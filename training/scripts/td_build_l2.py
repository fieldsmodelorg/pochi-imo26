#!/usr/bin/env python3
"""Build unified L2 training-intermediate parquet from L1 (v3 + cascade-2).

Preserves ALL original metadata (provenance / license / attribution / verified answer)
and emits uniform ~1GB shards.

L2 schema (one row per sample), partitioned dataset=<d>/domain=<dom>/:
  id, dataset, domain, generator, thinking_mode, has_tools, n_turns,
  license, upstream_source, expected_answer,
  messages (JSON, OpenAI-style structured), tools (JSON or ""),
  meta (JSON: every original non-messages column verbatim)

Tokenizer-independent; training renders via encoding_dsv4.encode_messages.

Robustness (host has load-correlated transient memory bit-flips): each output shard is
a BUNDLE of consecutive same-partition L1 parts (~1GB). For each bundle we derive the
per-row canonical-hash SUM three independent times -- normalize from a first source read
(and write the shard), normalize from an independent second source read, and read the
written shard back -- and require all three to match; mismatch retries the whole bundle.
Streaming via iter_batches bounds memory. Resumable via .done.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
import td_normalize as N

OUT_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("dataset", pa.string()),
    ("domain", pa.string()),
    ("generator", pa.string()),
    ("thinking_mode", pa.string()),
    ("has_tools", pa.bool_()),
    ("n_turns", pa.int32()),
    ("license", pa.string()),
    ("upstream_source", pa.string()),
    ("expected_answer", pa.string()),
    ("messages", pa.string()),
    ("tools", pa.string()),
    ("meta", pa.string()),
])

V3_GEN = {"without Python TIR": "DeepSeek-V3.2-Speciale", "with Python TIR": "DeepSeek-V3.2"}
V3_DOMTAG = {"without Python TIR": "cot", "with Python TIR": "tir"}
V3_META_COLS = ["problem", "expected_answer", "changed_answer_to_majority", "data_source",
                "license", "tool_usage", "url", "user_url", "user_name", "used_in"]
CASCADE_LICENSE = "NVIDIA Open Model License"
MOD = 1 << 256
TARGET_BYTES = 1_000_000_000  # ~1GB per output shard (by cumulative L1 input size)


def jdump(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def canon_row(r: dict) -> int:
    h = hashlib.blake2b(digest_size=16)
    for k in OUT_SCHEMA.names:
        h.update(str(r[k]).encode("utf-8")); h.update(b"\x00")
    return int.from_bytes(h.digest(), "big")


def input_columns(dataset):
    if dataset == "nemotron-math-v3":
        return ["uuid", "messages", "tools"] + V3_META_COLS
    return ["messages", "source"]


def build_rows(batch, dataset, domain, generator):
    if dataset == "nemotron-math-v3":
        col = {n: batch.column(n).to_pylist() for n in batch.schema.names}
        for i in range(batch.num_rows):
            raw = json.loads(col["messages"][i])
            tj = col["tools"][i]
            rtools = json.loads(tj) if tj else None
            rtools = rtools or None
            messages, tools = N.normalize_v3(raw, rtools)
            meta = {}
            for c in V3_META_COLS:
                v = col[c][i]
                if c == "used_in":
                    try:
                        v = json.loads(v) if isinstance(v, str) else v
                    except Exception:
                        pass
                meta[c] = v
            yield {
                "id": col["uuid"][i], "dataset": dataset, "domain": domain,
                "generator": generator, "thinking_mode": "thinking",
                "has_tools": tools is not None, "n_turns": N.n_assistant_turns(messages),
                "license": col["license"][i] or "",
                "upstream_source": col["data_source"][i] or "",
                "expected_answer": col["expected_answer"][i] or "",
                "messages": jdump(messages), "tools": jdump(tools) if tools else "",
                "meta": jdump(meta),
            }
    else:  # cascade-2
        mcol = batch.column("messages").to_pylist()
        scol = batch.column("source").to_pylist()
        for mj, up in zip(mcol, scol):
            raw = json.loads(mj)
            messages, tools = N.normalize_cascade(domain, raw)
            rid = hashlib.blake2b(mj.encode("utf-8", "surrogatepass"), digest_size=12).hexdigest()
            yield {
                "id": rid, "dataset": dataset, "domain": domain,
                "generator": generator, "thinking_mode": "thinking",
                "has_tools": tools is not None, "n_turns": N.n_assistant_turns(messages),
                "license": CASCADE_LICENSE, "upstream_source": up or "",
                "expected_answer": "",
                "messages": jdump(messages), "tools": jdump(tools) if tools else "",
                "meta": jdump({"upstream_source": up}),
            }


def normalize_pass(input_files, dataset, domain, generator, writer=None):
    cols = input_columns(dataset)
    cnt = 0
    hsum = 0
    buf = {n: [] for n in OUT_SCHEMA.names}

    def flush():
        if buf["id"]:
            writer.write_batch(pa.record_batch(
                [pa.array(buf[n], type=OUT_SCHEMA.field(n).type) for n in OUT_SCHEMA.names],
                schema=OUT_SCHEMA))
            for n in OUT_SCHEMA.names:
                buf[n].clear()

    for path in input_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=cols):
            for row in build_rows(batch, dataset, domain, generator):
                hsum = (hsum + canon_row(row)) % MOD
                cnt += 1
                if writer is not None:
                    for n in OUT_SCHEMA.names:
                        buf[n].append(row[n])
                    if len(buf["id"]) >= 512:
                        flush()
    if writer is not None:
        flush()
    return cnt, hsum


def readback_pass(path):
    cnt = 0
    hsum = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=OUT_SCHEMA.names):
        rows = {n: batch.column(n).to_pylist() for n in OUT_SCHEMA.names}
        for i in range(batch.num_rows):
            r = {n: rows[n][i] for n in OUT_SCHEMA.names}
            hsum = (hsum + canon_row(r)) % MOD
            cnt += 1
    return cnt, hsum


def worker(args):
    bid, input_files, dataset, domain, generator, out_root, done_dir = args
    done = os.path.join(done_dir, f"bundle-{bid:05d}")
    if os.path.exists(done):
        return {"bid": bid, "skipped": True, "rows": 0, "retries": 0}
    pdir = os.path.join(out_root, f"dataset={dataset}", f"domain={domain}")
    os.makedirs(pdir, exist_ok=True)
    final = os.path.join(pdir, f"part-{bid:05d}.parquet")
    tmp = final + ".tmp"
    last = ""
    for attempt in range(20):
        try:
            w = pq.ParquetWriter(tmp, OUT_SCHEMA, compression="zstd", compression_level=3)
            c1, h1 = normalize_pass(input_files, dataset, domain, generator, writer=w)
            w.close()
            c2, h2 = normalize_pass(input_files, dataset, domain, generator, writer=None)
            c3, h3 = readback_pass(tmp)
            if c1 == c2 == c3 and h1 == h2 == h3:
                os.replace(tmp, final)
                open(done, "w").close()
                return {"bid": bid, "rows": c1, "retries": attempt,
                        "dataset": dataset, "domain": domain}
            last = f"count {c1}/{c2}/{c3} eq12={h1==h2} eq13={h1==h3}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if os.path.exists(tmp):
            os.remove(tmp)
        time.sleep(0.05)
    raise RuntimeError(f"bundle {bid} ({input_files[0]}..) failed: {last}")


def bundle_partition(files_with_size):
    """Group (path,size) into bundles whose cumulative size ~ TARGET_BYTES."""
    bundles = []
    cur, cur_sz = [], 0
    for path, sz in files_with_size:
        if cur and cur_sz + sz > TARGET_BYTES:
            bundles.append(cur); cur, cur_sz = [], 0
        cur.append(path); cur_sz += sz
    if cur:
        bundles.append(cur)
    return bundles


def discover(cascade_root, v3_root, out_root, done_dir):
    tasks = []
    bid = 0
    # cascade: dataset=nemotron-cascade-2, domain from path
    parts_by_dom = {}
    for dp, _, fns in os.walk(cascade_root):
        rel = os.path.relpath(dp, cascade_root)
        kv = dict(p.split("=", 1) for p in rel.split(os.sep) if "=" in p)
        if "domain" not in kv:
            continue
        key = ("nemotron-cascade-2", kv["domain"], kv.get("generator", ""))
        for fn in sorted(fns):
            if fn.endswith(".parquet"):
                p = os.path.join(dp, fn)
                parts_by_dom.setdefault(key, []).append((p, os.path.getsize(p)))
    # v3: dataset=nemotron-math-v3, domain/generator from data_source/tool_usage path
    for dp, _, fns in os.walk(v3_root):
        rel = os.path.relpath(dp, v3_root)
        kv = dict(p.split("=", 1) for p in rel.split(os.sep) if "=" in p)
        if "data_source" not in kv or "tool_usage" not in kv:
            continue
        ds = unquote(kv["data_source"]); tu = unquote(kv["tool_usage"])
        domain = f"{ds.lower().replace('-', '_')}_{V3_DOMTAG[tu]}"
        key = ("nemotron-math-v3", domain, V3_GEN[tu])
        for fn in sorted(fns):
            if fn.endswith(".parquet"):
                p = os.path.join(dp, fn)
                parts_by_dom.setdefault(key, []).append((p, os.path.getsize(p)))

    for (dataset, domain, generator), fws in sorted(parts_by_dom.items()):
        fws.sort()
        for bundle in bundle_partition(fws):
            tasks.append((bid, bundle, dataset, domain, generator, out_root, done_dir))
            bid += 1
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade", required=True)
    ap.add_argument("--v3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    done_dir = os.path.join(args.out, ".done")
    os.makedirs(done_dir, exist_ok=True)
    tasks = discover(args.cascade, args.v3, args.out, done_dir)
    nd = sum(1 for t in tasks if os.path.exists(os.path.join(done_dir, f"bundle-{t[0]:05d}")))
    print(f"bundles={len(tasks)} done={nd} workers={args.workers} target=~1GB", flush=True)
    t0 = time.time()
    rows = retries = done = 0
    pc = {}
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(worker, tasks):
            done += 1
            rows += r.get("rows", 0)
            retries += r.get("retries", 0)
            if r.get("dataset"):
                k = f"{r['dataset']}/{r['domain']}"
                pc[k] = pc.get(k, 0) + r["rows"]
            if done % 10 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] rows={rows} retries={retries} "
                      f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    print(f"\nDONE rows={rows} retries={retries} elapsed={(time.time()-t0)/60:.1f}m")
    for k in sorted(pc):
        print(f"  {k}: {pc[k]}")


if __name__ == "__main__":
    main()
