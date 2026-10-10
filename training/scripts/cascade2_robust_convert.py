#!/usr/bin/env python3
"""Robust, self-verifying converter for flaky-memory hosts.

This machine exhibits transient, non-reproducible bit-flips when processing large
volumes of data (a handful of corrupted bytes per few hundred GB read/written,
load-correlated; SSD proven clean; likely marginal RAM). A normal single-pass
conversion therefore cannot be trusted, and a single read-back verification is
itself corrupted by the same faults.

Defence: redundancy + retry at small granularity. The source jsonl is ground
truth. Work is split into small chunks (byte ranges of the source files). For
each chunk we derive the canonical per-record hashes THREE independent times:
  h1 = from a first source read (also the records we write)
  h2 = from a second, independent source read
  hp = from reading the written parquet part back
A chunk is accepted only if h1 == h2 == hp elementwise (and counts match and all
messages are valid JSON). A transient flip in any single derivation breaks the
agreement and triggers a full retry of the chunk (re-reading the source afresh).
For three independent derivations to agree on a wrong value, the SAME flip would
have to occur in all three -- effectively impossible. Accepted chunks are thus
provably byte-exact w.r.t. the source.

Output: parquet_v2/domain=<d>/generator=<g>/part-NNNNN.parquet  (same schema as
cascade2_jsonl_to_parquet.py). Resumable: a chunk whose .done marker exists is
skipped. Keeps the original jsonl and the old parquet untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("domain", pa.string()),
    ("source", pa.string()),
    ("generator", pa.string()),
    ("messages", pa.string()),
])
KEEP = {"DeepSeek-V3.2", "DeepSeek-V3.2-Speciale"}
DOMAINS = {"math_notool", "math_tool", "math_proof", "swe_agentless", "terminal_agent"}
INPUTS = [
    "math/math_notool.jsonl",
    "math/math_tool.jsonl",
    "math/math_proof.jsonl",
    "swe/swe_agentless.jsonl",
    "terminal_agent/terminal_agent.jsonl",
]
CHUNK = 512 * 1024 * 1024  # ~512MB source byte ranges
MAX_RETRY = 20


def canon(domain, source, generator, messages_json):
    h = hashlib.blake2b(digest_size=16)
    h.update(domain.encode()); h.update(b"\x00")
    h.update(source.encode()); h.update(b"\x00")
    h.update(generator.encode()); h.update(b"\x00")
    h.update(messages_json.encode())
    return h.digest()


def read_source_range(path, start, end):
    """Return list of kept records as (domain, source, generator, messages_json)."""
    recs = []
    with open(path, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()  # partial line owned by previous chunk
        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            r = json.loads(raw)
            dom = r["domain"]; gen = r["generator"]
            if dom not in DOMAINS:
                raise ValueError(f"unexpected domain {dom!r} in {path}")
            if gen not in KEEP:
                continue
            msgs = r["messages"]
            for m in msgs:
                if set(m.keys()) != {"role", "content"}:
                    raise ValueError(f"unexpected msg keys {sorted(m.keys())} in {path}")
                if not isinstance(m["content"], str):
                    raise ValueError(f"non-str content in {path}")
            mj = json.dumps(msgs, ensure_ascii=False, separators=(",", ":"))
            recs.append((dom, r["source"], gen, mj))
    return recs


def hashes(recs):
    return [canon(*rec) for rec in recs]


def write_part(recs, tmp_path):
    cols = {n: [] for n in SCHEMA.names}
    for dom, src, gen, mj in recs:
        cols["domain"].append(dom); cols["source"].append(src)
        cols["generator"].append(gen); cols["messages"].append(mj)
    table = pa.table({n: pa.array(cols[n], type=SCHEMA.field(n).type) for n in SCHEMA.names},
                     schema=SCHEMA)
    pq.write_table(table, tmp_path, compression="zstd", compression_level=3)


def read_part(path):
    recs = []
    ok = True
    for b in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=SCHEMA.names):
        d = b.column("domain").to_pylist(); s = b.column("source").to_pylist()
        g = b.column("generator").to_pylist(); m = b.column("messages").to_pylist()
        for x in zip(d, s, g, m):
            recs.append(x)
            try:
                parsed = json.loads(x[3])
                for mm in parsed:
                    if set(mm.keys()) != {"role", "content"} or not isinstance(mm["content"], str):
                        ok = False
            except Exception:
                ok = False
    return recs, ok


def worker(args):
    cid, path, start, end, out_root, done_dir = args
    done_marker = os.path.join(done_dir, f"chunk-{cid:05d}")
    if os.path.exists(done_marker):
        return {"cid": cid, "skipped": True, "rows": 0, "retries": 0}

    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            recs1 = read_source_range(path, start, end)
            if not recs1:
                open(done_marker, "w").close()
                return {"cid": cid, "rows": 0, "retries": attempt, "empty": True}
            h1 = hashes(recs1)
            dom, _, gen, _ = recs1[0]
            pdir = os.path.join(out_root, f"domain={dom}", f"generator={gen}")
            os.makedirs(pdir, exist_ok=True)
            final = os.path.join(pdir, f"part-{cid:05d}.parquet")
            tmp = final + ".tmp"
            write_part(recs1, tmp)

            recs2 = read_source_range(path, start, end)
            h2 = hashes(recs2)
            recs_pq, ok = read_part(tmp)
            hp = hashes(recs_pq)

            if ok and len(h1) == len(h2) == len(hp) and h1 == h2 == hp:
                os.replace(tmp, final)
                open(done_marker, "w").close()
                return {"cid": cid, "rows": len(recs1), "retries": attempt,
                        "domain": dom, "generator": gen}
            # disagreement -> transient corruption somewhere; clean up and retry
            if os.path.exists(tmp):
                os.remove(tmp)
            last_err = (f"mismatch lens={len(h1)}/{len(h2)}/{len(hp)} "
                        f"valid={ok} eq12={h1==h2} eq1p={h1==hp}")
        except Exception as e:  # transient decode/parse error -> retry
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(0.05)
    raise RuntimeError(f"chunk {cid} ({path} [{start},{end})) failed after "
                       f"{MAX_RETRY} retries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    done_dir = os.path.join(args.out, ".done")
    os.makedirs(done_dir, exist_ok=True)

    tasks = []
    cid = 0
    for rel in INPUTS:
        path = os.path.join(args.root, rel)
        size = os.path.getsize(path)
        s = 0
        while s < size:
            e = min(s + CHUNK, size)
            tasks.append((cid, path, s, e, args.out, done_dir))
            cid += 1
            s = e
    n_done = sum(1 for t in tasks if os.path.exists(os.path.join(done_dir, f"chunk-{t[0]:05d}")))
    print(f"chunks={len(tasks)} already_done={n_done} workers={args.workers} "
          f"chunk_size={CHUNK//1024//1024}MB", flush=True)

    t0 = time.time()
    total_rows = 0
    total_retries = 0
    part_counts = {}
    done = 0
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(worker, tasks):
            done += 1
            total_rows += r.get("rows", 0)
            total_retries += r.get("retries", 0)
            if r.get("domain"):
                k = f"{r['domain']}|{r['generator']}"
                part_counts[k] = part_counts.get(k, 0) + r["rows"]
            if done % 50 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"[{done}/{len(tasks)}] rows={total_rows} retries={total_retries} "
                      f"elapsed={el/60:.1f}m", flush=True)

    print(f"\nDONE rows={total_rows} total_retries={total_retries} "
          f"elapsed={(time.time()-t0)/60:.1f}m")
    print("per-partition counts:")
    for k in sorted(part_counts):
        print(f"  {k}: {part_counts[k]}")


if __name__ == "__main__":
    main()
