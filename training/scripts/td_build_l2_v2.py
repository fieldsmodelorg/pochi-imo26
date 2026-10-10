#!/usr/bin/env python3
"""Build unified L2 (v2) parquet from 4 raw-JSONL Nemotron datasets.

New version, separate from v1 (training_l2). Reuses the SAME L2 OUT_SCHEMA /
canonical hash so the downstream L3 renderer is unchanged.

Sources (all already OpenAI-style messages: reasoning_content separate, structured
tool_calls / `tool` role -- NO inline reverse-parse needed, unlike cascade-2):

  nemotron-sft-math-v4      data/train.jsonl        gen=DeepSeek-V4-Pro
      domain = {aops|stackexchange_math}_{cot|tir}   (from source x subset)
  nemotron-math-proofs-v2   data/train.jsonl        gen=DeepSeek-V4-Pro
      domain = proof | verification | meta_verification  (from user-prompt
      signature; the upstream `subset` field is uniformly "proof")
  nemotron-sft-science-v2   {rqa,so,syn_mcq,vendor}.jsonl
      domain = file stem ; generator PER-ROW from metadata.generation_model;
      KEEP ONLY DeepSeek* rows (drops gpt-oss-120b / Kimi-K2)
  nemotron-sft-agentic-v2   {tool_calling,interactive_agent}.jsonl  gen=DeepSeek-V3.2
      domain = file stem ; search.jsonl SKIPPED (no model field, source unclear)

Robustness (host has load-correlated transient memory bit-flips): inputs are huge
JSONL, so we bundle by byte-range (~1GB). Each bundle is verified THREE independent
times -- normalize from a first read (writes shards), normalize from an independent
second read, and read the written shard(s) back -- requiring identical (count, hash-sum);
any mismatch retries the whole bundle. A single JSONL spans multiple L2 domains, so a
bundle may write several partition files (one per domain); the hash-sum covers all rows
regardless of domain (order-independent). Resumable via .done/.
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

sys.path.insert(0, os.path.dirname(__file__))
import td_normalize as N
from td_build_l2 import OUT_SCHEMA, canon_row, jdump, MOD, TARGET_BYTES

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")

# file stem -> upstream_source label for science
SCIENCE_UPSTREAM = {"so": "Math StackExchange", "rqa": "synthetic",
                    "syn_mcq": "synthetic", "vendor": "vendor"}
SRC_TAG = {"aops": "aops", "math stackexchange": "stackexchange_math"}


# ----- helpers ---------------------------------------------------------------

def norm_license(s: str) -> str:
    """'CC BY 4.0' -> 'cc-by-4.0', 'CC BY-SA 4.0' -> 'cc-by-sa-4.0', idempotent."""
    if not s:
        return ""
    return s.strip().lower().replace(" ", "-")


def get_id(raw: dict) -> str:
    if raw.get("uuid"):
        return raw["uuid"]
    md = raw.get("metadata")
    if isinstance(md, dict):
        for k in ("uuid", "id"):
            if md.get(k):
                return str(md[k])
    return hashlib.blake2b(jdump(raw["messages"]).encode("utf-8", "surrogatepass"),
                           digest_size=12).hexdigest()


def _tools_in(raw: dict):
    """Tools may be a list, '' (str), or absent -> normalize to list|None."""
    t = raw.get("tools")
    if not t or t == "":
        return None
    return t


def _l2(raw, dataset, domain, generator, license_, upstream, expected):
    """Assemble one L2 row. messages already OpenAI-style; reuse normalize_v3
    (attaches an empty system carrying tools for TIR when none present)."""
    messages, tools = N.normalize_v3(raw["messages"], _tools_in(raw))
    meta = {k: v for k, v in raw.items() if k not in ("messages", "tools")}
    return {
        "id": get_id(raw), "dataset": dataset, "domain": domain,
        "generator": generator, "thinking_mode": "thinking",
        "has_tools": tools is not None, "n_turns": N.n_assistant_turns(messages),
        "license": license_ or "", "upstream_source": upstream or "",
        "expected_answer": expected or "",
        "messages": jdump(messages), "tools": jdump(tools) if tools else "",
        "meta": jdump(meta),
    }


def _proofs_domain(user_prompt: str) -> str:
    """Classify a Nemotron-Math-Proofs-v2 record into proof / verification /
    meta_verification by its user-prompt signature (the `subset` field is unusably
    uniform). Raises on an unrecognized prompt so a new trace type fails loud."""
    if 'and you need to assess' in user_prompt and '"solution evaluation"' in user_prompt:
        return "meta_verification"
    if user_prompt.lstrip().startswith("## Instruction") and "evaluate the quality of a solution" in user_prompt:
        return "verification"
    if "Your task is to solve a given problem" in user_prompt:
        return "proof"
    raise RuntimeError(f"unclassified proofs prompt: {user_prompt[:120]!r}")


# ----- per-dataset row builders (dispatched by string -> picklable) ----------

def build_l2_row(raw: dict, dataset: str, subtag: str):
    """Return one L2 dict, or None to drop this raw row."""
    if dataset == "nemotron-sft-math-v4":
        src = (raw.get("source") or "").strip()
        sub = (raw.get("subset") or "").strip().lower()  # cot / tir
        dom = f"{SRC_TAG.get(src.lower(), src.lower().replace(' ', '_'))}_{sub}"
        return _l2(raw, dataset, dom, "DeepSeek-V4-Pro",
                   norm_license(raw.get("license")), src, raw.get("expected_answer", ""))

    if dataset == "nemotron-math-proofs-v2":
        # The upstream `subset` field is uniformly "proof" (nvidia bug); the three trace
        # types are instead distinguished by the user-prompt instruction. Counts via this
        # signature exactly match the dataset card (24,696 / 28,865 / 29,176).
        dom = _proofs_domain(raw["messages"][0]["content"])
        return _l2(raw, dataset, dom, "DeepSeek-V4-Pro",
                   norm_license(raw.get("license")), raw.get("source", "AoPS"), "")

    if dataset == "nemotron-sft-science-v2":
        md = raw.get("metadata") or {}
        gen = md.get("generation_model", "")
        if not gen.startswith("DeepSeek"):
            return None  # keep DeepSeek only
        return _l2(raw, dataset, subtag, gen,
                   norm_license(raw.get("license")), SCIENCE_UPSTREAM.get(subtag, ""), "")

    if dataset == "nemotron-sft-agentic-v2":
        return _l2(raw, dataset, subtag, "DeepSeek-V3.2", "cc-by-4.0", "", "")

    raise RuntimeError(f"unknown dataset {dataset!r}")


# ----- JSONL byte-range bundling --------------------------------------------

def scan_bundles(path, dataset, subtag, target=TARGET_BYTES):
    """Cut a JSONL into byte-range bundles of ~target bytes, on line boundaries."""
    bundles = []
    with open(path, "rb") as f:
        start = off = cur = 0
        while True:
            line = f.readline()
            if not line:
                break
            ln = len(line)
            if cur > 0 and cur + ln > target:
                bundles.append((path, dataset, subtag, start, off))
                start = off
                cur = 0
            cur += ln
            off += ln
        if cur > 0:
            bundles.append((path, dataset, subtag, start, off))
    return bundles


def iter_range(path, start, end):
    """Yield decoded non-empty lines whose start offset is within [start, end)."""
    with open(path, "rb") as f:
        f.seek(start)
        pos = start
        while pos < end:
            line = f.readline()
            if not line:
                break
            pos += len(line)
            s = line.strip()
            if s:
                yield s  # raw bytes; caller decodes / NUL-screens


# ----- bundle worker (3-way verified) ---------------------------------------

def _norm_pass(path, dataset, subtag, start, end, bid, out_root, write):
    """One pass over a byte range. If write, stream-write per-domain shards.
    Returns (count, hash_sum, {domain: tmp_path}, corrupt). `corrupt` counts records
    skipped because they contain NUL bytes -- upstream storage corruption (e.g. nvidia's
    tool_calling.jsonl line 1094 is a truncated record + ~228MB NUL hole). Deterministic
    skip -> the three verification passes still agree."""
    cnt = 0
    hsum = 0
    corrupt = 0
    writers = {}   # domain -> (ParquetWriter, tmp, final)
    bufs = {}      # domain -> {col: []}

    def flush(dom):
        b = bufs[dom]
        if b["id"]:
            writers[dom][0].write_batch(pa.record_batch(
                [pa.array(b[n], type=OUT_SCHEMA.field(n).type) for n in OUT_SCHEMA.names],
                schema=OUT_SCHEMA))
            for n in OUT_SCHEMA.names:
                b[n].clear()

    for ln in iter_range(path, start, end):
        if b"\x00" in ln:                 # upstream-corrupt record (NUL hole) -- skip, counted
            corrupt += 1
            continue
        raw = json.loads(ln)
        row = build_l2_row(raw, dataset, subtag)
        if row is None:
            continue
        hsum = (hsum + canon_row(row)) % MOD
        cnt += 1
        if write:
            dom = row["domain"]
            if dom not in writers:
                pdir = os.path.join(out_root, f"dataset={dataset}", f"domain={dom}")
                os.makedirs(pdir, exist_ok=True)
                final = os.path.join(pdir, f"part-{bid:05d}.parquet")
                tmp = final + ".tmp"
                writers[dom] = (pq.ParquetWriter(tmp, OUT_SCHEMA, compression="zstd",
                                                 compression_level=3), tmp, final)
                bufs[dom] = {n: [] for n in OUT_SCHEMA.names}
            b = bufs[dom]
            for n in OUT_SCHEMA.names:
                b[n].append(row[n])
            if len(b["id"]) >= 512:
                flush(dom)

    tmps = {}
    if write:
        for dom in writers:
            flush(dom)
            writers[dom][0].close()
            tmps[dom] = (writers[dom][1], writers[dom][2])
    return cnt, hsum, tmps, corrupt


def _readback(tmps):
    cnt = 0
    hsum = 0
    for tmp, _final in tmps.values():
        for batch in pq.ParquetFile(tmp).iter_batches(batch_size=512, columns=OUT_SCHEMA.names):
            rows = {n: batch.column(n).to_pylist() for n in OUT_SCHEMA.names}
            for i in range(batch.num_rows):
                hsum = (hsum + canon_row({n: rows[n][i] for n in OUT_SCHEMA.names})) % MOD
                cnt += 1
    return cnt, hsum


def worker(task):
    bid, path, dataset, subtag, start, end, out_root, done_dir = task
    done = os.path.join(done_dir, f"bundle-{bid:05d}")
    if os.path.exists(done):
        return {"bid": bid, "skipped": True, "rows": 0, "retries": 0}
    last = ""
    for attempt in range(20):
        tmps = {}
        try:
            c1, h1, tmps, k1 = _norm_pass(path, dataset, subtag, start, end, bid, out_root, write=True)
            c2, h2, _, k2 = _norm_pass(path, dataset, subtag, start, end, bid, out_root, write=False)
            c3, h3 = _readback(tmps)
            if c1 == c2 == c3 and h1 == h2 == h3 and k1 == k2:
                for tmp, final in tmps.values():
                    os.replace(tmp, final)
                open(done, "w").close()
                return {"bid": bid, "rows": c1, "retries": attempt, "corrupt": k1,
                        "dataset": dataset, "subtag": subtag}
            last = f"count {c1}/{c2}/{c3} eq12={h1==h2} eq13={h1==h3} corrupt={k1}/{k2}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        for tmp, _final in tmps.values():
            if os.path.exists(tmp):
                os.remove(tmp)
        time.sleep(0.05)
    raise RuntimeError(f"bundle {bid} ({path} [{start}:{end}]) failed: {last}")


# ----- discovery -------------------------------------------------------------

def discover(root, out_root, done_dir):
    files = [
        ("nemotron-sft-math-v4", "Nemotron-SFT-Math-v4/data/train.jsonl", ""),
        ("nemotron-math-proofs-v2", "Nemotron-Math-Proofs-v2/data/train.jsonl", ""),
        ("nemotron-sft-science-v2", "Nemotron-SFT-Science-v2/rqa.jsonl", "rqa"),
        ("nemotron-sft-science-v2", "Nemotron-SFT-Science-v2/so.jsonl", "so"),
        ("nemotron-sft-science-v2", "Nemotron-SFT-Science-v2/syn_mcq.jsonl", "syn_mcq"),
        ("nemotron-sft-science-v2", "Nemotron-SFT-Science-v2/vendor.jsonl", "vendor"),
        ("nemotron-sft-agentic-v2", "Nemotron-SFT-Agentic-v2/data/tool_calling.jsonl", "tool_calling"),
        ("nemotron-sft-agentic-v2", "Nemotron-SFT-Agentic-v2/data/interactive_agent.jsonl", "interactive_agent"),
        # search.jsonl intentionally skipped
    ]
    tasks = []
    bid = 0
    for dataset, rel, subtag in files:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"WARN missing {path}", flush=True)
            continue
        for (p, ds, st, s, e) in scan_bundles(path, dataset, subtag):
            tasks.append((bid, p, ds, st, s, e, out_root, done_dir))
            bid += 1
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="datasets/ root with the 4 source dirs")
    ap.add_argument("--out", required=True, help="output dir, e.g. datasets/training_l2_v2")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    done_dir = os.path.join(args.out, ".done")
    os.makedirs(done_dir, exist_ok=True)
    tasks = discover(args.root, args.out, done_dir)
    nd = sum(1 for t in tasks if os.path.exists(os.path.join(done_dir, f"bundle-{t[0]:05d}")))
    print(f"bundles={len(tasks)} done={nd} workers={args.workers} target=~1GB", flush=True)
    t0 = time.time()
    rows = retries = done = corrupt = 0
    pc = {}
    corrupt_by = {}
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(worker, tasks):
            done += 1
            rows += r.get("rows", 0)
            retries += r.get("retries", 0)
            corrupt += r.get("corrupt", 0)
            if r.get("dataset"):
                k = f"{r['dataset']}/{r['subtag']}" if r["subtag"] else r["dataset"]
                pc[k] = pc.get(k, 0) + r["rows"]
                if r.get("corrupt"):
                    corrupt_by[k] = corrupt_by.get(k, 0) + r["corrupt"]
            if done % 10 == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] rows={rows} retries={retries} "
                      f"corrupt={corrupt} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    print(f"\nDONE rows={rows} retries={retries} elapsed={(time.time()-t0)/60:.1f}m")
    if corrupt:
        print(f"!! SKIPPED {corrupt} NUL-corrupt upstream records:")
        for k in sorted(corrupt_by):
            print(f"   {k}: {corrupt_by[k]}")
    for k in sorted(pc):
        print(f"  {k}: {pc[k]}")


if __name__ == "__main__":
    main()
