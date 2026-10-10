#!/usr/bin/env python3
"""Full content verification: original train.jsonl  <->  parquet dataset.

Builds a per-uuid content hash from BOTH sources using an identical canonical
representation (all 13 stored fields; messages/tools/used_in serialized the same
way the converter did). Then proves a byte-level bijection:
  - every original uuid appears exactly once in parquet with the same hash
  - no extra / missing / duplicate uuids
Also validates that every parquet `messages` value is well-formed JSON with a
non-empty role sequence.

Original side is parsed in parallel over byte ranges; parquet side is read
sequentially (pyarrow C reader is fast) and checked against the original map.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import pyarrow.parquet as pq

SEP = b"\x1f"


def canon(uuid, problem, expected_answer, changed, data_source, license_, tool_usage,
          url, user_url, user_name, messages_s, tools_s, used_in_s) -> bytes:
    """Identical canonical byte representation regardless of source."""
    parts = [uuid, problem, expected_answer, "1" if changed else "0", data_source,
             license_, tool_usage, url, user_url, user_name, messages_s, tools_s, used_in_s]
    h = hashlib.blake2b(digest_size=16)
    for i, p in enumerate(parts):
        if i:
            h.update(SEP)
        h.update(p.encode("utf-8"))
    return h.digest()


def js(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def orig_worker(args):
    path, start, end = args
    out = {}
    dups = 0
    with open(path, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()
        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            r = json.loads(raw)
            uid = r["uuid"]
            hsh = canon(uid, r["problem"], r["expected_answer"],
                        bool(r["changed_answer_to_majority"]), r["data_source"],
                        r["license"], r["tool_usage"], r["url"], r["user_url"],
                        r["user_name"], js(r["messages"]), js(r["tools"]), js(r["used_in"]))
            if uid in out:
                dups += 1
            out[uid] = hsh
    return out, dups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    # ---- Phase 1: build original uuid -> hash (parallel) ----
    size = os.path.getsize(args.input)
    n = args.workers
    step = size // n
    ranges = [(args.input, i * step, size if i == n - 1 else (i + 1) * step) for i in range(n)]
    print(f"[phase1] hashing original {size/1e9:.1f}GB with {n} workers ...", flush=True)
    orig = {}
    orig_lines = 0
    intra_dups = 0
    with Pool(n) as pool:
        for part, dups in pool.map(orig_worker, ranges):
            orig_lines += len(part) + dups
            intra_dups += dups
            for k, v in part.items():
                if k in orig:
                    intra_dups += 1
                orig[k] = v
    print(f"[phase1] original lines={orig_lines} unique_uuids={len(orig)} dup_uuids={intra_dups}", flush=True)

    # ---- Phase 2: stream parquet, check against original ----
    files = sorted(glob.glob(args.parquet + "/**/*.parquet", recursive=True))
    print(f"[phase2] checking {len(files)} parquet files ...", flush=True)
    pq_rows = 0
    mismatched = 0
    missing_in_orig = 0
    bad_messages = 0
    pq_dups = 0
    seen = set()
    samples = {"mismatch": [], "missing": [], "bad_json": []}
    for fp in files:
        # read the file's own columns only (no partition discovery, so the
        # in-file data_source/tool_usage string columns are used as written)
        t = pq.ParquetFile(fp).read(columns=[
            "uuid", "problem", "expected_answer", "changed_answer_to_majority",
            "data_source", "license", "tool_usage", "url", "user_url", "user_name",
            "messages", "tools", "used_in"])
        d = t.to_pydict()
        for i in range(t.num_rows):
            uid = d["uuid"][i]
            pq_rows += 1
            if uid in seen:
                pq_dups += 1
            seen.add(uid)
            # validate messages JSON
            try:
                m = json.loads(d["messages"][i])
                if not isinstance(m, list) or len(m) == 0 or "role" not in m[0]:
                    raise ValueError("bad structure")
            except Exception as e:
                bad_messages += 1
                if len(samples["bad_json"]) < 5:
                    samples["bad_json"].append((uid, str(e)))
            hsh = canon(uid, d["problem"][i], d["expected_answer"][i],
                        bool(d["changed_answer_to_majority"][i]), d["data_source"][i],
                        d["license"][i], d["tool_usage"][i], d["url"][i], d["user_url"][i],
                        d["user_name"][i], d["messages"][i], d["tools"][i], d["used_in"][i])
            ov = orig.get(uid)
            if ov is None:
                missing_in_orig += 1
                if len(samples["missing"]) < 5:
                    samples["missing"].append(uid)
            elif ov != hsh:
                mismatched += 1
                if len(samples["mismatch"]) < 5:
                    samples["mismatch"].append(uid)
            else:
                del orig[uid]  # matched, remove so leftovers = missing-in-parquet

    missing_in_parquet = len(orig)
    print("\n===== VERIFICATION REPORT =====")
    print(f"original unique uuids : {orig_lines if False else '(see phase1)'}")
    print(f"parquet rows          : {pq_rows}")
    print(f"parquet duplicate uuids: {pq_dups}")
    print(f"hash mismatches       : {mismatched}")
    print(f"in parquet, not in orig: {missing_in_orig}")
    print(f"in orig, not in parquet: {missing_in_parquet}")
    print(f"malformed messages JSON: {bad_messages}")
    if any(samples.values()):
        print("samples:", {k: v for k, v in samples.items() if v})
    if missing_in_parquet:
        print("sample missing-in-parquet uuids:", list(orig.keys())[:5])

    ok = (intra_dups == 0 and pq_dups == 0 and mismatched == 0 and
          missing_in_orig == 0 and missing_in_parquet == 0 and bad_messages == 0)
    print("\nRESULT:", "PASS ✅ — parquet is a byte-exact bijection of the original" if ok else "FAIL ❌")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
