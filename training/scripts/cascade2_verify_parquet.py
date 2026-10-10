#!/usr/bin/env python3
"""Verify cascade-2 parquet == DeepSeek-filtered jsonl, byte-exact (no uuid).

This dataset has no uuid, so verification compares an order-independent multiset
of per-record canonical hashes between (a) the DeepSeek-filtered original jsonl
and (b) the parquet output.

Multiset equality is checked via the running SUM of the 128-bit canonical digests
(addition is commutative and respects multiplicity, unlike XOR), together with
the exact record count and per-(domain,generator) counts. A canonical digest is
blake2b over domain\x00source\x00generator\x00<compact-messages-json>; the
messages JSON on both sides uses the identical json.dumps(...) serialization, so
a matching record produces an identical digest. Also validates, on the parquet
side, that every messages value is valid JSON, every message has exactly
{role,content}, and every content is a str.

Run AFTER conversion. Only delete the source jsonl if this reports PASS.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import pyarrow.parquet as pq

KEEP = {"DeepSeek-V3.2", "DeepSeek-V3.2-Speciale"}
DOMAINS = {"math_notool", "math_tool", "math_proof", "swe_agentless", "terminal_agent"}

INPUTS = [
    "math/math_notool.jsonl",
    "math/math_tool.jsonl",
    "math/math_proof.jsonl",
    "swe/swe_agentless.jsonl",
    "terminal_agent/terminal_agent.jsonl",
]

MOD = 1 << 256  # bound the running sum size


def canon_int(domain: str, source: str, generator: str, messages_json: str) -> int:
    h = hashlib.blake2b(digest_size=16)
    h.update(domain.encode("utf-8")); h.update(b"\x00")
    h.update(source.encode("utf-8")); h.update(b"\x00")
    h.update(generator.encode("utf-8")); h.update(b"\x00")
    h.update(messages_json.encode("utf-8"))
    return int.from_bytes(h.digest(), "big")


def jsonl_worker(args):
    path, start, end = args
    hsum = 0
    kept = 0
    dropped = 0
    n_lines = 0
    pcounts = collections.Counter()
    with open(path, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()
        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            n_lines += 1
            r = json.loads(raw)
            gen = r["generator"]
            if gen not in KEEP:
                dropped += 1
                continue
            dom = r["domain"]
            mj = json.dumps(r["messages"], ensure_ascii=False, separators=(",", ":"))
            hsum = (hsum + canon_int(dom, r["source"], gen, mj)) % MOD
            kept += 1
            pcounts[(dom, gen)] += 1
    return hsum, kept, dropped, n_lines, pcounts


def parquet_worker(path):
    hsum = 0
    rows = 0
    bad_json = 0
    bad_keys = 0
    nonstr = 0
    pcounts = collections.Counter()
    # Stream in small batches (ParquetFile does NO hive-partition discovery, so the
    # in-file domain/generator string columns are used as-is rather than conflicting
    # with dictionary-typed columns inferred from the directory path). iter_batches
    # bounds peak memory to one batch -- messages can be tens of KB each, so reading
    # a whole file with to_pylist() across many workers blows up RAM.
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(
            batch_size=1000, columns=["domain", "source", "generator", "messages"]):
        doms = batch.column("domain").to_pylist()
        srcs = batch.column("source").to_pylist()
        gens = batch.column("generator").to_pylist()
        msgs = batch.column("messages").to_pylist()
        for dom, src, gen, mj in zip(doms, srcs, gens, msgs):
            rows += 1
            pcounts[(dom, gen)] += 1
            hsum = (hsum + canon_int(dom, src, gen, mj)) % MOD
            try:
                parsed = json.loads(mj)
            except Exception:
                bad_json += 1
                continue
            for m in parsed:
                if set(m.keys()) != {"role", "content"}:
                    bad_keys += 1
                elif not isinstance(m["content"], str):
                    nonstr += 1
    return hsum, rows, bad_json, bad_keys, nonstr, pcounts


def list_parquet(out_root):
    files = []
    for dp, _, fns in os.walk(out_root):
        for fn in fns:
            if fn.endswith(".parquet"):
                files.append(os.path.join(dp, fn))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset dir (jsonl source)")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--pq-workers", type=int, default=8,
                    help="parallelism for the parquet pass (kept low: each worker "
                         "buffers a batch of large message strings)")
    args = ap.parse_args()

    # ---- jsonl pass (byte-range parallel) ----
    tasks = []
    for rel in INPUTS:
        path = os.path.join(args.root, rel)
        size = os.path.getsize(path)
        step = size // args.workers
        for i in range(args.workers):
            s = i * step
            e = size if i == args.workers - 1 else (i + 1) * step
            tasks.append((path, s, e))
    with Pool(args.workers) as pool:
        jres = pool.map(jsonl_worker, tasks)
    j_sum = 0
    j_kept = j_dropped = j_lines = 0
    j_pc = collections.Counter()
    for hsum, kept, dropped, nl, pc in jres:
        j_sum = (j_sum + hsum) % MOD
        j_kept += kept; j_dropped += dropped; j_lines += nl
        j_pc.update(pc)

    # ---- parquet pass (per-file parallel) ----
    pfiles = list_parquet(args.parquet)
    with Pool(args.pq_workers) as pool:
        pres = pool.map(parquet_worker, pfiles)
    p_sum = 0
    p_rows = p_badjson = p_badkeys = p_nonstr = 0
    p_pc = collections.Counter()
    for hsum, rows, bj, bk, ns, pc in pres:
        p_sum = (p_sum + hsum) % MOD
        p_rows += rows; p_badjson += bj; p_badkeys += bk; p_nonstr += ns
        p_pc.update(pc)

    # ---- report ----
    print("=== jsonl side ===")
    print(f"  lines={j_lines} kept={j_kept} dropped={j_dropped}")
    print("=== parquet side ===")
    print(f"  rows={p_rows} bad_json={p_badjson} bad_keys={p_badkeys} nonstr_content={p_nonstr}")
    print("=== per-partition (domain|generator)  jsonl -> parquet ===")
    allkeys = sorted(set(j_pc) | set(p_pc))
    for k in allkeys:
        print(f"  {k[0]}|{k[1]}: {j_pc[k]} -> {p_pc[k]}")

    ok = True
    if j_kept + j_dropped != j_lines:
        print("FAIL: kept+dropped != lines"); ok = False
    if j_kept != p_rows:
        print(f"FAIL: kept {j_kept} != parquet rows {p_rows}"); ok = False
    if j_sum != p_sum:
        print("FAIL: canonical hash multiset sum differs"); ok = False
    if j_pc != p_pc:
        print("FAIL: per-partition counts differ"); ok = False
    if p_badjson or p_badkeys or p_nonstr:
        print("FAIL: parquet content validation failed"); ok = False

    if ok:
        print("\nPASS: parquet is byte-exact for the DeepSeek-filtered jsonl.")
        sys.exit(0)
    else:
        print("\nVERIFICATION FAILED -- do NOT delete source jsonl.")
        sys.exit(1)


if __name__ == "__main__":
    main()
