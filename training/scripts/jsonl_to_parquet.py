#!/usr/bin/env python3
"""Convert Nemotron-SFT-Math-v3 train.jsonl -> hive-partitioned parquet.

Layout:
  parquet/data_source=<AoPS|StackExchange-Math>/tool_usage=<no_tir|tir>/part-NNN.parquet

Schema: scalar metadata as native typed columns; messages/tools/used_in stored
as JSON strings (stable schema, fast conversion, training loader re-parses).

Strategy: split the input file into N byte ranges, one worker per range. Each
worker aligns to a line boundary (discarding the partial leading line; the next
worker's range starts there) and processes whole lines whose start offset falls
within [start, end) -- lossless, no overlap. Rows are bucketed into the 4
(data_source, tool_usage) partitions and flushed in batches to per-worker,
per-partition parquet files. Fail-loud on any unexpected field/value.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("uuid", pa.string()),
    ("problem", pa.string()),
    ("expected_answer", pa.string()),
    ("changed_answer_to_majority", pa.bool_()),
    ("data_source", pa.string()),
    ("license", pa.string()),
    ("tool_usage", pa.string()),
    ("url", pa.string()),
    ("user_url", pa.string()),
    ("user_name", pa.string()),
    ("messages", pa.string()),
    ("tools", pa.string()),
    ("used_in", pa.string()),
])

TOOL_TOKEN = {"without Python TIR": "no_tir", "with Python TIR": "tir"}
DATA_SOURCES = {"AoPS", "StackExchange-Math"}
BATCH_ROWS = 1000


def partition_dir(out_root: str, data_source: str, tool_token: str) -> str:
    return os.path.join(out_root, f"data_source={data_source}", f"tool_usage={tool_token}")


def worker(args):
    wid, path, start, end, out_root = args
    writers: dict[tuple, pq.ParquetWriter] = {}
    # column buffers per partition key
    buffers: dict[tuple, dict] = {}
    counts: dict[tuple, int] = {}
    n_lines = 0

    def new_buffer():
        return {c: [] for c in SCHEMA.names}

    def flush(key):
        buf = buffers[key]
        if not buf["uuid"]:
            return
        batch = pa.record_batch(
            [pa.array(buf[c], type=SCHEMA.field(c).type) for c in SCHEMA.names],
            schema=SCHEMA,
        )
        if key not in writers:
            ds, tok = key
            d = partition_dir(out_root, ds, tok)
            os.makedirs(d, exist_ok=True)
            fp = os.path.join(d, f"part-{wid:03d}.parquet")
            writers[key] = pq.ParquetWriter(
                fp, SCHEMA, compression="zstd", compression_level=3)
        writers[key].write_batch(batch)
        buffers[key] = new_buffer()

    with open(path, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()  # discard partial line; previous worker owns it
        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            r = json.loads(raw)
            ds = r["data_source"]
            tu = r["tool_usage"]
            if ds not in DATA_SOURCES:
                raise ValueError(f"unexpected data_source: {ds!r}")
            if tu not in TOOL_TOKEN:
                raise ValueError(f"unexpected tool_usage: {tu!r}")
            key = (ds, TOOL_TOKEN[tu])
            if key not in buffers:
                buffers[key] = new_buffer()
                counts[key] = 0
            b = buffers[key]
            b["uuid"].append(r["uuid"])
            b["problem"].append(r["problem"])
            b["expected_answer"].append(r["expected_answer"])
            b["changed_answer_to_majority"].append(bool(r["changed_answer_to_majority"]))
            b["data_source"].append(ds)
            b["license"].append(r["license"])
            b["tool_usage"].append(tu)
            b["url"].append(r["url"])
            b["user_url"].append(r["user_url"])
            b["user_name"].append(r["user_name"])
            b["messages"].append(json.dumps(r["messages"], ensure_ascii=False, separators=(",", ":")))
            b["tools"].append(json.dumps(r["tools"], ensure_ascii=False, separators=(",", ":")))
            b["used_in"].append(json.dumps(r["used_in"], ensure_ascii=False, separators=(",", ":")))
            counts[key] += 1
            n_lines += 1
            if len(b["uuid"]) >= BATCH_ROWS:
                flush(key)

    for key in list(buffers):
        flush(key)
    for w in writers.values():
        w.close()
    return {"wid": wid, "n_lines": n_lines, "counts": {f"{k[0]}|{k[1]}": v for k, v in counts.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    if os.path.exists(args.out) and os.listdir(args.out):
        sys.exit(f"output dir not empty: {args.out} (refusing to mix)")
    os.makedirs(args.out, exist_ok=True)

    size = os.path.getsize(args.input)
    n = args.workers
    step = size // n
    ranges = []
    for i in range(n):
        s = i * step
        e = size if i == n - 1 else (i + 1) * step
        ranges.append((i, args.input, s, e, args.out))

    print(f"input={args.input} size={size/1e9:.1f}GB workers={n} step={step/1e9:.2f}GB", flush=True)
    with Pool(n) as pool:
        results = pool.map(worker, ranges)

    total = sum(r["n_lines"] for r in results)
    part_counts: dict[str, int] = {}
    for r in results:
        for k, v in r["counts"].items():
            part_counts[k] = part_counts.get(k, 0) + v
    print(f"\nTOTAL parsed lines: {total}")
    print("per-partition counts:")
    for k in sorted(part_counts):
        print(f"  {k}: {part_counts[k]}")


if __name__ == "__main__":
    main()
