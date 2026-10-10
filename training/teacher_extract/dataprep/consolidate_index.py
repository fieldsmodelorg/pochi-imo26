# Copyright 2026 proof-pilot. Apache-2.0.
"""Consolidate a (possibly partial) sharded hidden extraction into one index.jsonl.

Robust to shards whose per-shard index.jsonl is missing (e.g. a shard whose client was
killed before write_index). Instead of reading every .pt, it joins the manifest (which has
the authoritative prompt_len/seq_len/target_len that extract_hidden used) against the .pt
files that actually exist on disk: row_idx r lives in shard (r % num_shards) as
{r:06d}.pt. Only rows with a present .pt are written. Reports per-shard coverage + gaps.

  .venv/bin/python training/teacher_extract/dataprep/consolidate_index.py \
    --manifest .../math-v4-aops-cot/manifest.parquet \
    --shard-dir .../math-v4-aops-cot/hidden_shards --num-shards 8 \
    --out .../math-v4-aops-cot/hidden/index.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pyarrow.parquet as pq

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))
from offpolicy_data import ExtractedDocMeta, write_index  # noqa: E402


def _s(v) -> str:
    return "" if v is None else str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--shard-dir", type=Path, required=True, help="dir holding shard_*/docs/*.pt")
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cols = ["row_idx", "prompt_len", "seq_len", "target_len",
            "run_id", "problem_id", "template", "effort"]
    metas: list[ExtractedDocMeta] = []
    per_shard_have = [0] * a.num_shards
    per_shard_total = [0] * a.num_shards
    missing: list[int] = []

    pf = pq.ParquetFile(a.manifest)
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        for r in batch.to_pylist():
            ri = int(r["row_idx"])
            sh = ri % a.num_shards
            per_shard_total[sh] += 1
            path = a.shard_dir / f"shard_{sh}" / "docs" / f"{ri:06d}.pt"
            if not path.exists():
                missing.append(ri)
                continue
            per_shard_have[sh] += 1
            tl = int(r["target_len"])
            metas.append(ExtractedDocMeta(
                row_idx=ri, path=str(path.resolve()),
                prompt_len=int(r["prompt_len"]), seq_len=int(r["seq_len"]),
                target_len=tl, teacher_seq_len=tl + 1,
                run_id=_s(r.get("run_id")), problem_id=_s(r.get("problem_id")),
                template=_s(r.get("template")), effort=_s(r.get("effort")),
            ))

    metas.sort(key=lambda m: m.row_idx)
    write_index(a.out, metas)
    print(f"[consolidate] wrote {len(metas)} rows -> {a.out}")
    for sh in range(a.num_shards):
        gap = per_shard_total[sh] - per_shard_have[sh]
        print(f"  shard_{sh}: {per_shard_have[sh]}/{per_shard_total[sh]}" + (f"  (missing {gap})" if gap else ""))
    print(f"  total kept={len(metas)}  missing={len(missing)}  target_tokens={sum(m.target_len for m in metas):,}")
    if missing:
        print(f"  missing row_idx (first 20): {sorted(missing)[:20]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
