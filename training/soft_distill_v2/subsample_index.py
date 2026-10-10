# Copyright 2026 proof-pilot. Apache-2.0.
"""Token-budgeted random subsample of a teacher-hidden pool index for replay mixing.

Picks rows uniformly at random (fixed seed) from a pool's `hidden/index.jsonl` until the
cumulative `target_len` reaches a token budget, then writes the selected ORIGINAL json lines
(paths/lengths untouched; make_combined_index.py renumbers row_idx globally). Used to build a
small anti-forgetting replay slice (e.g. ~50M tok from dsflash-v2-test) to mix with a new
emphasis pool.

    python subsample_index.py SRC/hidden/index.jsonl OUT.jsonl --budget-tok 50000000 --seed 20260623
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="source pool hidden/index.jsonl")
    ap.add_argument("out", type=Path)
    ap.add_argument("--budget-tok", type=int, required=True, help="stop once cumulative target_len >= this")
    ap.add_argument("--seed", type=int, default=20260623)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.src)]
    random.Random(args.seed).shuffle(rows)

    picked: list[dict] = []
    tok = 0
    for r in rows:
        picked.append(r)
        tok += int(r["target_len"])
        if tok >= args.budget_tok:
            break

    picked.sort(key=lambda r: r["row_idx"])  # stable, original order
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(tmp, "w") as w:
        for r in picked:
            w.write(json.dumps(r) + "\n")
    tmp.replace(args.out)

    print(f"src={args.src} total_rows={len(rows)}")
    print(f"picked {len(picked)} rows / {tok:,} target tok (budget {args.budget_tok:,}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
