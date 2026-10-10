# Copyright 2026 proof-pilot. Apache-2.0.
"""Merge extracted-hidden shard indices for offline training."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from offpolicy_data import ExtractedDocMeta, read_index, write_index  # noqa: E402

DEFAULT_GLOB = THIS_DIR / "work" / "dsflash-test" / "hidden_shards" / "shard_*" / "index.jsonl"
DEFAULT_OUT = THIS_DIR / "work" / "dsflash-test" / "hidden" / "index.jsonl"


def expand_inputs(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        matches = glob.glob(item)
        if not matches:
            matches = [item]
        for m in matches:
            p = Path(m)
            out.append(p / "index.jsonl" if p.is_dir() else p)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="index.jsonl files, shard dirs, or globs")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    inputs = expand_inputs(args.inputs or [str(DEFAULT_GLOB)])
    rows: dict[int, ExtractedDocMeta] = {}
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(p)
        for r in read_index(p):
            old = rows.get(r.row_idx)
            if old is not None:
                raise ValueError(f"duplicate row_idx={r.row_idx}: {old.path} and {r.path}")
            rows[r.row_idx] = r
    merged = [rows[k] for k in sorted(rows)]
    if not merged:
        raise SystemExit("no rows merged")
    write_index(args.out, merged)
    meta = {
        "inputs": [str(p) for p in inputs],
        "out": str(args.out),
        "n_rows": len(merged),
        "target_tokens": sum(r.target_len for r in merged),
        "teacher_rows": sum(r.teacher_seq_len for r in merged),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / "merge_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[merge] wrote {len(merged)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
