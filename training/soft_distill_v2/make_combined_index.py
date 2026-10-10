# Copyright 2026 proof-pilot. Apache-2.0.
"""Concatenate several teacher-hidden pools into one combined index for multi-pool training.

Each pool's `hidden/index.jsonl` references its per-doc `.pt` by ABSOLUTE path and carries
the same codec/teacher/dim, so pools are mixable by plain concatenation (EXTRACT doc §5). The
only fixup is `row_idx`: each pool numbers from 0, so we renumber globally. Every `.pt` path is
verified to exist (fail loud on a missing shard).

    python make_combined_index.py OUT.jsonl POOL1/hidden/index.jsonl POOL2/hidden/index.jsonl ...

Then point the trainer at the result: `--index OUT.jsonl`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        sys.exit(f"usage: {argv[0]} OUT.jsonl INDEX1.jsonl [INDEX2.jsonl ...]")
    out = Path(argv[1])
    srcs = [Path(p) for p in argv[2:]]
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    per_src = []
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w") as w:
        for s in srcs:
            c = 0
            for line in open(s):
                d = json.loads(line)
                if not os.path.exists(d["path"]):
                    sys.exit(f"missing .pt referenced by {s}: {d['path']}")
                d["row_idx"] = n
                n += 1
                c += 1
                w.write(json.dumps(d) + "\n")
            per_src.append((str(s), c))
    os.replace(tmp, out)
    print(f"wrote {out}: {n} rows")
    for s, c in per_src:
        print(f"  {c:>7} <- {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
