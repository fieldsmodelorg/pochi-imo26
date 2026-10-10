# Copyright 2026 proof-pilot. Apache-2.0.
"""Weighted multi-root data mix for stage-1 v2 (see ../PLAN.md §1).

A mix config (JSON) assigns a `repeat` weight to hive partitions
(`dataset=<d>/domain=<dom>`) across one or more L2 roots:

    {"entries": [
        {"root": "nemotron-deepseek-sft-mix",    "include": "*",                      "repeat": 0.5},
        {"root": "nemotron-deepseek-sft-mix-v2", "include": "nemotron-sft-math-v4/*", "repeat": 2}
    ]}

`root` matches the basename of a --dataset_path root; `include` is an fnmatch
glob over the partition key "<dataset>/<domain>". Unmatched partitions default
to repeat=1. Validation is fail-loud: an entry whose root is not provided or
whose include matches no partition is a config bug, and two entries matching
the same partition is an ambiguity -- all three raise.

Repeat semantics (repeat = k + f, integer k >= 0, fraction f in [0, 1)):
  - every shard of the partition appears k times unfiltered, plus (if f > 0)
    once more with a ROW-level filter that keeps fraction f of rows.
  - The row filter is a deterministic hash window, not a shard subset: each
    row's stable L2 `id` hashes to u in [0,1) and epoch e keeps the rows with
    ((u - e*f) mod 1) < f. Consecutive epochs slide the window, so for f=0.5
    epoch 0 and epoch 1 keep complementary halves and ceil(1/f) epochs cover
    the partition exactly once. Proportions are exact to ~1/sqrt(rows)
    (law of large numbers), independent of shard boundaries, and stable under
    re-sharding (the `id` column is content-derived).

Everything here is a pure function of (roots, entries, epoch) -- the shard
task list and the per-row decision are reproducible, so the trainer's
deterministic resume (bin-count reconstruction skip) keeps working unchanged.
`mix_fingerprint` is stored in the resume meta to fail loud if the mix changes
between continuations of one run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Optional

SHARD_GLOB = "dataset=*/domain=*/*.parquet"
_HASH_KEY = b"proof-pilot-mix-v1"  # keyed blake2b -> stable across runs/machines


# ---- row-level fractional filter ----
def _u01(row_id: str) -> float:
    """Stable hash of a row id to [0, 1). Keyed so it is independent of any
    other use of the id, and reproducible across Python processes (unlike hash())."""
    h = hashlib.blake2b(str(row_id).encode(), digest_size=8, key=_HASH_KEY).digest()
    return int.from_bytes(h, "big") / 2**64


def keep_row(row_id: str, frac: float, epoch: int) -> bool:
    """Epoch-sliding hash window: keeps `frac` of rows; consecutive epochs are
    complementary (ceil(1/frac) epochs cover everything exactly once)."""
    return (_u01(row_id) - epoch * frac) % 1.0 < frac


# ---- mix resolution ----
@dataclass(frozen=True)
class ShardTask:
    """One pass over one parquet shard; `frac` set -> keep that row fraction."""
    path: Path
    frac: Optional[float] = None  # None = all rows


def load_mix(path: str) -> list[dict]:
    """Read a mix JSON; returns its entries (underscore keys are commentary)."""
    with open(path) as f:
        cfg = json.load(f)
    entries = cfg.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"mix config {path} has no 'entries' list")
    for e in entries:
        missing = {"root", "include", "repeat"} - e.keys()
        if missing:
            raise ValueError(f"mix entry {e} missing keys {sorted(missing)}")
        if not (isinstance(e["repeat"], (int, float)) and e["repeat"] >= 0):
            raise ValueError(f"mix entry {e}: repeat must be a number >= 0 (0 = exclude)")
    return entries


def partitions(root: Path) -> dict[str, list[Path]]:
    """{ "<dataset>/<domain>": sorted [shard, ...] } under one L2 root."""
    parts: dict[str, list[Path]] = {}
    for shard in sorted(root.glob(SHARD_GLOB)):
        key = (f"{shard.parent.parent.name.split('=', 1)[1]}/"
               f"{shard.parent.name.split('=', 1)[1]}")
        parts.setdefault(key, []).append(shard)
    if not parts:
        raise FileNotFoundError(f"no parquet under {root}/{SHARD_GLOB}")
    return parts


def resolve_repeats(roots: list[Path], entries: list[dict],
                    ) -> dict[tuple[str, str], float]:
    """{(root_basename, partition_key): repeat} for every partition of every root.

    Fail-loud validation (config bugs must not silently change the data mix):
    duplicate root basenames, entry root not among `roots`, entry include
    matching no partition, and two entries matching the same partition.
    """
    names = [r.name for r in roots]
    if len(set(names)) != len(names):
        raise ValueError(f"dataset roots have duplicate basenames: {names}")
    parts_by_root = {r.name: partitions(r) for r in roots}

    repeats: dict[tuple[str, str], float] = {
        (rn, pk): 1.0 for rn, parts in parts_by_root.items() for pk in parts}
    claimed: dict[tuple[str, str], int] = {}
    for i, e in enumerate(entries):
        rn = e["root"]
        if rn not in parts_by_root:
            raise ValueError(f"mix entry {i} root {rn!r} not among dataset roots {names}")
        hits = [pk for pk in parts_by_root[rn] if fnmatchcase(pk, e["include"])]
        if not hits:
            raise ValueError(f"mix entry {i} include {e['include']!r} matches no "
                             f"partition of {rn} (have: {sorted(parts_by_root[rn])})")
        for pk in hits:
            if (rn, pk) in claimed:
                raise ValueError(f"mix entries {claimed[(rn, pk)]} and {i} both match "
                                 f"partition {rn}::{pk} -- overlapping includes")
            claimed[(rn, pk)] = i
            repeats[(rn, pk)] = float(e["repeat"])
    return repeats


def build_shard_tasks(roots: list[Path], entries: list[dict], epoch: int,
                      ) -> list[ShardTask]:
    """Deterministic (pre-shuffle) shard task list for one epoch.

    repeat = k + f -> k unfiltered passes per shard + one f-window pass.
    `epoch` only matters in that the CALLER passes it to keep_row via the task's
    frac; the task list itself is epoch-independent (the window slides per epoch
    inside keep_row). Order: roots as given, partitions sorted, shards sorted.
    """
    repeats = resolve_repeats(roots, entries)
    tasks: list[ShardTask] = []
    for root in roots:
        for pk, shards in sorted(partitions(root).items()):
            rep = repeats[(root.name, pk)]
            k, f = int(rep), rep - int(rep)
            for shard in shards:
                tasks.extend(ShardTask(shard) for _ in range(k))
                if f > 1e-9:
                    tasks.append(ShardTask(shard, frac=f))
    return tasks


def assign_tasks(tasks: list[ShardTask], world: int) -> list[list[ShardTask]]:
    """Deterministic size-balanced task assignment: LPT greedy into `world` buckets.

    Round-robin by index proved badly unbalanced (2026-06-05 job 79398): task sizes
    span ~25x (v1 ~70k-row fat shards vs v2 small shards vs fractional half-windows),
    so at 64 ranks the lightest rank exhausted its tasks at ~13% of the epoch and the
    rank-synced MIN-stop ended epoch 0 for everyone. LPT (sort by estimated rows desc,
    assign to the currently lightest bucket) bounds the imbalance to a few percent.

    Estimated rows = parquet metadata num_rows x (frac or 1) -- metadata-only, cheap.
    Pure function of (tasks, world): per-rank streams stay deterministic for resume.
    Note the assignment is epoch-independent (a rank keeps the same shards across
    epochs; within-rank order is still epoch-shuffled and frac windows slide).
    """
    import heapq

    import pyarrow.parquet as pq

    rows_cache: dict[Path, int] = {}

    def est(t: ShardTask) -> float:
        if t.path not in rows_cache:
            rows_cache[t.path] = pq.ParquetFile(str(t.path)).metadata.num_rows
        return rows_cache[t.path] * (t.frac if t.frac is not None else 1.0)

    sized = sorted(((est(t), i, t) for i, t in enumerate(tasks)),
                   key=lambda x: (-x[0], x[1]))  # size desc, original index tiebreak
    heap = [(0.0, r) for r in range(world)]  # (assigned rows, rank)
    heapq.heapify(heap)
    buckets: list[list[ShardTask]] = [[] for _ in range(world)]
    for rows, _i, t in sized:
        load, r = heapq.heappop(heap)
        buckets[r].append(t)
        heapq.heappush(heap, (load + rows, r))
    return buckets


def count_mix_docs(roots: list[Path], entries: list[dict]) -> float:
    """Sum of rows x repeat over all partitions (parquet metadata only).

    The fractional part is an expectation (the hash window keeps ~f of rows);
    over millions of rows the error is negligible for a progress metric."""
    import pyarrow.parquet as pq

    repeats = resolve_repeats(roots, entries)
    total = 0.0
    for root in roots:
        for pk, shards in partitions(root).items():
            rep = repeats[(root.name, pk)]
            total += rep * sum(pq.ParquetFile(str(s)).metadata.num_rows for s in shards)
    return total


def mix_fingerprint(roots: list[Path], entries: list[dict]) -> str:
    """Stable hash of the RESOLVED mix (per-partition repeats). Stored in the
    resume meta; a continuation with a different mix must fail loud, because the
    deterministic bin-skip resume replays the data stream from the mix."""
    repeats = resolve_repeats(roots, entries)
    canon = json.dumps(sorted((rn, pk, rep) for (rn, pk), rep in repeats.items()))
    return hashlib.blake2b(canon.encode(), digest_size=8).hexdigest()


def describe(roots: list[Path], entries: list[dict]) -> str:
    """Human-readable partition -> repeat/shards table (rank-0 startup log)."""
    repeats = resolve_repeats(roots, entries)
    lines = ["data mix (partition | repeat | shards):"]
    for root in roots:
        for pk, shards in sorted(partitions(root).items()):
            rep = repeats[(root.name, pk)]
            lines.append(f"  {root.name}::{pk:48} x{rep:<4g} {len(shards)} shards")
    return "\n".join(lines)
