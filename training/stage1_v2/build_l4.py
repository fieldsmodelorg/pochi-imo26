# Copyright 2026 proof-pilot. Apache-2.0.
"""Build L4: offline-rendered, globally-shuffled, pre-packed training bins.

Industrial-format data stage (Megatron .bin/.idx / OLMo memmap lineage): everything
the trainer's streaming path did per-rank at run time -- mix weighting (repeat +
fractional hash window), L3 render+mask, right-truncation to --max-len, greedy
packing into --micro-len rows -- is done ONCE here, with a GLOBAL row shuffle, and
written as fixed-shape records. The trainer then reads bins by a seeded global index
striped across ranks, which structurally removes the failure modes hit on 2026-06-05:

  * rank data imbalance (epoch ended when the lightest rank ran dry -- 13% / 67%
    utilization): bins are identical-cost units, every rank gets ceil/floor(N/world).
  * loss-curve nonstationarity (per-rank sequential shard walk): the global shuffle
    makes every bin an iid sample of the mix.
  * malformed upstream rows: dropped (loudly) at build time, not at step 455 of a
    64-GPU run.
  * train-time tokenizer CPU + replay-based resume: gone (bin cursor arithmetic).

Layout of <out>/:
  meta.json        params, counts, per-partition token shares, mix fingerprint
  input_ids.i32    [n_bins, micro_len] int32 (pad_id-padded rows)
  loss_mask.bits   [n_bins, micro_len/8] packed bits (1 = token carries loss)
  seg_ptr.i64      [n_bins+1] int64  -- ragged index into seg_lens
  seg_lens.i32     flat int32 doc lengths per bin (incl. trailing pad segment)

Reconstruction contract (must mirror olmo3_sink.sft_data.pack_to_tensors):
  labels       = where(mask, ids, -100)
  position_ids = concat(arange(L) for L in seg_lens[bin])   (pad segment included)
  n_docs       = len(seg_lens[bin]) - (1 if padded else 0)
  fixed-shape cu_seq_lens from seg_lens when max_segs is wanted.

Run (dev node, ~1 h; tokenize-bound):
  .venv/bin/python training/stage1_v2/build_l4.py \
      --roots data/nemotron-deepseek-sft-mix data/nemotron-deepseek-sft-mix-v2 \
      --mix training/stage1_v2/mix_g2.json \
      --tokenizer $DEEPSEEK_TOK_MODEL \
      --out data/l4-g2r05-ml12288-mc65536
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training"))                 # train_core
sys.path.insert(0, str(REPO / "training/stage1_v2/src"))   # data_mix

IGNORE = -100
N_BUCKETS = 64
REC_HDR = struct.Struct("<qiH")  # sigma int64, n_tok int32, partition_id uint16


# ---------------- pass A worker (per shard; runs in a process pool) ----------------
_W: dict = {}


def _winit(tokenizer_path: str, tmp_dir: str, max_len: int):
    """Pool initializer: per-worker tokenizer + one append-file per bucket."""
    from transformers import AutoTokenizer

    _W["tok"] = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _W["max_len"] = max_len
    pid = os.getpid()
    _W["files"] = [open(Path(tmp_dir) / f"b{b:02d}" / f"w{pid}.bin", "ab", buffering=1 << 20)
                   for b in range(N_BUCKETS)]


def _emit(sigma: int, part_id: int, ids: np.ndarray, mask: np.ndarray, n_bins_total: int):
    b = min(int(sigma * N_BUCKETS // n_bins_total), N_BUCKETS - 1)
    f = _W["files"][b]
    f.write(REC_HDR.pack(int(sigma), len(ids), part_id))
    f.write(ids.astype(np.int32).tobytes())
    f.write(np.packbits(mask).tobytes())


def render_shard(args) -> dict:
    """Render every selected row of one shard; emit one record per (row, sigma)."""
    shard, part_id, rows_idx, sigmas_per_row, n_total = args
    import pyarrow.parquet as pq

    from train_core.l3_render import render_and_mask

    tok, max_len = _W["tok"], _W["max_len"]
    want = dict(zip(rows_idx, sigmas_per_row))  # row -> list[sigma]
    kept = truncated = dropped = render_errors = 0
    tok_sum = 0
    pf = pq.ParquetFile(shard)
    row0 = 0
    for batch in pf.iter_batches(batch_size=256, columns=["messages", "tools"]):
        hit = [i for i in range(batch.num_rows) if (row0 + i) in want]
        if hit:
            msgs_col, tools_col = batch.column(0), batch.column(1)
            for i in hit:
                try:
                    msgs = json.loads(msgs_col[i].as_py())
                    traw = tools_col[i].as_py()
                    rendered, _ = render_and_mask(msgs, json.loads(traw) if traw else None,
                                                  tok, check_roundtrip=False)
                except Exception:  # noqa: BLE001 -- malformed upstream row: drop loudly
                    render_errors += 1
                    continue
                if rendered is None:
                    continue
                ids, labels = rendered.input_ids, rendered.labels
                if len(ids) > max_len:
                    ids, labels = ids[:max_len], labels[:max_len]
                    if all(l == IGNORE for l in labels):
                        dropped += 1
                        continue
                    truncated += 1
                kept += 1
                a = np.asarray(ids, dtype=np.int32)
                m = np.asarray(labels, dtype=np.int64) != IGNORE
                tok_sum += len(a) * len(want[row0 + i])
                for sigma in want[row0 + i]:
                    _emit(sigma, part_id, a, m, n_total)
        row0 += batch.num_rows
    for f in _W["files"]:
        f.flush()
    return {"kept": kept, "truncated": truncated, "dropped": dropped,
            "render_errors": render_errors, "tok": tok_sum, "part": part_id}


# ---------------- pass B: bucket -> sorted -> FFD pack -> memmap append ----------------
def read_bucket(bdir: Path) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """[(sigma, part_id, ids, mask_bool)] from all worker files of one bucket."""
    out = []
    for f in sorted(bdir.glob("w*.bin")):
        buf = f.read_bytes()
        off = 0
        while off < len(buf):
            sigma, n, pid = REC_HDR.unpack_from(buf, off)
            off += REC_HDR.size
            ids = np.frombuffer(buf, dtype=np.int32, count=n, offset=off)
            off += 4 * n
            nb = (n + 7) // 8
            mask = np.unpackbits(np.frombuffer(buf, dtype=np.uint8, count=nb, offset=off),
                                 count=n).astype(bool)
            off += nb
            out.append((sigma, pid, ids, mask))
    out.sort(key=lambda r: r[0])
    return out


def ffd_pack(items: list, cap: int) -> list[list]:
    """First-fit-decreasing -- mirrors sft_data.greedy_pack (records = (sigma,pid,ids,mask))."""
    items = sorted(items, key=lambda r: len(r[2]), reverse=True)
    bins: list[list] = []
    fill: list[int] = []
    for r in items:
        L = len(r[2])
        for i in range(len(bins)):
            if fill[i] + L <= cap:
                bins[i].append(r)
                fill[i] += L
                break
        else:
            bins.append([r])
            fill.append(L)
    return bins


def pack_stream(items: list, carry: list, cap: int, window: int, min_fill: float,
                last: bool) -> tuple[list[list], list]:
    """sigma-ordered windowed FFD: full bins out, underfull bins' items carried on.

    A whole-bucket FFD (~100k items) is O(n x bins) pure Python -- too slow. Windowed
    FFD at 2048 keeps fill ~99% (training used 256-windows at ~98%) and stays fast;
    underfull bins (< min_fill) are recycled into the next window/bucket so only the
    very last window may emit short bins."""
    out: list[list] = []
    pending = carry + items
    while pending:
        win, pending = pending[:window], pending[window:]
        bins = ffd_pack(win, cap)
        if pending or not last:
            keep = []
            for bn in bins:
                if sum(len(r[2]) for r in bn) >= min_fill * cap:
                    out.append(bn)
                else:
                    keep.extend(bn)
            if pending:
                pending = keep + pending
            else:
                return out, keep
        else:
            out.extend(bins)
    return out, []


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--micro-len", type=int, default=65536)
    ap.add_argument("--pad-id", type=int, default=None, help="default: tokenizer.pad_token_id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=0, help="frac-window epoch to materialize")
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--min-fill", type=float, default=0.98,
                    help="bins below this fill are recycled into the next bucket")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    from data_mix import build_shard_tasks, keep_row, load_mix, mix_fingerprint

    t0 = time.time()
    roots = [Path(r) for r in args.roots]
    entries = load_mix(args.mix)
    out = Path(args.out)
    tmp = out / "_tmp"
    for b in range(N_BUCKETS):
        (tmp / f"b{b:02d}").mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    pad_id = args.pad_id
    if pad_id is None:
        pad_id = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True).pad_token_id

    # ---- pass 0: enumerate selected (shard,row) pointers, global sigma permutation ----
    tasks = build_shard_tasks(roots, entries, args.epoch)
    by_path: dict[Path, list] = {}
    for t in tasks:
        by_path.setdefault(t.path, []).append(t)
    part_names: list[str] = []
    part_ids: dict[str, int] = {}
    pointers: list[tuple[int, int]] = []  # (shard_no, row_idx) one per copy kept
    shard_list = sorted(by_path)
    for sno, path in enumerate(shard_list):
        copies = by_path[path]
        pk = f"{path.parents[2].name}::{path.parent.parent.name.split('=',1)[1]}/" \
             f"{path.parent.name.split('=',1)[1]}"
        if pk not in part_ids:
            part_ids[pk] = len(part_names)
            part_names.append(pk)
        fracs = [t.frac for t in copies]
        if any(f is not None for f in fracs):
            # ParquetFile.read, NOT pq.read_table: the latter runs hive-partition inference
            # on the dataset=*/domain=* path and collides with the L2 `dataset` column.
            ids = pq.ParquetFile(str(path)).read(columns=["id"]).column(0).to_pylist()
            for r, rid in enumerate(ids):
                for f in fracs:
                    if f is None or keep_row(rid, f, args.epoch):
                        pointers.append((sno, r))
        else:
            n = pq.ParquetFile(str(path)).metadata.num_rows
            for r in range(n):
                pointers.extend([(sno, r)] * len(fracs))
    n_total = len(pointers)
    rng = np.random.RandomState(args.seed + 1009 * args.epoch)
    sigma = rng.permutation(n_total)
    print(f"[pass0] {n_total:,} selected row-copies from {len(shard_list)} shards "
          f"({time.time()-t0:.0f}s)", flush=True)

    # group per shard: row -> [sigmas]
    per_shard: dict[int, dict[int, list[int]]] = {}
    for p, ((sno, r), sg) in enumerate(zip(pointers, sigma)):
        per_shard.setdefault(sno, {}).setdefault(r, []).append(int(sg))
    del pointers, sigma

    # ---- pass A: parallel render -> bucket files ----
    from concurrent.futures import ProcessPoolExecutor

    jobs = []
    for sno, want in per_shard.items():
        path = shard_list[sno]
        pk = f"{path.parents[2].name}::{path.parent.parent.name.split('=',1)[1]}/" \
             f"{path.parent.name.split('=',1)[1]}"
        rows_idx = np.fromiter(want.keys(), dtype=np.int64)
        order = np.argsort(rows_idx)
        rows_idx = rows_idx[order]
        sigmas = [want[int(r)] for r in rows_idx]
        jobs.append((str(path), part_ids[pk], rows_idx.tolist(), sigmas, n_total))
    jobs.sort(key=lambda j: -len(j[2]))  # big shards first (pool tail latency)

    stats = Counter()
    part_tok = Counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_winit,
                             initargs=(args.tokenizer, str(tmp), args.max_len)) as ex:
        for i, res in enumerate(ex.map(render_shard, jobs, chunksize=1)):
            for k in ("kept", "truncated", "dropped", "render_errors"):
                stats[k] += res[k]
            part_tok[part_names[res["part"]]] += res["tok"]
            if (i + 1) % 25 == 0:
                print(f"[passA] {i+1}/{len(jobs)} shards, kept={stats['kept']:,} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[passA] done: {dict(stats)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- pass B: per bucket sort by sigma -> FFD pack -> append memmaps ----
    f_ids = open(out / "input_ids.i32", "wb")
    f_msk = open(out / "loss_mask.bits", "wb")
    seg_lens: list[int] = []
    seg_ptr: list[int] = [0]
    n_bins = n_docs_total = 0
    carry: list = []
    cap, mbytes = args.micro_len, args.micro_len // 8
    for b in range(N_BUCKETS):
        bins, carry = pack_stream(read_bucket(tmp / f"b{b:02d}"), carry, cap,
                                  window=2048, min_fill=args.min_fill,
                                  last=(b == N_BUCKETS - 1))
        for bn in bins:
            ids = np.full(cap, pad_id, dtype=np.int32)
            msk = np.zeros(cap, dtype=bool)
            offp = 0
            lens = []
            for _sg, pid, a, m in bn:
                ids[offp:offp + len(a)] = a
                msk[offp:offp + len(a)] = m
                lens.append(len(a))
                offp += len(a)
            if offp < cap:
                lens.append(cap - offp)  # trailing pad segment (position-reset, IGNORE)
            f_ids.write(ids.tobytes())
            f_msk.write(np.packbits(msk).tobytes())
            seg_lens.extend(lens)
            seg_ptr.append(len(seg_lens))
            n_bins += 1
            n_docs_total += len(bn)
        print(f"[passB] bucket {b+1}/{N_BUCKETS}: bins so far {n_bins:,} "
              f"({time.time()-t0:.0f}s)", flush=True)
    f_ids.close()
    f_msk.close()
    np.asarray(seg_ptr, dtype=np.int64).tofile(out / "seg_ptr.i64")
    np.asarray(seg_lens, dtype=np.int32).tofile(out / "seg_lens.i32")

    tot_tok = sum(part_tok.values())
    meta = {
        "format": "proof-pilot-l4-v1",
        "micro_len": args.micro_len,
        "max_len": args.max_len,
        "pad_id": int(pad_id),
        "seed": args.seed,
        "epoch_materialized": args.epoch,
        "n_bins": n_bins,
        "n_docs": n_docs_total,
        "tokenizer": args.tokenizer,
        "mix": args.mix,
        "mix_fingerprint": mix_fingerprint(roots, entries),
        "stats": dict(stats),
        "partition_token_share": {k: v / tot_tok for k, v in part_tok.most_common()},
        "fill": tot_tok / (n_bins * args.micro_len) if n_bins else 0.0,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    import shutil
    shutil.rmtree(tmp)
    print(f"[done] {n_bins:,} bins / {n_docs_total:,} docs / {tot_tok/1e9:.2f}B tok "
          f"(fill {meta['fill']:.1%}) in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
