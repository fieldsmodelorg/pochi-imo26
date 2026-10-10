# Copyright 2026 proof-pilot. Apache-2.0.
"""Build an L4 dataset from ALREADY-TOKENIZED sources (skip the render pass).

`stage1_v2/build_l4.py` renders L2 HF parquet (`messages`/`tools`) into L4 bins.
This builder takes sources that are *already tokenized* and only need the pass-B
FFD pack into `proof-pilot-l4-v1` bins that `training/dflash/data.py` reads:

  1. OPD rollout parquet (`opd_v2 rollout_store`): columns
     `input_ids: list<int32>`, `loss_mask: list<bool>`, `meta: str(JSON)`.
     Rows whose `meta.finish_reason` is in --drop-finish-reasons are dropped
     (default "length": cap-truncated / degenerate-loop rollouts — see
     opd_v2 DEEP_REVIEW_32B.md finding B3; their loss_mask is all-True over the
     truncated/looping completion and would teach the draft a bad distribution).
  2. dsflash teacher manifest (`teacher_extract/dataprep render_manifest`): columns
     `input_ids: list<int32>`, `prompt_len`, `seq_len`. loss_mask is constructed
     as [False]*prompt_len + [True]*(seq_len-prompt_len) (assistant proof span).

Both are vocab-129280 / DeepSeek-tokenizer aligned with the olmo3_sink 32B target,
so input_ids need no re-tokenization. Output layout is byte-identical to
build_l4.py (mirror its reconstruction contract):

  meta.json / input_ids.i32 / loss_mask.bits / seg_ptr.i64 / seg_lens.i32

Run (compute node):
  .venv/bin/python training/dflash/build_l4_pretokenized.py \
     --opd-roots training/opd_v2/runs/agentic_32b_lc140k/rollouts \
                 training/opd_v2/runs/agentic_32b_lc140k_v33/rollouts \
     --teacher-manifest training/teacher_extract/dataprep/work/dsflash-v2-test/manifest.parquet \
     --out data/l4-dflash32b-opd+dsflash-ml65536 \
     --max-len 65536 --micro-len 65536 --pad-id 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "training/stage1_v2"))
from build_l4 import pack_stream  # noqa: E402  (windowed-FFD packer; reused verbatim)


def _iter_opd(roots: list[str], drop_finish: set[str], stats: Counter):
    """Yield (ids:int32, mask:bool) per OPD rollout, dropping cap-hit finish reasons."""
    import pyarrow.parquet as pq

    files: list[str] = []
    for r in roots:
        files += sorted(glob.glob(os.path.join(r, "*.parquet")))
    for f in files:
        t = pq.read_table(f, columns=["input_ids", "loss_mask", "meta"])
        ids_col = t.column("input_ids").to_pylist()
        msk_col = t.column("loss_mask").to_pylist()
        meta_col = t.column("meta").to_pylist()
        for ids, msk, m in zip(ids_col, msk_col, meta_col):
            fr = ""
            if m:
                try:
                    fr = json.loads(m).get("finish_reason", "")
                except Exception:  # noqa: BLE001
                    fr = ""
            if fr in drop_finish:
                stats["opd_dropped_finish"] += 1
                continue
            yield (np.asarray(ids, dtype=np.int32),
                   np.asarray(msk, dtype=bool))
            stats["opd_kept"] += 1


def _iter_teacher(manifest: str, stats: Counter):
    """Yield (ids:int32, mask:bool) per teacher proof; mask = completion span."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(manifest)
    for batch in pf.iter_batches(batch_size=512, columns=["input_ids", "prompt_len", "seq_len"]):
        ids_col = batch.column(0).to_pylist()
        pl_col = batch.column(1).to_pylist()
        sl_col = batch.column(2).to_pylist()
        for ids, pl, sl in zip(ids_col, pl_col, sl_col):
            pl, sl = int(pl), int(sl)
            if pl >= sl:                      # no completion -> no loss token
                stats["teacher_skipped_noloss"] += 1
                continue
            a = np.asarray(ids, dtype=np.int32)
            m = np.arange(len(a)) >= pl       # prompt=False, completion(+EOS)=True
            yield (a, m)
            stats["teacher_kept"] += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opd-roots", nargs="*", default=[], help="OPD rollouts dir(s) (glob *.parquet)")
    ap.add_argument("--teacher-manifest", default=None, help="dsflash manifest.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=65536, help="per-doc right-truncation cap")
    ap.add_argument("--micro-len", type=int, default=65536, help="fixed bin length")
    ap.add_argument("--pad-id", type=int, default=2)
    ap.add_argument("--tokenizer", default=os.environ.get("DFLASH_TOKENIZER", "outputs/stage1-v2-32b-softdistill-v2test"),
                    help="informational (recorded in meta.json)")
    ap.add_argument("--drop-finish-reasons", nargs="*", default=["length"],
                    help="OPD finish_reason values to drop (default: length)")
    ap.add_argument("--opd-repeat", type=int, default=1)
    ap.add_argument("--teacher-repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-fill", type=float, default=0.98)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    drop_finish = set(args.drop_finish_reasons)
    stats: Counter = Counter()
    cap = args.micro_len

    # ---- collect already-tokenized docs (truncate to max_len, drop empty-loss) ----
    docs: list[tuple[int, int, np.ndarray, np.ndarray]] = []  # (pid, _, ids, mask) -- pid:0 teacher,1 opd
    tok_by_part: Counter = Counter()

    def _add(ids: np.ndarray, mask: np.ndarray, pid: int, repeat: int):
        if len(ids) > args.max_len:
            ids, mask = ids[: args.max_len], mask[: args.max_len]
            stats[f"truncated_p{pid}"] += 1
        if not mask.any():
            stats[f"dropped_noloss_p{pid}"] += 1
            return
        for _ in range(repeat):
            docs.append((pid, 0, ids, mask))
            tok_by_part[pid] += len(ids)

    if args.teacher_manifest:
        for ids, mask in _iter_teacher(args.teacher_manifest, stats):
            _add(ids, mask, 0, args.teacher_repeat)
    if args.opd_roots:
        for ids, mask in _iter_opd(args.opd_roots, drop_finish, stats):
            _add(ids, mask, 1, args.opd_repeat)

    n_docs = len(docs)
    if n_docs == 0:
        raise SystemExit("no docs collected")
    print(f"[collect] {n_docs:,} docs ({time.time()-t0:.0f}s) stats={dict(stats)}", flush=True)

    # ---- global shuffle (sigma) then windowed-FFD pack into micro_len bins ----
    rng = np.random.RandomState(args.seed)
    sigma = rng.permutation(n_docs)
    items = [(int(sigma[i]), docs[i][0], docs[i][2], docs[i][3]) for i in range(n_docs)]
    items.sort(key=lambda r: r[0])
    bins, leftover = pack_stream(items, [], cap, window=2048, min_fill=args.min_fill, last=True)
    # last=True drains everything (final window takes the else-branch, no recycling),
    # so leftover is always []. Assert the invariant rather than append it: leftover is
    # a flat item list, not a bin, so bins.append(leftover) would corrupt pass-B.
    assert not leftover, f"pack_stream(last=True) left {len(leftover)} unpacked items"
    print(f"[pack] {len(bins):,} bins ({time.time()-t0:.0f}s)", flush=True)

    # ---- pass-B write (mirror build_l4.py exactly) ----
    f_ids = open(out / "input_ids.i32", "wb")
    f_msk = open(out / "loss_mask.bits", "wb")
    seg_lens: list[int] = []
    seg_ptr: list[int] = [0]
    n_bins = n_docs_total = 0
    part_tok: Counter = Counter()
    for bn in bins:
        ids = np.full(cap, args.pad_id, dtype=np.int32)
        msk = np.zeros(cap, dtype=bool)
        offp = 0
        lens: list[int] = []
        for _sg, pid, a, m in bn:
            ids[offp : offp + len(a)] = a
            msk[offp : offp + len(a)] = m
            lens.append(len(a))
            part_tok[pid] += len(a)
            offp += len(a)
        if offp < cap:
            lens.append(cap - offp)  # trailing pad segment (position-reset, IGNORE)
        f_ids.write(ids.tobytes())
        f_msk.write(np.packbits(msk).tobytes())
        seg_lens.extend(lens)
        seg_ptr.append(len(seg_lens))
        n_bins += 1
        n_docs_total += len(bn)
    f_ids.close()
    f_msk.close()
    np.asarray(seg_ptr, dtype=np.int64).tofile(out / "seg_ptr.i64")
    np.asarray(seg_lens, dtype=np.int32).tofile(out / "seg_lens.i32")

    tot_tok = sum(part_tok.values())
    part_name = {0: "teacher:dsflash-v2-test", 1: "opd:rollouts"}
    meta = {
        "format": "proof-pilot-l4-v1",
        "micro_len": args.micro_len,
        "max_len": args.max_len,
        "pad_id": int(args.pad_id),
        "seed": args.seed,
        "n_bins": n_bins,
        "n_docs": n_docs_total,
        "tokenizer": args.tokenizer,
        "sources": {"opd_roots": args.opd_roots, "teacher_manifest": args.teacher_manifest,
                    "drop_finish_reasons": sorted(drop_finish),
                    "opd_repeat": args.opd_repeat, "teacher_repeat": args.teacher_repeat},
        "stats": dict(stats),
        "partition_token_share": {part_name[p]: v / tot_tok for p, v in part_tok.most_common()},
        "fill": tot_tok / (n_bins * args.micro_len) if n_bins else 0.0,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[done] {n_bins:,} bins / {n_docs_total:,} docs / {tot_tok/1e9:.2f}B tok "
          f"(fill {meta['fill']:.1%}) in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
