# Copyright 2026 proof-pilot. Apache-2.0.
"""Multiprocess version of render_manifest.py for the generation-schema HF datasets.

render_manifest.py is single-threaded; tokenizing ~1.4B token-encodes for the 53k-turn
`ycchen/dsflash-proof-distill-v2-test` per_turn config takes ~2h on one core. This driver
reuses render_manifest's PROVEN, gate-keeping `clean_row` / `encode_row` / `write_manifest`
verbatim (same strict `prompt_len == api prompt_tokens` fidelity check, same `effort=="max"`
prefix handling) and only fans the per-row tokenization out to a worker pool.

Input must be a parquet with the generation schema (messages_json + reasoning_content +
content + prompt_tokens, like render_manifest.py expects). row_idx is the ORIGINAL row
position in the input parquet (stable, unique; gaps where rows are dropped) so that
extract_hidden's `row_idx % num_shards` sharding is deterministic.

  .venv/bin/python training/teacher_extract/dataprep/render_manifest_mp.py \
      --input .../dsflash-v2-test/raw/data/per_turn.parquet \
      --out   .../dsflash-v2-test/manifest.parquet \
      --max-seq-len 200000 --workers 47
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
import os
from pathlib import Path
import statistics
import sys

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
for _p in (REPO / "training", THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse the validated single-threaded renderer's logic unchanged.
from render_manifest import clean_row, encode_row, write_manifest, _git_commit  # noqa: E402

DEFAULT_TOKENIZER = os.environ.get("DEEPSEEK_V4_FLASH", "/models/DeepSeek-V4-Flash")
DEFAULT_MAX_SEQ_LEN = 200_000

_TOK = None


def _init_worker(tokenizer_path: str) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _TOK
    from transformers import PreTrainedTokenizerFast
    _TOK = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def _render_task(task):
    src_idx, row, include_truncated = task
    if not include_truncated and not clean_row(row):
        return (src_idx, None, "unclean")
    try:
        enc = encode_row(row, _TOK)
        enc["row_idx"] = src_idx
        return (src_idx, enc, "ok")
    except Exception as exc:  # gate failures / template drift -> count, do not abort the run
        return (src_idx, None, f"err:{exc}")


def _iter_rows(path: Path, batch_size: int):
    """Yield (orig_idx, row_dict). iter_batches avoids the single-row-group 2GB string
    offset overflow that table.take()/read_table().to_pylist() hits on this file."""
    pf = pq.ParquetFile(path)
    idx = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield idx, row
            idx += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="generation-schema parquet (per_turn)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--batch-size", type=int, default=64, help="parquet read batch (memory vs throughput)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N input rows")
    ap.add_argument("--include-truncated", action="store_true",
                    help="debug only: keep error/truncated rows (production drops them)")
    args = ap.parse_args()

    from transformers import PreTrainedTokenizerFast
    PreTrainedTokenizerFast.from_pretrained(args.tokenizer)  # fail fast if tokenizer path is wrong

    gen = _iter_rows(args.input, args.batch_size)
    tasks = ((i, r, args.include_truncated) for i, r in gen
             if not args.limit or i < args.limit)

    rows: list[dict] = []
    n_seen = 0
    counts = {"ok": 0, "unclean": 0, "err": 0, "too_long": 0}
    err_samples: list[str] = []
    print(f"[render-mp] input={args.input} workers={args.workers} max_seq_len={args.max_seq_len}", flush=True)
    with Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        for src_idx, enc, status in pool.imap_unordered(_render_task, tasks, chunksize=4):
            n_seen += 1
            if status == "ok":
                if enc["seq_len"] > args.max_seq_len:
                    counts["too_long"] += 1
                else:
                    counts["ok"] += 1
                    rows.append(enc)
            elif status == "unclean":
                counts["unclean"] += 1
            else:
                counts["err"] += 1
                if len(err_samples) < 10:
                    err_samples.append(f"row {src_idx}: {status}")
            if n_seen % 5000 == 0:
                kept_tok = sum(r["target_len"] for r in rows)
                print(f"[render-mp] seen={n_seen} kept={counts['ok']} too_long={counts['too_long']} "
                      f"unclean={counts['unclean']} err={counts['err']} target_tok={kept_tok:,}", flush=True)

    if not rows:
        raise SystemExit("no rows rendered")
    rows.sort(key=lambda r: r["row_idx"])  # deterministic order by original input position

    target_lens = [r["target_len"] for r in rows]
    seq_lens = [r["seq_len"] for r in rows]
    deltas: dict[int, int] = {}
    for r in rows:
        d = r["completion_delta"]
        deltas[d] = deltas.get(d, 0) + 1
    meta = {
        "renderer": "render_manifest_mp.py",
        "input": str(args.input),
        "tokenizer": args.tokenizer,
        "max_seq_len": args.max_seq_len,
        "n_seen": n_seen,
        "n_rows": len(rows),
        "counts": counts,
        "target_tokens": int(sum(target_lens)),
        "seq_tokens": int(sum(seq_lens)),
        "completion_delta_counts": deltas,
        "est_flash_storage_gb": round(sum(target_lens) * 3328 / 1e9, 1),
        "git_commit": _git_commit(),
        "err_samples": err_samples,
    }
    write_manifest(rows, args.out, meta)
    (args.out.parent / "manifest_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[manifest] wrote {len(rows)} rows -> {args.out}\n"
        f"  counts={counts}\n"
        f"  target_tokens={sum(target_lens):,} seq_tokens={sum(seq_lens):,}\n"
        f"  seq_len mean={statistics.mean(seq_lens):.0f} p50={statistics.median(seq_lens):.0f} "
        f"max={max(seq_lens)}\n"
        f"  completion_delta_counts={deltas}\n"
        f"  est Flash had+int6 storage ~{meta['est_flash_storage_gb']} GB (3328 B/tok)",
        flush=True,
    )
    if err_samples:
        print("[render-mp] WARNING gate/encode errors (first 10):\n  " + "\n  ".join(err_samples), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
