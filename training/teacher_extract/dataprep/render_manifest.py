# Copyright 2026 proof-pilot. Apache-2.0.
"""Render dsflash-proof-distill rows into exact DeepSeek-V4 token manifests.

The HF dataset stores the API prompt (`messages_json`) and the teacher output split into
`reasoning_content` + final `content`. For off-policy distillation we train on the full
teacher assistant turn, including the thinking text and EOS.

Critical fidelity rule: rows with `effort == "max"` must be encoded with
`reasoning_effort="max"`, which prepends DeepSeek's max-effort instruction. High effort
adds no local prefix, so `None` and `"high"` tokenize identically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import os

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "training") not in sys.path:
    sys.path.insert(0, str(REPO / "training"))

from train_core.encoding_dsv4 import encode_messages  # noqa: E402

DEFAULT_DATASET = "ycchen/dsflash-proof-distill-test"
DEFAULT_REVISION = "734c59b0126d9ed47ebc0a8406769a6dade96bc0"
DEFAULT_TOKENIZER = os.environ.get("DEEPSEEK_V4_FLASH", "/models/DeepSeek-V4-Flash")
DEFAULT_MAX_SEQ_LEN = 262144


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def read_rows(args) -> list[dict]:
    if args.input:
        p = Path(args.input)
        if p.suffix == ".parquet":
            return pq.read_table(p).to_pylist()
        if p.suffix == ".arrow":
            with pa.memory_map(str(p), "r") as source:
                return ipc.open_stream(source).read_all().to_pylist()
        raise ValueError(f"unsupported --input suffix: {p}")

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split, revision=args.revision)
    return list(ds)


def clean_row(r: dict) -> bool:
    return r.get("error") is None and not r.get("truncated") and bool((r.get("content") or "").strip())


def encode_row(r: dict, tokenizer) -> dict:
    messages = json.loads(r["messages_json"])
    effort = "max" if r.get("effort") == "max" else None
    prompt_text = encode_messages(
        messages, thinking_mode="thinking", drop_thinking=False, reasoning_effort=effort)
    assistant = {
        "role": "assistant",
        "reasoning_content": r.get("reasoning_content") or "",
        "content": r.get("content") or "",
    }
    full_text = encode_messages(
        messages + [assistant],
        thinking_mode="thinking",
        drop_thinking=False,
        reasoning_effort=effort,
    )
    if not full_text.startswith(prompt_text):
        raise RuntimeError(f"full text does not start with prompt for {r.get('problem_id')}")
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    if prompt_len != r.get("prompt_tokens"):
        raise ValueError(
            f"prompt token mismatch row={r.get('problem_id')}: "
            f"api={r.get('prompt_tokens')} local={prompt_len} effort={r.get('effort')}"
        )
    if input_ids[:prompt_len] != prompt_ids:
        raise RuntimeError(f"tokenized prompt is not a prefix for {r.get('problem_id')}")
    target_len = len(input_ids) - prompt_len
    api_completion = r.get("completion_tokens")
    completion_delta = None if api_completion is None else target_len - api_completion
    return {
        "run_id": r.get("run_id"),
        "problem_id": r.get("problem_id"),
        "sample": r.get("sample"),
        "template": r.get("template"),
        "effort": r.get("effort"),
        "seed": r.get("seed"),
        "origin": r.get("origin"),
        "category": r.get("category"),
        "competition": r.get("competition"),
        "source": r.get("source"),
        "nm_uuid": r.get("nm_uuid"),
        "model": r.get("model"),
        "max_tokens": r.get("max_tokens"),
        "prompt_tokens_api": r.get("prompt_tokens"),
        "completion_tokens_api": api_completion,
        "reasoning_tokens_api": r.get("reasoning_tokens"),
        "prompt_len": prompt_len,
        "seq_len": len(input_ids),
        "target_len": target_len,
        "completion_delta": completion_delta,
        "self_score": r.get("self_score"),
        "input_ids": input_ids,
    }


def write_manifest(rows: list[dict], out: Path, meta: dict) -> None:
    schema = pa.schema([
        ("row_idx", pa.int64()),
        ("run_id", pa.string()),
        ("problem_id", pa.string()),
        ("sample", pa.int64()),
        ("template", pa.string()),
        ("effort", pa.string()),
        ("seed", pa.int64()),
        ("origin", pa.string()),
        ("category", pa.string()),
        ("competition", pa.string()),
        ("source", pa.string()),
        ("nm_uuid", pa.string()),
        ("model", pa.string()),
        ("max_tokens", pa.int64()),
        ("prompt_tokens_api", pa.int64()),
        ("completion_tokens_api", pa.int64()),
        ("reasoning_tokens_api", pa.int64()),
        ("prompt_len", pa.int32()),
        ("seq_len", pa.int32()),
        ("target_len", pa.int32()),
        ("completion_delta", pa.int64()),
        ("self_score", pa.string()),
        ("input_ids", pa.list_(pa.int32())),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    table = table.replace_schema_metadata({"off_policy_distill": json.dumps(meta, ensure_ascii=False)})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=16)
    back = pq.read_table(tmp, columns=["row_idx", "seq_len", "target_len"])
    if back.num_rows != len(rows):
        raise RuntimeError(f"readback row mismatch: {back.num_rows} != {len(rows)}")
    tmp.replace(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--split", default="train")
    ap.add_argument("--input", default="", help="Optional local .parquet or HF cache .arrow file")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-truncated", action="store_true",
                    help="Debug only: include truncated/error rows. Production keeps clean rows only.")
    ap.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    args = ap.parse_args()

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    raw = read_rows(args)
    if args.limit:
        raw = raw[:args.limit]
    raw_pairs = list(enumerate(raw))
    selected = raw_pairs if args.include_truncated else [(i, r) for i, r in raw_pairs if clean_row(r)]
    rows: list[dict] = []
    too_long = 0
    for src_idx, r in selected:
        enc = encode_row(r, tokenizer)
        enc["row_idx"] = src_idx
        if enc["seq_len"] > args.max_seq_len:
            too_long += 1
            continue
        rows.append(enc)

    if not rows:
        raise SystemExit("no rows rendered")

    target_lens = [r["target_len"] for r in rows]
    seq_lens = [r["seq_len"] for r in rows]
    deltas = {}
    for r in rows:
        d = r["completion_delta"]
        deltas[d] = deltas.get(d, 0) + 1
    meta = {
        "dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "input": args.input or None,
        "tokenizer": args.tokenizer,
        "max_seq_len": args.max_seq_len,
        "n_raw": len(raw),
        "n_selected": len(selected),
        "n_rows": len(rows),
        "n_too_long": too_long,
        "target_tokens": int(sum(target_lens)),
        "seq_tokens": int(sum(seq_lens)),
        "completion_delta_counts": deltas,
        "git_commit": _git_commit(),
    }
    write_manifest(rows, args.out, meta)
    print(
        f"[manifest] wrote {len(rows)} rows -> {args.out}\n"
        f"  target_tokens={sum(target_lens):,} seq_tokens={sum(seq_lens):,} too_long={too_long}\n"
        f"  seq_len mean={statistics.mean(seq_lens):.1f} p50={statistics.median(seq_lens):.0f} max={max(seq_lens)}\n"
        f"  completion_delta_counts={deltas}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
