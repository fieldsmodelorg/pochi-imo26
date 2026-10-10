# Copyright 2026 proof-pilot. Apache-2.0.
"""Render L2 single-turn docs into an extract_hidden manifest (soft-distill data scale-up).

General L2 -> manifest renderer (defaults to nemotron-math-proofs-v2 for back-compat, but
--root/--domains point it at any single-turn L2 dataset, e.g. nemotron-sft-math-v4/aops_cot).
Produces the SAME manifest schema as `render_manifest.py`, so `extract_hidden.py` and
`soft_distill_v2/data.py` consume it unchanged. Only works for SINGLE-TURN docs (last message
= assistant, prompt = everything before it); multi-turn / tool (TIR) data needs a different
renderer.

  # proofs-v2 (default): 4:2:1 sample to 30k
  ... --out work/proofs-v2-30k/manifest.parquet
  # aops_cot math-solving: take ALL (--total 0), single domain
  ... --root data/nemotron-deepseek-sft-mix-v2/dataset=nemotron-sft-math-v4 \
      --domains aops_cot --total 0 --ratio 1 --out work/math-v4-aops-cot/manifest.parquet

Semantics (must match build_window_specs, which treats ALL of [prompt_len:seq_len] as
target and ignores any loss mask):
  - prompt_len = tokens of the conversation rendered up to the assistant generation prompt
  - target     = the final assistant turn (reasoning_content + content + EOS), i.e. the tail
  - seq_len    = prompt_len + target_len
This is exactly render_manifest.py:encode_row, except the assistant turn is sourced from the
L2 `messages` list (reasoning_content already separated) rather than separate API columns.

Sampling is deterministic (blake2b of seed|id): per-domain quota by ratio (default 4:2:1,
30k total), smallest-key ids kept. Rows whose rendered seq_len exceeds --max-seq-len
(default 200000, the soft_distill_v2 whole-proof row cap) are dropped and counted, so the
final count can sit a few % under the target on the long proof tail (logged loudly).

CPU/tokenizer heavy (~2.3B tokens for 30k docs). Multiprocess: main does parquet I/O and
feeds rows to a worker pool (ordered imap keeps row_idx deterministic), each worker holds
its own tokenizer. Run under Slurm on a dev node (grab gpu:4 -> 48 CPU quota), not login.

  .venv/bin/python training/teacher_extract/dataprep/render_manifest_proofs_v2.py \
      --out training/teacher_extract/dataprep/work/proofs-v2-30k/manifest.parquet --workers 47
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from multiprocessing import Pool
import os
from pathlib import Path
import statistics
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "training") not in sys.path:
    sys.path.insert(0, str(REPO / "training"))

from train_core.encoding_dsv4 import encode_messages  # noqa: E402

DEFAULT_ROOT = REPO / "data" / "nemotron-deepseek-sft-mix-v2" / "dataset=nemotron-math-proofs-v2"
DEFAULT_TOKENIZER = os.environ.get("DEEPSEEK_V4_FLASH", "/models/DeepSeek-V4-Flash")
DEFAULT_OUT = REPO / "training" / "teacher_extract" / "dataprep" / "work" / "proofs-v2-30k" / "manifest.parquet"
DOMAINS = ("proof", "verification", "meta_verification")
DEFAULT_MAX_SEQ_LEN = 200_000
READ_COLS = ["id", "messages", "generator", "dataset", "upstream_source", "thinking_mode"]

# Identical schema/columns to render_manifest.write_manifest so the manifest is a drop-in.
SCHEMA = pa.schema([
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


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _key(seed: int, doc_id: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{seed}|{doc_id}".encode(), digest_size=8).digest(), "big")


def _shards(root: Path, domain: str, shards_limit: int) -> list[str]:
    s = sorted(glob.glob(str(root / f"domain={domain}" / "part-*.parquet")))
    return s[:shards_limit] if shards_limit else s


def _parse_ratio(ratio: str, n: int) -> list[int]:
    parts = [int(x) for x in ratio.split(":")]
    if len(parts) != n or any(p < 0 for p in parts) or sum(parts) == 0:
        raise SystemExit(f"--ratio must be {n} non-negative ints matching --domains")
    return parts


def select_ids(root: Path, total: int, ratio: list[int], seed: int,
               shards_limit: int, domains: list[str]) -> dict[str, set[str] | None]:
    """Pass A (id-only, cheap): pick the smallest-key quota of ids per domain.

    total <= 0 means TAKE ALL rows of every domain (no sampling) -> kept[dom]=None sentinel.
    """
    if total <= 0:
        kept: dict[str, set[str] | None] = {}
        for dom in domains:
            n = sum(pq.ParquetFile(s).metadata.num_rows for s in _shards(root, dom, shards_limit))
            kept[dom] = None
            print(f"[select] domain={dom:24s} pool={n:>8d} take=ALL", flush=True)
        return kept
    w = sum(ratio)
    quota, assigned = {}, 0
    for i, dom in enumerate(domains):
        q = total - assigned if i == len(domains) - 1 else round(total * ratio[i] / w)
        quota[dom] = q
        assigned += q
    kept = {}
    for dom in domains:
        ids: list[str] = []
        for sh in _shards(root, dom, shards_limit):
            for batch in pq.ParquetFile(sh).iter_batches(columns=["id"]):
                ids.extend(str(x) for x in batch.column("id").to_pylist())
        ids.sort(key=lambda d: _key(seed, d))
        q = min(quota[dom], len(ids))
        kept[dom] = set(ids[:q])
        print(f"[select] domain={dom:24s} pool={len(ids):>8d} quota={quota[dom]:>8d} kept={q:>8d}",
              flush=True)
    return kept


def _messages(row) -> list[dict]:
    m = row["messages"]
    return json.loads(m) if isinstance(m, str) else m


def encode_row(row, domain: str, tokenizer, seed: int) -> dict:
    msgs = _messages(row)
    if not msgs or msgs[-1].get("role") != "assistant":
        raise ValueError("last message is not assistant")
    assistant = msgs[-1]
    prompt_messages = msgs[:-1]
    tmode = row.get("thinking_mode") or "thinking"

    prompt_text = encode_messages(prompt_messages, thinking_mode=tmode, drop_thinking=False)
    full_text = encode_messages(prompt_messages + [assistant], thinking_mode=tmode,
                                drop_thinking=False)
    if not full_text.startswith(prompt_text):
        raise RuntimeError(f"full text does not start with prompt for {row.get('id')}")
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)
    if input_ids[:prompt_len] != prompt_ids:
        raise RuntimeError(f"tokenized prompt is not a prefix for {row.get('id')}")
    target_len = len(input_ids) - prompt_len
    if prompt_len <= 0 or target_len <= 0:
        raise ValueError(f"bad lengths prompt={prompt_len} target={target_len}")
    return {
        "run_id": None, "problem_id": str(row.get("id") or ""), "sample": None,
        "template": None, "effort": None, "seed": int(seed),
        "origin": str(row.get("upstream_source") or ""), "category": domain,
        "competition": None, "source": str(row.get("dataset") or ""),
        "nm_uuid": str(row.get("id") or ""), "model": str(row.get("generator") or ""),
        "max_tokens": None, "prompt_tokens_api": None, "completion_tokens_api": None,
        "reasoning_tokens_api": None, "prompt_len": prompt_len, "seq_len": len(input_ids),
        "target_len": target_len, "completion_delta": None, "self_score": None,
        "input_ids": input_ids,
    }


def clean(row) -> bool:
    try:
        msgs = _messages(row)
    except Exception:
        return False
    if not msgs or msgs[-1].get("role") != "assistant":
        return False
    a = msgs[-1]
    return bool(((a.get("reasoning_content") or "") + (a.get("content") or "")).strip())


# ---- worker pool ------------------------------------------------------------
_TOK = None


def _init_worker(tokenizer_path: str) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _TOK
    from transformers import PreTrainedTokenizerFast
    _TOK = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def _render_task(task):
    domain, row, seed = task
    if not clean(row):
        return (domain, None, "err")
    try:
        return (domain, encode_row(row, domain, _TOK, seed), "ok")
    except Exception as exc:
        return (domain, None, f"err:{exc}")


def _iter_candidates(root: Path, kept: dict[str, set[str] | None], shards_limit: int,
                     seed: int, domains: list[str]):
    """Yield (domain, row, seed) for kept ids (or ALL rows if kept[dom] is None),
    in deterministic domain/shard order."""
    for dom in domains:
        want = kept[dom]
        if want is None:  # take-all
            for sh in _shards(root, dom, shards_limit):
                for batch in pq.ParquetFile(sh).iter_batches(columns=READ_COLS):
                    for row in batch.to_pylist():
                        yield (dom, row, seed)
            continue
        seen: set[str] = set()
        for sh in _shards(root, dom, shards_limit):
            if len(seen) >= len(want):
                break
            for batch in pq.ParquetFile(sh).iter_batches(columns=READ_COLS):
                for row in batch.to_pylist():
                    rid = str(row.get("id") or "")
                    if rid not in want or rid in seen:
                        continue
                    seen.add(rid)
                    yield (dom, row, seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--domains", default=",".join(DOMAINS), help="comma-separated L2 domains under --root")
    ap.add_argument("--total", type=int, default=30_000, help="target doc count; <=0 = take ALL rows of every domain")
    ap.add_argument("--ratio", default="4:2:1", help="per-domain sampling weights (ignored when --total<=0)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--shards-limit", type=int, default=0, help="Debug: scan only first N shards/domain.")
    ap.add_argument("--flush-every", type=int, default=256)
    args = ap.parse_args()

    domains = [d for d in args.domains.split(",") if d]
    ratio = _parse_ratio(args.ratio, len(domains)) if args.total > 0 else [1] * len(domains)
    kept = select_ids(args.root, args.total, ratio, args.seed, args.shards_limit, domains)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    meta_seed = {
        "renderer": "render_manifest_proofs_v2.py", "root": str(args.root),
        "domains": args.domains, "tokenizer": args.tokenizer, "total_target": args.total,
        "ratio": args.ratio, "seed": args.seed, "max_seq_len": args.max_seq_len,
        "git_commit": _git_commit(),
    }
    writer = pq.ParquetWriter(
        tmp, SCHEMA.with_metadata({b"off_policy_distill": json.dumps(meta_seed).encode()}),
        compression="zstd")

    buf: list[dict] = []
    row_idx = 0
    counts = {d: {"kept": 0, "too_long": 0, "err": 0} for d in domains}
    seq_lens: list[int] = []
    target_lens: list[int] = []

    def flush():
        nonlocal buf
        if buf:
            writer.write_table(pa.Table.from_pylist(buf, SCHEMA))
            buf = []

    gen = _iter_candidates(args.root, kept, args.shards_limit, args.seed, domains)
    print(f"[render] domains={domains} workers={args.workers} max_seq_len={args.max_seq_len}", flush=True)
    try:
        with Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
            for domain, enc, status in pool.imap(_render_task, gen, chunksize=4):
                if enc is None:
                    counts[domain]["err"] += 1
                    continue
                if enc["seq_len"] > args.max_seq_len:
                    counts[domain]["too_long"] += 1
                    continue
                enc["row_idx"] = row_idx
                row_idx += 1
                counts[domain]["kept"] += 1
                seq_lens.append(enc["seq_len"])
                target_lens.append(enc["target_len"])
                buf.append(enc)
                if len(buf) >= args.flush_every:
                    flush()
                if row_idx % 2000 == 0:
                    print(f"[render] {row_idx} rows kept (tokens so far {sum(target_lens):,})",
                          flush=True)
        flush()
    finally:
        writer.close()

    for dom in domains:
        print(f"[render] domain={dom:24s} kept={counts[dom]['kept']:>8d} "
              f"too_long={counts[dom]['too_long']:>6d} err={counts[dom]['err']:>5d}", flush=True)
    if row_idx == 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit("no rows rendered")
    tmp.replace(args.out)

    meta = {
        **meta_seed, "n_rows": row_idx, "counts": counts,
        "target_tokens": int(sum(target_lens)), "seq_tokens": int(sum(seq_lens)),
        "est_flash_storage_gb": round(sum(target_lens) * 3328 / 1e9, 1),
    }
    (args.out.parent / "manifest_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[manifest] wrote {row_idx} rows -> {args.out}\n"
        f"  target_tokens={sum(target_lens):,} seq_tokens={sum(seq_lens):,}\n"
        f"  seq_len mean={statistics.mean(seq_lens):.0f} p50={statistics.median(seq_lens):.0f} "
        f"max={max(seq_lens)}\n"
        f"  est Flash had+int6 storage ~{meta['est_flash_storage_gb']} GB (3328 B/tok)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
