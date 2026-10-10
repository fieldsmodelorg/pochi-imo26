# Copyright 2026 proof-pilot. Apache-2.0.
"""Extract DeepSeek-V4-Flash hidden states for an off-policy manifest.

Input is the parquet written by `render_manifest.py`: each row already contains exact
DeepSeek chat-template token ids for prompt + teacher assistant output. This client sends
those token ids to patched sglang `/score` servers and stores one resumable `.pt` file per
document:

    input_ids: int32 [L]
    packed: uint8 [target_len + 1, 3072]
    scales: fp16 [target_len + 1, 128]
    teacher_top1: optional int32 [target_len + 1]

The extra teacher row is intentional: row 0 corresponds to hidden position `prompt_len-1`,
and the final row catches off-by-one mistakes in downstream G4 diagnostics.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from offpolicy_data import ExtractedDocMeta, write_index  # noqa: E402

OPD_SRC = REPO / "training" / "_vendor_opd"
if str(OPD_SRC) not in sys.path:
    sys.path.insert(0, str(OPD_SRC))
from opd.batch import PACKED_ROW_BYTES, SCALE_ROW_BYTES, TOP1_ROW_BYTES  # noqa: E402
from opd.config import HID_DIM  # noqa: E402

DEFAULT_MANIFEST = REPO / "training" / "teacher_extract" / "dataprep" / "work" / "dsflash-test" / "manifest.parquet"
DEFAULT_OUT = REPO / "training" / "teacher_extract" / "dataprep" / "work" / "dsflash-test" / "hidden"


@dataclass(frozen=True)
class ManifestRow:
    row_idx: int
    input_ids: list[int]
    prompt_len: int
    seq_len: int
    target_len: int
    run_id: str
    problem_id: str
    template: str
    effort: str

    @property
    def expected_teacher_seq_len(self) -> int:
        return self.target_len + 1


def _as_str(v: Any) -> str:
    return "" if v is None else str(v)


def iter_manifest(
    path: Path,
    *,
    batch_size: int,
    limit: int,
    shard_index: int,
    num_shards: int,
) -> list[ManifestRow]:
    cols = [
        "row_idx", "input_ids", "prompt_len", "seq_len", "target_len",
        "run_id", "problem_id", "template", "effort",
    ]
    rows: list[ManifestRow] = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        for r in batch.to_pylist():
            row_idx = int(r["row_idx"])
            if row_idx % num_shards != shard_index:
                continue
            ids = [int(x) for x in r["input_ids"]]
            prompt_len = int(r["prompt_len"])
            seq_len = int(r["seq_len"])
            target_len = int(r["target_len"])
            if len(ids) != seq_len:
                raise ValueError(f"row {row_idx}: input_ids len={len(ids)} seq_len={seq_len}")
            if prompt_len <= 0 or target_len <= 0 or prompt_len + target_len != seq_len:
                raise ValueError(
                    f"row {row_idx}: bad lengths prompt={prompt_len} target={target_len} seq={seq_len}"
                )
            rows.append(ManifestRow(
                row_idx=row_idx,
                input_ids=ids,
                prompt_len=prompt_len,
                seq_len=seq_len,
                target_len=target_len,
                run_id=_as_str(r.get("run_id")),
                problem_id=_as_str(r.get("problem_id")),
                template=_as_str(r.get("template")),
                effort=_as_str(r.get("effort")),
            ))
            if limit and len(rows) >= limit:
                return rows
    return rows


def doc_path_for(out_dir: Path, row_idx: int) -> Path:
    return out_dir / "docs" / f"{row_idx:06d}.pt"


def meta_for(row: ManifestRow, path: Path) -> ExtractedDocMeta:
    return ExtractedDocMeta(
        row_idx=row.row_idx,
        path=str(path.resolve()),
        prompt_len=row.prompt_len,
        seq_len=row.seq_len,
        target_len=row.target_len,
        teacher_seq_len=row.expected_teacher_seq_len,
        run_id=row.run_id,
        problem_id=row.problem_id,
        template=row.template,
        effort=row.effort,
    )


def _tensor_from_bytes(blob: bytes, dtype: np.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    arr = np.frombuffer(blob, dtype=dtype).reshape(shape).copy()
    return torch.from_numpy(arr)


def _header(headers: Mapping[str, str], name: str, default: str | None = None) -> str:
    val = headers.get(name) or headers.get(name.lower())
    if val is None:
        if default is not None:
            return default
        raise KeyError(name)
    return val


def parse_score_response(content: bytes, headers: Mapping[str, str], row: ManifestRow) -> dict[str, torch.Tensor | int]:
    seq_len = int(_header(headers, "X-Seq-Len"))
    packed_bytes = int(_header(headers, "X-Packed-Bytes"))
    top1_bytes = int(_header(headers, "X-Top1-Bytes", "0"))
    expected_seq = row.expected_teacher_seq_len
    if seq_len != expected_seq:
        raise RuntimeError(
            f"row {row.row_idx}: teacher seq_len={seq_len}, expected={expected_seq} "
            f"(target_len + 1)"
        )
    expected_packed = seq_len * PACKED_ROW_BYTES
    expected_scales = seq_len * SCALE_ROW_BYTES
    if packed_bytes != expected_packed:
        raise RuntimeError(
            f"row {row.row_idx}: packed bytes={packed_bytes}, expected={expected_packed}"
        )
    if top1_bytes not in (0, seq_len * TOP1_ROW_BYTES):
        raise RuntimeError(
            f"row {row.row_idx}: top1 bytes={top1_bytes}, expected={seq_len * TOP1_ROW_BYTES}"
        )
    expected_total = expected_packed + expected_scales + top1_bytes
    if len(content) != expected_total:
        raise RuntimeError(
            f"row {row.row_idx}: response bytes={len(content)}, expected={expected_total}"
        )

    packed_blob = content[:expected_packed]
    scale_blob = content[expected_packed:expected_packed + expected_scales]
    out: dict[str, torch.Tensor | int] = {
        "teacher_seq_len": seq_len,
        "packed": _tensor_from_bytes(packed_blob, np.uint8, (seq_len, PACKED_ROW_BYTES)),
        "scales": _tensor_from_bytes(scale_blob, np.float16, (seq_len, HID_DIM // 32)),
    }
    if top1_bytes:
        top1_blob = content[-top1_bytes:]
        out["teacher_top1"] = _tensor_from_bytes(top1_blob, np.int32, (seq_len,))
    return out


def save_doc(path: Path, row: ManifestRow, score: dict[str, torch.Tensor | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os_pid()}")
    payload: dict[str, Any] = {
        "format": "had+int6_blk32",
        "hidden_dim": HID_DIM,
        "row_idx": row.row_idx,
        "input_ids": torch.tensor(row.input_ids, dtype=torch.int32),
        "prompt_len": row.prompt_len,
        "seq_len": row.seq_len,
        "target_len": row.target_len,
        "teacher_seq_len": int(score["teacher_seq_len"]),
        "packed": score["packed"],
        "scales": score["scales"],
    }
    if "teacher_top1" in score:
        payload["teacher_top1"] = score["teacher_top1"]
    torch.save(payload, tmp)
    tmp.replace(path)


def os_pid() -> int:
    import os
    return os.getpid()


def validate_existing(path: Path, row: ManifestRow) -> None:
    doc = torch.load(path, map_location="cpu", weights_only=True)
    if int(doc["seq_len"]) != row.seq_len:
        raise RuntimeError(f"{path}: seq_len mismatch")
    if int(doc["target_len"]) != row.target_len:
        raise RuntimeError(f"{path}: target_len mismatch")
    if int(doc["teacher_seq_len"]) != row.expected_teacher_seq_len:
        raise RuntimeError(f"{path}: teacher_seq_len mismatch")
    if tuple(doc["packed"].shape) != (row.expected_teacher_seq_len, PACKED_ROW_BYTES):
        raise RuntimeError(f"{path}: packed shape mismatch")
    if tuple(doc["scales"].shape) != (row.expected_teacher_seq_len, HID_DIM // 32):
        raise RuntimeError(f"{path}: scales shape mismatch")


async def score_one(
    row: ManifestRow,
    *,
    client: Any,
    url: str,
    out_dir: Path,
    return_top1: bool,
    resume: bool,
    validate_resume: bool,
    retries: int,
) -> tuple[str, ExtractedDocMeta]:
    path = doc_path_for(out_dir, row.row_idx)
    if resume and path.exists():
        if validate_resume:
            validate_existing(path, row)
        return "skipped", meta_for(row, path)

    payload: dict[str, Any] = {
        "input_ids": row.input_ids,
        "start": row.prompt_len - 1,
    }
    if return_top1:
        payload["return_top1"] = True

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = await client.post(f"{url}/score", json=payload)
            resp.raise_for_status()
            score = parse_score_response(resp.content, resp.headers, row)
            save_doc(path, row, score)
            return "written", meta_for(row, path)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            await asyncio.sleep(min(30.0, 2.0 ** attempt))
    raise RuntimeError(f"row {row.row_idx} via {url} failed after {retries + 1} attempts: {last_error}")


async def run_extract(args: argparse.Namespace, rows: list[ManifestRow]) -> list[ExtractedDocMeta]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("extract_hidden.py requires httpx in the active environment") from exc

    urls = [u.rstrip("/") for u in args.teacher_url.split(",") if u.strip()]
    if not urls:
        raise ValueError("--teacher-url must contain at least one URL")
    limits = httpx.Limits(max_connections=max(args.concurrency, len(urls)), max_keepalive_connections=len(urls))
    clients = {
        u: httpx.AsyncClient(timeout=args.timeout, limits=limits)
        for u in urls
    }
    sem = asyncio.Semaphore(args.concurrency)
    metas: list[ExtractedDocMeta] = []
    counts = {"written": 0, "skipped": 0}
    start_time = time.time()

    async def wrapped(i: int, row: ManifestRow) -> tuple[str, ExtractedDocMeta]:
        url = urls[i % len(urls)]
        async with sem:
            return await score_one(
                row,
                client=clients[url],
                url=url,
                out_dir=args.out_dir,
                return_top1=args.return_top1,
                resume=args.resume,
                validate_resume=args.validate_resume,
                retries=args.retries,
            )

    tasks = [asyncio.create_task(wrapped(i, row)) for i, row in enumerate(rows)]
    try:
        for n_done, fut in enumerate(asyncio.as_completed(tasks), start=1):
            status, meta = await fut
            counts[status] = counts.get(status, 0) + 1
            metas.append(meta)
            if n_done == 1 or n_done % args.progress_every == 0 or n_done == len(tasks):
                dt = max(1e-6, time.time() - start_time)
                toks = sum(m.target_len for m in metas)
                print(
                    f"[extract] {n_done}/{len(tasks)} docs "
                    f"written={counts.get('written', 0)} skipped={counts.get('skipped', 0)} "
                    f"target_tokens={toks:,} rate={toks / dt:,.0f} tok/s",
                    flush=True,
                )
    finally:
        for client in clients.values():
            await client.aclose()
    return sorted(metas, key=lambda m: m.row_idx)


def write_run_meta(out_dir: Path, args: argparse.Namespace, rows: list[ManifestRow], metas: list[ExtractedDocMeta]) -> None:
    out = {
        "manifest": str(args.manifest),
        "teacher_url": args.teacher_url,
        "return_top1": args.return_top1,
        "format": "had+int6_blk32",
        "hidden_dim": HID_DIM,
        "packed_row_bytes": PACKED_ROW_BYTES,
        "scale_row_bytes": SCALE_ROW_BYTES,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "n_manifest_rows": len(rows),
        "n_index_rows": len(metas),
        "target_tokens": sum(m.target_len for m in metas),
        "teacher_rows": sum(m.teacher_seq_len for m in metas),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "extract_meta.json.tmp"
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_dir / "extract_meta.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--teacher-url", default="http://127.0.0.1:8100",
                    help="Comma-separated patched sglang /score base URLs.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest-batch-size", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--validate-resume", action="store_true",
                    help="When --resume skips an existing doc, load it and validate tensor shapes.")
    ap.add_argument("--return-top1", action="store_true",
                    help="Request teacher argmax ids from the patched /score route.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= shard_index < num_shards")

    rows = iter_manifest(
        args.manifest,
        batch_size=args.manifest_batch_size,
        limit=args.limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if not rows:
        raise SystemExit("no manifest rows selected")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[extract] selected {len(rows)} rows from {args.manifest} "
        f"(shard {args.shard_index}/{args.num_shards}, concurrency={args.concurrency}, "
        f"top1={args.return_top1})",
        flush=True,
    )
    metas = asyncio.run(run_extract(args, rows))
    if len(metas) != len(rows):
        raise RuntimeError(f"index row count mismatch: {len(metas)} != {len(rows)}")
    write_index(args.out_dir / "index.jsonl", metas)
    write_run_meta(args.out_dir, args, rows, metas)
    print(
        f"[extract] wrote index {args.out_dir / 'index.jsonl'} "
        f"docs={len(metas)} target_tokens={sum(m.target_len for m in metas):,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
