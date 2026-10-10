# Copyright 2026 proof-pilot. Apache-2.0.
"""Data helpers for off-policy distillation shards.

The extraction stage writes one `.pt` file per teacher-generated proof:

    input_ids: int32 [L]
    prompt_len: int
    target_len: int              # L - prompt_len
    teacher_seq_len: int         # target_len + 1
    packed: uint8 [teacher_seq_len, 3072]     # had+int6 rows
    scales: fp16 [teacher_seq_len, 128]

Trainer windows preserve absolute RoPE positions via `Trajectory.position_offset`.
For a target window `[target_start, target_end)` in original coordinates, teacher row
`target_start - prompt_len` corresponds to hidden position `target_start - 1`, so the
window needs `n_t + 1` rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import torch

import sys

REPO = Path(__file__).resolve().parents[3]
OPD_SRC = REPO / "training" / "_vendor_opd"
if str(OPD_SRC) not in sys.path:
    sys.path.insert(0, str(OPD_SRC))

from opd.buffer import Trajectory  # noqa: E402
from opd.batch import PACKED_ROW_BYTES, SCALE_ROW_BYTES  # noqa: E402
from opd.config import HID_DIM  # noqa: E402


@dataclass(frozen=True)
class ExtractedDocMeta:
    row_idx: int
    path: str
    prompt_len: int
    seq_len: int
    target_len: int
    teacher_seq_len: int
    run_id: str = ""
    problem_id: str = ""
    template: str = ""
    effort: str = ""


@dataclass(frozen=True)
class WindowSpec:
    meta: ExtractedDocMeta
    context_start: int
    target_start: int
    target_end: int

    @property
    def seq_len(self) -> int:
        return self.target_end - self.context_start

    @property
    def prompt_len(self) -> int:
        return self.target_start - self.context_start

    @property
    def n_targets(self) -> int:
        return self.target_end - self.target_start

    @property
    def row_start(self) -> int:
        return self.target_start - self.meta.prompt_len


def read_index(path: str | Path) -> list[ExtractedDocMeta]:
    rows: list[ExtractedDocMeta] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(ExtractedDocMeta(**r))
    return rows


def write_index(path: str | Path, rows: Iterable[ExtractedDocMeta]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    tmp.replace(p)


def build_window_specs(
    docs: list[ExtractedDocMeta],
    *,
    micro_len: int,
    max_seq_len: int = 262144,
    context_tokens: int = 4096,
    target_tokens: int | None = None,
) -> list[WindowSpec]:
    """Build deterministic train windows from extracted doc metadata."""
    if target_tokens is None:
        target_tokens = micro_len - min(context_tokens, micro_len - 1)
    context_tokens = max(1, min(context_tokens, micro_len - 1))
    target_tokens = max(1, min(target_tokens, micro_len - 1))

    out: list[WindowSpec] = []
    for d in docs:
        if d.seq_len > max_seq_len:
            continue
        if d.target_len <= 0:
            continue
        if d.teacher_seq_len != d.target_len + 1:
            raise ValueError(
                f"{d.path}: teacher_seq_len={d.teacher_seq_len} expected={d.target_len + 1}"
            )
        if d.seq_len <= micro_len:
            out.append(WindowSpec(d, context_start=0, target_start=d.prompt_len, target_end=d.seq_len))
            continue

        target_start = d.prompt_len
        while target_start < d.seq_len:
            context_start = max(0, target_start - context_tokens)
            local_prompt = target_start - context_start
            if local_prompt >= micro_len:
                context_start = target_start - (micro_len - 1)
                local_prompt = micro_len - 1
            n_t = min(target_tokens, micro_len - local_prompt, d.seq_len - target_start)
            if n_t <= 0:
                break
            target_end = target_start + n_t
            out.append(WindowSpec(d, context_start=context_start,
                                  target_start=target_start, target_end=target_end))
            target_start = target_end
    return out


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()


def _load_hidden_doc(path: str | Path) -> dict:
    """Load an extracted hidden file, using mmap when the local torch supports it."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def load_window(spec: WindowSpec) -> Trajectory:
    """Load one WindowSpec as an OPD Trajectory with int6 teacher bytes attached."""
    doc = _load_hidden_doc(spec.meta.path)
    if doc.get("format") != "had+int6_blk32":
        raise ValueError(f"{spec.meta.path}: unsupported hidden format {doc.get('format')!r}")
    if int(doc.get("hidden_dim", 0)) != HID_DIM:
        raise ValueError(
            f"{spec.meta.path}: hidden_dim={doc.get('hidden_dim')} expected={HID_DIM}"
        )
    for key in ("prompt_len", "seq_len", "target_len", "teacher_seq_len"):
        if int(doc.get(key, -1)) != int(getattr(spec.meta, key)):
            raise ValueError(
                f"{spec.meta.path}: {key}={doc.get(key)} does not match index "
                f"{getattr(spec.meta, key)}"
            )
    ids = doc["input_ids"]
    packed = doc["packed"]
    scales = doc["scales"]
    if int(ids.numel()) != spec.meta.seq_len:
        raise ValueError(
            f"{spec.meta.path}: input_ids length={int(ids.numel())} expected={spec.meta.seq_len}"
        )
    row_start = spec.row_start
    rows = spec.n_targets + 1
    if row_start < 0:
        raise ValueError(f"{spec.meta.path}: negative row_start={row_start}")
    if row_start + rows > packed.shape[0]:
        raise ValueError(
            f"{spec.meta.path}: teacher rows short: need {row_start + rows}, have {packed.shape[0]}"
        )
    local_ids = ids[spec.context_start:spec.target_end].to(torch.int32).tolist()
    traj = Trajectory(
        token_ids=local_ids,
        prompt_len=spec.prompt_len,
        weight_version=0,
        teacher_packed=_tensor_to_bytes(packed[row_start:row_start + rows]),
        teacher_scales=_tensor_to_bytes(scales[row_start:row_start + rows]),
        teacher_seq_len=rows,
        position_offset=spec.context_start,
        meta={
            "row_idx": spec.meta.row_idx,
            "problem_id": spec.meta.problem_id,
            "template": spec.meta.template,
            "effort": spec.meta.effort,
            "context_start": spec.context_start,
            "target_start": spec.target_start,
            "target_end": spec.target_end,
        },
    )
    top1 = doc.get("teacher_top1")
    if top1 is not None:
        traj.teacher_top1 = _tensor_to_bytes(top1[row_start:row_start + rows].to(torch.int32))

    # Cheap byte-shape guard before the trainer decodes on GPU.
    if len(traj.teacher_packed) != rows * PACKED_ROW_BYTES:
        raise ValueError(f"{spec.meta.path}: packed byte count mismatch")
    if len(traj.teacher_scales) != rows * SCALE_ROW_BYTES:
        raise ValueError(f"{spec.meta.path}: scales byte count mismatch")
    return traj
