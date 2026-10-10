# Copyright 2026 proof-pilot. Apache-2.0.
"""Assemble Trajectories into JSD-loss inputs — the position-alignment logic (the core of PLAN §6 G4).

The sharpest edge: teacher hidden and student hidden must sit on the **same s_t=(x,y_<t)** and
predict the **same next token y_t**. Alignment definition (matching
buffer.Trajectory.target_positions / the teacher_service start):

  trajectory token_ids[0..L-1], prompt_len = P.
  - What we learn is the generated span token_ids[P..L-1] (L-P tokens total).
  - student: hidden[t], through the head, predicts token_ids[t+1]. So the student positions that
    predict the generated span are t = P-1 .. L-2 (L-P of them), whose hidden predicts token_ids[P..L-1].
  - the teacher service is called with start=P-1 -> returns quant hidden for positions [P-1 .. L-1]
    (L-P+1 of them); take the first L-P ([P-1 .. L-2]) to align one-to-one with the student targets.
  - labels[i] = token_ids[P+i], i=0..L-P-1. student_pos[i]=P-1+i, teacher_pos[i]=P-1+i (same absolute
    position), and both hiddens predict labels[i].

So all three (student hidden slice, teacher hidden slice, labels) align per i. An off-by-one is exactly what G4 catches.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch

from opd.buffer import Trajectory
from opd.config import HID_DIM

# Reuse olmo3_sink's packing (Example/greedy_pack/pack_to_tensors) — importable from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from olmo3_sink.sft_data import Example, greedy_pack, pack_to_tensors  # noqa: E402

PACKED_ROW_BYTES = HID_DIM * 6 // 8
SCALE_ROW_BYTES = (HID_DIM // 32) * 2
TOP1_ROW_BYTES = 4


@dataclass
class AssembledTraj:
    input_ids: torch.Tensor       # [1, L]  the full sequence fed to the student model.model
    student_pos: torch.Tensor     # [n_t]   positions at which to take student hidden (= P-1 .. L-2)
    labels: torch.Tensor          # [n_t]   generated next-token (= token_ids[P..L-1])
    teacher_keep: int             # n_t: how many teacher hidden rows to keep (the service returned one extra at the end)

    @property
    def n_targets(self) -> int:
        return self.labels.numel()


def _slice_rows(blob: bytes | None, row_bytes: int, start: int, n_rows: int,
                name: str) -> bytes | None:
    if blob is None:
        return None
    a = start * row_bytes
    b = (start + n_rows) * row_bytes
    if b > len(blob):
        raise ValueError(f"{name} too short for row slice: need end={b} bytes, have={len(blob)}")
    return blob[a:b]


def window_trajectory(traj: Trajectory, micro_len: int, context_tokens: int = 4096,
                      target_tokens: int | None = None) -> list[Trajectory]:
    """Split an oversized scored trajectory into trainable target windows.

    The teacher service scored the full sequence from absolute position P-1, so row
    k in teacher bytes corresponds to original position P-1+k. Each returned window
    keeps enough left context for the student forward, slices teacher rows to the
    window target range, and preserves absolute RoPE positions via `position_offset`.
    """
    L, P = len(traj.token_ids), traj.prompt_len
    gen_len = L - P
    if gen_len <= 0:
        return []
    if not traj.scored():
        raise ValueError("window_trajectory() requires a scored trajectory")
    seq_len = traj.teacher_seq_len if traj.teacher_seq_len is not None else gen_len + 1
    if seq_len != gen_len + 1:
        raise ValueError(f"teacher seq_len mismatch: got={seq_len} expected={gen_len + 1}")
    if L <= micro_len:
        return [traj]

    context_tokens = max(1, min(context_tokens, micro_len - 1))
    if target_tokens is None:
        target_tokens = micro_len - context_tokens
    target_tokens = max(1, min(target_tokens, micro_len - 1))

    out: list[Trajectory] = []
    target_start = P
    while target_start < L:
        context_start = 0 if target_start == P else max(0, target_start - context_tokens)
        local_prompt = target_start - context_start
        if local_prompt >= micro_len:
            context_start = target_start - (micro_len - 1)
            local_prompt = micro_len - 1
        n_t = min(target_tokens, micro_len - local_prompt, L - target_start)
        if n_t <= 0:
            break
        target_end = target_start + n_t
        row_start = target_start - P
        rows = n_t + 1
        meta = dict(traj.meta)
        meta.update({
            "windowed_from_len": L,
            "window_context_start": context_start,
            "window_target_start": target_start,
            "window_target_end": target_end,
        })
        out.append(Trajectory(
            token_ids=traj.token_ids[context_start:target_end],
            prompt_len=local_prompt,
            weight_version=traj.weight_version,
            teacher_packed=_slice_rows(traj.teacher_packed, PACKED_ROW_BYTES, row_start, rows,
                                       "teacher_packed"),
            teacher_scales=_slice_rows(traj.teacher_scales, SCALE_ROW_BYTES, row_start, rows,
                                       "teacher_scales"),
            teacher_seq_len=rows,
            teacher_top1=_slice_rows(traj.teacher_top1, TOP1_ROW_BYTES, row_start, rows,
                                     "teacher_top1"),
            position_offset=traj.position_offset + context_start,
            meta=meta,
        ))
        target_start = target_end
    return out


def assemble(traj: Trajectory, device: str = "cpu") -> AssembledTraj:
    """Compute the alignment index/labels from a trajectory (does not touch teacher bytes; decode happens on the trainer side)."""
    L = len(traj.token_ids)
    P = traj.prompt_len
    n_t = L - P
    if n_t <= 0:
        raise ValueError(f"trajectory has no generated tokens: L={L} P={P}")
    ids = torch.tensor(traj.token_ids, device=device)
    student_pos = torch.arange(P - 1, L - 1, device=device)   # [n_t]
    labels = ids[P:L]                                          # [n_t]
    assert student_pos.numel() == n_t == labels.numel()
    return AssembledTraj(
        input_ids=ids.view(1, L),
        student_pos=student_pos,
        labels=labels,
        teacher_keep=n_t,           # the service returns L-P+1 rows from start=P-1; keep the first n_t
    )


def slice_teacher_hidden(decoded_hidden: torch.Tensor, a: AssembledTraj) -> torch.Tensor:
    """The hidden returned by the teacher service (positions [P-1 .. L-1], L-P+1 rows); keep the first n_t to align with the student."""
    if decoded_hidden.shape[0] < a.teacher_keep:
        raise ValueError(f"teacher hidden too short: {decoded_hidden.shape[0]} < {a.teacher_keep}")
    return decoded_hidden[: a.teacher_keep]


# ---- seq packing (reuses olmo3_sink; TRAINER_DESIGN §7c) ----
@dataclass
class Segment:
    """One trajectory inside a packed bin, with absolute-index alignment info (G4)."""
    off: int                       # the segment's absolute start index within the bin
    prompt_len: int
    n_t: int                       # = L - P (number of generated tokens)
    student_pos: torch.Tensor      # [n_t] absolute index = off + (P-1 .. L-2)
    labels: torch.Tensor           # [n_t] = token_ids[P:L] (generated next-token)
    teacher_packed: bytes
    teacher_scales: bytes
    teacher_seq_len: int | None = None
    teacher_top1: bytes | None = None
    position_offset: int = 0


@dataclass
class PackedBin:
    tensors: dict                  # pack_to_tensors output: input_ids/position_ids/cu_seq_lens_q/k/max_length_q/k [+labels/n_docs]
    segments: list[Segment] = field(default_factory=list)

    @property
    def total_targets(self) -> int:
        return sum(s.n_t for s in self.segments)


def pack_trajectories(trajs: list[Trajectory], micro_len: int, pad_id: int,
                      max_segs: int | None = None, device: str = "cpu") -> list[PackedBin]:
    """FFD-pack trajectories into micro_len bins (reusing olmo3_sink greedy_pack/pack_to_tensors),
    attaching (off, student_pos absolute index, labels, teacher bytes) to each segment.

    Contract: pack_to_tensors concatenates in the Example order of bin_, offset = cumsum of preceding
    segment lengths — consistent with the Example order within each bin returned by greedy_pack, so
    segment offsets can be reconstructed (G4 alignment). Trajectories longer than micro_len or with no
    generated tokens (L<=P) are dropped (greedy_pack drops over-length ones; here we pre-filter L<=P).
    """
    usable = [t for t in trajs if len(t.token_ids) > t.prompt_len and t.scored()]
    examples = [Example(input_ids=list(t.token_ids), prompt_len=t.prompt_len) for t in usable]
    ex2traj = {id(e): t for e, t in zip(examples, usable)}
    out: list[PackedBin] = []
    for bin_ in greedy_pack(examples, micro_len):
        tens = pack_to_tensors(bin_, micro_len, pad_id, device, max_segs)
        segs: list[Segment] = []
        off = 0
        for e in bin_:
            t = ex2traj[id(e)]
            P, L = t.prompt_len, len(t.token_ids)
            n_t = L - P
            tens["position_ids"][0, off:off + L] = torch.arange(
                t.position_offset, t.position_offset + L, device=device)
            segs.append(Segment(
                off=off, prompt_len=P, n_t=n_t,
                student_pos=torch.arange(off + P - 1, off + L - 1, device=device),
                labels=torch.tensor(t.token_ids[P:L], device=device),
                teacher_packed=t.teacher_packed, teacher_scales=t.teacher_scales,
                teacher_seq_len=t.teacher_seq_len,
                teacher_top1=t.teacher_top1,
                position_offset=t.position_offset,
            ))
            off += L
        out.append(PackedBin(tensors=tens, segments=segs))
    return out
