# Copyright 2026 proof-pilot. Apache-2.0.
"""soft_distill_v2 data layer: whole-proof packing into fixed micro_len (200k) rows.

No windowing. Each proof (seq_len <= truncate_len) is kept INTACT and packed into a
micro_len row, position-reset to 0 per proof so RoPE matches how the teacher was
extracted (teacher scored each proof from absolute position 0). Multiple short proofs
share a row; one long proof fills its own. Row assignment is deterministic (global
shuffle by seed) so every rank computes the same row list and takes a fixed slice =>
synchronized FSDP collectives, exactly like stage1's offline-packed L4.

Reuses: offpolicy_data.{read_index,build_window_specs,load_window},
opd.batch.pack_trajectories, opd.clients.teacher_client.decode_teacher_hidden.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "training", REPO / "training" / "_vendor_opd", REPO / "training" / "teacher_extract" / "dataprep", REPO / "training" / "_common"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from offpolicy_data import build_window_specs, read_index  # noqa: E402
from offpolicy_data import load_window  # noqa: E402
from opd.batch import PackedBin, pack_trajectories  # noqa: E402
from opd.clients.teacher_client import decode_teacher_hidden  # noqa: E402
from opd.config import HID_DIM  # noqa: E402

PAD_ID = 2


def build_rows(index_path, *, micro_len: int, truncate_len: int, seed: int):
    """Pack whole proofs (<= truncate_len) into fixed micro_len rows. Returns list[list[WindowSpec]]."""
    docs = read_index(index_path)
    # micro_len >= truncate_len so every kept proof falls into build_window_specs'
    # `seq_len <= micro_len` branch => one whole-proof spec (context_start=0, full proof).
    specs = build_window_specs(
        docs, micro_len=micro_len, max_seq_len=truncate_len, context_tokens=4096
    )
    rng = random.Random(seed)
    # FFD by full length into micro_len bins (rows).
    rows: list[list] = []
    fills: list[int] = []
    for s in sorted(specs, key=lambda s: s.seq_len, reverse=True):
        placed = False
        for i, f in enumerate(fills):
            if f + s.seq_len <= micro_len:
                rows[i].append(s)
                fills[i] += s.seq_len
                placed = True
                break
        if not placed:
            rows.append([s])
            fills.append(s.seq_len)
    rng.shuffle(rows)
    return rows


def _trim_bin(b: PackedBin) -> PackedBin:
    """Exact-trim isolated pad tail and rebuild dynamic varlen metadata."""
    if not b.segments:
        return b
    t = dict(b.tensors)
    seg_lens = [seg.prompt_len + seg.n_t for seg in b.segments]
    real_len = sum(seg_lens)
    if real_len <= 0:
        return b
    for key in ("input_ids", "position_ids", "labels"):
        val = t.get(key)
        if torch.is_tensor(val) and val.ndim == 2:
            t[key] = val[:, :real_len].contiguous()
    cu = torch.zeros(len(seg_lens) + 1, dtype=torch.int32, device=t["input_ids"].device)
    cu[1:] = torch.tensor(seg_lens, dtype=torch.int32, device=t["input_ids"].device).cumsum(0)
    t["cu_seq_lens_q"] = cu
    t["cu_seq_lens_k"] = cu
    t["max_length_q"] = max(seg_lens)
    t["max_length_k"] = max(seg_lens)
    return PackedBin(tensors=t, segments=b.segments)


def assemble_row_bin(row_specs, *, micro_len: int) -> PackedBin:
    """Load a row's proofs, CPU-pack into one bin, exact-trim. Asserts exactly one bin."""
    trajs = [load_window(s) for s in row_specs]
    max_segs = len(trajs) + 2
    bins = pack_trajectories(trajs, micro_len, pad_id=PAD_ID, max_segs=max_segs, device="cpu")
    if len(bins) != 1:
        raise RuntimeError(
            f"row packed into {len(bins)} bins (expected 1); FFD invariant broken: "
            f"sum_len={sum(s.seq_len for s in row_specs)} micro_len={micro_len}"
        )
    return _trim_bin(bins[0])


def assemble_kwargs(bin: PackedBin, device, w_rot, chunk_size):
    """Move tensors to GPU and decode per-segment teacher hidden into the JSD forward kwargs."""
    spos, th, lab = [], [], []
    for seg in bin.segments:
        spos.append(seg.student_pos.to(device))
        seq_len = seg.teacher_seq_len if seg.teacher_seq_len is not None else seg.n_t + 1
        dec = decode_teacher_hidden(seg.teacher_packed, seg.teacher_scales, seq_len,
                                    device=device, hid=HID_DIM)  # [n_t+1, H], rotated space
        th.append(dec[: seg.n_t])
        lab.append(seg.labels.to(device))
    t = bin.tensors
    return dict(
        input_ids=t["input_ids"].to(device),
        position_ids=t["position_ids"].to(device),
        cu_seq_lens_q=t["cu_seq_lens_q"].to(device),
        cu_seq_lens_k=t["cu_seq_lens_k"].to(device),
        max_length_q=t["max_length_q"],
        max_length_k=t["max_length_k"],
        opd_student_pos=torch.cat(spos),
        opd_teacher_hidden=torch.cat(th),
        opd_labels=torch.cat(lab),
        opd_w_rot=w_rot,
        opd_chunk_size=int(chunk_size),
    )
