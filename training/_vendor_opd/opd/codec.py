# Copyright 2026 proof-pilot. Apache-2.0.
"""Codec for quantized teacher hidden — directly reuses distill/hidden_codec (had+int6_blk32).

OPD wire format: the teacher service compresses the post-norm last-hidden `[L, 4096]` with `encode()`
into `had+int6` (3328 B/tok); the trainer's `decode()` returns **rotated-space** hidden, paired with a
pre-folded teacher head `W_rot` (`build_w_rot()`), such that `decode(packed) @ W_rotᵀ == h @ Wᵀ` (up to
int6 noise). The rotation never runs an inverse on the hot path (see the note at the top of hidden_codec.py).

This file only does two things: (1) re-export distill's Rotator/encode/decode; (2) provide a helper to
load `head.weight` from the teacher safetensors and fold it into `W_rot` (the trainer keeps this 1.06 GB bf16 resident).
"""
from __future__ import annotations

import json
import os
import sys

import torch

# distill is a sibling under training/ with no __init__ packaging -> add its directory to the path directly.
_DISTILL = os.path.join(os.path.dirname(__file__), "..", "..", "..", "distill")
sys.path.insert(0, os.path.abspath(_DISTILL))
from hidden_codec import BLOCK, ROT_SEED, Rotator, decode, encode  # noqa: E402

__all__ = ["BLOCK", "ROT_SEED", "Rotator", "encode", "decode", "build_w_rot", "load_teacher_head"]


def load_teacher_head(model_path: str, hid: int, device: str = "cpu") -> torch.Tensor:
    """Load `head.weight` bf16 `[V, hid]` from the teacher HF shards (same method as _validate_hidden.py)."""
    from safetensors import safe_open

    idx = json.load(open(f"{model_path}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{model_path}/{idx['head.weight']}", framework="pt", device="cpu") as f:
        w = f.get_tensor("head.weight")
    assert w.dtype == torch.bfloat16 and w.shape[1] == hid, (w.dtype, tuple(w.shape))
    return w.to(device)


def build_w_rot(model_path: str, hid: int, device: str = "cpu",
                seed: int = ROT_SEED) -> tuple[torch.Tensor, Rotator]:
    """Return (W_rot [V, hid] bf16, rotator). Called once at trainer startup; W_rot stays resident.

    `W_rot = fold_head(head.weight)`, after which `decode(packed,scales) @ W_rotᵀ` is directly the
    teacher logits (rotated-space hidden × rotated-space head, the rotation cancels). rotator is used
    for encoding on the teacher service side.
    """
    rot = Rotator(hid, device=device, seed=seed)
    head = load_teacher_head(model_path, hid, device=device)
    w_rot = rot.fold_head(head)  # [V, hid], same dtype as head (bf16)
    return w_rot, rot
