# Copyright 2026 proof-pilot. Apache-2.0.
"""codec for quantized teacher hidden — forwards to `_common/hidden_codec` (had+int6_blk32, 3328 B/tok).

Follows v1 `opd/src/opd/codec.py` (validated in G2/G3): on the teacher side `encode()` compresses the
post-norm last-hidden `[L,4096]` into had+int6; on the trainer side `decode()` returns hidden in **rotated
space**, paired with the pre-folded teacher head `W_rot` (`build_w_rot()`), such that
`decode(packed) @ W_rotᵀ == h @ Wᵀ` (up to int6 noise). The rotation never runs an inverse on the hot path.
"""
from __future__ import annotations

import json
import os
import sys

import torch

# _common is a sibling under training/ (no __init__): add its directory to the path.
# codec.py is at training/opd_v2/src/opd_v2/ -> go up 3 levels to training/, then into _common/.
_COMMON = os.path.join(os.path.dirname(__file__), "..", "..", "..", "_common")
sys.path.insert(0, os.path.abspath(_COMMON))
from hidden_codec import BLOCK, ROT_SEED, Rotator, decode, encode  # noqa: E402

__all__ = ["BLOCK", "ROT_SEED", "Rotator", "encode", "decode", "build_w_rot", "load_teacher_head"]


def load_teacher_head(model_path: str, hid: int, device: str = "cpu") -> torch.Tensor:
    """Load `head.weight` bf16 `[V, hid]` from the teacher HF shards."""
    from safetensors import safe_open

    idx = json.load(open(f"{model_path}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{model_path}/{idx['head.weight']}", framework="pt", device="cpu") as f:
        w = f.get_tensor("head.weight")
    assert w.dtype == torch.bfloat16 and w.shape[1] == hid, (w.dtype, tuple(w.shape))
    return w.to(device)


def build_w_rot(model_path: str, hid: int, device: str = "cpu",
                seed: int = ROT_SEED) -> tuple[torch.Tensor, Rotator]:
    """Returns (W_rot [V, hid] bf16, rotator). Called once at trainer startup; W_rot is kept resident."""
    rot = Rotator(hid, device=device, seed=seed)
    head = load_teacher_head(model_path, hid, device=device)
    w_rot = rot.fold_head(head)
    return w_rot, rot
