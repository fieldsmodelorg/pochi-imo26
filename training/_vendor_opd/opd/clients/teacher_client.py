# Copyright 2026 proof-pilot. Apache-2.0.
"""Client for the teacher scoring service (PLAN §4).

`score(input_ids, start)` -> returns (packed_bytes, scales_bytes, seq_len); the trainer reconstructs
teacher logits with `opd.codec.decode` + a resident `W_rot`. The bytes are stored directly in the
Trajectory (not decoded to a tensor on the client side, to save memory; the trainer decodes to GPU
only when it pulls a batch).
"""
from __future__ import annotations

import numpy as np
import torch

from opd.config import HID_DIM


class TeacherClient:
    def __init__(self, base_url: str, timeout: float = 600.0):
        import httpx
        self.base = base_url.rstrip("/")
        self._c = httpx.Client(timeout=timeout)

    def health(self) -> dict:
        return self._c.get(f"{self.base}/health").json()

    def score(self, input_ids: list[int], start: int = 0, return_top1: bool = False):
        """Return (packed_bytes, scales_bytes, seq_len), optionally with int32 teacher top1 bytes.

        When `return_top1=True`, a newer teacher appends `seq_len` int32 argmax ids after scales,
        returning (packed_bytes, scales_bytes, seq_len, top1_bytes). If an older teacher has no header,
        top1_bytes=None.
        """
        payload = {"input_ids": input_ids, "start": start}
        if return_top1:
            payload["return_top1"] = True
        r = self._c.post(f"{self.base}/score", json=payload)
        r.raise_for_status()
        seq = int(r.headers["X-Seq-Len"])
        npk = int(r.headers["X-Packed-Bytes"])
        ntop = int(r.headers.get("X-Top1-Bytes", "0"))
        if ntop:
            if ntop != seq * 4:
                raise RuntimeError(f"teacher top1 byte length mismatch: got={ntop} expected={seq * 4}")
            return r.content[:npk], r.content[npk:-ntop], seq, r.content[-ntop:]
        if return_top1:
            return r.content[:npk], r.content[npk:], seq, None
        return r.content[:npk], r.content[npk:], seq


def decode_teacher_hidden(packed_bytes: bytes, scales_bytes: bytes, seq_len: int,
                          device: str = "cuda", hid: int = HID_DIM) -> torch.Tensor:
    """bytes -> rotated-space bf16 hidden [seq_len, hid] (for use with the trainer's W_rot)."""
    from opd.codec import decode
    pcols = hid * 6 // 8
    scols = hid // 32
    packed = torch.from_numpy(
        np.frombuffer(packed_bytes, dtype=np.uint8).reshape(seq_len, pcols).copy())
    scales = torch.from_numpy(
        np.frombuffer(scales_bytes, dtype=np.float16).reshape(seq_len, scols).copy())
    return decode(packed.to(device), scales.to(device))


def decode_teacher_top1(top1_bytes: bytes, seq_len: int, device: str = "cuda") -> torch.Tensor:
    """int32 bytes -> int64 token ids [seq_len]."""
    top1 = torch.from_numpy(np.frombuffer(top1_bytes, dtype=np.int32).reshape(seq_len).copy())
    return top1.to(device=device, dtype=torch.long)
