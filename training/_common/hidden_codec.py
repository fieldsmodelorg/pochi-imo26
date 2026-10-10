# Copyright 2026 proof-pilot. Apache-2.0.
"""had+int6_blk32 codec for stored teacher hidden states (V4-Flash 4096 / V4-Pro 7168).

Storage format chosen by training/teacher_extract/_quant_study2.py (Flash) and
re-validated for Pro by _quant_study_pro.py: randomized orthonormal rotation
(gaussianizes outliers) + symmetric int6 with one fp16 scale per 32-dim block.
0.406x of bf16; reconstructed-logit KL = 1/7 (Flash) / 1/14 (Pro) of the teacher
engine's own rerun nondeterminism, with no measurable distribution bias.

Rotation by hidden dim d:
  - d = power of two (Flash 4096): Sylvester Hadamard Hn with random signs D.
    FROZEN — Flash shards exist; this path must stay bit-identical.
  - otherwise (Pro 7168 = 7*1024): no Sylvester Hadamard exists; use the
    Kronecker rotation M = kron(Q_m, H_p) with p the largest power-of-two
    factor and Q_m a seeded random orthogonal (float64 QR, det-sign fixed) —
    exactly _quant_study_pro.KronRotator, so the validated numbers carry over.

The rotation is free at training time: fold it into the teacher head once,
    W_rot = (W * D) @ M.T          # [V, H], one-time (M = Hn for Sylvester)
    logits = decode(packed, scales) @ W_rot.T   ==   h @ W.T   (up to int6 noise)
so decode() returns ROTATED-space hidden and no inverse rotation ever runs in the
hot path. Keep ROT_SEED stable forever once shards exist.

Packing: q in [-31, 31] -> +31 -> [0, 62] (6 bits); 4 values -> 3 bytes.
Per token: d*6/8 packed + (d/32) fp16 scales = 3328 B (4096) / 5824 B (7168).
"""
from __future__ import annotations

import torch

H_DIM = 4096                       # Flash default (back-compat)
BLOCK = 32
ROT_SEED = 7                       # frozen: shards are useless if this changes


def _sylvester(p: int, device: str) -> torch.Tensor:
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < p:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    assert H.shape[0] == p, f"p={p} must be a power of 2"
    return H / p ** 0.5


class Rotator:
    """x -> (x * D) @ M.T with random signs D and orthonormal M.

    d power of two: M = Hn (symmetric Sylvester Hadamard) — the frozen Flash
    path, bit-identical to the original 4096-only implementation.
    Otherwise: M = kron(Q_m, H_p), p = largest power-of-two factor of d."""

    def __init__(self, d: int = H_DIM, device: str = "cpu", seed: int = ROT_SEED):
        assert d % BLOCK == 0 and d % 4 == 0, d
        self.d = d
        g = torch.Generator(device="cpu").manual_seed(seed)
        if d & (d - 1) == 0:
            # FROZEN draw order: D right after seeding (Flash shards depend on it).
            self.Hn = _sylvester(d, device)
            self.M = self.Hn               # symmetric: M.T == M
            self.D = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).float().to(device)
        else:
            p = d & (-d)                   # largest power-of-two factor (7168 -> 1024)
            m = d // p
            Q, R = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
            Q = (Q * torch.sign(torch.diagonal(R))).float().contiguous()
            self.Hn = None
            self.M = torch.kron(Q, _sylvester(p, "cpu").contiguous()).to(device)
            # draw order matches _quant_study_pro.KronRotator: Q first, then D
            self.D = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).float().to(device)

    def to(self, device):
        self.M = self.M.to(device)
        if self.Hn is not None:
            self.Hn = self.M
        self.D = self.D.to(device)
        return self

    def fwd(self, h: torch.Tensor) -> torch.Tensor:
        if self.Hn is not None:
            return (h.float() * self.D) @ self.Hn      # frozen Flash path
        return (h.float() * self.D) @ self.M.T

    def fold_head(self, w: torch.Tensor) -> torch.Tensor:
        """W [V, H] -> W_rot [V, H] with decode(y) @ W_rot.T == h @ W.T."""
        if self.Hn is not None:
            return ((w.float() * self.D) @ self.Hn).to(w.dtype)
        return ((w.float() * self.D) @ self.M.T).to(w.dtype)


def encode(h: torch.Tensor, rot: Rotator) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16/float [n, d] -> (packed uint8 [n, d*6/8], scales fp16 [n, d/32])."""
    n, d = h.shape
    assert d == rot.d, (d, rot.d)
    nb = d // BLOCK
    y = rot.fwd(h).view(n, nb, BLOCK)
    s = (y.abs().amax(-1, keepdim=True) / 31.0).half()
    sf = s.float()
    sf = torch.where(sf == 0, torch.ones_like(sf), sf)
    q = (y / sf).round().clamp(-31, 31).to(torch.int32).view(n, d) + 31   # [0, 62]
    q4 = q.view(n, d // 4, 4)
    word = q4[..., 0] | (q4[..., 1] << 6) | (q4[..., 2] << 12) | (q4[..., 3] << 18)
    packed = torch.stack(
        [word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF], dim=-1
    ).to(torch.uint8).view(n, d * 6 // 8)
    return packed, s.view(n, nb)


def decode(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """-> bf16 [n, d] in ROTATED space (use with fold_head'ed weights)."""
    n = packed.shape[0]
    d = packed.shape[1] * 8 // 6
    b = packed.view(n, -1, 3).to(torch.int32)
    word = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    q = torch.stack(
        [word & 0x3F, (word >> 6) & 0x3F, (word >> 12) & 0x3F, (word >> 18) & 0x3F],
        dim=-1,
    ).view(n, d) - 31
    y = q.view(n, d // BLOCK, BLOCK).float() * scales.view(n, d // BLOCK, 1).float()
    return y.view(n, d).to(torch.bfloat16)


def _selftest_dim(d: int, dev: str):
    rot = Rotator(d, device=dev)
    h = (torch.randn(512, d, device=dev) * 5).to(torch.bfloat16)
    packed, scales = encode(h, rot)
    assert packed.shape == (512, d * 6 // 8) and packed.dtype == torch.uint8
    assert scales.shape == (512, d // BLOCK) and scales.dtype == torch.float16
    y = decode(packed.to(dev), scales.to(dev))

    # 1. roundtrip in rotated space: |y - rot(h)| within int6 grid (~scale/2)
    y_ref = rot.fwd(h)
    err = (y.float() - y_ref).abs().mean()
    grid = (scales.float().mean() / 2)
    print(f"[d={d}] roundtrip |dy| mean {err:.4e} (grid/2 = {grid:.4e})")
    assert err < grid * 1.2

    # 2. fold-head equivalence: decode(y) @ W_rot.T ~= h @ W.T
    w = (torch.randn(1000, d, device=dev) / d ** 0.5).to(torch.bfloat16)
    w_rot = rot.fold_head(w)
    lo_ref = h.float() @ w.float().T
    lo_q = y.float() @ w_rot.float().T
    rel = (lo_q - lo_ref).norm() / lo_ref.norm()
    print(f"[d={d}] fold-head logits rel err {rel:.4e}")
    assert rel < 3e-2   # iid-gaussian worst case; real-hidden KL measured in studies

    # 3. exact-rotation control (no quant): error should be ~fp rounding only
    lo_rot = y_ref @ w_rot.float().T
    rel0 = (lo_rot - lo_ref).norm() / lo_ref.norm()
    print(f"[d={d}] rotation-only logits rel err {rel0:.4e} (fp rounding)")
    assert rel0 < 2e-3

    bpt = d * 6 // 8 + d // BLOCK * 2
    print(f"[d={d}] bytes/token = {bpt} (ratio {bpt / (d * 2):.3f})")


def _selftest():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _selftest_dim(4096, dev)   # Flash (frozen Sylvester path)
    _selftest_dim(7168, dev)   # Pro (Kronecker path)
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
