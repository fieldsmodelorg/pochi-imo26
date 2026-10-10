# Copyright 2026 proof-pilot. Apache-2.0.
"""Regression test: the production batched no-dup FA3 kernel must match the
per-offset reference (forward + all 6 gradients) to bf16 tolerance.

The per-offset implementation is the verified ground truth; the batched kernel
(production default) collapses its offset loop into one batched flash call. This
test pins their equivalence so future kernel edits can't silently diverge.

Run (needs FA3 + a GPU):
  PYTHONPATH=$ROOT:$ROOT/training/dflash python training/dflash/tests/test_nodup_attn.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from fa3_nodup_attention import (
    no_dup_full_attention_dense_block,
    no_dup_full_attention_dense_block_per_offset,
)


def _run(fn, inp, window, scale, dev):
    xs = [t.detach().clone().requires_grad_(True) for t in inp]
    out = fn(*xs, window, scale)
    gen = torch.Generator(device=dev).manual_seed(7)
    go = torch.randn(out.shape, device=dev, dtype=out.dtype, generator=gen)
    grads = torch.autograd.grad(out, xs, go)
    return out, grads


def test_batched_matches_per_offset():
    assert torch.cuda.is_available(), "needs a GPU + FA3"
    torch.manual_seed(0)
    dev = "cuda:0"
    names = ["dq", "dk_ctx", "dv_ctx", "dk_blk", "dv_blk", "dsink"]
    # cover GQA 4:1 (7B geom) and 5:1 (32B geom), and a non-divisible chunk tail
    cases = [
        dict(B=10, C=128, hq=32, hkv=8, d=128, b=10, W=128),   # 7B
        dict(B=11, C=96, hq=40, hkv=8, d=128, b=11, W=512),    # 32B
        dict(B=11, C=37, hq=40, hkv=8, d=128, b=11, W=512),    # ragged last-chunk size
    ]
    n_checks = 0
    for cs in cases:
        B, C, hq, hkv, d, b, W = (cs[k] for k in ("B", "C", "hq", "hkv", "d", "b", "W"))
        K = C + W - 1
        scale = d ** -0.5

        def mk(*s):
            return torch.randn(*s, device=dev, dtype=torch.bfloat16, requires_grad=True)

        inp = [mk(B, C, hq, d), mk(K, hkv, d), mk(K, hkv, d), mk(C, b, hkv, d), mk(C, b, hkv, d), mk(hq)]
        o_ref, g_ref = _run(no_dup_full_attention_dense_block_per_offset, inp, W, scale, dev)
        o_bat, g_bat = _run(no_dup_full_attention_dense_block, inp, W, scale, dev)

        fwd_rel = (o_ref.float() - o_bat.float()).abs().max().item() / (o_ref.float().abs().max().item() + 1e-9)
        assert fwd_rel < 2e-2, f"{cs}: forward rel {fwd_rel:.2e}"
        n_checks += 1
        for name, a, c in zip(names, g_ref, g_bat):
            rel = (a.float() - c.float()).abs().max().item() / (a.float().abs().max().item() + 1e-9)
            assert rel < 6e-2, f"{cs}: grad {name} rel {rel:.2e}"
            n_checks += 1
        print(f"OK {cs}  fwd_rel={fwd_rel:.2e}")
    print(f"OK test_batched_matches_per_offset: {n_checks} checks passed across {len(cases)} geometries")


if __name__ == "__main__":
    test_batched_matches_per_offset()
