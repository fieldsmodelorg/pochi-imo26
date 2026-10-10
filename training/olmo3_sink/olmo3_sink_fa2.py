# Copyright 2026 proof-pilot. Apache-2.0.
"""olmo3_sink_fa2 — a transformers attn_implementation that carries the learnable sink on **stock FA2**,
torch.compile-safe.

Why this exists: transformers' `flash_attention_2` only applies `s_aux` when the installed flash
function accepts a sink arg (FA3's `s_aux` / FA4's `learnable_sink`). Stock FA2 has none, so HF
silently DROPS the sink (verified). Yi-Chia's `olmo3_sink_fa3` needs Hopper FA3. On B200 (no FA3) we
apply the sink as a **post-correction on FA2's (out, lse)** — the exact re-normalization from her
`fa3_sink.py`, which is flash-backend-agnostic:

    o_sink = o_nosink · exp(lse_base − logaddexp(lse_base, sink))     lse = logaddexp(lse_base, sink)

Forward/backward are each a `torch.library.custom_op` with a registered fake, so Dynamo treats them as
opaque nodes (**no graph break under torch.compile + gradient checkpointing**), matching the FA3 kernel.
The backward feeds FA2's native varlen backward the sink-corrected `out` + sink-inclusive `lse` (FA2
recomputes P = exp(scores − lse), so it reconstructs the sink weights → exact dq/dk/dv); `dsink` is a
closed-form Triton reduction. Verified fp64-exact vs the eager gpt-oss sink reference.

Register with `register_olmo3_sink_fa2()`, then load with `attn_implementation="olmo3_sink_fa2"`.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from transformers import AttentionInterface
from transformers.modeling_flash_attention_utils import (
    _is_packed_sequence,
    prepare_fa_kwargs_from_position_ids,
)

import flash_attn as _flash_attn
from flash_attn.flash_attn_interface import (  # stock FA2, unmodified
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)

# We call flash-attn's private varlen fwd/bwd POSITIONALLY, passing window_size as two separate ints
# (window_size_left, window_size_right) — the layout since flash-attn 2.7. On <=2.6 window_size was a
# single tuple, so our (wl, wr) would silently misalign onto (softcap, alibi) -> wrong attention.
# The private fn is wrapped (its signature reads as (*args, **kwargs)), so guard by VERSION, not by
# introspecting param names. Fails loud on an incompatible wheel instead of computing wrong results.
_FA_VER = tuple(int(x) for x in _flash_attn.__version__.split(".")[:2])
assert _FA_VER >= (2, 7), (
    f"olmo3_sink_fa2 needs flash-attn>=2.7 (split window_size_left/right args); "
    f"got {_flash_attn.__version__} — pin a compatible wheel."
)

ATTN_NAME = "olmo3_sink_fa2"


# --- closed-form dsink reduction (arch-agnostic Triton) ----------------------------------
@triton.jit
def _dsink_kernel(o_ptr, do_ptr, sink_ptr, lse2_ptr, dsink_ptr, T, D, so_t, so_h, sl_h, sl_t,
                  BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    d = tl.arange(0, BLOCK_D)
    tm, dm = t < T, d < D
    base = t[:, None] * so_t + h * so_h + d[None, :]
    m = tm[:, None] & dm[None, :]
    o = tl.load(o_ptr + base, mask=m, other=0.0).to(tl.float32)
    g = tl.load(do_ptr + base, mask=m, other=0.0).to(tl.float32)
    delta = tl.sum(o * g, axis=1)
    lse2 = tl.load(lse2_ptr + h * sl_h + t * sl_t, mask=tm, other=0.0)
    sink = tl.load(sink_ptr + h)
    tl.atomic_add(dsink_ptr + h, tl.sum(tl.where(tm, -tl.exp(sink - lse2) * delta, 0.0)))


def _dsink(o_sink, do, sink, lse2):
    """dsink_h = -Σ_t exp(sink_h − lse2_{h,t})·(o_sink·do)_{t,h}. o_sink,do [T,H,D]; lse2 [H,T]."""
    T, H, D = o_sink.shape
    dsink = torch.zeros(H, device=o_sink.device, dtype=torch.float32)
    _dsink_kernel[(H, triton.cdiv(T, 64))](
        o_sink, do, sink, lse2, dsink, T, D,
        o_sink.stride(0), o_sink.stride(1), lse2.stride(0), lse2.stride(1),
        BLOCK_T=64, BLOCK_D=triton.next_power_of_2(D))
    return dsink


# --- forward / backward as opaque, fake-backed custom ops (torch.compile-safe) -----------
@torch.library.custom_op("olmo3_sink_fa2::fwd", mutates_args=())
def _fa2_sink_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sink: torch.Tensor,
                  cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int,
                  scale: float, causal: bool, wl: int, wr: int) -> tuple[torch.Tensor, torch.Tensor]:
    r = _flash_attn_varlen_forward(q, k, v, cu_q, cu_k, max_q, max_k, 0.0, scale, causal, wl, wr,
                                   0.0, None, False, None)
    out, lse_base = r[0], r[1]                                            # out [T,H,D], lse [H,T]
    sink = sink.to(torch.float32).view(-1, 1)
    lse = torch.logaddexp(lse_base.to(torch.float32), sink)              # sink-inclusive lse
    alpha = torch.exp(lse_base.to(torch.float32) - lse).transpose(0, 1).unsqueeze(-1)
    # Rescale IN-PLACE on flash's own output (an internal, non-input tensor) -> no extra [T,H,D]
    # allocation. Safe under torch.compile: the mutation is inside the opaque custom_op.
    out.mul_(alpha.to(out.dtype))
    return out, lse


@_fa2_sink_fwd.register_fake
def _(q, k, v, sink, cu_q, cu_k, max_q, max_k, scale, causal, wl, wr):
    T, H = q.shape[0], q.shape[1]
    return torch.empty_like(q), q.new_empty((H, T), dtype=torch.float32)


@torch.library.custom_op("olmo3_sink_fa2::bwd", mutates_args=())
def _fa2_sink_bwd(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  out: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor,
                  cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int,
                  scale: float, causal: bool, wl: int, wr: int
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_varlen_backward(do, q, k, v, out, lse, dq, dk, dv, cu_q, cu_k, max_q, max_k,
                                0.0, scale, causal, wl, wr, 0.0, None, False, None)
    return dq, dk, dv, _dsink(out, do, sink, lse)


@_fa2_sink_bwd.register_fake
def _(do, q, k, v, out, lse, sink, cu_q, cu_k, max_q, max_k, scale, causal, wl, wr):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), torch.empty_like(sink)


def _setup_context(ctx, inputs, output):
    q, k, v, sink, cu_q, cu_k, max_q, max_k, scale, causal, wl, wr = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, sink, cu_q, cu_k)
    ctx.max_q, ctx.max_k, ctx.scale, ctx.causal, ctx.wl, ctx.wr = max_q, max_k, scale, causal, wl, wr


def _backward(ctx, grad_out, grad_lse):
    q, k, v, out, lse, sink, cu_q, cu_k = ctx.saved_tensors
    dq, dk, dv, ds = torch.ops.olmo3_sink_fa2.bwd(
        grad_out.contiguous(), q, k, v, out, lse, sink, cu_q, cu_k,
        ctx.max_q, ctx.max_k, ctx.scale, ctx.causal, ctx.wl, ctx.wr)
    return dq, dk, dv, ds, None, None, None, None, None, None, None, None


_fa2_sink_fwd.register_autograd(_backward, setup_context=_setup_context)


def fa2_varlen_attn_with_sink_kernel(q, k, v, sink, cu_q, cu_k, max_q, max_k,
                                     softmax_scale=None, causal=True, window_size=(-1, -1)):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    wl, wr = window_size
    out, _ = _fa2_sink_fwd(q, k, v, sink.to(torch.float32), cu_q, cu_k, int(max_q), int(max_k),
                           float(softmax_scale), bool(causal), int(wl), int(wr))
    return out


# --- transformers attention-interface adapter (mirrors attention.py, FA2 backend) --------
def fa2_sink_attention_forward(module, query, key, value, attention_mask=None,
                               scaling=None, dropout=0.0, sliding_window=None,
                               s_aux=None, **kwargs):
    B, Hq, S, D = query.shape
    Hkv = key.shape[1]
    sink = s_aux if s_aux is not None else module.sinks
    q = query.transpose(1, 2).reshape(B * S, Hq, D)
    k = key.transpose(1, 2).reshape(B * S, Hkv, D)
    v = value.transpose(1, 2).reshape(B * S, Hkv, D)
    cu_q = kwargs.get("cu_seq_lens_q")
    cu_k = kwargs.get("cu_seq_lens_k")
    max_q = kwargs.get("max_length_q")
    max_k = kwargs.get("max_length_k")
    if cu_q is None:
        position_ids = kwargs.get("position_ids")
        if position_ids is not None and _is_packed_sequence(position_ids, B):
            (cu_q, cu_k), (mq, mk) = prepare_fa_kwargs_from_position_ids(position_ids)
            max_q, max_k = int(mq), int(mk)
        else:
            cu_q = torch.arange(0, (B + 1) * S, S, device=q.device, dtype=torch.int32)
            cu_k = cu_q
            max_q = max_k = S
    window = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    out = fa2_varlen_attn_with_sink_kernel(q, k, v, sink, cu_q, cu_k, int(max_q), int(max_k),
                                           softmax_scale=scaling, causal=True, window_size=window)
    return out.reshape(B, S, Hq, D), None


def register_olmo3_sink_fa2() -> None:
    AttentionInterface.register(ATTN_NAME, fa2_sink_attention_forward)
