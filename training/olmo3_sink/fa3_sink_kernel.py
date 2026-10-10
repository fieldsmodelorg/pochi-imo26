# Copyright 2026 proof-pilot. Apache-2.0.
"""In-kernel attention sink on patched FA3 (targets >95% of no-sink FA3 for ALL packing).

The FA3 forward kernel epilogue is patched (see flash-attention/hopper/softmax.h finalize)
to add exp(sink-M) to the softmax denominator, so it directly outputs sink-corrected `o`
and sink-inclusive `lse` — NO forward post-pass. Backward feeds those to FA3's native
backward for exact dq/dk/dv (proven equivalent), plus a cheap dsink reduction.

torch.compile: forward and backward are each a `torch.library.custom_op` with a registered
fake (meta) impl, so Dynamo treats them as single opaque nodes with known output shapes
instead of graph-breaking on the fake-less `flash_attn_3.fwd`. The FA3/Triton internals
live in the eager op bodies and are never traced. This lets a compiled decoder layer fuse
everything around attention into one graph (critical for compile + gradient checkpointing,
which re-runs the forward in backward and would otherwise pay the graph-break cost twice).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from flash_attn_interface import _flash_attn_backward


@triton.jit
def _dsink_kernel(o_ptr, do_ptr, sink_ptr, lse2_ptr, dsink_ptr, T, D,
                  so_t, so_h, sl_h, sl_t, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    d = tl.arange(0, BLOCK_D)
    tm = t < T
    dm = d < D
    base = t[:, None] * so_t + h * so_h + d[None, :]
    m = tm[:, None] & dm[None, :]
    o = tl.load(o_ptr + base, mask=m, other=0.0).to(tl.float32)
    g = tl.load(do_ptr + base, mask=m, other=0.0).to(tl.float32)
    delta = tl.sum(o * g, axis=1)                              # [BLOCK_T]
    lse2 = tl.load(lse2_ptr + h * sl_h + t * sl_t, mask=tm, other=0.0)
    sink = tl.load(sink_ptr + h)
    contrib = tl.where(tm, -tl.exp(sink - lse2) * delta, 0.0)
    tl.atomic_add(dsink_ptr + h, tl.sum(contrib))


def _dsink(o_sink, do, sink, lse2):
    """dsink_h = -Σ_t exp(sink_h - lse2_{h,t})·(o_sink·do)_{t,h}. o_sink,do [T,H,D]; lse2 [H,T]."""
    T, H, D = o_sink.shape
    dsink = torch.zeros(H, device=o_sink.device, dtype=torch.float32)
    BLOCK_T = 64
    BLOCK_D = triton.next_power_of_2(D)
    grid = (H, triton.cdiv(T, BLOCK_T))
    _dsink_kernel[grid](o_sink, do, sink, lse2, dsink, T, D,
                        o_sink.stride(0), o_sink.stride(1), lse2.stride(0), lse2.stride(1),
                        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D)
    return dsink


# --- forward as an opaque, fake-backed custom op -----------------------------------------
@torch.library.custom_op("olmo3_sink::fa3_sink_fwd", mutates_args=())
def _fa3_sink_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sink: torch.Tensor,
    cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int,
    scale: float, causal: bool, wl: int, wr: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # patched op: `sink` is the trailing arg; kernel applies it in the epilogue.
    out, lse, _, _ = torch.ops.flash_attn_3.fwd(
        q, k, v,
        None, None, None, None,            # k_new, v_new, qv, out
        cu_q, cu_k, None,                  # cu_seqlens q/k/k_new
        None, None,                        # seqused q/k
        max_q, max_k,
        None, None, None,                  # page_table, kv_batch_idx, leftpad_k
        None, None, None,                  # rotary cos/sin, seqlens_rotary
        None, None, None,                  # q/k/v_descale
        scale, causal, wl, wr,
        0,                                 # attention_chunk
        0.0,                               # softcap
        False,                             # rotary_interleaved
        None,                              # scheduler_metadata
        1,                                 # num_splits (DISABLE_SPLIT -> must be 1)
        None,                              # pack_gqa
        0,                                 # sm_margin
        sink,                              # proof-pilot sink
    )
    return out, lse


@_fa3_sink_fwd.register_fake
def _(q, k, v, sink, cu_q, cu_k, max_q, max_k, scale, causal, wl, wr):
    # out matches q [total_q, H, D]; varlen lse is [H, total_q] (see _dsink indexing).
    T, H = q.shape[0], q.shape[1]
    return torch.empty_like(q), q.new_empty((H, T), dtype=torch.float32)


# --- backward as an opaque, fake-backed custom op ----------------------------------------
@torch.library.custom_op("olmo3_sink::fa3_sink_bwd", mutates_args=())
def _fa3_sink_bwd(
    do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    out: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor,
    cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int,
    scale: float, causal: bool, wl: int, wr: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(do, q, k, v, out, lse, cu_q, cu_k, None, None,
                         max_q, max_k, dq, dk, dv, scale, causal, wl, wr, 0.0, False, 0)
    dsink = _dsink(out, do, sink, lse)
    return dq, dk, dv, dsink


@_fa3_sink_bwd.register_fake
def _(do, q, k, v, out, lse, sink, cu_q, cu_k, max_q, max_k, scale, causal, wl, wr):
    return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v),
            torch.empty_like(sink))


def _setup_context(ctx, inputs, output):
    q, k, v, sink, cu_q, cu_k, max_q, max_k, scale, causal, wl, wr = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, sink, cu_q, cu_k)
    ctx.max_q, ctx.max_k = max_q, max_k
    ctx.scale, ctx.causal, ctx.wl, ctx.wr = scale, causal, wl, wr


def _backward(ctx, grad_out, grad_lse):
    q, k, v, out, lse, sink, cu_q, cu_k = ctx.saved_tensors
    dq, dk, dv, dsink = torch.ops.olmo3_sink.fa3_sink_bwd(
        grad_out.contiguous(), q, k, v, out, lse, sink, cu_q, cu_k,
        ctx.max_q, ctx.max_k, ctx.scale, ctx.causal, ctx.wl, ctx.wr)
    # grads for (q, k, v, sink, cu_q, cu_k, max_q, max_k, scale, causal, wl, wr)
    return dq, dk, dv, dsink, None, None, None, None, None, None, None, None


_fa3_sink_fwd.register_autograd(_backward, setup_context=_setup_context)


def fa3_varlen_attn_with_sink_kernel(
    q, k, v, sink, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    softmax_scale=None, causal=True, window_size=(-1, -1),
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    wl, wr = window_size
    out, _lse = _fa3_sink_fwd(
        q, k, v, sink.to(torch.float32), cu_seqlens_q, cu_seqlens_k,
        int(max_seqlen_q), int(max_seqlen_k), float(softmax_scale), bool(causal), int(wl), int(wr))
    return out
