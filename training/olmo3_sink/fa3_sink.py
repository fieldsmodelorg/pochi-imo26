# Copyright 2026 proof-pilot. Apache-2.0.
"""Trainable attention sinks on top of native FlashAttention 3.

Native FA3 has no sink (`s_aux`) support and vllm-fa3's sink is forward-only. This
module adds a fully-differentiable per-head sink as a cheap post-correction on FA3's
varlen output, recovering FA3 speed + trainable sinks.

Math
----
FA3 returns the no-sink output `o = Σ_i softmax(s)_i v_i` and `lse = log Σ_i exp(s_i)`.
A per-head sink adds one extra logit `sink` to the softmax denominator:

    D     = Σ_i exp(s_i) + exp(sink) = exp(lse) + exp(sink)
    scale = D / Σ_i exp(s_i)         = 1 + exp(sink - lse)
    o_sink = o / scale               (sink contributes 0 to the value mix)
    lse'   = lse + log(scale)        = log D     (sink-inclusive lse)

Key trick: FA3's own autograd saves `(q,k,v,out,softmax_lse)` and its backward enters
only through `out` and `softmax_lse`. If we overwrite those saved tensors in place with
the sink-corrected `o_sink` / `lse'` (via `.data.copy_`, which bypasses the autograd
version counter), FA3's native backward computes the EXACT dq/dk/dv for sink attention.
We then add the closed-form sink gradient:

    dL/dsink_h = -Σ_t p^sink_{h,t} · δ_{t,h},   p^sink = exp(sink - lse'),  δ = o_sink · do
"""

from __future__ import annotations

import torch

from flash_attn_interface import flash_attn_varlen_func


class _AttnSinkMerge(torch.autograd.Function):
    """Post-correct FA3's (out, lse) for a per-head sink; exact fwd+bwd.

    Shapes: out [T, H, D] (varlen packed), lse [H, T], sink [H].
    """

    @staticmethod
    def forward(ctx, out, lse, sink):
        z = sink.unsqueeze(1) - lse                               # [H, T]  (= sink - lse)
        inv_scale = torch.reciprocal(1.0 + torch.exp(z))          # [H, T]  = Σexp / D
        # Overwrite FA3's saved tensors IN PLACE -> its native backward becomes exact.
        # In-place mul_/add_ in bf16 (no [T,H,D] fp32 intermediate); `.data` keeps the
        # autograd version counter unchanged on purpose.
        out.data.mul_(inv_scale.transpose(0, 1).unsqueeze(-1).to(out.dtype))   # o_sink
        lse.data.add_(torch.log1p(torch.exp(z)))                  # lse' = lse + log(scale)
        ctx.save_for_backward(out, lse, sink)
        return out

    @staticmethod
    def backward(ctx, dout):
        out, lse, sink = ctx.saved_tensors                        # out=o_sink, lse=lse'
        delta = torch.sum(out * dout, dim=-1)                     # [T, H]
        p_sink = torch.exp(sink.unsqueeze(1) - lse)               # [H, T] = exp(sink)/D
        dsink = -(p_sink * delta.transpose(0, 1)).sum(dim=-1)     # [H]
        # `dout` is routed straight to `out`, i.e. into FA3's backward as its grad-output,
        # which (with the corrected saved out/lse) yields the right dq/dk/dv.
        return dout, None, dsink


def fa3_varlen_attn_with_sink(
    q, k, v, sink,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    softmax_scale=None, causal=True, window_size=(-1, -1),
):
    """Varlen FA3 attention with a trainable per-head sink.

    q,k,v: [total_tokens, H, D] (fp16/bf16). sink: [H] (any float, grads flow).
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    res = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        softmax_scale=softmax_scale, causal=causal, window_size=window_size,
        return_attn_probs=True,
    )
    out, lse = res[0], res[1]
    return _AttnSinkMerge.apply(out, lse, sink.to(lse.dtype))
