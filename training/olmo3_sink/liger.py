# Copyright 2026 proof-pilot. Apache-2.0.
"""Optional Liger-Kernel acceleration for Olmo3Sink.

Liger's Triton kernels (RoPE, RMSNorm, SwiGLU, fused-linear-cross-entropy) are fully
orthogonal to our `olmo3_sink_fa3` attention: Liger patches MLP / norms / RoPE /
lm_head-loss, while the sink attention is left untouched. Verified on Olmo-3-7B: loss
unchanged (diff 1e-4), grads still flow to sinks, ~20% lower peak memory (fused-linear-CE
avoids materializing the [B,S,vocab] logits) and ~9% faster step.

Note: Liger's olmo3 patcher does NOT patch attention q_norm/k_norm (they remain stock
Olmo3RMSNorm — tiny, no measurable effect). RoPE patching only works because
`Olmo3SinkAttention.forward` references `modeling_olmo3.apply_rotary_pos_emb` at call time.
"""
from __future__ import annotations

from transformers import PreTrainedModel
from liger_kernel.transformers import apply_liger_kernel_to_olmo3


def apply_liger(
    model: PreTrainedModel,
    *,
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    fused_linear_cross_entropy: bool = True,
    cross_entropy: bool = False,
) -> PreTrainedModel:
    """In-place patch an already-loaded Olmo3Sink model with Liger kernels."""
    apply_liger_kernel_to_olmo3(
        model=model,
        rope=rope,
        rms_norm=rms_norm,
        swiglu=swiglu,
        fused_linear_cross_entropy=fused_linear_cross_entropy,
        cross_entropy=cross_entropy,
    )
    return model
