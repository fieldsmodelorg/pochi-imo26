"""DFlash Draft Model — OLMo3 architecture variant.

Mirrors the upstream DFlash draft (Qwen3-based, SpecForge lineage) but uses
OLMo3 components so the draft matches the olmo3_sink target family:

- post-norm residual structure (norm AFTER attn/mlp output, before the residual
  add; no input_layernorm) — exactly `Olmo3DecoderLayer`
- full-projection QK-norm (RMSNorm over num_heads*head_dim, not per-head)
- no attention bias
- per-layer-type RoPE: sliding_attention layers use default unscaled RoPE
  (theta 500k), full_attention layers use the configured YaRN — mirroring
  olmo3_sink's fix of the transformers v5 per-layer RoPE regression
- optional gpt-oss-style learnable attention sink (per Q-head scalar, `s_aux`,
  flex_attention backend only)
- GQA (e.g. 32Q/8KV for the proof-pilot draft)

DFlash specifics (identical mechanism to the Qwen3 draft):
- cross-attention KV layout [target context | draft blocks]: K/V for the
  context come from the *fused target hidden states* (multi-layer concat ->
  shared `fc` -> RMSNorm), K/V for the block positions from the draft itself
- learnable `mask_embed` replacing the embedding at all mask slots
- no embedding / lm_head of its own — borrows the frozen target's

Self-contained: depends only on `transformers` (this file is copied into every
checkpoint directory).
"""

from collections.abc import Callable
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.olmo3.modeling_olmo3 import (
    Olmo3Config,
    Olmo3MLP,
    Olmo3RMSNorm,
    Olmo3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """RoPE with DFlash length handling: K covers [context | blocks] (full
    position range), Q only the blocks — Q uses the LAST q_len positions."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Olmo3DFlashRotaryEmbedding(Olmo3RotaryEmbedding):
    """`Olmo3RotaryEmbedding` + per-instance `rope_type` override (same shim as
    olmo3_sink / upstream huggingface/transformers#45945): the 5.9.0 base class
    can only read the rope type from the config, so it cannot express the
    default-RoPE sliding layers."""

    def __init__(self, config: Olmo3Config, device=None, rope_type: str | None = None):
        nn.Module.__init__(self)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = rope_type or self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)


class Olmo3DFlashAttention(nn.Module):
    """Cross-attention: Q from draft hidden, K/V from cat(fused target context,
    draft hidden). OLMo3 conventions: full-projection QK-norm, no bias."""

    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        bias = config.attention_bias
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=bias
        )
        # OLMo3-style: RMSNorm over the FULL projection (all heads share the
        # norm denominator), applied before the head reshape.
        self.q_norm = Olmo3RMSNorm(
            config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps
        )
        self.k_norm = Olmo3RMSNorm(
            config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps
        )
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )
        dflash_cfg = getattr(config, "dflash_config", {}) or {}
        self.use_attention_sink = dflash_cfg.get("use_attention_sink", False)
        if self.use_attention_sink:
            self.sinks = nn.Parameter(torch.zeros(config.num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        q = self.q_norm(self.q_proj(hidden_states))
        q = q.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        k = self.k_norm(torch.cat([k_ctx, k_noise], dim=1))
        k = k.view(bsz, ctx_len + q_len, -1, self.head_dim).transpose(1, 2)

        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        v = torch.cat([v_ctx, v_noise], dim=1)
        v = v.view(bsz, ctx_len + q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.use_attention_sink:
            if self.config._attn_implementation != "flex_attention":
                raise RuntimeError(
                    "Attention sink requires the flex_attention backend, got: "
                    f"{self.config._attn_implementation}"
                )
            kwargs["s_aux"] = self.sinks

        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Olmo3DFlashDecoderLayer(GradientCheckpointingLayer):
    """OLMo3 post-norm layer: h += norm(attn(h)); h += norm(mlp(h))."""

    def __init__(self, config: Olmo3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Olmo3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Olmo3MLP(config)
        self.post_attention_layernorm = Olmo3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Olmo3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    """Spread capture layers evenly over [1, num_target_layers - 3]."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[torch.Tensor], layer_ids: Optional[list[int]]
) -> torch.Tensor:
    """Select layers from an `output_hidden_states=True` tuple (index 0 is the
    embedding output, hence the +1 offset) and concat on the feature dim."""
    offset = 1
    return torch.cat([hidden_states[i + offset] for i in layer_ids], dim=-1)


class Olmo3DFlashDraftModel(PreTrainedModel):
    config_class = Olmo3Config
    base_model_prefix = "model"
    _no_split_modules = ["Olmo3DFlashDecoderLayer"]
    supports_gradient_checkpointing = True
    _supports_flex_attn = True
    _supports_sdpa = True
    _supports_flash_attn = False

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                Olmo3DFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.norm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Per-layer-type RoPE: sliding layers default/unscaled, full layers
        # use the configured (YaRN) parameters — matches the olmo3_sink target.
        self.rotary_embs = nn.ModuleDict(
            {
                "sliding_attention": Olmo3DFlashRotaryEmbedding(config, rope_type="default"),
                "full_attention": Olmo3DFlashRotaryEmbedding(config),
            }
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * target_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # mask_embed: learnable embedding for masked positions. Never leave it
        # zeros at train start (RMSNorm of an all-zero row is NaN); train.py
        # initializes it from the mean of the target embedding table when it is
        # absent from the checkpoint.
        self.mask_embed = nn.Parameter(torch.zeros(config.hidden_size))
        if target_hidden_size != config.hidden_size:
            self.input_proj = nn.Linear(target_hidden_size, config.hidden_size, bias=False)
            self.output_proj = nn.Linear(config.hidden_size, target_hidden_size, bias=False)
        else:
            self.input_proj = None
            self.output_proj = None
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.post_init()

    def _init_weights(self, module):
        # tf>=5 re-runs _init_weights over EVERY module after loading a
        # checkpoint and relies on `transformers.initialization` wrappers
        # (which no-op on params carrying `_is_hf_initialized`) to protect
        # loaded values. A raw `module.weight.data.normal_()` here silently
        # re-randomizes the whole loaded model (= the olmo3_sink sink-zeroing
        # bug, generalized). So: defer to the guarded default for standard
        # modules, and use guarded init for our custom params.
        from transformers import initialization as hf_init

        super()._init_weights(module)
        if isinstance(module, Olmo3DFlashAttention) and module.use_attention_sink:
            hf_init.zeros_(module.sinks)
        if isinstance(module, Olmo3DFlashDraftModel):
            hf_init.zeros_(module.mask_embed)

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        swa_attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        noise_mask: Optional[torch.BoolTensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.input_proj is not None:
            hidden_states = self.input_proj(noise_embedding)
        else:
            hidden_states = noise_embedding
        if noise_mask is not None:
            hidden_states = torch.where(
                noise_mask.unsqueeze(-1), self.mask_embed, hidden_states
            )
        target_hidden = self.hidden_norm(self.fc(target_hidden))

        position_embeddings = {
            ltype: self.rotary_embs[ltype](hidden_states, position_ids)
            for ltype in ("sliding_attention", "full_attention")
        }

        for layer_idx, layer in enumerate(self.layers):
            ltype = self.config.layer_types[layer_idx]
            if swa_attention_mask is not None and layer.self_attn.sliding_window is not None:
                layer_mask = swa_attention_mask
            else:
                layer_mask = attention_mask
            hidden_states = layer(
                target_hidden,
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings[ltype],
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        if self.output_proj is not None:
            hidden_states = self.output_proj(hidden_states)
        return hidden_states

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: Optional[list[int]] = None,
        temperature: float = 0.0,
    ):
        """Reference speculative-decoding loop (HF target with KV cache).

        NOTE: requires a cache-capable target attention backend (eager /
        flash); the training-only `olmo3_sink_fa3` adapter cannot decode.
        """
        from transformers import DynamicCache

        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size

        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=input_ids.device).unsqueeze(0)

        past_target = DynamicCache()

        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden_full = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        acceptance_lengths = []
        start = num_input_tokens
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            noise_mask = block_output_ids == self.mask_token_id

            ctx_len = target_hidden_full.shape[1]
            full_pos = torch.cat(
                [position_ids[:, :ctx_len], block_position_ids], dim=1
            )
            draft_hidden = self(
                position_ids=full_pos,
                noise_embedding=noise_embedding,
                noise_mask=noise_mask,
                target_hidden=target_hidden_full,
            )
            draft_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
            block_output_ids[:, 1:] = sample(draft_logits, temperature)

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_target,
                use_cache=True,
                output_hidden_states=True,
            )
            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
            start += acceptance_length + 1
            past_target.crop(start)
            new_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]
            target_hidden_full = torch.cat(
                [target_hidden_full[:, : start - acceptance_length - 1], new_hidden], dim=1
            )
            acceptance_lengths.append(acceptance_length + 1)
            if stop_token_ids is not None and any(
                stop_id in output_ids[0, num_input_tokens:start].tolist()
                for stop_id in stop_token_ids
            ):
                break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_t = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_idx = torch.isin(output_ids[0][num_input_tokens:], stop_t).nonzero(
                as_tuple=True
            )[0]
            if stop_idx.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_idx[0] + 1]
        return output_ids, acceptance_lengths
