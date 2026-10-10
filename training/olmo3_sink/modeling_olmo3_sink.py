# Copyright 2026 proof-pilot. Apache-2.0.
"""Olmo3 with (1) gpt-oss-style learnable attention sinks, (2) FlashAttention
packing-metadata reuse, and (3) restored per-layer-type RoPE.

Design: subclass the stock `transformers.models.olmo3` classes and override only
what changes. Everything else (RMSNorm, sliding-window layer_types, MLP,
generation, TP/PP plans) is inherited unchanged.

Three changes
-------------
1. Attention sink. Each attention module gets `self.sinks`, an
   `[num_attention_heads]` learnable parameter, passed to the attention kernel as
   `s_aux`. The kernel appends it as an extra column of softmax logits that is
   dropped after normalization (see gpt-oss). NOTE: the `sdpa` backend does NOT
   support `s_aux`; use `eager` (debug), `flash_attention_2`, `flash_attention_3`,
   or `flex_attention`. We assert this in the attention forward.

2. Packing-metadata reuse. On a flash backend with packed `position_ids`, derive
   the varlen `cu_seqlens`/`max_seqlen` once in `Olmo3SinkModel.forward` and put
   them in the kwargs threaded to every layer. `_flash_attention_forward` then
   takes its precomputed-varlen branch instead of re-deriving per layer.

3. Per-layer-type RoPE (bug fix). OLMo 3 applies RoPE scaling (YaRN) only to
   `full_attention` layers; `sliding_attention` layers use default unscaled RoPE
   (so it was pretrained, so vLLM serves it, and so transformers 4.57 ran it).
   The transformers v5 RoPE refactor (huggingface/transformers#39847) regressed
   this to one global scaled `rotary_emb` for every layer; the pinned 5.9.0 (and
   main as of 2026-06) is affected. We restore the 4.57 behavior here, mirroring
   the open upstream fix huggingface/transformers#45945: `rotary_embs` ModuleDict
   with `sliding_attention` forced to `rope_type="default"`, dispatched per
   `config.layer_types[i]` in `Olmo3SinkModel.forward`. Drop this override once
   transformers ships the fix.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import (
    _is_packed_sequence,
    prepare_fa_kwargs_from_position_ids,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.olmo3 import modeling_olmo3 as _olmo3_mod
from transformers.models.olmo3.modeling_olmo3 import (
    Olmo3Attention,
    Olmo3DecoderLayer,
    Olmo3ForCausalLM,
    Olmo3Model,
    Olmo3PreTrainedModel,
    Olmo3RotaryEmbedding,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs


def eager_attention_forward_with_sink(
    module,
    query: torch.Tensor,   # [B, Hq, S, D]
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    s_aux: torch.Tensor | None = None,
    **kwargs,
):
    """gpt-oss-style eager attention with a per-head sink (correct CPU/debug reference).

    Identical to Olmo3's `eager_attention_forward` except the per-head `sinks` scalar is
    appended as an extra softmax logit and dropped after normalization (so a head can
    attend to "nothing"). Done in fp32 with a max-subtract for stability, matching the
    in-kernel FA3 sink. Stock Olmo3 eager ignores the sink, so this is the reference that
    actually exercises `self_attn.sinks` (e.g. the `eager` backend / CPU). Defined here
    (not in attention.py) so it has NO FlashAttention import dependency."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., : key_states.shape[-2]]

    sink = s_aux if s_aux is not None else module.sinks  # [Hq]
    sinks = sink.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], 1)
    combined = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1).float()
    combined = combined - combined.amax(dim=-1, keepdim=True)
    probs = torch.softmax(combined, dim=-1)[..., :-1].to(query.dtype)  # drop sink column
    probs = nn.functional.dropout(probs, p=dropout, training=module.training)

    attn_output = torch.matmul(probs, value_states).transpose(1, 2).contiguous()
    return attn_output, probs
from transformers.utils.generic import TransformersKwargs

from .configuration_olmo3_sink import Olmo3SinkConfig


# OLMo 3 RoPE is identical to the stock rotary embedding, except:
# - RoPE scaling is not applied to sliding window attention layers.
class Olmo3SinkRotaryEmbedding(Olmo3RotaryEmbedding):
    """`Olmo3RotaryEmbedding` + per-instance `rope_type` override (mirrors upstream
    fix huggingface/transformers#45945; the 5.9.0 base class can only read the
    rope type from the config, so it cannot express the default-RoPE sliding
    layers)."""

    def __init__(self, config: Olmo3SinkConfig, device=None, rope_type: str | None = None):
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


class Olmo3SinkAttention(Olmo3Attention):
    """Olmo3 attention + per-head learnable sink (`s_aux`)."""

    def __init__(self, config: Olmo3SinkConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        # One scalar logit per query head. Initialized by `_init_weights`.
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Body identical to Olmo3Attention.forward; the only change is the
        # `s_aux=self.sinks` argument to the attention interface.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # Reference the module attribute at call time (not a bound import) so that
        # Liger-Kernel's `apply_liger_kernel_to_olmo3(rope=True)` monkey-patch takes effect.
        query_states, key_states = _olmo3_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attn_impl = self.config._attn_implementation
        if attn_impl == "sdpa":
            raise ValueError(
                "Olmo3Sink uses attention sinks (s_aux), which the `sdpa` backend does not support. "
                "Load with attn_implementation in {'eager','flash_attention_2','flash_attention_3','flex_attention'}."
            )

        # Fallback (incl. attn_impl="eager") is our SINK-AWARE eager, not Olmo3's
        # sink-less one -- so the eager backend is a correct reference for the sink.
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn_impl, eager_attention_forward_with_sink)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Olmo3SinkDecoderLayer(Olmo3DecoderLayer):
    def __init__(self, config: Olmo3SinkConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.self_attn = Olmo3SinkAttention(config=config, layer_idx=layer_idx)


class Olmo3SinkPreTrainedModel(Olmo3PreTrainedModel):
    config_class = Olmo3SinkConfig
    _no_split_modules = ["Olmo3SinkDecoderLayer"]

    def _init_weights(self, module):
        super()._init_weights(module)
        # The sink parameter is new (absent from base Olmo3 checkpoints); when
        # loading such a checkpoint it is a "missing key" and lands here.
        # transformers >= 5.x calls _init_weights on EVERY module after loading
        # (relying on guarded init fns to skip loaded params); a raw fill_()
        # bypasses that guard and would silently zero TRAINED sinks, so only
        # fill when the param was not loaded from the checkpoint.
        if isinstance(module, Olmo3SinkAttention) and not getattr(
            module.sinks, "_is_hf_initialized", False
        ):
            module.sinks.data.fill_(self.config.sink_init_value)


class Olmo3SinkModel(Olmo3SinkPreTrainedModel, Olmo3Model):
    def __init__(self, config: Olmo3SinkConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Olmo3SinkDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Per-layer-type RoPE (mirrors huggingface/transformers#45945): sliding
        # layers use default unscaled RoPE; full-attention layers use the
        # configured (e.g. YaRN) RoPE. Replaces the buggy global `rotary_emb`.
        self.rotary_embs = nn.ModuleDict(
            {
                "sliding_attention": Olmo3SinkRotaryEmbedding(config=config, rope_type="default"),
                "full_attention": Olmo3SinkRotaryEmbedding(config=config),
            }
        )
        del self.rotary_emb
        self.post_init()

    # Body mirrors stock `Olmo3Model.forward` (transformers 5.9.0, incl. its
    # decorators) with two changes: (a) the packing-metadata reuse block, (b)
    # per-layer-type `position_embeddings` from `self.rotary_embs` instead of one
    # global `self.rotary_emb`. Reimplemented (not `super().forward()`) because
    # the per-layer dispatch lives inside the layer loop.
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        impl = self.config._attn_implementation or ""
        if (
            getattr(self.config, "reuse_packing_metadata", True)
            and ("flash" in impl or impl == "olmo3_sink_fa3")
            and kwargs.get("cu_seq_lens_q") is None  # don't clobber model-provided varlen kwargs
            and _is_packed_sequence(position_ids, inputs_embeds.shape[0])
        ):
            (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(position_ids)
            # Compute the varlen metadata once here and thread it to every layer.
            # int() so per-layer attention avoids a device->host sync on max_seqlen.
            kwargs["cu_seq_lens_q"] = cu_q
            kwargs["cu_seq_lens_k"] = cu_k
            kwargs["max_length_q"] = int(max_q)
            kwargs["max_length_k"] = int(max_k)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings_mapping = {
            "sliding_attention": self.rotary_embs["sliding_attention"](hidden_states, position_ids),
            "full_attention": self.rotary_embs["full_attention"](hidden_states, position_ids),
        }

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings_mapping[self.config.layer_types[i]],
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Olmo3SinkForCausalLM(Olmo3SinkPreTrainedModel, Olmo3ForCausalLM):
    def __init__(self, config: Olmo3SinkConfig):
        super().__init__(config)
        self.model = Olmo3SinkModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


# Auto-register the in-kernel FA3 sink attention so `attn_implementation="olmo3_sink_fa3"`
# works on import (incl. trust_remote_code). Guarded: if the patched FA3 isn't installed,
# import still succeeds and other backends (eager) remain usable.
try:
    from .attention import register_fa3_sink_attention

    register_fa3_sink_attention()
except Exception:  # noqa: BLE001
    pass


__all__ = [
    "Olmo3SinkAttention",
    "Olmo3SinkDecoderLayer",
    "Olmo3SinkPreTrainedModel",
    "Olmo3SinkRotaryEmbedding",
    "Olmo3SinkModel",
    "Olmo3SinkForCausalLM",
]
