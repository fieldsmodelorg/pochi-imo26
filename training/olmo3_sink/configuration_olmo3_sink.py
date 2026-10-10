# Copyright 2026 proof-pilot. Apache-2.0.
"""Config for Olmo3 + attention-sink + packing-metadata-reuse.

This is a thin subclass of `Olmo3Config` (transformers 5.x, dataclass/`@strict`
style). It only adds two fields; all structural defaults come from Olmo3.
"""

from __future__ import annotations

from huggingface_hub.dataclasses import strict

from transformers.models.olmo3.configuration_olmo3 import Olmo3Config


@strict
class Olmo3SinkConfig(Olmo3Config):
    r"""Olmo3 with per-head learnable attention sinks (gpt-oss style).

    Args:
        sink_init_value (`float`, defaults to 0.0):
            Initial value for every `self_attn.sinks` entry. The sink is an extra
            logit appended to the attention softmax (mass that attends to
            "nothing"). With value `0.0` the sink competes on equal footing with a
            zero-score key at init (gpt-oss default). For a *warm start* from a
            checkpoint trained without sinks, use a strongly negative value
            (e.g. `-10.0`) so the sink starts as a near no-op and is learned in.
        reuse_packing_metadata (`bool`, defaults to True):
            When packing multiple documents per sequence and running on a
            FlashAttention backend, compute the varlen `cu_seqlens`/`max_seqlen`
            once in the model forward and thread it to every layer, instead of
            letting each layer re-derive it from `position_ids` (which incurs a
            device->host sync per layer). No effect on non-flash backends or when
            `position_ids` is not packed.
    """

    model_type = "olmo3_sink"

    sink_init_value: float = 0.0
    reuse_packing_metadata: bool = True


__all__ = ["Olmo3SinkConfig"]
