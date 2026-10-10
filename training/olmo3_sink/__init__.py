# Copyright 2026 proof-pilot. Apache-2.0.
"""Olmo3Sink: Olmo3 + attention sinks + packing-metadata reuse.

Importing this package has no side effects. For local/dev use call
`register_olmo3_sink()` once to wire the classes into the Auto* factories
(approach A). For portable delivery (approach B), use `convert.py` to bake the
code into a checkpoint via `trust_remote_code` / `auto_map`.
"""

from .configuration_olmo3_sink import Olmo3SinkConfig
from .modeling_olmo3_sink import (
    Olmo3SinkForCausalLM,
    Olmo3SinkModel,
    Olmo3SinkPreTrainedModel,
)
from .register import register_olmo3_sink

__all__ = [
    "Olmo3SinkConfig",
    "Olmo3SinkForCausalLM",
    "Olmo3SinkModel",
    "Olmo3SinkPreTrainedModel",
    "register_olmo3_sink",
]

# convenience re-export of the standalone in-kernel sink op (needs patched FA3)
try:
    from .fa3_sink_kernel import fa3_varlen_attn_with_sink_kernel  # noqa: F401

    __all__.append("fa3_varlen_attn_with_sink_kernel")
except Exception:  # noqa: BLE001
    pass

# optional Liger-Kernel acceleration (rope/rmsnorm/swiglu/fused-linear-CE)
try:
    from .liger import apply_liger  # noqa: F401

    __all__.append("apply_liger")
except Exception:  # noqa: BLE001
    pass
