# Copyright 2026 proof-pilot. Apache-2.0.
"""Approach A: in-process dynamic registration.

Call `register_olmo3_sink()` once at the top of a training / eval script so that
`AutoConfig`/`AutoModel*` recognize `model_type="olmo3_sink"`. This only affects
the current Python process; it does NOT make a checkpoint portable on its own
(for that, see `convert.py` / trust_remote_code).
"""

from __future__ import annotations

import warnings

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)

from .configuration_olmo3_sink import Olmo3SinkConfig
from .modeling_olmo3_sink import Olmo3SinkForCausalLM, Olmo3SinkModel

_REGISTERED = False


def register_olmo3_sink(exist_ok: bool = True) -> None:
    """Register Olmo3Sink config/model + the `olmo3_sink_fa3` attention impl."""
    global _REGISTERED
    if _REGISTERED:
        return
    AutoConfig.register("olmo3_sink", Olmo3SinkConfig, exist_ok=exist_ok)
    AutoModel.register(Olmo3SinkConfig, Olmo3SinkModel, exist_ok=exist_ok)
    AutoModelForCausalLM.register(Olmo3SinkConfig, Olmo3SinkForCausalLM, exist_ok=exist_ok)
    # The FA3 adapter (`attention.py`) imports `flash_attn_interface` (patched FA3 build)
    # at module scope. Import it lazily HERE so that `import olmo3_sink` + class
    # registration work on machines without FA3 (eager backend, as documented in the
    # README). Only a missing flash_attn_interface is tolerated -- anything else re-raises.
    # Requesting attn_implementation="olmo3_sink_fa3" on such a machine still fails loudly
    # inside transformers (the impl was never registered).
    try:
        from .attention import register_fa3_sink_attention
    except ModuleNotFoundError as e:
        if e.name != "flash_attn_interface":
            raise
        warnings.warn(
            "olmo3_sink: patched FA3 (flash_attn_interface) not available -- "
            "'olmo3_sink_fa3' attention backend NOT registered; eager backend only.",
            stacklevel=2,
        )
    else:
        register_fa3_sink_attention()
    # FA2 sink (B200/no-FA3): post-correction re-normalization on stock FA2 (all-arch). Registered
    # alongside FA3; requesting attn_implementation="olmo3_sink_fa2" needs the `flash_attn` (FA2) pkg.
    # NB: transformers' own `flash_attention_2` SILENTLY DROPS the sink (stock FA2 has no sink arg;
    # only FA3 `s_aux` / FA4 `learnable_sink` are honored) -> use this backend on Blackwell.
    try:
        from .olmo3_sink_fa2 import register_olmo3_sink_fa2
    except ModuleNotFoundError as e:
        if e.name != "flash_attn":
            raise
        warnings.warn(
            "olmo3_sink: flash_attn (FA2) not available -- 'olmo3_sink_fa2' backend NOT registered.",
            stacklevel=2,
        )
    else:
        register_olmo3_sink_fa2()
    _REGISTERED = True
