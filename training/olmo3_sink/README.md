# olmo3_sink

Custom OLMo 3 model used as the base for the whole proof-pilot training pipeline
(SFT → soft-distill → OPD). It is a thin subclass of `transformers` OLMo 3 that adds:

1. **Learnable attention sinks** (gpt-oss style) for stable long-context / packed attention.
2. **FlashAttention packing-metadata reuse** — `cu_seqlens` is computed once in `model.forward`
   and threaded down, so packed variable-length training is cheap.
3. A **patched FA3 backend** with the sink applied *in-kernel* (free on the forward pass);
   falls back to a sink-aware eager backend when FA3 is not built.

## Layout
| file | purpose |
|---|---|
| `modeling_olmo3_sink.py` | the model (sink + FA3 packing reuse + per-layer-type RoPE) |
| `configuration_olmo3_sink.py` | `Olmo3SinkConfig` |
| `attention.py` | attention module wiring the sink |
| `fa3_sink_kernel.py`, `fa3_sink.py`, `fa3_attention_sink.patch` | patched FA3 in-kernel sink backend + source patch |
| `liger.py` | Liger kernel integration (RoPE / RMSNorm / SwiGLU / fused-linear-CE) |
| `sft_data.py` | length-packing SFT collator |
| `register.py` | in-process `AutoModel` registration for `model_type="olmo3_sink"` |
| `convert.py` | bake the modeling code into a `trust_remote_code` checkpoint |
| `convert_to_gqa.py` | MHA → GQA conversion (mean-pool + uptrain) |
| `sink_calib.py`, `build_init_model.py` | sink-init calibration / build the initial model |
| `train_sft.py` | stand-alone SFT entrypoint (single-GPU + multi-GPU FSDP paths) |

## Usage
```bash
# Register the model class in-process, then load as usual:
python -c "from olmo3_sink.register import register_olmo3_sink; register_olmo3_sink()"

# SFT entrypoint (base model path via env var):
OLMO3_SINK_MODEL=/models/Olmo-3-7B python -m olmo3_sink.train_sft ...
```

## Notes
- The FA3 in-kernel sink backend must be built from patched sources (see `fa3_attention_sink.patch`);
  without it the model still imports and runs on the eager backend.
- The sink weight (`self_attn.sinks`, `[num_heads]`, bf16) is loader-compatible with gpt-oss / vLLM / SGLang.
