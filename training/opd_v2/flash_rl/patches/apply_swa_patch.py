#!/usr/bin/env python3
"""Enable the hybrid SWA KV pool for Olmo3Sink in sglang (KV capacity +1.6~3x -> higher concurrency/throughput).

stock sglang `is_hybrid_swa_model` is a hardcoded arch allowlist that doesn't include Olmo3SinkForCausalLM
-> the SWA pool is off and KV is allocated full across all 32 layers (wasteful). This patch makes two
anchored edits:
 1) add Olmo3SinkForCausalLM / Olmo2ForCausalLM to the hybrid_swa_archs set
 2) extend the GptOss branch of get_hybrid_layer_ids (which reads config.layer_types) to also cover Olmo (the deploy config already has layer_types)
The model side already applies the sliding window per layer_types (olmo2_sink.py); only sglang's KV pool allocator was missing it.

Usage (regenerate when switching image/version):
  apptainer exec <sif> cat /sgl-workspace/sglang/python/sglang/srt/configs/model_config.py > stock.py
  python apply_swa_patch.py stock.py model_config.py
"""
import sys
src = open(sys.argv[1]).read(); orig = src
a1 = '    hybrid_swa_archs = {\n        "Llama4ForConditionalGeneration",'
assert src.count(a1) == 1, f"anchor1 count={src.count(a1)} (sglang layout changed?)"
src = src.replace(a1, '    hybrid_swa_archs = {\n        "Olmo3SinkForCausalLM",\n        "Olmo2ForCausalLM",\n        "Llama4ForConditionalGeneration",')
a2 = '    elif "GptOssForCausalLM" in model_architectures:\n        layer_types = getattr(hf_text_config, "layer_types", [])'
assert src.count(a2) == 1, f"anchor2 count={src.count(a2)} (sglang layout changed?)"
src = src.replace(a2, '    elif "GptOssForCausalLM" in model_architectures or any(\n        a in ("Olmo3SinkForCausalLM", "Olmo2ForCausalLM") for a in model_architectures\n    ):\n        layer_types = getattr(hf_text_config, "layer_types", [])')
assert src != orig
compile(src, "model_config.py", "exec")
open(sys.argv[2], "w").write(src)
print("patched OK ->", sys.argv[2])
