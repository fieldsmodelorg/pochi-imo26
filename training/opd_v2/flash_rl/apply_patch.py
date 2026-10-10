#!/usr/bin/env python3
"""Apply the 3 flash_rl-from-disk fixes to a stock sglang loader.py.

The patched `patches/loader.py` in this dir was produced from the
sglang 0.5.12.post1 image. If you switch sglang versions, regenerate
instead of relying on the version-pinned full file:

    apptainer exec <img.sif> cat \
      /sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py > stock.py
    python apply_patch.py stock.py patches/loader.py

The 3 edits target QuantizedRLModelLoader (--load-format flash_rl) so that
`update_weights_from_disk` (disk -> re-quantize to fp8) works for repeated
reloads on a custom Olmo3Sink model. See README.md for the why.

Idempotent: re-running on an already-patched file is a no-op (it detects the
markers and refuses to double-apply).
"""
import sys

MARK = "proof-pilot patch"

# --- Edit 1: skip re-wrapping the load_weights proxy on reload -------------
E1_OLD = '''        logger.info("[QuantizedRL] Initial load with FP8 quantization")

        original_load_weights = model.load_weights'''
E1_NEW = '''        logger.info("[QuantizedRL] Initial load with FP8 quantization")

        # proof-pilot patch: on reload (update_weights_from_disk re-enters this
        # method) model.load_weights is ALREADY the reload proxy installed during
        # the initial load. Re-wrapping it nests a NEW proxy whose captured
        # original_load_weights is the PREVIOUS proxy -> the Nth reload recurses
        # N levels deep, each level re-materialising (list(weights)) and
        # re-quantizing every weight. That linear blowup is what OOMs after a few
        # reloads. On reload just invoke the single existing proxy (its
        # original_load_weights is the REAL load_weights, so rebinding bottoms out
        # with no recursion) and return -- this also skips the trailing
        # initial-load-only record + process_weights_after_loading, which would
        # otherwise re-quantize a non-contiguous view and trip is_contiguous().
        if getattr(model, "flash_rl_initial_load_complete", False):
            model.load_weights(weights)
            return

        original_load_weights = model.load_weights'''

# --- Edits 2 & 3: quantize only >=2-D weights, and move CPU->CUDA first -----
E2_OLD = '''                elif weight.dtype in [torch.bfloat16, torch.float32, torch.float16]:
                    qweight, scale = per_token_group_quant_fp8(weight, weight.shape[-1])'''
E2_NEW = '''                elif (
                    weight.dim() >= 2
                    and weight.dtype in [torch.bfloat16, torch.float32, torch.float16]
                ):
                    # proof-pilot patch (2 fixes vs stock flash_rl):
                    #  1) update_weights_from_disk yields CPU tensors, but
                    #     per_token_group_quant_fp8 is a CUDA-only fused kernel
                    #     -> move to the current device before quantizing.
                    #  2) stock SKIP_QUANTIZATION_PARAMS misses Olmo2/Olmo3 norms
                    #     (q_norm/k_norm/post_feedforward_layernorm), which are 1-D
                    #     and crash the 2-D quant kernel. Restrict quant to >=2-D
                    #     weights so any 1-D param falls through to the keep branch.
                    if not weight.is_cuda:
                        weight = weight.to(device=torch.cuda.current_device())
                    qweight, scale = per_token_group_quant_fp8(weight, weight.shape[-1])'''


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]
    text = open(src).read()
    if MARK in text:
        print(f"[apply_patch] {src} already contains '{MARK}' markers -> nothing to do")
        open(dst, "w").write(text)
        return
    for label, old, new in [("edit1(proxy)", E1_OLD, E1_NEW),
                            ("edit2+3(quant)", E2_OLD, E2_NEW)]:
        n = text.count(old)
        if n != 1:
            sys.exit(f"[apply_patch] FAILED: anchor for {label} matched {n} times "
                     f"(expected 1). The stock loader.py layout changed; update "
                     f"apply_patch.py for this sglang version.")
        text = text.replace(old, new)
    open(dst, "w").write(text)
    print(f"[apply_patch] wrote patched loader -> {dst} (3 edits applied)")


if __name__ == "__main__":
    main()
