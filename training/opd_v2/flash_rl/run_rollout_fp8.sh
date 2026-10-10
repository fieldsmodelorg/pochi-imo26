#!/usr/bin/env bash
# Launch an OPD student rollout server with FP8 quantization + working
# update_weights_from_disk (repeated bf16 reload + re-quantize to fp8).
#
# This is run_rollout_service.sh + two extra knobs:
#   --quantization fp8 --load-format flash_rl   (online fp8 + RL reload loader)
# plus a bind-mount of the patched flash_rl loader (see patches/ + README.md).
# Without the patch, `--quantization fp8` + update_weights_from_disk CRASHES
# (transpose assert); stock flash_rl from_disk OOMs after a few reloads.
#
#   CUDA_VISIBLE_DEVICES=4 ./run_rollout_fp8.sh --port 8200
#
# Tunables via env: SIF MODEL PORT TP MEMFRAC MAXRUN CUDA_GRAPH_MAX_BS CHUNKED_PREFILL CONTEXT_LEN
#                   KV_CACHE_DTYPE SWA_RATIO   (long-context KV memory savings; see below)
#   CUDA_GRAPH_MAX_BS defaults to MAXRUN: when conc(MAXRUN)>10 you **must** raise it too, otherwise the
#   cuda graph only captures up to the default bs and larger batches fall back to eager and slow down
#   (2026-06-20 KV-pool test: the graph only reached bs=10 = the default MAXRUN).
# Validated: sglang 0.5.12.post1, H200.
#   - TP=1 bf16-KV: original validation (10/10 reload bit-exact).
#   - TP=4 + fp8-KV(e4m3) + SWA-ratio: long-context measurement (2026-06-17) — all three flags can be
#     enabled together, update_weights_from_disk 6/6 success, reload->reload bit-exact, e4m3 keeps FA3. A TP8 head divides evenly and also works.
#   - TP4 + fp8 + e4m3 + SWA r=0.2 + ctx131072 + memfrac0.85 (measured 2026-06-20): hybrid SWA pool enabled,
#     full_layer_tokens=4,711,012 / swa_layer_tokens=942,202 (115GB, avail 20.6GB) -> long sequences bind the full pool:
#     conc*avg_len <= 4.71M (avg 64k -> ~72 per replica). OPD agentic sets MAXRUN=64 (see examples/run_agentic_mn_32b.sbatch).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

SIF=${SIF:-/images/sglang.sif}     # 0.5.12.post1
MODEL=${MODEL:-$ROOT/outputs/stage1-v2-7b-deploy}
PORT=8200; TP=1
while [ $# -gt 0 ]; do case "$1" in --port) PORT=$2; shift 2;; --tp) TP=$2; shift 2;; *) shift;; esac; done

SGL=${SGLANG_PKG_DIR:-/sgl-workspace/sglang/python/sglang}
SINK=$ROOT/deploy/target/olmo2_sink.py                  # in-engine sink target
LOADER=$HERE/patches/loader.py                          # patched flash_rl loader
LOADER_DST=$SGL/srt/model_loader/loader.py
MCFG=$HERE/patches/model_config.py                      # SWA-pool patch (enables the hybrid SWA KV pool for olmo3)
MCFG_DST=$SGL/srt/configs/model_config.py

# Regenerate the patched loader/model_config from THIS image if stale or you
# switched sglang versions (idempotent; see apply_patch.py / apply_swa_patch.py).
if [ "${REGEN_LOADER:-0}" = "1" ]; then
  apptainer exec "$SIF" cat "$LOADER_DST" > /tmp/_flashrl_stock.py
  python3 "$HERE/apply_patch.py" /tmp/_flashrl_stock.py "$LOADER"
  apptainer exec "$SIF" cat "$MCFG_DST" > /tmp/_mcfg_stock.py
  python3 "$HERE/patches/apply_swa_patch.py" /tmp/_mcfg_stock.py "$MCFG"
fi

PARSER_ARGS=(--reasoning-parser deepseek-r1 --tool-call-parser deepseekv4)
if [ "${SKIP_TOKENIZER_INIT:-0}" = "1" ]; then
  # OPD rollout is token-in/token-out; drop parsers (need a tokenizer/template).
  PARSER_ARGS=(--skip-tokenizer-init)
fi
EXTRA_ARGS=()
[ -n "${CONTEXT_LEN:-}" ] && EXTRA_ARGS+=(--context-length "$CONTEXT_LEN")
# long-context KV memory savings (measured 2026-06-17):
#  - fp8 KV cache: use fp8_e4m3 (FA3 supports it, 3-bit mantissa is less lossy); fp8_e5m2 forces the attn backend down to triton.
[ -n "${KV_CACHE_DTYPE:-}" ] && EXTRA_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
#  - hybrid SWA memory pool: ratio of SWA-layer KV tokens / full-layer KV tokens (olmo3 = 24 SWA : 8 full, window 4096).
[ -n "${SWA_RATIO:-}" ] && EXTRA_ARGS+=(--swa-full-tokens-ratio "$SWA_RATIO")
#  - TP>1 + multiple replicas/node: custom all-reduce uses CUDA IPC handles; two TP-groups on the same node
#    hit `custom_all_reduce.cuh: CUDA error: invalid argument` under cuda-graph capture (2026-06-20 job131796
#    real crash; a single replica can't reproduce it locally) -> fall back to NCCL all-reduce (7B TP4 decode
#    all-reduce is small, goes over NVLink, throughput impact negligible).
[ "${TP:-1}" -gt 1 ] && EXTRA_ARGS+=(--disable-custom-all-reduce)

# NOTE on memory: each reload needs transient scratch for the bf16->fp8
# re-quant. fp8 frees ~6GB of weights vs bf16; keep mem-fraction-static at a
# level that leaves headroom for the reload. 0.85 validated OK for 7B/TP1.
exec apptainer exec --nv \
  --bind /work \
  --bind "$SINK:$SGL/srt/models/olmo2.py" \
  --bind "$LOADER:$LOADER_DST" \
  --bind "$MCFG:$MCFG_DST" \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SIF" python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size "$TP" \
    --host 0.0.0.0 --port "$PORT" \
    --quantization fp8 \
    --load-format flash_rl \
    --mem-fraction-static "${MEMFRAC:-0.85}" \
    --max-running-requests "${MAXRUN:-10}" \
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-${MAXRUN:-10}}" \
    --chunked-prefill-size "${CHUNKED_PREFILL:-4096}" \
    --disable-radix-cache \
    "${EXTRA_ARGS[@]}" \
    "${PARSER_ARGS[@]}"
