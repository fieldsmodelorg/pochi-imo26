#!/usr/bin/env bash
# OPD v2 TEACHER /score server for an OLMO3-SINK teacher (self-distill test), in-container.
#
# Unlike run_teacher.sh (DeepSeek-V4-Flash: MLA + MoE + fp8 wo_a), the Olmo3 teacher is the SAME sink
# model as the rollout/student, so it needs the TRITON attention backend to apply the attention sink
# (like the rollout), served in bf16 for hidden fidelity. The hidden-extract PLUMBING (spool + /score)
# is model-agnostic and already baked (_patch_sglang_514.py patches scheduler/batch_result_processor/
# output_streamer/http_server). The DeepSeek-specific `patch_dsv4` is NOT used here (Olmo3 returns the
# plain post-norm last hidden via --enable-return-hidden-states), so SGLANG_DSV4_* / fp8-wo_a / MoE are
# all omitted. The trainer's build_w_rot reads THIS model's head; set OPD_HID_DIM to its hidden size
# (Olmo3-32B = 5120) and TEACHER_PATH to this model.
#
#   CUDA_VISIBLE_DEVICES=0 MODEL=<olmo3-deploy> ./run_teacher_olmo3.sh --tp 1 --port 8100
set -euo pipefail

# --- serve venv is CUDA-13 (sglang 0.5.14): use the CUDA-13 nvcc/CCCL for its runtime JIT kernels -----
# The base image defaults CUDA_HOME to cu128, but sglang JIT-compiles kernels that use CUDA-13 cuda::ptx
# intrinsics -> point it at the CUDA-13 toolkit baked in the image (the base/trainer venv stays cu128).
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:${PATH}

# --- CUDA-13 forward-compat (see Dockerfile cuda13compat stage) --------------------------------------
# sglang 0.5.14 is a CUDA-13 build and can't be pinned to cu128 (sgl-project/sglang#25069). If this
# node's driver predates CUDA 13, load the baked forward-compat libcuda so the serve stack runs on
# CUDA-12.x datacenter drivers (>=525). On a CUDA-13 driver this is skipped (native libcuda is used).
if [ -d /opt/cuda13-compat ]; then
  _cc=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]\{1,\}\).*/\1/p' | head -1)
  if [ -n "${_cc}" ] && [ "${_cc}" -lt 13 ] 2>/dev/null; then
    export LD_LIBRARY_PATH="/opt/cuda13-compat:${LD_LIBRARY_PATH:-}"
    echo "[cuda13-compat] node driver CUDA ${_cc}.x < 13 -> forward-compat libcuda enabled" >&2
  fi
fi

# --- persistent JIT compile cache (opt-in) — see run_teacher.sh for the full note ---------------------
# Set JIT_CACHE_DIR to a durable shared-FS path (Beaker Weka) so DeepGEMM/flashinfer/sglang/tvm-ffi
# compile caches persist across runs + instances. Unset -> default ~/.cache (unchanged). Arch+role scoped.
if [ -n "${JIT_CACHE_DIR:-}" ]; then
  _ccap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ') || true
  _jc="${JIT_CACHE_DIR%/}/sm${_ccap:-x}/teacher_olmo3"; mkdir -p "$HOME/.cache"
  for _d in deep_gemm flashinfer sglang tvm-ffi triton; do
    mkdir -p "$_jc/$_d"
    if [ -e "$HOME/.cache/$_d" ] && [ ! -L "$HOME/.cache/$_d" ]; then
      cp -an "$HOME/.cache/$_d/." "$_jc/$_d/" 2>/dev/null || true; rm -rf "$HOME/.cache/$_d"
    fi
    ln -sfn "$_jc/$_d" "$HOME/.cache/$_d"
  done
  mkdir -p "$_jc/inductor"
  export DG_JIT_CACHE_DIR="$HOME/.cache/deep_gemm" TRITON_CACHE_DIR="$HOME/.cache/triton" \
         TVM_FFI_CACHE_DIR="$HOME/.cache/tvm-ffi" TORCHINDUCTOR_CACHE_DIR="$_jc/inductor"
  echo "[jit-cache] persistent compile cache (sm${_ccap:-x}/teacher_olmo3) -> $_jc" >&2
fi
SERVE_PY="${SERVE_PY:-/opt/venv/serve/bin/python}"
OPD_REPO="${OPD_REPO:-/opt/opd/repo}"
MODEL="${MODEL:?set MODEL=<olmo3-sink deploy dir> (self-distill: same as the student)}"
TP="${TP:-1}"; PORT=8100
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"     # sink-correct on H200/B200 (not flashinfer)
while [ $# -gt 0 ]; do case "$1" in --tp) TP="$2"; shift 2;; --port) PORT="$2"; shift 2;; *) shift;; esac; done

SPOOL="${SPOOL:-/dev/shm/opd-v2-teacher-spool}"; mkdir -p "$SPOOL"
DIST_ARGS=()
[ -n "${DIST_INIT_ADDR:-}" ]  && DIST_ARGS+=(--dist-init-addr "$DIST_INIT_ADDR")
[ -n "${SGLANG_NCCL_PORT:-}" ] && DIST_ARGS+=(--nccl-port "$SGLANG_NCCL_PORT")
[ "${TP:-1}" -gt 1 ] && DIST_ARGS+=(--disable-custom-all-reduce)

# hidden-extract plumbing env (same as run_teacher.sh) — MINUS the DeepSeek-only knobs.
export MALLOC_ARENA_MAX=4 \
       SGLANG_HIDDEN_SPOOL_DIR="$SPOOL" \
       SGLANG_HIDDEN_CODEC_DIR="$OPD_REPO/training/_common" \
       OPD_V2_SRC="$OPD_REPO/training/opd_v2/src" \
       OPD_TEACHER_MODEL_PATH="$MODEL" \
       OPD_SCORE_TOP1_CHUNK="${OPD_SCORE_TOP1_CHUNK:-1024}" \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$SERVE_PY" -m sglang.launch_server --skip-server-warmup \
  --model-path "$MODEL" --tp-size "$TP" --host 0.0.0.0 --port "$PORT" \
  "${DIST_ARGS[@]}" \
  --attention-backend "$ATTENTION_BACKEND" \
  --enable-return-hidden-states --disable-radix-cache --disable-cuda-graph \
  --skip-tokenizer-init \
  --chunked-prefill-size "${CHUNKED_PREFILL:-11264}" --mem-fraction-static "${MEMFRAC:-0.80}" \
  --max-running-requests "${MAXRUN:-64}" \
  --context-length "${TEACHER_CONTEXT_LEN:-${CONTEXT_LEN:-45056}}" \
  --watchdog-timeout 1800 --log-level info
