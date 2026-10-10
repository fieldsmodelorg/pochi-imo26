#!/usr/bin/env bash
# OPD v2 teacher scoring service = native sglang server + hidden-extract patches,
# **but /score supports out_path -> writes shared FS server-side and returns a handle JSON** (P7 fix, V12).
#
# Differences vs v1 `training/opd/examples/run_teacher_service.sh`:
#   - http_server is swapped for opd_v2/teacher_patch/http_server.py (adds the out_path branch; the other 3
#     patches are reused from teacher_extract/_patched, same image version, verified consistent).
#   - additionally sets OPD_V2_SRC so /score inside the container can `from opd_v2.hidden_store import write_hidden`
#     (single source of truth for the hidden file format).
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_teacher_fs.sh --tp 4 --port 8100
set -euo pipefail
TP=4; PORT=8100
while [ $# -gt 0 ]; do case "$1" in --tp) TP="$2"; shift 2;; --port) PORT="$2"; shift 2;; *) echo "unknown arg: $1" >&2; exit 1;; esac; done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
SIF=${SIF:-/images/sglang.sif}
TE="$REPO/training/teacher_extract"
PDIR="$TE/_patched"                                  # deepseek_v4 / scheduler / output_processor
V2HTTP="$HERE/teacher_patch/http_server.py"          # opd_v2: out_path FS-write version
MODEL=${MODEL:-/models/DeepSeek-V4-Flash}
SPOOL=${SPOOL:-/dev/shm/opd-v2-teacher-spool}        # internal bf16 spool (node-local tmpfs; not the v2 shared hidden)

for f in deepseek_v4.py scheduler.py scheduler_output_processor_mixin.py; do
  [ -f "$PDIR/$f" ] || { echo "missing $PDIR/$f: regenerate by running REPATCH in teacher_extract first" >&2; exit 1; }
done
[ -f "$V2HTTP" ] || { echo "missing $V2HTTP: run the opd_v2 regen first (see README)" >&2; exit 1; }
mkdir -p "$SPOOL"
SGL=${SGLANG_PKG_DIR:-/sgl-workspace/sglang/python/sglang}

DIST_ARGS=()
[ -n "${DIST_INIT_ADDR:-}" ] && DIST_ARGS+=(--dist-init-addr "$DIST_INIT_ADDR")
[ -n "${SGLANG_NCCL_PORT:-}" ] && DIST_ARGS+=(--nccl-port "$SGLANG_NCCL_PORT")

exec apptainer exec --nv \
  --bind /work \
  --bind "$PDIR/deepseek_v4.py:$SGL/srt/models/deepseek_v4.py" \
  --bind "$PDIR/scheduler.py:$SGL/srt/managers/scheduler.py" \
  --bind "$PDIR/scheduler_output_processor_mixin.py:$SGL/srt/managers/scheduler_output_processor_mixin.py" \
  --bind "$V2HTTP:$SGL/srt/entrypoints/http_server.py" \
  --env SGLANG_DSV4_HIDDEN_POST_NORM=1 \
  --env SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 \
  --env MALLOC_ARENA_MAX=4 \
  --env SGLANG_HIDDEN_SPOOL_DIR="$SPOOL" \
  --env SGLANG_HIDDEN_CODEC_DIR="$REPO/training/_common" \
  --env OPD_V2_SRC="$HERE/src" \
  --env OPD_TEACHER_MODEL_PATH="$MODEL" \
  --env OPD_SCORE_TOP1_CHUNK="${OPD_SCORE_TOP1_CHUNK:-1024}" \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  "$SIF" python3 -m sglang.launch_server \
    --model-path "$MODEL" --tp-size "$TP" --host 0.0.0.0 --port "$PORT" \
    "${DIST_ARGS[@]}" \
    --enable-return-hidden-states --disable-radix-cache \
    --chunked-prefill-size "${CHUNKED_PREFILL:-11264}" --mem-fraction-static "${MEMFRAC:-0.80}" \
    --max-running-requests "${MAXRUN:-128}" --disable-cuda-graph \
    --context-length "${TEACHER_CONTEXT_LEN:-${CONTEXT_LEN:-69632}}" --moe-runner-backend marlin \
    --watchdog-timeout 1800 --log-level info
