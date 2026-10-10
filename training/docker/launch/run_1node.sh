#!/usr/bin/env bash
# OPD v2 single-node (8-GPU) integration launcher — the final smoke before the 64x B200 run.
#
# Same 4-process loop and launch ORDER as run_mn.sh (teacher -> rollout -> health gate ->
# make_config -> trainer -> orchestrator), but co-located on ONE node via CUDA_VISIBLE_DEVICES
# splits + background processes instead of slurm srun. Default 8-GPU layout (4:2:2):
#   teacher DeepSeek-V4-Flash TP4  -> GPU 0-3
#   rollout student (fp8, triton)  -> GPU 4-5  (TP2)
#   trainer (FSDP2, CPU offload)   -> GPU 6-7  (torchrun world=2)
#   orchestrator                   -> CPU
# Everything is env-overridable; source an env preset first, e.g.:
#   source /opt/opd/launch/env_1node_smoke.sh && bash /opt/opd/launch/run_1node.sh
set -uo pipefail
OPD_REPO="${OPD_REPO:-/opt/opd/repo}"
OPD_V2="$OPD_REPO/training/opd_v2"
ROLE_DIR="${ROLE_DIR:-/opt/opd/opd_serve}"
SRC="$OPD_V2/src"; OPD_SRC="$OPD_REPO/training/_vendor_opd"
TRAIN_PY="${OPD_TRAIN_PY:-python}"
export SERVE_PY="${SERVE_PY:-/opt/venv/serve/bin/python}"

RUN_NAME=${RUN_NAME:-opd_1node}
RUN_DIR=${RUN_DIR:?set RUN_DIR to a shared/scratch dir (holds hidden spool, weights, config, pool)}
mkdir -p "$RUN_DIR"; MAIN="$RUN_DIR/launch.log"
echo ">>> OPD v2 1-node launch run_dir=$RUN_DIR $(date)" | tee "$MAIN"
rm -f "$RUN_DIR/trainer_endpoint.json"

# ---- 8-GPU layout (4:2:2 = teacher : rollout : trainer) ----
TEACHER_GPUS=${TEACHER_GPUS:-0,1,2,3}; TEACHER_TP=${TEACHER_TP:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4,5};     ROLLOUT_TP=${ROLLOUT_TP:-2}
TRAINER_GPUS=${TRAINER_GPUS:-6,7};     TRAINER_NPROC=${TRAINER_NPROC:-2}
T_PORT=${T_PORT:-8100}; R_PORT=${R_PORT:-8200}
TRAINER_HTTP_PORT=${TRAINER_HTTP_PORT:-8300}; RDZV_PORT=${RDZV_PORT:-29500}

PIDS=()
cleanup() {
  echo ">>> cleanup $(date)" | tee -a "$MAIN"
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  pkill -9 -f sglang.launch_server 2>/dev/null
  pkill -9 -f opd_v2.trainer.service 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# ---- 1) teacher ----
# TEACHER_SCRIPT: run_teacher.sh (DeepSeek-V4-Flash, default) | run_teacher_olmo3.sh (Olmo3 self/x: triton sink).
# TEACHER_MODEL: the teacher checkpoint (DeepSeek default; for Olmo3 set to the teacher Olmo3-32B ckpt).
TEACHER_SCRIPT="${TEACHER_SCRIPT:-run_teacher.sh}"
echo ">>> teacher [$TEACHER_SCRIPT] on GPU $TEACHER_GPUS (TP$TEACHER_TP) :$T_PORT" | tee -a "$MAIN"
CUDA_VISIBLE_DEVICES=$TEACHER_GPUS SPOOL=/dev/shm/opd1node-tea MALLOC_ARENA_MAX=4 \
  SGLANG_NCCL_PORT=$((T_PORT+400)) \
  MODEL="${TEACHER_MODEL:-${DEEPSEEK_V4_FLASH:-}}" \
  MEMFRAC="${TEACHER_MEMFRAC:-}" MAXRUN="${TEACHER_MAXRUN:-}" \
  bash "$ROLE_DIR/$TEACHER_SCRIPT" --tp "$TEACHER_TP" --port "$T_PORT" > "$RUN_DIR/teacher.log" 2>&1 &
PIDS+=($!); TURLS="http://127.0.0.1:$T_PORT"

# ---- 2) rollout (student, fp8, triton sink) ----
echo ">>> rollout on GPU $ROLLOUT_GPUS (TP$ROLLOUT_TP) :$R_PORT" | tee -a "$MAIN"
CUDA_VISIBLE_DEVICES=$ROLLOUT_GPUS MALLOC_ARENA_MAX=4 \
  MODEL="${ROLLOUT_MODEL:?set ROLLOUT_MODEL (deploy-format student dir)}" \
  KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}" SWA_RATIO="${SWA_RATIO:-}" CONTEXT_LEN="${CONTEXT_LEN:-}" \
  MEMFRAC="${MEMFRAC:-}" MAXRUN="${ROLLOUT_MAXRUN:-}" \
  bash "$ROLE_DIR/run_rollout.sh" --tp "$ROLLOUT_TP" --port "$R_PORT" > "$RUN_DIR/rollout.log" 2>&1 &
PIDS+=($!); RURLS="http://127.0.0.1:$R_PORT"

# ---- 3) health gate (teacher cold start + JIT can be ~20min) ----
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-1800}
echo ">>> health gate ($HEALTH_TIMEOUT s) on $TURLS + $RURLS ..." | tee -a "$MAIN"
t0=$(date +%s)
while true; do
  bad=0
  for u in "$TURLS" "$RURLS"; do curl -s -m 3 "$u/health" >/dev/null 2>&1 || bad=$((bad+1)); done
  [ "$bad" = 0 ] && { echo "  both servers healthy" | tee -a "$MAIN"; break; }
  # surface early server crashes instead of waiting out the whole timeout
  for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null || { echo "  a server process died — see $RUN_DIR/{teacher,rollout}.log" | tee -a "$MAIN"; exit 1; }; done
  [ $(( $(date +%s) - t0 )) -gt "$HEALTH_TIMEOUT" ] && { echo "  health gate TIMEOUT ($bad unhealthy)" | tee -a "$MAIN"; exit 1; }
  sleep 10
done

# ---- 4) config.json (+ cu128 ATTN_IMPL) ----
RUN_DIR="$RUN_DIR" RUN_NAME="$RUN_NAME" ROLLOUT_URLS="$RURLS" TEACHER_URLS="$TURLS" \
  TRAINER_HTTP_PORT="$TRAINER_HTTP_PORT" ATTN_IMPL="${ATTN_IMPL:-olmo3_sink_fa2}" \
  "$TRAIN_PY" "$OPD_V2/examples/make_config.py" 2>&1 | tee -a "$MAIN"
[ -f "$RUN_DIR/config.json" ] || { echo "config.json not written; abort" | tee -a "$MAIN"; exit 1; }

# ---- 5) trainer (FSDP2 torchrun, world=$TRAINER_NPROC on $TRAINER_GPUS) ----
echo ">>> trainer on GPU $TRAINER_GPUS (world $TRAINER_NPROC)" | tee -a "$MAIN"
PYTHONPATH="$SRC:$OPD_SRC" OPD_RUN_DIR="$RUN_DIR" CUDA_VISIBLE_DEVICES="$TRAINER_GPUS" \
  TRITON_CACHE_DIR=/tmp/triton_opd1node PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$TRAIN_PY" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="$TRAINER_NPROC" \
    --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$RDZV_PORT" --rdzv-id=opd1node \
    -m opd_v2.trainer.service --run-dir "$RUN_DIR" > "$RUN_DIR/trainer.log" 2>&1 &
PIDS+=($!)
echo ">>> trainer pid=${PIDS[-1]} (log: $RUN_DIR/trainer.log)" | tee -a "$MAIN"

# ---- 6) orchestrator (foreground, CPU) ----
MAX_STEPS=${MAX_STEPS:-50}
echo ">>> orchestrator (max_steps=$MAX_STEPS)" | tee -a "$MAIN"
PYTHONPATH="$SRC:$OPD_SRC" OPD_RUN_DIR="$RUN_DIR" CUDA_VISIBLE_DEVICES= \
  "$TRAIN_PY" -m opd_v2.orchestrator --run-dir "$RUN_DIR" --max-steps "$MAX_STEPS" \
  2>&1 | tee -a "$RUN_DIR/orchestrator.log" | tee -a "$MAIN"
ORCH_RC=${PIPESTATUS[0]}
echo ">>> orchestrator exited rc=$ORCH_RC $(date)" | tee -a "$MAIN"
exit "$ORCH_RC"
