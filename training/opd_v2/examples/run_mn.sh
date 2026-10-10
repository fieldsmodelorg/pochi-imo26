#!/usr/bin/env bash
# OPD v2 multi-node e2e (single slurm job, V24). Inside one allocation, lay out four
# processes with srun --overlap:
#   teacher sglang(TP4 x K/node) | rollout sglang(fp8 TP1 x K/node) | trainer-as-service
#   (single cross-node srun, torchrun world=8 x NTR) | orchestrator (head node, CPU process).
#
# Topology (env-overridable): the allocation's nodes are split into teacher / rollout /
# trainer segments.
#   TEACHER_NNODES (default 1) + ROLLOUT_NNODES (default 1) + the rest = trainer.
#   TEACHERS_PER_NODE (default 2 x TP4), ROLLOUTS_PER_NODE (default 8 x TP1 fp8).
# Medium: hidden states + weights both live on the shared FS (under run_dir) -> rollout/
# trainer read them directly across nodes (P7 fix).
#
# Launch order: teacher -> rollout -> health gate (teacher cold start + JIT can be slow)
# -> make_config -> trainer -> orchestrator. An EXIT trap kills all server sruns.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_V2="$(cd "$HERE/.." && pwd)"
REPO="${PP_ROOT:-$(cd "$HERE/../../.." && pwd)}"
SRC="$OPD_V2/src"
OPD_SRC="$REPO/training/_vendor_opd"
VENV="${PP_PYTHON:-$REPO/.venv/bin/python}"

HOLDER=${SLURM_JOB_ID:?must run inside a slurm allocation}
TAG=${TAG:-opd_v2_$HOLDER}
RUN_NAME=${RUN_NAME:-$TAG}
RUN_DIR=${RUN_DIR:-$OPD_V2/runs/$RUN_NAME}        # must be on a shared FS
mkdir -p "$RUN_DIR"
MAIN="$RUN_DIR/launch.log"
echo ">>> OPD v2 mn launch tag=$TAG run_dir=$RUN_DIR $(date)" | tee "$MAIN"

# Remove the trainer self-registration file left over from a previous round/requeue:
# otherwise the orchestrator may read a stale URL (dead old port) before this round's
# trainer overwrites it -> it latches a dead endpoint and the scheduler never starts
# (observed in production). checkpoints/latest.json (used for resume) is left alone.
rm -f "$RUN_DIR/trainer_endpoint.json"

# ---- node topology ----
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NN=${#NODES[@]}
TEACHER_NNODES=${TEACHER_NNODES:-1}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
TRAINER_NNODES=$(( NN - TEACHER_NNODES - ROLLOUT_NNODES ))
if [ "$TRAINER_NNODES" -lt 1 ]; then echo "need >= TEACHER+ROLLOUT+1 nodes (have $NN)" | tee -a "$MAIN"; exit 1; fi
TEACHER_NODES=("${NODES[@]:0:TEACHER_NNODES}")
ROLLOUT_NODES=("${NODES[@]:TEACHER_NNODES:ROLLOUT_NNODES}")
TRAINER_NODES_ARR=("${NODES[@]:TEACHER_NNODES+ROLLOUT_NNODES:TRAINER_NNODES}")
TRAINER_NODES=$(IFS=,; echo "${TRAINER_NODES_ARR[*]}")
HEAD="${TRAINER_NODES_ARR[0]}"                      # orchestrator + rdzv head

TEACHER_TP=${TEACHER_TP:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
TEACHERS_PER_NODE=${TEACHERS_PER_NODE:-2}           # 2 x TP4 = 8 GPU/node
ROLLOUTS_PER_NODE=${ROLLOUTS_PER_NODE:-8}           # 8 x TP1 fp8 = 8 GPU/node
# PORT_SHIFT cap: the highest port (T_NCCL0 base 38600) + shift must stay <= 65535:
#   %800*32 -> max shift 25568 -> 38600+25568 = 64168 < 65535 (%1000*32=31968 would
#   overflow the teacher nccl port).
PORT_SHIFT=$(( (HOLDER % 800) * 32 ))
T_PORT0=$(( 8100 + PORT_SHIFT )); T_DIST0=$(( 38100 + PORT_SHIFT )); T_NCCL0=$(( 38600 + PORT_SHIFT ))
R_PORT0=$(( 8200 + PORT_SHIFT ))
TRAINER_RDZV_PORT=$(( 29500 + (PORT_SHIFT % 1000) ))
TRAINER_HTTP_PORT=${TRAINER_HTTP_PORT:-$(( 8300 + (PORT_SHIFT % 100) ))}

# Cluster-specific NCCL fabric tuning: set NCCL_IB_HCA / NCCL_SOCKET_IFNAME for your
# cluster (e.g. NCCL_IB_HCA=mlx5_0,... NCCL_SOCKET_IFNAME=eth0) before launching.
NCCL_ENV="NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
[ -n "${NCCL_IB_HCA:-}" ] && NCCL_ENV="NCCL_IB_HCA=$NCCL_IB_HCA $NCCL_ENV"
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && NCCL_ENV="NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME $NCCL_ENV"

echo ">>> nodes=$NN teacher=[${TEACHER_NODES[*]}]x$TEACHERS_PER_NODE(TP$TEACHER_TP) rollout=[${ROLLOUT_NODES[*]}]x$ROLLOUTS_PER_NODE(TP$ROLLOUT_TP fp8) trainer=[$TRAINER_NODES](world=$((8*TRAINER_NNODES))) head=$HEAD" | tee -a "$MAIN"

SERVER_PIDS=()
cleanup() {
  echo ">>> cleanup tag=$TAG: kill server/trainer srun $(date)" | tee -a "$MAIN"
  for p in "${SERVER_PIDS[@]}"; do kill "$p" 2>/dev/null; done
  for n in "${TEACHER_NODES[@]}" "${ROLLOUT_NODES[@]}" "${TRAINER_NODES_ARR[@]}"; do
    srun --jobid="$HOLDER" --overlap --nodelist="$n" --nodes=1 --ntasks=1 \
      bash -c 'pkill -9 -f sglang.launch_server; pkill -9 -f opd_v2.trainer.service' 2>/dev/null &
  done
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

# ---- 1) teacher servers ----
TURLS=""
for n in "${TEACHER_NODES[@]}"; do
  srun --jobid="$HOLDER" --overlap --nodelist="$n" --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=96 \
    bash -c '
      for i in $(seq 0 '"$((TEACHERS_PER_NODE-1))"'); do
        gpus=$(seq -s, $((i*'"$TEACHER_TP"')) $((i*'"$TEACHER_TP"'+'"$TEACHER_TP"'-1)))
        port=$(('"$T_PORT0"'+i)); dist=$(('"$T_DIST0"'+i)); nccl=$(('"$T_NCCL0"'+i))
        CUDA_VISIBLE_DEVICES=$gpus SPOOL=/dev/shm/opd-v2-tea-$port MALLOC_ARENA_MAX=4 \
          DIST_INIT_PORT=$dist SGLANG_NCCL_PORT=$nccl \
          MEMFRAC='"${TEACHER_MEMFRAC:-}"' MAXRUN='"${TEACHER_MAXRUN:-}"' \
          bash '"$OPD_V2"'/run_teacher_fs.sh --tp '"$TEACHER_TP"' --port $port \
          > '"$RUN_DIR"'/teacher_'"$n"'_$port.log 2>&1 &
      done; wait
    ' > "$RUN_DIR/teachersrun_$n.log" 2>&1 &
  SERVER_PIDS+=($!)
  for i in $(seq 0 $((TEACHERS_PER_NODE-1))); do TURLS+="http://$n:$((T_PORT0+i)),"; done
  echo ">>> teacher srun on $n pid=${SERVER_PIDS[-1]}" | tee -a "$MAIN"
done
TURLS=${TURLS%,}

# ---- 2) rollout servers (fp8 flash_rl) ----
RURLS=""
for n in "${ROLLOUT_NODES[@]}"; do
  srun --jobid="$HOLDER" --overlap --nodelist="$n" --nodes=1 --ntasks=1 --gres=gpu:8 --cpus-per-task=96 \
    bash -c '
      for i in $(seq 0 '"$((ROLLOUTS_PER_NODE-1))"'); do
        gpus=$(seq -s, $((i*'"$ROLLOUT_TP"')) $((i*'"$ROLLOUT_TP"'+'"$ROLLOUT_TP"'-1)))
        port=$(('"$R_PORT0"'+i))
        CUDA_VISIBLE_DEVICES=$gpus MALLOC_ARENA_MAX=4 \
        MODEL='"${ROLLOUT_MODEL:-}"' KV_CACHE_DTYPE='"${KV_CACHE_DTYPE:-}"' SWA_RATIO='"${SWA_RATIO:-}"' CONTEXT_LEN='"${CONTEXT_LEN:-}"' MEMFRAC='"${MEMFRAC:-}"' MAXRUN='"${ROLLOUT_MAXRUN:-}"' \
          bash '"$OPD_V2"'/flash_rl/run_rollout_fp8.sh --port $port --tp '"$ROLLOUT_TP"' \
          > '"$RUN_DIR"'/rollout_'"$n"'_$port.log 2>&1 &
      done; wait
    ' > "$RUN_DIR/rolloutsrun_$n.log" 2>&1 &
  SERVER_PIDS+=($!)
  for i in $(seq 0 $((ROLLOUTS_PER_NODE-1))); do RURLS+="http://$n:$((R_PORT0+i)),"; done
  echo ">>> rollout srun on $n pid=${SERVER_PIDS[-1]}" | tee -a "$MAIN"
done
RURLS=${RURLS%,}

# ---- 3) health gate (from head; teacher cold start + JIT can be ~20min) ----
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-1800}
echo ">>> health gate ($HEALTH_TIMEOUT s) ..." | tee -a "$MAIN"
srun --jobid="$HOLDER" --overlap --nodelist="$HEAD" --nodes=1 --ntasks=1 bash -c '
  urls=$(echo "'"$TURLS,$RURLS"'" | tr "," " ")
  t0=$(date +%s)
  while true; do
    bad=0
    for u in $urls; do curl -s -m 3 "$u/health" >/dev/null 2>&1 || bad=$((bad+1)); done
    [ "$bad" = 0 ] && { echo "all $(echo $urls|wc -w) servers healthy"; exit 0; }
    [ $(( $(date +%s) - t0 )) -gt '"$HEALTH_TIMEOUT"' ] && { echo "health gate TIMEOUT ($bad unhealthy)"; exit 1; }
    sleep 10
  done
' 2>&1 | tee -a "$MAIN"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "health gate failed; abort" | tee -a "$MAIN"; exit 1; }

# ---- 4) config.json (single source of truth, with resolved URLs) ----
RUN_DIR="$RUN_DIR" RUN_NAME="$RUN_NAME" ROLLOUT_URLS="$RURLS" TEACHER_URLS="$TURLS" \
  TRAINER_HTTP_PORT="$TRAINER_HTTP_PORT" \
  "$VENV" "$HERE/make_config.py" 2>&1 | tee -a "$MAIN"
[ -f "$RUN_DIR/config.json" ] || { echo "config.json not written; abort" | tee -a "$MAIN"; exit 1; }

# ---- 5) trainer: single cross-node srun, torchrun c10d (NOT dual-srun; v1 IB-hang lesson) ----
echo ">>> launch trainer (world=$((8*TRAINER_NNODES)) on $TRAINER_NODES)" | tee -a "$MAIN"
srun --jobid="$HOLDER" --overlap --nodelist="$TRAINER_NODES" --nodes="$TRAINER_NNODES" --ntasks-per-node=1 \
     --gres=gpu:8 --cpus-per-task=96 \
  bash -c '
    export PYTHONPATH='"$SRC:$OPD_SRC"' '"$NCCL_ENV"' OPD_RUN_DIR='"$RUN_DIR"'
    export TRITON_CACHE_DIR=/tmp/triton_opdv2_$SLURM_NODEID
    '"$VENV"' -m torch.distributed.run \
      --nnodes='"$TRAINER_NNODES"' --nproc_per_node=8 \
      --rdzv-backend=c10d --rdzv-endpoint='"$HEAD"':'"$TRAINER_RDZV_PORT"' --rdzv-id='"$HOLDER"' \
      -m opd_v2.trainer.service --run-dir '"$RUN_DIR"' \
      > '"$RUN_DIR"'/trainer_$SLURM_NODEID.log 2>&1
  ' > "$RUN_DIR/trainersrun.log" 2>&1 &
TRAINER_SRUN_PID=$!
echo ">>> trainer srun pid=$TRAINER_SRUN_PID" | tee -a "$MAIN"

# ---- 6) orchestrator (head node, CPU process; reads config + discovers trainer endpoint) ----
MAX_STEPS=${MAX_STEPS:-100000}
echo ">>> launch orchestrator on $HEAD (max_steps=$MAX_STEPS)" | tee -a "$MAIN"
srun --jobid="$HOLDER" --overlap --nodelist="$HEAD" --nodes=1 --ntasks=1 --cpus-per-task=16 \
  bash -c '
    export PYTHONPATH='"$SRC:$OPD_SRC"' OPD_RUN_DIR='"$RUN_DIR"' CUDA_VISIBLE_DEVICES=
    '"$VENV"' -m opd_v2.orchestrator --run-dir '"$RUN_DIR"' --max-steps '"$MAX_STEPS"'
  ' 2>&1 | tee -a "$RUN_DIR/orchestrator.log" | tee -a "$MAIN"
ORCH_RC=${PIPESTATUS[0]}
echo ">>> orchestrator exited rc=$ORCH_RC $(date)" | tee -a "$MAIN"
exit "$ORCH_RC"
