#!/usr/bin/env bash
# OPD v2 multi-node e2e WITHOUT a scheduler — run this SAME script once on every node (e.g. via
# launch_ssh.sh, pdsh, or your own scheduler), inside the image. Port of run_mn_cu128.sh (her V24/V33 loop)
# for clusters with no slurm fan-out.
#
# Each node is told who it is:
#   NODE_RANK  0..NNODES-1 (rank ordering == the slurm node ordering in run_mn_cu128.sh)
#   NNODES     total node count (8 for her V33 topology)
#   LAUNCH_ID  any string, IDENTICAL on all nodes and UNIQUE per launch (launch_ssh.sh uses a timestamp)
#   RUN_DIR    a SHARED, WRITABLE FS path, identical on all nodes
# and runs ONLY the role its rank maps to:
#     rank 0 .. TEACHER_NNODES-1                       -> teacher node(s)
#     rank TEACHER_NNODES .. +ROLLOUT_NNODES-1         -> rollout node(s)
#     the rest                                          -> trainer node(s)
#     the FIRST trainer node (trainer-local-rank 0)     -> ALSO the rdzv head + orchestrator
#
# EVERYTHING ELSE IS COPIED VERBATIM FROM run_mn_cu128.sh — DO NOT re-tune any OPD knob here:
#   role scripts /opt/opd/opd_serve/run_{teacher,rollout}.sh, the PORT_SHIFT/port scheme, the per-role
#   env, the health gate, the launch ORDER (teacher+rollout -> health -> make_config -> trainer ->
#   orchestrator), make_config, opd_v2.trainer.service (c10d torchrun), the orchestrator invocation,
#   NCCL env, and the cleanup pkill patterns. Only node discovery/placement differ from slurm:
#
#   1. NODE LIST: `scontrol show hostnames` -> a shared-FS rank->hostname gather barrier: every node
#      publishes its routable hostname (NODE_HOST, default `hostname -f`) and all read them back in rank
#      order. The health gate is the empirical routability check (fails fast on a wrong hostname).
#   2. HOLDER: slurm's numeric $SLURM_JOB_ID (PORT_SHIFT + rdzv-id + tag) -> a numeric hash of LAUNCH_ID.
#   3. PLACEMENT: `srun --overlap --nodelist=$n <body>` -> `<body>` run locally on the node whose rank
#      maps to that role.
#   4. TRAINER: one cross-node `srun ... torchrun` -> each trainer node runs its OWN torchrun; they
#      c10d-rendezvous into one world exactly as before (rdzv-endpoint = the first trainer node).
#   5. TEARDOWN: the slurm EXIT trap's srun-fanned pkill -> per-launch sentinels in RUN_DIR:
#      the head writes STOP_<holder> when the orchestrator exits (graceful teardown everywhere); any node
#      whose child dies writes FAILED_<holder>_rank<r>, which stops every other node and the orchestrator.
#   Every sentinel/gather file is keyed by LAUNCH_ID, so re-launching into the same RUN_DIR (resume) never
#   reads a previous launch's hostnames, STOP, or config.json.
#
# Source an env preset first, exactly as with the slurm launcher, e.g.:
#   source /opt/opd/launch/env_v33.sh && NODE_RANK=3 NNODES=8 LAUNCH_ID=... bash /opt/opd/launch/run_mn_nodes.sh
set -uo pipefail
OPD_REPO="${OPD_REPO:-/opt/opd/repo}"
OPD_V2="$OPD_REPO/training/opd_v2"
ROLE_DIR="${ROLE_DIR:-/opt/opd/opd_serve}"          # baked de-apptainer'd role scripts
SRC="$OPD_V2/src"
OPD_SRC="$OPD_REPO/training/_vendor_opd"
TRAIN_PY="${OPD_TRAIN_PY:-python}"                   # image train python (base conda, transformers 5.9)
export SERVE_PY="${SERVE_PY:-/opt/venv/serve/bin/python}"   # role scripts default to this too

# ---- node identity (replaces slurm SLURM_JOB_ID / SLURM_JOB_NODELIST / SLURM_NODEID) ----
RANK=${NODE_RANK:?set NODE_RANK (0..NNODES-1) on each node}
RCOUNT=${NNODES:?set NNODES (total node count, same on all nodes)}
LAUNCH_ID=${LAUNCH_ID:?set LAUNCH_ID (identical on all nodes, unique per launch, e.g. \$(date +%s))}
HOLDER=${OPD_HOLDER:-$(( 16#$(printf '%s' "$LAUNCH_ID" | md5sum | cut -c1-7) ))}   # numeric; same on every node
TAG=${TAG:-opd_v2_$HOLDER}
RUN_NAME=${RUN_NAME:-$TAG}
RUN_DIR=${RUN_DIR:?set RUN_DIR to a SHARED, WRITABLE FS path visible at the same path on all nodes}
mkdir -p "$RUN_DIR"
MAIN="$RUN_DIR/launch_rank${RANK}.log"              # per-node log (slurm wrote one head log)
STOP="$RUN_DIR/STOP_$HOLDER"
FAILED_GLOB="$RUN_DIR/FAILED_${HOLDER}_rank*"
CONFIG_READY="$RUN_DIR/config_ready_$HOLDER"
echo ">>> OPD v2 node launch rank=$RANK/$RCOUNT launch_id=$LAUNCH_ID tag=$TAG run_dir=$RUN_DIR $(date)" | tee "$MAIN"

# ---- node topology: rebuild the ordered NODES[] that `scontrol show hostnames` gave on slurm ----
HOSTS_DIR="$RUN_DIR/.hosts_$HOLDER"
mkdir -p "$HOSTS_DIR"
MY_HOST="${NODE_HOST:-$(hostname -f 2>/dev/null || hostname)}"
printf '%s\n' "$MY_HOST" > "$HOSTS_DIR/rank_${RANK}.tmp" && mv "$HOSTS_DIR/rank_${RANK}.tmp" "$HOSTS_DIR/rank_${RANK}"
echo ">>> [rank $RANK] published host '$MY_HOST'; waiting for all $RCOUNT nodes ..." | tee -a "$MAIN"
GATHER_TIMEOUT=${GATHER_TIMEOUT:-600}
t0=$(date +%s)
while :; do
  present=$(find "$HOSTS_DIR" -maxdepth 1 -name 'rank_[0-9]*' ! -name '*.tmp' 2>/dev/null | wc -l)
  [ "$present" -ge "$RCOUNT" ] && break
  [ $(( $(date +%s) - t0 )) -gt "$GATHER_TIMEOUT" ] && { echo "host-gather TIMEOUT ($present/$RCOUNT after ${GATHER_TIMEOUT}s)" | tee -a "$MAIN"; exit 1; }
  sleep 2
done
NODES=()
for r in $(seq 0 $((RCOUNT-1))); do NODES+=("$(cat "$HOSTS_DIR/rank_$r")"); done
NN=$RCOUNT

TEACHER_NNODES=${TEACHER_NNODES:-1}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
TRAINER_NNODES=$(( NN - TEACHER_NNODES - ROLLOUT_NNODES ))
if [ "$TRAINER_NNODES" -lt 1 ]; then echo "need >= TEACHER+ROLLOUT+1 nodes (have $NN)" | tee -a "$MAIN"; exit 1; fi
TEACHER_NODES=("${NODES[@]:0:TEACHER_NNODES}")
ROLLOUT_NODES=("${NODES[@]:TEACHER_NNODES:ROLLOUT_NNODES}")
TRAINER_NODES_ARR=("${NODES[@]:TEACHER_NNODES+ROLLOUT_NNODES:TRAINER_NNODES}")
TRAINER_NODES=$(IFS=,; echo "${TRAINER_NODES_ARR[*]}")
HEAD="${TRAINER_NODES_ARR[0]}"                      # trainer rdzv head + orchestrator host

TEACHER_TP=${TEACHER_TP:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
TEACHERS_PER_NODE=${TEACHERS_PER_NODE:-2}
ROLLOUTS_PER_NODE=${ROLLOUTS_PER_NODE:-8}
PORT_SHIFT=$(( (HOLDER % 800) * 32 ))
T_PORT0=$(( 8100 + PORT_SHIFT )); T_DIST0=$(( 38100 + PORT_SHIFT )); T_NCCL0=$(( 38600 + PORT_SHIFT ))
R_PORT0=$(( 8200 + PORT_SHIFT ))
TRAINER_RDZV_PORT=$(( 29500 + (PORT_SHIFT % 1000) ))
TRAINER_HTTP_PORT=${TRAINER_HTTP_PORT:-$(( 8300 + (PORT_SHIFT % 100) ))}

NCCL_ENV="NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
[ -n "${NCCL_IB_HCA:-}" ] && NCCL_ENV="NCCL_IB_HCA=$NCCL_IB_HCA $NCCL_ENV"
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && NCCL_ENV="NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME $NCCL_ENV"

# ---- role of THIS node (rank ordering == slurm node ordering) ----
ROLE=trainer
if   [ "$RANK" -lt "$TEACHER_NNODES" ]; then ROLE=teacher
elif [ "$RANK" -lt $((TEACHER_NNODES+ROLLOUT_NNODES)) ]; then ROLE=rollout
fi
TRAINER_LOCAL_RANK=$(( RANK - TEACHER_NNODES - ROLLOUT_NNODES ))   # slurm's SLURM_NODEID analog (trainer only)
IS_HEAD=0; { [ "$ROLE" = trainer ] && [ "$TRAINER_LOCAL_RANK" = 0 ]; } && IS_HEAD=1
MY_NODE="${NODES[$RANK]}"

# Teacher/rollout endpoint URL lists (built EXACTLY as hers: http://<node>:<base+i>).
TURLS=""
for ti in $(seq 0 $((TEACHER_NNODES-1))); do n="${TEACHER_NODES[$ti]}"
  for i in $(seq 0 $((TEACHERS_PER_NODE-1))); do TURLS+="http://$n:$((T_PORT0+i)),"; done; done
TURLS=${TURLS%,}
RURLS=""
for ri in $(seq 0 $((ROLLOUT_NNODES-1))); do n="${ROLLOUT_NODES[$ri]}"
  for i in $(seq 0 $((ROLLOUTS_PER_NODE-1))); do RURLS+="http://$n:$((R_PORT0+i)),"; done; done
RURLS=${RURLS%,}

echo ">>> rank=$RANK role=$ROLE node=$MY_NODE | teacher=[${TEACHER_NODES[*]}]x$TEACHERS_PER_NODE(TP$TEACHER_TP) rollout=[${ROLLOUT_NODES[*]}]x$ROLLOUTS_PER_NODE(TP$ROLLOUT_TP fp8) trainer=[$TRAINER_NODES](world=$((8*TRAINER_NNODES))) head=$HEAD" | tee -a "$MAIN"

stop_requested() { [ -f "$STOP" ] || compgen -G "$FAILED_GLOB" >/dev/null; }

# ---- cleanup: local pkill (her patterns); head signals STOP, a failing node signals FAILED ----
cleanup() {
  local rc=$?
  echo ">>> cleanup rank=$RANK role=$ROLE tag=$TAG rc=$rc $(date)" | tee -a "$MAIN"
  [ "$IS_HEAD" = 1 ] && : > "$STOP" 2>/dev/null
  [ "$rc" != 0 ] && ! stop_requested && : > "$RUN_DIR/FAILED_${HOLDER}_rank${RANK}" 2>/dev/null
  pkill -9 -f sglang.launch_server 2>/dev/null
  pkill -9 -f opd_v2.trainer.service 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# block until STOP / any FAILED, or until one of the given PIDs dies (return 1 = a child died).
serve_wait() {
  local pids=("$@")
  while :; do
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo ">>> [rank $RANK] a child ($p) died -> exit" | tee -a "$MAIN"; return 1; }; done
    stop_requested && { echo ">>> [rank $RANK] STOP/FAILED seen -> shutting down" | tee -a "$MAIN"; return 0; }
    sleep 15
  done
}

# =====================================================================================
# TEACHER node  (== the slurm teacher srun body, run locally)
# =====================================================================================
if [ "$ROLE" = teacher ]; then
  echo ">>> [rank $RANK] TEACHER on $MY_NODE : $TEACHERS_PER_NODE x TP$TEACHER_TP (ports $T_PORT0+)" | tee -a "$MAIN"
  PIDS=()
  for i in $(seq 0 $((TEACHERS_PER_NODE-1))); do
    gpus=$(seq -s, $((i*TEACHER_TP)) $((i*TEACHER_TP+TEACHER_TP-1)))
    port=$((T_PORT0+i)); dist=$((T_DIST0+i)); nccl=$((T_NCCL0+i))
    CUDA_VISIBLE_DEVICES=$gpus SPOOL=/dev/shm/opd-v2-tea-$port MALLOC_ARENA_MAX=4 \
      DIST_INIT_PORT=$dist SGLANG_NCCL_PORT=$nccl \
      MEMFRAC="${TEACHER_MEMFRAC:-}" MAXRUN="${TEACHER_MAXRUN:-}" \
      bash "$ROLE_DIR/run_teacher.sh" --tp "$TEACHER_TP" --port "$port" \
      > "$RUN_DIR/teacher_${MY_NODE}_$port.log" 2>&1 &
    PIDS+=($!)
  done
  serve_wait "${PIDS[@]}"; exit $?
fi

# =====================================================================================
# ROLLOUT node  (== the slurm rollout srun body, run locally)
# =====================================================================================
if [ "$ROLE" = rollout ]; then
  echo ">>> [rank $RANK] ROLLOUT on $MY_NODE : $ROLLOUTS_PER_NODE x TP$ROLLOUT_TP fp8 (ports $R_PORT0+)" | tee -a "$MAIN"
  PIDS=()
  for i in $(seq 0 $((ROLLOUTS_PER_NODE-1))); do
    gpus=$(seq -s, $((i*ROLLOUT_TP)) $((i*ROLLOUT_TP+ROLLOUT_TP-1)))
    port=$((R_PORT0+i))
    CUDA_VISIBLE_DEVICES=$gpus MALLOC_ARENA_MAX=4 \
      MODEL="${ROLLOUT_MODEL:-}" KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}" SWA_RATIO="${SWA_RATIO:-}" \
      CONTEXT_LEN="${CONTEXT_LEN:-}" MEMFRAC="${MEMFRAC:-}" MAXRUN="${ROLLOUT_MAXRUN:-}" \
      bash "$ROLE_DIR/run_rollout.sh" --port "$port" --tp "$ROLLOUT_TP" \
      > "$RUN_DIR/rollout_${MY_NODE}_$port.log" 2>&1 &
    PIDS+=($!)
  done
  serve_wait "${PIDS[@]}"; exit $?
fi

# =====================================================================================
# TRAINER node. The head (trainer-local-rank 0) runs the health gate + make_config BEFORE its torchrun
# and the orchestrator AFTER; the other trainer nodes wait for this launch's config, then join c10d.
# =====================================================================================
run_trainer() {   # one 8-GPU trainer node; c10d rendezvous assigns node-rank (as in hers)
  PYTHONPATH="$SRC:$OPD_SRC" OPD_RUN_DIR="$RUN_DIR" \
  TRITON_CACHE_DIR="/tmp/triton_opdv2_$TRAINER_LOCAL_RANK" \
  env $NCCL_ENV \
    "$TRAIN_PY" -m torch.distributed.run \
      --nnodes="$TRAINER_NNODES" --nproc_per_node=8 \
      --rdzv-backend=c10d --rdzv-endpoint="$HEAD:$TRAINER_RDZV_PORT" --rdzv-id="$HOLDER" \
      -m opd_v2.trainer.service --run-dir "$RUN_DIR"
}

if [ "$IS_HEAD" = 1 ]; then
  rm -f "$RUN_DIR/trainer_endpoint.json"              # drop a stale trainer URL from a previous launch

  # ---- 3) health gate (from head; teacher cold start + JIT can be ~20min) ----
  HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-1800}
  echo ">>> [head] health gate ($HEALTH_TIMEOUT s) over $(echo "$TURLS,$RURLS" | tr ',' ' ' | wc -w) servers ..." | tee -a "$MAIN"
  urls=$(echo "$TURLS,$RURLS" | tr ',' ' ')
  t0=$(date +%s)
  while :; do
    bad=0
    for u in $urls; do curl -s -m 3 "$u/health" >/dev/null 2>&1 || bad=$((bad+1)); done
    [ "$bad" = 0 ] && { echo "all $(echo $urls|wc -w) servers healthy" | tee -a "$MAIN"; break; }
    stop_requested && { echo "STOP/FAILED during health gate; abort" | tee -a "$MAIN"; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt "$HEALTH_TIMEOUT" ] && { echo "health gate TIMEOUT ($bad unhealthy)" | tee -a "$MAIN"; exit 1; }
    sleep 10
  done

  # ---- 4) config.json (single source of truth) + cu128 ATTN_IMPL ----
  RUN_DIR="$RUN_DIR" RUN_NAME="$RUN_NAME" ROLLOUT_URLS="$RURLS" TEACHER_URLS="$TURLS" \
    TRAINER_HTTP_PORT="$TRAINER_HTTP_PORT" ATTN_IMPL="${ATTN_IMPL:-olmo3_sink_fa2}" \
    "$TRAIN_PY" "$OPD_V2/examples/make_config.py" 2>&1 | tee -a "$MAIN"
  [ "${PIPESTATUS[0]}" = 0 ] && [ -f "$RUN_DIR/config.json" ] || { echo "config.json not written; abort" | tee -a "$MAIN"; exit 1; }
  : > "$CONFIG_READY"                                   # releases the other trainer nodes (this launch only)
else
  # ---- non-head trainer: wait for THIS launch's config (reproduces make_config -> trainer order) ----
  CONFIG_WAIT=${CONFIG_WAIT:-2400}    # must exceed HEALTH_TIMEOUT (teacher cold start) + make_config
  echo ">>> [rank $RANK] trainer waiting for $CONFIG_READY (<=${CONFIG_WAIT}s) ..." | tee -a "$MAIN"
  t0=$(date +%s)
  until [ -f "$CONFIG_READY" ]; do
    stop_requested && { echo "STOP/FAILED before config; abort" | tee -a "$MAIN"; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt "$CONFIG_WAIT" ] && { echo "config wait TIMEOUT; abort" | tee -a "$MAIN"; exit 1; }
    sleep 5
  done
fi

# ---- 5) trainer: this node's torchrun (c10d rendezvous -> one world across trainer nodes) ----
echo ">>> [rank $RANK] trainer torchrun (local-rank $TRAINER_LOCAL_RANK, world=$((8*TRAINER_NNODES))) -> rdzv $HEAD:$TRAINER_RDZV_PORT" | tee -a "$MAIN"
if [ "$IS_HEAD" = 1 ]; then
  run_trainer > "$RUN_DIR/trainer_$TRAINER_LOCAL_RANK.log" 2>&1 &
  TRAINER_PID=$!
  echo ">>> [head] trainer pid=$TRAINER_PID (log: trainer_$TRAINER_LOCAL_RANK.log)" | tee -a "$MAIN"

  # ---- 6) orchestrator (head node, CPU process). Watched so a FAILED node or a dead trainer stops it. ----
  MAX_STEPS=${MAX_STEPS:-100000}
  echo ">>> [head] launch orchestrator on $HEAD (max_steps=$MAX_STEPS)" | tee -a "$MAIN"
  PYTHONPATH="$SRC:$OPD_SRC" OPD_RUN_DIR="$RUN_DIR" CUDA_VISIBLE_DEVICES= \
    "$TRAIN_PY" -m opd_v2.orchestrator --run-dir "$RUN_DIR" --max-steps "$MAX_STEPS" \
    > >(tee -a "$RUN_DIR/orchestrator.log" | tee -a "$MAIN") 2>&1 &
  ORCH_PID=$!
  stop_orch() {   # TERM, then KILL if it hasn't exited within 60s
    echo ">>> [head] $1 -> stopping orchestrator" | tee -a "$MAIN"
    kill "$ORCH_PID" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$ORCH_PID" 2>/dev/null || return 0; sleep 1; done
    kill -9 "$ORCH_PID" 2>/dev/null
  }
  while kill -0 "$ORCH_PID" 2>/dev/null; do
    if compgen -G "$FAILED_GLOB" >/dev/null; then
      stop_orch "FAILED from $(ls $FAILED_GLOB | xargs -n1 basename | tr '\n' ' ')"; break
    fi
    if ! kill -0 "$TRAINER_PID" 2>/dev/null; then
      stop_orch "local trainer died"; break
    fi
    sleep 15
  done
  wait "$ORCH_PID"; ORCH_RC=$?
  [ "$ORCH_RC" = 0 ] && compgen -G "$FAILED_GLOB" >/dev/null && ORCH_RC=1   # a peer failed: never report success
  echo ">>> orchestrator exited rc=$ORCH_RC $(date)" | tee -a "$MAIN"   # cleanup trap writes STOP -> tears down peers
  exit "$ORCH_RC"
else
  run_trainer > "$RUN_DIR/trainer_$TRAINER_LOCAL_RANK.log" 2>&1 &
  serve_wait $!; exit $?
fi
