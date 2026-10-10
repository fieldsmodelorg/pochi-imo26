#!/usr/bin/env bash
# Start OPD v2 on N nodes over ssh + docker (no scheduler). Each node runs run_mn_nodes.sh in the image
# with its rank = its line number in HOSTFILE (0-based; line order == slurm node order: teacher first,
# then rollout, then trainer — the first trainer host also runs the orchestrator).
#
#   OPD_IMAGE=chankhavu/ycchen-opd@sha256:908516a710f3f6c4157a92c0f9723ff862f84a9c3bf347601c6b2b9ca0ef1a25 OPD_ENV=env_max_b200.sh \
#   RUN_DIR=/shared/runs/opd_max_b200 JIT_CACHE_DIR=/shared/jit_cache \
#   STUDENT_PATH=/shared/models/student-deploy DEEPSEEK_V4_FLASH=/shared/models/DeepSeek-V4-Flash \
#   bash training/docker/launch/launch_ssh.sh hosts.txt          # start (prints the LAUNCH_ID)
#   bash training/docker/launch/launch_ssh.sh hosts.txt stop     # remove this launcher's containers
#
# Requirements: passwordless ssh to every host, docker + NVIDIA container toolkit on every host, the image
# present (or pullable) on every host, and RUN_DIR / JIT_CACHE_DIR / model dirs on a shared FS at the
# same path everywhere. Hosts must reach each other by the names in HOSTFILE (set NODE_HOST per line as
# "host routable-name" if the ssh name differs from the name other nodes should use).
# Progress: $RUN_DIR/launch_rank<r>.log, $RUN_DIR/orchestrator.log (head), and `docker logs` on each host.
set -euo pipefail
HOSTFILE=${1:?usage: launch_ssh.sh HOSTFILE [start|stop]}
ACTION=${2:-start}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$HOSTFILE")
NNODES=${#LINES[@]}
NAME_PREFIX=${OPD_CONTAINER_PREFIX:-opd_v2}

if [ "$ACTION" = stop ]; then
  for line in "${LINES[@]}"; do
    h=${line%% *}
    ssh -o BatchMode=yes "$h" "docker ps -aq --filter name=^${NAME_PREFIX}_ | xargs -r docker rm -f" &
  done
  wait; exit 0
fi

: "${OPD_IMAGE:?set OPD_IMAGE (e.g. chankhavu/ycchen-opd:cu128)}"
: "${RUN_DIR:?set RUN_DIR (shared FS; distinct per preset — 128k and 160k pools must not mix)}"
: "${STUDENT_PATH:?set STUDENT_PATH (deploy-format student dir)}"
: "${DEEPSEEK_V4_FLASH:?set DEEPSEEK_V4_FLASH (teacher checkpoint dir)}"
OPD_ENV=${OPD_ENV:-env_max_b200.sh}
[ -f "$HERE/$OPD_ENV" ] || { echo "no preset $HERE/$OPD_ENV" >&2; exit 1; }
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-$STUDENT_PATH}"
export STUDENT_DEPLOY_PATH="${STUDENT_DEPLOY_PATH:-$STUDENT_PATH}"
export JIT_CACHE_DIR="${JIT_CACHE_DIR:-$(dirname "$RUN_DIR")/jit_cache}"
LAUNCH_ID=${LAUNCH_ID:-$(date +%Y%m%d-%H%M%S)-$$}
mkdir -p "$RUN_DIR" "$JIT_CACHE_DIR"

# Same-path bind mounts (dedup'd) + the checkout's launch/ dir over the baked one (launcher from the checkout,
# everything else from the image — as with opd_v2.sbatch).
mounts=()
for d in "$RUN_DIR" "$JIT_CACHE_DIR" "$STUDENT_PATH" "$ROLLOUT_MODEL" "$STUDENT_DEPLOY_PATH" "$DEEPSEEK_V4_FLASH"; do
  [ -e "$d" ] || { echo "missing path: $d" >&2; exit 1; }
  m="$(readlink -f "$d"):$d"                  # same path inside as given (configs record it), real target behind it
  [[ " ${mounts[*]-} " == *" -v $m "* ]] || mounts+=(-v "$m")
done
mounts+=(-v "$HERE:/opt/opd/launch:ro")
EXTRA_DOCKER_MOUNTS=${EXTRA_DOCKER_MOUNTS:-}
for m in ${EXTRA_DOCKER_MOUNTS//,/ }; do mounts+=(-v "$m"); done

# Env forwarded into every container via a 0600 env-file on the shared FS (keeps secrets off command lines).
# Tuning knobs belong in the preset; this forwards only paths, cluster, run-control and credentials.
FWD=(RUN_DIR JIT_CACHE_DIR STUDENT_PATH ROLLOUT_MODEL STUDENT_DEPLOY_PATH DEEPSEEK_V4_FLASH
     NCCL_IB_HCA NCCL_SOCKET_IFNAME NCCL_DEBUG MAX_STEPS RESUME RESUME_FROM RUN_NAME SEED_SOURCE
     WANDB_API_KEY WANDB_MODE WANDB_PROJECT WANDB_ENTITY HF_TOKEN HEALTH_TIMEOUT CONFIG_WAIT GATHER_TIMEOUT
     TEACHER_NNODES ROLLOUT_NNODES MOE_BACKEND JIT_CACHE_SEED JIT_AUTOTUNE_SEED ${OPD_FORWARD_VARS:-})
ENVF="$RUN_DIR/.launch_env_$LAUNCH_ID"
( umask 077; : > "$ENVF"
  for v in "${FWD[@]}"; do [ -n "${!v+x}" ] && printf '%s=%s\n' "$v" "${!v}" >> "$ENVF"; done
  printf 'NNODES=%s\nLAUNCH_ID=%s\n' "$NNODES" "$LAUNCH_ID" >> "$ENVF" )

echo ">>> launch_id=$LAUNCH_ID nnodes=$NNODES preset=$OPD_ENV image=$OPD_IMAGE run_dir=$RUN_DIR"
rank=0; PIDS=(); HOSTS=()
for line in "${LINES[@]}"; do
  read -r h node_host <<< "$line"
  node_host=${node_host:-$h}
  cmd=(docker run -d --name "${NAME_PREFIX}_${LAUNCH_ID}_r${rank}"
       --gpus all --network host --ipc host --ulimit memlock=-1 --ulimit stack=67108864
       --env-file "$ENVF" -e NODE_RANK="$rank" -e NODE_HOST="$node_host"
       "${mounts[@]}" "$OPD_IMAGE"
       bash -lc "source /opt/opd/launch/$OPD_ENV && exec bash /opt/opd/launch/run_mn_nodes.sh")
  echo ">>> rank $rank -> $h (as $node_host)"
  ssh -o BatchMode=yes "$h" "$(printf '%q ' "${cmd[@]}")" >/dev/null &
  PIDS+=($!); HOSTS+=("$h"); rank=$((rank+1))
done
fail=0
for i in "${!PIDS[@]}"; do wait "${PIDS[$i]}" || { echo "!!! rank $i on ${HOSTS[$i]}: docker run failed" >&2; fail=1; }; done
rm -f "$ENVF"            # docker has read it at create time
[ "$fail" = 0 ] || { echo "!!! not all nodes started; tear down with: $0 $HOSTFILE stop" >&2; exit 1; }
echo ">>> started. Logs: $RUN_DIR/launch_rank*.log, $RUN_DIR/orchestrator.log. Stop: $0 $HOSTFILE stop"
