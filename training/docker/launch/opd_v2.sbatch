#!/usr/bin/env bash
#SBATCH --job-name=opd_v2
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --output=opd_v2_%j.out
#
# OPD v2 on 8 nodes x 8 GPU (H200 or B200) with the cu128 image, via pyxis/enroot.
# 1 teacher (DeepSeek-V4-Flash TP4x2) + 4 rollout (student TP4x2 fp8) + 3 trainer (FSDP2 world 24); the
# orchestrator runs on the first trainer node. See training/docker/launch/README.md.
#
# Pick the preset with OPD_ENV (default = the B200 run):
#   env_max_b200.sh  her V33 + 160k rollout — our 64x B200 run. B200/B300 only.
#   env_v33.sh       her V33 (128k) — faithful reproduction; fits H200 and B200.
#
# Submit from the repo root (or set PP_ROOT), with the shared-FS paths filled in:
#   OPD_IMAGE=/shared/ycchen-opd+cu128.sqsh \
#   RUN_DIR=/shared/runs/opd_max_b200 JIT_CACHE_DIR=/shared/jit_cache \
#   STUDENT_PATH=/shared/models/student-deploy DEEPSEEK_V4_FLASH=/shared/models/DeepSeek-V4-Flash \
#   sbatch training/docker/launch/opd_v2.sbatch
#
# OPD_IMAGE: a squashfs imported once on the shared FS (recommended; avoids every node pulling ~40GB):
#   enroot import -o /shared/ycchen-opd+cu128.sqsh docker://chankhavu/ycchen-opd:cu128
# or a registry ref pyxis understands (e.g. chankhavu/ycchen-opd:cu128).
# The launcher itself runs on the batch host (needs srun/scontrol); every srun step runs inside the image.
set -uo pipefail
unset CUDA_VISIBLE_DEVICES                     # scrub submit-environment pollution (as in her sbatch)

REPO="${PP_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
LAUNCH="$REPO/training/docker/launch"
OPD_ENV="${OPD_ENV:-env_max_b200.sh}"

: "${OPD_IMAGE:?set OPD_IMAGE (enroot .sqsh path or registry ref of chankhavu/ycchen-opd:cu128)}"
: "${RUN_DIR:?set RUN_DIR (shared FS; keep it distinct per preset — 128k and 160k pools must not mix)}"
: "${STUDENT_PATH:?set STUDENT_PATH (deploy-format student dir, e.g. an hf download of chankhavu/yccchen-olmo3-deploy)}"
: "${DEEPSEEK_V4_FLASH:?set DEEPSEEK_V4_FLASH (deepseek-ai/DeepSeek-V4-Flash checkpoint dir)}"
export STUDENT_PATH DEEPSEEK_V4_FLASH RUN_DIR
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-$STUDENT_PATH}"              # one deploy dir serves trainer + rollout
export STUDENT_DEPLOY_PATH="${STUDENT_DEPLOY_PATH:-$STUDENT_PATH}"  # config/tokenizer source for weight-sync saves
export JIT_CACHE_DIR="${JIT_CACHE_DIR:-$(dirname "$RUN_DIR")/jit_cache}"   # FIXED path, persists across runs
mkdir -p "$RUN_DIR" "$JIT_CACHE_DIR"

# Bind every host path the steps touch at the same path inside the container (dedup'd), plus any extras.
mounts=()
for d in "$RUN_DIR" "$JIT_CACHE_DIR" "$STUDENT_PATH" "$ROLLOUT_MODEL" "$STUDENT_DEPLOY_PATH" "$DEEPSEEK_V4_FLASH"; do
  d="$(readlink -f "$d")"
  [[ " ${mounts[*]} " == *" $d:$d "* ]] || mounts+=("$d:$d")
done
MOUNTS="$(IFS=,; echo "${mounts[*]}")${EXTRA_CONTAINER_MOUNTS:+,$EXTRA_CONTAINER_MOUNTS}"
export SRUN_CONTAINER_ARGS="--container-image=$OPD_IMAGE --container-mounts=$MOUNTS --no-container-mount-home ${EXTRA_CONTAINER_ARGS:-}"

source "$LAUNCH/$OPD_ENV"

echo "=== OPD v2 job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST) preset=$OPD_ENV run=$RUN_NAME $(date) ==="
echo "image=$OPD_IMAGE"
echo "ctx=$CONTEXT_LEN gen=$MAX_NEW_TOKENS traj=$MAX_TRAJ_TOKENS micro=$MICRO teacher_ctx=$TEACHER_CONTEXT_LEN"
echo "mounts=$MOUNTS"

exec bash "$LAUNCH/run_mn_cu128.sh"
