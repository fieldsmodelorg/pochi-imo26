#!/usr/bin/env bash
#SBATCH --job-name=dflash-32b-v2t-phaseL
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=88
#SBATCH --mem=1600G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/dflash-32b-v2test-phaseL-64g-%j.log
#
# Phase L (long-context specialization) for the 32B v2test DFlash draft.
# Curriculum step 2: warm-start from the Phase S short-context draft (acc 0.64)
# and specialize on the REAL deployment distribution (long proof generation).
#   * WARM   = Phase S final.pt (dflash-canonical-32b-v2test-short-ml4096-64g-s12000,
#              acc 0.64 / pos0 0.86). Loaded before compile; ckpt is compile-agnostic.
#   * DATA   = data/l4-dflash32b-opd-dsflash-ml65536 (OPD 32B rollouts, finish_reason
#              =length filtered, + dsflash-v2-test 65,836 teacher proofs; micro 65536).
#   * GAMMA  = 20 (was 5): flatten per-position loss decay so pos8/9 get ~2.5x more
#              weight (exp(-9/20)=0.64 vs exp(-9/5)=0.17). Near positions still
#              dominate via greedy-prefix gating (train_ratio pos0 0.81 vs pos9 0.24).
#   * COMPILE= --compile-layers (batched FA3 + torch.compile per draft layer;
#              long-ctx draft train_s ~4.3->2.4s, draft peak ~68->47GiB). Explicit
#              though it defaults on. Step 1 pays compile overhead (minutes);
#              watch startup.
#   * MAX_STARTS = 8192 (amortize the expensive long-ctx target forward; chunked by
#              start_chunk_size 2048 so peak stays bounded).
# Requeue-safe (auto-resume from latest.pt); preflight GPU probe excludes wedged nodes.
#
# Submit from the repo root (or set PP_ROOT); mkdir -p logs first.
set -euo pipefail

# Scrub submit-env pollution: when launched from inside another job/session, sbatch
# --export=ALL can leak CUDA_VISIBLE_DEVICES (and agent-session env) into this job.
# A leaked CUDA_VISIBLE_DEVICES makes torch see 0 GPUs -> "CUDA unknown error /
# devices zero" -> crash in ~30s. MUST precede any srun.
unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER || true
unset CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_SESSION_ID CLAUDE_CODE_ENTRYPOINT \
      CLAUDE_CODE_EXECPATH CLAUDECODE CLAUDE_EFFORT AI_AGENT || true

ROOT="${PP_ROOT:-$PWD}"
cd "$ROOT"
source .venv/bin/activate
# Optional redistributed nvcc (FA3 JIT build workaround); skipped if absent.
REDIST=$ROOT/.cuda_redist/cuda_nvcc-linux-x86_64-12.6.85-archive
if [ -d "$REDIST" ]; then
  export CUDA_HOME=$REDIST
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
fi
export PYTHONPATH=$ROOT/training:$ROOT/training/dflash

# Cluster-specific NCCL fabric tuning: set NCCL_IB_HCA / NCCL_SOCKET_IFNAME for your
# cluster (e.g. NCCL_IB_HCA=mlx5_0,... NCCL_SOCKET_IFNAME=eth0) before submitting.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29433
echo "[sbatch] job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR"

set +e
BAD=$(srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 --kill-on-bad-exit=0 \
  python -c "
import torch, socket, sys
try:
    for i in range(torch.cuda.device_count()):
        torch.zeros(1, device=f'cuda:{i}')
except Exception:
    print(socket.gethostname(), flush=True); sys.exit(1)
" 2>/dev/null | sort -u | paste -sd, -)
set -e
if [ -n "$BAD" ]; then
  echo "[sbatch] preflight FAILED on node(s): $BAD"
  RQ=$ROOT/logs/.requeue-count-$SLURM_JOB_ID
  N=$(($(cat "$RQ" 2>/dev/null || echo 0) + 1)); echo $N > "$RQ"
  if [ "$N" -le 6 ]; then
    PREV=$(squeue -j $SLURM_JOB_ID -h -o %X | grep -v '(null)' || true)
    NEW="$BAD"; [ -n "$PREV" ] && NEW="$PREV,$BAD"
    echo "[sbatch] excluding $NEW and requeueing"
    scontrol update JobId=$SLURM_JOB_ID ExcNodeList=$NEW 2>/dev/null || true
    scontrol requeue $SLURM_JOB_ID
    scontrol update JobId=$SLURM_JOB_ID ExcNodeList=$NEW 2>/dev/null || true
    sleep 10
    exit 0
  fi
  echo "[sbatch] too many preflight requeues ($N); giving up"
  exit 1
fi
echo "[sbatch] preflight OK on all nodes"

STEPS=${STEPS:-3000}
GAMMA=${GAMMA:-20}
DATA=${DATA:-$ROOT/data/l4-dflash32b-opd-dsflash-ml65536}
TARGET=${TARGET:-$ROOT/outputs/stage1-v2-32b-softdistill-v2test}
INIT_FROM=${INIT_FROM:-$ROOT/outputs/dflash-canonical-32b-v2test-short-ml4096-64g-s12000/final.pt}
OUT=${OUT:-$ROOT/outputs/dflash-canonical-32b-v2test-phaseL-g${GAMMA}-64g-s$STEPS}

srun --kill-on-bad-exit=1 \
  python -m torch.distributed.run \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  training/dflash/canonical_fa3_train_prod.py \
  --target-model-path $TARGET \
  --draft-config-path $ROOT/training/dflash/configs/dflash-olmo3-32b-8L-bs10-swa512-sink.json \
  --train-data-path $DATA \
  --output-dir $OUT \
  --init-from $INIT_FROM \
  --steps $STEPS \
  --window-size 512 \
  --block-size 11 \
  --start-chunk-size 2048 \
  --max-starts-per-step 8192 \
  --learning-rate 6e-4 \
  --warmup-steps 200 \
  --max-grad-norm 1.0 \
  --loss-decay-gamma $GAMMA \
  --compile-layers \
  --log-interval 20 \
  --save-interval 200 \
  --report-to wandb \
  --wandb-project dflash-olmo3 \
  --wandb-name dflash-canon-32b-v2test-phaseL-g${GAMMA}-64g-s$STEPS \
  "$@"
