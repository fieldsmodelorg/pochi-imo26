#!/usr/bin/env bash
#SBATCH --job-name=dflash-32b-v2t-short
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=88
#SBATCH --mem=1600G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/dflash-32b-v2test-short-64g-%j.log
#
# SHORT-CONTEXT warm-up phase for the 32B v2test DFlash draft (curriculum step 1).
# Cheap short forwards (micro_len 8192 vs 65536 -> target forward ~8x shorter ->
# step ~2-4s) let us run many more steps to push draft acc up fast on abundant
# SFT data, then a later phase warm-starts from this and specializes on long
# proof context (OPD+teacher L4).
#   * DATA   = data/l4-g2-ml4096-mc8192 (the SFT mix re-rendered at max_len 4096 /
#              micro_len 8192 by stage1_v2/build_l4.py; ~3M bins).
#   * TARGET = stage1-v2-32b-softdistill-v2test (same v2test target as the long run).
#   * WARM   = --init-from a half-trained 32B draft checkpoint (latest.pt).
#   * SWA    = 512 (--window-size 512, swa512 config).
#   * MAX_STARTS = 8192 (= full position coverage of an 8192 bin -> max learning
#              per expensive target forward; loss is a weight-normalized MEAN over
#              starts so this needs NO LR change).
# Requeue-safe (auto-resume from latest.pt); preflight GPU probe excludes wedged
# nodes. STEPS generous (cheap steps) -> monitor acc, early-stop on plateau.
#
# Submit from the repo root (or set PP_ROOT); mkdir -p logs first.
set -euo pipefail

ROOT="${PP_ROOT:-$PWD}"
cd "$ROOT"
source training/.venv/bin/activate   # uv sync in training/ (pyproject.toml + uv.lock live there)
# Optional redistributed nvcc (FA3 JIT build workaround); skipped if absent.
REDIST=$ROOT/.cuda_redist/cuda_nvcc-linux-x86_64-12.6.85-archive
if [ -d "$REDIST" ]; then
  export CUDA_HOME=$REDIST
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
fi
export PYTHONPATH=$ROOT/training/dflash:$ROOT/training   # dflash/ first: its dflash.py must win over the dflash/ package

# Cluster-specific NCCL fabric tuning: set NCCL_IB_HCA / NCCL_SOCKET_IFNAME for your
# cluster (e.g. NCCL_IB_HCA=mlx5_0,... NCCL_SOCKET_IFNAME=eth0) before submitting.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29423
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

STEPS=${STEPS:-12000}
DATA=${DATA:-$ROOT/data/l4-g2-ml4096-mc8192}
OUT=${OUT:-$ROOT/outputs/dflash-canonical-32b-v2test-short-ml4096-64g-s$STEPS}
TARGET=${TARGET:-$ROOT/outputs/stage1-v2-32b-softdistill-v2test}
# Optional warm start from an earlier draft checkpoint (the original run warm-started
# from a half-trained 32B canonical draft); leave unset to train from scratch.
INIT_ARGS=()
[ -n "${INIT_FROM:-}" ] && INIT_ARGS=(--init-from "$INIT_FROM")

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
  ${INIT_ARGS[@]+"${INIT_ARGS[@]}"} \
  --steps $STEPS \
  --window-size 512 \
  --block-size 11 \
  --start-chunk-size 2048 \
  --max-starts-per-step 8192 \
  --learning-rate 6e-4 \
  --warmup-steps 200 \
  --max-grad-norm 1.0 \
  --loss-decay-gamma 5.0 \
  --log-interval 20 \
  --save-interval 200 \
  --report-to wandb \
  --wandb-project dflash-olmo3 \
  --wandb-name dflash-canon-32b-v2test-short-ml4096-64g-s$STEPS \
  "$@"
