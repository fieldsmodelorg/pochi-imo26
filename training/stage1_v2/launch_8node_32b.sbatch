#!/usr/bin/env bash
# Stage-1 v2 SFT REAL RUN: fused 32B base (Olmo-3.1-32B-Think-deepseekTok-sink) on
# 8 nodes x 8 H200 = 64 GPU, 1 epoch of the G2-r05 weighted mix (L4 pre-packed).
# HSDP: FSDP2 8-way shard WITHIN each node x replicate ACROSS the 8 nodes -- per-GPU
# memory equals the single-node 8-way shard (~130/141 GB with expandable_segments).
#
# LR: peak 5e-5, ~500-step warmup (ratio 0.0417, matching the validated 7B recipe),
# LINEAR decay to 0.1*peak over --max-steps 12000
# (~1 epoch: 50.4B tok / 4.19M tok/step at 64 GPU). 32B is ~4.3x the 7B FLOPs ->
# expect ~1.8k tok/s/GPU (~115k global, ~36s/step) -> ~5 days wall-clock = ~5 requeue
# cycles on a 1-day partition limit; --requeue + --resume continue exactly each time
# (back up _resume before each manual resubmit -- it is overwritten in place).
# NB: the fp32 _resume DCP is ~390 GB (master+AdamW); SAVE_MIN=60 (a 30-min cadence
# outran the NCCL watchdog on the first attempt).
#   sbatch training/stage1_v2/launch_8node_32b.sbatch              # submit from repo root
#   MAX_STEPS=20 SAVE_MIN=0 sbatch training/stage1_v2/launch_8node_32b.sbatch  # smoke
# (the --output log dir must exist before submission: mkdir -p logs/stage1-v2-32b)
#SBATCH --job-name=s1v2-32b-64g
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1600G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=logs/stage1-v2-32b/slurm-%j.out
#SBATCH --error=logs/stage1-v2-32b/slurm-%j.out

set -euo pipefail
REPO="${PP_ROOT:-$PWD}"   # submit from the repo root, or set PP_ROOT
cd "$REPO"
mkdir -p logs/stage1-v2-32b outputs/stage1-v2-32b

# Scrub env leaked from the submitting shell (interactive alloc + --export=ALL drags
# CUDA_VISIBLE_DEVICES / SLURM_* into the new job -> workers see the wrong GPU count).
unset CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL \
      SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE \
      SLURM_DISTRIBUTION SLURM_GPUS SLURM_GPUS_PER_NODE SLURM_GPUS_ON_NODE \
      SLURM_STEP_GPUS SLURM_JOB_GPUS

source .venv/bin/activate
# Optional redistributed nvcc (FA3 JIT build workaround); skipped if absent.
REDIST="$REPO/.cuda_redist/cuda_nvcc-linux-x86_64-12.6.85-archive"
if [ -d "$REDIST" ]; then
  export CUDA_HOME="$REDIST" PATH="$REDIST/bin:$PATH" LD_LIBRARY_PATH="$REDIST/lib64:${LD_LIBRARY_PATH:-}"
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=12
# 32B at micro 65536 runs ~130/141 GB; expandable_segments prevents the fragmentation
# creep that OOM'd the first 4-node attempt after ~50 steps.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL over native IB: set NCCL_IB_HCA / NCCL_SOCKET_IFNAME for your cluster's fabric
# (e.g. NCCL_IB_HCA=mlx5_0,... NCCL_SOCKET_IFNAME=eth0).
export NCCL_DEBUG=INFO

export WANDB_PROJECT="${WANDB_PROJECT:-proof-pilot-stage1}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-s1v2-32b-64g-${SLURM_JOB_ID}}"
export WANDB_MODE="${WANDB_MODE:-online}"

# ---- tunables (env-overridable; defaults = full run) ----
MICRO_LEN="${MICRO_LEN:-65536}"
MAX_LEN="${MAX_LEN:-12288}"
MAX_HOURS="${MAX_HOURS:-23.5}"
SAVE_MIN="${SAVE_MIN:-60}"
MAX_STEPS="${MAX_STEPS:-12000}"      # decay horizon (~1 epoch at 64 GPU); set small for a smoke run
LR="${LR:-5e-5}"
MODEL="${MODEL:-${OLMO3_SINK_32B_MODEL:?set MODEL or OLMO3_SINK_32B_MODEL to the fused 32B base checkpoint}}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
echo "nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR:$MASTER_PORT nnodes=$SLURM_NNODES model=$MODEL mix=G2-r05 lr=$LR linear->0.1 max_steps=$MAX_STEPS micro_len=$MICRO_LEN"

srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
  python -m torch.distributed.run \
    --nnodes='"$SLURM_NNODES"' --nproc_per_node=8 --node_rank=$SLURM_NODEID \
    --rdzv_id='"$SLURM_JOB_ID"' --rdzv_backend=c10d --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
    '"$REPO"'/training/stage1_v2/src/train.py \
      --model_path '"$MODEL"' \
      --dataset_path '"$REPO"'/data/l4-g2r05-ml12288-mc65536 \
      --output_path '"$REPO"'/outputs/stage1-v2-32b \
      --logdir '"$REPO"'/logs/stage1-v2-32b \
      --micro-len '"$MICRO_LEN"' --max-len '"$MAX_LEN"' \
      --num_train_epochs 1 --max-steps '"$MAX_STEPS"' \
      --learning_rate '"$LR"' --lr-decay linear --min-lr-ratio 0.1 --warmup-ratio 0.0417 \
      --max-hours '"$MAX_HOURS"' --save-minutes '"$SAVE_MIN"' --log-every 5 \
      --master-dtype auto --resume \
      --wandb-project "'"$WANDB_PROJECT"'" --wandb-run-name "'"$WANDB_RUN_NAME"'" --wandb-mode "'"$WANDB_MODE"'"
'
echo "exit=$?"
