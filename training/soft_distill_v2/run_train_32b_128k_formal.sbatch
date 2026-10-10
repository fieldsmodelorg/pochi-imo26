#!/bin/bash
# soft_distill_v2 — FORMAL 32B @128k run. 64-GPU (8 node) HSDP + FSDP2 CPUOffloadPolicy
# (params+grad+optstate -> CPU, gloo CPU collective), whole-proof 128k rows, chunked
# pure-soft forward-KL JSD. Mirrors the 7B formal recipe (combined proofs-v2+aops_cot data,
# LR 2e-5, HF checkpoint per epoch + final) with the 32B-specific knobs:
#   --cpu-offload --master-dtype bf16 --micro/truncate 131072 --chunk-size 2048.
#
# Data at 128k: 77,737 of 80,582 docs survive (seq_len<=131072; 2,845 long proofs dropped),
# -> ~21,682 rows, world=64 -> steps_per_epoch~338. 2 epochs (~676 steps, ~19h @ ~100s/step
# + checkpoint saves) fits a 24h partition wall; 3 epochs would need DCP resume (not built).
#
# Smoke reuse (validate 64-GPU gloo + the cpu-offload save path before the real run):
#   EPOCHS=0 MAX_STEPS=3 SAVE_EVERY=2 OUT_DIR=outputs/_sdv2_32b_smoke \
#   WANDB_MODE=disabled sbatch --time=00:40:00 --job-name=sdv2-32b-smoke run_train_32b_128k_formal.sbatch
#
# v2test continuation (the stage1-v2-32b-softdistill-v2test checkpoint) reused this
# script with env overrides -- new data index + init from the previous epoch-1
# checkpoint on 7 nodes / 56 GPU:
#   INDEX=<dsflash-v2-test index.jsonl> \
#   STUDENT=outputs/stage1-v2-32b-softdistill-v2/ep1 \
#   OUT_DIR=outputs/stage1-v2-32b-softdistill-v2test \
#   sbatch --nodes=7 run_train_32b_128k_formal.sbatch
#
# Submit from the repo root (or set PP_ROOT); mkdir -p logs/soft_distill_v2 first.
#SBATCH --job-name=sdv2-32b-formal
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1600G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/soft_distill_v2/formal32b_%j.log
#SBATCH --error=logs/soft_distill_v2/formal32b_%j.log
set -euo pipefail

REPO="${PP_ROOT:-$PWD}"
cd "$REPO"
mkdir -p "$REPO/logs/soft_distill_v2"

# Scrub parent-job env pollution (interactive alloc + --export=ALL drags CUDA_VISIBLE_DEVICES /
# SLURM_* into the new job -> workers see wrong GPU count).
unset CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL \
      SLURM_NTASKS SLURM_NPROCS SLURM_NTASKS_PER_NODE SLURM_TASKS_PER_NODE \
      SLURM_DISTRIBUTION SLURM_GPUS SLURM_GPUS_PER_NODE SLURM_GPUS_ON_NODE \
      SLURM_STEP_GPUS SLURM_JOB_GPUS

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SLURM_CPU_BIND=none
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Cluster-specific fabric tuning: set NCCL_IB_HCA for your IB HCAs if needed.
# gloo CPU collective (CPUOffloadPolicy grads/params) goes over the socket iface (TCP, not IB):
# set NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME to your node interconnect interface.
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-soft-distill-v2}"
export WANDB_MODE="${WANDB_MODE:-online}"

# INDEX: teacher-hidden training index (teacher_extract/dataprep output; the formal run
# used the combined proofs-v2 + aops_cot pool, merged with make_combined_index.py).
INDEX="${INDEX:-$REPO/data/hidden/index.jsonl}"
OUT_DIR="${OUT_DIR:-$REPO/outputs/stage1-v2-32b-softdistill-v2}"
STUDENT="${STUDENT:-$REPO/outputs/stage1-v2-32b-y256k-base}"
EPOCHS="${EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:-0}"
SAVE_EVERY="${SAVE_EVERY:-0}"
MICRO="${MICRO:-131072}"
TRUNCATE="${TRUNCATE:-131072}"
CHUNK="${CHUNK:-2048}"
LR="${LR:-2e-5}"
BETA="${BETA:-0.0}"   # jsd_kernel beta: 0=forward-KL (default), 0.5=JSD, 1=reverse-KL
WARMUP="${WARMUP:-25}"
SEED="${SEED:-${SLURM_JOB_ID:-20260617}}"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT="${MASTER_PORT:-29561}"
echo "[sdv2-32b-formal] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR:$MASTER_PORT"
echo "[sdv2-32b-formal] student=$STUDENT index=$INDEX out=$OUT_DIR epochs=$EPOCHS max_steps=$MAX_STEPS save_every=$SAVE_EVERY lr=$LR beta=$BETA warmup=$WARMUP micro=$MICRO truncate=$TRUNCATE chunk=$CHUNK"

srun --cpu-bind=none --cpus-per-task="${SLURM_CPUS_PER_TASK:-96}" \
     --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -c '
    '"$REPO"'/.venv/bin/python -m torch.distributed.run \
      --nnodes='"$SLURM_NNODES"' --nproc_per_node=8 --node_rank=$SLURM_NODEID \
      --rdzv_id='"$SLURM_JOB_ID"' --rdzv_backend=c10d --rdzv_endpoint='"$MASTER_ADDR:$MASTER_PORT"' \
      '"$REPO"'/training/soft_distill_v2/train_soft_distill.py \
        --index '"$INDEX"' --out-dir '"$OUT_DIR"' --student-path '"$STUDENT"' \
        --cpu-offload --master-dtype bf16 \
        --epochs '"$EPOCHS"' --max-steps '"$MAX_STEPS"' --save-every '"$SAVE_EVERY"' \
        --micro-len '"$MICRO"' --truncate-len '"$TRUNCATE"' --chunk-size '"$CHUNK"' \
        --lr '"$LR"' --beta '"$BETA"' --warmup-steps '"$WARMUP"' --seed '"$SEED"'
  '
echo "[sdv2-32b-formal] srun returned $?"
