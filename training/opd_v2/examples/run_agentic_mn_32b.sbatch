#!/usr/bin/env bash
#SBATCH --job-name=opd_agentic_32b
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1600G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=training/opd_v2/runs/agentic_32b_lc140k_v33/slurm_%j.out
#
# Agentic semi-on-policy OPD -- 32B student (stage1-v2-32b-softdistill-v2test).
# 64 GPU / 8 nodes: 1 teacher + 4 rollout + 3 trainer (V33 rebalance). 140k long-CoT +
# SWA + fp8-KV, conc 64/replica.
# vs the 7B run: student=32B, ctx 131k->140k, gen->128k, teacher ctx->150k,
#   rollout gen timeout 3600->6000, MICRO 200k->148k (>=max_traj to avoid lpt drop +
#   fit 32B memory).
# V33: fast sharded save_weights + cap-hit admission filter (DROP_FINISH_REASONS) +
# topology rebalance (4 rollout / 3 trainer).
#
#   sbatch training/opd_v2/examples/run_agentic_mn_32b.sbatch   # submit from repo root
#   (the --output dir must exist before submission: mkdir -p training/opd_v2/runs/agentic_32b_lc140k_v33)
#
# * Before submitting (once): (1) build the deploy variant (legacy-rope config, used
#   for rollout loading + weight-sync saves):
#     python deploy/make_olmo3sink_deploy.py \
#       --src outputs/stage1-v2-32b-softdistill-v2test \
#       --dst outputs/stage1-v2-32b-softdistill-v2test-deploy
#   (2) pre-build the pool seed.jsonl (avoids headless private-HF auth; or copy the
#   same-source seed from a previous run):
#     RUN=$PWD/training/opd_v2/runs/agentic_32b_lc140k_v33
#     PYTHONPATH=$PWD/training/opd_v2/src \
#       .venv/bin/python -m opd_v2.agentic.seed --run-dir "$RUN"
#
# RUN_NAME is fixed -> pool + DCP checkpoints persist across requeues. RESUME=0 here
# starts fresh (set back to 1 to continue after a requeue).
# NOTE: rollout fp8 32B TP4 + update_weights_from_disk -- watch the rollout logs at
# startup on a new setup.
set -uo pipefail

# Scrub submit-environment pollution.
unset CUDA_VISIBLE_DEVICES

REPO="${PP_ROOT:-$PWD}"   # submit from the repo root, or set PP_ROOT

# ---- producer: agentic ----
export PRODUCER=agentic

# ---- models (32B softdistill-v2test, yarn 256k window) ----
#   trainer build_model uses the training format (rope_parameters); rollout serving +
#   weight-sync configs use the deploy format (legacy rope_scaling). Build the deploy
#   variant with make_olmo3sink_deploy.py first (see header note (1)).
export STUDENT_PATH=$REPO/outputs/stage1-v2-32b-softdistill-v2test
export ROLLOUT_MODEL=$REPO/outputs/stage1-v2-32b-softdistill-v2test-deploy
export STUDENT_DEPLOY_PATH=$REPO/outputs/stage1-v2-32b-softdistill-v2test-deploy

# ---- topology: 1 teacher(2xTP4) + 4 rollout(2xTP4 fp8 = 8 replicas) + 3 trainer(world=24, HSDP 3x8) ----
#   V33 rebalance: trainer 2->3 (bottleneck was the trainer: starved_frac~0.5%, teacher
#   idle), rollout 5->4 (had headroom).
#   Note: faster trainer + fewer rollouts + dropping cap-hits can flip the run to
#   rollout-bound; watch starved_frac at startup and go back to 5 rollout / 2 trainer
#   if it spikes.
export TEACHER_NNODES=1  TEACHER_TP=4  TEACHERS_PER_NODE=2
export ROLLOUT_NNODES=4  ROLLOUT_TP=4  ROLLOUTS_PER_NODE=2
# (trainer = 8 - 1 - 4 = 3 nodes; run_mn.sh computes it -> world 24)

# ---- rollout server: 140k long-context (SWA pool + fp8 KV). 32B GQA-8 -> KV-friendly ----
export CONTEXT_LEN=130816
export KV_CACHE_DTYPE=fp8_e4m3      # must be e4m3: keeps FA3 + sink
export SWA_RATIO=0.2                # long CoT: maximize the full pool (check the log that the hybrid SWA pool is active)
export ROLLOUT_MAXRUN=64            # conc 64/replica (= --max-running-requests + --cuda-graph-max-bs)
export MEMFRAC=0.82                 # 0.92->0.82: at 0.92 only ~10.9GB free, but the fp8 reload peak is
#   ~18-26GB (old fp8 copy + loader clone + bf16 re-quant) -> the whole rollout fleet OOMs.
#   0.82 leaves ~25GB to cover the reload; the KV full pool is still ~8.5M tok > worst-case
#   conc64 x 130816 = 8.37M (pool pressure only preempts, never crashes).

# ---- teacher: 150k score (> max_traj 140k -> the teacher never length-fails) ----
export TEACHER_CONTEXT_LEN=150000
export TEACHER_MEMFRAC=0.6  TEACHER_MAXRUN=16

# ---- data-plane / trajectory ----
export MAX_TRAJ_TOKENS=130816       # = rollout ctx (128k-256) -> produce never over-requests
export MAX_NEW_TOKENS=128000        # generation cap 128k (produce clamps to max_traj-prompt)
export ROLLOUT_GEN_TIMEOUT=6000     # per-rollout aiohttp timeout; long CoT must not be cut mid-generation
export TARGET_INFLIGHT=512          # keep-N-in-flight ~= 8 replicas x 64
export STARVE_TIMEOUT=7200          # > gen timeout 6000 (otherwise all-slots-long looks like a stuck producer)
export DROP_FINISH_REASONS=length   # V33: window-truncated rollouts (incl. degenerate loops, ~5.7%) never enter training -> break self-amplification

# ---- trainer (32B @140k -> CPU offload; HSDP) ----
export MICRO=131072                 # = the validated soft_distill_v2 32B micro-len. MUST be >= MAX_TRAJ_TOKENS (130816).
export CHUNK_SIZE=2048              # JSD chunk = the soft_distill 32B value (opd_v2 default 4096 uses more memory; saves ~2GB)
#   memory note: HID_DIM=4096 -> opd_v2 carries teacher_hidden+w_rot ~2GB more than
#   soft_distill; chunk 2048 roughly cancels it -> 131072 sits at the soft_distill
#   ~139.4GB edge (~0-1GB margin). If it still OOMs: lower MICRO and MAX_TRAJ_TOKENS
#   together by ~8k (the single knob; no re-analysis needed).
export CPU_OFFLOAD=1                 # required for 32B@140k (FSDP2 CPUOffloadPolicy)
export TRAIN_BATCH_TRAJS=64          # LPT packs short trajectories into per-rank bins
export BETA=1.0                      # reverse-KL (on-policy canonical)
export LR=1e-5  WEIGHT_SYNC_EVERY=4  G4_EVERY=5  LOG_EVERY=1
export CHECKPOINT_EVERY=50  CHECKPOINT_KEEP=2  HF_EXPORT=1  RESUME=0   # fresh run; set RESUME=1 to continue

# ---- run identity (fixed -> pool + checkpoints persist across requeues) + monitoring ----
export RUN_NAME=${RUN_NAME:-agentic_32b_lc140k_v33}
export MAX_STEPS=${MAX_STEPS:-100000}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-opd-v2-agentic}

mkdir -p "$REPO/training/opd_v2/runs/$RUN_NAME"

HERE=$REPO/training/opd_v2/examples
echo "=== agentic OPD 32B sbatch job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES nodelist=$SLURM_JOB_NODELIST $(date) ==="
echo "layout: 1 teacher(TP4x2) + 4 rollout(TP4x2=8 replicas, conc64) + 3 trainer(world24) | run=$RUN_NAME"
echo "student=$STUDENT_PATH  rollout=$ROLLOUT_MODEL"
echo "ctx=$CONTEXT_LEN gen=$MAX_NEW_TOKENS traj=$MAX_TRAJ_TOKENS teacher_ctx=$TEACHER_CONTEXT_LEN gen_timeout=$ROLLOUT_GEN_TIMEOUT micro=$MICRO"

exec bash "$HERE/run_mn.sh"
