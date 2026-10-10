# OPD v2 — her V33 production config (training/opd_v2/examples/run_agentic_mn_32b.sbatch), for H200 AND B200.
# Faithful to her best OPD-32B run (V33 ran on 8x H200). Source before run_mn_cu128.sh inside a >=8-node
# slurm allocation (or let opd_v2.sbatch do it):
#   source training/docker/launch/env_v33.sh && bash training/docker/launch/run_mn_cu128.sh
#
# Hardware: nothing in this file is GPU-specific. The per-arch choices are AUTO-DETECTED where they matter:
#   * teacher MoE backend  — run_teacher.sh: sm_100 (B200/B300) -> flashinfer_mxfp4, else auto -> marlin (her H200)
#   * JIT compile cache    — run_{teacher,rollout}.sh scope it per arch (sm_90a vs sm_100) + seed sm_100 from the image
#   * trainer attention    — olmo3_sink_fa2 (stock FA2 built for sm_90 + sm_100) on both
# The 128k sizes below are her H200 numbers (trainer at the ~139GB H200 edge), so they fit H200 and B200.
# For the longer B200-only variant (160k) use env_max_b200.sh, which sources this file.
#
# REQUIRED (cluster paths): STUDENT_PATH, STUDENT_DEPLOY_PATH, ROLLOUT_MODEL, DEEPSEEK_V4_FLASH, RUN_DIR (shared FS).
# The student deploy dir works for BOTH trainer and rollout (transformers reconstructs rope_parameters
# from the legacy rope_scaling; sglang needs the legacy form) — so STUDENT_PATH may equal ROLLOUT_MODEL.

export PRODUCER=agentic
export SEED_SOURCE=${SEED_SOURCE:-chankhavu/ycchen-dsflash-proof-distill-v2-test}   # OPD prompt dataset — byte-faithful, public mirror of ycchen/dsflash-proof-distill-v2-test (the original is no longer public)
export SEED_HF_CONFIG=${SEED_HF_CONFIG:-per_problem}
export ATTN_IMPL=olmo3_sink_fa2          # cu128 delta #1: post-correction sink on stock FA2 (B200 has no FA3; same kernel on H200)
# (cu128 delta #2 is the rollout attention backend = triton, already the default in run_rollout.sh)

# ---- topology (her V33 default): 1 teacher(TP4x2) + 4 rollout(TP4x2 fp8 = 8 replicas) + 3 trainer(world 24) ----
# Overridable per-run (defaults = her V33). Trainer = total_nodes - teacher - rollout (launcher computes it), so
# ROLLOUT_NNODES is the ONE knob for the balance: 4 -> 1+4+3 (V33, trainer-heavy = her optimized); 5 -> 1+5+2
# (her pre-V33 fallback, rollout-heavy). BOTH are memory-safe (HSDP shards within-node at 8, so the trainer count
# doesn't change per-GPU memory). Start V33, watch starved_frac; if it spikes (rollout-bound), set ROLLOUT_NNODES=5.
export TEACHER_NNODES=${TEACHER_NNODES:-1} TEACHER_TP=${TEACHER_TP:-4} TEACHERS_PER_NODE=${TEACHERS_PER_NODE:-2}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-4} ROLLOUT_TP=${ROLLOUT_TP:-4} ROLLOUTS_PER_NODE=${ROLLOUTS_PER_NODE:-2}

# ---- rollout: 128k long-context, fp8 KV, hybrid SWA ----
export CONTEXT_LEN=130816 KV_CACHE_DTYPE=fp8_e4m3 SWA_RATIO=0.2 ROLLOUT_MAXRUN=64 MEMFRAC=0.82
export MAX_TRAJ_TOKENS=130816 MAX_NEW_TOKENS=128000 ROLLOUT_GEN_TIMEOUT=6000
export TARGET_INFLIGHT=512 STARVE_TIMEOUT=7200 DROP_FINISH_REASONS=length

# ---- teacher: score window > max_traj so it never length-fails ----
export TEACHER_CONTEXT_LEN=150000 TEACHER_MEMFRAC=0.6 TEACHER_MAXRUN=16
# B200 MoE backend (REQUIRED on sm_100 — validated 2026-07-13 on 4xB200 TP4, clean Euclid proof).
# DeepSeek-V4-Flash stores its EXPERTS IN FP4 (only attn/router/dense are fp8). On Blackwell `auto`
# mis-resolves to the fp8 triton MoE runner, which CRASHES ("Hidden size mismatch"); deep_gemm
# (swiglu_limit/JIT-EP shape guard) and flashinfer_trtllm (format_is_bypassed) also fail — all are
# fp8 expert runners vs fp4-packed weights (fp8 MoE for V4 is unsupported by design, sglang #25704/#23743).
# flashinfer_mxfp4 is the fp4-NATIVE Blackwell path — SAME precision as her Hopper marlin fp4 experts,
# NOT a downgrade. First launch does a one-time ~15min flashinfer fp4 autotune (looks frozen between
# ~2:18 profiles — it's warming, not hung); persisted via JIT_CACHE_DIR. NVFP4 support (#25820) is
# already in 0.5.14 — no sglang bump needed.
# MoE backend is AUTO-DETECTED in run_teacher.sh (compute_cap major==10 = Blackwell sm_10x = B200/B300 ->
# flashinfer_mxfp4; else auto -> marlin on Hopper). So this env does NOT set MOE_BACKEND, and a b300 wrapper
# that sources this stays thin. Force a specific one with:  export MOE_BACKEND=<backend>
#   export MAX_PREFILL_TOKENS=8192              # only if the FlashMLA mixed decode+prefill crash appears (#23743)
#   add --disable-flashinfer-autotune to run_teacher.sh for a ~1-2min cold start (skips the tune; slightly slower kernels)

# ---- trainer: 32B @140k -> CPU offload, HSDP ----
export MICRO=131072 CHUNK_SIZE=2048 CPU_OFFLOAD=1 TRAIN_BATCH_TRAJS=64
export BETA=1.0 LR=1e-5 WEIGHT_SYNC_EVERY=4 G4_EVERY=5 LOG_EVERY=1
export CHECKPOINT_EVERY=25 CHECKPOINT_KEEP=2 HF_EXPORT=1 RESUME=${RESUME:-0}   # RESUME=1 to continue a requeued run; 25 (was her 50): more frequent durable ckpts; operational-only, no training-dynamics change

export RUN_NAME=${RUN_NAME:-agentic_32b_lc140k_v33}
export MAX_STEPS=${MAX_STEPS:-100000}
export WANDB_PROJECT=${WANDB_PROJECT:-opd-v2-agentic} WANDB_MODE=${WANDB_MODE:-online}
