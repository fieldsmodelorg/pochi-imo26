# OPD v2 — single-node 8-GPU (4:2:2) end-to-end shakeout. Source before run_1node.sh:
#   source /opt/opd/launch/env_1node_smoke.sh && bash /opt/opd/launch/run_1node.sh
#
# Purpose: exercise the full loop end-to-end (rollout -> teacher hidden-extract -> JSD -> FSDP2 step
# -> weight sync -> checkpoint) on one node. Runs the SAME producer as
# production: the AGENTIC self-play pool (prove/verify/refine/select) seeded from the dsflash dataset.
# Scaled down: ~57k seq-len (the agentic min_gen_room floor), small batch, ~20 steps. Env-overridable.
# REQUIRED cluster paths: STUDENT_PATH, DEEPSEEK_V4_FLASH, RUN_DIR (scratch/shared).

# Producer = agentic (her production path). Problems + warm-start proofs come from the seed dataset
# chankhavu/ycchen-dsflash-proof-distill-v2-test — a public mirror of Yi-Chia's
# ycchen/dsflash-proof-distill-v2-test (the original is no longer public), built into the pool at runtime by opd_v2.agentic.seed
# -> needs network on the node (or pre-seed once: python -m opd_v2.agentic.seed --run-dir $RUN_DIR).
export PRODUCER="${PRODUCER:-agentic}"
export SEED_SOURCE="${SEED_SOURCE:-chankhavu/ycchen-dsflash-proof-distill-v2-test}"   # the OPD prompt dataset (user-owned mirror)
export SEED_HF_CONFIG="${SEED_HF_CONFIG:-per_problem}"
export ATTN_IMPL="${ATTN_IMPL:-olmo3_sink_fa2}"
export STUDENT_PATH="${STUDENT_PATH:?set to the deploy-format student dir (e.g. chankhavu/yccchen-olmo3-deploy checkout)}"
export STUDENT_DEPLOY_PATH="${STUDENT_DEPLOY_PATH:-$STUDENT_PATH}"   # same deploy dir works for weight-sync saves
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-$STUDENT_PATH}"
export DEEPSEEK_V4_FLASH="${DEEPSEEK_V4_FLASH:?set to the DeepSeek-V4-Flash checkpoint dir}"

# ---- 8-GPU layout: teacher TP4 [0-3] | rollout TP2 [4-5] | trainer world2 [6-7] ----
export TEACHER_GPUS="${TEACHER_GPUS:-0,1,2,3}" TEACHER_TP="${TEACHER_TP:-4}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-4,5}"     ROLLOUT_TP="${ROLLOUT_TP:-2}"
export TRAINER_GPUS="${TRAINER_GPUS:-6,7}"     TRAINER_NPROC="${TRAINER_NPROC:-2}"
export MOE_BACKEND="${MOE_BACKEND:-marlin}"    # Hopper/H200: marlin OK. B200: pick a Blackwell backend.

# ---- scale: ~57k seq-len = the agentic floor min_gen_room(48000)+max(bundle caps). MICRO >= MAX_TRAJ. ----
# Reduced refine/select bundle caps (her prod 40k/50k) pull the floor down to ~56k so it fits 8 GPUs.
export REFINE_BUNDLE_CAP="${REFINE_BUNDLE_CAP:-8000}" SELECT_BUNDLE_CAP="${SELECT_BUNDLE_CAP:-8000}"
export CONTEXT_LEN="${CONTEXT_LEN:-57344}" MAX_TRAJ_TOKENS="${MAX_TRAJ_TOKENS:-57344}" MICRO="${MICRO:-57344}"
# agentic render-prompt cap MUST be <= MAX_TRAJ_TOKENS (orchestrator guard, else a prompt leaves budget<=0
# for generation and is silently dropped). config.py's default (100000) is sized for her 130k-ctx prod
# max_traj; at this scaled smoke max_traj it would trip the guard -> scale to half of MAX_TRAJ (permissive:
# well above the ~8k-bundle render prompt, so nothing legit is dropped). Her env_v33 leaves it at 100000
# (100000 < 130816, fine) -- this default only applies to the smaller smoke.
export AGENTIC_MAX_PROMPT_TOKENS="${AGENTIC_MAX_PROMPT_TOKENS:-$(( MAX_TRAJ_TOKENS / 2 ))}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-49152}"
# MEMFRAC 0.70 here, NOT her 0.82 (set for TP4 replicas): at TP2 the fp8 weights (~16GB/GPU) sit on top of the
# static KV pool and the fp8 weight-sync reload peaks at ~18-26GB (old fp8 copy + loader clone + bf16 re-quant),
# so 0.70 leaves the same ~25GB reload headroom her 0.82 leaves at TP4.
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}" SWA_RATIO="${SWA_RATIO:-0.2}" MEMFRAC="${MEMFRAC:-0.70}"
export TEACHER_CONTEXT_LEN="${TEACHER_CONTEXT_LEN:-61440}" TEACHER_MEMFRAC="${TEACHER_MEMFRAC:-0.78}" TEACHER_MAXRUN="${TEACHER_MAXRUN:-8}"
export ROLLOUT_N="${ROLLOUT_N:-2}"                 # samples per produce_sample task
export ROLLOUT_MAXRUN="${ROLLOUT_MAXRUN:-4}" TARGET_INFLIGHT="${TARGET_INFLIGHT:-8}"
export ROLLOUT_GEN_TIMEOUT="${ROLLOUT_GEN_TIMEOUT:-3000}" STARVE_TIMEOUT="${STARVE_TIMEOUT:-3600}" DROP_FINISH_REASONS="${DROP_FINISH_REASONS-length}"
export CHUNK_SIZE="${CHUNK_SIZE:-2048}" CPU_OFFLOAD="${CPU_OFFLOAD:-1}" TRAIN_BATCH_TRAJS="${TRAIN_BATCH_TRAJS:-4}"
# WEIGHT_SYNC_EVERY=1 for the smoke (prod = 4): a plumbing test should exercise the trainer->rollout weight
# transfer at step 1, not hide it until step 4. Every step reloads student weights into the rollout (fp8
# re-quant) — the most memory-intensive edge, so we want it validated immediately + on every step.
export BETA="${BETA:-1.0}" LR="${LR:-1e-5}" WEIGHT_SYNC_EVERY="${WEIGHT_SYNC_EVERY:-1}" G4_EVERY="${G4_EVERY:-5}" LOG_EVERY="${LOG_EVERY:-1}"
# CHECKPOINT_EVERY=0 -> skip the durable DCP resume ckpt (a 32B ckpt = ~475GB of fp32 optim+master;
# not needed for a 20-step shakeout). Weight-sync still exercises the rolling weights buffer (~128GB).
# Set CHECKPOINT_EVERY=10 (+ ensure ~500GB-1TB free) to also validate the checkpoint-save path.
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}" CHECKPOINT_KEEP="${CHECKPOINT_KEEP:-1}" HF_EXPORT="${HF_EXPORT:-0}" RESUME="${RESUME:-0}"
export RUN_NAME="${RUN_NAME:-opd_1node_smoke}" MAX_STEPS="${MAX_STEPS:-20}"
# online by default (matches opd_v2 config.py wandb_mode="online"); needs WANDB_API_KEY in the env (or
# `wandb login`). Set WANDB_MODE=offline for a no-network run; `wandb sync <dir>` uploads it afterward.
export WANDB_MODE="${WANDB_MODE:-online}" WANDB_PROJECT="${WANDB_PROJECT:-opd-v2-smoke}"

# ---- LIGHTER validation without the dataset: the single_round producer ----
# single_round has NO min_gen_room floor -> smaller context, no seed/network needed. It draws prompts
# from the in-repo training/distill_gen/problems/problems.parquet (NOT dsflash). Use it for a fast plumbing check:
#   export PRODUCER=single_round
#   export CONTEXT_LEN=40960 MAX_TRAJ_TOKENS=40960 MICRO=40960 MAX_NEW_TOKENS=36864 TEACHER_CONTEXT_LEN=45056
# If the world-2 trainer OOMs at 57k agentic: lower MICRO/MAX_TRAJ_TOKENS, drop TRAIN_BATCH_TRAJS, or set
# TRAINER_NPROC=3 (steal a rollout GPU: ROLLOUT_GPUS=4 ROLLOUT_TP=1 TRAINER_GPUS=5,6,7).
