# OPD v2 — single-node 4-GPU end-to-end test: Olmo3-32B <- Olmo3-32B distillation.
#   source /opt/opd/launch/env_4gpu_olmo3.sh && bash /opt/opd/launch/run_1node.sh
#
# Teacher and student are both Olmo3-32B (hidden 5120, codec-clean; vocab 129280). Point TEACHER_MODEL /
# TEACHER_PATH at a DIFFERENT checkpoint than STUDENT_PATH for a REAL distillation signal (non-zero JSD);
# leave them unset -> self-distill (JSD ~= 0, a correctness check of the hidden-extract + W_rot codec).
# DeepSeek is NOT used here (it needs TP4 = all 4 GPUs), so the teacher is served via run_teacher_olmo3.sh
# (triton sink + return-hidden + bf16), NOT the DeepSeek run_teacher.sh.
# REQUIRED: STUDENT_PATH (deploy-format Olmo3-32B), RUN_DIR (scratch).

# Producer = single_round here BY NECESSITY, not choice: the dsflash dataset is only reachable via the
# AGENTIC producer, which has a hard min_gen_room(48000)+bundle-caps ~= 56k context floor that will NOT
# fit a 32B trainer on 2 GPUs. This 4-GPU test validates the Olmo3 hidden-extract + W_rot codec (self-
# distill JSD ~= 0) — problem-set-independent — so single_round (in-repo problems.parquet) is correct.
# The dsflash/agentic dataset path is exercised by the 8-GPU smoke (env_1node_smoke.sh) instead.
export PRODUCER="${PRODUCER:-single_round}"
export ATTN_IMPL="${ATTN_IMPL:-olmo3_sink_fa2}"
export STUDENT_PATH="${STUDENT_PATH:?set to the STUDENT Olmo3-32B deploy dir (checkpoint B)}"
export STUDENT_DEPLOY_PATH="${STUDENT_DEPLOY_PATH:-$STUDENT_PATH}"
export ROLLOUT_MODEL="${ROLLOUT_MODEL:-$STUDENT_PATH}"
# teacher = a (different) Olmo3-32B checkpoint A; defaults to the student -> self-distill.
export TEACHER_MODEL="${TEACHER_MODEL:-$STUDENT_PATH}"    # served by the sglang teacher
export TEACHER_PATH="${TEACHER_PATH:-$TEACHER_MODEL}"     # read by the trainer's build_w_rot(head)
export OPD_HID_DIM="${OPD_HID_DIM:-5120}"                 # Olmo3-32B teacher hidden dim (codec/W_rot)
export TEACHER_SCRIPT="${TEACHER_SCRIPT:-run_teacher_olmo3.sh}"

# ---- 4-GPU layout: teacher TP1 [0] | rollout TP1 [1] | trainer world2 [2-3] ----
export TEACHER_GPUS="${TEACHER_GPUS:-0}" TEACHER_TP="${TEACHER_TP:-1}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}" ROLLOUT_TP="${ROLLOUT_TP:-1}"
export TRAINER_GPUS="${TRAINER_GPUS:-2,3}" TRAINER_NPROC="${TRAINER_NPROC:-2}"

# ---- reduced scale (all-32B on 4 GPUs is tight; MICRO must be >= MAX_TRAJ_TOKENS) ----
export CONTEXT_LEN="${CONTEXT_LEN:-24576}" MAX_TRAJ_TOKENS="${MAX_TRAJ_TOKENS:-24576}" MICRO="${MICRO:-24576}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20480}"
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}" SWA_RATIO="${SWA_RATIO:-0.2}" MEMFRAC="${MEMFRAC:-0.85}"
export TEACHER_CONTEXT_LEN="${TEACHER_CONTEXT_LEN:-28672}" TEACHER_MEMFRAC="${TEACHER_MEMFRAC:-0.85}" TEACHER_MAXRUN="${TEACHER_MAXRUN:-8}"
export ROLLOUT_N="${ROLLOUT_N:-2}" ROLLOUT_MAXRUN="${ROLLOUT_MAXRUN:-4}" TARGET_INFLIGHT="${TARGET_INFLIGHT:-8}"
export ROLLOUT_GEN_TIMEOUT="${ROLLOUT_GEN_TIMEOUT:-1800}" STARVE_TIMEOUT="${STARVE_TIMEOUT:-3600}" DROP_FINISH_REASONS="${DROP_FINISH_REASONS-length}"
export CHUNK_SIZE="${CHUNK_SIZE:-2048}" CPU_OFFLOAD="${CPU_OFFLOAD:-1}" TRAIN_BATCH_TRAJS="${TRAIN_BATCH_TRAJS:-4}"
export BETA="${BETA:-1.0}" LR="${LR:-1e-5}" WEIGHT_SYNC_EVERY="${WEIGHT_SYNC_EVERY:-4}" G4_EVERY="${G4_EVERY:-5}" LOG_EVERY="${LOG_EVERY:-1}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}" CHECKPOINT_KEEP="${CHECKPOINT_KEEP:-1}" HF_EXPORT="${HF_EXPORT:-0}" RESUME="${RESUME:-0}"
export RUN_NAME="${RUN_NAME:-opd_4gpu_olmo3}" MAX_STEPS="${MAX_STEPS:-20}"
# online by default (matches opd_v2 config.py wandb_mode="online"); needs WANDB_API_KEY in the env (or
# `wandb login`). Set WANDB_MODE=offline for a no-network run; `wandb sync <dir>` uploads it afterward.
export WANDB_MODE="${WANDB_MODE:-online}" WANDB_PROJECT="${WANDB_PROJECT:-opd-v2-4gpu}"
