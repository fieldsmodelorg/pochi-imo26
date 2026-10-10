# OPD v2 "MAX" — her V33 + ~25% LONGER ROLLOUT (128k -> 160k). B200/B300 ONLY. This is the config of our
# 64x B200 run. Source before run_mn_cu128.sh (or let opd_v2.sbatch do it):
#   source training/docker/launch/env_max_b200.sh && bash training/docker/launch/run_mn_cu128.sh
#
# Everything (LR, beta, batch, topology, backends, checkpointing) is env_v33.sh; only the 5 hard-coupled length
# knobs change. Why it's safe on B200: the student's yarn ceiling is 262144, so 160k is in-distribution; the
# teacher supports 1M. Her V33 trainer sat at the H200 ~139GB edge, +25% MICRO is ~+35GB worst case
# -> ~174GB < B200 ~180GB. That does NOT fit H200 (~141GB): on H200 use env_v33.sh.
# Timeouts are unchanged: B200 decode (~1.67x H200 HBM bandwidth) finishes the longer gen inside her 6000s/7200s.
#   ! VALIDATE on step 1: if the trainer OOMs at MICRO=163840, back off MICRO and MAX_TRAJ_TOKENS TOGETHER by
#     ~8k (her documented single knob) until it fits.
_user_run_name=${RUN_NAME:-}
source "$(dirname "${BASH_SOURCE[0]}")/env_v33.sh"

export CONTEXT_LEN=163584 MAX_TRAJ_TOKENS=163584   # rollout window = prompt + gen cap
export MAX_NEW_TOKENS=160000                        # gen cap (produce clamps to max_traj - prompt)
export MICRO=163840                                 # trainer bin (160*1024); MUST stay >= MAX_TRAJ_TOKENS (core.py hard-blocks)
export TEACHER_CONTEXT_LEN=190000                   # MUST exceed max_traj so the teacher never length-fails
export RUN_NAME=${_user_run_name:-agentic_32b_lc160k_max}   # keep a caller-set RUN_NAME; else the MAX default
unset _user_run_name

_ccmaj=$( (nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true) | head -1 | cut -d. -f1)
if [[ "${_ccmaj:-}" =~ ^[0-9]+$ ]] && [ "$_ccmaj" != "10" ]; then   # skip silently if nvidia-smi is unusable
  echo "WARNING: env_max_b200.sh sizes the trainer for B200 (~174GB/GPU); this node is sm_${_ccmaj}x — use env_v33.sh" >&2
fi
unset _ccmaj
