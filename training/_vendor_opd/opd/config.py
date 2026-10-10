# Copyright 2026 proof-pilot. Apache-2.0.
"""Global OPD configuration.

All knobs live in one place; the orchestrator, the three services, and the loss all read from
here. Defaults follow a lowest-risk, maximum-reuse-of-existing-infra recipe: full-vocab
via quant-hidden, JSD(β=0.5), Flash teacher, token-in-token-out, disk-based weight sync,
near-on-policy (bounded max_staleness).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---- Resource paths (override via environment variables) ----
STUDENT_PATH = os.environ.get("STUDENT_PATH", "/models/olmo3-sink-stage1-v2-7b")
TEACHER_PATH = os.environ.get("DEEPSEEK_V4_FLASH", "/models/DeepSeek-V4-Flash")
SGLANG_SIF = os.environ.get("SGLANG_SIF", "/images/sglang.sif")
HID_DIM = 4096          # Flash / student are both 4096 (JSD compares on vocab logits; hidden dim need not match)
VOCAB_SIZE = 129280     # teacher == student


@dataclass
class TeacherCfg:
    """DeepSeek-V4-Flash teacher scoring service (sglang, in-container)."""
    model_path: str = TEACHER_PATH
    tp_size: int = 4
    # Flash configuration validated in _validate_hidden.py:
    chunked_prefill_size: int = 11264   # flash_mla get_decoding_sched_meta smem cap ~11.6k;
                                        # shorter than the doc is fine (patch 2/3 accumulate hidden across chunks). 16384 raises CUDA invalid argument; -1 crashes
    mem_fraction_static: float = 0.80
    max_running_requests: int = 128
    context_length: int = 69632        # >= max_traj_tokens + margin (Flash itself supports 1M; the student is the bottleneck)
    moe_runner_backend: str = "marlin"  # consumes the native fp4 checkpoint on Hopper
    watchdog_timeout: int = 1800        # cold-start JIT compilation >5min
    host: str = "0.0.0.0"
    port: int = 8100
    max_traj_tokens: int = 65536        # cap beyond this. student yarn limit is 65536 -> align the whole stack to it
    # Concurrency/batching is delegated to sglang's built-in continuous-batching scheduler (/score async +
    # async_generate); max_running_requests is the engine's in-flight limit, so no server-side batch knob is needed.


@dataclass
class RolloutCfg:
    """student sglang server (deploy/target olmo2_sink), token-in-token-out."""
    model_path: str = STUDENT_PATH
    tp_size: int = 1
    temperature: float = 1.0            # rollout diversity
    top_p: float = 0.95                 # nucleus: top_p=1.0 sampling the tail produces junk the teacher can't score effectively
    ignore_eos: bool = False            # used for the 100k smoke; normal training respects EOS by default
    n: int = 4                          # rollouts per prompt
    max_new_tokens: int = 4096
    host: str = "0.0.0.0"
    port: int = 8200
    # deploy-required flags (see docs/stage1_deploy_test): legacy rope, reasoning/tool parser
    reasoning_parser: str = "deepseek-r1"
    tool_call_parser: str = "deepseekv4"


@dataclass
class LossCfg:
    """Liger fused-linear JSD."""
    beta: float = 0.5                   # 0=forward KL, 0.5=JSD, 1=reverse KL (canonical OPD)
    temperature: float = 1.0            # distillation temperature (teacher/student logits divided by the same value)
    length_normalize: bool = True       # length-normalized mean over generated tokens
    chunk_size: int = 1024              # Liger per-token chunk size (does not materialize [L,V])


@dataclass
class TrainerCfg:
    """HSDP trainer (fork of stage1_v2/train.py)."""
    student_path: str = STUDENT_PATH
    lr: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    weight_sync_every: int = 8          # save weights to the rollout every N steps
    micro_batch_tokens: int = 32768
    grad_accum: int = 1
    grad_ckpt: bool = True
    master_dtype: str = "fp32"          # fp32 master + bf16 compute
    weight_out_dir: str = "/tmp/opd-weights"  # local fast disk; cross-node needs shared storage
    mask_easy: bool = False             # experimental, non-canonical: mask positions where teacher-argmax == student-token.
                                        # canonical OPD = RKL mean over all generated tokens; off by default, enable if the data says so
    compute_frac_agree: bool = False    # without cached teacher_top1, per-step argmax does one extra full-vocab
                                        # teacher head matmul; off by default, low-frequency diagnostics via diag_every only.
    require_teacher_hidden: bool = True # CE-only speed baselines can skip int6 decode + teacher-hidden tensors.
    # Which non-weight metadata (config/tokenizer) to copy into the sync dir when writing weights. Empty = use
    # student_path (training config, tf5 rope_parameters); set it to the deploy dir (legacy rope) so a rollout
    # update_weights re-read does not trip the sglang Olmo3Config yarn validation bug (see docs/stage1_deploy_test).
    deploy_config_src: str = ""


@dataclass
class OrchestratorCfg:
    """The brain: buffer / staleness / rates."""
    buffer_capacity: int = 4096         # max number of trajectories in the buffer (backpressure)
    buffer_capacity_tokens: int = 8_000_000  # token cap (long-CoT dominated: ~3.3KB/token teacher bytes ~= 26GB RAM)
    max_staleness: int = 16             # drop trajectories where cur_step - weight_version > this value
    prompts_per_pull: int = 32          # how many prompts to pull for rollout each time
    train_batch_trajs: int = 8          # trajectories the trainer consumes per step
    long_windowing: bool = True         # window long trajectories exceeding micro_batch_tokens instead of dropping them whole
    window_context_tokens: int = 4096   # how much left-side context each non-first window keeps
    window_target_tokens: int = 32768   # how many generated target tokens each window trains on


@dataclass
class OPDConfig:
    teacher: TeacherCfg = field(default_factory=TeacherCfg)
    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    orch: OrchestratorCfg = field(default_factory=OrchestratorCfg)

    # prompt source: 'problems' = distill_gen deduplicated proof problem bank (the OPD training corpus, with a
    #                prover template applied); 'l2' = L2 messages mix (built-in math/proof defaults in prompts.py)
    prompt_source: str = "problems"
    prompt_seed: int = 0                  # prompt shuffle/template seed; the launcher can use SLURM_JOB_ID to avoid cross-run repeats
    problems_parquet: str = os.environ.get("PROBLEMS_PARQUET", "training/distill_gen/problems/problems.parquet")
    prover_template_pool: tuple[str, ...] = ("proofbench_generator", "dsmv2_a1", "imo25_prover")
    # L2 source (used when prompt_source='l2')
    dataset_roots: tuple[str, ...] = (
        os.environ.get("SFT_MIX", "data/nemotron-deepseek-sft-mix"),
        os.environ.get("SFT_MIX_V2", "data/nemotron-deepseek-sft-mix-v2"),
    )
    mix_config: str = ""                # empty = use the built-in math/proof defaults (see prompts.py)
    wandb_project: str = "opd-olmo3"

    def hidden_dim(self) -> int:
        return HID_DIM
