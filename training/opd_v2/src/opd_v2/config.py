# Copyright 2026 proof-pilot. Apache-2.0.
"""OPD v2 configuration — the **single source of truth** (fixes v1's P10 "three drifting default sets").

All four processes (rollout sglang / teacher sglang / trainer server / orchestrator) read from the
**same** `OPDConfig`. Mechanism: the launcher parses once -> writes `<run>/config.json` -> each of the
four processes calls `OPDConfig.load(run_dir)` at startup to read back the same file, so no process
carries its own defaults.

This file does **not import torch** (the orchestrator is a pure-CPU async process and must be importable
on a GPU-less host). Path/dimension constants follow v1 (validated): student=stage1-v2-7b,
teacher=DeepSeek-V4-Flash, hid=4096, vocab=129280 (teacher==student, verified in G1).
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field

# ---- Fixed constants (this cluster, inherited from v1) ----
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
STUDENT_PATH = f"{REPO}/outputs/stage1-v2-7b"
STUDENT_DEPLOY_PATH = f"{REPO}/outputs/stage1-v2-7b-deploy"  # legacy-rope config (safe for rollout reload)
TEACHER_PATH = os.environ.get("DEEPSEEK_V4_FLASH", "/models/DeepSeek-V4-Flash")
SGLANG_SIF = os.environ.get("SGLANG_SIF", "/images/sglang.sif")
HID_DIM = int(os.environ.get("OPD_HID_DIM", "4096"))   # TEACHER hidden dim (codec/W_rot). Default 4096 =
                                                        # DeepSeek-V4-Flash (unchanged). Set OPD_HID_DIM=5120
                                                        # for an Olmo3-32B self-distill teacher (H200 test).
VOCAB_SIZE = 129280
PAD_ID = 2          # student pad_token_id (used to pad the packing tail; masked out by cu_seqlens/IGNORE)
EOS_ID = 1


@dataclass
class RolloutCfg:
    """student rollout sglang server (fp8 flash_rl deployment, token-in-token-out)."""
    urls: list[str] = field(default_factory=lambda: ["http://127.0.0.1:8200"])
    tp_size: int = 1
    fp8: bool = True                    # flash_rl fp8 deployment (V30; opd_v2/flash_rl)
    # sampling (V1: one request = one trajectory; the N samples are fanned out in the client)
    n_samples: int = 4                  # number of independent atoms to run per prompt
    temperature: float = 1.0
    top_p: float = 0.95                 # top_p=1.0 tail-sampling produces garbage the teacher cannot score
    top_k: int = -1
    # The whole proof must be able to finish (teacher high-effort proofs use up to 65536; the student
    # yarn window is also 65536). produce_sample clamps per request to max_traj_tokens - len(prompt),
    # so the effective cap = the entire remaining window.
    # 4096 was a leftover dev value: proofs average ~40k tok, so 4096 means training on truncated half-proofs.
    max_new_tokens: int = 65536
    ignore_eos: bool = False
    # aiohttp client timeout (seconds) for a single rollout. Hitting it = doomed/slow trajectories get
    # killed, produce nothing, and free the slot.
    # Under long CoT (128k gen) decode is slow; the default 3600 may time out before a legitimately long
    # proof finishes (aborting wastes the whole computed segment).
    gen_timeout_s: float = 3600.0
    # per-replica in-flight limit (aligned with sglang --max-running-requests)
    max_inflight_per_replica: int = 8
    # ---- training-buffer admission (V33; does not touch the rollout sampling distribution, only decides
    #      which on-policy samples enter the gradient) ----
    # A finish_reason matching this is dropped right after generation, before teacher/buffer (before the
    # teacher -> saves a hidden-state disk write).
    # Default ("length",): window-truncated half-proofs / degenerate long loops = the main source of OPD
    #   self-amplification (measured ~5.7%, of which ~2% are truly degenerate loops). Set () to disable.
    #   Extension point: see produce._admission_drop (token-level loop detection could be added later).
    drop_finish_reasons: tuple[str, ...] = ("length",)


@dataclass
class TeacherCfg:
    """DeepSeek-V4-Flash teacher scoring sglang server (+ /score FS-write patch)."""
    urls: list[str] = field(default_factory=lambda: ["http://127.0.0.1:8100"])
    tp_size: int = 4
    max_inflight_per_replica: int = 64   # engine continuous-batching capacity (v1 measured 64-conc ~22.9k tok/s)


@dataclass
class DataPlaneCfg:
    """Data production (produce_sample atom + two load-aware pools + scheduler)."""
    target_inflight: int = 64            # keep-N-in-flight: number of atoms in flight at once
    rollout_concurrency: int = 0         # global rollout semaphore (0 = sum(replica max_inflight))
    teacher_concurrency: int = 0         # global teacher semaphore (0 = sum(replica max_inflight))
    max_traj_tokens: int = 65536         # prompt+gen cap (student yarn limit); anything longer is truncated
    dead_until_seconds: float = 10.0     # how long to skip a replica after consecutive errors
    starve_timeout_s: float = 600.0      # how long the buffer must starve before we treat the producer as
                                         # stuck and raise; for long CoT (a single 100k decode is slow, the
                                         # first batch may take a while) set this higher (e.g. 3600+)


@dataclass
class BufferCfg:
    """Lightweight trajectory buffer (stores only handles, no bytes, V16) + bounded staleness."""
    capacity: int = 4096                 # max number of trajectories
    capacity_tokens: int = 16_000_000    # token cap (backpressure; stores only ids+handle, so it can be larger than v1)
    max_staleness: int = 0               # drop if cur_step - wv > this value; **0=disabled** (OPD has no
                                         # importance ratio, staleness is not a correctness requirement, long-CoT
                                         # rollouts are expensive to discard, so off by default; see is_stale)
    near_full_frac: float = 0.9          # producer backpressure threshold


@dataclass
class LossCfg:
    """full-vocab JSD(β) + V34 routed-OPD stabilization (skew-KL base + routed top-K FKL + EOS/tail reweight).
    repo chunked fp32-softmax kernel (V26, not Liger). **All V34 knobs default to 0/off -> bit-identical
    back to β OPD.** Design: see `V34_PLAN.md`; root cause = length self-amplification / EOS under-training
    (DEEP_REVIEW §A2)."""
    beta: float = 1.0                    # 0=fwdKL 0.5=JSD 1=revKL (on-policy canonical OPD)
    temperature: float = 1.0
    hard_weight: float = 0.0             # pure distillation (CE anchor off by default)
    soft_weight: float = 1.0
    chunk_size: int = 4096               # token chunk for chunked JSD (soft_v2 uses 4096)
    mask_easy: bool = False              # experimental, non-canonical; off by default
    # ---- V34 routed-OPD loss-side root-cause fix (all default to 0/off = falls back to naive β) ----
    # skew reverse-KL: change the base to KL(student ‖ (1-α)·teacher + α·student); α≈0.1 removes the
    # zero-avoiding pathology on teacher-near-zero tokens (the proper fix for length self-amplification;
    # not "just lower β", it preserves signal strength). **Only active when beta==1.**
    skew_alpha: float = 0.0
    # routed top-K forward-KL: overlay FKL on high-entropy / overconfident-wrong / severe-outlier tokens
    # (advice §2). **The whole routing package is gated by fkl_lambda>0** (including base down-weighting on outliers).
    fkl_lambda: float = 0.0              # 0=off; 0.15~0.25=on
    fkl_top_k: int = 64
    route_high_ent_nats: float = 2.5     # teacher entropy(nats) > this -> high-entropy (+FKL)
    route_oc_hs_nats: float = 0.30       # student entropy < this and ...
    route_oc_js: float = 0.30            # ... top-K JS > this -> overconfident-wrong (+FKL, ↓base)
    route_outlier_nll: float = 8.0       # teacher's -logp on the actually-sampled token > this -> severe outlier (+FKL, ↓base)
    base_outlier_down: float = 1.0       # base RKL weight of outlier/oc tokens is multiplied by (1−this); 1=fully disable base (advice)
    # ---- EOS / tail reweight (training-side, on-policy safe; uses seg.labels token-id, computed in the trainer) ----
    clean_eos_reweight: float = 0.0      # for clean-EOS (traj tail label==eos), scale the last K tokens' soft loss ×(1+this)
    clean_eos_k: int = 64                # number of tail tokens for clean-EOS
    tail_loop_mask: bool = False         # degenerate tail (verbatim periodic loop) weight->0, replaces produce-side whole-traj drop
    tail_loop_period_max: int = 64       # max loop period to detect
    tail_loop_min_repeats: int = 4       # minimum number of repeats in the tail to be judged a loop
    eos_region_n: int = 64               # EOS-region diagnostic: number of tail tokens to take per seg (total rows cap 512)


@dataclass
class TrainerCfg:
    """HSDP trainer-as-service (rank-0 HTTP ingress + command-loop across all ranks)."""
    student_path: str = STUDENT_PATH
    teacher_path: str = TEACHER_PATH
    attn: str = "olmo3_sink_fa3"
    lr: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    grad_ckpt: bool = True
    master_dtype: str = "fp32"           # fp32 master + bf16 compute
    cpu_offload: bool = False            # only enable for 32B / very long context (FSDP2 CPUOffloadPolicy, V27)
    micro_batch_tokens: int = 65536      # max packed-bin length (whole traj is not windowed, V25)
    train_batch_trajs: int = 8           # trajectories consumed globally per step (rank-0 LPT splits into world shares)
    weight_sync_every: int = 1           # orchestrator triggers a weight sync every N steps (N=1=most on-policy, V22)
    log_every: int = 1                   # print one line + log to wandb every N steps (1=every step)
    g4_every: int = 5                    # compute g4 agreement + learning diagnostics (entropy/bidirectional KL)
                                         # every N steps; this is a reuse-hidden no_grad small GEMM (cap 4096),
                                         # 5 steps is dense enough yet cheap
    lr_schedule: str = "constant"        # constant | cosine | warmup_cosine
    warmup_steps: int = 0
    total_steps: int = 100000
    http_port: int = 8300                # rank-0 ingress port
    # which config/tokenizer to copy for weight sync (empty=student; set to a deploy dir to avoid the sglang rope validation bug)
    deploy_config_src: str = STUDENT_DEPLOY_PATH
    # durable checkpoint / resume (DCP model+optim+sched, completely separate from the _a/_b rolling buffer, never overwritten, V32)
    # —— the rolling buffer is overwritten every weight_sync and has no optim state; this is the real
    #    time-limit/crash-resistant checkpoint + exact resume.
    checkpoint_every: int = 50           # write a durable ckpt to <run>/checkpoints/step_<N>/ every N steps (0 = off)
    checkpoint_keep: int = 2             # keep the most recent N step_* dirs (-1 = keep all; **prune only after committing latest**)
    checkpoint_dir: str = ""             # empty = <run>/checkpoints
    hf_export: bool = True               # each ckpt also exports a consolidated bf16 HF (step_N/hf/, for eval/serve)
    resume: bool = True                  # on startup, automatically resume from checkpoints/latest.json (model+optim+sched+step)
    resume_from: str = ""                # explicit step dir (empty = auto-find via latest.json)


@dataclass
class RolloutDumpCfg:
    """Dump **all** rollouts (prompt+response token ids) to dflash-native parquet (side channel, decoupled from hidden GC).

    For post-hoc analysis / spec-decode draft (dflash) training. The dump point is in produce_sample
    (after rollout success+truncate, before teacher score) -> even rollouts whose teacher failed are stored.
    See rollout_store.py.
    """
    enabled: bool = True
    dir: str = ""                        # empty = <run_dir>/rollouts
    rows_per_file: int = 1000            # trajectories per parquet file (file-count vs memory/small-file tradeoff)
    flush_interval_s: float = 60.0       # flush periodically even at low rate (avoid lingering in memory)
    store_meta: bool = True              # also store meta (problem_id/template) (JSON column)
    compression: str = "zstd"            # pyarrow built-in codec; = dflash convert_dataset convention


@dataclass
class AgenticCfg:
    """agentic semi-on-policy OPD (pool-based multi-role distillation) — only enabled when producer="agentic".

    Pushes single-round prover OPD to the whole math_3r loop (prove/verify/refine/select): maintains a
    per-problem pool (problem->proofs->verifies, refined); each atom picks a role, assembles context from
    the pool (using math_3r's XML template + rank/bundle), the student generates on-policy, and the teacher
    /scores. Parse-passing student generations are written back to the pool (answer only, think stripped)
    -> the pool deepens and context becomes progressively on-policy. The mix is auto-balanced by fill-fraction
    sampling (verify naturally dominates because of its fan-out, so no un-verified proofs pile up).
    Design: see PLAN § (V33+)/DECISIONS.
    """
    # role target weights (fill_fraction = student_count(role) / weight; pick the available role with the lowest fill_fraction).
    # 22/44/20/14: verify=2×prove (= the fan-out of 2 verifies per proof, so verify keeps up with proof, zero un-verified backlog).
    role_mix: dict = field(default_factory=lambda: {
        "prove": 22.0, "verify": 44.0, "refine": 20.0, "select": 14.0})
    softmax_temp: float = 0.5            # softmax temperature for role selection (>0 adds randomness to avoid thrash; ->0 = argmin)
    # per-problem / per-proof "expansion caps" — only used to spread work within a role, not a hard gate
    max_proofs_per_problem: int = 6
    max_verifies_per_proof: int = 2
    max_refined_per_problem: int = 4
    # context source preference: True = prefer student-source artifacts as context (drives on-policy transfer)
    prefer_student_context: bool = True
    # bundle truncation cap (est tokens = chars//4; < student 128k window, leaves room for long reasoning)
    refine_bundle_cap_tokens: int = 40000
    select_bundle_cap_tokens: int = 50000
    max_prompt_tokens: int = 100000     # skip prompts whose rendered token count exceeds this (rare, safety valve)
    min_gen_room: int = 48000           # startup guard: max_traj_tokens must be ≥ max(bundle_cap)+this,
                                        # otherwise refine/select long reasoning gets truncated (see orchestrator guard)
    max_artifact_chars: int = 200000    # char cap for proof/refined content entering the pool (guards against a
                                        # pathologically long proof blowing up render -> starving that role; 200k chars
                                        # ≈50k tok, a normal proof is far below that)
    # seed (cold-start): fill entirely from DeepSeek r3_hard2000 nested data
    seed_format: str = "hf_per_problem"  # "hf_per_problem" | "records_jsonl"
    seed_source: str = "ycchen/dsflash-proof-distill-v2-test"  # HF repo (per_problem config) or records.jsonl path
    seed_hf_config: str = "per_problem"
    pool_dir: str = ""                   # empty = <run>/pool


@dataclass
class EvalCfg:
    """in-loop ProofBench eval (fixes P11)."""
    enabled: bool = False
    every_weight_versions: int = 50
    teacher_ceiling: float = 4.64        # dsv4-flash high_notool ceiling (/7)


@dataclass
class OPDConfig:
    # run identity + directories
    run_name: str = "opd_v2_dev"
    run_dir: str = ""                    # empty = <opd_v2>/runs/<run_name> (filled in by resolve())
    seed: int = 0
    producer: str = "single_round"       # "single_round" (single-turn prover OPD) | "agentic" (pool-based multi-role)
    prompt_source: str = "problems"
    problems_parquet: str = f"{REPO}/training/distill_gen/problems/problems.parquet"
    prover_template_pool: tuple[str, ...] = ("proofbench_generator", "dsmv2_a1", "imo25_prover")
    wandb_project: str = "opd-v2"
    wandb_mode: str = "online"

    rollout: RolloutCfg = field(default_factory=RolloutCfg)
    teacher: TeacherCfg = field(default_factory=TeacherCfg)
    data_plane: DataPlaneCfg = field(default_factory=DataPlaneCfg)
    buffer: BufferCfg = field(default_factory=BufferCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    trainer: TrainerCfg = field(default_factory=TrainerCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    rollout_dump: RolloutDumpCfg = field(default_factory=RolloutDumpCfg)
    agentic: AgenticCfg = field(default_factory=AgenticCfg)

    # ---- Derived paths (under run_dir; all on shared-FS WekaFS) ----
    def resolve(self) -> "OPDConfig":
        """Fill in the default run_dir and ensure it is an absolute shared-FS path. Called once at launcher startup."""
        if not self.run_dir:
            self.run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "runs", self.run_name)
        self.run_dir = os.path.abspath(self.run_dir)
        return self

    @property
    def hidden_dir(self) -> str:
        return os.path.join(self.run_dir, "hidden")

    @property
    def weights_dir(self) -> str:
        return os.path.join(self.run_dir, "weights")

    @property
    def checkpoints_dir(self) -> str:
        """durable ckpt root (separate from weights_dir's rolling _a/_b)."""
        return self.trainer.checkpoint_dir or os.path.join(self.run_dir, "checkpoints")

    @property
    def rollouts_dir(self) -> str:
        return self.rollout_dump.dir or os.path.join(self.run_dir, "rollouts")

    @property
    def pool_dir(self) -> str:
        """agentic pool root (per-problem graph as append-only JSONL + index)."""
        return self.agentic.pool_dir or os.path.join(self.run_dir, "pool")

    @property
    def trainer_endpoint_file(self) -> str:
        return os.path.join(self.run_dir, "trainer_endpoint.json")

    @property
    def config_file(self) -> str:
        return os.path.join(self.run_dir, "config.json")

    def hidden_dim(self) -> int:
        return HID_DIM

    # ---- JSON round-trip (single source of truth persisted to disk) ----
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self) -> str:
        """After resolve, write <run>/config.json. Returns the path."""
        self.resolve()
        os.makedirs(self.run_dir, exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return self.config_file

    @classmethod
    def from_dict(cls, d: dict) -> "OPDConfig":
        sub = {
            "rollout": RolloutCfg, "teacher": TeacherCfg, "data_plane": DataPlaneCfg,
            "buffer": BufferCfg, "loss": LossCfg, "trainer": TrainerCfg, "eval": EvalCfg,
            "rollout_dump": RolloutDumpCfg, "agentic": AgenticCfg,
        }
        kw = dict(d)
        for k, klass in sub.items():
            if k in kw and isinstance(kw[k], dict):
                kw[k] = klass(**kw[k])
        # restore tuple fields (json turns them into lists)
        if "prover_template_pool" in kw and isinstance(kw["prover_template_pool"], list):
            kw["prover_template_pool"] = tuple(kw["prover_template_pool"])
        return cls(**kw)

    @classmethod
    def load(cls, run_dir: str) -> "OPDConfig":
        """Each of the four processes reads back the same resolved config at startup."""
        path = run_dir if run_dir.endswith(".json") else os.path.join(run_dir, "config.json")
        with open(path) as f:
            return cls.from_dict(json.load(f)).resolve()
