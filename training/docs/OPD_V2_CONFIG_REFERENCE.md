# OPD v2 — config & knob reference (+ her best OPD-32B run)

Single source of truth = `OPDConfig` (`training/opd_v2/src/opd_v2/config.py`): the launcher parses env
→ writes `<run>/config.json`; all four processes read the same file. The env-override subset (the
practical tuning surface) is `training/opd_v2/examples/make_config.py`. `config.json` is **generated
per run — not committed**; her config-of-record is the production sbatch (§3).

## 1. Env-exposed knobs (env var → field, default)

### Rollout / sampling (`RolloutCfg`)
| env | field | default |
|---|---|---|
| `ROLLOUT_N` | `n_samples` | 4 |
| `ROLLOUT_T` | `temperature` | 1.0 |
| `ROLLOUT_TOP_P` | `top_p` | 0.95 |
| `MAX_NEW_TOKENS` | `max_new_tokens` | 65536 |
| `ROLLOUT_GEN_TIMEOUT` | `gen_timeout_s` | 3600 |
| `ROLLOUT_MAXRUN` | `max_inflight_per_replica` | 8 |
| `ROLLOUT_TP` / `ROLLOUT_FP8` | `tp_size` / `fp8` | 1 / on |
| `DROP_FINISH_REASONS` | `drop_finish_reasons` | `("length",)` |

### Data-plane / buffer / teacher
| env | field | default |
|---|---|---|
| `TARGET_INFLIGHT` | `data_plane.target_inflight` | 64 |
| `MAX_TRAJ_TOKENS` | `data_plane.max_traj_tokens` | 65536 |
| `STARVE_TIMEOUT` | `data_plane.starve_timeout_s` | 600 |
| `BUF_CAPACITY` / `BUF_CAPACITY_TOKENS` | `buffer.capacity` / `capacity_tokens` | 4096 / 16M |
| `MAX_STALENESS` | `buffer.max_staleness` | **0 (disabled)** |
| `TEACHER_TP` / `TEACHER_MAXRUN` | `teacher.tp_size` / `max_inflight_per_replica` | 4 / 64 |

### Trainer (`TrainerCfg`)
| env | field | default |
|---|---|---|
| `LR` | `lr` | 1e-5 |
| `MICRO` | `micro_batch_tokens` | 65536 |
| `TRAIN_BATCH_TRAJS` | `train_batch_trajs` | 8 |
| `WEIGHT_SYNC_EVERY` | `weight_sync_every` | 1 |
| `ATTN_IMPL` | `attn` | `olmo3_sink_fa3` |
| `CPU_OFFLOAD` | `cpu_offload` | off |
| `LR_SCHEDULE` / `WARMUP_STEPS` / `MAX_STEPS` | `lr_schedule` / `warmup_steps` / `total_steps` | constant / 0 / 100000 |
| `G4_EVERY` / `LOG_EVERY` | diagnostics / logging cadence | 5 / 1 |
| `CHECKPOINT_EVERY` / `CHECKPOINT_KEEP` / `CHECKPOINT_DIR` | durable DCP ckpt | 50 / 2 / `<run>/checkpoints` |
| `HF_EXPORT` / `RESUME` / `RESUME_FROM` | consolidated bf16 export / auto-resume | on / on / auto |
| `STUDENT_PATH` / `STUDENT_DEPLOY_PATH` | student / rope-safe deploy config | 7B paths |
| `TRAINER_HTTP_PORT` | rank-0 ingress | 8300 |

### Loss (`LossCfg`)
`BETA` (`loss.beta`, default **1.0** = reverse-KL OPD; 0.5=JSD, 0=forward-KL) · `CHUNK_SIZE`
(`loss.chunk_size`, 4096). **V34 routed-OPD package — all default 0/off = bit-identical to naïve β:**
`SKEW_ALPHA`, `FKL_LAMBDA`, `FKL_TOP_K`, `ROUTE_HIGH_ENT_NATS`, `ROUTE_OC_HS_NATS`, `ROUTE_OC_JS`,
`ROUTE_OUTLIER_NLL`, `BASE_OUTLIER_DOWN`, `CLEAN_EOS_REWEIGHT`, `CLEAN_EOS_K`, `TAIL_LOOP_MASK`,
`TAIL_LOOP_PERIOD_MAX`, `TAIL_LOOP_MIN_REPEATS`, `EOS_REGION_N`.

### Agentic (`AgenticCfg`, only when `PRODUCER=agentic`)
`PRODUCER` (`single_round`|`agentic`) · `ROLE_MIX` (`prove:22,verify:44,refine:20,select:14`) ·
`ROLE_SOFTMAX_TEMP` (0.5) · `MAX_PROOFS_PER_PROBLEM`/`MAX_VERIFIES_PER_PROOF`/`MAX_REFINED_PER_PROBLEM`
(6/2/4) · `PREFER_STUDENT_CONTEXT` (on) · `REFINE_BUNDLE_CAP`/`SELECT_BUNDLE_CAP` (40k/50k) ·
`AGENTIC_MAX_PROMPT_TOKENS` (100k) · `SEED_SOURCE`/`SEED_FORMAT`/`SEED_HF_CONFIG` · `POOL_DIR`.

### Rollout dump (`RolloutDumpCfg`, DFlash side channel)
`ROLLOUT_DUMP` (on) · `ROLLOUT_DUMP_DIR` · `ROLLOUT_DUMP_ROWS` (1000) · `ROLLOUT_DUMP_FLUSH_S` (60) ·
`ROLLOUT_DUMP_META` (on).

### Misc
`RUN_NAME` · `RUN_DIR` (required) · `ROLLOUT_URLS` / `TEACHER_URLS` (required) · `SEED` · `WANDB_MODE` /
`WANDB_PROJECT`.

## 2. NOT env-exposed (edit `config.json` directly)
- `trainer`: `adam_beta1` (0.9), `adam_beta2` (0.95), `weight_decay` (0.0), `grad_clip` (1.0),
  `grad_ckpt` (True), `master_dtype` (fp32).
- `loss`: `temperature` (1.0), `hard_weight` (0.0), `soft_weight` (1.0), `mask_easy` (False).
- `rollout`: `top_k` (−1), `ignore_eos` (False).
- `data_plane`: `rollout_concurrency` / `teacher_concurrency` (0 = sum of replica capacities),
  `dead_until_seconds` (10).
- **all of `EvalCfg`** (`enabled` False, `every_weight_versions` 50, `teacher_ceiling` 4.64) — the
  in-loop ProofBench eval has no env wiring.

## 3. Her best OPD training: `run_agentic_mn_32b.sbatch` (V33)

The config-of-record for the **delivered OPD-32B** (`origin/main`, 8 nodes / 64 GPU: 1 teacher +
4 rollout + 3 trainer, world 24). It is the env overrides below on top of the §1 defaults:

```bash
PRODUCER=agentic                                  # full prove→verify→refine→select loop
STUDENT_PATH=outputs/stage1-v2-32b-softdistill-v2test
ROLLOUT_MODEL=STUDENT_DEPLOY_PATH=…-deploy         # legacy-rope deploy variant for rollout/weight-sync

# topology (env for run_mn.sh)
TEACHER_NNODES=1 TEACHER_TP=4 TEACHERS_PER_NODE=2
ROLLOUT_NNODES=4 ROLLOUT_TP=4 ROLLOUTS_PER_NODE=2  # 8 fp8 replicas, conc 64
# trainer = 3 nodes → torchrun world 24 (HSDP 3×8)

# rollout — 140k long-context, fp8 KV, hybrid SWA
CONTEXT_LEN=130816  KV_CACHE_DTYPE=fp8_e4m3  SWA_RATIO=0.2  ROLLOUT_MAXRUN=64  MEMFRAC=0.82
MAX_TRAJ_TOKENS=130816  MAX_NEW_TOKENS=128000  ROLLOUT_GEN_TIMEOUT=6000
TARGET_INFLIGHT=512  STARVE_TIMEOUT=7200  DROP_FINISH_REASONS=length

# teacher — score window > max_traj so it never length-fails
TEACHER_CONTEXT_LEN=150000  TEACHER_MEMFRAC=0.6  TEACHER_MAXRUN=16

# trainer — 32B @140k → CPU offload, HSDP
MICRO=131072  CHUNK_SIZE=2048  CPU_OFFLOAD=1  TRAIN_BATCH_TRAJS=64
BETA=1.0  LR=1e-5  WEIGHT_SYNC_EVERY=4  G4_EVERY=5  LOG_EVERY=1
CHECKPOINT_EVERY=50  CHECKPOINT_KEEP=2  HF_EXPORT=1  RESUME=0
RUN_NAME=agentic_32b_lc140k_v33  MAX_STEPS=100000  WANDB_PROJECT=opd-v2-agentic
```

**Everything she did NOT set falls to the §1 defaults**, notably: `n_samples=4`, `temperature=1.0`,
`top_p=0.95`, `role_mix=22/44/20/14`, caps `6/2/4`, `MAX_STALENESS=0` (unbounded), and **all V34 loss
knobs off**. So her best run is **canonical β=1 reverse-KL OPD**, agentic producer, `length` admission
filter, 140k ctx / 128k gen, 32B student.

**`env_v33.sh` vs her config.** `training/docker/launch/env_v33.sh` resolves (via `make_config.py`) to
the same training-relevant `config.json` fields as her `run_agentic_mn_32b.sbatch`, except three intended
deltas:
1. `trainer.attn`: `olmo3_sink_fa3` → `olmo3_sink_fa2` (the B200 kernel swap; same math);
2. `trainer.checkpoint_every`: 50 → 25 (operational: more frequent durable checkpoints; does not touch the
   loss, optimizer, LR schedule or sampling);
3. `agentic.seed_source`: `ycchen/dsflash-proof-distill-v2-test` → `chankhavu/ycchen-dsflash-proof-distill-v2-test`
   (a mirror of the original, which is no longer public).

The remaining differing fields are cluster paths (`student_path`, `deploy_config_src`, `teacher_path`,
`problems_parquet`). Her sbatch pointed the trainer at the training-format checkpoint and the rollout at the
deploy-format one; the presets here use the deploy-format dir for both (transformers rebuilds
`rope_parameters` from the legacy `rope_scaling`). Her resolved `config.json` lived in
`ycchen/proof-pilot-datasets` → `step10-opd-pool-rollouts/agentic_32b_lc140k_v33/config.json` (no longer
public).

Notable non-default choices:
- `WEIGHT_SYNC_EVERY=4` (not the default 1) — with `TARGET_INFLIGHT=512` and long-CoT gen time, this
  sets the effective (unbounded but pipeline-limited) staleness.
- `CHUNK_SIZE=2048` (not 4096) — 32B `HID_DIM=4096` carries teacher_hidden+w_rot ~2GB more than
  soft_distill; chunk 2048 cancels it so `MICRO=131072` fits (~0–1GB margin at ~139.4GB).
- `MICRO=131072` must be ≥ `MAX_TRAJ_TOKENS=130816` (whole trajectory un-windowed).
- `RESUME=0` starts fresh; set `1` to continue after a requeue (fixed `RUN_NAME` → pool + DCP
  checkpoints persist across requeues).

**Prerequisites her sbatch documents** (run once): build the deploy variant
(`deploy/make_olmo3sink_deploy.py`) for rollout + weight-sync; pre-seed the pool
(`python -m opd_v2.agentic.seed --run-dir $RUN`) to avoid headless private-HF auth.

## 4. To reproduce her best run on the cu128/B200 image

Use `env_v33.sh`: it carries the §3 values **verbatim** apart from the deltas listed in §3. The two
B200 kernel substitutions are:
- `ATTN_IMPL=olmo3_sink_fa2` (trainer sink — B200 has no FA3);
- rollout `--attention-backend triton` (already the default in `run_rollout.sh`).

Everything else — β, LR, weight_sync cadence, admission filter, role mix, staleness — stays exactly
hers. The `config.py` **defaults are the 7B baseline** (`stage1-v2-7b`, `max_traj=65536`); the 32B run
is entirely the sbatch overrides.

## 5. Data & checkpoints — everything to reproduce or continue (all public)

OPD is on-policy: there is **no training corpus** — the student self-generates rollouts, scored live.
The complete input set is just models + the prompt seed + the config:

| what | HF id | notes |
|---|---|---|
| Teacher | `deepseek-ai/DeepSeek-V4-Flash` | frozen scorer; hidden dim 4096 |
| Prompt dataset (agentic seed) | `chankhavu/ycchen-dsflash-proof-distill-v2-test` | mirror of `ycchen/dsflash-proof-distill-v2-test` (no longer public); `per_problem` config: 1,776 hard AoPS/olympiad problems + DeepSeek proofs |
| **OPD start** (pre-OPD student) | `fieldsmodelorg/proof-pilot-checkpoints/step09-softdistill-v2test-32b` | = her `stage1-v2-32b-softdistill-v2test`; ~65 GB bf16 |
| **OPD final** (delivered 32B) | `fieldsmodelorg/proof-pilot-checkpoints/step10-opd-32b-s150` | her result after ~150 OPD steps; ~65 GB bf16 |
| Her resolved config, OPD pool + admitted rollouts | `ycchen/proof-pilot-datasets/step10-opd-pool-rollouts/` | **no longer public**; not needed to run (config = `env_v33.sh` §3, pool regenerates from the seed) |

- **Reproduce** her run → start from **step09** + teacher + dataset + `env_v33.sh`.
- **Continue training** → warm-start from **step10-opd-32b-s150** + same teacher/dataset/config.
- **Caveat — warm-start, not bit-exact resume.** The DCP checkpoint (fp32 optimizer + master weights +
  scheduler) is multi-TB and **not** published (only the bf16 consolidated weights are). OPD is
  async-nondeterministic (rollout requests carry no sampling seed; staleness and weight-sync timing depend
  on the pipeline), so a rerun reproduces the recipe statistically, not bit-for-bit — warm-start from the
  published weights (fresh optimizer) is the intended path.
- **Student format:** the deploy-format student `chankhavu/yccchen-olmo3-deploy` (legacy-rope, for
  rollout + weight-sync) works for both roles in these presets; step09/step10 are the training-format
  consolidated checkpoints. Convert training→deploy with `deploy/make_olmo3sink_deploy.py` if needed.
