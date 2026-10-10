# OPD v2 — algorithm, environment, and rollout (faithful reference)

How Yi-Chia Chen's OPD v2 actually works, read out of the code (`training/opd_v2/`). This is the
training stage that produces the delivered **OPD-32B** proof model. Nothing here is a proposal — it
describes her code as-is, so we can port it to cu128/B200 without changing behavior.

**One-line summary:** a student (OLMo3-sink) generates proof trajectories on its *own* rollouts inside
a self-play artifact pool; a frozen **DeepSeek-V4-Flash teacher** scores each trajectory's hidden
states; the trainer minimizes a full-vocab **JSD(β)** between student and teacher over the student's
own tokens. It is **distillation, not RL** — there is no reward and no importance ratio.

## 1. Four decoupled processes

| process | role | code |
|---|---|---|
| **rollout** | student sglang server (fp8, `flash_rl/`), token-in / token-out | `data_plane/clients.py` |
| **teacher** | DeepSeek-V4-Flash sglang (TP4); `/score` writes hidden states to shared FS, returns a handle | `run_teacher_fs.sh`, teacher_extract patches |
| **trainer** | trainer-as-service: rank-0 HTTP ingress + all-rank command loop; FSDP2/HSDP + JSD | `trainer/` |
| **orchestrator** | CPU-async driver: fills the trajectory buffer, POSTs `/train_step`, drives weight sync | `orchestrator.py` |

Teacher hidden states move over a **shared filesystem**, never the control plane — the buffer stores
only `(ids, handle)`, never bytes. The trainer is a service the orchestrator drives.

## 2. The environment — a self-play artifact pool (agentic mode)

Active when `producer="agentic"` (the alternative, `single_round`, is plain prover-only OPD). It is a
**per-problem artifact graph** (`agentic/pool.py`), not a reward-bearing RL env:

```
problem ─▶ proofs ─▶ verifies          problem ─▶ refined          select: no node (sample only)
```

Every node carries provenance `(id, wv, source, step)` with `source ∈ {deepseek_seed, student}`.

**Four roles** (`agentic/roles.py`), each rendered through the student tokenizer (math_3r XML templates):
- **prove** — problem → proof (`<solution>` + `<self_eval>`).
- **verify** — problem + 1 proof → verification (`score ∈ {0, .5, 1}` + evaluation text).
- **refine** — problem + its proofs (ranked, top-4 bundled with verifier reviews) → refined proof.
- **select** — problem + its ≥2 refined → pick the best (training sample only; nothing consumes it).

Role dependencies are the environment "dynamics": a proof unlocks verify; a *verified* proof unlocks
refine; ≥2 refined unlock select (`pool.available_roles`).

**Curriculum control** (`agentic/sampler.py`) is **flow-balanced**, not fixed ratios:
- pick the role with the lowest `fill_fraction = student_count(role) / role_mix_weight(role)`
  (softmax over `−fill/temp`), so the four roles advance in lockstep toward `role_mix` (default
  `22/44/20/14` — verify = 2× prove because there are 2 verifies per proof, so no un-verified backlog);
- within a role, pick the **largest-deficit item** (least-verified proof, least-refined problem) to
  spread work.

**On-policy transfer:** per-problem caps (`max_proofs_per_problem=6`, `max_verifies_per_proof=2`,
`max_refined_per_problem=4`) count **only `student`-source** nodes — seed artifacts provide *context*
but never satisfy the student quota, so the pool drifts seed-dominated → student-dominated.
`prefer_student_context=True` further prefers student artifacts when assembling refine/select bundles.

**Key faithfulness note:** verifier scores are **curriculum signal, not reward** — they only
rank/bundle which proofs get refined. The training signal is purely the JSD to the teacher.

**Persistence:** append-only JSONL — `seed.jsonl` (immutable cold-start, DeepSeek `r3_hard2000`) +
`artifacts.jsonl` (student appends). Replayed on load → resume-safe. Cold-start seed:
`ycchen/dsflash-proof-distill-v2-test`.

## 3. How a rollout works — the `produce_sample` atom

The unit of work (`data_plane/produce.py`) is **"one rollout + one teacher score" as one fully
independent async coroutine.** N samples of a prompt = N independent atoms (`fan_out`); each finishes
and enters the buffer on its own — no gating the group on the slowest trajectory (this is what lowers
staleness). Steps:

1. **Prompt** — the sampler yields `Prompt(ids, meta)`: role + pool context → student chat template →
   `input_ids` (dropped if over `agentic.max_prompt_tokens`).
2. **Rollout slot** — acquire a student replica from a load-aware pool. Gen budget =
   `max_traj_tokens − prompt_len`, clamped by `max_new_tokens` (so `prompt+gen ≤ max_traj_tokens`).
3. **Generate** — `client.generate_one(ids, temperature/top_p/top_k, max_new, ignore_eos, timeout)`
   → the student sglang server (fp8; **triton** attention-sink on B200) does token-in/token-out,
   returning `gen_ids`, `wv` (the weight version that generated them), and `finish_reason`.
4. **Assemble** — `full = prompt.ids + gen_ids`, clamped to `max_traj_tokens`.
5. **Dump** (side channel) — `rollout_store` records **every** rollout (even ones later dropped /
   evicted / teacher-failed) for post-hoc analysis + DFlash draft training.
6. **Admission filter** — drop if `finish_reason ∈ drop_finish_reasons` (default `{length}`:
   window-truncated rollouts are the main source of OPD self-amplification, ~5.7%). A *drop* ≠ *fail*:
   valid generation deliberately excluded from the gradient; **does not** alter the sampling
   distribution (pure rejection filter).
7. **Pool write-back** (agentic) — parse the generation answer-only + validity-gate (`writeback.py`)
   and `admit_*` it into the pool as a new proof/verify/refined node → context for downstream roles.
   Happens *before* the teacher, independent of teacher success.
8. **Teacher score** — `store.new_path()` → `teacher.score(full, start=plen−1, out_path, wv)`: the
   teacher scores the trajectory, extracts hidden states over the generated span (from `plen−1`),
   **writes them to the shared FS**, returns a lightweight **handle** (not bytes).
9. **Return** `ScoredTrajectory(ids, prompt_len, wv, handle, meta, finish_reason)` → the buffer.

## 4. The training signal — full-vocab JSD distillation

The trainer (`trainer/`) reads the teacher hidden from FS, reconstructs full-vocab teacher logits via a
quantized-hidden `W_rot` codec, and computes a **chunked fused-linear JSD(β)** against the student's
logits over the student's own tokens (repo fp32-softmax kernel, not Liger). Canonical setting
`beta=1.0` = reverse-KL on-policy OPD; `0.5`=JSD, `0`=forward-KL. Whole trajectories are trained
un-windowed (packed varlen, `micro_batch_tokens`). FSDP2 / HSDP; fp32 master + bf16 compute; CPU
offload for 32B/long-context.

The **V34 loss package** (skew-KL base, routed top-K FKL, EOS/tail reweight) exists in `LossCfg` but
**all knobs default to 0/off = bit-identical to naïve β-OPD**; her best run does not use it.

## 5. Weight sync & staleness (her design — kept faithfully)

**Weight sync** (`orchestrator.weight_sync`): every `weight_sync_every` steps the trainer saves a
checkpoint and each rollout replica reloads it. Her design, restored on our branch:
- `pause_generation("in_place")` — in-flight generations **keep running** across the swap;
- `update_weights_from_disk(flush_cache=False)` — old-weight KV is **not** flushed ("under in_place a
  failed flush asserts and kills the scheduler");
- best-effort: count successful reloads, advance `weight_version` even on partial failure.

So a trajectory can span two weight versions and reuse old-weight KV — deliberate **semi-on-policy**
staleness, in line with async-RL practice. (A stricter abort/flush fail-closed *barrier* variant was
prototyped locally; it changes this behavior and was reverted to stay faithful — preserved on branch
`wip/weight-sync-barrier` as an explicit opt-in. See [OPD_V2_PARITY_STATUS.md](OPD_V2_PARITY_STATUS.md).)

**Staleness bound** (`buffer.is_stale`): `max_staleness=0 = disabled` by default → **nothing is ever
dropped for age.** OPD has no importance ratio (it doesn't store generation logprobs) and the teacher
target is frozen, so a rollout from an old `wv` is equally valid distillation data — unbounded
staleness is *theoretically* fine here, a step past bounded-staleness async-RL. The **effective**
staleness is set by pipeline depth (`target_inflight`, gen time, `weight_sync_every`), not by a drop
threshold. Set `MAX_STALENESS=N>0` to enforce `drop iff cur_step − wv > N`.

## 6. Data flow (end to end)

```
PoolSampler ─prompt→ produce_sample ─generate→ student(sglang fp8, triton sink)
   ↑ writeback (answer-only, validity gate)         │ full = prompt+gen, admission filter
   └──────────── pool (problem→proof→verify/refine) │
                                                     ▼ teacher.score(full, start=plen−1) → hidden→FS, handle
                          buffer (ids+handle) ◀──────┘
                                │  orchestrator POST /train_step (ids + handles)
                                ▼
   trainer: FS hidden → W_rot → teacher logits → chunked fused-linear JSD(β) vs student → FSDP2/HSDP step
                                │  every weight_sync_every steps
                                ▼  weight_sync → rollout replicas (in_place, no-flush, semi-on-policy)
```

See [OPD_V2_CONFIG_REFERENCE.md](OPD_V2_CONFIG_REFERENCE.md) for every knob and her best-run values.
