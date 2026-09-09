# Degenerate-loop filtering (gzip-based) + server watchdog

How this harness detects and drops **degenerate generations** — outputs stuck in
a repetition or runaway-enumeration loop — and why.

**One-line summary:** a proof/verify generation that falls into a loop is detected
by the **gzip compression ratio** of its text (loops compress far more than real
reasoning) using fixed thresholds,
and is stopped two ways: **live** while streaming (`search.stream_detect`, default
on — abort the request and salvage a proof from the clean prefix) and **post-hoc**
as a backstop (`search.filter_degenerate`, default on — never pool/refine/score a
degenerate output). A separate `server.watchdog_timeout` bump keeps a long
generation from crashing the server. All three are per-config switchable.

---

## 1. Why this exists — the crash

A run of the FM-Pochi-32B-IMO26 (step225) checkpoint **crashed mid-P6**: the
SGLang server SIGQUIT'd itself (exit -3). Root cause from the node's server log:

- A **verifier generation ran away to 131,484 tokens over ~880 s** in a degenerate
  enumeration loop (`… gcd(594,102)=6, gcd(594,105)=3, gcd(594,108)=54, …` for
  hundreds of cases).
- DFlash speculative-decode acceptance collapsed (~0.1–0.4 — the draft can't
  predict a runaway), so throughput fell to **~40 tok/s**.
- A scheduler forward pass then made **no progress for >300 s**, so SGLang's
  **watchdog (`watchdog_timeout=300`, the default) killed the whole server.**

It was **not** OOM (KV-cache usage was 8%) and **not** a CUDA fault. The
`py-spy … Permission Denied` lines in the log are the watchdog's *failed attempt*
to dump the hung stack — a symptom, not the cause. The deeper problem is that the
model **degenerates into loops on hard problems**, and nothing stopped it.

## 2. Two fixes, both config-gated

| Fix | Knob | Effect |
|---|---|---|
| **Watchdog headroom** | `server.watchdog_timeout: 1200` (→ `--watchdog-timeout`) | A legitimately slow long forward no longer trips the watchdog. The same 880 s runaway would *complete* instead of killing the server. Prevents the crash directly. |
| **Streaming abort + salvage** | `search.stream_detect: true` | Detects the loop **live** and aborts the request, salvaging a proof from the clean prefix. Reclaims the wasted compute and prevents the stall at the source (§7). |
| **Post-hoc filter** | `search.filter_degenerate: true` | Backstop on finished text: degenerate output never pools / seeds a refine / scores a proof (§5). |

Independent and composable: the watchdog stops the *crash*; streaming stops the
*runaway as it happens*; the post-hoc filter is the *backstop* for anything that
slips through.

## 3. The signal: gzip, not a repetition penalty

The detector's signal is the **zlib/gzip compression ratio** of the text:
`ratio = len(zlib.compress(bytes, level=6)) / len(bytes)` (**compressed / raw**;
**lower = more repetitive**). LZ77 collapses repeated substrings, so a window
stuck in a loop compresses to ~0 while genuine varied reasoning stays ~0.3. It is
cheap (~tens of µs on a 12 KB window), language-agnostic, and model-free.

**Why not a repetition penalty.** Two reasons:
1. **Math reasoning is inherently repetitive** — enumerations, re-derivations,
   `a₁=…, a₂=…`. A repetition penalty would corrupt exactly the reasoning we want.
2. **A penalty warps the sampling distribution**; aborting/rejecting a doomed
   generation only *truncates* it and does **not** change the distribution, so it
   is safe on on-policy OPD rollouts. (DFlash also cannot apply penalties —
   `patch_dflash_sampling.py` — the penalty state breaks speculative decoding.)

Gzip-reject is the distribution-neutral tool; a penalty is the wrong one.

## 4. The two detectors (`evaluation/harness/loop_detect.py`)

`is_degenerate(text)` is `True` if **either** detector fires. Thresholds are taken
**verbatim** from the reference v2 loop detectors (`zlib_runaway_detector` +
`loopguard`).

**Tier 1 — zlib sliding window** (`zlib_runaway`): the primary, general detector.

| const | value | role |
|---|---|---|
| `WINDOW_CHARS` | 12,000 | window scanned |
| `STEP_CHARS` | 1,000 | re-check every 1 k chars |
| `HARD_RATIO` | **0.05** | ratio below → degenerate immediately (hard token loop) |
| `SOFT_RATIO` | **0.18** | ratio below… |
| `SOFT_PERSIST` | **20** | …for ≥ 20 consecutive checks → degenerate |

The **SOFT persistence requirement is what spares legitimate long math**: a real
enumeration dips below 0.18 but recovers within a few checks; a true loop stays
sub-0.18 for 60+.

**Tier 2 — loopguard local-density backstop** (`loopguard_degenerate`): catches a
verbatim segment repeated densely, even inside otherwise-varied text.

| const | value | role |
|---|---|---|
| `LG_CHUNK` | 25 | a 25-char verbatim segment… |
| `LG_STEP` | 5 | sampled every 5 chars |
| `LG_THRESHOLD` | **8** | …recurring > 8× → degenerate |
| `LG_SPAN` | 1,500 | …within a 1,500-char window |

Calibrated (measurements on OPD traces): genuine small-case enumeration
tops out at ~4 dense recurrences; real loops sit at 20+. `threshold=8` is a 2×
safety margin below genuine.

## 5. Where it runs

Applied **post-hoc** on the finished text (`reasoning_content + "\n" + content`)
at the two choke points in `proof_search.py`, both gated by
`search.filter_degenerate`:

- **`_admit_candidate`** — a degenerate proof is rejected, so it never enters the
  pool or seeds a refinement. (Truncated generations, `finish_reason != "stop"`,
  are already dropped separately.)
- **`_verify_proof`** — a degenerate verifier output is treated as invalid and
  its parsed score is discarded, so a looping verifier can't pollute a proof's
  `mean_score`.

## 6. Config flags (how to disable)

Two independent booleans under `search:`, both **default `true`**, both present and
commented in **every** shipped config and type-validated by the strict schema
(`eval_config.py`):

| flag | default | disables |
|---|---|---|
| `filter_degenerate` | `true` | the **post-hoc** filter — the admit-side + verify-side checks (§5) |
| `stream_detect` | `true` | the **streaming** live-abort + salvage (§7); calls fall back to plain blocking |

```yaml
search:
  filter_degenerate: true   # post-hoc backstop (keep on)
  stream_detect: true       # real-time abort+salvage; false = blocking calls only
```

They compose: with both on, streaming aborts most loops live and the post-hoc
filter is the backstop. Turning `stream_detect` off keeps the (validated) blocking
path + post-hoc filter. Turning `filter_degenerate` off removes *all* degenerate
handling. The server knob `server.watchdog_timeout` is likewise exposed and
commented in every config. Setting `filter_degenerate: false` removes all
degenerate handling (no loop filter at all).

## 7. Real-time streaming detection + salvage (`search.stream_detect`)

The post-hoc filter (§5) keeps degenerate output out of results but can't stop a
runaway *while it generates* — the wasted compute (and the stall risk the watchdog
covers) remain. `search.stream_detect` (default **on**) adds the live path:

- **`async_client.chat_stream`** streams `/chat/completions` (`stream: true`,
  `stream_options.include_usage`), feeds each delta to a `RunawayDetector`
  (`loop_detect`, the same thresholds), and the moment it aborts, closes the stream
  (SGLang aborts the request on client disconnect) and POSTs `/abort_request`
  (belt-and-suspenders).
- **Salvage** (`_salvage_stream_loop`): recover a proof from the clean pre-loop
  prefix instead of discarding the whole call. Cut at a verbatim `find_loop_cut`
  where there is one, else `loop_onset` — a **length-relative** zlib estimate (the
  recent window + sustained-soft run at the tail of *this* stream; it must not use
  the global `verdict.position`, which counts reasoning+content and would over-cut).
  Then:
  - loop in the **content** (a `<solution>` body was already written): keep that
    clean prefix and **force-close from it** so its closing `</solution>…<score>`
    gets written — a bare truncation has no `<score>` and would fail to parse.
  - loop in the **reasoning**: drop the looping tail and force-close a *fresh*
    solution.
  Both reuse the existing `/generate` continuation; a force-close that itself loops
  is cut again (guard). (The content path was hardened per the audit — §10.)
- Wired in **`CallStore.perform`**: when `stream_detect` is on, generations and
  verifications go through `chat_stream` instead of `chat_raw`; the length-
  continuation logic is unchanged. A **degenerate verifier** is folded into the
  disposition here (`skipped_degenerate`, not `accepted`) so `mean_score`, the
  `_verify_proof` filter, and the `final.json` valid/invalid tally read one
  consistent source.

The **post-hoc `filter_degenerate` still runs as the backstop** with streaming on
(defense in depth: a loop that completes inside a single window, or a salvage that
slips through, is still caught before it can score a proof).

> ✅ **Live-server smoke-tested (P1 + P6, both determinism modes).** Validated with
> `stream_detect: true` under both deterministic and nondeterministic inference:
> **P1 solves (self=1.0)** — byte-identical to blocking under deterministic inference,
> same outcome nondeterministic; **P6 aborts + salvages 40+ loops with zero server
> crashes** (the exact problem that crashed the blocking run); streamed proofs parse.
> Fallback if ever needed: `stream_detect: false` → the blocking path + post-hoc
> `filter_degenerate` + `watchdog_timeout`.

## 8. Provenance

Ported from the Proof-Pilot v2 loop detectors: `zlib_runaway_detector` (Tier 1)
and `loopguard` (Tier 2). All constants and the two-tier decision logic are kept
verbatim; only the invocation site differs (post-hoc `is_degenerate()` here vs the
original streaming `feed()` + selection-time `degenerate()`). Tests:
`tests/test_loop_detect.py`.

## 9. Measured stats — scale & false-positive check

Detector replayed over **all 29,455 model outputs** from the three IMO-2026 runs
— FM-Pochi-32B-ProofPilot (deploy), FM-Pochi-32B-IMO26 (step225), and an
earlier step-125 checkpoint from the same training run (not published). These
are *measurements on real production output*, not synthetic.

**Overall:** **244 / 29,455 flagged degenerate (0.83 %)**.

**False-positive check — flag rate by problem × type** (P1 is the easy, clean
problem; any flag there would be a likely false positive):

| problem / type | flagged / total | rate |
|---|---|---|
| **P1 / gen** | **0 / 768** | **0 %** ✅ |
| P2 / gen | 52 / 384 | 14 % |
| P3 / gen | 19 / 384 | 5 % |
| P4 / gen | 3 / 256 | 1 % |
| P5 / gen | 20 / 320 | 6 % |
| **P6 / gen** | **138 / 352** | **39 %** |
| P3 / verify | 5 / 6,080 | 0.08 % |
| P4 / verify | 1 / 4,064 | 0.02 % |
| P5 / verify | 1 / 5,072 | 0.02 % |
| P6 / verify | 5 / 4,783 | 0.10 % |

**Zero flags on P1**, and flags concentrate exactly on the hard problems where the
model is *known* to degenerate (P6 generations at 39 %, the same problem that
crashed the server). This is the signature of a filter catching real pathology,
not clean work. The clean 106 k-token P5 proof was **not** flagged; the 131 k
crash-trigger verifier **was**.

**Detector attribution** (of the 244): zlib (Tier 1) fired on **234**, loopguard
(Tier 2) on **128** — **116 zlib-only, 118 both, and only 10 loopguard-only**. The
10 loopguard-only cases (the softest, most-borderline signal) were every one a
200 k+ char generation on a hard problem — extreme-tail output that is low-quality
regardless.

**Repetitiveness separation** (raw/compressed ratio, *higher* = more repetitive):
clean (unflagged, n=29,211) median **3.51**, p99 4.66, max 16.67; flagged (n=244)
median **7.84**, max **72.83** (pure `1, 1, 1, …` loops). Note the ranges *overlap*
at the edges (clean max 16.67 > flagged min 3.59) — **because the detector uses a
sliding window, not the whole-text ratio.** A clean proof can be uniformly, mildly
compressible (repeated LaTeX/notation) yet never have a *local* loop; a flagged
output can be varied overall but contain a dense loop burst that a 12 k window
catches. This divergence is exactly why the detector is windowed rather than
a single whole-text gzip check.

**Scale — compute lost to degeneracy.** Counting *generation* tokens (the
expensive part), **23.2 % of all generation compute — 30.5 M of 131.6 M tokens —
went to degenerate output** now discarded by the filter. It is wildly uneven by
problem:

| problem | degenerate gen tokens | % of that problem's gen tokens |
|---|---|---|
| **P6** | 18.5 M | **53.3 %** |
| P2 | 6.7 M | 21.6 % |
| P5 | 2.6 M | 13.4 % |
| P3 | 2.4 M | 7.9 % |
| P4 | 0.3 M | 2.1 % |
| P1 | 0 | 0 % |

Over **half of P6's generation budget was spent looping** — which is both why P6
never solves and why it was the problem that crashed the server. Verifiers loop far
less often (≤ 0.1 % of verify calls) but cost ~1.5 M tokens more, and a *single*
131 k-token verifier loop is what stalled the scheduler. Post-hoc filtering stops
this output from polluting results; a streaming early-abort (§7) would additionally
*reclaim* this compute.

**Takeaway:** on real data the filter has **effectively zero false positives on
normal output** and cleanly separates degenerate loops. The thresholds hold up on
our traces unchanged.

## 10. Adversarial audit & fixes

A 6-dimension adversarial council (loop detector, SSE client, salvage, `proof_search`
wiring, config validation, determinism-removal safety), each finding then verified by
a skeptic, found **no correctness or crash bug** — streaming reassembly, the abort
path, the filter gates, and the nondeterministic-DFlash change all held up. Two real
issues were fixed (commit `67a25c8`):

- **① (medium) — salvage discarded a semantic content-loop.** When a loop tripped
  mid-`<solution>` but was *non-verbatim* (zlib soft tier, so `find_loop_cut` returned
  None), the old code fell through to a reasoning-only force-close — and because
  `loop_onset` used the global `verdict.position`, it clamped to 0, force-closing from
  *empty* reasoning and throwing the good proof away. This was the mechanism behind the
  ~47 % P6 salvage rate. **Fix:** keep the clean `<solution>` prefix (via `loop_onset`
  when there's no verbatim cut) and force-close *from it*; `loop_onset` is now
  length-relative so it never clamps to 0 for a single stream (§7).
- **③ (low) — verifier tally inconsistency.** A parseable-but-degenerate verifier was
  dropped from `mean_score` but still counted `accepted` in `final.json`'s
  valid/invalid tally. **Fix:** the degeneracy check moved into `perform`'s disposition
  (`skipped_degenerate`), so all three consumers agree.

Regression tests for both: `tests/test_stream_detect.py` (verbatim + semantic
content-loop force-close), `tests/test_loop_detect.py` (`loop_onset` clamping),
`tests/test_proof_search.py` (`skipped_degenerate` disposition).
