# IMO 2025 — FM-Pochi-32B-IMO26 (step225), high budget

Evaluation of **FM-Pochi-32B-IMO26 (step225)** — a proof model trained from
`Olmo-3.1-32B-Think`; see [Models](../../README.md#models) — on the six **IMO 2025**
problems, each graded 0–7. The provided proofs are one rollout per problem at the
**high** inference budget of the generate–verify–refine harness.

Inference was run with the **high**-budget production config
[`config-model-step225-budget-high.yaml`](../../config-model-step225-budget-high.yaml)
— `step225` is this checkpoint's token in the preset filenames.

## Headline

| Problem | Score /7 | Result |
|---|--:|---|
| P1 | **7** | complete — sunny lines, k ∈ {0,1,3} |
| P2 | **7** | complete **coordinate-geometry** proof of the tangency |
| P3 | **2** | construction (c ≥ 4) only; upper-bound crux incomplete |
| P4 | **7** | complete — answer 6·12ᴷ·J, gcd(J,10)=1 |
| P5 | **7** | complete — game threshold λ = 1/√2 |
| P6 | **0** | wrong answer (4048; correct 2112) |
| **Total** | **30 / 42** | |

**30 / 42 → Silver** (2025 medal cutoffs: Bronze 19 / Silver 28 / Gold 35).

Graded by **GPT-5.6-sol** (xhigh reasoning); partial credit follows the
[MathArena](https://matharena.ai) `imo_2025` checkpoint markschemes.

## Grading methodology

Each proof was graded **0–7 by GPT-5.6-sol** (xhigh reasoning, 1 grader/problem) using the
[MathArena](https://matharena.ai) `imo_2025` **checkpoint markschemes**, which are embedded
in [`grading/prompts.jsonl`](grading/prompts.jsonl) — so the scores are **reproducible**:
run those prompts through GPT-5.6-sol (xhigh). The rubric (full text in
[`grading/prompt_template.md`](grading/prompt_template.md)):

- a **complete rigorous proof by any method** scores **7** — a fully-carried-out
  coordinate/analytic bash counts, even if it matches none of the synthetic checkpoints;
  minor non-load-bearing errors don't reduce it;
- otherwise, **partial credit** = the sum of the checkpoints genuinely earned (a valid
  construction earns its checkpoint even if the rest of the argument is incomplete or
  fabricated).

The correct answers were verified against reference solutions from
[Evan Chen](https://web.evanchen.cc/exams/IMO-2025-notes.pdf),
[Gemini Deep Think](https://storage.googleapis.com/deepmind-media/gemini/IMO_2025.pdf),
and [OpenAI](https://github.com/aw31/openai-imo-2025-proofs), and each grade was
**cross-checked by independent Claude verifiers** that re-derived every finite claim,
coordinate identity, and game strategy in Python.

## Reading the scores

- **P2 (coordinate proof).** The model solves P2 by a complete coordinate/analytic bash
  — a legitimate alternative that OpenAI and Gemini also used. A checkpoint-literal
  autograder under-scores it (~2, matching none of the *synthetic* checkpoints); it is a
  complete proof → **7**.
- **P3 (partial).** Right answer (c = 4) and a valid construction (c ≥ 4), but the hard
  upper-bound crux — the identity-function branch — is "closed" by a false arithmetic
  step. Only the construction checkpoint is earned → **2** (range 0–4 across graders).
- **P6 (wrong).** The proof reduces to a fixed diagonal permutation via an invalid
  "WLOG" (row/column permutations don't preserve rectangles), giving 4048; the correct
  answer is **2112** = ⌈n + 2√n − 3⌉. This was the hardest problem — neither OpenAI nor
  Gemini publicly solved it either.

## Contents

- `solutions.csv` — the 6 proofs with per-problem score and note
  (`Problem ID, score_0_7, note, proof`).
- `grading/SUMMARY.md` — the score table + notes.
- `grading/scores.json` — machine-readable per-problem scores + total.
- `grading/prompt_template.md` — the full grading rubric + prompt template.
- `grading/prompts.jsonl` — the exact `[system, user]` grading input per problem,
  **self-contained**: the MathArena checkpoint markscheme (CC-BY) is embedded, so
  re-running these 6 prompts through GPT-5.6-sol (xhigh) reproduces the scores.

## Caveats

- Scores are **LLM-graded** (GPT-5.6-sol + Claude cross-check), not human coordination;
  the P3 and one-line P1/P5 calls are the most judgment-dependent.
- One rollout per problem; generation variance is not captured.
