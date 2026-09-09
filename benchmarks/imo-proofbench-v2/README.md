# IMO-ProofBench-V2 — FM-Pochi-32B-IMO26 (step225), medium budget

Evaluation of **FM-Pochi-32B-IMO26 (step225)** on
[IMO-ProofBench-V2](https://github.com/google-deepmind/superhuman/tree/main/imobench)
(Luong et al., 2025, [arXiv:2511.01846](https://arxiv.org/abs/2511.01846)) — 60
olympiad problems (30 Basic + 30 Advanced), each graded 0–7. The model is a
proof model trained from `Olmo-3.1-32B-Think`; see
[Models](../../README.md#models).

Inference was run with the **medium**-budget production config
[`config-model-step225-budget-medium.yaml`](../../config-model-step225-budget-medium.yaml)
— `step225` is this checkpoint's token in the preset filenames.

## Headline

| Subset | Problems | Mean / 7 | % of max |
|---|--:|--:|--:|
| **Overall** | 60 | **4.65** | **66.4%** |
| Basic | 30 | 6.45 | 92.1% |
| Advanced | 30 | 2.84 | 40.6% |

### By level

| Level | Problems | Mean / 7 | % of max |
|---|--:|--:|--:|
| pre-IMO | 8 | 6.97 | 99.6% |
| IMO-easy | 24 | 6.54 | 93.5% |
| IMO-medium | 18 | 3.56 | 50.8% |
| IMO-hard | 10 | 0.20 | 2.9% |

### By category

| Category | Problems | Mean / 7 | % of max |
|---|--:|--:|--:|
| Algebra | 16 | 5.58 | 79.7% |
| Number theory | 14 | 4.57 | 65.3% |
| Geometry | 14 | 4.27 | 61.0% |
| Combinatorics | 16 | 4.11 | 58.7% |

Scores decline monotonically with difficulty (near-ceiling on pre-IMO/easy, a
cliff at IMO-hard), and the model is strongest in Algebra, weakest in
Combinatorics.

## Grading methodology

Graded with **GPT-5.6-sol** as the autograder, following the ProofAutoGrader
prompt from arXiv:2511.01846 **Appendix B.5** verbatim (see
`grading/prompt_template.md`):

- **Scale 0–7**, with the paper's coarse rubric anchors: **7** Correct · **6**
  Almost · **1** Partial (a single point, only for guideline-named milestones) ·
  **0** Incorrect. Partial credit is *not* a graded 1–5 band — this is
  deliberate and is what makes the Advanced numbers strict.
- The grader is given the **problem, the model's proof, a reference solution,
  and the problem-specific grading guidelines**; it emits free-text analysis
  ending in a single `<points>N out of 7</points>` block.
- **4 graders per problem**, arithmetic mean. Reasoning effort **high**.

**Grader stability.** The autograder is near-deterministic: 57/60 problems get a
unanimous score across the 4 judgments, all 60 within 1 point. A separate
single-grader **xhigh** pass reproduced every problem's score exactly (identical
66.4% overall), so the number is invariant to grader count and reasoning tier.
Note this measures grader *self-consistency*, not correctness against a human.

## Contents

- `solutions.csv` — the 60 model proofs with their per-problem score
  (`Problem ID, subset, category, level, score_0_7, proof`).
- `grading/SUMMARY.md` — the tables above.
- `grading/scores.json` — machine-readable aggregates + per-problem means.
- `grading/prompt_template.md` — the exact system + user prompt template and run
  config.
- `grading/prompts.jsonl` — one line per problem (`{problem_id, messages}`): the
  exact `[system, user]` input sent to the grader, embedding the problem,
  reference solution, and grading guidelines (reproducibility).

The benchmark problems, reference solutions, and grading guidelines are from
[google-deepmind/superhuman/imobench/proofbench_v2.csv](https://github.com/google-deepmind/superhuman/blob/main/imobench/proofbench_v2.csv)
(CC-BY 4.0).

## Caveats

- Numbers are the **GPT-5.6-sol autograder**, not verified human grades. The
  Basic/Advanced and by-difficulty *patterns* are reliable; individual Advanced
  scores may be over- or under-credited and warrant spot-checking.
- One rollout per problem (medium budget); generation variance is not captured.
