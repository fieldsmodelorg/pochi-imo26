# IMO 2026 model submissions - graded artefacts

Model-generated solutions to IMO 2026 Problems 1-6, independently graded. This
folder collects the graded artefacts for public sharing.

## Contents

```
solutions/   # per-problem solution text, verbatim raw model output
             #   <submission>/<submission>_problem_<n>.txt
pdfs/        # typeset PDF of each full submission
raw/         # the original submission CSVs (id, proof) as downloaded
grading/     # assessment prompt template, prompt-builder script, and the
             #   official IMO 2026 score distribution
scores.csv   # machine-readable per-problem scores (with uncertainty notes)
llm_partial_marks_report.md
             # spot check of the LLM-awarded partial marks in deedy/imo-2026
README.md    # this file
```

## Submissions

| Submission | Model | Problems | Budget | Source |
| --- | --- | --- | --- | --- |
| `imo2026-deploy-budget-high-tournament` | FM-Pochi-32B-ProofPilot (deploy) | 1-6 | high | [submission.csv](raw/imo2026-deploy-budget-high-tournament_submission.csv) |
| `imo2026-step225-budget-high-tournament` | FM-Pochi-32B-IMO26 (step225) | 1-6 | high | [submission.csv](raw/imo2026-step225-budget-high-tournament_submission.csv) |
| `step225_run2` (rerun) | FM-Pochi-32B-IMO26 (step225) | 1, 4, 5 | xhigh | [submission.csv](raw/step225_run2_submission.csv) |
| `step225_run3` (rerun) | FM-Pochi-32B-IMO26 (step225) | 5 | xhigh | [submission.csv](raw/step225_run3_submission.csv) |

Both are checkpoints of the same 32B proof model, run through the same
generate–verify–refine harness — see [Models](../README.md#models). The budget
column names a [production preset](../README.md#production-configs): the runs
above used `config-model-{deploy,step225}-budget-{high,xhigh}.yaml`.

## Grading methodology

The submissions have been graded according to broad IMO standards. We've not had
access to the official IMO markschemes (they aren't made public), so we can't
grade against them - where this creates uncertainty it's flagged explicitly
below and in `scores.csv`.

The IMO 2026 medal boundaries were: **Bronze 16, Silver 23, Gold 29** (out
of 42).

### Process

Grading was twofold:

1. An expert human (two time IMO Problem Selection Committee member) read and assessed every solution in
   full.
2. As a cross-check, each per-problem solution was also run past **GPT 5.6 Sol
   Pro** via ChatGPT with a structured assessment prompt, to make sure nothing
   had been missed.

The final marks are the human's - the LLM pass was a safety net rather than a
grader. Each prompt pairs the official problem statement with the model's
verbatim output and asks for errors and omissions to be classified as 'major'
(undermining the validity of the proof) or 'minor' (fixable by a routine
clarification), with no negative marking. The template is in
[`grading/assessment_prompt_template.txt`](grading/assessment_prompt_template.txt)
and the (rough) script that builds one prompt per problem per submission is
[`grading/build_prompts.py`](grading/build_prompts.py).

### Why the marks are almost all 0 or 7

Whether a solution is complete and correct is a reasonably objective question,
so that's what we've marked on: 7 for a complete solution, 0 otherwise. Without
the markschemes, any intermediate mark is guesswork.

The exception is the 5 awarded to FM-Pochi-32B-ProofPilot (deploy) on P5. We've
kept it because it doesn't affect the medal colour either way, and because
there's some evidence it's the mark that would actually have been awarded: 75 of
the 666 contestants scored exactly 5 on P5 (against only 26 at 3 and 14 at 4),
which looks like a standard deduction landing at 5 rather than graders spreading
marks evenly, and some anecdotal discussions we've had point the same way. The
uncertainty is flagged in the footnote below and in `scores.csv`.

For reference, the full IMO 2026 score distribution (from
[imo-official.org](https://www.imo-official.org/results/individual/year/2026/),
also in
[`grading/imo2026_score_distribution.csv`](grading/imo2026_score_distribution.csv)):

| Problem | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | 65 | 26 | 7 | 46 | 47 | 11 | 21 | 443 |
| P2 | 288 | 183 | 54 | 26 | 13 | 0 | 2 | 100 |
| P3 | 507 | 86 | 7 | 9 | 39 | 2 | 3 | 13 |
| P4 | 39 | 81 | 45 | 10 | 18 | 75 | 59 | 339 |
| P5 | 151 | 132 | 105 | 26 | 14 | 75 | 12 | 151 |
| P6 | 557 | 73 | 8 | 4 | 0 | 0 | 1 | 23 |

### A caution on grading purely with LLMs

A lot of published 'IMO scores' for models are based entirely on LLM grading.
In our experience this needs a lot of care: LLM graders tend to be too generous
at the lower end, and can occasionally be too harsh at the top.

The lower-end problem isn't that partial marks are rare - the table above shows
2s and 3s are awarded not that infrequently - it's that without the markscheme
we don't know what they're awarded *for*. Partial credit at the IMO is very
non-linear and its shape differs a lot from problem to problem: P2 partials
cluster at 1-2 with essentially nothing at 4-6, P4 spikes at 5-6, and P6
partial credit is almost entirely a single mark of 1. The bar is often higher
than an LLM assumes, so it awards credit too early - a model that reproduces
half of a solution will happily be given 2 or 3, when it may be the much easier
half and actually worth 0 or 1. In practice LLM graders invent their own
partial-progress criteria and mark against those instead.

We saw this directly in our runs: ChatGPT leaned towards 3/7 for some of the
computational geometry solutions on P2, which would certainly score 0 at the
IMO - computational work on geometry earns no partial credit unless the intermediate
results are interpreted synthetically.

So: judging whether a solution is complete is reasonably objective (though it
does depend on the strength of the grading model), but LLM-awarded partial
marks like 1/7 or 3/7, given without access to the markscheme, are guesswork.
They should be treated with extreme scepticism, and they tend to inflate
totals.

We've written this up in more detail in
[`llm_partial_marks_report.md`](llm_partial_marks_report.md) - a spot check of
the LLM-awarded partial marks in one published comparison
([deedy/imo-2026](https://github.com/deedy/imo-2026)), with concrete examples
of marks that would be zeros at the actual IMO.

## Scores

We evaluated **FM-Pochi-32B-ProofPilot (deploy)** and **FM-Pochi-32B-IMO26
(step225)**, both at the high budget, on the full set of problems. There were
lots of technical details in P2, P3 and P6 on which the models could fail, which
made it unlikely that an LLM could make sufficient progress on a further run, so
we skipped them in the reruns.

| Model | Budget | P1 | P2 | P3 | P4 | P5 | P6 | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FM-Pochi-32B-ProofPilot (deploy) | high | 7 | 0 | 0 | 7 | 5\* | 0† | **19** |
| FM-Pochi-32B-IMO26 (step225) | high | 7 | 0 | 0 | 7 | 7 | 0† | **21** |
| FM-Pochi-32B-IMO26 (step225), `step225_run2` | xhigh | 7 | - | - | 7 | 7 | - | **21** |

Additionally, we re-evaluated `step225_run3` a third time on problem P5, on
which it scored 7.

\* **P5, FM-Pochi-32B-ProofPilot (deploy)** - defaulted to 5 in the total.
Without access to the official markscheme this score is uncertain: it could
reasonably be as low as 2, though it is very unlikely to be lower given the
score distribution on the actual IMO. Even at 2, the total (16) would still be a
Bronze-medal score.

† **P6** - both the FM-Pochi-32B-ProofPilot (deploy) and FM-Pochi-32B-IMO26
(step225) submissions are probable 0s, but each contains some useful work, so
there is a small chance either could score 1.

Plausible ranges: FM-Pochi-32B-ProofPilot (deploy) 16-20, FM-Pochi-32B-IMO26
(step225) 21-22. Both reach a Bronze-medal score, with FM-Pochi-32B-IMO26
(step225) close to (but not reaching) Silver.

## Per-problem commentary

We analyse **FM-Pochi-32B-ProofPilot (deploy)** and **FM-Pochi-32B-IMO26
(step225)**, both at the high budget.

- **P1** - Both main submissions are complete solutions: 7 each.
- **P2** - Both score 0. FM-Pochi-32B-ProofPilot (deploy) mis-converts one of
  the angle relations into complex numbers at the outset. FM-Pochi-32B-IMO26
  (step225) is correct for longer, but the algebra simplifying \(BC - AD = 0\)
  is invalid. Computational approaches to geometry problems are very rarely
  awarded partial credit unless the intermediate results are interpreted
  synthetically (i.e. back in terms of the diagram rather than pure algebra), so
  both would certainly score 0.
- **P3** - Both score 0. Neither submission goes beyond the \(n = 1\) case; both
  give the wrong answer, with no progress towards the optimal bound. The
  \(n = 1\) case is not credit-worthy at the IMO on a hard problem.
- **P4** - Both are complete solutions: 7 each. The FM-Pochi-32B-IMO26 (step225)
  write-up is hard to follow in places, but the IMO applies no style penalties
  or negative marking.
- **P5** - The FM-Pochi-32B-IMO26 (step225) solution is complete: 7. The
  FM-Pochi-32B-ProofPilot (deploy) solution fails to handle the possibility
  \(f(x) - x \in \{0, c\}\) for some constant \(c \ge 0\), and its Case 1
  argument is incorrect (the claimed inequality fails, e.g., for \(a = 10\),
  \(b = 2\), \(c = 1\)). Without the official markscheme the penalty for this is
  uncertain: the score would very likely be in the range 2-5, with 5 the most
  likely.
- **P6** - Both submissions contain fundamental errors and are incomplete,
  though each has some useful ideas — particularly FM-Pochi-32B-IMO26 (step225).
  The bar for a partial mark on a hard problem is usually reasonably high, but
  73 contestants earned one this year, so there is a small chance of a 1. The
  most likely mark for both is 0.
- **xhigh reruns** - `step225_run2` (P1, P4, P5) and `step225_run3` (P5) were
  also reviewed: all are complete solutions, 7 each.

## Reproducibility

The PDFs in `pdfs/` were typeset from the raw CSVs (Markdown embedding LaTeX
math) via pandoc and `pdflatex`. The per-problem files in `solutions/` are the
verbatim raw model output, extracted unchanged from the `proof` column of the
CSVs in `raw/`.
