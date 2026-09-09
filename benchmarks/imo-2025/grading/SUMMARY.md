# IMO 2025 — FM-Pochi-32B-IMO26 (step225), high budget

Grader: **GPT-5.6-sol** (xhigh reasoning, 1 grader/problem), using the
[MathArena](https://matharena.ai) `imo_2025` **checkpoint markschemes** (embedded in
`prompts.jsonl`, so the scores are reproducible). Rubric: complete-proof-by-any-method
= 7, wrong-answer = 0, else partial = checkpoints genuinely earned. Correct answers
verified against reference solutions
([Evan Chen](https://web.evanchen.cc/exams/IMO-2025-notes.pdf),
[Gemini Deep Think](https://storage.googleapis.com/deepmind-media/gemini/IMO_2025.pdf),
[OpenAI](https://github.com/aw31/openai-imo-2025-proofs)); each grade cross-checked by
independent Claude verifiers that computationally re-checked every finite claim,
coordinate identity, and game strategy. Scale 0–7.

## Per-problem

| Problem | Score /7 | Notes |
|---|--:|---|
| P1 | **7** | complete — sunny lines, answer k ∈ {0,1,3} |
| P2 | **7** | complete **coordinate-geometry** proof (a valid alternative to the synthetic solution; both OpenAI and Gemini also solved P2 this way) |
| P3 | **2** | construction (c ≥ 4) checkpoint earned; the upper-bound **crux** (the identity branch) is incomplete — "closed" by a false arithmetic step |
| P4 | **7** | complete — answer 6·12ᴷ·J with gcd(J,10)=1 |
| P5 | **7** | complete — inekoalaty game, threshold λ = 1/√2 |
| P6 | **0** | wrong answer (4048; correct **2112**), via a false "WLOG diagonal" reduction |
| **Total** | **30 / 42** | |

**30 / 42 → Silver** (2025 medal cutoffs: Bronze 19 / Silver 28 / Gold 35).

**Grading spread.** The score is method-dependent at two problems. **P2**: the
checkpoint-literal autograder scores the coordinate proof ~2 (it matches none of the
synthetic checkpoints), but it is a complete proof → 7. **P3**: ranges 0 (holistic:
the fabricated upper bound is fatal) to 4 (crediting the conditional tail); the
markscheme value — construction checkpoint only — is **2**, used here. P1/P5 have
one-line "trivially-fixable" blemishes some strict graders dock; we treat them as 7.
