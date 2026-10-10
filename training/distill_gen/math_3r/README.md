# math_3r — prompt/parse/rank/bundle modules for the agentic OPD roles

Trimmed from Yi-Chia Chen's `distill_gen/math_3r` (the multi-agent proof-data generator). Only the pure
modules that OPD v2's agentic producer imports are kept (`opd_v2/agentic/{roles,seed,writeback}.py`):

| file | purpose |
|------|---------|
| `prompts.py` | Template loading + render + system/user split. Templates are read from the repo's shared `evaluation/prompts/ycchen_math_3r/` (`{prover,verifier,refiner,selector,fallback}.txt`) — the same files the inference harness serves, so training and deployment prompts are byte-identical. |
| `parser.py` | Dataclasses + XML-tag parsing (`<solution>`, `<self_eval>`, `<score>`, `<evaluation>`, `<selected_id>`) + validity checks. |
| `rank.py` | Candidate ranking (verifier mean/min, self_score, length). |
| `bundle.py` | Builds refine/select bundles (top-4, token-budget truncation). |
| `clean.py` | Deterministic cleaning (strips self-eval/boxed/XML/meta) + fallback. |

The generation pipeline itself (`pipeline.py`, `run.py`, `gen_refsel.py`, `select_hard.py`, `make_report.py`)
and its problem parquets are not included: they drive LLM backends to build the distillation data, not
training. The resulting data is on HF at `chankhavu/ycchen-dsflash-proof-distill-v2-test` (a mirror of
`ycchen/dsflash-proof-distill-v2-test`, which is no longer public), which is what OPD seeds from.
