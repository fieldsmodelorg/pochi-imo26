# scripts — data-pipeline utilities

Standalone scripts for converting, verifying, and building the training-data corpora. They import the
shared `train_core` library (repo root is resolved from `__file__`).

| script | purpose |
|--------|---------|
| `jsonl_to_parquet.py` | Convert raw Nemotron-SFT-Math-v3 JSONL to hive-partitioned Parquet (parallel byte-range workers). |
| `verify_parquet.py` | Full byte-exact (blake2b) row-by-row verification of the Parquet conversion. |
| `cascade2_jsonl_to_parquet.py` | Convert the Nemotron-Cascade-2 DeepSeek subset to Parquet. |
| `cascade2_robust_convert.py` | Robust variant of the Cascade-2 conversion (handles upstream data flaws). |
| `cascade2_verify_parquet.py` | Verify the Cascade-2 Parquet conversion. |
| `td_normalize.py` | Normalize sources into the tokenizer-agnostic unified **L2** schema (OpenAI-style messages). |
| `td_build_l2.py` | Materialize the unified L2 dataset (streaming three-way self-verification + retry). |
| `td_build_l2_v2.py` | Materialize **L2-v2** (DeepSeek-V4-Pro generation: math-v4 / proofs-v2 / science-v2 / agentic-v2). |
| `td_upload_hf.py` | Publish an L2 dataset to the Hugging Face Hub with post-upload integrity checks. |
| `build_distill_prompts.py` | Build the distillation prompt set. |
| `tok_overlap.py` | Byte-overlap analysis between donor and base tokenizer vocabularies. |
| `number_tok.py` | Compare how the two tokenizers split numbers. |
| `number_tok_exhaustive.py` | Exhaustive number-splitting comparison. |

## Usage

```bash
uv run python scripts/jsonl_to_parquet.py --help
uv run python scripts/td_build_l2.py --help
```
