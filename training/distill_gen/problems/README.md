# problems — deduplicated proof-problem prompt pool

The **unique proof-problem prompt corpus** for off-policy distillation (DeepSeek-V4 teacher). Problems from two source datasets are merged, semantically deduplicated across datasets, and stored one problem per row with provenance kept, ready for generating teacher data via the DeepSeek API.

## Sources

| Source | unique problems | Notes |
|------|------------:|------|
| [`lm-provers/FineProofs-SFT`](https://huggingface.co/datasets/lm-provers/FineProofs-SFT) (`data` config) | 4,281 | olympiad / national-competition problems (apache-2.0); locally under `datasets/FineProofs-SFT/` |
| [`Nemotron-Math-Proofs-v2`](https://huggingface.co/datasets/nvidia/Nemotron-Math-Proofs-v2) | 5,752 | AoPS subset (CC BY 4.0); locally under `datasets/Nemotron-Math-Proofs-v2/`, 82,737 traces / 5,752 problems |

## Deduplication method

The same famous olympiad problem is often written with different LaTeX conventions, variable names, and phrasings, so plain string matching only catches 9 pairs (all LaTeX whitespace differences). We instead use embedding-based semantic dedup:

- **Model**: `Qwen/Qwen3-Embedding-0.6B` (fp16, a single 4090).
- **Cross-source**: FineProofs problems are encoded as query+instruction, Nemotron problems as document; for each FineProofs problem we take the nearest-neighbor cosine to a Nemotron problem, and **>= 0.87 is treated as the same problem**.
  - The threshold is calibrated on the 9 known duplicates that are "string-identical (after stripping LaTeX)": those 9 pairs have cosine in **0.891–0.945**, so a reliable same-problem cutoff is ≈ 0.88; a manual spot-check of the 0.87–0.90 band shows all real duplicates.
  - Result: **193 pairs** of cross-dataset duplicates.
- **Intra-source**: **string normalization only** (lowercase, strip LaTeX delimiters and punctuation), catching the LaTeX-whitespace variants that exact matching misses — 2 pairs in FP, 4 in NM.
  - ⚠️ **We deliberately do NOT use embeddings intra-source**: self-similarity of same-source same-encoding items inflates, and olympiad inequality problems are highly structurally similar (e.g. `a²+b²+c²=3` vs `a+b+c=3` have self-cosine 0.96 yet are different problems), so a 0.87 self-sim would falsely kill many. Each source was already deduplicated by its original authors; intra only patches the safe string variants.
- **Clustering**: connected components of the cross + intra edges above via union-find; each cluster keeps one representative problem (the longest problem text = the most complete statement), with all other members' provenance recorded in `members`.

### Statistics

```
raw merged input : 4,281 (FP) + 5,752 (NM) = 10,033
merged away      :   199  (193 cross-source embedding + 6 intra-source string)
unique problems  : 9,834
  ├─ FineProofs only            : 4,086
  ├─ Nemotron-Math-Proofs-v2 only: 5,562
  └─ both (same problem across datasets): 186   ← of 191 merged clusters, 183 are 2-member and 8 are 3-member
```

## Schema (`problems.parquet`, 9,834 rows, zstd)

| Column | Type | Notes |
|------|------|------|
| `problem` | string | representative problem text (the longest statement in the cluster) |
| `origin` | string | `FineProofs` / `Nemotron-Math-Proofs-v2` / `both` |
| `rep_source` | string | which source the representative text came from |
| `category` | string | FP's domain category (Inequalities/Combinatorics…); null for non-FP |
| `competition` | string | FP's competition origin (e.g. `Germany_TST-2020`); null for non-FP |
| `source` | string | FP=`olympiads` etc. / NM=`AoPS` |
| `fp_gemini_grade` | int64 | FineProofs' `gemini-3-pro-grade` (0/1/6/7); null for non-FP |
| `fp_qwen_reward` | double | FineProofs' `qwen3-4b-thinking-reward@128`; null for non-FP |
| `nm_uuid` | string | Nemotron sample uuid (joins back to the original traces); null for non-NM |
| `nm_n_traces` | int64 | number of traces for this problem in Nemotron (proof/verify/meta-verify combined) |
| `n_members` | int64 | how many original problems this cluster merged (1 = not merged) |
| `merge_max_cosine` | double | max edge cosine within the cluster (embedding sim for cross, 1.0 for intra); null if not merged |
| `members` | string (JSON) | provenance + original text of every member in the cluster, for traceability/audit |

## Tracing back to source data

- **Nemotron**: join back via `nm_uuid` or `problem` to `datasets/Nemotron-Math-Proofs-v2/data/train.jsonl` (each problem has proof / verification / meta-verification traces, `nm_n_traces` of them).
- **FineProofs**: join back via `problem` / `competition` to `datasets/FineProofs-SFT/` (the `all` config has multiple reference solutions per problem + grade, reward).
- For `origin=both` rows, the original text from both sources is in `members`.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/build_distill_prompts.py
```

Script: embed (cached at `/tmp/*_e.npy`) → cross/intra edges → union-find → write parquet (with a read-back check to guard against local bit-flips). The threshold and other parameters are at the top of the file.
