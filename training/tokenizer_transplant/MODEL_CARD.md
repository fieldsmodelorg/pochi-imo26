# Olmo-3-1025-7B + DeepSeek-V4-Flash tokenizer (centered-OMP transplant)

> This is the vendored copy of the model card stored alongside the produced model at
> `/models/Olmo-3-1025-7B-deepseekTok`. Method analysis and the mergekit comparison are in
> `../../docs/tokenizer/transplant.md`.

This model replaces the tokenizer/vocabulary of **`allenai/Olmo-3-1025-7B`** with the tokenizer of
**`deepseek-ai/DeepSeek-V4-Flash`** (vocab 100,278 → 129,280). All transformer body weights keep the
original Olmo weights; only `embed_tokens` and `lm_head` are rebuilt to match the new vocab, via a
**mean-centered Orthogonal Matching Pursuit (OMP)** embedding transplant.

- **Base (weights kept)**: allenai/Olmo-3-1025-7B (olmo3, 32 layers, hidden 4096, untied embeddings)
- **Donor (tokenizer + reference embeddings)**: deepseek-ai/DeepSeek-V4-Flash
- **Tokenizer**: DeepSeek-V4-Flash byte-level BPE, vocab 129,280, `bos/eos/pad = 0/1/1`
- **Motivation**: the DeepSeek tokenizer compresses CJK/multilingual text far better than Olmo's cl100k-family tokenizer (~2.6× fewer tokens for Chinese).

> ⚠️ **This is a warm-start initialization, not a finished model.** Zero-shot it can generate fluent
> English/code, but the 70k new (mostly CJK) tokens are only *initialized* and their output
> distribution is not yet calibrated. Strong multilingual generation requires continued pretraining
> first. See the **Limitations** section.

---

## Method

Training-free tokenizer transplant via Orthogonal Matching Pursuit
(Goddard & Fernandes Neto 2025, [arXiv:2506.06607](https://arxiv.org/abs/2506.06607)).

For each output matrix (`embed_tokens` and `lm_head`, each processed independently):

1. **Anchors** = tokens whose string exists in both vocabularies (**59,084** of them), compared by
   byte-level surface form (both tokenizers share the same `Ġ` byte-mapping, verified via decode).
2. **Shared tokens** → **copy** the base (Olmo) row verbatim to the new vocab id.
3. **New tokens** (**70,196** of them, 54.3% of the vocab) → OMP. Express the new token's *donor*
   embedding as a sparse (k=64) linear combination of anchors' donor embeddings; then apply the
   **same coefficients** to those anchors' *base* embeddings to obtain a row in Olmo space.
4. **Mean-centering**: before OMP, subtract the mean from both the donor and base anchor spaces; after
   reconstruction, add the base mean back.

### Faithfulness note (important)

This is **not** raw-canonical OMP but a **centered variant**. It shares the mean-centering idea with
mergekit's `SPARSE_TOKEN_BASIS` (STB) but is **not equivalent** to STB: STB projects each new token via
dense lstsq onto a shared basis obtained by SVD, whereas this method runs sparse k=64 OMP directly on
the anchors. In mergekit terms it is closest to `--approximation-method omp` **plus centering**.
Differences from canonical OMP and their measured impact:

| Difference | Canonical | This method | Impact (measured) |
|---|---|---|---|
| Atom selection | raw inner product | normalized (cosine) | fidelity-neutral |
| Least squares | ridge-free incremental QR | normal-eq + adaptive ridge 1e-3 | fidelity-neutral |
| **Mean-centering** | none | **yes** | output-logit calibration; fidelity-neutral |

The centered-vs-uncentered and mergekit comparison experiments are in `../../docs/tokenizer/transplant.md`.

---

## How it's built

- The donor embeddings are extracted from only **2 of 46 shards** (`embed.weight`, `head.weight`, both
  dense `[129280, 4096]` bf16) — about 2 GB, not the full 160 GB.
- Build code: this package `tokenizer_transplant/`, pure PyTorch, no mergekit dependency (DeepSeek's
  custom `deepseek_v4` arch is not recognized by mergekit-tokensurgeon — see docs).
- Runtime: ~196 seconds on a single H100 80GB.

Reproduce (from the repo root):
```bash
uv run python -m tokenizer_transplant full \
  --config tokenizer_transplant/configs/olmo3_7b__deepseek_v4_flash.yaml
uv run python -m tokenizer_transplant selftest \
  --config tokenizer_transplant/configs/olmo3_7b__deepseek_v4_flash.yaml --which embed
```

---

## Validation

**Correctness**
- Shared-token rows are **bitwise identical** to the original Olmo rows (embed + head).
- The anchor split is exact and mutually exclusive (59,084 + 70,196 = 129,280).
- For a prompt made entirely of shared tokens, the transplant's last hidden state is **bit-identical**
  to the original Olmo (cosine 1.0) — embeddings + body were untouched.

**Reconstruction fidelity** (held-out shared tokens, cosine to the real base row; random floor 0.024):
embed **0.59**, lm_head **0.70**. In agreement with canonical OMP within noise.

**Zero-shot generation (fluent):**
```
The capital of France is  → known for its rich history and culture...
def add(a, b):            → return a + b\ndef subtract(a, b): return a - b\ndef multiply...
The three primary colors are → red, yellow, and blue. These colors are called primary because...
Q: What is 2+2?\nA:       → 4
深度学习是一种            → 模式识别技术，它将输入数据映射到输出数据。在机器学习中...
# (a Chinese prompt "Deep learning is a" -> a fluent Simplified-Chinese continuation; kept verbatim as a real model output)
```

---

## Limitations

1. **New tokens' outputs are uncalibrated.** Teacher-forced next-token statistics on multilingual text:

   | target token | perplexity | top-1 | top-10 |
   |---|---|---|---|
   | shared | ~7 | ~50% | ~88% |
   | **new** | **~3400** | **~2%** | ~19% |

   The model *understands* text made of new tokens, and its argmax is usually reasonable (hence fluent
   free generation), but it cannot assign a sharp probability to a specific gold token.
2. **Bits-per-byte vs original Olmo** (tokenizer-independent, lower is better): en 0.53→0.64,
   zh 0.70→1.63, ja 0.76→1.53, ru 0.32→1.04. Multilingual is clearly worse on the output side.
3. **Greedy decoding drifts/loops where new tokens are dense** (distinct-token ratio ~0.45). Before CPT,
   use `repetition_penalty≈1.3` or sampling for usable output.
4. **Digits are grouped left-aligned in threes** (`\p{N}{1,3}`, `1000`→`100|0`, `1000000`→`100|000|0`).
   Note this is **identical to Olmo** — Olmo / DeepSeek / Kimi all split digits identically byte-for-byte
   (exhaustively verified, see `../../docs/tokenizer/comparison.md`), so the transplant **does not change
   digit handling**. The real weakness is that left-aligned grouping misaligns place-value across lengths
   (the last digit gets isolated), which hurts multi-digit arithmetic — but this is shared by all three
   and cannot be fixed by swapping tokenizers, only mitigated via training data / CoT format.
5. Even with Traditional Chinese input it outputs **Simplified** (due to DeepSeek's training distribution).

---

## Suggested next step: continued pretraining (CPT)

A short low-LR CPT run is needed to calibrate the 70k new token rows and adapt to the digit grouping.
Suggested recipe (per the paper, ~2B tokens, LR ~4e-7):
- optionally freeze the transformer body first, training only `embed_tokens` + `lm_head`, then unfreeze;
- corpus biased toward CJK / multilingual + math;
- evaluate GSM8K and multilingual BPB before and after CPT to quantify recovery.

---

## Files (in the produced model directory)
- `model.safetensors` — transplanted weights (vocab 129,280)
- `config.json` / `generation_config.json` — olmo3 config, `vocab_size=129280`, `bos/eos/pad=0/1/1`
- `tokenizer.json` / `tokenizer_config.json` — DeepSeek-V4-Flash tokenizer
