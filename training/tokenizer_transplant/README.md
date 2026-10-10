# tokenizer_transplant

Training-free **OMP tokenizer transplant**: swap a base model's tokenizer for a donor's
while keeping the transformer body untouched — only the two vocab-sized matrices
(`embed_tokens`, `lm_head`) are rebuilt. Shared tokens (string match) are copied verbatim;
new tokens are reconstructed by **centered Orthogonal Matching Pursuit** over the shared
anchors. Used here to give OLMo 3 the DeepSeek-V4 vocabulary (the distillation linchpin).

## Layout
| file | purpose |
|---|---|
| `omp.py` | core (audited) centered-OMP numerics: cosine selection + ridge least-squares |
| `transplant.py` | end-to-end pipeline: rebuild embed/head, special-token map, chat template |
| `cli.py`, `__main__.py` | YAML-config-driven CLI (`python -m tokenizer_transplant <config.yaml>`) |
| `selftest.py` | fidelity self-checks (shared-row bitwise equality, held-out reconstruction cosine) |
| `configs/*.yaml` | base/donor/out configs for OLMo-3 7B & 32B (Instruct + Think variants) |
| `MODEL_CARD.md` | model card for the transplanted checkpoints |

## Usage
```bash
# Edit a config to point base/donor/out at your local model dirs, then:
python -m tokenizer_transplant full --config tokenizer_transplant/configs/olmo3_think_7b__deepseek_v4_flash.yaml

# 32B production transplant (the checkpoint behind step01 of the final pipeline):
python -m tokenizer_transplant full --config tokenizer_transplant/configs/olmo3_think_32b__deepseek_v4_flash.yaml
```

## Notes
- The config `base` / `donor_weights` / `out` paths are placeholders (`/models/...`) — set them to your paths.
- Shared tokens are copied bitwise; only genuinely new tokens go through OMP, so English capability is
  near-native and the residual gap is closed by later calibration/SFT, not by initialization.
- Reference: *Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit* (arXiv:2506.06607).
