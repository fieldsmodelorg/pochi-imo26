# train_core

Small shared training-core library reused across the pipeline (SFT, soft-distill, OPD,
data rendering). Tokenizer-independent L2 messages → tokenized, loss-masked L3 examples.

## Layout
| file | purpose |
|---|---|
| `l3_render.py` | render one L2 (OpenAI-style messages) example into `(input_ids, labels)` with an exact, offset-based assistant-only loss mask |
| `encoding_dsv4.py` | vendored DeepSeek-V4 `encode_messages` (official chat rendering) used by L3 |

## Usage
```python
from train_core.l3_render import render_l3   # -> input_ids + per-token loss mask
```

## Notes
- "L2" is the tokenizer-independent unified message format; "L3" is the tokenized + masked form
  fed to `olmo3_sink` training. The mask covers assistant turns only, computed by character offsets
  and verified against an `encode_messages` round-trip.
