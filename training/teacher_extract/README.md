# teacher_extract

Extracts **DeepSeek-V4 teacher last-hidden-states** so the OLMo student can be distilled on the
teacher's full-vocab distribution (used by `soft_distill_v2` for offline distillation and by `opd_v2`
as the teacher `/score` service for on-policy distillation).

The teacher runs inside the stock `lmsysorg/sglang` container; we monkey-patch a few sglang files at
launch so the engine emits post-norm hidden states, then encode them with the `had+int6` codec
(`../_common/hidden_codec.py`) to keep them small (~14× smaller than bf16).

## Layout
| path | purpose |
|---|---|
| `_patch_sglang.py` | The canonical patch generator. Four anchored patches applied to sglang: (1) post-norm hidden + `wo_a` streaming, (2) scheduler hidden append, (3) `/score` HTTP endpoint (encodes hidden → codec, optional server-side write to shared FS), (4) top-1 head materialization. Anchors are asserted so an upstream change fails loudly. |
| `_validate_hidden.py` | Correctness harness (A/B/C checks) for `return_hidden_states` on the patched engine. |
| `_extract_e2e.py` | End-to-end smoke: docs json → sglang spool → `had+int6` shard. |
| `_render_docs.py` | Pre-render a few L2 docs to token ids (JSON) as validation input, decoupling L3 rendering from the engine runs. |
| `run_in_container.sh` | Launch the patched teacher service inside the sglang apptainer image. |
| `dataprep/` | Off-policy **data-prep** pipeline (below): HF rows → exact token manifest → teacher hidden → index. |

### `dataprep/` — off-policy hidden extraction at scale
`render_manifest*.py` render HF distillation rows (prompt + teacher reasoning + answer) into **exact
DeepSeek-V4 chat-template token manifests** (`_mp` = multiprocess), then `extract_hidden.py` streams
those token ids to the patched `/score` servers and stores one resumable `.pt` per document
(`input_ids`, codec-`packed`, `scales`, optional `teacher_top1`). `consolidate_index.py` /
`merge_indices.py` build the `index.jsonl` that `soft_distill_v2` and `dflash` consume.

## Usage
```bash
# 1. render HF rows -> exact token manifest (needs the DeepSeek tokenizer)
DEEPSEEK_V4_FLASH=/path/to/DeepSeek-V4-Flash \
  python training/teacher_extract/dataprep/render_manifest_mp.py --out work/manifest.parquet --workers 47

# 2. launch the patched teacher /score service (inside the sglang container)
SGLANG_SIF=/path/to/sglang.sif training/teacher_extract/run_in_container.sh

# 3. extract teacher hidden for the manifest
python training/teacher_extract/dataprep/extract_hidden.py --manifest work/manifest.parquet --out work/hidden
```

## Notes
- Env vars: `DEEPSEEK_V4_FLASH` (teacher weights), `SGLANG_SIF` (container image), `SGLANG_HIDDEN_CODEC_DIR`
  (points at `../_common` for `hidden_codec`), `OPD_V2_SRC` (points at `../opd_v2/src` for the shared-FS writer).
- The codec (`had+int6`, Hadamard rotation + int6 block-quant) lives in `../_common/hidden_codec.py`; the
  opd byte-layout constants are vendored under `../_vendor_opd/opd/`.
- Correctness was validated against a fresh engine run on the same doc/node (bf16-head `mean|Δ|` at engine
  rerun-nondeterminism floor).
