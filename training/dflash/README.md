# training/dflash — DFlash speculative-decoding draft training

Trains a **DFlash draft model** for an `olmo3_sink` target. DFlash is single-step
block-diffusion drafting: one draft forward proposes a whole block of tokens, which the
target then verifies. The draft here is an OLMo3-style 8-layer model with the same attention
sink as the target, trained with greedy-prefix filtering and a no-duplicate FA3 full-attention
block. The trained draft is served by the inference stack's DFlash patches (`sglang_patches/` / `vllm_patches/` at the repo root).

## Layout
| file | purpose |
|------|---------|
| `canonical_fa3_train_prod.py` | **production trainer for the delivered drafts** (canonical block convention: block-0 = the verified token's target embedding, predicts `block_size-1` tokens; weights are drop-in for the stock sglang DFlash worker); checkpoint/resume + `--init-from` warm start |
| `experiments/canonical_fa3_train.py` | canonical draft model + loss/metrics core, imported by `canonical_fa3_train_prod.py` |
| `nodup_fa3_train_prod.py` | no-dup-convention production trainer (earlier convention) |
| `nodup_fa3_train.py` | no-dup model + loss/metrics core, imported by both prod trainers |
| `fa3_nodup_attention.py` | no-duplicate full-attention FA3 block used during training |
| `draft_model_olmo3.py` | the OLMo3-style DFlash draft model |
| `target_model.py`, `target_utils.py` | frozen target (FSDP2) + embedding/head sharing |
| `data.py`, `build_l4_pretokenized.py` | L4 packed-data loader + builder (from rollout dumps / teacher manifests) |
| `optimizer.py`, `lr_scheduler.py`, `tracker.py`, `utils.py`, `train.py` | training-loop support |
| `configs/` | draft configs (7B/32B × SWA window 128/512) |
| `examples/` | 32B two-phase curriculum sbatch scripts (phase S / phase L) |
| `tests/` | unit tests (attention, sampler resume, target, misc) |

## Usage — 32B canonical draft (two-phase curriculum)

The delivered 32B draft is trained with `canonical_fa3_train_prod.py` in two phases
(`examples/run_32b_v2test_swa512_short_64g.sbatch`, then
`examples/run_32b_v2test_phaseL_64g.sbatch`): block size 11 (predicts 10), SWA window 512.

```bash
# Phase S — short-context warm-up: cheap 8k target forwards, many steps, SFT-mix data.
torchrun --nnodes=8 --nproc_per_node=8 training/dflash/canonical_fa3_train_prod.py \
    --target-model-path outputs/stage1-v2-32b-softdistill-v2test \
    --draft-config-path training/dflash/configs/dflash-olmo3-32b-8L-bs10-swa512-sink.json \
    --train-data-path data/l4-g2-ml4096-mc8192 \
    --output-dir outputs/dflash-canonical-32b-phaseS \
    --steps 12000 --window-size 512 --block-size 11 \
    --max-starts-per-step 8192 --start-chunk-size 2048 \
    --learning-rate 6e-4 --loss-decay-gamma 5.0

# Phase L — long-context specialization: warm-start from phase S, train on the real
# deployment distribution (long proof rollouts + teacher proofs), flatter position decay.
torchrun --nnodes=8 --nproc_per_node=8 training/dflash/canonical_fa3_train_prod.py \
    --target-model-path outputs/stage1-v2-32b-softdistill-v2test \
    --draft-config-path training/dflash/configs/dflash-olmo3-32b-8L-bs10-swa512-sink.json \
    --train-data-path data/l4-dflash32b-opd-dsflash-ml65536 \
    --output-dir outputs/dflash-canonical-32b-phaseL \
    --init-from outputs/dflash-canonical-32b-phaseS/final.pt \
    --steps 3000 --window-size 512 --block-size 11 \
    --max-starts-per-step 8192 --start-chunk-size 2048 \
    --learning-rate 6e-4 --loss-decay-gamma 20.0
```

- `--block-size 11` is the TOTAL slot count: 1 verified block-0 + 10 predicted slots.
- `--max-starts-per-step 8192` = full position coverage of an 8k bin; the loss is a
  weight-normalized mean over starts, so raising it needs no LR change.
- `--init-from` loads model weights only (and re-syncs the optimizer's fp32 masters);
  if a `latest.pt` exists in `--output-dir`, auto-resume takes precedence.
- Phase L uses `--loss-decay-gamma 20` (vs 5) to flatten the per-position loss decay so
  far positions (pos 8/9) get ~2.5× more weight.
- The legacy no-dup trainer is invoked the same way via `nodup_fa3_train_prod.py`
  (`--block-size 10`, swa128 configs).

### Short-context (phase S) data recipe

Phase S reuses the stage-1 SFT mix, re-rendered by `stage1_v2/build_l4.py` at short
lengths so the frozen-target forward is ~8× cheaper:

```bash
python training/stage1_v2/build_l4.py \
    --roots data/nemotron-deepseek-sft-mix data/nemotron-deepseek-sft-mix-v2 \
    --mix training/stage1_v2/mix_g2.json \
    --tokenizer "$DEEPSEEK_TOK_MODEL" \
    --max-len 4096 --micro-len 8192 \
    --out data/l4-g2-ml4096-mc8192
```

(`--max-len 4096` caps each document, `--micro-len 8192` packs them into 8k bins;
~3M bins from the G2 mix.) Phase L data comes from `build_l4_pretokenized.py` over OPD
rollout dumps (finish_reason=length filtered) + teacher-proof manifests at micro-len 65536.

## Notes
- Resource paths come from env vars (`TARGET_MODEL_PATH`, `DFLASH_TRAIN_DATA`,
  `DFLASH_OUTPUT_DIR`, `DFLASH_TOKENIZER`); configs are resolved relative to this directory.
- `build_l4_pretokenized.py` consumes a teacher manifest produced by
  [`teacher_extract/dataprep`](../teacher_extract) and reuses the FFD packer from `stage1_v2/build_l4.py`.
