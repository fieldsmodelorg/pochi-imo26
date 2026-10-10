# training/stage1_v2 — Stage-1 SFT (FMI train-variant container)

Stage-1 supervised fine-tuning: trains the `olmo3_sink` model (learnable attention sink +
FA3 packing-metadata reuse) on the public L2 SFT mix (`nemotron-deepseek-sft-mix`). This is
the production SFT that produced the `stage1-v2-7b` / `stage1-v2-32b` checkpoints, which in
turn seed the soft-distillation and OPD stages.

The trainer imports shared code from its canonical top-level homes (`olmo3_sink/`,
`train_core/`); the self-contained container `/app` is materialized at packaging time by
`make_pkg.py` from `pkg.manifest`.

## Layout
| file | purpose |
|------|---------|
| `src/train.py` | stage-1 SFT entrypoint: streaming L2→L3 render/loss-mask, length-packing, FSDP2/HSDP loop, DCP resume, HF save |
| `src/data_mix.py` | data-mix partition scanner |
| `build_l4.py` | pre-tokenize + pack the SFT mix into the L4 packed format |
| `mix_g2.json`, `mix_lc256k.json` | data-mix weight configs |
| `make_pkg.py`, `pkg.manifest`, `olmo3sink-sft_train.def` | Singularity container packaging (FMI train variant) |
| `fa3/build_fa3.sh` | build the patched FlashAttention-3 (in-kernel attention sink) |
| `requirements.txt` | container pip pins |

## Usage
```bash
# 1. Build the L4 packed dataset
python training/stage1_v2/build_l4.py \
    --roots data/nemotron-deepseek-sft-mix data/nemotron-deepseek-sft-mix-v2 \
    --mix training/stage1_v2/mix_g2.json \
    --tokenizer "$DEEPSEEK_TOK_MODEL" \
    --out data/l4-g2r05-ml12288-mc65536

# 2. Train (HSDP: intra-node FSDP2 full-shard x inter-node replicate)
torchrun --nnodes=8 --nproc_per_node=8 training/stage1_v2/src/train.py \
    --model_path "$OLMO3_SINK_MODEL" \
    --dataset_path data/l4-g2r05-ml12288-mc65536 \
    --output_path outputs/stage1-v2-7b \
    --logdir logs/stage1-v2-7b

# 3. Production 32B run (the exact stage1-v2-32b hyperparameters; 8 nodes x 8 H200,
#    see launch_8node_32b.sbatch). NOTE: once --max-steps is set, warmup comes from
#    --warmup-ratio (0.0417 x 12000 ≈ 500 steps), not --warmup-steps.
torchrun --nnodes=8 --nproc_per_node=8 training/stage1_v2/src/train.py \
    --model_path "$OLMO3_SINK_32B_MODEL" \
    --dataset_path data/l4-g2r05-ml12288-mc65536 \
    --output_path outputs/stage1-v2-32b \
    --logdir logs/stage1-v2-32b \
    --learning_rate 5e-5 --max-steps 12000 --lr-decay linear \
    --min-lr-ratio 0.1 --warmup-ratio 0.0417 --micro-len 65536
```

## Notes
- Model/tokenizer paths are supplied via env vars / CLI; `REPO` is resolved from `__file__`.
- L3 = offset-based assistant-only loss mask over the L2 messages format.
