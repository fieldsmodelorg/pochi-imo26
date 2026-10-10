# training — FM-Pochi-32B training code

Yi-Chia Chen's proof-pilot training pipeline, ported from the `hav4ik/ycchen-proof-pilot-codes` fork
(branch `opd/b200-cu128` @ `ba818f5`), plus the CUDA 12.8 image and launchers we used for the
64× B200 OPD run. Delivery stack: **stage-1 SFT → soft distillation → OPD v2 → deploy**.

| path | stage / role |
|---|---|
| [`stage1_v2/`](stage1_v2/) | Stage-1 SFT of the `olmo3_sink` model (L2 → L3 render/mask → L4 packed memmaps → FSDP2 trainer) |
| [`soft_distill_v2/`](soft_distill_v2/) | offline soft distillation (forward-KL on DeepSeek-V4-Flash hidden states) |
| [`opd_v2/`](opd_v2/) | **on-policy distillation** — the stage that produces the delivered model ([algorithm](docs/OPD_V2_ALGORITHM.md), [config](docs/OPD_V2_CONFIG_REFERENCE.md)) |
| [`teacher_extract/`](teacher_extract/) | DeepSeek-V4 teacher hidden-state extraction + the sglang patches for the teacher `/score` endpoint |
| [`dflash/`](dflash/) | DFlash speculative-decoding draft training (deploy-time; not part of the student's training) |
| [`olmo3_sink/`](olmo3_sink/) | the model: OLMo 3 + learnable attention sink (HF modeling, FA3 / FA2 sink kernels, Liger, packing) |
| [`train_core/`](train_core/) | DeepSeek-V4 chat encoding + L3 render/mask shared by SFT, soft-distill and teacher extraction |
| [`distill_gen/`](distill_gen/) | OPD prompt sources: `math_3r` role modules, prover templates, `problems/` |
| [`_common/`](_common/), [`_vendor_opd/`](_vendor_opd/) | JSD kernel + hidden codec; vendored v1 OPD leaf modules |
| [`deploy/make_olmo3sink_deploy.py`](deploy/make_olmo3sink_deploy.py) | training-format checkpoint → deploy format (legacy rope) for sglang serving + rollout |
| [`docker/`](docker/) | cu128 image for H200 + B200, slurm launcher, `opd_v2.sbatch`, env presets ([launch guide](docker/launch/README.md)) |
| [`docs/`](docs/) | OPD v2 algorithm, config reference (her best-run values), cu13 ↔ cu128 parity matrix |

## Layout conventions

- Scripts resolve paths from their own file (`sys.path` from `__file__`), with the **repo root** as the
  anchor: `training/...` paths are relative to it, exactly as in Yi-Chia's tree.
- `olmo3_sink/` and `train_core/` are top-level packages in her repo; here they live under `training/`.
  `stage1_v2/src/train.py` puts the nearest directory holding a `pyproject.toml` on `sys.path`, which is
  `training/` (her `pyproject.toml` + `uv.lock`, the dev environment), so `import olmo3_sink` resolves.
- The math_3r prompt templates are **not duplicated**: `distill_gen/math_3r/prompts.py` reads the repo's
  `evaluation/prompts/ycchen_math_3r/`, the same files the inference harness uses.

## Running OPD

The image (`chankhavu/ycchen-opd:cu128`) carries this tree at `/opt/opd/repo`. On 8 nodes × 8 GPU:

```bash
OPD_IMAGE=/shared/ycchen-opd+cu128.sqsh OPD_ENV=env_max_b200.sh \
RUN_DIR=/shared/runs/opd_max_b200 STUDENT_PATH=... DEEPSEEK_V4_FLASH=... \
sbatch training/docker/launch/opd_v2.sbatch
```

`env_max_b200.sh` is our B200 run (her V33 + 160k rollout); `env_v33.sh` is her V33 (128k), for H200
or a faithful reproduction. Details, requirements and the single-node smoke: [`docker/launch/README.md`](docker/launch/README.md).

## Not included

Beaker specs, the large binaries (FA2 wheel, JIT-cache tarball — see [`docker/README.md`](docker/README.md#build)),
the math_3r data-generation pipeline (its output dataset is on HF), and the rest of Yi-Chia's repo
(evaluation, Kaggle, quantization, tokenizer transplant). `olmo3_sink/build_init_model.py` lazily imports
her `tokenizer_transplant` package, which is only needed to rebuild the initial model from scratch.
