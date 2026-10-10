# training — FM-Pochi-32B training code

Yi-Chia Chen's proof-pilot training pipeline, plus the CUDA 12.8 image and launchers we used for the
64× B200 OPD run. Delivery stack: **tokenizer transplant → stage-1 SFT → soft distillation → OPD v2 → deploy**.

| path | stage / role |
|---|---|
| [`tokenizer_transplant/`](tokenizer_transplant/) | stage 0: OMP transplant of the DeepSeek-V4 tokenizer onto Olmo-3.1-32B-Think (configs in `configs/`) |
| [`olmo3_sink/`](olmo3_sink/) | the model: OLMo 3 + learnable attention sink (HF modeling, FA3 / FA2 sink kernels, Liger, packing); `build_init_model.py` = transplant + sink calibration (`calib/` holds the calibration corpus) |
| [`scripts/`](scripts/) | data pipeline: raw → unified **L2** schema (`td_normalize.py`, `td_build_l2{,_v2}.py`), Parquet conversion/verification, distill prompt set |
| [`stage1_v2/`](stage1_v2/) | Stage-1 SFT (L2 → L3 render/mask → L4 packed memmaps → FSDP2 trainer) |
| [`train_core/`](train_core/) | DeepSeek-V4 chat encoding + L3 render/mask shared by SFT, soft-distill and teacher extraction |
| [`teacher_extract/`](teacher_extract/) | DeepSeek-V4 teacher hidden-state extraction + the sglang patches for the teacher `/score` endpoint |
| [`soft_distill_v2/`](soft_distill_v2/) | offline soft distillation (forward-KL on the extracted teacher hidden states) |
| [`opd_v2/`](opd_v2/) | **on-policy distillation** — the stage that produces the delivered model ([algorithm](docs/OPD_V2_ALGORITHM.md), [config](docs/OPD_V2_CONFIG_REFERENCE.md)) |
| [`distill_gen/`](distill_gen/) | OPD prompt sources: `math_3r` role modules, prover templates, `problems/` |
| [`_common/`](_common/), [`_vendor_opd/`](_vendor_opd/) | JSD kernel + hidden codec; vendored v1 OPD leaf modules |
| [`deploy/`](deploy/) | `make_olmo3sink_deploy.py` (training → deploy format), `prepare_yarn_deploy.py` (YaRN-256k config), `target/olmo2_sink.py` (reference sglang sink model + parity test) |
| [`dflash/`](dflash/) | DFlash speculative-decoding draft training (deploy-time; not part of the student's training) |
| [`docker/`](docker/) | cu128 image for H200 + B200 (+ its base image in `base/`) and the OPD launchers ([image](docker/README.md), [launch guide](docker/launch/README.md)) |
| [`docs/`](docs/) | OPD v2 algorithm, config reference (her best-run values), cu13 ↔ cu128 parity matrix |

## Layout conventions

Yi-Chia's repo had `training/` next to top-level packages (`olmo3_sink/`, `train_core/`, `tokenizer_transplant/`,
`scripts/`, `deploy/`, `distill_gen/`). Here the **repo root stands in for her repo root**, and her other
top-level directories live under `training/`. So:

- Commands written as `training/...` (most launchers and docstrings) run **from the repo root**, unchanged.
- Her top-level tools run **from `training/`**: `python -m tokenizer_transplant full --config tokenizer_transplant/configs/...`,
  `python -m olmo3_sink.build_init_model ...`, `python scripts/td_build_l2_v2.py ...`, `python deploy/prepare_yarn_deploy.py ...`.
- The dev environment is `training/pyproject.toml` + `uv.lock` (`uv sync` inside `training/` → `training/.venv`, which
  the bare-metal sbatch files use). `stage1_v2/src/train.py` puts the nearest directory holding a `pyproject.toml`
  on `sys.path`, which is `training/`, so `import olmo3_sink` / `train_core` resolve from anywhere.
- The math_3r prompt templates are **not duplicated**: `distill_gen/math_3r/prompts.py` reads the repo's
  `evaluation/prompts/ycchen_math_3r/`, the same files the inference harness uses.

## Model lineage

| model | how it was produced |
|---|---|
| `ycchen/proof-pilot-checkpoints` step01 → step09 | tokenizer transplant → sink init → stage-1 SFT → YaRN-256k → soft distillation (her pipeline above) |
| `chankhavu/yccchen-olmo3-deploy` (= `fieldsmodelorg/FM-Pochi-32B-ProofPilot`) | her OPD v2 from step09, `env_v33.sh` settings, OPD step 200, deploy format |
| `fieldsmodelorg/FM-Pochi-32B-IMO26` (`opd-32b-bf16-step-*`) | our OPD v2 on 64× B200, warm-started from the deploy model above (her OPD step 200) |

## Running OPD

The image `chankhavu/ycchen-opd@sha256:908516a7…` is the one our B200 run used; it carries the training code at
`/opt/opd/repo`. Three equivalent multi-node launchers (8 nodes × 8 GPU), all running the same role logic:

| launcher | when |
|---|---|
| `docker/launch/opd_v2.sbatch` | slurm + pyxis/enroot |
| `docker/launch/launch_ssh.sh HOSTFILE` | no scheduler: ssh + docker on each node |
| `docker/launch/run_mn_nodes.sh` | your own scheduler: run it on every node with `NODE_RANK`/`NNODES`/`LAUNCH_ID` |

Presets: `env_max_b200.sh` is our B200 run (her V33 + 160k rollout); `env_v33.sh` is her V33 (128k), for H200 or a
faithful reproduction of her run. Both resolve to `config.json` files that were checked byte-for-byte against the
originals. Details, requirements and the single-node smoke: [`docker/launch/README.md`](docker/launch/README.md).

## Reproducibility notes

- **Pinned:** the run image (digest above), its base image (by digest in the Dockerfile), and every Python package
  of both image environments (`docker/constraints-{train,serve}.txt`, frozen from the run image).
- **Not bit-exact:** OPD is asynchronous — rollout requests carry no sampling seed, and staleness and weight-sync
  timing depend on the pipeline — so a rerun reproduces the recipe statistically, not bit-for-bit. Stage-1 SFT
  and soft distillation are seeded (data order, packing, row shuffle).
- **Data:** the OPD seed is served from the public mirror `chankhavu/ycchen-dsflash-proof-distill-v2-test` (her
  original repo is no longer public). Her `ycchen/proof-pilot-datasets` (resolved V33 config, OPD pool/rollouts)
  and `ycchen/dsflash-proof-distill-test` (the v1 dataset `teacher_extract/dataprep/render_manifest.py` defaults
  to) are no longer public.

## Not included

The large binaries (FA2 wheel, JIT-cache tarball — see [`docker/README.md`](docker/README.md#build)), the math_3r
data-generation pipeline (its output dataset is on HF), and the rest of Yi-Chia's repo (evaluation, Kaggle,
quantization).
