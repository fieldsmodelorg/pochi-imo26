# training/docker — OPD v2 image (CUDA 12.8, H200 + B200)

Containerizes the **full OPD v2 loop** (FSDP2 trainer + orchestrator + rollout sglang +
DeepSeek-V4-Flash teacher) into one **cu128** image built for both Hopper `sm_90` (H200) and
Blackwell `sm_100` (B200). Student = `chankhavu/yccchen-olmo3-deploy`. Published as
`chankhavu/ycchen-opd:cu128`.

## Why cu128 (not Yi-Chia's cu13 production stack)

The target cluster ceiling is **CUDA 12.8**, which is also the floor for Blackwell `sm_100`. B200 has
**no FA3** (Hopper-only), so the image swaps two kernels vs. her H200 run, and uses the same ones on
H200 so both GPUs run one code path:

| component | her H200 run (cu13) | this image (cu128) |
|---|---|---|
| trainer sink | patched-FA3 in-kernel | **`olmo3_sink_fa2`** — post-correction on **stock** flash-attn-2 (no kernel patch) |
| rollout attn | FA3 default (honors sink) | **`--attention-backend triton`** — honors the sink in **prefill *and* decode**; flashinfer drops it |
| teacher MoE | marlin (Hopper fp4) | auto-detected in `run_teacher.sh`: `flashinfer_mxfp4` on sm_100, `auto`→marlin on Hopper |

Numerics are validated fp64-exact vs eager and bit-exact on the OPD JSD loss. The rollout sglang uses
Yi-Chia's patched sources, re-anchored to the **0.5.14** release. Full parity matrix:
[`../docs/OPD_V2_PARITY_STATUS.md`](../docs/OPD_V2_PARITY_STATUS.md).

## Build

Two binaries are built/fetched out-of-band and are gitignored:

```bash
# 1) stock FA2 wheel for sm_90 + sm_100 (the build asserts both arches are present)
OUT=training/docker/wheels bash training/docker/build_fa2_wheel.sh
# 2) optional: pre-warmed sm_100 JIT cache (teacher reaches /health in ~1-2 min instead of ~25-35 min)
hf download chankhavu/opd-jit-cache-sm100 opd-jit-sm100-tp4-autotune.tgz --repo-type dataset \
  --local-dir training/docker/jit-cache
```

Then, from the **repo root** (BuildKit uses `Dockerfile.ycchen-opd.dockerignore`, a whitelist of
`training/` + `evaluation/prompts/ycchen_math_3r/`, instead of the root `.dockerignore`):

```bash
DOCKER_BUILDKIT=1 docker build -f training/docker/Dockerfile.ycchen-opd -t ycchen-opd:cu128 .
```

**BuildKit is required** (it reads the per-Dockerfile ignore file; the classic builder would use the root
`.dockerignore`, which excludes `training/`, and fail at the first `COPY training/...`).

Base: `chankhavu/olmo3-olmocore:cu128-fa2-sink`, pinned by digest (`sha256:e67d3894…`, the exact base of the
published run image) — torch 2.10+cu128, flash_attn 2.8.x, olmo_core; override with `--build-arg BASE=...`.
Python packages are pinned by `constraints-train.txt` / `constraints-serve.txt`, frozen from the published run image
`chankhavu/ycchen-opd@sha256:908516a7…`; a fresh resolve of the serve venv against them reproduces all 193
packages at the published versions. The build self-verifies both venvs (trainer imports + sink
registration + math_3r prompts; sglang serve patches incl. the hybrid-SWA KV-pool patch; teacher
hidden-extract patches). In the image the repo lives at `/opt/opd/repo` (same layout as here).

## Layout

| path | purpose |
|---|---|
| `Dockerfile.ycchen-opd` | the image (two venvs: base trainer/orchestrator, `/opt/venv/serve` sglang 0.5.14) |
| `launch/` | multi-node slurm launcher + `opd_v2.sbatch` + env presets → [`launch/README.md`](launch/README.md) |
| `opd_serve/run_teacher.sh` | DeepSeek-V4-Flash `/score` teacher (hidden-state extraction → shared FS) |
| `opd_serve/run_rollout.sh` | student rollout (fp8 + flash_rl loader; `--attention-backend triton`) |
| `opd_serve/run_teacher_olmo3.sh` | Olmo3-32B teacher (for the 4-GPU self-distill codec test) |
| `opd_serve/sglang_patches/` | Yi-Chia's patched sglang sources (olmo2 sink model, flash_rl loader, load_config, server_args) |
| `opd_v2_train_smoke.py` | fp64-exactness + OPD JSD + FA2-parity + compile smoke (GPU) |
| `test_attention_sink.py` | central sink-test launcher (`--group all`; the image's default CMD) |
| `tests/bench_teacher_prefill.py` | teacher `/score` prefill-throughput benchmark |

The two anchored source patchers live in the training tree and run at build time:
`training/opd_v2/flash_rl/patches/apply_swa_patch.py` (rollout SWA KV-pool) and
`training/teacher_extract/_patch_sglang_514.py` (teacher hidden-extract, 5 patches).
These are training-side patches, separate from the inference `sglang_patches/` at the repo root.

## Launch

Multi-node (8 nodes, slurm): see [`launch/README.md`](launch/README.md). Single node, one GPU per role:

```bash
# teacher (DeepSeek-V4-Flash)
CUDA_VISIBLE_DEVICES=3 /opt/opd/opd_serve/run_teacher.sh --tp 1 --port 8100
# rollout (student, deploy-format checkpoint)
CUDA_VISIBLE_DEVICES=1 MODEL=<student-deploy> /opt/opd/opd_serve/run_rollout.sh --port 8201
# trainer
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m opd_v2.trainer.service --run-dir $OPD_RUN_DIR
# orchestrator
ATTN_IMPL=olmo3_sink_fa2 ROLLOUT_URLS=... TEACHER_URLS=... python examples/make_config.py \
  && python -m opd_v2.orchestrator --run-dir $OPD_RUN_DIR
```
