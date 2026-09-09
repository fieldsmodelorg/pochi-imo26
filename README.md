# AIMO Proof Pilot Inference

This repository packages the generate-verify-refine proof harness as a Docker
image. The submission path reads `test.csv`, runs the selected harness, and
writes `submission.csv` without calling an external grader. The checked-in
configuration uses eight H200 GPUs as four TP2 replicas, BF16 target and draft
weights, DFlash speculative decoding, and FlashAttention 3. Inference has only
been tested on H200.

> ### 📦 Prebuilt image
> **`ghcr.io/fieldsmodelorg/aimo-proof-pilot:sha-463682b`** &nbsp;·&nbsp; built from `main` [`463682b`](https://github.com/fieldsmodelorg/AIMO-Proof-Pilot/commit/463682bbf4137dac6366246ee7aefa1b0d0a4a68) &nbsp;·&nbsp; [package on GHCR](https://github.com/fieldsmodelorg/AIMO-Proof-Pilot/pkgs/container/aimo-proof-pilot)
> ```bash
> docker pull ghcr.io/fieldsmodelorg/aimo-proof-pilot:sha-463682b
> ```
> Public, pullable with no login. Bakes the SGLang runtime + the committed IMO-2026
> problem set; only model weights are downloaded at run time.

## Quick start (8×H200)

You need an 8×H200 node with Docker and NVIDIA GPU access. The image is
self-contained — the SGLang runtime and every dependency are baked in; only the
model weights are downloaded, into a folder you mount. No HuggingFace token is
needed at any step.

**1. Start the container**, mounting a host folder at `/workspace` (it holds the
downloaded models and all run outputs, and persists across restarts):

```bash
mkdir -p data
docker run --rm -it --gpus all --ipc=host --shm-size=32g \
  --entrypoint bash \
  -v "$PWD/data:/workspace" \
  ghcr.io/fieldsmodelorg/aimo-proof-pilot:sha-463682b
```

(`--entrypoint bash` opens a shell in the repo directory; the automated
`serve` / `submission` entrypoints in [Docker usage](#docker-usage) require a
`CONFIG` instead.)

**2. Download the model weights** (public repos, no token needed):

```bash
./download_models.sh                # both targets + shared draft  ->  /workspace/models
# ./download_models.sh step225      # just the step225 target + draft (skip deploy)
```

**3. Run inference.** The recommended best setting is
**FM-Pochi-32B-IMO26 (step225)** at the **xhigh** budget:

```bash
./scheduler.sh config-model-step225-budget-xhigh.yaml /workspace/runs/step225-xhigh
```

That is the whole flow. The scheduler starts the server, waits for it to be healthy,
runs all six IMO-2026 problems (the committed `evaluation/data/imo2026-latex-test.csv`),
and writes `/workspace/runs/step225-xhigh/submission.csv` (with `artifacts/` and
`server.log` alongside), then shuts the server down. Pass a third argument to run
your own `id,problem` CSV instead.

**If a run is interrupted** (a crash or a node reboot), continue it — no restart
from scratch:

```bash
./scheduler.sh --resume /workspace/runs/step225-xhigh
```

Finished problems are skipped and a partially-done problem resumes from its last
completed round. (Running outside the container? Point `VENV` at the runtime venv,
or `source` its `activate-env.sh` and set `PYTHON`; everything else is identical.)

### Models

`download_models.sh` fetches these into `/workspace/models/` from public,
un-gated HuggingFace repos (pinned to a fixed revision for reproducibility):

| model | local folder | source repo (revision) |
|---|---|---|
| **FM-Pochi-32B-ProofPilot (deploy)** | `opd-32b-deploy` | `fieldsmodelorg/FM-Pochi-32B-ProofPilot` (`87707b80`) |
| **FM-Pochi-32B-IMO26 (step225)** | `opd-32b-bf16-step-225` | `fieldsmodelorg/FM-Pochi-32B-IMO26` (`f14030d3`) |
| DFlash **draft**, shared by both targets | `dflash-32b-draft-v2test-phaseL` | `fieldsmodelorg/FM-Pochi-32B-ProofPilot` (`87707b80`) |

Each source repo holds several checkpoints as subfolders, so the repo name alone
does not identify a model — the **local folder** column is also the subfolder name
inside the repo. **FM-Pochi-32B-IMO26 (step225)** is the `opd-32b-bf16-step-225`
subfolder of `FM-Pochi-32B-IMO26`; **FM-Pochi-32B-ProofPilot (deploy)** is the
`opd-32b-deploy` subfolder of `FM-Pochi-32B-ProofPilot`, which also ships the
shared DFlash draft.

`./download_models.sh` (default `all`) fetches both targets + draft; pass
`step225` for just FM-Pochi-32B-IMO26 (step225) + draft, or `deploy` for just
FM-Pochi-32B-ProofPilot (deploy) + draft. Budget on disk: roughly ~64 GB per
target checkpoint plus ~5 GB for the draft (so ~135 GB for the default `all`).

### Production configs

Six presets, `config-model-<model>-budget-<budget>.yaml`, that vary only the
search budget (exact knobs in [Budget presets](#budget-presets)). `<model>` is
the parenthesised short name from the table above — `deploy` or `step225` — and
the same token `download_models.sh` takes as its argument:

| checkpoint | medium | high | xhigh |
|---|---|---|---|
| **FM-Pochi-32B-ProofPilot (deploy)** | [`…deploy-budget-medium`](config-model-deploy-budget-medium.yaml) | [`…deploy-budget-high`](config-model-deploy-budget-high.yaml) | [`…deploy-budget-xhigh`](config-model-deploy-budget-xhigh.yaml) |
| **FM-Pochi-32B-IMO26 (step225)** | [`…step225-budget-medium`](config-model-step225-budget-medium.yaml) | [`…step225-budget-high`](config-model-step225-budget-high.yaml) | [**`…step225-budget-xhigh`**](config-model-step225-budget-xhigh.yaml) |

**FM-Pochi-32B-IMO26 (step225)** is the strongest checkpoint and `xhigh` the
largest budget, so **`config-model-step225-budget-xhigh.yaml`** is the
recommended best setting.

## Docker usage

> The [Quick start](#quick-start-8h200) above is the recommended path. This section
> documents the lower-level, fully-automated container entrypoint (`serve` /
> `submission`) for reference.


### Select the harness commit

The image is built on demand (a `v*` release tag or a manual **Run workflow** in
the Actions tab — the baked image is ~19 GB, so it is not built per commit) and
published to **`ghcr.io/fieldsmodelorg/aimo-proof-pilot`** (public, no login), tagged
`sha-<7-character-commit>`. The current build is `sha-463682b`. Set `COMMIT` to the
full commit of a build that completed successfully:

```bash
export COMMIT=463682bbf4137dac6366246ee7aefa1b0d0a4a68
export IMAGE=ghcr.io/fieldsmodelorg/aimo-proof-pilot:sha-${COMMIT:0:7}

docker pull "$IMAGE"
test "$(docker image inspect "$IMAGE" \
  --format "{{ index .Config.Labels \"org.opencontainers.image.revision\" }}")" = "$COMMIT"
```

The runtime venv (patched SGLang + kernels) is **baked into the image** at
`/opt/pp`, inherited (`COPY --from`) from a pinned container base image
(`RUNTIME_BASE_IMAGE`; see [runtime/README.md](runtime/README.md)) and topped with
the pinned PyPI deps at build time, so the final image is self-contained: **no
runtime download and no `HF_TOKEN` for the runtime**, and every image tag carries
an identical, frozen SGLang. Only the
(public) model weights are fetched at boot -- so a plain
`docker run ... submission` needs no secrets at all. At boot the entrypoint just
applies the checked-in SGLang patches (fast, in-place) and resolves the models.

Those patches are not optional. Neither checkpoint is stock
`Olmo-3.1-32B-Think`, the open model they were trained from: both use
**attention sinks**, an architectural change no upstream inference stack
implements, so the Olmo3Sink model patch below and the custom FlashAttention-3
kernels baked into the runtime are what make their outputs numerically correct
(the same support is packaged for vLLM in [`vllm_patches/`](vllm_patches/)).
Those kernels are compiled for Hopper (`sm90a`) — the H200 GPU generation.

> **Always launch through the `serve`/`submission` entrypoint or `scheduler.sh`.**
> The SGLang patches (including the **required** Olmo3Sink model patch) are applied
> at boot by those launchers, not baked into the venv at rest. A hand-rolled
> `python -m sglang.launch_server` / `launch_server.py` that bypasses both would run
> **unpatched** (no attention sinks) and silently produce wrong numerics. `scheduler.sh`
> applies the patch set itself and now **fails loudly** if the runtime is missing the
> expected patch helper, rather than skipping.

### Prepare persistent storage

Mount persistent storage at the internal `/workspace` path. It holds the models,
caches, `test.csv`, `submission.csv`, and resumable search artifacts. (The SGLang
runtime is baked into the image at `/opt/pp`, not under `/workspace`.) The host
path is arbitrary; these examples use `$PWD/workspace`. Allow at least 200 GB for
the checked-in default model pair, caches, and artifacts.

Fetch the selected commit configuration, then edit any values needed for the
run:

```bash
mkdir -p "$PWD/workspace"
curl -fsSL \
  "https://raw.githubusercontent.com/fieldsmodelorg/AIMO-Proof-Pilot/$COMMIT/config.yaml" \
  -o "$PWD/workspace/config.yaml"
```

`config.yaml` is the minimal base (8×H200, DFlash, selector **off**) and targets
**FM-Pochi-32B-IMO26 (step225)** by default. For the LLM
final-solution selector and the tuned search budgets, use the production configs —
the `config-model-{deploy,step225}-budget-{medium,high,xhigh}.yaml` presets in the
repo root (see [Budget presets](#budget-presets)) — or set `search.llm_selector: true`
(+ `selection_*` knobs) in your own copy.

`CONFIG` is mandatory. The container has no fallback configuration. It validates
the supplied YAML but never copies, rewrites, clamps, or overrides its values.
All model paths in the YAML are absolute container paths. Put custom target and
draft assets at the corresponding locations under the mounted storage. The
container downloads the checked-in default model pair only when the YAML uses
the default paths and those assets are missing.

**By default the container runs the committed IMO-2026 set** — the exact 6-problem
`evaluation/data/imo2026-latex-test.csv` that was run on the cluster from the [LLMC](https://llmc.nii.ac.jp/en/) at [NII](https://nii.ac.jp/en/). You do **not** need to
create anything to reproduce our results; leave `/workspace/test.csv` absent and the
`submission` entrypoint falls back to the committed CSV automatically.

To run your **own** problems instead, mount a `test.csv` at `$PWD/workspace/test.csv`
(or set `INPUT_CSV`) with exactly two lowercase columns:

```csv
id,problem
0,"First complete problem statement"
1,"Second complete problem statement"
```

IDs must be nonempty and unique. Quote fields containing commas or newlines. Do
not add answers, rubrics, reference solutions, or metadata columns. Note: the harness
keys its deterministic RNG on CSV **row order**, so to reproduce a specific run use the
same problems in the same order (the committed CSV is byte-exact for the NII set).

### Generate submission.csv

```bash
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v "$PWD/workspace:/workspace" \
  -e CONFIG=/workspace/config.yaml \
  "$IMAGE" submission
```

The command resolves the configured models, applies the selected commit patches,
starts and validates SGLang, and
processes input rows sequentially. It writes exactly these columns to
`$PWD/workspace/submission.csv`:

```csv
id,proof
```

The submission workflow does not call an external grader. Multiline proofs are
CSV-quoted. After every completed search round, the current problem row is
atomically replaced with the top-ranked cumulative-pool proof; the final
selection replaces it once more when the search completes.

## Configuration

The selected commit YAML is the complete runtime contract. Candidate labels
identify harness policy, not model identity; record the target and draft model
revisions separately when comparing model pairs.

The current `main` defaults are:

| Setting | Value |
|---|---|
| Hardware | 8 x NVIDIA H200 |
| Default target | FM-Pochi-32B-IMO26 (step225) — `opd-32b-bf16-step-225` (`f14030d3`) |
| Model mode | BF16 target and BF16 DFlash draft |
| Parallelism | TP2 x DP4 |
| Attention | FA3, page size 1, non-deterministic inference |
| Server context | 262,144 tokens |
| Server concurrency | 64 running requests per DP replica |
| Search concurrency | 96 requests cluster-wide |
| Search policy | 32 proofs, 16 verifications per proof, top 8, 4 refine parents × 3 reviews, up to 4 rounds |
| Sampling | temperature 1.0, top-p 0.95 |
| First output segment | 128,000 tokens |
| Solution continuation | 16,384 tokens |
| Verifier continuation | 16,384 tokens |

Users may change every YAML value. Validation retains type, range, schema, and
implementation compatibility checks, including:

```text
top_proofs           <= proofs_per_round
refine_parents       <= top_proofs
reviews_per_refine_parent <= verifications_per_proof
min_valid_verifications   <= verifications_per_proof
FA3: page_size=1     (deterministic_inference optional; the configs run it false)
FA4: page_size=128
```

The configured server context is a total input-plus-output limit.

### Budget presets

The `config-model-{deploy,step225}-budget-{medium,high,xhigh}.yaml` configs are a
matrix that varies **only the search budget**. `refine_parents` (4) ×
`reviews_per_refine_parent` (3) — the training limit — and everything else (server
topology, sampling, the LLM selector) are held constant, so runs differ only by
compute. Pick one by name with `scheduler.sh` (or as the container `CONFIG`).

| preset | proofs_per_round | verifications_per_proof | top_proofs | refine_parents | reviews/parent | max_rounds |
|---|---|---|---|---|---|---|
| **medium** | 32 | 16 | 8 | 4 | 3 | 4 |
| **high** | 64 | 32 | 16 | 4 | 3 | 8 |
| **xhigh** | 128 | 64 | 32 | 4 | 3 | 8 |

- `medium` is the original run policy.
- `proofs_per_round` is both the round-1 prover count and the per-round refinement
  count; `top_proofs` is the pool refinement parents are stratified-sampled from.
- `refine_review_strategy` is `random_nonideal` (each refine parent is paired with 3
  reviews drawn from its `<1`-score verifications).
- `max_rounds` counts round 1 (generation): `4` = 1 gen + 3 refine, `8` = 1 gen + 7 refine.
- Two checkpoints — FM-Pochi-32B-ProofPilot (deploy) and FM-Pochi-32B-IMO26
  (step225) — × three budgets = the six configs.

## Resume and outputs

Search state is stored in `/workspace/submission_artifacts`. If a run stops
before its configured final round, `submission.csv` retains the top proof from
the latest completed round. Re-run the same
image, YAML, `test.csv`, and command to reuse completed work and retry missing or
failed work. For a different input set or policy, use a new directory:

```bash
-e ARTIFACTS_DIR=/workspace/submission_artifacts_candidate_2
```

The runner rejects mismatched inputs or configuration rather than silently
mixing runs.

## Other commands

All commands except `help` require `CONFIG`:

| Command | Purpose |
|---|---|
| `submission` | Start the server and generate `submission.csv` |
| `serve` | Start and validate only the configured SGLang server |
| `bootstrap` | Prepare the runtime and configured models without GPUs |
| `validate` | Validate an already running configured server |
| `shell` | Prepare the runtime and open a shell |
| `help` | Show entrypoint help |

The SGLang API has no application authentication. Do not publish its configured
port directly; use private networking or an authenticated reverse proxy.

## Troubleshooting

**`CONFIG is required`:** mount the YAML into the container and pass its absolute
container path with `-e CONFIG=/workspace/config.yaml`.

**Configured model is incomplete:** ensure each active model path contains
`config.json` and safetensor weights. Custom paths are never replaced or
downloaded automatically.

**CUDA device-count mismatch:** the number of visible GPUs must equal
`tensor_parallel_size * data_parallel_size` from the YAML.

**Resume input mismatch:** restore the exact image, YAML, and `test.csv`, or use a
new `ARTIFACTS_DIR`.

**Server validation reports missing DFlash markers:** inspect the complete log at
`/workspace/opd32b-eval.log` and confirm that the configured target, draft,
attention backend, and DFlash settings are compatible.

## Repository layout

A tour of the tree (top level):

```text
README.md                     this file
scheduler.sh                  one command to run a full inference — starts the SGLang server,
                              waits for health, validates the config, then runs the submission
                              (resumable with --resume)
run_submission.sh             thin wrapper: run the proof-search harness over an input CSV
download_models.sh            fetch model weights into /workspace/models (all | step225 | deploy)

config.yaml                   minimal base config (8×H200, DFlash, selector off); targets step225
config-model-*-budget-*.yaml  the six production presets — {deploy,step225} × {medium,high,xhigh}
config-blackwell.yaml         Blackwell (sm120) profile — auto data-parallel, fp8 KV cache
config-dynamic.yaml           dynamic profile — adapts parallelism to the node's GPU count

evaluation/                   the generate-verify-refine harness and its assets
  harness/                    proof_search.py (the search loop), eval_config.py, launch_server.py,
                              run_submission.py, the LLM selector, loop detection, trace uploader
  harness_vllm/               alternative vLLM-backed runner (fallback backend)
  data/                       the IMO-2026 test set (CSV + rendered PDF)
  prompts/                    the proof-generation / verification prompt set
  DEGENERATE_FILTER.md        how degenerate generations are detected and dropped

Dockerfile                    multi-stage build; inherits the baked SGLang runtime at /opt/pp
docker/                       entrypoint.sh (serve | bootstrap | validate) + inspect_config.py
install/                      install the inference venv on bare nodes + verify_models.py
runtime/                      how the baked SGLang runtime image is built / reused

sglang_patches/               SGLang source patches — attention sink, DFlash speculative decoding,
                              W4A8 — plus apply_patches.sh
vllm_patches/                 the same attention-sink + DFlash support as an installable vLLM plugin

benchmarks/                   IMO-2025 and ProofBench-V2 solution sets + grading prompts/scores
imo_2026_eval/                IMO-2026 results — raw submissions, typeset PDFs, grading + scores
proof_pilot_kaggle/           Kaggle competition materials — prompts, solutions, grades, markschemes

tests/                        the pytest suite (selector, traces, submission runner, patches)
.github/workflows/            container build + publish CI
```
