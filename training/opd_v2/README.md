# opd_v2 — On-Policy Distillation (the delivered OPD-32B model)

On-policy distillation of an OLMo3-sink student toward a DeepSeek-V4-Flash teacher, on the
student's *own* rollouts. This is the training stage that produces the delivered OPD-32B proof
model. It is a clean rewrite of an earlier prototype: four decoupled processes instead of a
co-located SPMD orchestrator, teacher hidden states moved over a shared filesystem (never through
the control plane), and the trainer exposed as a service.

## Architecture — four processes

| process | role |
|---|---|
| **rollout** | student sglang server (fp8, via `flash_rl/`), token-in / token-out |
| **teacher** | DeepSeek-V4-Flash sglang server (TP4); `/score` writes hidden states to shared FS and returns a small handle (`run_teacher_fs.sh`) |
| **trainer** | trainer-as-service: rank-0 HTTP ingress + all-rank gloo command loop; FSDP2 / HSDP (`src/opd_v2/trainer/`) |
| **orchestrator** | CPU-async driver: fills a trajectory buffer, POSTs `/train_step` with ids + handles, drives weight sync (`src/opd_v2/orchestrator.py`) |

Full-vocab distillation via a quantized-hidden codec + chunked fused-linear JSD; near-on-policy with
bounded staleness; whole trajectories trained un-windowed. **Agentic mode** (`src/opd_v2/agentic/`)
runs a pool-based semi-on-policy loop over the entire prove → verify → refine → select trajectory
(role prompts reused from `distill_gen/math_3r`), not just single-round generation.

## Layout
| path | purpose |
|---|---|
| `src/opd_v2/config.py` | all knobs (external paths via env vars) |
| `src/opd_v2/data_plane/` | rollout/teacher clients, load-aware pools, `produce`, scheduler |
| `src/opd_v2/trainer/` | trainer core (HSDP + JSD forward) + HTTP/gloo service |
| `src/opd_v2/orchestrator.py` | the driver process |
| `src/opd_v2/agentic/` | pool + roles + sampler + writeback for the agentic loop |
| `src/opd_v2/{codec,hidden_store,buffer,rollout_store}.py` | hidden codec/W_rot, shared-FS handle store, buffer, rollout dump |
| `flash_rl/` | fp8 rollout serving + small `update_weights_from_disk` loader patch |
| `run_teacher_fs.sh` | launch the teacher `/score` service |
| `examples/make_config.py` | generate a run config |
| `servers/`, `tests/` | mock servers + unit tests |

## Dependencies (sibling shared libs, wired via `sys.path` from `__file__`)
`../_common` (jsd_kernel, hidden_codec) · `../_vendor_opd` (`opd.*` leaf modules + `data_mix`) ·
`../stage1_v2/src` (train.py build/save helpers) · `../distill_gen/math_3r` (agentic role prompts; templates in the repo's `evaluation/prompts/ycchen_math_3r/`).

## Run (sketch)
Multi-node: start the teacher (`run_teacher_fs.sh`), one or more fp8 rollout replicas
(`flash_rl/run_rollout_fp8.sh`), the trainer service on all ranks, then the orchestrator — it
discovers the trainer endpoint and drives the loop. Configure paths via env (`DEEPSEEK_V4_FLASH`,
`SGLANG_SIF`, `STUDENT_PATH`) and generate a config with `python examples/make_config.py`.
`examples/run_mn.sh` automates all of the above inside one slurm allocation, and
`examples/run_agentic_mn_32b.sbatch` is the production 32B agentic submission script.

Regenerate `teacher_patch/http_server.py` (the `/score` + FS-handle `out_path` patch that
`run_teacher_fs.sh` bind-mounts) from pristine sglang sources with the canonical patch generator:
`python3 training/teacher_extract/_patch_sglang.py <sglang_src_dir> <out_dir>` — `<sglang_src_dir>`
holds the pristine `deepseek_v4.py`, `scheduler.py`, `scheduler_output_processor_mixin.py`,
`http_server.py` copied out of the sglang image; copy the patched `http_server.py` from
`<out_dir>` to `training/opd_v2/teacher_patch/http_server.py` (the other three patched files
live in `training/teacher_extract/_patched/`, regenerated via
`REPATCH=1 training/teacher_extract/run_in_container.sh`).
