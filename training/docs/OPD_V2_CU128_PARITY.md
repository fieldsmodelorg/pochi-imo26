# OPD v2 cu128 container — components, patches, and cu13↔0.5.14 parity

Container: **`ycchen-opd`** (cu128, B200-target). Two isolated venvs, all 4 OPD processes.
Student = `chankhavu/yccchen-olmo3-deploy`; teacher = DeepSeek-V4-Flash.

## 1. Components
| process | venv | stack |
|---|---|---|
| trainer + orchestrator | base `/app/.venv` | torch 2.10+cu128 · transformers **5.9.0** (her pin) · `olmo3_sink_fa2` · liger |
| rollout (student) | `/opt/venv/serve` | **sglang 0.5.14** + her patched `olmo2.py`/`loader.py`/`load_config.py`/`server_args.py` · triton sink |
| teacher (DeepSeek-V4) | `/opt/venv/serve` | sglang 0.5.14 + re-anchored hidden-extract (`_patch_sglang_514.py`, 5 patches) |

Exact pins: `training/docker/constraints-train.txt` (trainer) and `training/docker/constraints-serve.txt` (serve).

## 2. Trainer sink — `olmo3_sink_fa2`
FA3 is Hopper-only and stock transformers `flash_attention_2` **silently drops the sink** (only FA3 `s_aux` / FA4 `learnable_sink` are honored). So the sink runs as a **post-correction re-normalization on stock FA2** (`o_sink = out·exp(lse−logaddexp(lse,sink))`), wrapped in a `torch.library.custom_op` (torch.compile-safe), with an in-place rescale (no extra memory). The math matches eager attention-with-sink, so the OPD JSD loss is unchanged; it handles packed varlen batches without cross-document leakage.

## 3. Yi-Chia's cu13 version vs the cu128 0.5.14 version — the differences
| aspect | her (cu13, sglang 0.5.14.dev / SIF 0.5.12.post1) | this image (cu128, sglang 0.5.14 release) | why |
|---|---|---|---|
| CUDA | 13.0 (won't run on cu128 cluster) | 12.8 | AI2 ceiling; cu13 > cu128 |
| trainer sink | patched-FA3 in-kernel (Hopper) | `olmo3_sink_fa2` post-correction on stock FA2 | B200 has no FA3 |
| rollout sink model | `olmo2.py` (her patch) | **same file, byte-identical** | copied hers |
| rollout attn backend | **FA3** (H200 default; her `run_rollout_fp8.sh` does NOT pin `--attention-backend`, and its comments optimize KV dtype to *keep FA3*: `fp8_e4m3` keeps FA3, `fp8_e5m2` would force triton). sglang's FA3 backend honors the sink. | **triton** (B200 has no FA3; triton is the Blackwell-capable backend that still honors the sink — flashinfer drops it) | **a real, forced substitution**. Correctness rests on triton applying the sink identically to FA3 (0.5.14 native radix `sinks` + triton sink path). |
| teacher patches | `_patch_sglang.py` on 0.5.12 layout | `_patch_sglang_514.py` re-anchored (5 patches) | 0.5.14 moved files; only where the patches attach changed, not their logic |
| teacher emit-rank | `self.attn_tp_rank` (Scheduler) | `get_attention_tp_rank()` | 0.5.14 moved the emit into a frozen-slots dataclass with no `attn_tp_rank`; the global lookup keeps only attention-TP rank 0 writing the spool file |
| teacher chunk truncation | owned by her 0.5.12 output path | +5th patch to `output_streamer.py` | 0.5.14 truncates the emitted hidden to `finished_len` (= 1 for `/score`, `max_new_tokens=1`) while the patch appends one entry per prefill chunk; the 5th patch skips that truncation in spool mode so prompts longer than `--chunked-prefill-size` (11264) keep every chunk |
| `wo_a` precision | dequant-to-bf16 (her default) | **`SGLANG_OPT_FP8_WO_A_GEMM=0`** → dequant-to-bf16 | fidelity + parity; fp8 GEMM would bias the teacher hidden (OPD policy bias) |
| rollout SWA KV pool | `flash_rl/patches/model_config.py` (cu13-patched file, `--bind`) | her idempotent `apply_swa_patch.py` run against **stock 0.5.14** `model_config.py` at build (`is_hybrid_swa_model(Olmo3Sink)=True`) | avoids cu13→cu128 version-skew of copying her 73KB file; adds Olmo3Sink to `hybrid_swa_archs` so `--swa-full-tokens-ratio` isn't silently inert → 140k rollout fits |
| teacher MoE backend | marlin (Hopper fp4) | auto-selected per arch in `run_teacher.sh`: sm_100 → `flashinfer_mxfp4`, else `auto` → marlin | sglang's `auto` picks an fp8 runner on sm_100 that cannot run V4's fp4 experts |

The multi-node launcher is `training/docker/launch/run_mn_cu128.sh` (+ `opd_v2.sbatch` for pyxis/enroot
slurm); `run_mn_nodes.sh` is the scheduler-free equivalent with the same role logic.
