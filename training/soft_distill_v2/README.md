# soft_distill_v2 — Offline Soft Distillation

Offline soft distillation shaped like SFT: whole proofs are packed into fixed large rows and the
student matches the teacher's full-vocab distribution via a chunked fused-linear JSD loss. This is
the offline step whose checkpoint initializes on-policy distillation (`../opd_v2`).

Each proof is kept **intact** (no windowing / no 4k truncation) and position-reset to 0 so RoPE
matches how the teacher hidden states were extracted (100% full-context exact). Row assignment is
deterministic, so every FSDP2 rank computes the same row list and takes a fixed slice —
synchronized collectives, exactly like stage-1's offline-packed data.

## Layout
| file | purpose |
|---|---|
| `data.py` | whole-proof packing into fixed micro_len rows (per-proof position reset) |
| `train_soft_distill.py` | SPMD `torchrun` trainer (single clean forward + chunked JSD) |
| `make_combined_index.py` | merge multiple teacher-hidden pools into one training index |
| `subsample_index.py` | token-budget random subsample (e.g. replay mix) |
| `bench_loss_heads.py`, `verify_jsd_soft.py` | JSD-kernel micro-bench + fp32 correctness check |

## Dependencies
`../_common` (jsd_kernel) · `../_vendor_opd` (`opd.*` + `data_mix`) · `../teacher_extract/dataprep`
(offpolicy_data index/window helpers) · `../stage1_v2/src` (train.py). The loss is the repo's chunked
`fused_linear_jsd_fp32_softmax` (bf16 GEMM + forced fp32 log-softmax/KL; never materializes `[B*T, V]`).

## Run (sketch)
```
torchrun --nproc_per_node=8 [--nnodes N ...] train_soft_distill.py \
    --index <index.jsonl> --student <student_ckpt> --out-dir outputs/softdistill \
    --epochs 3 --micro-len 200000 [--cpu-offload]      # 32B uses FSDP2 CPUOffloadPolicy
```
`--index` is the teacher-hidden index produced by `../teacher_extract/dataprep` (+ optionally merged
with `make_combined_index.py`).
