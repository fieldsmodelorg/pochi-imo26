# Copyright 2026 proof-pilot. Apache-2.0.
"""soft_distill_v2 trainer: stage1-style FSDP2 loop, whole-proof 200k rows, chunked JSD.

SPMD torchrun entry. Every rank builds the same deterministic row list (whole proofs
packed into micro_len bins) and takes a fixed `rows[rank::world]` slice, so every rank
runs the same number of forward_backward steps => synchronized FSDP collectives, like
stage1. The forward is a single clean wrapper (no layered monkeypatch): backbone hidden
-> gather target positions -> chunked fused-linear JSD against the teacher head W_rot.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import logging
import math
import os
import random
import sys
import time
import types
from pathlib import Path

local = int(os.environ.get("LOCAL_RANK", "0"))
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_sdv2_rank{local}")
Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for _p in (
    REPO / "training",
    REPO / "training" / "stage1_v2" / "src",
    REPO / "training" / "_vendor_opd",
    REPO / "training" / "_common",
    REPO / "training" / "soft_distill_v2",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train import build_model, setup_parallelism, save_consolidated  # noqa: E402  (stage1_v2)
from jsd_kernel import fused_linear_jsd_fp32_softmax  # noqa: E402
from opd.codec import build_w_rot  # noqa: E402
from opd.config import HID_DIM, TEACHER_PATH  # noqa: E402
from opd.loss import IGNORE  # noqa: E402
from data import assemble_kwargs, assemble_row_bin, build_rows  # noqa: E402

DEFAULT_STUDENT = REPO / "outputs" / "stage1-v2-7b-y256k-base"
DEFAULT_INDEX = REPO / "data" / "hidden" / "index.jsonl"  # override with --index


def soft_distill_forward(self, input_ids, position_ids, cu_seq_lens_q, cu_seq_lens_k,
                         max_length_q, max_length_k, opd_student_pos, opd_teacher_hidden,
                         opd_labels, opd_w_rot, opd_chunk_size):
    """Backbone forward (inside FSDP root unshard) + chunked CE+forward-KL JSD."""
    h = self.model(
        input_ids=input_ids, position_ids=position_ids,
        cu_seq_lens_q=cu_seq_lens_q, cu_seq_lens_k=cu_seq_lens_k,
        max_length_q=max_length_q, max_length_k=max_length_k,
    ).last_hidden_state
    sh = h[0, opd_student_pos]
    hard, soft, beta = self._sd_hard, self._sd_soft, self._sd_beta
    return fused_linear_jsd_fp32_softmax(
        sh.bfloat16(), self.lm_head.weight.bfloat16(),
        opd_teacher_hidden.bfloat16(), opd_w_rot.bfloat16(), opd_labels,
        weight_hard_loss=hard, weight_soft_loss=soft, beta=beta, ignore_index=IGNORE,
        temperature=1.0, compiled=False, chunk_size=opd_chunk_size, compute_ce_loss=hard != 0.0,
    )


def init_dist(cpu_offload=False):
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    torch.cuda.set_device(local)
    if world > 1 and not torch.distributed.is_initialized():
        # CPU-offloaded params/grads need a CPU collective backend (gloo) for the offloaded tensors.
        backend = "cpu:gloo,cuda:nccl" if cpu_offload else "nccl"
        torch.distributed.init_process_group(
            backend, device_id=torch.device(f"cuda:{local}"),
            timeout=_datetime.timedelta(minutes=int(os.environ.get("DIST_TIMEOUT_MIN", "60"))),
        )
    return rank, local, world


def setup_fsdp_cpu_offload(model, world, local):
    """Mirror stage1_v2 setup_parallelism's mesh, + CPUOffloadPolicy (params+grad+optstate -> CPU).

    Frees ~all GPU resident so 32B fits longer context (128k). fwd/bwd stay full-speed (param
    streaming overlaps with compute); only the AdamW step runs on CPU. model.to(cuda) BEFORE
    fully_shard so the root unshards embed/lm_head to GPU on forward.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
    gpn = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count())) or 1
    n_nodes = world // gpn
    if n_nodes > 1:
        mesh = init_device_mesh("cuda", (n_nodes, gpn), mesh_dim_names=("replicate", "shard"))
        logging.getLogger("sdv2").info("HSDP+CPUOffload mesh: replicate=%d x shard=%d", n_nodes, gpn)
    else:
        mesh = init_device_mesh("cuda", (gpn,), mesh_dim_names=("shard",))
        logging.getLogger("sdv2").info("FSDP2+CPUOffload mesh: shard=%d", gpn)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    off = CPUOffloadPolicy(pin_memory=True)
    model = model.to(f"cuda:{local}")
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp, offload_policy=off)
    fully_shard(model, mesh=mesh, mp_policy=mp, offload_policy=off)
    return model


def init_wandb(args, *, world, n_rows, steps_per_epoch):
    """Rank-0 wandb init. Best-effort: failure falls back to stdout-only, never kills training."""
    if args.no_wandb or args.wandb_mode == "disabled" or os.environ.get("OPD_WANDB", "1") == "0":
        return None
    try:
        import wandb
        jid = os.environ.get("SLURM_JOB_ID", "")
        run_id = os.environ.get("WANDB_RUN_ID") or (f"sdv2-{jid}" if jid else None)
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"sdv2-{jid or 'local'}",
            id=run_id,
            resume=os.environ.get("WANDB_RESUME") or ("allow" if run_id else None),
            mode=args.wandb_mode,
            config={
                "micro_len": args.micro_len, "truncate_len": args.truncate_len,
                "world": world, "n_rows": n_rows, "steps_per_epoch": steps_per_epoch,
                "max_steps": args.max_steps, "lr": args.lr, "warmup_steps": args.warmup_steps,
                "chunk_size": args.chunk_size, "grad_ckpt": args.grad_ckpt,
                "master_dtype": args.master_dtype, "seed": args.seed,
                "student_path": str(args.student_path), "teacher_path": str(args.teacher_path),
                "hard_weight": args.hard_weight, "soft_weight": args.soft_weight, "beta": args.beta,
                "cpu_offload": args.cpu_offload, "matmul": "bf16",
                "slurm_job_id": jid, "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
            },
        )
        logging.getLogger("sdv2").info("wandb run: %s", getattr(run, "url", "(offline)"))
        return run
    except Exception:
        logging.getLogger("sdv2").exception("wandb init failed; continuing stdout-only")
        return None


def build_sched(opt, warmup, total):
    def lr_lambda(s):
        if warmup > 0 and s < warmup:
            return (s + 1) / warmup
        prog = min(1.0, max(0.0, (s - warmup) / max(1, total - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--student-path", type=Path, default=DEFAULT_STUDENT)
    ap.add_argument("--teacher-path", type=Path, default=Path(TEACHER_PATH))
    ap.add_argument("--micro-len", type=int, default=204800)
    ap.add_argument("--truncate-len", type=int, default=200000)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="explicit step cap; 0 = derive from --epochs * steps_per_epoch")
    ap.add_argument("--epochs", type=int, default=0,
                    help="full passes over the row list; sets max_steps = epochs * steps_per_epoch")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="HF checkpoint dir (deliverable). None = no save (smoke/bench). Final -> root, "
                         "intermediate epochs -> {out_dir}/ep{e}.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="extra step-based save to {out_dir}/step{N} (0 = off; epoch ends + final always save)")
    ap.add_argument("--tokenizer-path", type=Path, default=None, help="tokenizer source (default = --student-path)")
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--adam-beta1", type=float, default=0.9)
    ap.add_argument("--adam-beta2", type=float, default=0.95)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=4096)
    # Loss weights. Default = canonical pure-soft forward-KL:
    # hard=0 soft=1 beta=0. Set both >0 (e.g. hard=soft=0.5) to mix in a CE anchor.
    ap.add_argument("--hard-weight", type=float, default=0.0, help="CE weight on teacher tokens (0 = pure soft)")
    ap.add_argument("--soft-weight", type=float, default=1.0, help="soft JSD/KL weight")
    ap.add_argument("--beta", type=float, default=0.0, help="Liger JSD beta: 0=forward-KL, 0.5=JSD, 1=reverse-KL")
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--master-dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--cpu-offload", action="store_true",
                    help="FSDP2 CPUOffloadPolicy: params+grad+optstate -> CPU (for 32B long-context). "
                         "Needs gloo CPU backend + CPU AdamW step. fwd/bwd stay full-speed.")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--no-wandb", action="store_true", help="disable rank-0 wandb (env OPD_WANDB=0 also disables)")
    ap.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "soft-distill-v2"))
    ap.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"),
                    choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-run-name", default=os.environ.get("WANDB_RUN_NAME", ""))
    args = ap.parse_args()

    rank, local, world = init_dist(cpu_offload=args.cpu_offload)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [rank{rank}] %(message)s")
    log = logging.getLogger("sdv2")
    dev = torch.device("cuda", local)

    rows = build_rows(args.index, micro_len=args.micro_len, truncate_len=args.truncate_len, seed=args.seed)
    usable = (len(rows) // world) * world
    if usable == 0:
        raise RuntimeError(f"not enough rows: {len(rows)} < world {world}")
    my_rows = rows[rank:usable:world]
    steps_per_epoch = usable // world
    if args.epochs > 0:
        args.max_steps = args.epochs * steps_per_epoch
    if args.max_steps <= 0:
        raise SystemExit("set --epochs (>0) or --max-steps (>0)")
    total_epochs = args.epochs if args.epochs > 0 else math.ceil(args.max_steps / max(1, steps_per_epoch))
    if rank == 0:
        n_proofs = sum(len(r) for r in rows)
        fills = [sum(s.seq_len for s in r) for r in rows]
        log.info("rows=%d usable=%d steps_per_epoch=%d max_steps=%d epochs=%d proofs=%d "
                 "fill mean=%.0f/%d (%.1f%%) per_rank=%d",
                 len(rows), usable, steps_per_epoch, args.max_steps, total_epochs, n_proofs,
                 sum(fills) / len(fills), args.micro_len,
                 100 * sum(fills) / (len(fills) * args.micro_len), len(my_rows))
    wb = init_wandb(args, world=world, n_rows=len(rows), steps_per_epoch=steps_per_epoch) if rank == 0 else None

    mp = torch.float32 if args.master_dtype == "fp32" else torch.bfloat16
    model = build_model(str(args.student_path), attn="olmo3_sink_fa3", liger=True, master_dtype=mp)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False})
    model.train()
    model = setup_fsdp_cpu_offload(model, world, local) if args.cpu_offload else setup_parallelism(model, world, local)
    model.forward = types.MethodType(soft_distill_forward, model)
    model._sd_hard, model._sd_soft, model._sd_beta = args.hard_weight, args.soft_weight, args.beta
    if world > 1:
        from torch.distributed.tensor import DTensor
        if not isinstance(next(model.parameters()), DTensor):
            raise RuntimeError("FSDP2 did not engage")

    # CPU-offloaded params -> optimizer step runs on CPU (fused is CUDA-only); foreach is the fast CPU path.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=not args.cpu_offload,
                            foreach=args.cpu_offload or None,
                            betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    sched = build_sched(opt, args.warmup_steps, args.max_steps)
    w_rot, _ = build_w_rot(str(args.teacher_path), HID_DIM, device=str(dev))
    w_rot = w_rot.to(dev)
    if rank == 0:
        log.info("built: student=%s grad_ckpt=%s master=%s resident=%.1fGB",
                 args.student_path, args.grad_ckpt, args.master_dtype,
                 torch.cuda.memory_allocated() / 1e9)
    tok = None
    if args.out_dir is not None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(args.tokenizer_path or args.student_path))

    def maybe_save(step_done):
        """Collective HF save (all ranks). step_done = number of completed steps (step+1)."""
        if args.out_dir is None:
            return
        is_final = step_done >= args.max_steps
        is_epoch_end = steps_per_epoch and step_done % steps_per_epoch == 0
        if is_final:
            dst = args.out_dir
        elif is_epoch_end:
            dst = args.out_dir / f"ep{step_done // steps_per_epoch}"
        elif args.save_every and step_done % args.save_every == 0:
            dst = args.out_dir / f"step{step_done}"
        else:
            return
        if rank == 0:
            log.info("saving HF checkpoint (step %d) -> %s", step_done, dst)
        save_consolidated(model, tok, str(dst), world, rank)

    if world > 1:
        torch.distributed.barrier()

    t_train = time.time()
    for step in range(args.max_steps):
        if step > 0 and steps_per_epoch and step % steps_per_epoch == 0:
            # Deterministic per-epoch reshuffle. len(my_rows) is identical on every rank, so the
            # reorder never changes step counts => FSDP collectives stay in lockstep.
            random.Random(args.seed * 1_000_003 + step // steps_per_epoch).shuffle(my_rows)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        row = my_rows[step % len(my_rows)]
        bin_ = assemble_row_bin(row, micro_len=args.micro_len)
        kw = assemble_kwargs(bin_, dev, w_rot, args.chunk_size)
        t_data = time.time()

        loss = model(**kw)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        gnorm = gnorm.full_tensor().item() if hasattr(gnorm, "full_tensor") else float(gnorm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0

        # per-rank stats; reduce for a global view
        real_tokens = sum(seg.prompt_len + seg.n_t for seg in bin_.segments)
        target_tokens = sum(seg.n_t for seg in bin_.segments)
        n_segs = len(bin_.segments)
        stat = torch.tensor([float(loss.detach()), real_tokens, target_tokens, n_segs], device=dev)
        if world > 1:
            torch.distributed.all_reduce(stat, op=torch.distributed.ReduceOp.SUM)
        loss_g = stat[0].item() / world
        real_g = int(stat[1].item())
        tgt_g = int(stat[2].item())
        segs_g = int(stat[3].item())
        if rank == 0:
            peak = torch.cuda.max_memory_allocated() / 1e9
            lr_now = sched.get_last_lr()[0]
            log.info(
                "step=%d loss=%.4f lr=%.2e gnorm=%.2f | segs=%d real_tok=%d target_tok=%d "
                "| target_tok/s/gpu=%.0f real_tok/s/gpu=%.0f | dt=%.1fs data=%.1fs peak=%.1fGB",
                step, loss_g, lr_now, gnorm, segs_g, real_g, tgt_g,
                tgt_g / world / dt, real_g / world / dt, dt, t_data - t0, peak,
            )
            if wb is not None:
                wb.log({
                    "train/loss": loss_g, "train/lr": lr_now, "train/grad_norm": gnorm,
                    "perf/target_tok_s_gpu": tgt_g / world / dt, "perf/real_tok_s_gpu": real_g / world / dt,
                    "perf/peak_gb": peak, "perf/seconds": dt, "time/data_seconds": t_data - t0,
                    "tokens/target_global": tgt_g, "tokens/real_global": real_g, "tokens/segments_global": segs_g,
                    "epoch": step // steps_per_epoch,
                }, step=step)

        maybe_save(step + 1)  # collective: all ranks (epoch ends + final + optional --save-every)

    if rank == 0 and wb is not None:
        wb.finish()
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if rank == 0:
        log.info("DONE %d steps in %.1fs -- training ran smoothly", args.max_steps, time.time() - t_train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
