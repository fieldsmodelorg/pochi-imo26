#!/usr/bin/env python3
"""Production all-SWA no-dup FA3 DFlash trainer (multi-node, requeue-safe).

Reuses the validated model/loss code from ``experiments/nodup_fa3_train.py``
bit-for-bit (imported, not copied) and adds what the experiments trainer lacks
for long unattended multi-node runs:

  * periodic atomic checkpoint (latest.pt: bf16 model + full AdamW state +
    fp32 master weights + scheduler + step). Under DDP every rank holds the
    identical optimizer state (grads are all-reduced before step), so a
    single-rank full save is complete — unlike the per-rank FSDP1 trainer.
  * auto-resume from latest.pt in --output-dir: restores model, fp32 masters,
    AdamW moments, LR scheduler, and skips the already-consumed bins at the
    sampler level (no data is read for skipped bins).
  * SIGTERM/SIGUSR1 graceful stop with cross-rank consensus (a 1-scalar
    all_reduce per step) so slurm preemption/requeue resumes cleanly.

Launch (per node, under sbatch):
  torchrun --nnodes=$N --nproc_per_node=8 --rdzv_backend=c10d ... \
      training/dflash/nodup_fa3_train_prod.py --steps 12000 ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoConfig

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.abspath(os.path.join(HERE, "..")))   # training/ (olmo3_sink); after HERE so dflash.py wins over the dflash/ package

from data import L4Dataset, L4StripeSampler, epoch_resumable_iter, packed_collate_fn
from nodup_fa3_train import NoDupFA3Draft, compute_loss_backward_and_metrics
from optimizer import BF16Optimizer
from target_model import FSDP2TargetModel
from target_utils import TargetEmbeddingsAndHead
from train import _chunked_greedy
from tracker import create_tracker
from utils import print_on_rank0, print_with_rank


class ResumableStripeSampler(L4StripeSampler):
    """L4StripeSampler that can skip the first N already-consumed bins."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.skip = 0

    def __iter__(self):
        order = self._order()
        return iter(order[self.skip :].tolist())

    def __len__(self):
        # max(0, ...): epoch_resumable_iter maps a wrapped skip (skip >= per_rank)
        # to a within-epoch offset before the first iter(), but guard the transient
        # state so a stray len() never raises ValueError(__len__ < 0).
        return max(0, self.per_rank - self.skip)


def build_resumable_dataloader(data_path: str, seed: int, skip: int) -> DataLoader:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dataset = L4Dataset(data_path, max_bins=None)
    sampler = ResumableStripeSampler(len(dataset), rank, world_size, seed=seed, shuffle=True)
    sampler.skip = skip
    if rank == 0:
        print(
            f"[data] L4 {data_path}: {len(dataset)} bins x {dataset.micro_len} tokens, "
            f"{sampler.per_rank} bins/rank, skipping {skip} consumed bins",
            flush=True,
        )
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=2,
        collate_fn=packed_collate_fn,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )


def save_checkpoint(args, draft, opt, cfg, step: int, final: bool = False) -> None:
    """rank0 atomic save. Optimizer state is DDP-replicated, so this is complete."""
    name = "final.pt" if final else "latest.pt"
    path = os.path.join(args.output_dir, name)
    tmp = path + ".tmp"
    t0 = time.time()
    # Strip the ``_orig_mod.`` prefix torch.compile adds so checkpoints are
    # compile-agnostic: keys match the uncompiled draft loaded at resume (load
    # happens before compile) and stay backward-compatible with pre-compile ckpts.
    model_sd = {k.replace("_orig_mod.", ""): v for k, v in draft.state_dict().items()}
    torch.save(
        {
            "step": step,
            "model": model_sd,
            "optimizer": opt.optimizer_only_state_dict(),
            "fp32_master": [p.data.cpu() for p in opt.fp32_params],
            "scheduler": opt.state_dict(),
            "args": vars(args),
            "config": cfg.to_dict(),
        },
        tmp,
    )
    os.replace(tmp, path)
    print(f"[ckpt] saved {name} at step {step} in {time.time() - t0:.1f}s", flush=True)


def load_checkpoint(args, draft, opt) -> int:
    """Restore from latest.pt if present. Returns the step to start from (1-based)."""
    path = os.path.join(args.output_dir, "latest.pt")
    if not os.path.exists(path):
        return 1
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    draft.load_state_dict(ckpt["model"])
    with torch.no_grad():
        for p, saved in zip(opt.fp32_params, ckpt["fp32_master"]):
            p.data.copy_(saved.to(p.device))
    opt.load_optimizer_only_state_dict(ckpt["optimizer"])
    opt.load_state_dict(ckpt["scheduler"])
    start = ckpt["step"] + 1
    print_on_rank0(f"[ckpt] resumed from {path} (step {ckpt['step']}, next step {start})")
    return start


_STOP = {"flag": False}


def _signal_handler(signum, frame):
    _STOP["flag"] = True
    print(f"[signal] received {signum}, will checkpoint and exit at step boundary", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-model-path", default=os.environ.get("TARGET_MODEL_PATH", "outputs/stage1-v2-7b"))
    p.add_argument(
        "--draft-config-path",
        default=os.path.join(HERE, "configs", "dflash-olmo3-7b-8L-bs10-swa128-sink.json"),
    )
    p.add_argument("--train-data-path", default=os.environ.get("DFLASH_TRAIN_DATA", "data/l4-g2r05-ml12288-mc65536"))
    p.add_argument("--output-dir", default=os.environ.get("DFLASH_OUTPUT_DIR", "outputs/dflash-nodup-fa3-7b-v2"))
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=10)
    p.add_argument("--start-chunk-size", type=int, default=2048)
    p.add_argument("--max-starts-per-step", type=int, default=4096)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--loss-decay-gamma", type=float, default=5.0)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--metrics-mode", choices=["full", "loss_only"], default="full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-target-fsdp", action="store_true")
    p.add_argument(
        "--compile-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile each draft layer (fuses MLP/proj/norm around the opaque "
        "FA3 attention; ~1.4x faster draft + ~28%% less memory). --no-compile-layers to disable.",
    )
    p.add_argument("--dist-timeout", type=int, default=60)
    p.add_argument("--report-to", choices=["none", "wandb", "tensorboard"], default="none")
    p.add_argument("--wandb-project", type=str, default="dflash-olmo3")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-key", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)

    from datetime import timedelta

    dist.init_process_group("nccl", timeout=timedelta(minutes=args.dist_timeout))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGUSR1, _signal_handler)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    target_model = FSDP2TargetModel.from_pretrained(
        args.target_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="olmo3_sink_fa3",
        fsdp=not args.no_target_fsdp,
    )

    cfg = AutoConfig.from_pretrained(args.draft_config_path)
    cfg.block_size = args.block_size
    cfg.layer_types = ["sliding_attention"] * cfg.num_hidden_layers
    cfg.sliding_window = args.window_size
    cfg._attn_implementation = "fa3_nodup"
    draft = NoDupFA3Draft(cfg, window_size=args.window_size).cuda().to(torch.bfloat16)
    target_model.set_capture_layers(draft.target_layer_ids, capture_final_norm=True)

    components = TargetEmbeddingsAndHead.from_pretrained(args.target_model_path, device="cuda")
    with torch.no_grad():
        draft.mask_embed.copy_(components.embed_tokens.weight.mean(dim=0))
    lm_head = components.lm_head

    opt = BF16Optimizer(
        draft,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        total_optimizer_steps=args.steps,
    )

    # Resume BEFORE the DDP wrap: DDP broadcasts rank0 params at construction,
    # and every rank loads the identical checkpoint anyway.
    start_step = load_checkpoint(args, draft, opt)
    is_resume = start_step > 1
    if start_step > args.steps:
        print_on_rank0(f"[ckpt] start_step {start_step} > steps {args.steps}; nothing to do")
        dist.destroy_process_group()
        return

    # Compile after load, before DDP (see canonical_fa3_train_prod for rationale).
    if args.compile_layers:
        for i in range(len(draft.layers)):
            draft.layers[i] = torch.compile(draft.layers[i])
        print_on_rank0(f"[compile] torch.compile applied to {len(draft.layers)} draft layers")

    ddp = DDP(draft, device_ids=[local_rank])
    tracker = create_tracker(args, args.output_dir)
    if rank == 0 and os.path.exists(metrics_path) and not is_resume:
        os.remove(metrics_path)

    from cut_cross_entropy import linear_cross_entropy

    loader = build_resumable_dataloader(args.train_data_path, seed=args.seed, skip=start_step - 1)
    data_iter = epoch_resumable_iter(loader)

    print_on_rank0(
        f"No-dup FA3 prod train: steps={args.steps} start={start_step} "
        f"chunk={args.start_chunk_size} max_starts={args.max_starts_per_step} "
        f"B={args.block_size} W={args.window_size} world={dist.get_world_size()} "
        f"params={sum(p.numel() for p in draft.parameters()):,}"
    )

    stop_flag = torch.zeros(1, device="cuda")
    t_last = time.time()
    for step in range(start_step, args.steps + 1):
        torch.cuda.reset_peak_memory_stats()
        step_t0 = time.time()
        data = next(data_iter)
        input_ids = data["input_ids"].cuda(non_blocking=True)
        loss_mask = data["loss_mask"].cuda(non_blocking=True)
        document_ids = data["document_ids"].cuda(non_blocking=True)
        position_ids = data["position_ids"].cuda(non_blocking=True)

        target_t0 = time.time()
        target_out = target_model.generate_hidden_states(input_ids, loss_mask, position_ids)
        torch.cuda.synchronize()
        target_s = time.time() - target_t0
        greedy_t0 = time.time()
        greedy, _ = _chunked_greedy(components.lm_head.weight.data, target_out.last_hidden)
        torch.cuda.synchronize()
        greedy_s = time.time() - greedy_t0

        should_log = step == start_step or step % args.log_interval == 0 or step == args.steps
        compute_metrics = should_log and args.metrics_mode == "full"
        train_t0 = time.time()
        loss, metrics = compute_loss_backward_and_metrics(
            ddp,
            lm_head,
            input_ids,
            loss_mask,
            document_ids,
            position_ids,
            target_out.hidden_states,
            greedy,
            args.start_chunk_size,
            args.max_starts_per_step,
            args.window_size,
            args.loss_decay_gamma,
            compute_metrics,
            linear_cross_entropy,
        )
        torch.cuda.synchronize()
        train_s = time.time() - train_t0
        opt_t0 = time.time()
        opt.step()
        torch.cuda.synchronize()
        opt_s = time.time() - opt_t0

        # Cross-rank consensus on graceful stop (signal delivery is per-process).
        stop_flag.fill_(1.0 if _STOP["flag"] else 0.0)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        stopping = bool(stop_flag.item() > 0)

        peak_mem = torch.tensor(
            [torch.cuda.max_memory_allocated() / (1024**3)], device="cuda", dtype=torch.float32
        )
        dist.all_reduce(peak_mem, op=dist.ReduceOp.MAX)

        if should_log:
            keys = sorted(metrics.keys()) if metrics is not None else []
            packed = torch.stack([loss.detach()] + [metrics[k].detach() for k in keys])
            dist.all_reduce(packed)
            packed /= dist.get_world_size()
            log = {
                "step": step,
                "loss": packed[0].item(),
                "lr": opt.get_learning_rate(),
                "grad_norm": opt.get_grad_norm().item() if opt.get_grad_norm() is not None else None,
                "peak_mem_gib": peak_mem.item(),
                "time/target_s": target_s,
                "time/greedy_s": greedy_s,
                "time/train_s": train_s,
                "time/opt_s": opt_s,
                "time/step_s": time.time() - step_t0,
                "elapsed_since_last_log_s": time.time() - t_last,
            }
            for i, k in enumerate(keys):
                log[k] = packed[i + 1].item()
            t_last = time.time()
            if rank == 0:
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(log, sort_keys=True) + "\n")
                tracker.log(log, step=step)
                print(
                    f"Step {step}/{args.steps} | loss={log['loss']:.4f} "
                    f"acc={log.get('accuracy', 0):.4f} "
                    f"prefix={log.get('greedy/mean_prefix_len', 0):.2f} "
                    f"match={log.get('greedy/match_rate', 0):.3f} "
                    f"peak={log['peak_mem_gib']:.1f}GiB "
                    f"step={log['time/step_s']:.1f}s "
                    f"lr={log['lr']:.2e}",
                    flush=True,
                )

        if step % args.save_interval == 0 or stopping or step == args.steps:
            dist.barrier()
            if rank == 0:
                save_checkpoint(args, draft, opt, cfg, step, final=(step == args.steps))
            dist.barrier()
        if stopping:
            print_on_rank0(f"[signal] graceful stop after step {step}")
            break

    if rank == 0:
        with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        # Keep the self-contained modeling code next to the result for audit.
        shutil.copy(__file__, os.path.join(args.output_dir, "nodup_fa3_train_prod.py"))
        shutil.copy(
            os.path.join(HERE, "nodup_fa3_train.py"),
            os.path.join(args.output_dir, "nodup_fa3_train.py"),
        )
        shutil.copy(
            os.path.join(HERE, "fa3_nodup_attention.py"),
            os.path.join(args.output_dir, "fa3_nodup_attention.py"),
        )
    print_on_rank0("No-dup FA3 prod training done (or gracefully stopped).")
    tracker.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
