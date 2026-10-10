#!/usr/bin/env python3
"""Production canonical-convention DFlash trainer (multi-node, requeue-safe).

Same harness as ``nodup_fa3_train_prod.py`` (atomic latest.pt, auto-resume with
sampler-skip, per-step SIGTERM cross-rank consensus) — those helpers are imported
verbatim — but it trains the **canonical** block convention via
``experiments/canonical_fa3_train.py`` (block-0 = verified token's target
embedding, predict block_size-1, context strictly before start). The resulting
weights are drop-in for the all-SWA sglang DFlash serving stack at one target
forward/cycle.

Warm-start: ``--init-from <no-dup final.pt>`` loads the no-dup checkpoint's model
weights as init (params are identical between the two draft classes), so the run
keeps all learned representations and only re-learns the block-0 convention. If a
``latest.pt`` already exists in --output-dir, resume takes precedence and
--init-from is ignored.

Launch (per node, under sbatch):
  torchrun --nnodes=$N --nproc_per_node=8 --rdzv_backend=c10d ... \
      training/dflash/canonical_fa3_train_prod.py --steps 12000 \
      --init-from outputs/dflash-nodup-fa3-7b-v2-32g-s12000/final.pt ...
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
from transformers import AutoConfig

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, "experiments"))
sys.path.insert(0, HERE)
sys.path.append(os.path.abspath(os.path.join(HERE, "..")))   # training/ (olmo3_sink); after HERE so dflash.py wins over the dflash/ package

from canonical_fa3_train import (
    CanonicalFA3Draft,
    compute_loss_backward_and_metrics_canonical,
)
from optimizer import BF16Optimizer
from target_model import FSDP2TargetModel
from target_utils import TargetEmbeddingsAndHead
from train import _chunked_greedy
from tracker import create_tracker
from utils import print_on_rank0, print_with_rank
from data import epoch_resumable_iter

# Reuse the requeue-safe harness verbatim from the no-dup prod trainer.
from nodup_fa3_train_prod import (
    _STOP,
    _signal_handler,
    build_resumable_dataloader,
    load_checkpoint,
    save_checkpoint,
)


def load_init_weights(path: str, draft, opt) -> None:
    """Warm-start: load model weights only from an external checkpoint (no
    optimizer/scheduler/step). Params are identical between NoDupFA3Draft and
    CanonicalFA3Draft, so the no-dup final.pt loads 1:1.

    Critically, re-sync the optimizer's fp32 master copies to the loaded weights:
    ``BF16Optimizer.__init__`` snapshots fp32 masters from the (random) init at
    construction, and ``opt.step`` writes the master back into the model. Without
    this re-sync the first step would overwrite the warm-started weights with the
    random master (otherwise accuracy collapses to random within a few steps)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = draft.load_state_dict(sd, strict=False)
    bad = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    assert not bad, f"warm-start missing params: {bad}"
    assert not unexpected, f"warm-start unexpected params: {unexpected}"
    with torch.no_grad():
        for p, mp in zip(opt.model_params, opt.fp32_params):
            mp.data.copy_(p.data.to(torch.float32))
    print_on_rank0(f"[init] warm-started draft from {path} (+ fp32 master re-synced)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-model-path", default=os.environ.get("TARGET_MODEL_PATH", "outputs/stage1-v2-7b"))
    p.add_argument(
        "--draft-config-path",
        default=os.path.join(HERE, "configs", "dflash-olmo3-7b-8L-bs10-swa128-sink.json"),
    )
    p.add_argument("--train-data-path", default=os.environ.get("DFLASH_TRAIN_DATA", "data/l4-g2r05-ml12288-mc65536"))
    p.add_argument("--output-dir", default=os.environ.get("DFLASH_OUTPUT_DIR", "outputs/dflash-canonical-7b-v2"))
    p.add_argument("--init-from", default=None, help="warm-start model weights from this final.pt")
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument(
        "--block-size",
        type=int,
        default=11,
        help="TOTAL slots = 1 verified block-0 + predicted mask slots; predicts block_size-1.",
    )
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
    draft = CanonicalFA3Draft(cfg, window_size=args.window_size).cuda().to(torch.bfloat16)
    target_model.set_capture_layers(draft.target_layer_ids, capture_final_norm=True)

    components = TargetEmbeddingsAndHead.from_pretrained(args.target_model_path, device="cuda")
    with torch.no_grad():
        draft.mask_embed.copy_(components.embed_tokens.weight.mean(dim=0))
    lm_head = components.lm_head
    embed_tokens = components.embed_tokens

    opt = BF16Optimizer(
        draft,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        total_optimizer_steps=args.steps,
    )

    # Resume (full state) takes precedence over warm-start init.
    start_step = load_checkpoint(args, draft, opt)
    is_resume = start_step > 1
    if not is_resume and args.init_from:
        load_init_weights(args.init_from, draft, opt)
    if start_step > args.steps:
        print_on_rank0(f"[ckpt] start_step {start_step} > steps {args.steps}; nothing to do")
        dist.destroy_process_group()
        return

    # Compile AFTER load (weights/optimizer refs are on the uncompiled modules) and
    # BEFORE DDP. torch.compile wraps each layer's forward (MLP/proj/qk-norm/residual
    # fuse) and treats the FA3 attention autograd.Function as opaque. Params are not
    # replaced, so the BF16Optimizer refs stay valid. Checkpoint keys are kept
    # compile-agnostic by stripping ``_orig_mod.`` in save_checkpoint.
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
        f"Canonical FA3 prod train: steps={args.steps} start={start_step} "
        f"chunk={args.start_chunk_size} max_starts={args.max_starts_per_step} "
        f"block_size={args.block_size} (predict {args.block_size - 1}) W={args.window_size} "
        f"world={dist.get_world_size()} init_from={args.init_from if not is_resume else '(resumed)'} "
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
        loss, metrics = compute_loss_backward_and_metrics_canonical(
            ddp,
            lm_head,
            embed_tokens,
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
        for fn in ("canonical_fa3_train_prod.py",):
            shutil.copy(__file__, os.path.join(args.output_dir, fn))
        shutil.copy(
            os.path.join(HERE, "experiments", "canonical_fa3_train.py"),
            os.path.join(args.output_dir, "canonical_fa3_train.py"),
        )
        shutil.copy(
            os.path.join(HERE, "nodup_fa3_train.py"),
            os.path.join(args.output_dir, "nodup_fa3_train.py"),
        )
        shutil.copy(
            os.path.join(HERE, "fa3_nodup_attention.py"),
            os.path.join(args.output_dir, "fa3_nodup_attention.py"),
        )
    print_on_rank0("Canonical FA3 prod training done (or gracefully stopped).")
    tracker.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
