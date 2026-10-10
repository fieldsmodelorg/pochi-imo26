#!/usr/bin/env python3
"""DFlash draft training for olmo3_sink targets (FSDP2 frozen target + DDP draft).

Ported from dflash-train (gpt-oss-120b era) with proof-pilot adaptations:
  - target: olmo3_sink checkpoint (FSDP2 per-layer sharding, hook capture,
    in-kernel sink FA3, packing via per-document position_ids)
  - draft: Olmo3DFlashDraftModel (OLMo3-style, SWA+sink, GQA)
  - data: L4 pre-packed bins read directly (no runtime packing)
  - mask_embed init: mean of the target embedding table (the mask token's OMP
    row is transplant garbage — usable as an id, not as an init)

Run (see examples/):
  torchrun --standalone --nproc_per_node=8 train.py \
      --target-model-path outputs/stage1-7b-4n \
      --draft-config-path training/dflash/configs/dflash-olmo3-7b-8L-bs10-swa128-sink.json \
      --train-data-path data/l4-g2r05-ml12288-mc65536 ...
"""

import argparse
import itertools
import logging
import os
import shutil
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoConfig

from data import build_dataloader
from dflash import OnlineDFlashModel
from draft_model_olmo3 import Olmo3DFlashDraftModel
from optimizer import BF16Optimizer
from target_model import FSDP2TargetModel
from target_utils import TargetEmbeddingsAndHead
from tracker import create_tracker
from utils import get_last_checkpoint, print_on_rank0, print_with_rank


@torch.no_grad()
def _chunked_greedy(
    lm_head_weight: torch.Tensor,
    last_hidden: torch.Tensor,
    chunk_size: int = 1024,
    input_ids: torch.Tensor = None,
) -> tuple:
    """Greedy tokens (argmax over the full vocab) computed chunk-by-chunk so the
    (B, S, V) logits tensor never materializes. Optionally also P(sample token)
    for --greedy-match-threshold."""
    B, S, H = last_hidden.shape
    greedy = torch.empty(B, S, dtype=torch.long, device=last_hidden.device)
    sample_probs = (
        torch.zeros(B, S, device=last_hidden.device) if input_ids is not None else None
    )
    for i in range(0, S, chunk_size):
        chunk = last_hidden[:, i : i + chunk_size]
        logits = torch.nn.functional.linear(chunk, lm_head_weight)  # (B, chunk, V)
        greedy[:, i : i + chunk_size] = logits.argmax(dim=-1)

        if sample_probs is not None:
            actual_chunk = logits.size(1)
            usable = min(actual_chunk, S - i - 1)
            if usable > 0:
                next_tokens = input_ids[:, i + 1 : i + 1 + usable]
                tok_logit = logits[:, :usable].gather(
                    -1, next_tokens.unsqueeze(-1)
                ).squeeze(-1)
                lse = torch.logsumexp(logits[:, :usable], dim=-1)
                sample_probs[:, i : i + usable] = (tok_logit - lse).exp()
        del logits
    return greedy, sample_probs


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash draft for olmo3_sink")

    g = parser.add_argument_group("model")
    g.add_argument("--target-model-path", type=str, required=True)
    g.add_argument("--draft-config-path", type=str, required=True)
    g.add_argument("--mask-token-id", type=int, default=128000,
                    help="DeepSeek <place_holder_no_0>; census-verified zero "
                         "occurrences in the L4 mix (scripts/l4_mask_census.py).")
    g.add_argument("--attention-backend", type=str, default="flex_attention",
                    choices=["eager", "sdpa", "flex_attention"])
    g.add_argument("--num-anchors", type=int, default=4096)
    g.add_argument("--loss-decay-gamma", type=float, default=5.0)
    g.add_argument("--greedy-match-threshold", type=float, default=None)
    g.add_argument("--use-cce", action="store_true", default=False)
    g.add_argument("--focal-gamma", type=float, default=None)
    g.add_argument("--gradient-checkpointing", action="store_true", default=False)
    g.add_argument("--causal", action="store_true", default=False)
    g.add_argument("--target-attn-implementation", type=str, default="olmo3_sink_fa3")
    g.add_argument("--no-target-fsdp", action="store_true", default=False,
                    help="Replicate the target per rank instead of FSDP2 sharding "
                         "(fine for 7B dev on H200).")

    g = parser.add_argument_group("data")
    g.add_argument("--train-data-path", type=str, required=True, help="L4 root dir")
    g.add_argument("--max-bins", type=int, default=None,
                    help="Use only the first N bins (global shuffle at build time "
                         "makes any prefix iid) — dev-scale runs.")
    g.add_argument("--dataloader-num-workers", type=int, default=2)

    g = parser.add_argument_group("training")
    g.add_argument("--num-epochs", type=int, default=1)
    g.add_argument("--batch-size", type=int, default=1)
    g.add_argument("--accumulation-steps", type=int, default=1)
    g.add_argument("--learning-rate", type=float, default=6e-4)
    g.add_argument("--pretrained-lr", type=float, default=None)
    g.add_argument("--warmup-ratio", type=float, default=0.04)
    g.add_argument("--warmup-steps", type=int, default=None)
    g.add_argument("--max-grad-norm", type=float, default=1.0)
    g.add_argument("--total-optimizer-steps", type=int, default=None)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--resume", action="store_true")
    g.add_argument("--ckpt-dir", type=str, default=None)

    g = parser.add_argument_group("output")
    g.add_argument("--output-dir", type=str, required=True)
    g.add_argument("--log-interval", type=int, default=20)
    g.add_argument("--save-interval", type=int, default=1000)
    g.add_argument("--save-latest-interval", type=int, default=200)

    g = parser.add_argument_group("distributed")
    g.add_argument("--dist-timeout", type=int, default=60)

    g = parser.add_argument_group("tracker")
    g.add_argument("--report-to", type=str, default="none",
                    choices=["none", "wandb", "tensorboard"])
    g.add_argument("--wandb-project", type=str, default="dflash-olmo3")
    g.add_argument("--wandb-name", type=str, default=None)
    g.add_argument("--wandb-key", type=str, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_models(args):
    print_on_rank0(f"Loading target model from {args.target_model_path}")
    target_model = FSDP2TargetModel.from_pretrained(
        args.target_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.target_attn_implementation,
        fsdp=not args.no_target_fsdp,
    )

    draft_config = AutoConfig.from_pretrained(args.draft_config_path)
    target_config = AutoConfig.from_pretrained(args.target_model_path)
    assert draft_config.num_target_layers == target_config.num_hidden_layers, (
        f"draft config num_target_layers={draft_config.num_target_layers} != "
        f"target num_hidden_layers={target_config.num_hidden_layers}"
    )
    assert getattr(draft_config, "target_hidden_size", draft_config.hidden_size) == \
        target_config.hidden_size
    assert draft_config.vocab_size == target_config.vocab_size

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}
    draft_config._attn_implementation = args.attention_backend

    draft_model = Olmo3DFlashDraftModel(draft_config).cuda().to(torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids, capture_final_norm=True)

    print_on_rank0(
        f"Draft: block_size={draft_config.block_size}, layers={draft_config.num_hidden_layers}, "
        f"layer_types={draft_config.layer_types}, swa={draft_config.sliding_window}, "
        f"sink={draft_config.dflash_config.get('use_attention_sink')}, "
        f"target_layers={draft_model.target_layer_ids}, "
        f"params={sum(p.numel() for p in draft_model.parameters()):,}"
    )
    return target_model, draft_model


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(args, epoch, step, dflash_model, draft_model, optimizer, save_dirs=None):
    """Save model + optimizer checkpoint (DDP: replicated, rank-0 writes all).

    Layout per checkpoint dir:
        config.json, model.safetensors, draft_model_olmo3.py
        training_state.pt        (epoch/step/scheduler/rng)
        optimizer.pt             (AdamW state over fp32 masters; identical on
                                  every rank under DDP -> single file, no
                                  world_size coupling)
    """
    if save_dirs is None:
        save_dirs = [os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")]

    rank = dist.get_rank()

    if rank == 0:
        draft_state_dict = {
            k: v.detach().cpu() for k, v in draft_model.state_dict().items()
        }
        training_state = {
            "epoch": epoch,
            "global_step": step,
            "args": args,
            "rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            **optimizer.state_dict(),
        }
        modeling_src = os.path.join(os.path.dirname(__file__), "draft_model_olmo3.py")
        for save_dir in save_dirs:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)
            if os.path.exists(modeling_src):
                shutil.copy(modeling_src, os.path.join(save_dir, "draft_model_olmo3.py"))
            torch.save(
                {"optimizer_state_dict": optimizer.optimizer_only_state_dict()},
                os.path.join(save_dir, "optimizer.pt"),
            )
            print_on_rank0(f"Saved checkpoint to {save_dir}")
    dist.barrier()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)

    from datetime import timedelta
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=args.dist_timeout))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    target_model, draft_model = build_models(args)

    # Checkpoint discovery
    draft_model_last_checkpoint = None
    if args.ckpt_dir is not None and os.path.isdir(args.ckpt_dir):
        draft_model_last_checkpoint = args.ckpt_dir
        print_on_rank0(f"Using checkpoint: {draft_model_last_checkpoint}")
    if args.resume and os.path.isdir(args.output_dir):
        ckpt = get_last_checkpoint(args.output_dir, prefix=r"epoch_\d+_step")
        if ckpt:
            draft_model_last_checkpoint = ckpt
            print_on_rank0(f"Resuming from: {draft_model_last_checkpoint}")

    resume_state = None
    _mask_embed_missing = True
    if draft_model_last_checkpoint:
        loaded_model = Olmo3DFlashDraftModel.from_pretrained(
            draft_model_last_checkpoint, dtype=torch.bfloat16
        )
        import glob as _glob
        _ckpt_files = _glob.glob(os.path.join(draft_model_last_checkpoint, "*.safetensors"))
        if _ckpt_files:
            from safetensors import safe_open
            with safe_open(_ckpt_files[0], framework="pt") as _sf:
                _mask_embed_missing = "mask_embed" not in _sf.keys()
        draft_model.load_state_dict(loaded_model.state_dict(), strict=False)
        del loaded_model
        print_on_rank0(
            f"Loaded draft weights from checkpoint (mask_embed_missing={_mask_embed_missing})"
        )

        training_state_path = os.path.join(draft_model_last_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            resume_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
            print_on_rank0(
                f"Will resume from epoch {resume_state['epoch']}, step {resume_state['global_step']}"
            )

    mask_token_id = args.mask_token_id
    print_on_rank0(f"mask_token_id: {mask_token_id}")
    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids

    # Data
    train_dataloader = build_dataloader(
        data_path=args.train_data_path,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        seed=args.seed,
        shuffle=True,
        max_bins=args.max_bins,
    )

    bins_per_epoch = len(train_dataloader) * args.batch_size
    if args.total_optimizer_steps is not None:
        total_optimizer_steps = args.total_optimizer_steps
    else:
        total_optimizer_steps = (args.num_epochs * bins_per_epoch) // args.accumulation_steps
    print_on_rank0(
        f"Bins/epoch/rank: {bins_per_epoch}, accumulation: {args.accumulation_steps}, "
        f"total optimizer steps: {total_optimizer_steps}"
    )

    # Frozen embed + lm_head
    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path, device="cuda"
    )
    greedy_lm_head_weight = target_components.lm_head.weight.data

    # mask_embed init: mean of the embedding table. NOT embed_tokens(mask_id) —
    # the placeholder token's transplanted row is OMP garbage (dead donor
    # vector); the table mean is a well-scaled neutral start. The garbage row
    # itself never enters the graph (mask positions are torch.where-replaced).
    if _mask_embed_missing:
        with torch.no_grad():
            draft_model.mask_embed.data.copy_(
                target_components.embed_tokens.weight.mean(dim=0)
            )
        print_on_rank0("Initialized mask_embed from embedding-table mean")

    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        greedy_match_threshold=args.greedy_match_threshold,
        use_cce=args.use_cce,
        sliding_window=getattr(draft_model.config, "sliding_window", None),
        causal=args.causal,
        focal_gamma=args.focal_gamma,
    )

    # DDP, not FSDP1 (upstream used FSDP1 SHARD_GRAD_OP): the draft is small
    # (~1.6B; ~25GB/rank incl. fp32 masters + AdamW), so full replication is
    # cheap, the saved state_dict is trivially the live weights, the optimizer
    # state is identical on every rank (single-file checkpoint, no world_size
    # lock on resume), and we avoid FSDP1's deprecated state-dict APIs on
    # torch 2.12.
    dflash_model = dflash_model.cuda()
    dflash_model = DDP(dflash_model, device_ids=[local_rank])
    print_with_rank("Initialized DDP for draft model")

    if args.pretrained_lr is not None:
        pretrained_names = {"layers", "norm"}
        pretrained_params, new_params = [], []
        for name, param in draft_model.named_parameters():
            (pretrained_params if any(name.startswith(p) for p in pretrained_names)
             else new_params).append(param)
        print_on_rank0(
            f"Differential LR: pretrained={len(pretrained_params)} @ {args.pretrained_lr}, "
            f"new={len(new_params)} @ {args.learning_rate}"
        )
        param_groups = [
            {"params": pretrained_params, "lr": args.pretrained_lr},
            {"params": new_params, "lr": args.learning_rate},
        ]
        optimizer = BF16Optimizer(
            draft_model, lr=args.learning_rate, max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio, warmup_steps=args.warmup_steps,
            total_optimizer_steps=total_optimizer_steps, param_groups=param_groups,
        )
    else:
        optimizer = BF16Optimizer(
            draft_model, lr=args.learning_rate, max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio, warmup_steps=args.warmup_steps,
            total_optimizer_steps=total_optimizer_steps,
        )

    # Resume optimizer + scheduler state
    start_epoch = 0
    global_step = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state)
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        if "rng_state" in resume_state:
            torch.random.set_rng_state(resume_state["rng_state"])
            torch.cuda.set_rng_state(resume_state["cuda_rng_state"])

        opt_path = os.path.join(draft_model_last_checkpoint, "optimizer.pt")
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location="cpu", weights_only=False)
            optimizer.load_optimizer_only_state_dict(opt_state["optimizer_state_dict"])
            print_on_rank0("Loaded optimizer.pt")
        else:
            print_on_rank0(
                "WARNING: optimizer.pt not found; AdamW momentum starts fresh."
            )
        del resume_state
        print_on_rank0(f"Restored scheduler, lr={optimizer.get_learning_rate():.6f}")

    if args.gradient_checkpointing:
        draft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print_on_rank0("Gradient checkpointing enabled")

    tracker = create_tracker(args, args.output_dir)

    skip_bins = global_step * args.accumulation_steps
    skip_batches = skip_bins // args.batch_size

    # ======================================================================
    # Training loop
    # ======================================================================
    accum = args.accumulation_steps
    last_time = time.time()
    reached_total = False  # --total-optimizer-steps is a hard stop, not just the LR horizon
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        if reached_total:
            break
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()

        total_batches = len(train_dataloader)
        epoch_skip = skip_batches if epoch == start_epoch else 0
        if epoch_skip > 0:
            print_on_rank0(f"Skipping {epoch_skip}/{total_batches} batches in epoch {epoch}")
            data_iter = itertools.islice(train_dataloader, epoch_skip, None)
        else:
            data_iter = train_dataloader

        remaining_batches = total_batches - epoch_skip
        remaining_opt_steps = (remaining_batches * args.batch_size) // accum
        progress_bar = (
            tqdm(total=remaining_opt_steps, desc=f"Epoch {epoch}", leave=True)
            if dist.get_rank() == 0
            else None
        )

        accum_loss = torch.tensor(0.0, device="cuda")
        accum_metrics = {}
        accum_count = 0

        # Clear stale gradients from a partial accumulation window at epoch end
        dflash_model.zero_grad(set_to_none=True)

        for data in data_iter:
            input_ids = data["input_ids"].cuda(non_blocking=True)
            loss_mask = data["loss_mask"].cuda(non_blocking=True)
            document_ids = data["document_ids"].cuda(non_blocking=True)
            position_ids = data["position_ids"].cuda(non_blocking=True)

            num_bins_in_batch = input_ids.shape[0]

            for micro_idx in range(num_bins_in_batch):
                accum_count += 1
                is_last_accum = (accum_count == accum)
                should_log = is_last_accum and ((global_step + 1) % args.log_interval == 0)

                micro_input = input_ids[micro_idx : micro_idx + 1]
                micro_lm = loss_mask[micro_idx : micro_idx + 1]
                micro_doc = document_ids[micro_idx : micro_idx + 1]
                micro_pos = position_ids[micro_idx : micro_idx + 1]

                # Frozen target forward (per-bin to bound peak memory)
                target_output = target_model.generate_hidden_states(
                    micro_input, micro_lm, position_ids=micro_pos
                )
                micro_hidden = target_output.hidden_states

                micro_greedy = None
                micro_sample_probs = None
                if target_output.last_hidden is not None:
                    micro_greedy, micro_sample_probs = _chunked_greedy(
                        greedy_lm_head_weight,
                        target_output.last_hidden,
                        input_ids=micro_input if args.greedy_match_threshold is not None else None,
                    )

                sync_ctx = nullcontext if is_last_accum else dflash_model.no_sync
                with sync_ctx():
                    micro_loss, micro_metrics = dflash_model(
                        input_ids=micro_input,
                        hidden_states=micro_hidden,
                        loss_mask=micro_lm,
                        compute_accuracy=should_log,
                        document_ids=micro_doc,
                        context_position_ids=micro_pos,
                        greedy_tokens=micro_greedy,
                        sample_probs=micro_sample_probs,
                    )
                    micro_loss.backward(torch.tensor(1.0 / accum, device="cuda"))

                accum_loss += micro_loss.detach()
                if micro_metrics is not None:
                    for k, v in micro_metrics.items():
                        if k not in accum_metrics:
                            accum_metrics[k] = torch.tensor(0.0, device="cuda")
                        accum_metrics[k] += v.detach() if isinstance(v, torch.Tensor) else v

                if is_last_accum:
                    optimizer.step()
                    global_step += 1

                    if should_log:
                        _metric_keys = sorted(accum_metrics.keys())
                        _sum_metrics = {"effective_tokens"}
                        _packed = torch.stack(
                            [accum_loss / accum] + [accum_metrics[k] for k in _metric_keys]
                        )
                        dist.all_reduce(_packed)
                        _ws = dist.get_world_size()
                        _packed[0] /= _ws
                        for _i, mk in enumerate(_metric_keys):
                            if mk not in _sum_metrics:
                                _packed[_i + 1] /= _ws

                        avg_loss = _packed[0]
                        log_dict = {
                            "train/loss": avg_loss.item(),
                            "train/lr": optimizer.get_learning_rate(),
                        }
                        grad_norm = optimizer.get_grad_norm()
                        if grad_norm is not None:
                            log_dict["train/grad_norm"] = grad_norm.item()

                        for _i, mk in enumerate(_metric_keys):
                            v = _packed[_i + 1].item()
                            if mk.startswith("acc/pos_"):
                                log_dict[f"pos_acc/{mk.split('_')[-1]}"] = v
                            elif mk.startswith("train_ratio/pos_"):
                                log_dict[f"pos_ratio/{mk.split('_')[-1]}"] = v
                            elif mk.startswith("greedy/"):
                                log_dict[mk] = v
                            else:
                                log_dict[f"train/{mk}"] = v

                        acc_str = f" acc={log_dict.get('train/accuracy', 0):.4f}"
                        prefix_str = ""
                        if "greedy/mean_prefix_len" in log_dict:
                            prefix_str = f" prefix_len={log_dict['greedy/mean_prefix_len']:.2f}"
                        print_on_rank0(
                            f"Step {global_step}/{total_optimizer_steps} | "
                            f"loss={avg_loss.item():.4f}{acc_str}{prefix_str} "
                            f"lr={optimizer.get_learning_rate():.2e}"
                        )
                        tracker.log(log_dict, step=global_step)

                    if progress_bar is not None:
                        elapsed = time.time() - last_time
                        last_time = time.time()
                        progress_bar.update(1)
                        progress_bar.set_postfix({
                            "loss": f"{(accum_loss / accum).item():.4f}",
                            "time": f"{elapsed:.2f}s",
                        })

                    if global_step % args.save_latest_interval == 0:
                        save_dirs = [os.path.join(args.output_dir, "latest")]
                        if global_step % args.save_interval == 0:
                            save_dirs.append(
                                os.path.join(args.output_dir, f"epoch_{epoch}_step_{global_step}")
                            )
                        save_checkpoint(
                            args, epoch, global_step, dflash_model, draft_model, optimizer,
                            save_dirs=save_dirs,
                        )
                    elif global_step % args.save_interval == 0:
                        save_checkpoint(
                            args, epoch, global_step, dflash_model, draft_model, optimizer,
                        )

                    accum_loss.zero_()
                    accum_metrics.clear()
                    accum_count = 0

                    if global_step >= total_optimizer_steps:
                        reached_total = True
                        print_on_rank0(f"Reached total_optimizer_steps={total_optimizer_steps}, stopping.")
                        break

            if reached_total:
                break

        if progress_bar is not None:
            progress_bar.close()

    # On an early stop (total_optimizer_steps) save the CURRENT epoch so a
    # later resume with a higher step target continues mid-epoch; only a fully
    # completed run records num_epochs. NOTE: the resume batch-skip assumes
    # sub-1-epoch runs (skip = global_step*accum within the saved epoch).
    final_epoch = epoch if reached_total else args.num_epochs
    save_checkpoint(
        args, final_epoch, global_step, dflash_model, draft_model, optimizer,
        save_dirs=[
            os.path.join(args.output_dir, "latest"),
            os.path.join(args.output_dir, f"epoch_{final_epoch}_step_{global_step}"),
        ],
    )

    tracker.close()
    print_on_rank0("Training complete.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
