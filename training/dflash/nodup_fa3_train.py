#!/usr/bin/env python3
"""Trainer core for all-SWA no-dup FA3 DFlash (replaces the anchor + FlexAttention
trainer with a simpler all-SWA layout):

  start = last verified token
  Q     = B mask positions at start+1..start+B
  KV    = W target-hidden context tokens start-W+1..start plus the B draft tokens

There are no anchors and no FlexAttention BlockMask in this path. All valid
start positions are available; for throughput we may sample a contiguous subset
per optimizer step. The attention layer does not materialize overlapping context
windows: context attention is FA3 local attention over the original packed
sequence, while the small per-start draft block attention is merged exactly with
a dense BxB partial.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import shutil
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.utils import set_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoConfig
from transformers.models.olmo3.modeling_olmo3 import Olmo3MLP, Olmo3RMSNorm

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.abspath(os.path.join(HERE, "..")))   # training/ (olmo3_sink); after HERE so dflash.py wins over the dflash/ package

from data import build_dataloader
from draft_model_olmo3 import (
    Olmo3DFlashRotaryEmbedding,
    build_target_layer_ids,
)
from fa3_nodup_attention import no_dup_full_attention_dense_block
from optimizer import BF16Optimizer
from target_model import FSDP2TargetModel
from target_utils import TargetEmbeddingsAndHead
from train import _chunked_greedy
from tracker import create_tracker
from utils import print_on_rank0, print_with_rank


def apply_rope_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x [..., heads, D] with cos/sin [..., D]."""
    x_type = x.dtype
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos + rotated * sin).to(x_type)


class NoDupFA3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = Olmo3RMSNorm(
            config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps
        )
        self.k_norm = Olmo3RMSNorm(
            config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps
        )
        self.sinks = nn.Parameter(torch.zeros(config.num_attention_heads))

    def forward(
        self,
        block_hidden: torch.Tensor,       # [C, B, H]
        context_hidden: torch.Tensor,     # [K, H], K = C + left overlap
        context_pos: torch.Tensor,        # [K]
        block_pos: torch.Tensor,          # [C, B]
        rotary_emb: Olmo3DFlashRotaryEmbedding,
    ) -> torch.Tensor:
        c, b, _ = block_hidden.shape
        hq = self.config.num_attention_heads
        hkv = self.config.num_key_value_heads
        d = self.head_dim

        q = self.q_norm(self.q_proj(block_hidden))
        q = q.view(c, b, hq, d)

        k_ctx = self.k_proj(context_hidden)
        k_blk = self.k_proj(block_hidden)
        k_ctx = self.k_norm(k_ctx).view(context_hidden.shape[0], hkv, d)
        k_blk = self.k_norm(k_blk).view(c, b, hkv, d)

        v_ctx = self.v_proj(context_hidden).view(context_hidden.shape[0], hkv, d)
        v_blk = self.v_proj(block_hidden)
        v_blk = v_blk.view(c, b, hkv, d)

        ctx_cos, ctx_sin = rotary_emb(context_hidden.unsqueeze(0), context_pos.unsqueeze(0))
        blk_cos, blk_sin = rotary_emb(block_hidden, block_pos)
        q = apply_rope_one(q, blk_cos, blk_sin)
        k_blk = apply_rope_one(k_blk, blk_cos, blk_sin)
        k_ctx = apply_rope_one(k_ctx, ctx_cos.squeeze(0), ctx_sin.squeeze(0))

        out = no_dup_full_attention_dense_block(
            q.transpose(0, 1).contiguous(),  # [B,C,Hq,D]
            k_ctx.contiguous(),
            v_ctx.contiguous(),
            k_blk.contiguous(),
            v_blk.contiguous(),
            self.sinks,
            self.config.sliding_window,
            self.scaling,
        )
        out = out.transpose(0, 1).reshape(c, b, hq * d)
        return self.o_proj(out)


class NoDupFA3Layer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = NoDupFA3Attention(config, layer_idx)
        self.mlp = Olmo3MLP(config)
        self.post_attention_layernorm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, block_hidden, context_hidden, context_pos, block_pos, rotary_emb):
        residual = block_hidden
        block_hidden = self.self_attn(block_hidden, context_hidden, context_pos, block_pos, rotary_emb)
        block_hidden = self.post_attention_layernorm(block_hidden)
        block_hidden = residual + block_hidden

        residual = block_hidden
        block_hidden = self.mlp(block_hidden)
        block_hidden = self.post_feedforward_layernorm(block_hidden)
        block_hidden = residual + block_hidden
        return block_hidden


class NoDupFA3Draft(nn.Module):
    def __init__(self, config, window_size: int):
        super().__init__()
        self.config = config
        self.window_size = window_size
        self.block_size = config.block_size
        self.target_layer_ids = config.dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * target_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mask_embed = nn.Parameter(torch.zeros(config.hidden_size))
        self.rotary_emb = Olmo3DFlashRotaryEmbedding(config, rope_type="default")
        self.layers = nn.ModuleList(
            [NoDupFA3Layer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: torch.Tensor,        # [1, S, n_layers*H]
        context_position_ids: torch.Tensor, # [1, S]
        start_positions: torch.Tensor,      # [C], contiguous token indices of last verified token
    ) -> torch.Tensor:
        w = self.window_size
        b = self.block_size
        device = target_hidden.device
        c = start_positions.numel()
        lo = int(start_positions[0].item())
        hi = int(start_positions[-1].item()) + 1
        k0 = max(0, lo - w + 1)

        selected = target_hidden[0, k0:hi]
        context_hidden = self.hidden_norm(self.fc(selected))
        context_pos = context_position_ids[0, k0:hi]
        block_hidden = self.mask_embed.view(1, 1, -1).expand(c, b, -1)
        start_pos = context_position_ids[0, start_positions]
        block_pos = start_pos[:, None] + torch.arange(1, b + 1, device=device)[None, :]

        for layer in self.layers:
            block_hidden = layer(block_hidden, context_hidden, context_pos, block_pos, self.rotary_emb)
        return self.norm(block_hidden)


def build_all_start_training_tensors(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    greedy_tokens: torch.Tensor,
    window_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build all contiguous start positions; invalid starts get zero weight."""
    assert loss_mask.shape[0] == 1
    device = loss_mask.device
    S = loss_mask.shape[1]
    starts = torch.arange(window_size - 1, S - block_size, device=device)
    doc = document_ids[0]
    same_doc = (
        (doc[starts] >= 0)
        & (doc[starts - window_size + 1] == doc[starts])
        & (doc[starts + block_size] == doc[starts])
    )
    label_idx = starts[:, None] + torch.arange(1, block_size + 1, device=device)[None, :]
    prev_idx = starts[:, None] + torch.arange(0, block_size, device=device)[None, :]
    target_ids = input_ids[0].gather(0, label_idx.reshape(-1)).view(-1, block_size)
    greedy_ids = greedy_tokens[0].gather(0, prev_idx.reshape(-1)).view(-1, block_size)
    base_weight = loss_mask[0].gather(0, label_idx.reshape(-1)).view(-1, block_size).float()
    base_weight = base_weight * same_doc[:, None].float()
    if base_weight.sum().item() == 0:
        raise RuntimeError("No valid start positions in this bin.")
    return starts, target_ids, greedy_ids, base_weight, same_doc


def compute_loss_backward_and_metrics(
    draft: nn.Module,
    lm_head: nn.Module,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    document_ids: torch.Tensor,
    position_ids: torch.Tensor,
    target_hidden: torch.Tensor,
    greedy_tokens: torch.Tensor,
    start_chunk_size: int,
    max_starts_per_step: int,
    window_size: int,
    loss_decay_gamma: float,
    compute_metrics: bool,
    linear_cross_entropy,
):
    draft_model = draft.module if hasattr(draft, "module") else draft
    starts, target_ids_raw, greedy_ids, base_weight, same_doc = build_all_start_training_tensors(
        input_ids, loss_mask, document_ids, greedy_tokens, window_size, draft_model.block_size
    )
    source_n = starts.numel()
    if max_starts_per_step > 0 and source_n > max_starts_per_step:
        # This is stochastic contiguous subsampling, not a fixed stride. Across
        # steps, every start can be selected while one step remains bounded.
        max_offset = source_n - max_starts_per_step
        start_offset = int(torch.randint(0, max_offset + 1, (), device=starts.device).item())
        stop_offset = start_offset + max_starts_per_step
        starts = starts[start_offset:stop_offset]
        target_ids_raw = target_ids_raw[start_offset:stop_offset]
        greedy_ids = greedy_ids[start_offset:stop_offset]
        base_weight = base_weight[start_offset:stop_offset]
        same_doc = same_doc[start_offset:stop_offset]

    b = draft_model.block_size
    n = starts.numel()
    match = target_ids_raw == greedy_ids
    prefix_match = match.long().cumprod(dim=-1)
    greedy_mask = torch.ones_like(prefix_match, dtype=torch.float32)
    greedy_mask[:, 1:] = prefix_match[:, :-1].float()

    target_ids = greedy_ids
    weight = base_weight * greedy_mask
    binary_mask = weight > 0

    if loss_decay_gamma > 0:
        k = torch.arange(b, device=starts.device).view(1, -1)
        weight = weight * torch.exp(-k.float() / loss_decay_gamma)

    total_weight = weight.sum() + 1e-6
    loss_sum_total = torch.zeros((), device=starts.device)
    correct_total = torch.zeros((), device=starts.device)
    effective_total = torch.zeros((), device=starts.device)
    pos_correct = torch.zeros(b, device=starts.device)
    pos_count = torch.zeros(b, device=starts.device)

    for lo in range(0, n, start_chunk_size):
        hi = min(lo + start_chunk_size, n)
        sync_this_chunk = hi == n
        sync_ctx = nullcontext() if sync_this_chunk or not hasattr(draft, "no_sync") else draft.no_sync()
        with sync_ctx:
            out = draft(target_hidden, position_ids, starts[lo:hi])
            cce_targets = target_ids[lo:hi].clone()
            cce_targets[~binary_mask[lo:hi]] = -100
            flat_hidden = out.reshape(-1, out.shape[-1])
            flat_targets = cce_targets.reshape(-1)
            loss_per_token = linear_cross_entropy(
                flat_hidden,
                lm_head.weight,
                flat_targets,
                reduction="none",
            )
            flat_weight = weight[lo:hi].reshape(-1)
            loss_sum = (loss_per_token * flat_weight).sum()
            (loss_sum / total_weight).backward()
        loss_sum_total = loss_sum_total + loss_sum.detach()

        if compute_metrics:
            with torch.no_grad():
                pred = torch.empty(flat_hidden.shape[0], dtype=torch.long, device=flat_hidden.device)
                for i in range(0, flat_hidden.shape[0], 1024):
                    logits = lm_head(flat_hidden[i : i + 1024])
                    pred[i : i + 1024] = logits.argmax(dim=-1)
                pred_block = pred.view(hi - lo, b)
                m = binary_mask[lo:hi]
                correct = (pred_block == target_ids[lo:hi]) & m
                correct_total = correct_total + correct.sum().float()
                effective_total = effective_total + m.sum().float()
                for k in range(b):
                    pos_correct[k] += correct[:, k].sum().float()
                    pos_count[k] += m[:, k].sum().float()

    loss = loss_sum_total / total_weight.detach()

    metrics = None
    if compute_metrics:
        metrics = {
            "accuracy": correct_total / (effective_total + 1e-6),
            "effective_tokens": effective_total,
            "starts/source_total": torch.tensor(float(source_n), device=starts.device),
            "starts/total": torch.tensor(float(n), device=starts.device),
            "starts/valid_same_doc": same_doc.sum().float(),
            "greedy/mean_prefix_len": binary_mask.sum(dim=-1).float().mean(),
            "greedy/match_rate": effective_total / (base_weight.sum() + 1e-6),
        }
        for k in range(b):
            metrics[f"acc/pos_{k}"] = pos_correct[k] / (pos_count[k] + 1e-6)
            metrics[f"train_ratio/pos_{k}"] = pos_count[k] / n
    return loss, metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-model-path", default=os.environ.get("TARGET_MODEL_PATH", "outputs/stage1-v2-7b"))
    p.add_argument(
        "--draft-config-path",
        default=os.path.join(HERE, "configs", "dflash-olmo3-7b-8L-bs10-swa128-sink.json"),
    )
    p.add_argument("--train-data-path", default=os.environ.get("DFLASH_TRAIN_DATA", "data/l4-g2r05-ml12288-mc65536"))
    p.add_argument("--output-dir", default=os.environ.get("DFLASH_OUTPUT_DIR", "outputs/dflash-nodup-fa3-7b-dev"))
    p.add_argument("--window-size", type=int, default=128, help="SWA context window W.")
    p.add_argument("--block-size", type=int, default=10, help="Draft tokens predicted per start B.")
    p.add_argument(
        "--start-chunk-size",
        type=int,
        default=2048,
        help="Backward micro-chunk size over contiguous start positions.",
    )
    p.add_argument(
        "--max-starts-per-step",
        type=int,
        default=4096,
        help="Random contiguous start subset per bin; 0 trains all starts in the bin.",
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--max-bins", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--loss-decay-gamma", type=float, default=5.0)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--metrics-mode", choices=["full", "loss_only"], default="full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-target-fsdp", action="store_true")
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
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    if rank == 0 and os.path.exists(metrics_path):
        os.remove(metrics_path)
    tracker = create_tracker(args, args.output_dir)

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

    ddp = DDP(draft, device_ids=[local_rank])
    opt = BF16Optimizer(
        draft,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        total_optimizer_steps=args.steps,
    )

    from cut_cross_entropy import linear_cross_entropy

    max_bins = args.max_bins or (args.steps * dist.get_world_size() + 1000)
    loader = build_dataloader(
        args.train_data_path,
        batch_size=1,
        num_workers=2,
        seed=args.seed,
        shuffle=True,
        max_bins=max_bins,
    )
    loader.sampler.set_epoch(0)
    data_iter = itertools.cycle(loader)

    print_on_rank0(
        f"No-dup FA3 train: steps={args.steps} chunk={args.start_chunk_size} "
        f"max_starts={args.max_starts_per_step} B={args.block_size} W={args.window_size} "
        f"params={sum(p.numel() for p in draft.parameters()):,}"
    )
    t_last = time.time()
    pbar = tqdm(total=args.steps, desc="nodup-fa3", disable=(rank != 0))
    for step in range(1, args.steps + 1):
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

        should_log = step == 1 or step % args.log_interval == 0 or step == args.steps
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
        peak_mem = torch.tensor(
            [torch.cuda.max_memory_allocated() / (1024**3)],
            device="cuda",
            dtype=torch.float32,
        )
        dist.all_reduce(peak_mem, op=dist.ReduceOp.MAX)

        log = None
        if should_log:
            keys = sorted(metrics.keys()) if metrics is not None else []
            packed = torch.stack([loss.detach()] + [metrics[k].detach() for k in keys])
            dist.all_reduce(packed)
            packed /= dist.get_world_size()
            log = {
                "step": step,
                "loss": packed[0].item(),
                "lr": opt.get_learning_rate(),
                "grad_norm": opt.get_grad_norm().item()
                if opt.get_grad_norm() is not None
                else None,
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
                    f"train={log['time/train_s']:.1f}s "
                    f"lr={log['lr']:.2e}",
                    flush=True,
                )
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    pbar.close()
    if rank == 0:
        torch.save(
            {
                "model": draft.state_dict(),
                "args": vars(args),
                "config": cfg.to_dict(),
            },
            os.path.join(args.output_dir, "final.pt"),
        )
        with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        # Keep the self-contained modeling code next to the result for reproducibility.
        shutil.copy(__file__, os.path.join(args.output_dir, "nodup_fa3_train.py"))
        shutil.copy(
            os.path.join(HERE, "..", "fa3_nodup_attention.py"),
            os.path.join(args.output_dir, "fa3_nodup_attention.py"),
        )
    print_on_rank0("No-dup FA3 training complete.")
    tracker.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
