#!/usr/bin/env python3
"""Canonical-convention DFlash draft, ported onto the no-dup FA3 harness.

Background: the no-dup trainer builds the
draft block as ``B`` ``mask_embed`` queries at positions ``start+1..start+B`` with
the verified token present only as context, and predicts ``B`` tokens. sglang's
stock DFlash worker (and the canonical ``Olmo3DFlashDraftModel``) instead uses the
**canonical** convention: block-0 is the verified token's *target embedding* as a
real query at position ``start``, slots ``1..B-1`` are ``mask_embed``, and the
draft predicts ``B-1`` tokens; the context window is strictly *before* ``start``
(the verified token's own target-hidden is never in context, because it is not
available at inference). Training in the canonical convention makes the weights
drop-in for the existing all-SWA sglang serving stack at one target forward/cycle.

This module reuses the no-dup model/attention/data/loss machinery verbatim and
changes ONLY the block construction + context boundary + label slicing:

  * ``CanonicalFA3Draft`` (subclasses ``NoDupFA3Draft``): block-0 fed the verified
    token's embedding, block positions ``start..start+B-1``, context sliced to
    ``[k0:last_start]`` (exclusive of each start's own row — the FA3 context kernel
    is right-aligned + left-windowed, so shifting the buffer end down by one and
    ``k0`` down by one lands every start's window on ``[s_i-W, s_i-1]``). The FA3
    kernel (``fa3_nodup_attention.py``) is UNCHANGED.
  * ``compute_loss_backward_and_metrics_canonical``: the label/greedy tensors are
    identical to the no-dup B-1 predictions; the draft emits ``B`` slots and the
    loss/metrics use ``out[:, 1:]`` (block-0 carries no loss).

The all-start / L4-direct-read / FA3-local-context efficiency is fully preserved.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext

import torch
import torch.nn as nn

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.append(os.path.abspath(os.path.join(HERE, "../..")))   # training/ (olmo3_sink); last so dflash.py wins over the dflash/ package

from nodup_fa3_train import (  # noqa: E402
    NoDupFA3Draft,
    build_all_start_training_tensors,
)


class CanonicalFA3Draft(NoDupFA3Draft):
    """No-dup draft re-cast to the canonical block convention.

    ``config.block_size`` is the TOTAL slot count (verified block-0 + predicted
    mask slots). To predict the same ``P`` tokens the no-dup draft predicted, set
    ``block_size = P + 1`` (e.g. 11 to predict 10). All parameters are identical to
    ``NoDupFA3Draft`` so a no-dup checkpoint warm-starts 1:1.
    """

    def forward(
        self,
        target_hidden: torch.Tensor,         # [1, S, n_layers*H]
        context_position_ids: torch.Tensor,  # [1, S]
        start_positions: torch.Tensor,       # [C] contiguous last-verified indices
        verified_embed: torch.Tensor,        # [C, H] target embedding of input_ids[start]
    ) -> torch.Tensor:
        w = self.window_size
        b = self.block_size
        device = target_hidden.device
        c = start_positions.numel()
        lo = int(start_positions[0].item())
        sN = int(start_positions[-1].item())

        # Context strictly BEFORE each start (canonical: verified token enters via
        # block-0, never as context). Buffer end = last start (exclusive); k0 one
        # token further left than no-dup so query 0's window [s0-W, s0-1] is covered.
        # The FA3 context kernel is right-aligned, so query i lands on token s_i-1.
        k0 = max(0, lo - w)
        selected = target_hidden[0, k0:sN]
        context_hidden = self.hidden_norm(self.fc(selected))
        context_pos = context_position_ids[0, k0:sN]

        # Block: slot 0 = verified token embedding (real query at position start),
        # slots 1..b-1 = learnable mask_embed.
        block_hidden = self.mask_embed.view(1, 1, -1).expand(c, b, -1).clone()
        block_hidden[:, 0] = verified_embed.to(block_hidden.dtype)

        start_pos = context_position_ids[0, start_positions]
        block_pos = start_pos[:, None] + torch.arange(0, b, device=device)[None, :]

        for layer in self.layers:
            block_hidden = layer(block_hidden, context_hidden, context_pos, block_pos, self.rotary_emb)
        return self.norm(block_hidden)


def compute_loss_backward_and_metrics_canonical(
    draft: nn.Module,
    lm_head: nn.Module,
    embed_tokens: nn.Module,
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
    """Canonical variant of the no-dup loss.

    The draft emits ``block_size`` slots; slot 0 is the verified token (no loss),
    slots ``1..block_size-1`` predict ``block_size-1`` tokens. The label/greedy
    construction is identical to the no-dup path with ``block_size = pred`` where
    ``pred = draft.block_size - 1`` — so we build P-wide labels and compare against
    ``out[:, 1:]``.
    """
    draft_model = draft.module if hasattr(draft, "module") else draft
    pred = draft_model.block_size - 1  # number of predicted tokens per block

    starts, target_ids_raw, greedy_ids, base_weight, same_doc = build_all_start_training_tensors(
        input_ids, loss_mask, document_ids, greedy_tokens, window_size, pred
    )
    # build_all_start_training_tensors uses starts = arange(W-1, S-pred) and checks
    # doc[start-W+1]. The canonical context reaches one token further left
    # (start-W), so drop the single start whose extra-left token leaves the doc.
    doc = document_ids[0]
    extra_left_ok = (starts - window_size) >= 0
    safe_idx = extra_left_ok.clone()
    safe_idx[extra_left_ok] = doc[(starts[extra_left_ok] - window_size)] == doc[starts[extra_left_ok]]
    base_weight = base_weight * safe_idx[:, None].float()

    source_n = starts.numel()
    if max_starts_per_step > 0 and source_n > max_starts_per_step:
        max_offset = source_n - max_starts_per_step
        start_offset = int(torch.randint(0, max_offset + 1, (), device=starts.device).item())
        stop_offset = start_offset + max_starts_per_step
        starts = starts[start_offset:stop_offset]
        target_ids_raw = target_ids_raw[start_offset:stop_offset]
        greedy_ids = greedy_ids[start_offset:stop_offset]
        base_weight = base_weight[start_offset:stop_offset]
        same_doc = same_doc[start_offset:stop_offset]

    n = starts.numel()
    match = target_ids_raw == greedy_ids
    prefix_match = match.long().cumprod(dim=-1)
    greedy_mask = torch.ones_like(prefix_match, dtype=torch.float32)
    greedy_mask[:, 1:] = prefix_match[:, :-1].float()

    target_ids = greedy_ids
    weight = base_weight * greedy_mask
    binary_mask = weight > 0

    if loss_decay_gamma > 0:
        k = torch.arange(pred, device=starts.device).view(1, -1)
        weight = weight * torch.exp(-k.float() / loss_decay_gamma)

    total_weight = weight.sum() + 1e-6
    loss_sum_total = torch.zeros((), device=starts.device)
    correct_total = torch.zeros((), device=starts.device)
    effective_total = torch.zeros((), device=starts.device)
    pos_correct = torch.zeros(pred, device=starts.device)
    pos_count = torch.zeros(pred, device=starts.device)

    embed_w = embed_tokens.weight

    for lo in range(0, n, start_chunk_size):
        hi = min(lo + start_chunk_size, n)
        sync_this_chunk = hi == n
        sync_ctx = nullcontext() if sync_this_chunk or not hasattr(draft, "no_sync") else draft.no_sync()
        chunk_starts = starts[lo:hi]
        verified_embed = embed_w[input_ids[0, chunk_starts]]  # [chunk, H]
        with sync_ctx:
            out_full = draft(target_hidden, position_ids, chunk_starts, verified_embed)
            out = out_full[:, 1:]  # drop block-0 (verified token, no loss)
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
                pred_tok = torch.empty(flat_hidden.shape[0], dtype=torch.long, device=flat_hidden.device)
                for i in range(0, flat_hidden.shape[0], 1024):
                    logits = lm_head(flat_hidden[i : i + 1024])
                    pred_tok[i : i + 1024] = logits.argmax(dim=-1)
                pred_block = pred_tok.view(hi - lo, pred)
                m = binary_mask[lo:hi]
                correct = (pred_block == target_ids[lo:hi]) & m
                correct_total = correct_total + correct.sum().float()
                effective_total = effective_total + m.sum().float()
                for k in range(pred):
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
        for k in range(pred):
            metrics[f"acc/pos_{k}"] = pos_correct[k] / (pos_count[k] + 1e-6)
            metrics[f"train_ratio/pos_{k}"] = pos_count[k] / n
    return loss, metrics
