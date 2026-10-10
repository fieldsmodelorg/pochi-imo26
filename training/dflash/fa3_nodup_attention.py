"""No-dup FA3 attention used by the all-SWA DFlash trainer.

This module implements the attention pattern used by
``experiments/nodup_fa3_train.py``:

* each draft offset attends the original target-hidden context with FA3 local
  attention, so the W-token context window is not materialized per start;
* the small per-start draft block attends itself with dense BxB math;
* the context, block, and sink partial attentions are merged exactly by LSE.

The public wrapper expects chunked contiguous starts. For a chunk of C starts,
``q_all`` has shape [B, C, Hq, D], context K/V have shape [K, Hkv, D], and
block K/V have shape [C, B, Hkv, D]. Context K must include the left overlap
needed by ``window`` plus the C rows aligned to the C starts.
"""

from __future__ import annotations

import torch
from flash_attn_interface import _flash_attn_backward, flash_attn_func

__all__ = [
    "no_dup_full_attention_dense_block",
    "no_dup_full_attention_dense_block_per_offset",
]


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    return torch.repeat_interleave(x, groups, dim=-2)


class _NoDupFullAttentionDenseBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_all, k_ctx, v_ctx, k_blk, v_blk, sink, window: int, scale: float):
        block_size, seq_len, hq, _head_dim = q_all.shape
        hkv = k_ctx.shape[-2]
        groups = hq // hkv

        out_rows = []
        lse_ctx_rows = []
        lse_blk_rows = []
        lse_total_rows = []

        k_blk_rep = _repeat_kv(k_blk.float(), groups)
        v_blk_rep = _repeat_kv(v_blk.float(), groups)
        for off in range(block_size):
            out_ctx, lse_ctx = flash_attn_func(
                q_all[off].unsqueeze(0),
                k_ctx.unsqueeze(0),
                v_ctx.unsqueeze(0),
                softmax_scale=scale,
                causal=False,
                window_size=(window - 1, 0),
                return_attn_probs=True,
            )
            lse_ctx = lse_ctx.squeeze(0).contiguous()  # [H,S]

            scores_blk = torch.einsum(
                "shd,sbhd->shb", q_all[off].float(), k_blk_rep
            ) * scale
            lse_blk = torch.logsumexp(scores_blk, dim=-1).transpose(0, 1).contiguous()
            lse_total = torch.logsumexp(
                torch.stack(
                    [
                        lse_ctx,
                        lse_blk,
                        sink.float().view(hq, 1).expand(hq, seq_len),
                    ],
                    dim=0,
                ),
                dim=0,
            ).contiguous()

            alpha_ctx = torch.exp(lse_ctx - lse_total).transpose(0, 1).unsqueeze(-1)
            p_blk_total = torch.exp(scores_blk - lse_total.transpose(0, 1).unsqueeze(-1))
            out_blk = torch.einsum("shb,sbhd->shd", p_blk_total, v_blk_rep)
            out = (out_ctx.squeeze(0).float() * alpha_ctx + out_blk).to(q_all.dtype)

            out_rows.append(out)
            lse_ctx_rows.append(lse_ctx)
            lse_blk_rows.append(lse_blk)
            lse_total_rows.append(lse_total)

        out_total = torch.stack(out_rows, dim=0).contiguous()
        ctx.save_for_backward(
            q_all,
            k_ctx,
            v_ctx,
            k_blk,
            v_blk,
            sink,
            out_total,
            torch.stack(lse_ctx_rows, dim=0).contiguous(),
            torch.stack(lse_blk_rows, dim=0).contiguous(),
            torch.stack(lse_total_rows, dim=0).contiguous(),
        )
        ctx.window = int(window)
        ctx.scale = float(scale)
        return out_total

    @staticmethod
    def backward(ctx, grad_out):
        (
            q_all,
            k_ctx,
            v_ctx,
            k_blk,
            v_blk,
            sink,
            out_total,
            lse_ctx_all,
            _lse_blk_all,
            lse_total_all,
        ) = ctx.saved_tensors
        block_size, seq_len, hq, head_dim = q_all.shape
        ctx_len = k_ctx.shape[0]
        hkv = k_ctx.shape[-2]
        groups = hq // hkv
        window = ctx.window
        scale = ctx.scale

        grad_out = grad_out.contiguous()
        dq_all = torch.zeros_like(q_all)
        dk_ctx_all = torch.zeros_like(k_ctx)
        dv_ctx_all = torch.zeros_like(v_ctx)
        dk_blk_all = torch.zeros_like(k_blk)
        dv_blk_all = torch.zeros_like(v_blk)
        dsink = torch.zeros_like(sink, dtype=torch.float32)
        k_blk_rep = _repeat_kv(k_blk.float(), groups)
        v_blk_rep = _repeat_kv(v_blk.float(), groups)

        for off in range(block_size):
            do = grad_out[off]
            out = out_total[off].contiguous()
            lse_ctx = lse_ctx_all[off]
            lse_total = lse_total_all[off]

            delta = (out.float() * do.float()).sum(dim=-1)  # [S,H]
            alpha_ctx = torch.exp(lse_ctx - lse_total).transpose(0, 1).unsqueeze(-1)
            alpha_sink = torch.exp(sink.float().view(hq, 1) - lse_total)
            dsink = dsink - (alpha_sink * delta.transpose(0, 1)).sum(dim=-1)

            do_ctx = (do.float() * alpha_ctx).to(do.dtype)
            dq_ctx = torch.empty((1, seq_len, hq, head_dim), device=q_all.device, dtype=q_all.dtype)
            dk_ctx = torch.empty((1, ctx_len, hkv, head_dim), device=q_all.device, dtype=k_ctx.dtype)
            dv_ctx = torch.empty((1, ctx_len, hkv, head_dim), device=q_all.device, dtype=v_ctx.dtype)
            _flash_attn_backward(
                do_ctx.unsqueeze(0).contiguous(),
                q_all[off].unsqueeze(0).contiguous(),
                k_ctx.unsqueeze(0).contiguous(),
                v_ctx.unsqueeze(0).contiguous(),
                out.unsqueeze(0),
                lse_ctx.unsqueeze(0).contiguous(),
                None,
                None,
                None,
                None,
                None,
                None,
                dq_ctx,
                dk_ctx,
                dv_ctx,
                scale,
                False,
                window - 1,
                0,
                0.0,
                False,
                0,
            )
            dq_all[off] = dq_all[off] + dq_ctx.squeeze(0)
            dk_ctx_all = dk_ctx_all + dk_ctx.squeeze(0)
            dv_ctx_all = dv_ctx_all + dv_ctx.squeeze(0)

            scores_blk = torch.einsum(
                "shd,sbhd->shb", q_all[off].float(), k_blk_rep
            ) * scale
            p_blk_total = torch.exp(scores_blk - lse_total.transpose(0, 1).unsqueeze(-1))
            v_dot_do = torch.einsum("shd,sbhd->shb", do.float(), v_blk_rep)
            dscores = p_blk_total * (v_dot_do - delta.unsqueeze(-1))
            dq_blk = torch.einsum("shb,sbhd->shd", dscores, k_blk_rep) * scale
            dk_blk_rep = torch.einsum("shb,shd->sbhd", dscores, q_all[off].float()) * scale
            dv_blk_rep = torch.einsum("shb,shd->sbhd", p_blk_total, do.float())

            dq_all[off] = dq_all[off] + dq_blk.to(dq_all.dtype)
            dk_blk_all = dk_blk_all + dk_blk_rep.view(
                seq_len, block_size, hkv, groups, head_dim
            ).sum(dim=3).to(k_blk.dtype)
            dv_blk_all = dv_blk_all + dv_blk_rep.view(
                seq_len, block_size, hkv, groups, head_dim
            ).sum(dim=3).to(v_blk.dtype)

        return dq_all, dk_ctx_all, dv_ctx_all, dk_blk_all, dv_blk_all, dsink.to(sink.dtype), None, None


class _NoDupBatched(torch.autograd.Function):
    """Batched-offset equivalent of ``_NoDupFullAttentionDenseBlock``.

    All ``block_size`` offsets share the same k_ctx/v_ctx/window and differ only in
    ``q_all[off]``, so the per-offset loop collapses into ONE batched flash call
    (offset = flash batch dim) + batched einsums. Mathematically identical to the
    per-offset reference (forward bit-exact; backward replicated in batched form,
    all 6 grads match to bf16). Far higher MFU: one big op instead of block_size
    small flash calls + looped eager merge. See DFLASH_ATTENTION_LAYOUT.md.
    """

    @staticmethod
    def forward(ctx, q_all, k_ctx, v_ctx, k_blk, v_blk, sink, window: int, scale: float):
        B, C, hq, d = q_all.shape
        hkv = k_ctx.shape[-2]
        groups = hq // hkv
        kc = k_ctx.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        vc = v_ctx.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        out_ctx, lse_ctx = flash_attn_func(
            q_all, kc, vc, softmax_scale=scale, causal=False,
            window_size=(window - 1, 0), return_attn_probs=True,
        )  # out_ctx [B,C,hq,d], lse_ctx [B,hq,C]
        k_blk_rep = _repeat_kv(k_blk.float(), groups)
        v_blk_rep = _repeat_kv(v_blk.float(), groups)
        scores_blk = torch.einsum("oshd,sbhd->oshb", q_all.float(), k_blk_rep) * scale  # [B,C,hq,b]
        lse_blk = torch.logsumexp(scores_blk, dim=-1).transpose(1, 2)                    # [B,hq,C]
        sink_e = sink.float().view(1, hq, 1).expand(B, hq, C)
        lse_total = torch.logsumexp(torch.stack([lse_ctx.float(), lse_blk, sink_e], 0), 0)
        alpha_ctx = torch.exp(lse_ctx.float() - lse_total).transpose(1, 2).unsqueeze(-1)
        p_blk = torch.exp(scores_blk - lse_total.transpose(1, 2).unsqueeze(-1))
        out_blk = torch.einsum("oshb,sbhd->oshd", p_blk, v_blk_rep)
        out = (out_ctx.float() * alpha_ctx + out_blk).to(q_all.dtype)
        ctx.save_for_backward(q_all, k_ctx, v_ctx, k_blk, v_blk, sink, out, lse_ctx, lse_total)
        ctx.window = int(window)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q_all, k_ctx, v_ctx, k_blk, v_blk, sink, out, lse_ctx, lse_total = ctx.saved_tensors
        B, C, hq, d = q_all.shape
        hkv = k_ctx.shape[-2]
        groups = hq // hkv
        K = k_ctx.shape[0]
        window = ctx.window
        scale = ctx.scale
        grad_out = grad_out.contiguous()

        delta = (out.float() * grad_out.float()).sum(-1)                       # [B,C,hq]
        alpha_ctx = torch.exp(lse_ctx.float() - lse_total).transpose(1, 2).unsqueeze(-1)
        alpha_sink = torch.exp(sink.float().view(1, hq, 1) - lse_total)        # [B,hq,C]
        dsink = -(alpha_sink * delta.transpose(1, 2)).sum(dim=(0, 2))          # [hq]

        do_ctx = (grad_out.float() * alpha_ctx).to(grad_out.dtype).contiguous()
        kc = k_ctx.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        vc = v_ctx.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        dq_ctx = torch.empty_like(q_all)
        dk_ctx_b = torch.empty((B, K, hkv, d), device=q_all.device, dtype=k_ctx.dtype)
        dv_ctx_b = torch.empty((B, K, hkv, d), device=q_all.device, dtype=v_ctx.dtype)
        _flash_attn_backward(
            do_ctx, q_all.contiguous(), kc, vc, out.contiguous(), lse_ctx.contiguous(),
            None, None, None, None, None, None,
            dq_ctx, dk_ctx_b, dv_ctx_b, scale, False, window - 1, 0, 0.0, False, 0,
        )
        dk_ctx = dk_ctx_b.sum(0)
        dv_ctx = dv_ctx_b.sum(0)
        dq_all = dq_ctx.float()

        k_blk_rep = _repeat_kv(k_blk.float(), groups)
        v_blk_rep = _repeat_kv(v_blk.float(), groups)
        scores_blk = torch.einsum("oshd,sbhd->oshb", q_all.float(), k_blk_rep) * scale
        p_blk = torch.exp(scores_blk - lse_total.transpose(1, 2).unsqueeze(-1))
        v_dot_do = torch.einsum("oshd,sbhd->oshb", grad_out.float(), v_blk_rep)
        dscores = p_blk * (v_dot_do - delta.unsqueeze(-1))                     # [B,C,hq,b]
        dq_all = (dq_all + torch.einsum("oshb,sbhd->oshd", dscores, k_blk_rep) * scale).to(q_all.dtype)
        b = k_blk.shape[1]
        dk_blk = (torch.einsum("oshb,oshd->sbhd", dscores, q_all.float()) * scale).view(
            C, b, hkv, groups, d).sum(3).to(k_blk.dtype)
        dv_blk = torch.einsum("oshb,oshd->sbhd", p_blk, grad_out.float()).view(
            C, b, hkv, groups, d).sum(3).to(v_blk.dtype)
        return dq_all, dk_ctx, dv_ctx, dk_blk, dv_blk, dsink.to(sink.dtype), None, None


def no_dup_full_attention_dense_block(
    q_all: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    k_blk: torch.Tensor,
    v_blk: torch.Tensor,
    sink: torch.Tensor,
    window: int,
    scale: float,
) -> torch.Tensor:
    """Run exact no-dup DFlash attention for one contiguous start chunk.

    Uses the batched-offset kernel (drop-in, numerically identical to the
    per-offset reference but ~1.4x faster on its own and the path that composes
    with ``torch.compile``-d layers). The per-offset reference is retained as
    ``no_dup_full_attention_dense_block_per_offset`` for regression testing.
    """
    return _NoDupBatched.apply(
        q_all, k_ctx, v_ctx, k_blk, v_blk, sink, int(window), float(scale)
    )


def no_dup_full_attention_dense_block_per_offset(
    q_all: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    k_blk: torch.Tensor,
    v_blk: torch.Tensor,
    sink: torch.Tensor,
    window: int,
    scale: float,
) -> torch.Tensor:
    """Per-offset reference implementation (ground truth for tests/debugging)."""
    return _NoDupFullAttentionDenseBlock.apply(
        q_all, k_ctx, v_ctx, k_blk, v_blk, sink, int(window), float(scale)
    )
