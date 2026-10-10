# Copyright 2026 proof-pilot. Apache-2.0.
"""Local JSD distillation kernel wrapper.

The stock Liger chunked JSD base is structurally correct for OPD, but it always
computes a student log-softmax for the hard CE path before entering the soft JSD
loss. JSD then computes another student log-softmax. For our common T=1 mixed
hard+soft objective those two values are identical, and for soft-only runs the
first one is pure waste.

This wrapper keeps the useful Liger pattern: compute chunk-local gradients with
`torch.func.grad_and_value`, save only gradients, and never materialize the full
[tokens, vocab] logits outside a chunk. It differs by:
- letting callers choose the GEMM dtype (bf16 fast path, fp32 baseline);
- forcing the log-softmax/KL math to fp32;
- sharing the student log-probs between hard CE and soft JSD when T=1.

V34 routed-OPD (proof-pilot, see `training/opd_v2/V34_PLAN.md`): the soft term can
additionally do **skew reverse-KL** (mix a little teacher into the RKL reference so
teacher-near-zero tokens stop blowing up entropy/length), add a **routed top-K
forward-KL** on high-entropy / overconfident-wrong / severe-outlier tokens, and
apply an optional **per-token weight** (clean-EOS reweight). All new knobs default
to a no-op, so the call is bit-identical to plain JSD(β) when they are off. The new
terms are differentiated automatically by `grad_and_value` — no hand-derived backward.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class FusedLinearJSDWithFP32Softmax(torch.autograd.Function):
    """Chunked full-vocab JSD with caller-controlled GEMM dtype."""

    # ---- per-token soft divergences (return [tokens]; ignore-mask applied by caller) ----
    @staticmethod
    def _rkl_per_token(student_log_probs, teacher_log_probs, skew_alpha):
        """KL(student ‖ ref); ref = teacher (α=0) or (1-α)·teacher + α·student (skew, α>0)."""
        if skew_alpha > 0.0:
            ref = torch.logsumexp(
                torch.stack([
                    teacher_log_probs + math.log(1.0 - skew_alpha),
                    student_log_probs + math.log(skew_alpha),
                ], dim=0),
                dim=0,
            )
        else:
            ref = teacher_log_probs
        return F.kl_div(ref, student_log_probs, reduction="none", log_target=True).sum(dim=-1)

    @staticmethod
    def _jsd_per_token(student_log_probs, teacher_log_probs, *, beta, skew_alpha):
        """soft divergence per token. β=1 reverse-KL (skew-able), β=0 forward-KL, 0<β<1 generalized JSD."""
        if beta == 1:
            return FusedLinearJSDWithFP32Softmax._rkl_per_token(student_log_probs, teacher_log_probs, skew_alpha)
        if beta == 0:
            return F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)
        log_mean_probs = torch.logsumexp(
            torch.stack([
                student_log_probs + math.log(1 - beta),
                teacher_log_probs + math.log(beta),
            ], dim=0),
            dim=0,
        )
        student_kl = F.kl_div(log_mean_probs, student_log_probs, reduction="none", log_target=True).sum(dim=-1)
        teacher_kl = F.kl_div(log_mean_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)
        return beta * teacher_kl + (1 - beta) * student_kl

    @staticmethod
    def _topk_fkl_per_token(student_log_probs, teacher_log_probs, k):
        """forward-KL(teacher ‖ student) restricted to teacher top-K (renormalized). Grad flows via student."""
        k = min(int(k), teacher_log_probs.shape[-1])
        t_val, idx = teacher_log_probs.topk(k, dim=-1)               # [C,k] log p_T
        t_renorm = t_val - torch.logsumexp(t_val, dim=-1, keepdim=True)
        s_at = student_log_probs.gather(-1, idx)                     # [C,k] log p_S (carries grad)
        s_renorm = s_at - torch.logsumexp(s_at, dim=-1, keepdim=True)
        return (t_renorm.exp() * (t_renorm - s_renorm)).sum(dim=-1)

    @staticmethod
    def _route_weights(student_log_probs, teacher_log_probs, target_chunk, *,
                       fkl_top_k, high_ent_nats, oc_hs_nats, oc_js, outlier_nll,
                       base_outlier_down, ignore_index):
        """token routing (advice §2), all detached (the gate is stop-grad). Returns (fkl_gate, base_scale)."""
        sp = student_log_probs.exp()
        tp = teacher_log_probs.exp()
        h_s = -(sp * student_log_probs).sum(dim=-1)
        h_t = -(tp * teacher_log_probs).sum(dim=-1)
        k = min(int(fkl_top_k), teacher_log_probs.shape[-1])
        t_val, idx = teacher_log_probs.topk(k, dim=-1)
        t_rn = t_val - torch.logsumexp(t_val, dim=-1, keepdim=True)
        s_rn = student_log_probs.gather(-1, idx)
        s_rn = s_rn - torch.logsumexp(s_rn, dim=-1, keepdim=True)
        m = torch.logsumexp(torch.stack([t_rn + math.log(0.5), s_rn + math.log(0.5)], dim=0), dim=0)
        js = 0.5 * (t_rn.exp() * (t_rn - m)).sum(dim=-1) + 0.5 * (s_rn.exp() * (s_rn - m)).sum(dim=-1)
        teacher_nll = -teacher_log_probs.gather(-1, target_chunk.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        high_ent = h_t > high_ent_nats
        oc_wrong = (h_s < oc_hs_nats) & (js > oc_js)
        outlier = teacher_nll > outlier_nll
        fkl_gate = (high_ent | oc_wrong | outlier).float()
        base_scale = 1.0 - base_outlier_down * (oc_wrong | outlier).float()
        return fkl_gate, base_scale

    @staticmethod
    def _compute_loss(
        student_input_chunk: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input_chunk: torch.Tensor,
        teacher_weight: torch.Tensor,
        target_chunk: torch.Tensor,
        tok_weight_chunk: torch.Tensor | None,
        *,
        num_valid_tokens: torch.Tensor,
        weight_hard_loss: float,
        weight_soft_loss: float,
        beta: float,
        ignore_index: int,
        temperature: float,
        compute_ce_loss: bool,
        skew_alpha: float,
        fkl_lambda: float,
        fkl_top_k: int,
        route_high_ent_nats: float,
        route_oc_hs_nats: float,
        route_oc_js: float,
        route_outlier_nll: float,
        base_outlier_down: float,
    ):
        student_logits = student_input_chunk @ student_weight.t()
        with torch.no_grad():
            teacher_logits = teacher_input_chunk @ teacher_weight.t()

        if temperature == 1.0:
            student_log_probs_soft = F.log_softmax(student_logits.float(), dim=-1)
        else:
            student_log_probs_soft = F.log_softmax((student_logits / temperature).float(), dim=-1)
        teacher_log_probs = F.log_softmax((teacher_logits / temperature).float(), dim=-1)

        hard_loss = student_logits.new_zeros(())
        if compute_ce_loss:
            hard_log_probs = (
                student_log_probs_soft
                if temperature == 1.0
                else F.log_softmax(student_logits.float(), dim=-1)
            )
            hard_loss = F.nll_loss(
                hard_log_probs.view(-1, hard_log_probs.shape[-1]),
                target_chunk.view(-1),
                reduction="sum",
                ignore_index=ignore_index,
            )

        # ---- soft term: skew-able base divergence (+ routed top-K FKL) ----
        soft_per_tok = FusedLinearJSDWithFP32Softmax._jsd_per_token(
            student_log_probs_soft, teacher_log_probs, beta=beta, skew_alpha=skew_alpha)
        if fkl_lambda != 0.0:
            with torch.no_grad():
                fkl_gate, base_scale = FusedLinearJSDWithFP32Softmax._route_weights(
                    student_log_probs_soft, teacher_log_probs, target_chunk,
                    fkl_top_k=fkl_top_k, high_ent_nats=route_high_ent_nats,
                    oc_hs_nats=route_oc_hs_nats, oc_js=route_oc_js, outlier_nll=route_outlier_nll,
                    base_outlier_down=base_outlier_down, ignore_index=ignore_index)
            soft_per_tok = soft_per_tok * base_scale
            fkl_per_tok = FusedLinearJSDWithFP32Softmax._topk_fkl_per_token(
                student_log_probs_soft, teacher_log_probs, fkl_top_k)
            soft_per_tok = soft_per_tok + fkl_lambda * fkl_per_tok * fkl_gate
        if tok_weight_chunk is not None:
            soft_per_tok = soft_per_tok * tok_weight_chunk
        soft_loss = soft_per_tok.masked_fill(target_chunk == ignore_index, 0.0).sum()

        loss = weight_hard_loss * hard_loss / num_valid_tokens + weight_soft_loss * soft_loss / num_valid_tokens
        return loss, (soft_loss / num_valid_tokens, hard_loss / num_valid_tokens)

    @staticmethod
    def forward(
        ctx,
        student_input: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_input: torch.Tensor,
        teacher_weight: torch.Tensor,
        true_labels: torch.LongTensor,
        student_bias: torch.Tensor | None,
        teacher_bias: torch.Tensor | None,
        weight_hard_loss: float = 0.5,
        weight_soft_loss: float = 0.5,
        beta: float = 0.5,
        ignore_index: int = -100,
        temperature: float = 1.0,
        compiled: bool = False,
        chunk_size: int = 1024,
        compute_ce_loss: bool = True,
        return_soft_hard_loss: bool = False,
        tok_weight: torch.Tensor | None = None,
        skew_alpha: float = 0.0,
        fkl_lambda: float = 0.0,
        fkl_top_k: int = 64,
        route_high_ent_nats: float = 2.5,
        route_oc_hs_nats: float = 0.30,
        route_oc_js: float = 0.30,
        route_outlier_nll: float = 8.0,
        base_outlier_down: float = 1.0,
    ):
        if student_bias is not None or teacher_bias is not None:
            raise NotImplementedError("OPD JSD wrapper supports bias-free lm heads only")

        grad_weight = torch.zeros_like(student_weight)
        grad_input = torch.empty_like(student_input)
        loss_acc = torch.zeros((), device=student_input.device)
        soft_loss_acc = torch.zeros((), device=student_input.device) if return_soft_hard_loss else None
        hard_loss_acc = torch.zeros((), device=student_input.device) if return_soft_hard_loss else None
        num_valid_tokens = (true_labels != ignore_index).sum().clamp_min(1)

        def accumulate_chunk(student_input_chunk, teacher_input_chunk, target_chunk, tok_weight_chunk):
            (chunk_grad_input, chunk_grad_weight), (chunk_loss, (chunk_soft_loss, chunk_hard_loss)) = (
                torch.func.grad_and_value(
                    FusedLinearJSDWithFP32Softmax._compute_loss,
                    argnums=(0, 1),
                    has_aux=True,
                )(
                    student_input_chunk,
                    student_weight,
                    teacher_input_chunk,
                    teacher_weight,
                    target_chunk,
                    tok_weight_chunk,
                    num_valid_tokens=num_valid_tokens,
                    weight_hard_loss=weight_hard_loss,
                    weight_soft_loss=weight_soft_loss,
                    beta=beta,
                    ignore_index=ignore_index,
                    temperature=temperature,
                    compute_ce_loss=compute_ce_loss,
                    skew_alpha=skew_alpha,
                    fkl_lambda=fkl_lambda,
                    fkl_top_k=fkl_top_k,
                    route_high_ent_nats=route_high_ent_nats,
                    route_oc_hs_nats=route_oc_hs_nats,
                    route_oc_js=route_oc_js,
                    route_outlier_nll=route_outlier_nll,
                    base_outlier_down=base_outlier_down,
                )
            )
            grad_weight.add_(chunk_grad_weight)
            loss_acc.add_(chunk_loss)
            if return_soft_hard_loss:
                soft_loss_acc.add_(chunk_soft_loss)
                hard_loss_acc.add_(chunk_hard_loss)
            return chunk_grad_input

        if compiled:
            accumulate_chunk = torch.compile(accumulate_chunk)

        chunk_size = max(1, int(chunk_size))
        for start in range(0, student_input.shape[0], chunk_size):
            end = min(start + chunk_size, student_input.shape[0])
            tw_chunk = tok_weight[start:end] if tok_weight is not None else None
            grad_input[start:end].copy_(
                accumulate_chunk(
                    student_input[start:end],
                    teacher_input[start:end],
                    true_labels[start:end],
                    tw_chunk,
                )
            )

        ctx.save_for_backward(grad_input, grad_weight)
        if return_soft_hard_loss:
            return loss_acc, soft_loss_acc, hard_loss_acc
        return loss_acc

    @staticmethod
    def backward(ctx, grad_output, *args):
        grad_input, grad_weight = ctx.saved_tensors
        grad_input = grad_input * grad_output
        grad_weight = grad_weight * grad_output
        return (
            grad_input,
            grad_weight,
            None,  # teacher_input
            None,  # teacher_weight
            None,  # true_labels
            None,  # student_bias
            None,  # teacher_bias
            None,  # weight_hard_loss
            None,  # weight_soft_loss
            None,  # beta
            None,  # ignore_index
            None,  # temperature
            None,  # compiled
            None,  # chunk_size
            None,  # compute_ce_loss
            None,  # return_soft_hard_loss
            None,  # tok_weight
            None,  # skew_alpha
            None,  # fkl_lambda
            None,  # fkl_top_k
            None,  # route_high_ent_nats
            None,  # route_oc_hs_nats
            None,  # route_oc_js
            None,  # route_outlier_nll
            None,  # base_outlier_down
        )


def fused_linear_jsd_fp32_softmax(
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight_hard_loss: float,
    weight_soft_loss: float,
    beta: float,
    ignore_index: int,
    temperature: float,
    compiled: bool,
    chunk_size: int,
    compute_ce_loss: bool,
    tok_weight: torch.Tensor | None = None,
    skew_alpha: float = 0.0,
    fkl_lambda: float = 0.0,
    fkl_top_k: int = 64,
    route_high_ent_nats: float = 2.5,
    route_oc_hs_nats: float = 0.30,
    route_oc_js: float = 0.30,
    route_outlier_nll: float = 8.0,
    base_outlier_down: float = 1.0,
) -> torch.Tensor:
    return FusedLinearJSDWithFP32Softmax.apply(
        student_input,
        student_weight,
        teacher_input,
        teacher_weight,
        labels,
        None,
        None,
        float(weight_hard_loss),
        float(weight_soft_loss),
        float(beta),
        int(ignore_index),
        float(temperature),
        bool(compiled),
        int(chunk_size),
        bool(compute_ce_loss),
        False,
        tok_weight,
        float(skew_alpha),
        float(fkl_lambda),
        int(fkl_top_k),
        float(route_high_ent_nats),
        float(route_oc_hs_nats),
        float(route_oc_js),
        float(route_outlier_nll),
        float(base_outlier_down),
    )
