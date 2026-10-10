# Copyright 2026 proof-pilot. Apache-2.0.
"""OPD full-vocab divergence loss — Liger fused-linear JSD(β) (PLAN §1/§2, D2/D3).

GKD directly-differentiable divergence (not PPO/IS): at every generated position of the student
rollout, minimize `D_JSD(β)(π_T ‖ π_θ)`. Uses `LigerFusedLinearJSDFunction` to reconstruct both
sides' logits per-token-chunk and compute the divergence inside the kernel, **without materializing
[BT,V]** (numerically validated in distill/_liger_jsd_test).

OPD is pure distillation (the teacher distribution is the target, no ground-truth label):
weight_hard(CE)=0, weight_soft=1. labels are only used for the ignore mask — prompt/pad positions
are set to IGNORE and skipped, generated positions get their next-token (the value does not enter
the loss, it only marks "to be computed"). Liger's mean reduction is length-normalized (D8).

Inputs (assembled on the trainer side; teacher hidden is the rotated-space value from codec.decode,
paired with W_rot):
  student_hidden     [BT, H_s]   student hidden at generated positions
  student_head_w     [V, H_s]    student lm_head.weight
  teacher_hidden_rot [BT, H_t]   decode(quant) rotated-space hidden
  w_rot              [V, H_t]    fold_head(teacher head) (rotated space)
  labels             [BT]        generated next-token id; masked positions = IGNORE

β: 0=forward KL (mode-covering), 0.5=JSD, 1=reverse KL (mode-seeking, canonical OPD).
"""
from __future__ import annotations

import torch

from liger_kernel.chunked_loss.jsd_loss import LigerFusedLinearJSDFunction

IGNORE = -100


def opd_jsd_loss(student_hidden: torch.Tensor, student_head_w: torch.Tensor,
                 teacher_hidden_rot: torch.Tensor, w_rot: torch.Tensor,
                 labels: torch.Tensor, beta: float = 0.5, temperature: float = 1.0,
                 chunk_size: int = 1024, compiled: bool = False) -> torch.Tensor:
    # compiled=False: BT (the number of targets) differs per bin -> dynamo recompiles every step and
    # the third compilation hits the produce_guards IndexError (torch dynamic-shape bug; mnlong1 step1
    # blows up on all ranks). eager chunked is a bit slower but shape-immune.
    """Pure-distillation full-vocab JSD(β), length-normalized scalar.

    arg order matches distill/_liger_jsd_test:
      apply(x_s, w_s, x_t, w_t, labels, bias_s, bias_t, w_hard, w_soft, beta,
            ignore_index, temperature, compiled, chunk_size, <accum bool>)
    """
    return LigerFusedLinearJSDFunction.apply(
        student_hidden, student_head_w, teacher_hidden_rot, w_rot, labels,
        None, None, 0.0, 1.0, beta, IGNORE, temperature, compiled, chunk_size, False,
    )


class OPDLoss:
    """A thin wrapper holding a LossCfg."""

    def __init__(self, beta: float = 0.5, temperature: float = 1.0, chunk_size: int = 1024):
        self.beta = beta
        self.temperature = temperature
        self.chunk_size = chunk_size

    def __call__(self, student_hidden, student_head_w, teacher_hidden_rot, w_rot, labels):
        return opd_jsd_loss(student_hidden, student_head_w, teacher_hidden_rot, w_rot, labels,
                            beta=self.beta, temperature=self.temperature,
                            chunk_size=self.chunk_size)
