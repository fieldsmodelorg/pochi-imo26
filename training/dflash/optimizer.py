"""BF16 mixed-precision optimizer with FP32 master weights."""

import torch

from lr_scheduler import CosineAnnealingWarmupLR
from utils import print_on_rank0


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_optimizer_steps=800_000,
        warmup_ratio=0.015,
        warmup_steps=None,
        param_groups=None,
    ):
        self.model = model
        self.max_grad_norm = max_grad_norm

        if param_groups is not None:
            self.model_params = []
            self.fp32_params = []
            optimizer_groups = []
            for group in param_groups:
                group_model_params = [p for p in group["params"] if p.requires_grad]
                group_fp32_params = [
                    p.detach().clone().to(torch.float32) for p in group_model_params
                ]
                for mp in group_fp32_params:
                    mp.requires_grad = True
                self.model_params.extend(group_model_params)
                self.fp32_params.extend(group_fp32_params)
                opt_group = {k: v for k, v in group.items() if k != "params"}
                opt_group["params"] = group_fp32_params
                opt_group.setdefault("weight_decay", weight_decay)
                optimizer_groups.append(opt_group)
            self.optimizer = torch.optim.AdamW(optimizer_groups)
        else:
            self.model_params = [p for p in model.parameters() if p.requires_grad]
            self.fp32_params = [
                p.detach().clone().to(torch.float32) for p in self.model_params
            ]
            for mp in self.fp32_params:
                mp.requires_grad = True
            self.optimizer = torch.optim.AdamW(
                self.fp32_params, lr=lr, weight_decay=weight_decay
            )

        effective_warmup = (
            warmup_steps
            if warmup_steps is not None
            else int(warmup_ratio * total_optimizer_steps)
        )
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_optimizer_steps=total_optimizer_steps,
            warmup_steps=effective_warmup,
        )

    def step(self):
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )
        self._last_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.fp32_params, self.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None

    def state_dict(self):
        """Returns scheduler state only.

        Optimizer state is per-rank (FSDP1 sharded fp32 master copies) and must
        be saved/loaded per-rank via `optimizer_only_state_dict()` and
        `load_optimizer_only_state_dict()`. Saving optimizer state from a
        single rank loses 7/8 of the data and causes shape mismatches on resume.
        """
        return {
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state. Optimizer state must be loaded separately."""
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Loaded scheduler state_dict.")

    def optimizer_only_state_dict(self):
        """Returns this rank's local AdamW state (sharded fp32 master copies)."""
        return self.optimizer.state_dict()

    def load_optimizer_only_state_dict(self, opt_state):
        """Load this rank's local AdamW state."""
        self.optimizer.load_state_dict(opt_state)

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_grad_norm(self):
        return self._last_grad_norm if hasattr(self, "_last_grad_norm") else None
