"""Learning rate schedulers."""

from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler as _LRScheduler


class _enable_get_lr_call:
    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = int(warmup_epochs)
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()
        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)

    def state_dict(self):
        state_dict = {
            key: value for key, value in self.__dict__.items() if key not in "optimizer"
        }
        if isinstance(state_dict["after_scheduler"], _LRScheduler):
            state_dict["after_scheduler_type"] = type(
                state_dict["after_scheduler"]
            ).__name__
            state_dict["after_scheduler_dict"] = state_dict[
                "after_scheduler"
            ].state_dict()
            del state_dict["after_scheduler"]
        else:
            raise NotImplementedError()
        return state_dict

    def load_state_dict(self, state_dict):
        if "after_scheduler_dict" in state_dict:
            self.after_scheduler.load_state_dict(state_dict["after_scheduler_dict"])
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key not in ("after_scheduler_type", "after_scheduler_dict")
        }
        super().load_state_dict(state_dict)


class CosineAnnealingWarmupLR(WarmupScheduler):
    def __init__(
        self,
        optimizer,
        total_optimizer_steps: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        base_scheduler = _CosineAnnealingLR(
            optimizer,
            total_optimizer_steps - warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
        super().__init__(optimizer, warmup_steps, base_scheduler, last_epoch=last_epoch)
