"""Experiment tracking (wandb / tensorboard / none)."""

import abc
import netrc
import os
from typing import Any, Dict, Optional

import torch.distributed as dist

try:
    import wandb
except ImportError:
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class Tracker(abc.ABC):
    def __init__(self, args, output_dir: str):
        self.args = args
        self.output_dir = output_dir
        self.rank = dist.get_rank()
        self.is_initialized = False

    @abc.abstractmethod
    def log(self, log_dict: Dict[str, Any], step: Optional[int] = None) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class NoOpTracker(Tracker):
    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        self.is_initialized = True

    def log(self, log_dict, step=None):
        pass

    def close(self):
        pass


class WandbTracker(Tracker):
    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        if self.rank == 0:
            if args.wandb_key:
                wandb.login(key=args.wandb_key)
            wandb.init(
                project=args.wandb_project, name=args.wandb_name, config=vars(args)
            )
            self.is_initialized = True

    def log(self, log_dict, step=None):
        if self.rank == 0 and self.is_initialized:
            wandb.log(log_dict, step=step)

    def close(self):
        if self.rank == 0 and self.is_initialized and wandb.run:
            wandb.finish()
            self.is_initialized = False


class TensorboardTracker(Tracker):
    def __init__(self, args, output_dir: str):
        super().__init__(args, output_dir)
        if self.rank == 0:
            log_dir = os.path.join(output_dir, "runs")
            self.writer = SummaryWriter(log_dir=log_dir)
            self.is_initialized = True

    def log(self, log_dict, step=None):
        if self.rank == 0 and self.is_initialized:
            for key, value in log_dict.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, global_step=step)

    def close(self):
        if self.rank == 0 and self.is_initialized:
            self.writer.close()
            self.is_initialized = False


def create_tracker(args, output_dir: str) -> Tracker:
    report_to = getattr(args, "report_to", "none")
    if report_to == "wandb":
        if wandb is None:
            raise ImportError("pip install wandb")
        # Auto-detect key
        if not getattr(args, "wandb_key", None):
            args.wandb_key = os.environ.get("WANDB_API_KEY")
            if not args.wandb_key:
                try:
                    nrc = netrc.netrc(os.path.expanduser("~/.netrc"))
                    if "api.wandb.ai" in nrc.hosts:
                        _, _, args.wandb_key = nrc.authenticators("api.wandb.ai")
                except Exception:
                    pass
        return WandbTracker(args, output_dir)
    elif report_to == "tensorboard":
        if SummaryWriter is None:
            raise ImportError("pip install tensorboard")
        return TensorboardTracker(args, output_dir)
    else:
        return NoOpTracker(args, output_dir)
