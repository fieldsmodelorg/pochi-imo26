"""Utility helpers for DFlash training."""

import logging
import os
import re
from contextlib import contextmanager

import torch.distributed as dist

logger = logging.getLogger(__name__)


def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        print(f"[rank {dist.get_rank()}] {message}", flush=True)
    else:
        print(f"[non-distributed] {message}", flush=True)


def print_on_rank0(message):
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        print(f"[rank 0] {message}", flush=True)


@contextmanager
def rank_0_priority():
    rank = dist.get_rank()
    if rank == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


def get_last_checkpoint(folder, prefix=r"epoch_\d+_step"):
    # Prefer latest/ checkpoint (updated most frequently, for resume)
    latest = os.path.join(folder, "latest")
    if os.path.isdir(latest):
        return latest

    # Fallback: find permanent checkpoint with highest step number
    content = os.listdir(folder)
    _re_checkpoint = re.compile(r"^" + prefix + r"_(\d+)$")
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return None
    return os.path.join(
        folder,
        max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])),
    )
