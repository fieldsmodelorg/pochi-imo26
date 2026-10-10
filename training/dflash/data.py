"""Data loading for DFlash training — reads proof-pilot L4 pre-packed bins directly.

L4 format (see training/stage1_v2 `build_l4.py`): offline mix-weighted render +
global row shuffle + FFD packing into fixed-length bins.

    input_ids.i32   [n_bins, micro_len] int32 memmap
    loss_mask.bits  [n_bins, micro_len/8] bit-packed (1 = assistant target token)
    seg_ptr.i64 / seg_lens.i32   per-bin segment (document) lengths; a trailing
                                 pad segment is present iff the last segment
                                 carries no loss token (real docs always have >=1)
    meta.json       n_bins / micro_len / pad_id / tokenizer / mix fingerprint

Each __getitem__ returns one packed bin:
    input_ids      (L,) int64
    loss_mask      (L,) int64
    document_ids   (L,) int64, -1 on the trailing pad segment
    position_ids   (L,) int64, per-document reset (matches the L4/stage-1 layout
                   and lets the target model derive packing cu_seqlens)
    attention_mask (L,) int64 (document_ids >= 0)

Rank striping mirrors stage1_v2 `iter_l4_bins`: a seeded global permutation per
epoch, striped rank::world and truncated to floor(N/world) so every rank sees
the same number of identical-cost bins (no end-of-epoch collective mismatch).
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler


class L4Dataset(Dataset):
    """One item = one pre-packed L4 bin (memmap-backed, lazy per-process open)."""

    def __init__(self, root: str, max_bins: int | None = None):
        self.root = root
        with open(os.path.join(root, "meta.json")) as f:
            self.meta = json.load(f)
        assert self.meta.get("format") == "proof-pilot-l4-v1", self.meta.get("format")
        self.n_bins_total = self.meta["n_bins"]
        self.micro_len = self.meta["micro_len"]
        self.pad_id = self.meta["pad_id"]
        # Bins were globally shuffled at build time -> a prefix is an iid
        # subsample of the mix. max_bins gives cheap dev-scale runs.
        self.n_bins = min(self.n_bins_total, max_bins) if max_bins else self.n_bins_total
        self.seg_ptr = np.fromfile(os.path.join(root, "seg_ptr.i64"), dtype=np.int64)
        self.seg_lens = np.fromfile(os.path.join(root, "seg_lens.i32"), dtype=np.int32)
        self._ids_mm = None
        self._msk_mm = None

    def _ensure_open(self):
        if self._ids_mm is None:
            N, L = self.n_bins_total, self.micro_len
            self._ids_mm = np.memmap(
                os.path.join(self.root, "input_ids.i32"), dtype=np.int32, mode="r", shape=(N, L)
            )
            self._msk_mm = np.memmap(
                os.path.join(self.root, "loss_mask.bits"), dtype=np.uint8, mode="r", shape=(N, L // 8)
            )

    def __len__(self):
        return self.n_bins

    def __getitem__(self, j: int) -> Dict[str, Any]:
        self._ensure_open()
        L = self.micro_len
        row = np.asarray(self._ids_mm[j])
        mask = np.unpackbits(np.asarray(self._msk_mm[j]), count=L).astype(bool)
        lens = self.seg_lens[self.seg_ptr[j] : self.seg_ptr[j + 1]]
        assert int(lens.sum()) == L, f"bin {j}: segment lengths sum {lens.sum()} != {L}"

        # Trailing pad segment iff the last segment carries no loss token
        # (identical rule to stage1_v2 iter_l4_bins).
        padded = not mask[L - int(lens[-1]) :].any()

        doc_ids = np.repeat(np.arange(len(lens), dtype=np.int64), lens)
        if padded:
            doc_ids[L - int(lens[-1]) :] = -1
        pos = np.concatenate([np.arange(int(l), dtype=np.int64) for l in lens])

        return {
            "input_ids": torch.from_numpy(row.astype(np.int64)),
            "loss_mask": torch.from_numpy(mask.astype(np.int64)),
            "document_ids": torch.from_numpy(doc_ids),
            "position_ids": torch.from_numpy(pos),
            "attention_mask": torch.from_numpy((doc_ids >= 0).astype(np.int64)),
        }


class L4StripeSampler(Sampler):
    """Per-epoch seeded global permutation, striped rank::world, truncated to
    floor(N/world) so all ranks iterate the same number of bins."""

    def __init__(self, n_bins: int, rank: int, world_size: int, seed: int = 42, shuffle: bool = True):
        self.n_bins = n_bins
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self._epoch = 0
        self.per_rank = n_bins // world_size

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _order(self) -> np.ndarray:
        if self.shuffle:
            perm = np.random.RandomState(self.seed + self._epoch).permutation(self.n_bins)
        else:
            perm = np.arange(self.n_bins)
        return perm[self.rank :: self.world_size][: self.per_rank]

    def __iter__(self):
        return iter(self._order().tolist())

    def __len__(self):
        return self.per_rank


def resume_epoch_offset(consumed: int, per_rank: int) -> tuple[int, int]:
    """Map a per-rank *consumed bins* count to ``(epoch, within_epoch_offset)``.

    Pure / deterministic so resume is reproducible. ``consumed`` may exceed
    ``per_rank`` (a run that wrapped one or more epochs before being requeued).
    """
    if per_rank <= 0:
        return 0, 0
    return consumed // per_rank, consumed % per_rank


def epoch_resumable_iter(loader):
    """Yield batches forever, reshuffling the stripe permutation every epoch and,
    on resume, starting at the correct ``(epoch, offset)``.

    The sampler's ``.skip`` (set by ``build_resumable_dataloader`` to
    ``start_step - 1`` = per-rank bins already consumed) is read as a *global*
    consumed count and mapped onto an epoch + within-epoch offset. This fixes two
    bugs in the old ``set_epoch(0)`` + ``while True: yield from dl`` loop that
    only set the epoch once and never reset ``.skip``:

    1. every epoch replayed the *same* permutation (no reshuffle) -> verbatim
       data repetition on multi-epoch runs (e.g. the 65k-ctx OPD build is only
       ~556 bins/rank, so a 4000-step run wrapped it ~7x in identical order);
    2. resuming past one epoch left ``order[skip:]`` empty (``skip >= per_rank``)
       -> the generator spun forever yielding nothing = a silent training hang on
       any requeue after the first epoch.

    NOT ``itertools.cycle``: cycle caches every yielded (pinned) bin -> tens of
    GiB of page-locked host RAM over a long run.
    """
    sampler = loader.sampler
    epoch, within = resume_epoch_offset(getattr(sampler, "skip", 0), sampler.per_rank)
    while True:
        sampler.set_epoch(epoch)
        sampler.skip = within
        yield from loader
        epoch += 1
        within = 0


def packed_collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {key: torch.stack([f[key] for f in features]) for key in features[0].keys()}


def build_dataloader(
    data_path: str,
    batch_size: int,
    num_workers: int = 2,
    seed: int = 42,
    shuffle: bool = True,
    max_bins: int | None = None,
) -> DataLoader:
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    dataset = L4Dataset(data_path, max_bins=max_bins)
    sampler = L4StripeSampler(len(dataset), rank, world_size, seed=seed, shuffle=shuffle)

    if rank == 0:
        print(
            f"[data] L4 {data_path}: {len(dataset)}/{dataset.n_bins_total} bins x "
            f"{dataset.micro_len} tokens, {sampler.per_rank} bins/rank",
            flush=True,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=packed_collate_fn,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
