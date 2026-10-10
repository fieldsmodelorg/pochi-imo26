# Copyright 2026 proof-pilot. Apache-2.0.
"""Unit tests for the epoch-resumable stripe iterator (data.epoch_resumable_iter).

Pins the two bugs the rewrite fixed in the old `set_epoch(0)` + `while True:
yield from dl` loop:
  (1) every epoch replayed the same permutation (no reshuffle);
  (2) resuming past one epoch (skip >= per_rank) hung forever yielding nothing.

Light-weight: no torch.distributed, no trainer import. The local
``_FakeResumableSampler`` mirrors ``nodup_fa3_train_prod.ResumableStripeSampler``
(a 3-line skip slice over the real ``L4StripeSampler._order``); the real
``resume_epoch_offset`` / ``epoch_resumable_iter`` are exercised directly.

Run:
  PYTHONPATH=training/dflash python training/dflash/tests/test_sampler_resume.py
"""
from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import L4StripeSampler, epoch_resumable_iter, resume_epoch_offset


class _FakeResumableSampler(L4StripeSampler):
    """Mirror of ResumableStripeSampler kept local so the test stays import-light."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.skip = 0

    def __iter__(self):
        return iter(self._order()[self.skip :].tolist())


class _FakeLoader:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        return iter(self.sampler)


def _order_for_epoch(n_bins, rank, world, seed, epoch):
    s = L4StripeSampler(n_bins, rank, world, seed=seed, shuffle=True)
    s.set_epoch(epoch)
    return s._order().tolist()


def _checks():
    n_bins, rank, world, seed = 40, 1, 4, 123
    per_rank = n_bins // world  # 10
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, "FAIL: " + msg
        n += 1

    # --- resume_epoch_offset pure mapping ---
    ok(resume_epoch_offset(0, per_rank) == (0, 0), "fresh -> (0,0)")
    ok(resume_epoch_offset(per_rank, per_rank) == (1, 0), "exactly one epoch -> (1,0)")
    ok(resume_epoch_offset(13, per_rank) == (1, 3), "13/10 -> (1,3)")
    ok(resume_epoch_offset(25, per_rank) == (2, 5), "25/10 -> (2,5)")
    ok(resume_epoch_offset(5, 0) == (0, 0), "per_rank=0 guard -> (0,0)")

    # --- (1) reshuffle: consecutive epochs use different permutations ---
    # _order() stripes rank::world over a *fresh global* permutation each epoch,
    # so a rank sees both a different ORDER and a different SUBSET of bins per
    # epoch (full-dataset coverage rotates across epochs).
    e0 = _order_for_epoch(n_bins, rank, world, seed, 0)
    e1 = _order_for_epoch(n_bins, rank, world, seed, 1)
    ok(e0 != e1, "epoch 0 and 1 differ (reshuffle)")
    ok(set(e0) != set(e1), "epoch 0 and 1 select different bin subsets (global reshuffle)")
    ok(len(set(e0)) == per_rank and len(e0) == per_rank, "epoch has per_rank distinct bins")

    # fresh run: first per_rank bins == epoch0 order, next per_rank == epoch1 order
    samp = _FakeResumableSampler(n_bins, rank, world, seed=seed, shuffle=True)
    it = epoch_resumable_iter(_FakeLoader(samp))
    got = list(itertools.islice(it, 2 * per_rank))
    ok(got[:per_rank] == e0, "fresh epoch 0 stream == e0")
    ok(got[per_rank : 2 * per_rank] == e1, "fresh epoch 1 stream == e1 (reshuffled)")

    # uninterrupted reference stream over 3 epochs
    e2 = _order_for_epoch(n_bins, rank, world, seed, 2)
    uninterrupted = e0 + e1 + e2  # 30 bins

    # --- (2) resume past one epoch must NOT hang and must continue correctly ---
    for consumed in (0, 3, 10, 13, 20, 25):
        samp_r = _FakeResumableSampler(n_bins, rank, world, seed=seed, shuffle=True)
        samp_r.skip = consumed  # what build_resumable_dataloader sets on resume
        it_r = epoch_resumable_iter(_FakeLoader(samp_r))
        remaining = len(uninterrupted) - consumed
        got_r = list(itertools.islice(it_r, remaining))  # would hang-equivalent (empty) under old bug
        ok(
            got_r == uninterrupted[consumed:],
            f"resume at consumed={consumed} reconstructs the uninterrupted continuation",
        )

    # explicit regression for the hang: skip >= per_rank used to yield empty forever
    samp_h = _FakeResumableSampler(n_bins, rank, world, seed=seed, shuffle=True)
    samp_h.skip = per_rank + 3  # 13 > 10
    it_h = epoch_resumable_iter(_FakeLoader(samp_h))
    first = next(it_h)  # must produce a bin, not spin on an empty order[skip:]
    ok(first == e1[3], "skip>=per_rank resumes into epoch 1 (no hang), first bin == e1[3]")

    print(f"OK test_sampler_resume: {n}/{n} checks passed")


if __name__ == "__main__":
    _checks()
