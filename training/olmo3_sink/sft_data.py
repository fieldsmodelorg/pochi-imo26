# Copyright 2026 proof-pilot. Apache-2.0.
"""Sequence-packing for Olmo3Sink SFT.

Greedy first-fit packing of variable-length tokenized examples into fixed-length
rows. Emits exactly what the `olmo3_sink_fa3` varlen path wants:

  - input_ids   [1, L]         several docs concatenated into one row
  - position_ids[1, L]         **reset to 0 at each doc start** -> the model derives
                               varlen cu_seqlens from this (packing-metadata reuse), so
                               attention never crosses doc boundaries
  - labels      [1, L]         -100 on padding AND (optionally) on prompt tokens

The per-doc position reset is the contract the whole sink/packing stack relies on
(`Olmo3SinkModel.forward` -> `_is_packed_sequence` -> `prepare_fa_kwargs_from_position_ids`).

A "doc" here = one training example = {prompt_ids, response_ids}. We pack whole docs
only (never split a doc across rows) because attention + loss must see a doc intact.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

IGNORE = -100


@dataclass
class Example:
    """One tokenized SFT example.

    Loss masking is one of two mutually-exclusive forms:
      - `labels` given: explicit per-token labels (IGNORE where masked). This is what the
        L3 renderer emits for multi-turn / multi-span SFT (assistant spans only).
      - else `prompt_len`: a single leading prefix of `prompt_len` tokens is masked
        (the simple single-turn / smoke-test case).
    """
    input_ids: list[int]
    prompt_len: int = 0  # leading tokens to mask with IGNORE (ignored if `labels` set)
    labels: list[int] | None = None  # explicit mask; must match len(input_ids) if set

    def __post_init__(self) -> None:
        if self.labels is not None and len(self.labels) != len(self.input_ids):
            raise ValueError(f"labels len {len(self.labels)} != input_ids len {len(self.input_ids)}")

    def __len__(self) -> int:
        return len(self.input_ids)


def greedy_pack(examples: list[Example], max_len: int) -> list[list[Example]]:
    """First-fit-decreasing packing of whole examples into bins of capacity max_len.

    Examples longer than max_len are dropped (caller should pre-truncate). Returns a
    list of bins; each bin is a list of Examples whose total length <= max_len.
    """
    kept = [e for e in examples if 0 < len(e) <= max_len]
    kept.sort(key=len, reverse=True)
    bins: list[list[Example]] = []
    bin_fill: list[int] = []
    for e in kept:
        placed = False
        for i, fill in enumerate(bin_fill):
            if fill + len(e) <= max_len:
                bins[i].append(e)
                bin_fill[i] += len(e)
                placed = True
                break
        if not placed:
            bins.append([e])
            bin_fill.append(len(e))
    return bins


def pack_to_tensors(bin_: list[Example], max_len: int, pad_id: int, device="cpu",
                    max_segs: int | None = None):
    """Materialize one packed bin into (input_ids, position_ids, labels), all [1, max_len].

    Padding fills the tail to max_len; padding positions get position 0 and label IGNORE.
    The pad tail is itself a trailing "doc" (position reset) so it can't attend into real docs.

    If `max_segs` is given, also emit FIXED-shape varlen metadata so torch.compile sees one
    static structure across steps (otherwise the varying #docs changes `cu_seqlens` length +
    `max_seqlen`, which blows past Dynamo's recompile cache limit -> eager fallback). The
    `cu_seqlens` is padded to `max_segs+1` with trailing zero-length docs (cu repeats `max_len`),
    and `max_length_*` is the constant `max_len`. FA3 varlen skips zero-length segments and an
    over-estimated max_seqlen is safe (only affects scheduling). The model sees these in the
    batch and reuses them (skips its own per-step varlen recompute)."""
    ids: list[int] = []
    pos: list[int] = []
    lab: list[int] = []
    seglens: list[int] = []
    for e in bin_:
        ids.extend(e.input_ids)
        pos.extend(range(len(e)))
        if e.labels is not None:
            lab.extend(e.labels)
        else:
            lab.extend([IGNORE] * e.prompt_len + e.input_ids[e.prompt_len:])
        seglens.append(len(e))
    pad = max_len - len(ids)
    if pad > 0:
        ids.extend([pad_id] * pad)
        pos.extend(range(pad))      # own position-reset segment -> isolated from real docs
        lab.extend([IGNORE] * pad)
        seglens.append(pad)
    t = lambda x, dt=torch.long: torch.tensor(x, dtype=dt, device=device)[None]
    out = {"input_ids": t(ids), "position_ids": t(pos), "labels": t(lab)}
    # real docs packed into this bin (excludes the trailing pad segment, if any) -- used by the
    # trainer to log a fractional epoch. Plain int; survives _to_device (only tensors are moved).
    out["n_docs"] = len(seglens) - (1 if pad > 0 else 0)

    if max_segs is not None:
        n = len(seglens)
        assert n <= max_segs, f"bin has {n} segments > max_segs={max_segs}"
        cu = torch.zeros(max_segs + 1, dtype=torch.int32, device=device)
        cu[1:n + 1] = torch.tensor(seglens, dtype=torch.int32, device=device).cumsum(0)
        cu[n + 1:] = max_len  # trailing zero-length docs -> FA3 skips them
        out["cu_seq_lens_q"] = cu
        out["cu_seq_lens_k"] = cu
        out["max_length_q"] = max_len
        out["max_length_k"] = max_len
    return out


class PackedCollator:
    """Collate pre-tokenized Examples into packed fixed-length rows.

    Usage (single row / batch=1, which is the SFT packing convention):
        coll = PackedCollator(max_len=8192, pad_id=tok.pad_token_id)
        batch = coll(list_of_examples)   # -> dict of [1, max_len] tensors (one bin)

    For grad-accumulation over a packed dataset, iterate `coll.iter_bins(...)`.
    """

    def __init__(self, max_len: int, pad_id: int, device="cpu", max_segs: int | None = None):
        self.max_len = max_len
        self.pad_id = pad_id
        self.device = device
        self.max_segs = max_segs  # set (>= max #docs/bin) to emit fixed-shape varlen (compile-stable)

    def __call__(self, examples: list[Example]) -> dict:
        # one call -> one packed row; assumes caller passed examples that fit one bin
        return pack_to_tensors(examples, self.max_len, self.pad_id, self.device, self.max_segs)

    def iter_bins(self, examples: list[Example]):
        for b in greedy_pack(examples, self.max_len):
            yield pack_to_tensors(b, self.max_len, self.pad_id, self.device, self.max_segs)
