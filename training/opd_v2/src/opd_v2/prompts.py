# Copyright 2026 proof-pilot. Apache-2.0.
"""prompt source — reuses v1 `opd.prompts.ProblemPromptLoader` (distill_gen proof-problem bank + prover
template -> student chat template -> input_ids), wrapped as an **infinite** iterator (reshuffle each epoch).

OPD rollout needs the prompt's token_ids (token-in-token-out). The v1 loader is already validated, so it is reused directly.
"""
from __future__ import annotations

import os
import sys
from typing import Iterator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "_vendor_opd")))

from opd_v2.config import OPDConfig
from opd_v2.data_plane.produce import Prompt


def iter_prompts_forever(cfg: OPDConfig) -> Iterator[Prompt]:
    """Infinite prompt stream: after one pass over the problem bank, reshuffle with a new seed and repeat (for continuous training)."""
    from opd.prompts import ProblemPromptLoader

    epoch = 0
    while True:
        loader = ProblemPromptLoader(
            cfg.trainer.student_path, cfg.problems_parquet,
            template_pool=list(cfg.prover_template_pool),
            seed=cfg.seed + epoch * 100003)
        n = 0
        for p in loader.iter_prompts(shuffle=True):
            n += 1
            yield Prompt(ids=p.input_ids,
                         meta={"id": p.id, "domain": p.domain, **(p.meta or {})})
        if n == 0:
            raise RuntimeError(f"prompt source produced 0 prompts (parquet={cfg.problems_parquet})")
        epoch += 1


def iter_prompts_debug(n: int = 64, plen: int = 48, seed: int = 0) -> Iterator[Prompt]:
    """Fake prompt stream with no tokenizer/parquet dependency (for unit tests / quick smoke)."""
    import random
    rng = random.Random(seed)
    i = 0
    while i < n:
        yield Prompt(ids=[rng.randint(3, 129279) for _ in range(plen)], meta={"id": f"dbg-{i}"})
        i += 1
