# Copyright 2026 proof-pilot. Apache-2.0.
"""PoolSampler — flow-balanced sampling (replaces single-round's iter_prompts_forever).

Each atom: pick a role -> assemble context from the pool -> yield a Prompt (token-in-token-out).

**The mix is auto-balanced by fill-fraction, not hand-picked fixed ratios** (PLAN agentic section):
- `fill_fraction(role) = student_count(role) / role_mix_weight(role)`; pick the available role with the
  **lowest fill_fraction** (softmax_temp>0 adds randomness to avoid thrash). -> the four roles lockstep
  toward the role_mix ratios.
- Within a role, pick the **item with the largest deficit** (verify picks the least-verified proof; refine
  picks a verified-but-under-refined problem) -> spread, **avoiding "generate proofs like crazy, verify
  can't keep up, un-verified proofs pile up"** (the failure mode the user pointed out).
- role_mix 22/44/20/14: verify=2×prove (= the fan-out of 2 verifies per proof) -> verify naturally
  dominates and keeps up with proof.

Note: `next()` includes tokenizer.encode (synchronous, may slightly block the event loop) — same as the
existing ProblemPromptLoader pattern; under long-CoT atoms are sparse (a single decode takes minutes), so
it's cheap. The iterator never StopIterations (prove is always yieldable).
"""
from __future__ import annotations

import logging
import math
import random
from typing import Iterator

from opd_v2.agentic.pool import PoolStore
from opd_v2.agentic.roles import RolePromptBuilder
from opd_v2.config import OPDConfig
from opd_v2.data_plane.produce import Prompt

log = logging.getLogger("opd_v2.agentic.sampler")

_ORDER = ("prove", "verify", "refine", "select")   # fixed order for tie-breaking


class PoolSampler:
    def __init__(self, cfg: OPDConfig, pool: PoolStore, builder: RolePromptBuilder | None = None):
        self.cfg = cfg
        self.pool = pool
        self.builder = builder or RolePromptBuilder(cfg.trainer.student_path, cfg)
        self.rng = random.Random(cfg.seed)
        self.n_yielded = {r: 0 for r in _ORDER}        # count of yielded roles (observation; readable by wandb)
        self.n_render_dropped = {r: 0 for r in _ORDER}  # build_* returned None (prompt too long / no item) -> that role was dropped
        self.n_fallback_prove = 0                       # number of times all roles failed and we fell back to prove

    # ---- role selection ----
    def _role_weights(self) -> dict:
        rm = self.cfg.agentic.role_mix
        return {r: float(rm.get(r, 0.0)) for r in _ORDER if float(rm.get(r, 0.0)) > 0.0}

    def _pick_role(self, available: set, counts: dict) -> str | None:
        weights = self._role_weights()
        cands = [r for r in _ORDER if r in available and r in weights]
        if not cands:
            return None
        fill = {r: counts.get(r, 0) / weights[r] for r in cands}   # lower = more behind = more due for sampling
        temp = max(1e-3, self.cfg.agentic.softmax_temp)
        # softmax over (-fill/temp), numerically stable
        xs = [-fill[r] / temp for r in cands]
        mx = max(xs)
        es = [math.exp(x - mx) for x in xs]
        s = sum(es) or 1.0
        probs = [e / s for e in es]
        return self.rng.choices(cands, weights=probs, k=1)[0]

    # ---- role -> Prompt ----
    def _build(self, role: str) -> Prompt | None:
        p = self.pool
        if role == "prove":
            prob = p.pick_prove_problem(self.cfg, self.rng)
            return self.builder.build_prove(prob) if prob else None
        if role == "verify":
            tgt = p.pick_verify_target(self.cfg, self.rng)
            return self.builder.build_verify(*tgt) if tgt else None
        if role == "refine":
            prob = p.pick_refine_problem(self.cfg, self.rng)
            return self.builder.build_refine(prob) if prob else None
        if role == "select":
            prob = p.pick_select_problem(self.cfg, self.rng)
            return self.builder.build_select(prob) if prob else None
        return None

    def next_prompt(self) -> Prompt | None:
        """Pick a role -> assemble a Prompt. Failure within a role (no item / prompt too long) -> record a render-drop and fall back to another role."""
        available = self.pool.available_roles(self.cfg)
        if not available:
            return None                       # pool is empty (no problems) — the caller should have seeded it
        counts = self.pool.student_counts()   # O(1), computed once per atom (not recomputed inside the retry loop, P1)
        tried: set = set()
        for _ in range(len(_ORDER) + 1):
            remaining = available - tried
            if not remaining:
                break
            role = self._pick_role(remaining, counts)
            if role is None:
                break
            tried.add(role)
            prompt = self._build(role)
            if prompt is not None:
                self.n_yielded[role] += 1
                return prompt
            self.n_render_dropped[role] += 1   # build returned None (no item / prompt too long) -> make it visible (B3)
        # all failed -> last resort: prove (simplest context, almost never None)
        prob = self.pool.pick_prove_problem(self.cfg, self.rng)
        prompt = self.builder.build_prove(prob) if prob else None
        if prompt is not None:
            self.n_yielded["prove"] += 1
            self.n_fallback_prove += 1
        return prompt

    def stats(self) -> dict:
        return {"yielded": dict(self.n_yielded), "render_dropped": dict(self.n_render_dropped),
                "fallback_prove": self.n_fallback_prove}

    def iter_forever(self) -> Iterator[Prompt]:
        """Infinite prompt stream (used by the scheduler). next_prompt only returns None if the pool has no
        problems at all (= no seed / misconfiguration) -> fatal, raise so the orchestrator fails clearly
        (don't quietly treat it as exhaustion)."""
        while True:
            prompt = self.next_prompt()
            if prompt is None:
                raise RuntimeError("PoolSampler: pool has no problems (seed not loaded?)")
            yield prompt
