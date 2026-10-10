# Copyright 2026 proof-pilot. Apache-2.0.
"""PoolIngestor — rollout -> decode(answer-only) -> parse -> validity-gate -> pool.admit (write-back).

Design (PLAN agentic section, mirrors rollout_store's non-blocking writer + produce.py's dump hook):
- **gate**: every generation is a training sample (teacher /score runs as usual); **only parse-passing
  artifacts are written into the pool** as downstream context (the user's rule). Truncated
  (finish_reason=="length") -> not entered into the pool (but still trained on, still dumped by rollout_store).
- **Two extractions**: the training sample = the full continuation (with think, handled by the trainer);
  the **pool artifact = only the parsed answer (<solution>/<score>...), think discarded** — the downstream
  verifier should not see the prover's private reasoning (same as math_3r).
- **Non-blocking**: `append()` only enqueues (microseconds on the event loop); a background coroutine
  `run_in_executor`s the token->text decode, parses (cheap, on loop), and `pool.admit_*` (on loop; the
  index mutation and the sampler read share the same event loop -> lock-free).
- admit is independent of teacher success/failure (called **before** produce.py's teacher /score): whether a
  proof can serve as context is unrelated to its teacher score. A teacher failure only drops the training
  sample, not the pool artifact.

select: no pool node (nothing downstream consumes it) -> not parsed, only `admit_select` (a count, for fill_fraction).
"""
from __future__ import annotations

import asyncio
import logging

from opd_v2.agentic.pool import PoolStore
from opd_v2.config import OPDConfig

log = logging.getLogger("opd_v2.agentic.writeback")

_STOP = object()


def parse_artifact(text: str, stage: str, finish_reason: str | None):
    """Parse a student generation (answer-only text) into a pool-artifact dict; return None if invalid.

    Reuses the math_3r parser (does not re-implement it): prove/refine use _two_section validity
    (not-truncated ∧ has <solution> ∧ score∈{0,.5,1} ∧ len>500); verify needs a parseable score and no truncation.
    """
    import os
    import sys
    m3r = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "distill_gen", "math_3r"))
    if m3r not in sys.path:
        sys.path.insert(0, m3r)
    from parser import parse_proof_package, parse_refined_package, parse_verification  # noqa: E402

    call = {"content": text or "", "finish_reason": finish_reason, "error": None}
    if stage in ("prove", "refine"):
        pkg = (parse_proof_package if stage == "prove" else parse_refined_package)(call, "X0")
        if not pkg.valid:
            return None
        return {"content": pkg.proof, "self_eval": pkg.self_eval, "self_score": pkg.self_score}
    if stage == "verify":
        if finish_reason == "length":
            return None
        v = parse_verification(call, "X0", 0)
        if v.score is None:
            return None
        return {"score": v.score, "text": v.text}
    return None


class PoolIngestor:
    """Write student rollouts back into the pool as a side channel (non-blocking, background decode/parse/admit)."""

    def __init__(self, pool: PoolStore, tokenizer, cfg: OPDConfig):
        self.pool = pool
        self.tok = tokenizer
        self.cfg = cfg
        self._q: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.n_admitted = {"prove": 0, "verify": 0, "refine": 0, "select": 0}
        self.n_rejected = 0
        self.n_seen = 0

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._q = asyncio.Queue()
        self._task = asyncio.create_task(self._run(), name="pool_ingestor")

    def append(self, full_ids: list, prompt_len: int, wv: int, meta: dict,
               finish_reason: str | None) -> None:
        """Non-blocking enqueue (mirrors dump.append). Called before produce.py's teacher /score."""
        if self._q is None:
            return
        self.n_seen += 1
        self._q.put_nowait((list(full_ids), int(prompt_len), int(wv), dict(meta or {}), finish_reason))

    async def _run(self) -> None:
        while True:
            item = await self._q.get()
            if item is _STOP:
                break
            try:
                await self._ingest(*item)
            except Exception:
                log.exception("pool ingest failed (skipped)")

    async def _ingest(self, full_ids, prompt_len, wv, meta, finish_reason) -> None:
        stage = meta.get("stage")
        problem_id = meta.get("problem_id")
        if stage is None or problem_id is None:
            return
        if stage == "select":
            if finish_reason == "length":      # a truncated select doesn't count as progress (so fill_fraction doesn't overestimate select, R3)
                self.n_rejected += 1
                return
            self.pool.admit_select(problem_id, wv=wv, source="student")
            self.n_admitted["select"] += 1
            return
        gen_ids = full_ids[prompt_len:]
        if not gen_ids:
            self.n_rejected += 1
            return
        text = await self._loop.run_in_executor(None, self._decode, gen_ids)
        art = parse_artifact(text, stage, finish_reason)
        if art is None:
            self.n_rejected += 1
            return
        if stage == "prove":
            ok = self.pool.admit_proof(problem_id, art["content"], art["self_eval"],
                                       art["self_score"], wv=wv, source="student")
        elif stage == "verify":
            refs = meta.get("refs") or []
            ok = self.pool.admit_verify(problem_id, refs[0], art["score"], art["text"],
                                        wv=wv, source="student") if refs else None
        elif stage == "refine":
            ok = self.pool.admit_refined(problem_id, meta.get("refs") or [], art["content"],
                                         art["self_eval"], art["self_score"], wv=wv, source="student")
        else:
            ok = None
        if ok is not None:
            self.n_admitted[stage] += 1
        else:
            self.n_rejected += 1

    def _decode(self, gen_ids: list) -> str:
        return self.tok.decode(gen_ids, skip_special_tokens=False)

    async def close(self) -> None:
        if self._task is None:
            return
        self._q.put_nowait(_STOP)
        try:
            await self._task
        except Exception:
            log.exception("pool ingestor close error")
        self._task = None

    def stats(self) -> dict:
        return {"seen": self.n_seen, "rejected": self.n_rejected,
                "queue": (self._q.qsize() if self._q is not None else 0), **self.n_admitted}
