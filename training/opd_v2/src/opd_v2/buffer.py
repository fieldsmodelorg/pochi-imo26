# Copyright 2026 proof-pilot. Apache-2.0.
"""Lightweight trajectory buffer (V16) — stores only `{ids, prompt_len, wv, handle}`, **no teacher bytes**.

The v1 buffer stored the whole teacher hidden bytes inside the Trajectory (~3.3KB/token), so 4096 long-CoT
trajectories ate hundreds of GB of RAM. The v2 buffer stores only token ids + a HiddenHandle (pointing to
the hidden file on shared FS), so the trajectory/token caps can be much larger and the scatter is
ultra-light. The bytes are read directly from FS by the owning trainer rank via the handle (see hidden_store).

near-on-policy: no importance ratio; drops trajectories that are too old via `max_staleness`
(`cur_step - wv > max_staleness`). The staleness rule is factored into a pure function `is_stale` for easy
unit testing. The orchestrator is a single asyncio event loop (single-threaded); the lock is only for safety
(there is no await inside the async critical section, so it is already atomic).
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from opd_v2.hidden_store import HiddenHandle


@dataclass
class ScoredTrajectory:
    """One student rollout + its already-scored teacher hidden handle (pointing to shared FS)."""
    ids: list[int]               # full sequence (prompt + generated), token-in-token-out
    prompt_len: int              # the first prompt_len tokens are the prompt
    wv: int                      # the weight_version when the rollout server generated it (server-reported, V6)
    handle: HiddenHandle         # handle to the teacher hidden on shared FS (no bytes)
    meta: dict = field(default_factory=dict)
    finish_reason: str = ""      # rollout stop reason: "stop"=EOS / "length"=window hit (truncation monitor; not sent on the wire)

    @property
    def gen_len(self) -> int:
        return len(self.ids) - self.prompt_len

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    def to_wire(self) -> dict:
        """The minimal representation sent to the trainer (used by both the HTTP body and gloo scatter; no bytes)."""
        return {"ids": self.ids, "prompt_len": self.prompt_len, "wv": self.wv,
                "handle": self.handle.to_dict()}

    @classmethod
    def from_wire(cls, d: dict) -> "ScoredTrajectory":
        return cls(ids=list(d["ids"]), prompt_len=int(d["prompt_len"]), wv=int(d["wv"]),
                   handle=HiddenHandle.from_dict(d["handle"]))


def is_stale(wv: int, cur_step: int, max_staleness: int) -> bool:
    """Whether a trajectory is too old and should be dropped. Pure function, easy to unit-test.

    `max_staleness <= 0` = **disabled** (never stale). OPD does not use an importance ratio (it doesn't store
    generation logprobs) -> staleness is not a correctness requirement, the teacher hidden is frozen and
    always valid, so a rollout from an old wv is equally legitimate distillation data. Long-CoT rollouts are
    very expensive, so by default this is off and nothing is dropped (see config default 0). Set a positive
    value to enable "prefer fresh".
    """
    return max_staleness > 0 and (cur_step - wv) > max_staleness


class TrajectoryBuffer:
    """Bounded trajectory queue; does staleness-dropping when a batch is pulled. FIFO + prefer-fresh (drop the oldest when full)."""

    def __init__(self, capacity: int, capacity_tokens: int | None = None):
        self.capacity = capacity
        self.capacity_tokens = capacity_tokens
        self._dq: deque[ScoredTrajectory] = deque()
        self._tok = 0
        self._lock = threading.Lock()
        self.n_put = 0
        self.n_dropped_stale = 0
        self.n_dropped_overflow = 0
        self.n_served = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)

    def token_count(self) -> int:
        with self._lock:
            return self._tok

    def near_full(self, frac: float = 0.9) -> bool:
        """Producer backpressure check: either the trajectory count or the token count approaches the cap."""
        with self._lock:
            if len(self._dq) >= self.capacity * frac:
                return True
            return bool(self.capacity_tokens) and self._tok >= self.capacity_tokens * frac

    def put(self, traj: ScoredTrajectory) -> list[ScoredTrajectory]:
        """Add one. If full, drop the oldest (backpressure + prefer-fresh). Returns the **evicted trajectories** (the caller GCs their handles)."""
        L = traj.n_tokens
        evicted: list[ScoredTrajectory] = []
        with self._lock:
            while self._dq and (len(self._dq) >= self.capacity
                                or (self.capacity_tokens and self._tok + L > self.capacity_tokens)):
                old = self._dq.popleft()
                self._tok -= old.n_tokens
                self.n_dropped_overflow += 1
                evicted.append(old)
            self._dq.append(traj)
            self._tok += L
            self.n_put += 1
        return evicted

    def get_batch(self, n: int, cur_step: int, max_staleness: int
                  ) -> tuple[list[ScoredTrajectory], list[ScoredTrajectory]]:
        """Pull up to n fresh-enough trajectories; drop any stale ones encountered. Returns (kept, dropped_stale).

        dropped_stale is returned too so the caller can GC their hidden files (V14: unlink after stale-drop).
        """
        out: list[ScoredTrajectory] = []
        stale: list[ScoredTrajectory] = []
        with self._lock:
            while self._dq and len(out) < n:
                t = self._dq.popleft()
                self._tok -= t.n_tokens
                if is_stale(t.wv, cur_step, max_staleness):
                    self.n_dropped_stale += 1
                    stale.append(t)
                    continue
                out.append(t)
            self.n_served += len(out)
        return out, stale

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._dq),
                "tokens": self._tok,
                "n_put": self.n_put,
                "n_served": self.n_served,
                "n_dropped_stale": self.n_dropped_stale,
                "n_dropped_overflow": self.n_dropped_overflow,
            }
