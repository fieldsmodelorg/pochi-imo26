# Copyright 2026 proof-pilot. Apache-2.0.
"""Async trajectory buffer with weight_version tagging + max_staleness dropping (PLAN §2/§4 D4).

OPD is near-on-policy: rollouts are drawn from a weight version a few steps behind the trainer.
There is no importance-ratio correction (full-vocab GKD, direct divergence), so "on-policy enough"
is achieved by **bounding staleness** rather than by IS weights. Each trajectory is tagged with the
`weight_version` at sampling time; when the trainer pulls a batch at step `cur_step`, it drops any
trajectory with `cur_step - weight_version > max_staleness`.

Pure logic, thread-safe (multiple async producers in the orchestrator + a trainer consumer); the
staleness rule is factored into the pure function `is_stale` for easy unit testing.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trajectory:
    """One student rollout + (after scoring) its corresponding quant teacher hidden.

    token_ids is the **full sequence** (prompt + generated), token-in-token-out throughout (D12).
    teacher_* are filled in after the teacher service scores it (packed/scales, rotated had+int6 space).
    """
    token_ids: list[int]                 # prompt + generated (full)
    prompt_len: int                      # the first prompt_len tokens are the prompt
    weight_version: int                  # trainer step on the rollout server at sampling time
    teacher_packed: Optional[bytes] = None   # had+int6 packed (after scoring)
    teacher_scales: Optional[bytes] = None   # fp16 scales (after scoring)
    teacher_seq_len: Optional[int] = None     # number of positions returned by the teacher service (usually = gen_len + 1)
    teacher_top1: Optional[bytes] = None      # optional int32 teacher argmax ids (same length as teacher_seq_len)
    position_offset: int = 0                  # RoPE position corresponding to token_ids[0]; used by long-window crop
    meta: dict = field(default_factory=dict)

    @property
    def gen_len(self) -> int:
        return len(self.token_ids) - self.prompt_len

    def target_positions(self) -> range:
        """Hidden positions that predict generated tokens: prompt_len-1 .. len-2 (inclusive),
        whose hidden, through the head, predicts token_ids[prompt_len .. len-1] (i.e. the generated span).
        This is the G4-alignment definition: teacher hidden[t] and student hidden[t] both predict token_ids[t+1]."""
        return range(self.prompt_len - 1, len(self.token_ids) - 1)

    def scored(self) -> bool:
        return self.teacher_packed is not None


def is_stale(weight_version: int, cur_step: int, max_staleness: int) -> bool:
    """Whether a trajectory is too old and should be dropped. Pure function, easy to unit-test."""
    return (cur_step - weight_version) > max_staleness


class TrajectoryBuffer:
    """A bounded, thread-safe trajectory queue that drops stale entries when a batch is pulled."""

    def __init__(self, capacity: int, capacity_tokens: int | None = None):
        """capacity = max number of trajectories; capacity_tokens = total token cap. For long CoT the token
        cap dominates: teacher hidden bytes ~3.3KB/token, so 4096 trajectories x 32k tokens would eat
        hundreds of GB of RAM and must be bounded by token count."""
        self.capacity = capacity
        self.capacity_tokens = capacity_tokens
        self._dq: deque[Trajectory] = deque()
        self._tok = 0                       # total tokens in the queue (maintained under the same lock as _dq)
        self._lock = threading.Lock()
        # statistics (for logging)
        self.n_put = 0
        self.n_dropped_stale = 0
        self.n_dropped_overflow = 0
        self.n_served = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)

    def near_full(self, frac: float = 0.9) -> bool:
        """Producer backpressure check: either the count or the token total is approaching the cap."""
        with self._lock:
            if len(self._dq) >= self.capacity * frac:
                return True
            return bool(self.capacity_tokens) and self._tok >= self.capacity_tokens * frac

    def put(self, traj: Trajectory) -> None:
        """Add one (must already be scored). If full, drop the oldest (backpressure + prefer freshness)."""
        if not traj.scored():
            raise ValueError("put() only accepts already-scored trajectories")
        L = len(traj.token_ids)
        with self._lock:
            while self._dq and (len(self._dq) >= self.capacity
                                or (self.capacity_tokens and self._tok + L > self.capacity_tokens)):
                old = self._dq.popleft()
                self._tok -= len(old.token_ids)
                self.n_dropped_overflow += 1
            self._dq.append(traj)
            self._tok += L
            self.n_put += 1

    def get_batch(self, n: int, cur_step: int, max_staleness: int) -> list[Trajectory]:
        """Pull up to n fresh-enough trajectories; drop any stale ones encountered (counted). FIFO."""
        out: list[Trajectory] = []
        with self._lock:
            while self._dq and len(out) < n:
                t = self._dq.popleft()
                self._tok -= len(t.token_ids)
                if is_stale(t.weight_version, cur_step, max_staleness):
                    self.n_dropped_stale += 1
                    continue
                out.append(t)
            self.n_served += len(out)
        return out

    def note_overflow_drop(self, n: int = 1) -> None:
        """Record a producer-side drop that happened before put(), e.g. failed long-window slicing."""
        with self._lock:
            self.n_dropped_overflow += n

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
