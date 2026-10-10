# Copyright 2026 proof-pilot. Apache-2.0.
"""keep-N-in-flight scheduler + backpressure (PLAN §5.3).

Keeps ~`target_inflight` atoms in flight; as each finishes it is collected into the buffer; when the buffer
is `near_full` it stops issuing new atoms (backpressure). Each prompt fans out into `n_samples` independent
atoms (V1). Pure asyncio worker-pool, no blocking threads (fixes v1 P5's 40~64 OS threads).

GC hangs off here: when the buffer is full and evicts an old traj, the scheduler unlinks its hidden file
immediately (overflow GC); stale-drop / post-batch-consumption GC happens on the orchestrator side (V14).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Iterator

from opd_v2.buffer import ScoredTrajectory, TrajectoryBuffer
from opd_v2.data_plane.produce import ProduceResult, Prompt
from opd_v2.hidden_store import HiddenStore

log = logging.getLogger("opd_v2.scheduler")


class Scheduler:
    def __init__(self, *, prompts: Iterator[Prompt],
                 produce: Callable[[Prompt], Awaitable[ProduceResult | None]],
                 buffer: TrajectoryBuffer, store: HiddenStore,
                 target_inflight: int, n_samples: int, near_full_frac: float = 0.9):
        self.prompts = prompts
        self.produce = produce
        self.buf = buffer
        self.store = store
        self.target = max(1, target_inflight)
        self.n = max(1, n_samples)
        self.frac = near_full_frac
        self._pending: list[Prompt] = []        # atoms awaiting dispatch after fan-out
        self._exhausted = False
        self.n_produced = 0
        self.n_failed = 0
        self._gen_sum = 0          # cumulative rollout generation length (degeneration/length-collapse monitor)
        self._gen_max = 0
        self._gen_n = 0
        self._fr = {"stop": 0, "length": 0, "other": 0}   # cumulative finish_reason (EOS/length-stop ratio)
        self._n_admit_drop: dict[str, int] = {}           # admission-filter deliberate drops (by reason; ≠ fail)

    def _refill_pending(self) -> bool:
        """Pull one problem from the prompt source, fan it out into n atoms into pending. Returns whether prompts remain."""
        if self._exhausted:
            return False
        try:
            p = next(self.prompts)
        except StopIteration:
            self._exhausted = True
            return False
        self._pending.extend([p] * self.n)
        return True

    async def run(self, stop: asyncio.Event) -> None:
        """Run until stop is set (orchestrator teardown) or the prompt source is exhausted and in-flight drains."""
        inflight: set[asyncio.Task] = set()
        try:
            while not stop.is_set():
                # within capacity and not under backpressure, top up in-flight as much as possible
                while len(inflight) < self.target and not self.buf.near_full(self.frac):
                    if not self._pending and not self._refill_pending():
                        break
                    p = self._pending.pop()
                    inflight.add(asyncio.create_task(self.produce(p)))

                if not inflight:
                    if self._exhausted and not self._pending:
                        break                      # exhausted: nothing left to run
                    await asyncio.sleep(0.05)       # under backpressure / no prompts for now: yield
                    continue

                done, inflight = await asyncio.wait(
                    inflight, return_when=asyncio.FIRST_COMPLETED, timeout=1.0)
                for t in done:
                    try:
                        res = t.result()
                    except Exception:
                        res = None
                    if res is None:                        # exception / early return (prompt over window, generation errored)
                        self.n_failed += 1
                        continue
                    # record finish_reason for **all** completed generations (including drop/fail -> eos/length reflects the generation side and isn't distorted by the filter)
                    fr = res.finish_reason
                    if fr:
                        self._fr[fr if fr in self._fr else "other"] += 1
                    if res.drop_reason:                    # admission deliberate drop (≠ fail; dropped before teacher, no hidden to GC)
                        self._n_admit_drop[res.drop_reason] = self._n_admit_drop.get(res.drop_reason, 0) + 1
                        continue
                    traj = res.traj
                    if traj is None:                       # generation/teacher failed
                        self.n_failed += 1
                        continue
                    evicted = self.buf.put(traj)           # if full, evicts the oldest
                    if evicted:
                        self.store.delete_handles([e.handle for e in evicted])  # overflow GC (V14)
                    self.n_produced += 1
                    g = traj.gen_len
                    self._gen_sum += g
                    self._gen_max = max(self._gen_max, g)
                    self._gen_n += 1
        finally:
            for t in inflight:
                t.cancel()
            res = await asyncio.gather(*inflight, return_exceptions=True)
            # cancelled but actually already produced (race): GC their files to avoid orphans
            for r in res:
                if isinstance(r, ProduceResult) and r.traj is not None:
                    self.store.delete(r.traj.handle.path)

    def stats(self) -> dict:
        return {"produced": self.n_produced, "failed": self.n_failed,
                "pending": len(self._pending), "exhausted": self._exhausted,
                "gen_len_mean": (self._gen_sum / self._gen_n) if self._gen_n else 0.0,
                "gen_len_max": self._gen_max,
                "fr_stop": self._fr["stop"], "fr_length": self._fr["length"],
                "fr_other": self._fr["other"],
                # admission filter: deliberate drops (by reason) + total (≠ failed). drop rate = dropped/(produced+dropped).
                "admit_dropped": dict(self._n_admit_drop),
                "admit_dropped_total": sum(self._n_admit_drop.values())}
