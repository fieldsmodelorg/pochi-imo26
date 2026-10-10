# Copyright 2026 proof-pilot. Apache-2.0.
"""load-aware concurrency pools (fixes v1 P9: teacher's static by-wid round-robin -> load-aware).

One `Pool` per service (rollout / teacher), each holding several replicas. `slot()` is an async context manager:
1. `await` the global semaphore (limits total in-flight);
2. pick the **emptiest live replica** (smallest in-flight count; a replica with consecutive errors is skipped during its `dead_until`);
3. yield that replica's client;
4. on exit, in-flight -1 and release the semaphore; if an exception is raised mid-way -> mark that replica dead (briefly sidelined, self-healing).

Uses "in-flight count" rather than `/v1/loads` as the load signal: always available, zero extra round-trip,
and pick+inc is atomic under the single event loop. (`/v1/loads` is kept for a future upgrade.) The two pools
have independent semaphores -> rollout (TP1, slow) / teacher (TP4, fast) capacities are tuned separately
(PLAN §5.3/§5.4).
"""
from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field

import aiohttp

from opd_v2.config import OPDConfig
from opd_v2.data_plane.clients import RolloutClient, TeacherClient

log = logging.getLogger("opd_v2.pools")


@dataclass
class Replica:
    base_url: str
    client: object                 # RolloutClient | TeacherClient
    cap: int = 8                   # per-replica soft cap (in-flight)
    in_flight: int = 0
    dead_until: float = 0.0
    n_errors: int = 0
    n_done: int = 0

    def live(self, now: float) -> bool:
        return now >= self.dead_until


class Pool:
    def __init__(self, replicas: list[Replica], concurrency: int, dead_seconds: float = 10.0,
                 name: str = "pool"):
        if not replicas:
            raise ValueError(f"{name}: no replicas")
        self.replicas = replicas
        self.name = name
        self.dead_seconds = dead_seconds
        self._sem = __import__("asyncio").Semaphore(max(1, concurrency))
        self.concurrency = max(1, concurrency)

    def _pick(self) -> Replica:
        now = time.monotonic()
        live = [r for r in self.replicas if r.live(now)]
        if live:
            # emptiest (smallest in_flight); ties go to the first
            return min(live, key=lambda r: r.in_flight)
        # all dead: optimistically retry the one that revives soonest (avoid a full stall)
        return min(self.replicas, key=lambda r: r.dead_until)

    @contextlib.asynccontextmanager
    async def slot(self):
        """`async with pool.slot() as client:` — rate limiting + load-aware pick + failure sideline."""
        await self._sem.acquire()
        r = self._pick()
        r.in_flight += 1
        try:
            yield r.client
            r.n_done += 1
        except Exception as e:
            r.n_errors += 1
            r.dead_until = time.monotonic() + self.dead_seconds
            log.warning("%s slot err on %s: %r", self.name, r.base_url, e)   # V33: record the exception type to help diagnose the error source
            raise
        finally:
            r.in_flight -= 1
            self._sem.release()

    def stats(self) -> dict:
        now = time.monotonic()
        return {
            "concurrency": self.concurrency,
            "live": sum(1 for r in self.replicas if r.live(now)),
            "replicas": len(self.replicas),
            "in_flight": sum(r.in_flight for r in self.replicas),
            "errors": sum(r.n_errors for r in self.replicas),
            "done": sum(r.n_done for r in self.replicas),
        }


def build_pools(session: aiohttp.ClientSession, cfg: OPDConfig) -> tuple[Pool, Pool]:
    """Build the rollout/teacher pools from config (sharing one aiohttp session)."""
    r_caps = cfg.rollout.max_inflight_per_replica
    rollout_reps = [Replica(u, RolloutClient(session, u), cap=r_caps) for u in cfg.rollout.urls]
    r_conc = cfg.data_plane.rollout_concurrency or sum(r.cap for r in rollout_reps)
    rollout_pool = Pool(rollout_reps, r_conc, cfg.data_plane.dead_until_seconds, name="rollout")

    t_caps = cfg.teacher.max_inflight_per_replica
    teacher_reps = [Replica(u, TeacherClient(session, u), cap=t_caps) for u in cfg.teacher.urls]
    t_conc = cfg.data_plane.teacher_concurrency or sum(r.cap for r in teacher_reps)
    teacher_pool = Pool(teacher_reps, t_conc, cfg.data_plane.dead_until_seconds, name="teacher")
    return rollout_pool, teacher_pool
