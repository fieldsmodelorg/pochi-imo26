# Copyright 2026 proof-pilot. Apache-2.0.
"""P0 integration test: data-plane (atom independence / backpressure / failover / handle round-trip / GC), all mock, no GPU needed.

run:  PYTHONPATH=src .venv/bin/python tests/test_p0_dataplane.py
"""
from __future__ import annotations

import asyncio
import functools
import os
import random
import socket
import sys
import tempfile
import time

import aiohttp
from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "servers"))

from opd_v2.buffer import TrajectoryBuffer
from opd_v2.config import OPDConfig
from opd_v2.data_plane.pools import Pool, Replica, build_pools
from opd_v2.data_plane.clients import RolloutClient, TeacherClient
from opd_v2.data_plane.produce import Prompt, produce_sample, fan_out
from opd_v2.data_plane.scheduler import Scheduler
from opd_v2.hidden_store import HiddenStore, read_hidden
import mock_rollout
import mock_teacher


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def start(app):
    runner = web.AppRunner(app)
    await runner.setup()
    port = free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner, f"http://127.0.0.1:{port}"


def fs_count(d: str) -> int:
    return len([f for f in os.listdir(d) if f.endswith(".bin")]) if os.path.isdir(d) else 0


def rand_prompts(seed=0, plen=32):
    rng = random.Random(seed)
    while True:
        yield Prompt(ids=[rng.randint(3, 129279) for _ in range(plen)])


async def main() -> int:
    tmp = tempfile.mkdtemp(prefix="opd_v2_p0_")
    fails = []

    def check(name, ok, extra=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
        if not ok:
            fails.append(name)

    # ---- servers ----
    r_runner, r_url = await start(mock_rollout.make_app(base_latency=0.0, max_new=64, seed=1))
    t_runner, t_url = await start(mock_teacher.make_app(base_latency=0.0, seed=2))
    # slow rollout server for the concurrency test
    rs_runner, rs_url = await start(mock_rollout.make_app(base_latency=0.20, max_new=32, seed=3))
    ts_runner, ts_url = await start(mock_teacher.make_app(base_latency=0.05, seed=4))
    # failing rollout replica (always 500)
    rf_runner, rf_url = await start(mock_rollout.make_app(fail_rate=1.0, seed=5))

    session = aiohttp.ClientSession()
    store = HiddenStore(os.path.join(tmp, "hidden"))
    step = {"v": 0}

    try:
        # ============ 1) single atom: handle round-trip ============
        print("\n[1] produce_sample atom + handle round-trip")
        cfg = OPDConfig(run_dir=tmp)
        cfg.rollout.urls = [r_url]; cfg.teacher.urls = [t_url]
        cfg.rollout.max_new_tokens = 64
        rp, tp = build_pools(session, cfg)
        prompt = Prompt(ids=list(range(50)))
        traj = await produce_sample(prompt, rollout_pool=rp, teacher_pool=tp, store=store,
                                    cfg=cfg, default_wv=lambda: step["v"])
        ok = traj is not None and os.path.exists(traj.handle.path)
        check("atom returns ScoredTrajectory + file on FS", ok)
        if ok:
            pb, sb, tb, seq, hid = read_hidden(traj.handle.path)
            n_t = len(traj.ids) - traj.prompt_len
            check("handle.seq_len == n_t+1 (G4 alignment)", seq == n_t + 1, f"seq={seq} n_t={n_t}")
            check("file seq matches handle", seq == traj.handle.seq_len)
            check("teacher bytes NOT in trajectory (P7: handle only)",
                  not hasattr(traj, "teacher_packed"))
            store.delete(traj.handle.path)

        # ============ 2) fan_out N atoms run CONCURRENTLY (not slowest-gated) ============
        print("\n[2] fan_out concurrency (atoms independent, wall≈max not sum)")
        cfg2 = OPDConfig(run_dir=tmp)
        cfg2.rollout.urls = [rs_url]; cfg2.teacher.urls = [ts_url]
        cfg2.rollout.max_new_tokens = 32
        cfg2.rollout.max_inflight_per_replica = 32
        cfg2.teacher.max_inflight_per_replica = 32
        rp2, tp2 = build_pools(session, cfg2)
        N = 16
        atoms = [produce_sample(p, rollout_pool=rp2, teacher_pool=tp2, store=store, cfg=cfg2,
                                default_wv=lambda: step["v"]) for p in fan_out(Prompt(ids=list(range(40))), N)]
        t0 = time.perf_counter()
        res = await asyncio.gather(*atoms)
        dt = time.perf_counter() - t0
        got = [r for r in res if r is not None]
        # each atom: rollout 0.20s + teacher 0.05s = 0.25s serial would be 16*0.25=4.0s
        check(f"all {N} atoms produced", len(got) == N, f"got={len(got)}")
        check("ran concurrently (wall<1.0s vs serial 4.0s)", dt < 1.0, f"wall={dt:.2f}s")
        store.delete_handles([r.handle for r in got])

        # ============ 3) scheduler: keep-N-in-flight + backpressure + overflow GC ============
        print("\n[3] scheduler keep-N-in-flight + backpressure + overflow GC")
        cfg3 = OPDConfig(run_dir=tmp)
        cfg3.rollout.urls = [r_url]; cfg3.teacher.urls = [t_url]
        cfg3.rollout.max_new_tokens = 64
        cfg3.rollout.max_inflight_per_replica = 64
        cfg3.teacher.max_inflight_per_replica = 64
        rp3, tp3 = build_pools(session, cfg3)
        buf = TrajectoryBuffer(capacity=20, capacity_tokens=10_000_000)
        store3 = HiddenStore(os.path.join(tmp, "hidden3"))
        pfn = functools.partial(produce_sample, rollout_pool=rp3, teacher_pool=tp3, store=store3,
                                cfg=cfg3, default_wv=lambda: step["v"])
        sched = Scheduler(prompts=rand_prompts(seed=7), produce=pfn, buffer=buf, store=store3,
                          target_inflight=16, n_samples=4, near_full_frac=0.9)
        stop = asyncio.Event()
        runner_task = asyncio.create_task(sched.run(stop))
        await asyncio.sleep(1.2)
        # mid-run invariant: buffer never exceeds capacity
        mid_size = len(buf)
        stop.set()
        await runner_task
        files = fs_count(store3.dir)
        check("buffer bounded (<= capacity)", len(buf) <= buf.capacity, f"size={len(buf)} cap={buf.capacity}")
        check("buffer churned (produced > capacity => keep-N-in-flight worked)",
              sched.n_produced > buf.capacity, f"produced={sched.n_produced}")
        check("overflow GC: FS files ≈ buffer size (no orphan leak)",
              files <= buf.capacity + 4, f"files={files} bufsize={len(buf)}")
        check("mid-run buffer was bounded too", mid_size <= buf.capacity, f"mid={mid_size}")
        # drain buffer + GC -> 0 files
        kept, stale = buf.get_batch(10_000, cur_step=0, max_staleness=10_000)
        store3.delete_handles([t.handle for t in kept])
        check("explicit GC clears FS", fs_count(store3.dir) == 0, f"left={fs_count(store3.dir)}")

        # ============ 4) failover: dead replica sidelined, production continues ============
        print("\n[4] failover (one rollout replica always-500 -> sidelined)")
        cfg4 = OPDConfig(run_dir=tmp)
        cfg4.rollout.urls = [rf_url, r_url]   # failing first, healthy second
        cfg4.teacher.urls = [t_url]
        cfg4.rollout.max_new_tokens = 64
        cfg4.data_plane.dead_until_seconds = 30.0
        rp4, tp4 = build_pools(session, cfg4)
        store4 = HiddenStore(os.path.join(tmp, "hidden4"))
        produced = 0
        for i in range(30):
            tr = await produce_sample(Prompt(ids=list(range(30))), rollout_pool=rp4, teacher_pool=tp4,
                                      store=store4, cfg=cfg4, default_wv=lambda: step["v"])
            if tr:
                produced += 1
                store4.delete(tr.handle.path)
        rstats = rp4.stats()
        check("failover: most atoms still produced via healthy replica", produced >= 25,
              f"produced={produced}/30")
        check("failover: failing replica recorded errors", rstats["errors"] >= 1, str(rstats))

        # ============ 5) staleness drop returns handles for GC ============
        print("\n[5] staleness drop -> handles returned for GC")
        buf5 = TrajectoryBuffer(capacity=100)
        store5 = HiddenStore(os.path.join(tmp, "hidden5"))
        for i in range(6):   # 6 real trajs with wv 0..5 (override server wv to control staleness)
            tr = await produce_sample(Prompt(ids=list(range(20))), rollout_pool=rp3, teacher_pool=tp3,
                                      store=store5, cfg=cfg3, default_wv=lambda: 0)
            assert tr is not None
            tr.wv = i
            buf5.put(tr)
        kept, stale = buf5.get_batch(100, cur_step=20, max_staleness=16)   # wv<=3 stale (20-wv>16)
        store5.delete_handles([t.handle for t in stale])
        check("staleness: stale handles identified", len(stale) == 4 and len(kept) == 2,
              f"kept={len(kept)} stale={len(stale)}")
        store5.delete_handles([t.handle for t in kept])
        check("staleness GC clears FS", fs_count(store5.dir) == 0, f"left={fs_count(store5.dir)}")

    finally:
        await session.close()
        for rr in (r_runner, t_runner, rs_runner, ts_runner, rf_runner):
            await rr.cleanup()

    print("\n" + ("=" * 50))
    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("ALL P0 DATA-PLANE TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
