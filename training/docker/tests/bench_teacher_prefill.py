#!/usr/bin/env python3
"""Teacher `/score` PREFILL-throughput benchmark for OPD.

The OPD teacher is **prefill-only**: `/score` runs one forward over the full trajectory and writes
the post-norm hidden to FS — it never decodes. So decode tok/s and sglang's per-chunk
`input throughput` (idle-polluted: tokens / time-since-last-activity) are NOT the metric that
matters. This measures **aggregate prefill tok/s under concurrency**, by wall-clock, across
sequence lengths — the real teacher-side throughput gate for OPD.

Run on the TEACHER node in the serve venv (has aiohttp):
  /opt/venv/serve/bin/python bench_teacher_prefill.py --url http://localhost:8100 \
      --seq-lens 8192,16384,32768 --concurrency 1,8,16 --requests 32

Notes:
- A warmup pass is discarded (clears idle-pollution + any first-shape JIT).
- `--concurrency` should bracket the server's `--max-running-requests` (TEACHER_MAXRUN): peak
  prefill throughput is near it; beyond it requests just queue.
- The teacher chunks prefill at `--chunked-prefill-size` (default 11264), so seq_lens around and
  above that are compute-bound and representative of production trajectories.
- Hidden files are written to a tmpfs out-dir and removed; nothing touches the shared FS.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import time

import aiohttp


async def _one_score(session, url, seq_len, vocab, out_dir, idx):
    ids = [random.randint(1, vocab - 1) for _ in range(seq_len)]
    out_path = os.path.join(out_dir, f"bench_{idx}.bin")
    payload = {"input_ids": ids, "start": 0, "out_path": out_path, "return_top1": False}
    t0 = time.monotonic()
    async with session.post(f"{url}/score", json=payload,
                            timeout=aiohttp.ClientTimeout(total=1800)) as r:
        if r.status != 200:
            raise RuntimeError(f"/score -> {r.status}: {(await r.text())[:200]}")
        meta = await r.json()
    dt = time.monotonic() - t0
    try:
        os.remove(out_path)
    except OSError:
        pass
    return int(meta.get("seq_len", seq_len)), dt


async def _run(url, seq_len, concurrency, requests, vocab, out_dir):
    sem = asyncio.Semaphore(concurrency)

    async def worker(i):
        async with sem:
            return await _one_score(session, url, seq_len, vocab, out_dir, i)

    async with aiohttp.ClientSession() as session:
        t0 = time.monotonic()
        results = await asyncio.gather(*[worker(i) for i in range(requests)], return_exceptions=True)
        wall = time.monotonic() - t0

    ok = [r for r in results if not isinstance(r, Exception)]
    errs = [r for r in results if isinstance(r, Exception)]
    if errs:
        print(f"  !! {len(errs)}/{requests} requests failed, e.g. {errs[0]!r}", flush=True)
    total_tokens = sum(sl for sl, _ in ok)
    lats = sorted(dt for _, dt in ok) or [0.0]
    return {
        "seq_len": seq_len, "concurrency": concurrency, "ok": len(ok),
        "wall_s": wall, "total_tokens": total_tokens,
        "prefill_tok_s": (total_tokens / wall) if wall > 0 else 0.0,
        "p50": lats[len(lats) // 2], "p95": lats[min(len(lats) - 1, int(len(lats) * 0.95))],
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8100")
    ap.add_argument("--seq-lens", default="8192,16384,32768")
    ap.add_argument("--concurrency", default="1,8,16")
    ap.add_argument("--requests", type=int, default=32, help="requests per (seq_len, concurrency) point")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=100000, help="random token id upper bound (< model vocab)")
    ap.add_argument("--out-dir", default="/dev/shm/opd-bench")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    concs = [int(x) for x in args.concurrency.split(",")]

    print(f"warmup: {args.warmup} req @ seq_len={seq_lens[0]} (discarded) ...", flush=True)
    await _run(args.url, seq_lens[0], min(args.warmup, 4), args.warmup, args.vocab, args.out_dir)

    hdr = f"{'seq_len':>8} {'conc':>5} {'ok':>4} {'wall_s':>8} {'PREFILL tok/s':>14} {'p50_s':>8} {'p95_s':>8}"
    print("\n" + hdr + "\n" + "-" * len(hdr), flush=True)
    for sl in seq_lens:
        for c in concs:
            r = await _run(args.url, sl, c, args.requests, args.vocab, args.out_dir)
            print(f"{r['seq_len']:>8} {r['concurrency']:>5} {r['ok']:>4} {r['wall_s']:>8.1f} "
                  f"{r['prefill_tok_s']:>14.0f} {r['p50']:>8.2f} {r['p95']:>8.2f}", flush=True)
    print("\nPREFILL tok/s = total input tokens / wall time (the OPD teacher-side throughput gate).")


if __name__ == "__main__":
    asyncio.run(main())
