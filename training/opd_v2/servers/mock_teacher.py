# Copyright 2026 proof-pilot. Apache-2.0.
"""Mock teacher sglang server (P0: exercises the data-plane's teacher->FS->handle path without GPU/real sglang).

Mimics the v2 teacher /score contract (**write to FS server-side, return handle metadata**, V12):
- POST /score {input_ids, start, out_path, return_top1} -> write random bytes of had+int6 size to out_path,
  return {seq_len, packed_bytes, scales_bytes, top1_bytes}. seq_len = len(input_ids) - start.
- GET /health -> 200

Key: **bytes do not go back to the orchestrator via the HTTP body**, only JSON metadata (which is exactly what the P7 fix must verify).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random

from aiohttp import web

from opd_v2.config import HID_DIM
from opd_v2.hidden_store import packed_row_bytes, scale_row_bytes, write_hidden


def make_app(*, base_latency: float = 0.0, jitter: float = 0.0, hid: int = HID_DIM,
             fail_rate: float = 0.0, seed: int = 0) -> web.Application:
    state = {"n_score": 0}
    rng = random.Random(seed)
    app = web.Application()

    async def health(_req):
        return web.Response(status=200, text="ok")

    async def score(req):
        body = await req.json()
        ids = body["input_ids"]
        start = int(body.get("start", 0))
        out_path = body["out_path"]
        return_top1 = bool(body.get("return_top1", False))
        if fail_rate and rng.random() < fail_rate:
            return web.Response(status=500, text="mock injected failure")
        seq_len = len(ids) - start
        if seq_len <= 0:
            return web.Response(status=400, text=f"bad start={start} for len={len(ids)}")
        if base_latency:
            await asyncio.sleep(base_latency + (rng.random() * jitter if jitter else 0.0))
        packed = os.urandom(seq_len * packed_row_bytes(hid))
        scales = os.urandom(seq_len * scale_row_bytes(hid))
        top1 = os.urandom(seq_len * 4) if return_top1 else b""
        # server-side write to shared FS (atomic)
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: write_hidden(out_path, packed, scales, seq_len, top1=top1, hid=hid))
        state["n_score"] += 1
        return web.json_response({"seq_len": seq_len, "packed_bytes": len(packed),
                                  "scales_bytes": len(scales), "top1_bytes": len(top1)})

    app.router.add_get("/health", health)
    app.router.add_post("/score", score)
    app["state"] = state
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--latency", type=float, default=0.03)
    a = ap.parse_args()
    web.run_app(make_app(base_latency=a.latency), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
