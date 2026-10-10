# Copyright 2026 proof-pilot. Apache-2.0.
"""Mock rollout sglang server (P0: exercises the data-plane without GPU/real sglang).

Mimics the real rollout server's HTTP contract:
- POST /generate  {input_ids, sampling_params, stream} -> {output_ids:[...], meta_info:{weight_version}}
  output_ids length is random (simulates variable-length rollouts, so atoms finish out of order); a server-side sleep simulates decode latency.
- GET  /health -> 200
- POST /update_weights_from_disk {model_path, weight_version, flush_cache} -> bump weight_version
- POST /pause_generation / /continue_generation -> {}

Uses aiohttp.web; can run standalone (`python mock_rollout.py --port 8200`) or be started in-process by a test.
"""
from __future__ import annotations

import argparse
import asyncio
import random

from aiohttp import web


def make_app(*, base_latency: float = 0.0, jitter: float = 0.0, min_new: int = 5,
             max_new: int = 64, fail_rate: float = 0.0, seed: int = 0) -> web.Application:
    state = {"wv": 0, "n_gen": 0}
    rng = random.Random(seed)
    app = web.Application()

    async def health(_req):
        return web.Response(status=200, text="ok")

    async def generate(req):
        body = await req.json()
        sp = body.get("sampling_params", {})
        if fail_rate and rng.random() < fail_rate:
            return web.Response(status=500, text="mock injected failure")
        cap = int(sp.get("max_new_tokens", max_new))
        eff = max(min_new, min(max_new, cap))      # the mock's own effective generation cap
        n = rng.randint(min_new, eff)
        if base_latency:
            await asyncio.sleep(base_latency + (rng.random() * jitter if jitter else 0.0))
        out = [rng.randint(3, 129279) for _ in range(n)]
        state["n_gen"] += 1
        fr = "length" if n >= eff else "stop"      # hit the cap=length-stop, otherwise EOS-stop (simulates the truncation monitor)
        return web.json_response({"output_ids": out,
                                  "meta_info": {"weight_version": str(state["wv"]),
                                                "finish_reason": {"type": fr}}})

    async def update_weights(req):
        body = await req.json()
        wv = body.get("weight_version")
        if wv not in (None, ""):
            state["wv"] = int(wv)
        return web.json_response({"success": True, "weight_version": str(state["wv"])})

    async def loads(_req):
        return web.json_response({"loads": []})

    async def noop(_req):
        return web.json_response({})

    app.router.add_get("/health", health)
    app.router.add_post("/generate", generate)
    app.router.add_post("/update_weights_from_disk", update_weights)
    app.router.add_get("/v1/loads", loads)
    app.router.add_post("/pause_generation", noop)
    app.router.add_post("/continue_generation", noop)
    app["state"] = state
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--latency", type=float, default=0.05)
    ap.add_argument("--jitter", type=float, default=0.05)
    a = ap.parse_args()
    web.run_app(make_app(base_latency=a.latency, jitter=a.jitter), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
