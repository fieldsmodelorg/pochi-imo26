# Copyright 2026 proof-pilot. Apache-2.0.
"""async (aiohttp) clients — rollout and teacher sglang servers.

Design principles (PLAN §5.5):
- **rollout**: `/generate` is one request = one rollout (**no `n`**, V1); **reads `output_ids` directly**
  (V2, not the v1 return_logprob/logprob_start_len/skip-tokenizer-init hack); `wv` is taken from the
  server-reported `meta_info.weight_version` (V6).
- **teacher**: `/score` with an `out_path` -> the teacher **writes to shared FS server-side** and returns
  handle metadata JSON (**not bytes**, V12). The client assembles a `HiddenHandle` (path created by the
  orchestrator, stamped with the rollout's wv).

Fully async, sharing one `aiohttp.ClientSession` (created by the orchestrator, passed into pool/clients).
Each method has its own timeout (a 32k rollout generation can exceed 20min; teacher score is faster). A
non-2xx response raises directly -> handled by the pool/produce side.
"""
from __future__ import annotations

import aiohttp

from opd_v2.hidden_store import HiddenHandle


class RolloutError(RuntimeError):
    pass


class TeacherError(RuntimeError):
    pass


class RolloutClient:
    """token-in-token-out student rollout (fp8 flash_rl deployment; the client does not send load_format, V30/§5.6)."""

    def __init__(self, session: aiohttp.ClientSession, base_url: str):
        self.s = session
        self.base = base_url.rstrip("/")

    async def health(self, timeout: float = 5.0) -> bool:
        try:
            async with self.s.get(f"{self.base}/health",
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return r.status == 200
        except Exception:
            return False

    async def generate_one(self, input_ids: list[int], *, temperature: float, top_p: float,
                           top_k: int, max_new_tokens: int, ignore_eos: bool,
                           timeout: float = 3600.0) -> tuple[list[int], int | None, str | None]:
        """One rollout. Returns (output_ids, weight_version|None, finish_reason|None).

        finish_reason = sglang `meta_info.finish_reason.type` ("stop"=EOS/stop-token, "length"=hit
        max_new_tokens/context, "abort", etc.) -> used to compute the EOS-stop / length-stop ratio (truncation monitor).
        """
        sp = {
            "temperature": temperature, "top_p": top_p,
            "max_new_tokens": max_new_tokens, "ignore_eos": ignore_eos,
        }
        if top_k and top_k > 0:
            sp["top_k"] = top_k
        payload = {"input_ids": input_ids, "sampling_params": sp, "stream": False}
        async with self.s.post(f"{self.base}/generate", json=payload,
                               timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                raise RolloutError(f"/generate -> {r.status}: {(await r.text())[:200]}")
            data = await r.json()
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            raise RolloutError(f"unexpected /generate response type {type(data)}")
        meta = data.get("meta_info") or {}
        out = data.get("output_ids")
        if out is None:                       # some sglang versions put output_ids in meta_info
            out = meta.get("output_ids")
        if out is None:
            raise RolloutError(f"/generate response has no output_ids (keys={list(data)})")
        # sglang reports "default" until first update_weights_from_disk(weight_version=...) ->
        # anything non-numeric becomes None, and the produce side falls back to the orchestrator's current weight_version.
        wv = None
        wv_raw = meta.get("weight_version")
        if wv_raw not in (None, ""):
            try:
                wv = int(wv_raw)
            except (ValueError, TypeError):
                wv = None
        fr = meta.get("finish_reason")
        if isinstance(fr, dict):
            fr = fr.get("type")
        return list(out), wv, (fr if isinstance(fr, str) else None)

    # ---- weight sync (orchestrator-driven; parallel across all replicas, V22) ----
    async def pause_generation(self, mode: str = "in_place", timeout: float = 120.0) -> dict:
        async with self.s.post(f"{self.base}/pause_generation", json={"mode": mode},
                               timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            r.raise_for_status()
            return await r.json()

    async def continue_generation(self, timeout: float = 120.0) -> dict:
        # an empty body is required (sglang otherwise returns 422)
        async with self.s.post(f"{self.base}/continue_generation", json={},
                               timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            r.raise_for_status()
            return await r.json()

    async def update_weights_from_disk(self, path: str, weight_version: int,
                                       flush_cache: bool = False, timeout: float = 1800.0) -> dict:
        """flush_cache=False for in_place pause (required: under in_place a failed flush asserts and kills the scheduler).
        **Do not send load_format**: keeps the server's flash_rl fp8 loader (sending auto goes through DefaultLoader and blows up, §5.6)."""
        payload = {"model_path": path, "flush_cache": flush_cache,
                   "weight_version": str(weight_version)}
        async with self.s.post(f"{self.base}/update_weights_from_disk", json=payload,
                               timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            r.raise_for_status()
            return await r.json()


class TrainerHTTPClient:
    """orchestrator -> trainer-as-service (rank-0 HTTP ingress). Control plane, small data + handle (V23)."""

    def __init__(self, session: aiohttp.ClientSession, base_url: str):
        self.s = session
        self.base = base_url.rstrip("/")

    async def health(self, timeout: float = 5.0) -> dict | None:
        try:
            async with self.s.get(f"{self.base}/health",
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return await r.json() if r.status == 200 else None
        except Exception:
            return None

    async def train_step(self, trajs_wire: list[dict], want_g4: bool = False,
                         timeout: float = 7200.0) -> dict:
        payload = {"trajs": trajs_wire, "want_g4": want_g4}
        async with self.s.post(f"{self.base}/train_step", json=payload,
                               timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            r.raise_for_status()
            return await r.json()

    async def save(self, slot: str | None = None, timeout: float = 1800.0) -> dict:
        async with self.s.post(f"{self.base}/save", json={"slot": slot},
                               timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            r.raise_for_status()
            return await r.json()

    async def checkpoint(self, hf: bool = True, keep: int = -1, timeout: float = 3600.0) -> dict:
        """durable DCP+HF ckpt (model+optim+sched). Slower than /save (gather+HF), so the timeout is larger."""
        async with self.s.post(f"{self.base}/checkpoint", json={"hf": hf, "keep": keep},
                               timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            r.raise_for_status()
            return await r.json()

    async def stop(self, timeout: float = 120.0) -> dict:
        async with self.s.post(f"{self.base}/stop", json={},
                               timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            r.raise_for_status()
            return await r.json()


class TeacherClient:
    """DeepSeek-V4-Flash teacher scoring (/score writes shared FS, returns a handle, V12)."""

    def __init__(self, session: aiohttp.ClientSession, base_url: str):
        self.s = session
        self.base = base_url.rstrip("/")

    async def health(self, timeout: float = 5.0) -> bool:
        try:
            async with self.s.get(f"{self.base}/health",
                                  timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return r.status == 200
        except Exception:
            return False

    async def score(self, input_ids: list[int], *, start: int, out_path: str, wv: int,
                    return_top1: bool = False, timeout: float = 1200.0) -> HiddenHandle:
        """The teacher computes hidden over full_ids, **writes out_path server-side**, and returns handle metadata.

        Response JSON: `{seq_len, packed_bytes, scales_bytes, top1_bytes}` (the file is already on FS). The
        client assembles `HiddenHandle(path=out_path, seq_len, wv)` — the bytes never pass through the
        orchestrator (P7 fix).
        """
        payload = {"input_ids": input_ids, "start": start, "out_path": out_path,
                   "return_top1": return_top1}
        async with self.s.post(f"{self.base}/score", json=payload,
                               timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            if r.status != 200:
                raise TeacherError(f"/score -> {r.status}: {(await r.text())[:200]}")
            meta = await r.json()
        seq_len = int(meta["seq_len"])
        return HiddenHandle(path=out_path, seq_len=seq_len, wv=wv)
