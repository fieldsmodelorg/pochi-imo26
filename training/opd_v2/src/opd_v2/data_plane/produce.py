# Copyright 2026 proof-pilot. Apache-2.0.
"""produce_sample atom — "one rollout + one teacher score" = one independent async coroutine (V3).

This is the core of the "elegant design" the user wanted (PLAN §5):
- **Fully independent**: does not depend on other tasks; the only shared state = the two pools' semaphores
  + the output buffer.
- **As soon as one trajectory finishes, send it straight to the teacher** (don't wait for the other samples
  of the same prompt) -> eliminates v1's "gate the whole group on the slowest trajectory" (P3/P4).
- **Finish early, enter the buffer early** -> lowers staleness.
- **Return a handle, not bytes** (teacher writes to FS server-side) -> buffer/scatter is ultra-light (P7 fix).

N samples = N independent atoms (`fan_out`, V1). Any failed step -> return None (scheduler counts fail, cleans the half-written file).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from opd_v2.buffer import ScoredTrajectory
from opd_v2.config import OPDConfig
from opd_v2.data_plane.pools import Pool
from opd_v2.hidden_store import HiddenStore


@dataclass
class Prompt:
    ids: list[int]               # already-rendered+tokenized prompt token ids
    meta: dict = field(default_factory=dict)


@dataclass
class ProduceResult:
    """The result of produce_sample (V33). Makes the three states "success / admission-drop / fail" explicit,
    so the scheduler can both record finish_reason for **all** generations (EOS/length generation-side ratio
    monitoring stays intact) and count "deliberately dropped" separately from "errored".
    - traj: successfully scored and admitted -> ScoredTrajectory; otherwise None.
    - finish_reason: always returned (even on drop/fail), for generation-side eos/length ratio monitoring.
    - drop_reason: the label of an admission-filter deliberate drop (e.g. "length"); "" = not dropped.
      drop ≠ fail: drop is "valid generation, deliberately not trained on", fail is "generation/score errored".
    """
    traj: ScoredTrajectory | None = None
    finish_reason: str = ""
    drop_reason: str = ""


def _admission_drop(finish_reason: str, drop_finish_reasons) -> str:
    """training-buffer admission policy (pure function, easy to unit-test and extend). Returns the drop label; "" = accept.

    Current rule: drop if finish_reason ∈ drop_finish_reasons (default {"length"}: window truncation = the
    main source of OPD self-amplification). Does not touch the rollout sampling distribution — only decides
    which on-policy samples enter the gradient (a rejection filter). Extension point: to add "token-level loop
    detection" later, take extra ids/prompt_len params here and return "loop" on a hit; the upstream
    produce/scheduler/wandb counting pipeline needs no changes (drop_reason is a free-form string).
    """
    if finish_reason in drop_finish_reasons:
        return finish_reason
    return ""


async def produce_sample(prompt: Prompt, *, rollout_pool: Pool, teacher_pool: Pool,
                         store: HiddenStore, cfg: OPDConfig,
                         default_wv: Callable[[], int], dump=None, pool_ingest=None
                         ) -> ProduceResult:
    """One rollout + one teacher score. Returns a ProduceResult (three states: success traj / admission-drop / fail).

    `dump` (RolloutDumpWriter | None): if given, as soon as the rollout is generated (before teacher score)
    the ids are dumped as a side channel -> even rollouts whose teacher failed / that were later evicted from
    the buffer / GC'd are captured (rollout_store.py).
    `pool_ingest` (agentic.PoolIngestor | None): only given in agentic mode; as soon as the rollout is
    generated (before teacher) it is written back to the artifact pool (parse answer-only, validity-gate) ->
    context for downstream roles. Independent of teacher success/failure (same location as dump); non-blocking enqueue.
    """
    rc = cfg.rollout
    plen = len(prompt.ids)
    # gen budget = the entire remaining window (max_traj_tokens - prompt), further clamped by cfg.max_new_tokens.
    # This lets a proof use the full window to finish, without the request exceeding context (prompt+gen ≤ max_traj_tokens).
    budget = cfg.data_plane.max_traj_tokens - plen
    if budget <= 0:                            # the prompt itself already exceeds the window -> nothing to generate
        return ProduceResult()
    max_new = min(rc.max_new_tokens, budget)
    # 1) one rollout (one request, no n, read output_ids directly)
    try:
        async with rollout_pool.slot() as client:
            gen_ids, wv, finish_reason = await client.generate_one(
                prompt.ids, temperature=rc.temperature, top_p=rc.top_p, top_k=rc.top_k,
                max_new_tokens=max_new, ignore_eos=rc.ignore_eos, timeout=rc.gen_timeout_s)
    except Exception:
        return ProduceResult()                 # generation errored (finish_reason unknown) -> fail
    if not gen_ids:
        return ProduceResult(finish_reason=finish_reason)

    full = prompt.ids + list(gen_ids)
    if len(full) > cfg.data_plane.max_traj_tokens:
        full = full[: cfg.data_plane.max_traj_tokens]
    if len(full) <= plen:                      # no generated token left after truncation -> nothing to learn
        return ProduceResult(finish_reason=finish_reason)
    wv = wv if wv is not None else int(default_wv())

    # 1b) dump as soon as the rollout is generated (before teacher score) -> stores "all" rollouts (including dropped ones), decoupled from teacher/buffer/GC
    if dump is not None:
        dump.append(full, plen, wv, {**prompt.meta, "finish_reason": finish_reason})

    # 1c) admission filter: window-truncated etc. don't enter training (before teacher -> saves a hidden disk write; doesn't touch the sampling distribution).
    #     Still returns finish_reason for generation-side ratio monitoring; drop_reason lets the scheduler and wandb count independently (≠ fail).
    drop = _admission_drop(finish_reason, rc.drop_finish_reasons)
    if drop:
        return ProduceResult(finish_reason=finish_reason, drop_reason=drop)

    # 1d) agentic: write this trajectory back to the artifact pool (before teacher; parse answer-only + validity-gate; non-blocking)
    if pool_ingest is not None:
        pool_ingest.append(full, plen, wv, prompt.meta, finish_reason)

    # 2) once done, send it straight to the teacher (don't wait for the other samples of the same prompt); teacher writes to FS server-side, returns a handle
    out_path = store.new_path()
    try:
        async with teacher_pool.slot() as client:
            handle = await client.score(full, start=plen - 1, out_path=out_path, wv=wv)
    except Exception:
        store.delete(out_path)                 # teacher may have half-written -> clean it up
        return ProduceResult(finish_reason=finish_reason)

    return ProduceResult(
        traj=ScoredTrajectory(ids=full, prompt_len=plen, wv=wv, handle=handle,
                              meta=dict(prompt.meta), finish_reason=finish_reason or ""),
        finish_reason=finish_reason or "")


def fan_out(prompt: Prompt, n: int) -> list[Prompt]:
    """N samples = the input for N independent atoms (same problem, each independent, each finishes and enters the buffer on its own, V1)."""
    return [prompt] * n
