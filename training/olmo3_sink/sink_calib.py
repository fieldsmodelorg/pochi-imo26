# Copyright 2026 proof-pilot. Apache-2.0.
"""Attention-sink calibration library: measure where an Olmo3(Sink) model dumps
its no-op attention mass and derive the per-head `s_aux` warm-start init.

Shared by the production builder (`build_init_model.py`) and the research CLIs
under `study/`. Method and empirical grounding: docs/attn_sink_study.md
(§5 drafted-token sinks, §6/§9 s_init derivation, §10 e2e builder).

  measure()         per-layer received-mass / sink-absorption stats (eager probs)
  build_sink_keys() the drafted "sink token" keyset per layer
  derive_logZ_D()   per-head logZ + dump mass D, sliding steady state, GQA-aware
  s_init_from()     s = logZ + logit(clip(D, p_floor, p_cap))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

TEXT = Path(__file__).resolve().parent.parent / "evaluation" / "data" / "deepseek-math-v2.txt"


def lm_loss(model, windows, device) -> float:
    """Mean causal-LM loss over full-window token tensors."""
    with torch.no_grad():
        return sum(model(input_ids=w[None].to(device), labels=w[None].to(device))
                   .loss.float().item() for w in windows) / len(windows)


def measure(model, ids: torch.Tensor, device: str, skip: int = 16, chunk: int = 1024):
    """Per layer: head sink-mass (in-window/evicted), P(pos0), received [H,S].

    Requires the eager backend (hooks read the per-row probs the sink fn returns;
    rowsum deficit = mass absorbed by s_aux)."""
    cfg = model.config
    window = cfg.sliding_window
    layer_types = cfg.layer_types
    S = ids.shape[0]
    stats: list[dict] = [dict() for _ in layer_types]

    def make_hook(li, layer_type):
        def hook(module, a, kw, output):
            pb = output[1][0]  # [H,S,S]; rowsum = 1 - sink mass (sink col dropped)
            H = pb.shape[0]
            dev = pb.device
            qi = torch.arange(S, device=dev)
            eff = window if layer_type == "sliding_attention" else S
            acc = {k: torch.zeros(H, device=dev) for k in ("sk_iw", "sk_ev", "p0_iw")}
            recv = torch.zeros(H, S, device=dev)
            n_iw = n_ev = 0
            for s0 in range(0, S, chunk):
                s1 = min(s0 + chunk, S)
                p = pb[:, s0:s1, :].float()
                rows = qi[s0:s1]
                sink = (1.0 - p.sum(-1)).clamp(min=0)
                iw = (rows >= skip) & (rows < eff)
                ev = rows >= eff
                if iw.any():
                    acc["sk_iw"] += sink[:, iw].sum(1)
                    acc["p0_iw"] += p[:, iw, 0].sum(1)
                    n_iw += int(iw.sum())
                if ev.any():
                    acc["sk_ev"] += sink[:, ev].sum(1)
                    n_ev += int(ev.sum())
                recv += p.sum(1)
            seen = torch.minimum(torch.tensor(S, device=dev) - qi,
                                 torch.tensor(eff, device=dev)).clamp(min=1)
            st = stats[li]
            st["sink_inwin"] = (acc["sk_iw"] / max(n_iw, 1)).cpu().numpy()
            st["sink_evict"] = (acc["sk_ev"] / max(n_ev, 1)).cpu().numpy() if n_ev else None
            st["p0_inwin"] = (acc["p0_iw"] / max(n_iw, 1)).cpu().numpy()
            st["received"] = (recv / seen).clamp(max=1.0).cpu().numpy()
        return hook

    hs = [l.self_attn.register_forward_hook(make_hook(i, layer_types[i]), with_kwargs=True)
          for i, l in enumerate(model.model.layers)]
    with torch.no_grad():
        model.model(input_ids=ids[None].to(device))
    for h in hs:
        h.remove()
    return stats


def build_sink_keys(received_per_layer: list, eps: float, tail: int = 256) -> dict[int, np.ndarray]:
    """received_per_layer: list of [H, S] arrays. Keys whose mean received mass
    exceeds eps for ANY head, excluding the recency-noisy tail."""
    S = received_per_layer[0].shape[1]
    return {li: np.where((r[:, : S - tail] > eps).any(0))[0]
            for li, r in enumerate(received_per_layer)}


def derive_logZ_D(model, ids, sink_keys: dict, qmin: int, chunk: int = 1024) -> list[dict]:
    """One forward pass; per layer/head: logZ (logsumexp of visible logits) and D
    (mass dumped on that layer's sink keyset), over query rows >= qmin.

    Captures post-RoPE q/k by wrapping the modeling module's apply_rotary_pos_emb;
    handles GQA (expands kv heads to query heads, repeat_kv order)."""
    from transformers.models.olmo3 import modeling_olmo3 as olmo3_mod

    cfg = model.config
    layer_types, window = cfg.layer_types, cfg.sliding_window
    H = cfg.num_attention_heads
    ids_t = torch.as_tensor(ids).view(1, -1)
    S = ids_t.shape[-1]
    dev = next(model.parameters()).device

    res: list[dict] = []
    orig = olmo3_mod.apply_rotary_pos_emb

    def wrapped(q, k, cos, sin, *a, **kw):
        qe, ke = orig(q, k, cos, sin, *a, **kw)
        li = len(res)
        with torch.no_grad():
            Q, K = qe[0].float(), ke[0].float()
            if K.shape[0] != Q.shape[0]:  # GQA (e.g. 32B 40Q/8KV): expand kv heads
                K = K.repeat_interleave(Q.shape[0] // K.shape[0], dim=0)
            scale = Q.shape[-1] ** -0.5
            eff = window if layer_types[li] == "sliding_attention" else S
            sk = torch.as_tensor(sink_keys[li], device=Q.device, dtype=torch.long)
            logZ_s = torch.zeros(H, device=Q.device)
            dump_s = torch.zeros(H, device=Q.device)
            n = 0
            for s0 in range(qmin, S, chunk):
                s1 = min(s0 + chunk, S)
                rows = torch.arange(s0, s1, device=Q.device)
                lg = torch.matmul(Q[:, s0:s1], K.transpose(-1, -2)) * scale
                keys = torch.arange(S, device=Q.device).view(1, -1)
                vis = (keys <= rows.view(-1, 1)) & (keys > rows.view(-1, 1) - eff)
                lg = lg.masked_fill(~vis, float("-inf"))
                logZ = lg.logsumexp(-1)
                logZ_s += logZ.sum(1)
                if len(sk):
                    lgs = lg[:, :, sk].masked_fill(~vis[:, sk], float("-inf"))
                    dump_s += (lgs - logZ.unsqueeze(-1)).exp().sum(-1).sum(1)
                n += s1 - s0
            res.append({"logZ": (logZ_s / n).cpu().numpy(), "D": (dump_s / n).cpu().numpy()})
        return qe, ke

    olmo3_mod.apply_rotary_pos_emb = wrapped
    try:
        with torch.no_grad():
            model.model(input_ids=ids_t.to(dev))
    finally:
        olmo3_mod.apply_rotary_pos_emb = orig
    assert len(res) == cfg.num_hidden_layers, f"rope fired {len(res)}x"
    return res


def s_init_from(res: list[dict], p_floor: float, p_cap: float) -> np.ndarray:
    """[L, H] warm-start sink logits: s = logZ + logit(clip(D, p_floor, p_cap))."""
    out = []
    for r in res:
        p = np.clip(r["D"], p_floor, p_cap)
        out.append(r["logZ"] + np.log(p / (1 - p)))
    return np.stack(out)
