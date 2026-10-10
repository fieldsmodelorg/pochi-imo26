#!/usr/bin/env python
"""Convert an olmo3_sink MHA checkpoint to GQA by mean-pooling KV heads.

The 7B stage1 checkpoints are full MHA (num_attention_heads == num_key_value_heads
== 32). This produces a GQA checkpoint (Q unchanged, KV heads reduced to --kv-heads,
default 8, matching the 32B) via the canonical mean-pool conversion (Ainslie et al.
2023, arXiv:2305.13245): each surviving KV head = mean of the `group` source KV heads
that the corresponding query-head group used.

What changes (per layer), everything else copied verbatim:
  k_proj  (n_kv*hd, H) -> (kv*hd, H)   mean-pool contiguous groups of `group` heads
  v_proj  (n_kv*hd, H) -> (kv*hd, H)   same
  k_norm  (n_kv*hd,)   -> (kv*hd,)     QK-norm is over the *flat* KV projection (applied
                                        before the head reshape in the forward), and its
                                        dim scales with KV heads (7B: 4096 -> 1024); pool
                                        its per-head weight slices the same way.
Unchanged: q_proj, q_norm (Q heads stay 32), o_proj, sinks (per-Q-head [32]), mlp,
norms, embed, lm_head, and the config except num_key_value_heads.

RoPE note: standard RoPE rotates every head by the *same* position-dependent rotation
(it depends on position and the within-head dim index, not the head index), so pooling
pre-RoPE K projections does not mix rotational frames -- the mean-pool is RoPE-safe for
this architecture. Init quality is still verified empirically (see _gqa_initcheck.py).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _pool_heads(w: torch.Tensor, n_src: int, n_dst: int, head_dim: int) -> torch.Tensor:
    """Mean-pool contiguous head groups. `w` is (n_src*head_dim, ...) or (n_src*head_dim,)."""
    group = n_src // n_dst
    assert n_src % n_dst == 0, f"{n_src} not divisible by {n_dst}"
    rest = w.shape[1:]
    wf = w.float().reshape(n_src, head_dim, *rest)          # (n_src, hd, ...)
    wf = wf.reshape(n_dst, group, head_dim, *rest)          # (n_dst, group, hd, ...)
    pooled = wf.mean(dim=1)                                 # (n_dst, hd, ...)
    return pooled.reshape(n_dst * head_dim, *rest).to(w.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source MHA checkpoint dir")
    ap.add_argument("--out", required=True, help="output GQA checkpoint dir")
    ap.add_argument("--kv-heads", type=int, default=8, help="target num_key_value_heads")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    n_q = cfg["num_attention_heads"]
    n_kv_src = cfg["num_key_value_heads"]
    head_dim = cfg["hidden_size"] // n_q
    n_kv_dst = args.kv_heads
    assert n_q % n_kv_dst == 0 and n_kv_src % n_kv_dst == 0
    print(f"convert {src.name}: Q={n_q} KV {n_kv_src}->{n_kv_dst} (head_dim {head_dim}, "
          f"group {n_kv_src // n_kv_dst})")

    # single-file checkpoint (7B is one model.safetensors)
    st_path = src / "model.safetensors"
    assert st_path.exists(), f"expected single-file {st_path}"
    new_tensors: dict[str, torch.Tensor] = {}
    n_pooled = 0
    with safe_open(str(st_path), "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight",
                           "self_attn.k_norm.weight")):
                t = _pool_heads(t, n_kv_src, n_kv_dst, head_dim)
                n_pooled += 1
            new_tensors[k] = t
    print(f"pooled {n_pooled} tensors (expect {3 * cfg['num_hidden_layers']})")
    assert n_pooled == 3 * cfg["num_hidden_layers"], "missing attn tensors -- check key names"

    save_file(new_tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    cfg["num_key_value_heads"] = n_kv_dst
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for fn in ["chat_template.jinja", "generation_config.json", "tokenizer_config.json",
               "tokenizer.json"]:
        if (src / fn).exists():
            shutil.copy2(src / fn, out / fn)
    print(f"wrote GQA checkpoint -> {out}  (num_key_value_heads={n_kv_dst})")


if __name__ == "__main__":
    main()
