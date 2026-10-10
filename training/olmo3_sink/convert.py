# Copyright 2026 proof-pilot. Apache-2.0.
"""Approach B: warm-start an Olmo3Sink checkpoint from a stock Olmo3 checkpoint
and bake in the custom code so it loads anywhere via `trust_remote_code=True`
(Kaggle / FMI container, without a patched transformers install).

Usage:
    uv run python -m olmo3_sink.convert \
        --src allenai/Olmo-3-1025-7B \
        --dst /work/.../Olmo-3-7B-sink \
        --sink-init-value -10.0

The transformer body + embeddings load bit-for-bit from `--src`; only the new
per-head `sinks` parameters are freshly initialized (to `--sink-init-value`).
With a strongly negative init the model is numerically ~identical to the base
Olmo3 at step 0, and SFT learns the sinks in.

With `--sinks-npz` the sinks are instead BAKED from a measured per-head warm
start (`_attn_sink_init.py` output; see docs/attn_sink_study.md §9 — scalar
inits 0.0/-10 are dead starts for SFT-time sink injection). `--src` may be a
tokenizer-transplant output dir, yielding a single ready-to-train
"deepseek-tokenizer + s_aux" artifact:

    uv run python -m olmo3_sink.convert \
        --src models/Olmo-3-7B-Think-deepseekTok \
        --dst models/Olmo-3-7B-Think-deepseekTok-sink \
        --sinks-npz outputs/attn_sink_study/sink_init_7b_dstok_s8192.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .configuration_olmo3_sink import Olmo3SinkConfig
from .modeling_olmo3_sink import Olmo3SinkForCausalLM


def convert(src: str, dst: str, sink_init_value: float, dtype: str,
            sinks_npz: str | None = None) -> None:
    base_cfg = AutoConfig.from_pretrained(src)
    cfg_dict = base_cfg.to_dict()
    cfg_dict.pop("model_type", None)
    cfg_dict.pop("architectures", None)
    cfg = Olmo3SinkConfig(**cfg_dict)
    cfg.sink_init_value = sink_init_value

    torch_dtype = getattr(torch, dtype)
    # Loads matching weights from `src`; `sinks` are missing -> _init_weights
    # fills them with sink_init_value.
    model = Olmo3SinkForCausalLM.from_pretrained(
        src, config=cfg, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )

    provenance = None
    if sinks_npz is not None:
        # Bake measured per-head warm-start sinks (from _attn_sink_init.py) into the
        # checkpoint, so the artifact ships ready to train (docs/attn_sink_study.md §9:
        # scalar init 0.0/-10 are dead starts for SFT-time sink injection).
        import hashlib
        import json

        import numpy as np

        z = np.load(sinks_npz)
        sinks = z["sinks_init"]  # [L, H]
        L, H = cfg.num_hidden_layers, cfg.num_attention_heads
        if sinks.shape != (L, H):
            raise ValueError(f"sinks_init shape {sinks.shape} != model ({L}, {H})")
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.sinks.data.copy_(
                torch.from_numpy(sinks[i]).to(layer.self_attn.sinks.dtype))
        provenance = {
            "sinks_npz": str(Path(sinks_npz).resolve()),
            "sinks_npz_sha256": hashlib.sha256(Path(sinks_npz).read_bytes()).hexdigest()[:16],
            "derivation": json.loads(str(z["meta"])) if "meta" in z else None,
            "src": str(src),
            "sinks_baked_mean": float(sinks.mean()),
            "sinks_baked_min": float(sinks.min()),
            "sinks_baked_max": float(sinks.max()),
        }
        print(f"[ok] baked measured sinks [L={L},H={H}] "
              f"mean {sinks.mean():.2f} range {sinks.min():.2f}..{sinks.max():.2f}")

    # Bake the defining modules into the checkpoint (writes config.json auto_map +
    # copies configuration_olmo3_sink.py / modeling_olmo3_sink.py next to it).
    cfg.register_for_auto_class()
    model.register_for_auto_class("AutoModelForCausalLM")

    Path(dst).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst)
    try:
        AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    except Exception as exc:  # tokenizer is optional here
        print(f"[warn] tokenizer not copied: {exc}")
    if provenance is not None:
        import json
        import subprocess
        try:
            provenance["git_commit"] = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            provenance["git_commit"] = None
        (Path(dst) / "sink_provenance.json").write_text(json.dumps(provenance, indent=1))
        print(f"[ok] wrote sink_provenance.json")
    print(f"[ok] wrote Olmo3Sink checkpoint to {dst}")
    print("      load with: AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="stock Olmo3 checkpoint (path or hub id)")
    ap.add_argument("--dst", required=True, help="output dir for the Olmo3Sink checkpoint")
    ap.add_argument("--sink-init-value", type=float, default=-10.0,
                    help="scalar fill for sinks missing from --src (ignored where --sinks-npz applies)")
    ap.add_argument("--sinks-npz", default=None,
                    help="bake measured per-head sinks_init [L,H] from _attn_sink_init.py "
                         "(writes sink_provenance.json next to the weights)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()
    convert(args.src, args.dst, args.sink_init_value, args.dtype, sinks_npz=args.sinks_npz)


if __name__ == "__main__":
    main()
