#!/usr/bin/env python3
# Source: ycchen-tw/proof-pilot-codes@bc03a2c71a076990deaad3d712c6889682e12c69
# Original path: deploy/dflash/convert_draft.py
"""Convert a trained DFlash checkpoint into an SGLang-loadable draft directory.

The patched SGLang draft class (`dflash_sink.py`, bind-mounted over
`srt/models/dflash.py`) keeps the same submodule names as the trained draft, so
weights load 1:1 except q/k/v and gate/up, which SGLang's loader fuses from the
split projections. The converter writes only `checkpoint["model"]` to
safetensors and derives a minimal DFlash config from the checkpoint config and
tensor shapes.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
from safetensors.torch import save_file

DEFAULT_DRAFT_DIR = "outputs/dflash-canonical-7b-v2-32g-s12000"
DEFAULT_OUT_DIR = "outputs/dflash-canonical-sink-sglang-draft"


def _load_checkpoint(path: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _strip_module(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        k[len("module."):] if k.startswith("module.") else k: v.contiguous()
        for k, v in sd.items()
    }


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg[key] if key in cfg and cfg[key] is not None else default
    value = getattr(cfg, key, None)
    return value if value is not None else default


def _rope_theta(cfg: dict[str, Any]) -> float:
    rp = _cfg_get(cfg, "rope_parameters", {}) or {}
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        return float(rp["rope_theta"])
    return float(_cfg_get(cfg, "rope_theta", 500000))


def _infer_config(sd: dict[str, torch.Tensor], ckpt: dict[str, Any], block_size: int | None) -> dict[str, Any]:
    cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    dcfg = dict(_cfg_get(cfg, "dflash_config", {}) or {})

    n_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers."))
    hidden_size = int(_cfg_get(cfg, "hidden_size", sd["fc.weight"].shape[0]))
    intermediate_size = int(
        _cfg_get(cfg, "intermediate_size", sd["layers.0.mlp.gate_proj.weight"].shape[0])
    )
    num_attention_heads = int(
        _cfg_get(cfg, "num_attention_heads", sd["layers.0.self_attn.sinks"].shape[0])
    )
    head_dim = int(_cfg_get(cfg, "head_dim", sd["layers.0.self_attn.q_proj.weight"].shape[0] // num_attention_heads))
    num_key_value_heads = int(
        _cfg_get(cfg, "num_key_value_heads", sd["layers.0.self_attn.k_proj.weight"].shape[0] // head_dim)
    )
    resolved_block_size = int(
        block_size
        if block_size is not None
        else _cfg_get(ckpt.get("args", {}) or {}, "block_size", _cfg_get(cfg, "block_size", 11))
    )
    num_target_layers = int(
        _cfg_get(cfg, "num_target_layers", dcfg.get("num_target_layers", n_layers))
    )
    target_layer_ids = dcfg.get("target_layer_ids")
    if target_layer_ids is None:
        target_layer_ids = [
            min(num_target_layers - 1, round((i + 1) * num_target_layers / (n_layers + 1)) - 1)
            for i in range(n_layers)
        ]
    sliding_window = int(_cfg_get(cfg, "sliding_window", dcfg.get("sliding_window", 128)))

    return {
        "architectures": ["DFlashDraftModel"],
        # model_type only drives AutoConfig parsing; the model class is selected
        # by architectures -> DFlashDraftModel from the bind-mounted dflash_sink.py.
        "model_type": "qwen3",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": n_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "hidden_act": _cfg_get(cfg, "hidden_act", "silu"),
        "rms_norm_eps": float(_cfg_get(cfg, "rms_norm_eps", 1e-6)),
        "attention_bias": bool(_cfg_get(cfg, "attention_bias", False)),
        "vocab_size": int(_cfg_get(cfg, "vocab_size", 129280)),
        "max_position_embeddings": int(_cfg_get(cfg, "max_position_embeddings", 65536)),
        "rope_theta": _rope_theta(cfg),
        "rope_scaling": None,
        "sliding_window": sliding_window,
        "layer_types": ["sliding_attention"] * n_layers,
        "block_size": resolved_block_size,
        "num_target_layers": num_target_layers,
        "target_hidden_size": int(_cfg_get(cfg, "target_hidden_size", hidden_size)),
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": bool(_cfg_get(cfg, "tie_word_embeddings", False)),
        "dflash_config": {
            "mask_token_id": int(dcfg.get("mask_token_id", 128000)),
            "target_layer_ids": [int(x) for x in target_layer_ids],
            "num_target_layers": num_target_layers,
            "block_size": resolved_block_size,
            "use_attention_sink": bool(dcfg.get("use_attention_sink", True)),
            "sliding_window": sliding_window,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-dir", default=DEFAULT_DRAFT_DIR)
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint path. Defaults to final.pt if present, else latest.pt in --draft-dir.")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--block-size", type=int, default=None,
                    help="sglang DFlash block_size; it drafts block_size-1 tokens, "
                         "so 11 -> 10 drafted positions matching canonical B=10.")
    args = ap.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        final_path = os.path.join(args.draft_dir, "final.pt")
        latest_path = os.path.join(args.draft_dir, "latest.pt")
        ckpt_path = final_path if os.path.exists(final_path) else latest_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    os.makedirs(args.out, exist_ok=True)
    ckpt = _load_checkpoint(ckpt_path)
    sd = _strip_module(ckpt["model"])
    config = _infer_config(sd, ckpt, block_size=args.block_size)

    save_file(sd, os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    n_layers = int(config["num_hidden_layers"])
    print(f"wrote {len(sd)} tensors + config to {args.out}")
    print(f"  checkpoint={ckpt_path}")
    print(f"  checkpoint_step={ckpt.get('step')}")
    print(f"  hidden_size={config['hidden_size']}, num_hidden_layers={n_layers}, block_size={config['block_size']}")
    print(f"  num_target_layers={config['num_target_layers']}, target_layer_ids={config['dflash_config']['target_layer_ids']}")
    print(f"  fc.weight={tuple(sd['fc.weight'].shape)}")
    print(f"  layers.0.self_attn.sinks={tuple(sd['layers.0.self_attn.sinks'].shape)}")
    print(f"  mask_embed={tuple(sd['mask_embed'].shape)}")


if __name__ == "__main__":
    main()
