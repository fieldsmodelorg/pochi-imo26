#!/usr/bin/env python3
# Copyright 2026 proof-pilot. Apache-2.0.
"""Create a long-context YaRN deploy directory for OPD rollout servers.

The OPD rollout container expects the legacy deploy config shape used by
`deploy/target/olmo2_sink.py`: top-level `rope_theta` plus `rope_scaling`.
Training checkpoints use `rope_parameters`. This tool links all checkpoint files
into a new directory and rewrites only config.json, so preparing a 131k/160k
deploy variant does not duplicate the large safetensors file.

Example:
  python deploy/prepare_yarn_deploy.py \
    --src outputs/opd-mnlong6-s220 \
    --dst outputs/opd-mnlong6-s220-ctx131k-deploy \
    --max-position 131072
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path


def _link_or_copy(src: Path, dst: Path, force: bool) -> str:
    if dst.exists() or dst.is_symlink():
        if not force:
            return "exists"
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        try:
            os.symlink(src.resolve(), dst)
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def _rope_base(cfg: dict) -> dict:
    rope = dict(cfg.get("rope_scaling") or cfg.get("rope_parameters") or {})
    if "rope_theta" not in rope:
        rope["rope_theta"] = cfg.get("rope_theta", 500000)
    rope.setdefault("rope_type", "yarn")
    rope.setdefault("original_max_position_embeddings", 8192)
    rope.setdefault("beta_fast", 32.0)
    rope.setdefault("beta_slow", 1.0)
    return rope


def _rewrite_config(cfg: dict, max_position: int, fmt: str, attention_factor: float | None) -> dict:
    rope = _rope_base(cfg)
    original = int(rope["original_max_position_embeddings"])
    factor = max_position / original
    rope["factor"] = factor
    rope["attention_factor"] = (
        float(attention_factor) if attention_factor is not None else 0.1 * math.log(factor) + 1.0
    )

    out = dict(cfg)
    out["max_position_embeddings"] = max_position
    if fmt == "deploy":
        if out.get("model_type") == "olmo3_sink":
            out["model_type"] = "olmo3"
        out["rope_theta"] = rope["rope_theta"]
        out["rope_scaling"] = {
            "attention_factor": rope["attention_factor"],
            "beta_fast": rope["beta_fast"],
            "beta_slow": rope["beta_slow"],
            "factor": rope["factor"],
            "original_max_position_embeddings": original,
            "rope_type": rope["rope_type"],
        }
        out.pop("rope_parameters", None)
        out["use_cache"] = True
        out.setdefault("torch_dtype", "bfloat16")
    else:
        out["rope_parameters"] = {
            "attention_factor": rope["attention_factor"],
            "beta_fast": rope["beta_fast"],
            "beta_slow": rope["beta_slow"],
            "factor": rope["factor"],
            "original_max_position_embeddings": original,
            "rope_theta": rope["rope_theta"],
            "rope_type": rope["rope_type"],
        }
        out.pop("rope_scaling", None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source checkpoint/deploy directory")
    ap.add_argument("--dst", required=True, help="destination deploy directory")
    ap.add_argument("--max-position", type=int, default=131072)
    ap.add_argument("--format", choices=["deploy", "train"], default="deploy")
    ap.add_argument("--attention-factor", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not (src / "config.json").is_file():
        raise FileNotFoundError(src / "config.json")
    dst.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for item in src.iterdir():
        if item.name == "config.json":
            continue
        target = dst / item.name
        if item.is_dir():
            if target.exists() and not args.force:
                counts["exists"] = counts.get("exists", 0) + 1
                continue
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target, symlinks=True)
            counts["copytree"] = counts.get("copytree", 0) + 1
        elif item.is_file():
            mode = _link_or_copy(item, target, args.force)
            counts[mode] = counts.get(mode, 0) + 1

    cfg = json.loads((src / "config.json").read_text())
    out = _rewrite_config(cfg, args.max_position, args.format, args.attention_factor)
    (dst / "config.json").write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")

    rope = out.get("rope_scaling") or out.get("rope_parameters")
    print(
        f"wrote {dst}/config.json max_position={out['max_position_embeddings']} "
        f"factor={rope['factor']:.6g} attention_factor={rope['attention_factor']:.6g} links={counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
