#!/usr/bin/env python3
# Copyright 2026 proof-pilot. Apache-2.0.
"""Build an sglang-servable deploy dir from an olmo3_sink training checkpoint.

Generalizes deploy/dflash/make_target_deploy.py (which hardcodes stage1-v2-7b) to any
src/dst, single- or multi-shard. Weights are hardlinked (original untouched, no 65GB copy);
config.json is rewritten per the stage-1 deploy recipe (docs/stage1_deploy_test.md):

  - rope_parameters -> legacy top-level rope_theta + rope_scaling (dodges sglang's
    Olmo3Config yarn-validation order bug; tf5 rebuilds rope_parameters from the aliases,
    so olmo2_sink.py's config.rope_parameters still works). **rope values copied VERBATIM**
    (factor / attention_factor preserved exactly as trained — no recompute from max-position).
  - drop auto_map (use sglang's patched olmo2 class, not trust_remote_code)
  - model_type=olmo3, architectures=[Olmo3SinkForCausalLM], dtype/torch_dtype=bfloat16,
    use_cache=true; max_position_embeddings kept as-is.

Serve (per the 7B PASS recipe), 32B may need --tp 2:
  apptainer exec --nv --bind $PP_ROOT \
    --bind deploy/target/olmo2_sink.py:/sgl-workspace/sglang/python/sglang/srt/models/olmo2.py \
    $SGLANG_SIF \
    python -m sglang.launch_server --model-path DST --tp 1 --attention-backend fa3 \
      --mem-fraction-static 0.85 --context-length 32768 --reasoning-parser deepseek-r1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _link_or_copy(src: Path, dst: Path, force: bool) -> str:
    if dst.exists() or dst.is_symlink():
        if not force:
            return "exists"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source olmo3_sink checkpoint dir")
    ap.add_argument("--dst", required=True, help="destination deploy dir")
    ap.add_argument("--force", action="store_true", help="overwrite existing linked files")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not (src / "config.json").is_file():
        raise FileNotFoundError(src / "config.json")
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    # rope: accept either training (rope_parameters) or already-legacy (rope_scaling) form.
    rp = dict(cfg.pop("rope_parameters", None) or cfg.get("rope_scaling") or {})
    if not rp:
        raise ValueError("no rope_parameters/rope_scaling in source config")
    cfg.pop("auto_map", None)
    cfg["model_type"] = "olmo3"
    cfg["architectures"] = ["Olmo3SinkForCausalLM"]
    cfg["dtype"] = "bfloat16"
    cfg["torch_dtype"] = "bfloat16"
    cfg["use_cache"] = True
    cfg["rope_theta"] = rp.get("rope_theta", cfg.get("rope_theta", 500000))
    cfg["rope_scaling"] = {k: v for k, v in rp.items() if k != "rope_theta"}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    counts: dict[str, int] = {}
    for item in src.iterdir():
        if item.name == "config.json" or item.name.startswith("_resume") or item.is_dir():
            continue
        if item.name.endswith(".safetensors"):
            mode = _link_or_copy(item, dst / item.name, args.force)  # hardlink the big weights
        else:
            shutil.copy2(item, dst / item.name)  # small aux: tokenizer / index / chat template
            mode = "copied"
        counts[mode] = counts.get(mode, 0) + 1

    rs = cfg["rope_scaling"]
    print(f"wrote deploy dir {dst}")
    print(f"  model_type={cfg['model_type']} dtype={cfg['dtype']} use_cache={cfg['use_cache']} "
          f"max_pos={cfg['max_position_embeddings']} auto_map={'auto_map' in cfg}")
    print(f"  rope_theta={cfg['rope_theta']} rope_type={rs.get('rope_type')} "
          f"factor={rs.get('factor')} attention_factor={rs.get('attention_factor')}")
    print(f"  files: {sorted(os.listdir(dst))}  links={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
