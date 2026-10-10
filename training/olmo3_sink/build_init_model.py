# Copyright 2026 proof-pilot. Apache-2.0.
"""One-command e2e builder: native Olmo 3 -> DeepSeek-tokenizer + measured-s_aux
trainable checkpoint.

Chain (single process; sinks are written DIRECTLY into the weights, no
intermediate npz):

  1. OMP tokenizer transplant (tokenizer_transplant YAML config; reused if the
     out dir already exists, --redo-transplant to force)
  2. load the transplant as Olmo3Sink (eager, sinks=-1e4 == stock behavior)
  3. measure per-head received mass at --seq-len (sliding steady state)
  4. derive per-head s_init = logZ + logit(clip(D, p_floor, p_cap)) over rows
     >= --qmin and copy it into `self_attn.sinks`
  5. save the fused checkpoint: olmo3_sink model_type + trust_remote_code
     modules + tokenizer/chat template + sink_provenance.json, and report the
     step-0 LM-loss delta (expect ~+0.03 on a fresh transplant base; the
     dead-start alternative was measured to never train -- docs §9)

Method, measurements and the gate rationale: docs/attn_sink_study.md §6/§9/§10.

Usage (GPU required; 7B ~15 min, 32B ~1.5 h):
  CUDA_VISIBLE_DEVICES=4 uv run python -m olmo3_sink.build_init_model \
      --transplant-config tokenizer_transplant/configs/olmo3_think_7b__deepseek_v4_flash.yaml
  # output defaults to <transplant out>-sink
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .sink_calib import TEXT, build_sink_keys, derive_logZ_D, lm_loss, measure, s_init_from


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transplant-config", required=True,
                    help="tokenizer_transplant YAML (defines base / donor / transplant out)")
    ap.add_argument("--dst", default=None, help="fused output dir (default: <transplant out>-sink)")
    ap.add_argument("--redo-transplant", action="store_true",
                    help="re-run the transplant even if its out dir exists")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--qmin", type=int, default=4096,
                    help="derivation rows (>= sliding window => slid steady state)")
    ap.add_argument("--eps", type=float, default=0.04, help="received-mass sink-key threshold")
    ap.add_argument("--p-floor", type=float, default=0.02)
    ap.add_argument("--p-cap", type=float, default=0.10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from tokenizer_transplant.transplant import TransplantConfig
    from tokenizer_transplant.transplant import run as transplant_run

    from .configuration_olmo3_sink import Olmo3SinkConfig
    from .modeling_olmo3_sink import Olmo3SinkForCausalLM

    tcfg = TransplantConfig.from_yaml(args.transplant_config)
    dst = args.dst or f"{tcfg.out.rstrip('/')}-sink"

    # ---- 1. transplant (idempotent) ----
    if args.redo_transplant or not (Path(tcfg.out) / "config.json").exists():
        print(f"[1/5] transplant: {tcfg.base} -> {tcfg.out}")
        transplant_run(tcfg, device=args.device.split(":")[0])
    else:
        print(f"[1/5] transplant: reusing existing {tcfg.out}")

    # ---- 2. load as Olmo3Sink, stock behavior ----
    print(f"[2/5] load {tcfg.out} (eager, sinks=-1e4 == stock)")
    cfg = Olmo3SinkConfig.from_pretrained(tcfg.out)
    cfg.sink_init_value = -1e4
    cfg._attn_implementation = "eager"
    model = Olmo3SinkForCausalLM.from_pretrained(tcfg.out, config=cfg, dtype=torch.bfloat16)
    model = model.to(args.device).eval()

    tok = AutoTokenizer.from_pretrained(tcfg.out)
    ids_all = tok(TEXT.read_text(), return_tensors="pt").input_ids[0]
    S = args.seq_len
    ids = ids_all[:S]
    loss_windows = [ids_all[i * S:(i + 1) * S] for i in range(2)]
    loss_before = lm_loss(model, loss_windows, args.device)

    # ---- 3. received-mass measurement (which keys absorb the no-op dump) ----
    print(f"[3/5] measure received mass @ seq {S}")
    stats = measure(model, ids, args.device)
    sink_keys = build_sink_keys([st["received"] for st in stats], args.eps)
    n_keys = {li: len(k) for li, k in sink_keys.items() if len(k)}
    print(f"      sink keys/layer: mean {np.mean(list(n_keys.values() or [0])):.1f} "
          f"({len(n_keys)}/{cfg.num_hidden_layers} layers non-empty)")

    # ---- 4. derive s_init and write it into the weights ----
    print(f"[4/5] derive s_init (rows >= {args.qmin}) and write into sinks")
    res = derive_logZ_D(model, ids, sink_keys, qmin=args.qmin)
    sinks = s_init_from(res, args.p_floor, args.p_cap)  # [L, H]
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.sinks.data.copy_(
            torch.from_numpy(sinks[i]).to(layer.self_attn.sinks.dtype))
    loss_after = lm_loss(model, loss_windows, args.device)
    D = np.stack([r["D"] for r in res])
    print(f"      s_init mean {sinks.mean():.2f} p10 {np.percentile(sinks, 10):.2f} "
          f"p90 {np.percentile(sinks, 90):.2f} | D mean {D.mean():.3f}")
    print(f"      step-0 LM loss: {loss_before:.4f} -> {loss_after:.4f} "
          f"(delta {loss_after - loss_before:+.4f})")

    # ---- 5. save fused checkpoint ----
    print(f"[5/5] save -> {dst}")
    cfg.sink_init_value = 0.0  # only relevant for hypothetical future missing keys
    cfg.register_for_auto_class()
    model.register_for_auto_class("AutoModelForCausalLM")
    Path(dst).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst)
    tok.save_pretrained(dst)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git = None
    (Path(dst) / "sink_provenance.json").write_text(json.dumps({
        "pipeline": "olmo3_sink.build_init_model (e2e: native -> transplant -> measured sinks)",
        "base": tcfg.base,
        "transplant_config": str(Path(args.transplant_config).resolve()),
        "transplant_out": tcfg.out,
        "calib_text": str(TEXT),
        "calib_text_sha256": hashlib.sha256(TEXT.read_bytes()).hexdigest()[:16],
        "seq_len": S, "qmin": args.qmin, "eps": args.eps,
        "p_floor": args.p_floor, "p_cap": args.p_cap,
        "sinks_mean": float(sinks.mean()), "sinks_min": float(sinks.min()),
        "sinks_max": float(sinks.max()), "D_mean": float(D.mean()),
        "lm_loss_before": loss_before, "lm_loss_after": loss_after,
        "git_commit": git,
    }, indent=1))
    print(f"[done] trainable fused model at {dst}")
    print(f"       train: point stage1_v2 --model_path here (baked sinks survive load; "
          f"SINK_INIT only fills missing keys)")


if __name__ == "__main__":
    main()
