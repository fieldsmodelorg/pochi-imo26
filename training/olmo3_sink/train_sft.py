# Copyright 2026 proof-pilot. Apache-2.0.
"""Olmo3Sink SFT training entrypoint, wired to the throughput findings in
`docs/train_bench.md`.

Two regimes (auto-selected by world size), both Liger + olmo3_sink_fa3:

  SINGLE GPU  (python -m olmo3_sink.train_sft ...)
    pack to 65536, gradient checkpointing ON, Liger fused-linear-CE only, the decoder
    layers torch.compile'd, PagedAdamW8bit (optimizer state on CPU). ~9,056 tok/s on one
    H100, peak ~44 GB. Why this combo: compile fuses rope/norm/swiglu/residual itself
    (Liger's Triton kernels are opaque to compile, so we leave only the CE to Liger to
    keep logits from materializing), and a compiled fused layer makes checkpointing's
    recompute cheap -- recovering most of the 20% recompute tax. Requires the FA3 sink
    kernel to be compile-able (custom_op + register_fake; see fa3_sink_kernel.py).
    Do NOT add ckpt-skip (OOMs with compile) and do NOT use full Liger (blocks fusion).

  MULTI GPU   (torchrun --nproc_per_node=N -m olmo3_sink.train_sft --fsdp ...)
    FSDP2 full-shard (params+grads+optim sharded across ranks). Because sharding frees
    the memory that checkpointing was buying, we DROP activation checkpointing and use
    a per-rank microbatch of 8192 (one doc) with grad accumulation -> ~9,500 tok/s/GPU
    (+18% vs the single-GPU checkpointed path). microbatch must be >= 1 doc.

Data: by default a synthetic random-token dataset (smoke test). Swap `load_examples`
for your tokenized SFT set -> list[Example] (set prompt_len to mask prompts).
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from olmo3_sink.sft_data import Example, PackedCollator

MODEL_PATH = os.environ.get("OLMO3_SINK_MODEL", "/models/Olmo-3-1025-7B")


def load_examples(n: int, vocab: int, doc_len: int) -> list[Example]:
    """Placeholder synthetic dataset (docs ~doc_len long). Replace with your tokenized
    SFT set -> list[Example]. NOTE attention cost is O(sum doc_i^2), so real throughput
    tracks the *doc-length distribution*, not just total tokens; the 8192-doc spec keeps
    per-doc attention bounded (8x8192^2 << one 65536^2)."""
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        L = int(torch.randint(max(1, doc_len // 2), doc_len + 1, (1,), generator=g))
        ids = torch.randint(0, vocab, (L,), generator=g).tolist()
        out.append(Example(ids, prompt_len=L // 4))  # mask first quarter as "prompt"
    return out


def build_model(attn="olmo3_sink_fa3", liger="ce-only"):
    from transformers import AutoConfig, AutoModelForCausalLM
    from olmo3_sink import Olmo3SinkConfig, register_olmo3_sink
    register_olmo3_sink()
    d = AutoConfig.from_pretrained(MODEL_PATH).to_dict()
    d.pop("model_type", None)
    d.pop("architectures", None)
    cfg = Olmo3SinkConfig(**d)
    cfg.sink_init_value = -10.0  # warm-start ~ stock Olmo3
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, config=cfg, dtype=torch.bfloat16, attn_implementation=attn)
    if liger:
        from olmo3_sink import apply_liger
        if liger == "ce-only":  # leave rope/norm/swiglu for torch.compile to fuse
            apply_liger(model, rope=False, rms_norm=False, swiglu=False,
                        fused_linear_cross_entropy=True)
        else:
            apply_liger(model)
    return model


def setup_fsdp(model):
    """FSDP2 full-shard: shard each decoder layer + the root. No activation ckpt."""
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsdp", action="store_true", help="multi-GPU FSDP no-ckpt path")
    ap.add_argument("--micro-len", type=int, default=None,
                    help="tokens per microbatch row (default: 65536 single / 8192 fsdp)")
    ap.add_argument("--accum", type=int, default=None,
                    help="grad-accum microbatches per step (default: 1 single / 8 fsdp)")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                    help="torch.compile the decoder layers (single-GPU best; +15%%)")
    ap.add_argument("--doc-len", type=int, default=8192, help="synthetic doc length (~spec)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed warmup steps (raise above #distinct pack structures so compile fully amortizes)")
    ap.add_argument("--n-examples", type=int, default=512)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    use_fsdp = args.fsdp and world > 1
    micro_len = args.micro_len or (8192 if use_fsdp else 65536)
    accum = args.accum or (8 if use_fsdp else 1)

    if world > 1:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(rank)
    dev = f"cuda:{rank}" if world > 1 else "cuda"

    # ce-only Liger pairs with compile (compile fuses rope/norm/swiglu); without compile
    # use full Liger so those ops are still fused. FSDP path keeps full Liger (robust).
    liger_mode = "ce-only" if (args.compile and not use_fsdp) else True
    model = build_model(liger=liger_mode).to(dev)
    model.train()

    if use_fsdp:
        model = setup_fsdp(model)
        model.config.use_cache = False
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)  # state sharded by FSDP
    else:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": False})
        model.config.use_cache = False
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr)

    if args.compile:
        # regional compile of each decoder layer; the FA3 sink kernel is compile-able
        # (custom_op + register_fake) so each layer fuses into one graph with no break.
        for i in range(len(model.model.layers)):
            model.model.layers[i] = torch.compile(model.model.layers[i], dynamic=False)

    vocab = model.config.vocab_size
    # one doc max = micro_len so each microbatch row is one full packed bin
    doc_len = min(args.doc_len, micro_len)
    examples = load_examples(args.n_examples, vocab, doc_len=doc_len)
    # When compiling, emit FIXED-shape varlen metadata so torch.compile sees one static
    # structure (varying #docs would exceed Dynamo's recompile cache -> eager fallback).
    # Size max_segs to the shortest doc we allow, +slack for the pad segment.
    max_segs = (micro_len // max(1, doc_len // 2) + 2) if args.compile else None
    coll = PackedCollator(max_len=micro_len, pad_id=0, device=dev, max_segs=max_segs)
    bins = list(coll.iter_bins(examples))
    if rank == 0:
        print(f"[train] fsdp={use_fsdp} world={world} micro_len={micro_len} accum={accum} "
              f"bins={len(bins)} tok/step={micro_len*accum}")

    bi = 0
    for step in range(args.steps + args.warmup):
        if step == args.warmup:
            torch.cuda.synchronize(); t0 = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        last = None
        for _ in range(accum):
            batch = bins[bi % len(bins)]; bi += 1
            out = model(**batch)
            (out.loss / accum).backward()
            last = out.loss.detach()
        opt.step()
        if rank == 0:
            print(f"  step {step} loss {last.item():.4f}")

    if step >= args.warmup:
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.steps
        peak = torch.cuda.max_memory_allocated() / 1e9
        if rank == 0:
            print(f"[train] {dt*1000:.0f} ms/step | {micro_len*accum/dt:,.0f} tok/s/GPU "
                  f"| peak {peak:.1f} GB" + (f" | {micro_len*accum*world/dt:,.0f} tok/s global" if world > 1 else ""))

    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
