# Copyright 2026 proof-pilot. Apache-2.0.
"""Validate sglang return_hidden_states for DeepSeek-V4 teacher extraction (Flash/Pro).

Runs inside the lmsysorg/sglang apptainer image (see run_in_container.sh); docs are
pre-rendered by _render_docs.py so this script needs only sglang/torch/numpy.
Hidden dim comes from --model's config (Flash 4096 / Pro 7168). Pro needs:
  --tp 8 --mem-frac 0.94 --chunk 4096 --max-running 1 --load-threads 1
(chunk 4096: the full-prompt-logprob B check OOMs at 8192; max-running 1 +
mem-frac 0.94: hybrid-SWA pool stall, see README pitfall #14).

Checks, against a TP=N offline Engine on real L2 docs:
  A. hidden shape/dtype is [seq, 4096] post-hc_head post-norm (needs the
     SGLANG_DSV4_HIDDEN_POST_NORM=1 patch bind-mounted over the image's
     deepseek_v4.py, see sglang_dsv4_post_norm_hidden.patch)
  B. logits reconstructed as hidden @ head.weight.T reproduce the engine's own
     input_token_logprobs (proves we captured the exact tensor that feeds lm_head)
  C. bs=1 vs batched submission return identical hidden for the same doc
     (guards the historical sglang batched-prefill hidden bugs #8066/#4997)
"""
import argparse
import faulthandler
import json
import os
import signal

os.environ["SGLANG_DSV4_HIDDEN_POST_NORM"] = "1"
# Full DeepGEMM warmup with chunk=16384 sweeps M up to 2*chunk and exhausts VRAM
# (manifests as flash_mla get_decoding_sched_meta "invalid argument" + scheduler crash).
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_FAST_WARMUP", "1")  # must be set before engine workers spawn

import numpy as np
import torch

MODEL = os.environ.get("DEEPSEEK_V4_FLASH", "/models/DeepSeek-V4-Flash")
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_docs.json")


def load_head_weight(model, hid):
    """head.weight bf16 [129280, hid] from the HF shards, upcast fp32 (as inference does)."""
    from safetensors import safe_open
    idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{model}/{idx['head.weight']}", framework="pt", device="cpu") as f:
        w = f.get_tensor("head.weight")
    assert w.dtype == torch.bfloat16 and w.shape == (129280, hid), (w.dtype, w.shape)
    return w.float()


def get_hidden(out):
    """Normalize the hidden_states field to a [seq, dim] float32 tensor.

    Pieces are nested lists (default path) or .pt file paths (spool mode)."""
    hs = out["meta_info"]["hidden_states"]
    if isinstance(hs, list):
        hs = [torch.load(h, weights_only=True) if isinstance(h, str)
              else torch.as_tensor(np.asarray(h)) for h in hs]
        hs = torch.cat([h if h.ndim == 2 else h.unsqueeze(0) for h in hs], dim=0)
    return hs.float()


def main():
    faulthandler.register(signal.SIGUSR1)  # kill -USR1 <main pid> dumps the stack
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--docs", default=DOCS)
    ap.add_argument("--mem-frac", type=float, default=0.80)
    ap.add_argument("--chunk", type=int, default=11264)
    ap.add_argument("--max-running", type=int, default=128)
    ap.add_argument("--load-threads", type=int, default=0,
                    help="serialize weight loading (strict-overcommit hosts); 0 = sglang default")
    a = ap.parse_args()
    hid = json.load(open(f"{a.model}/config.json"))["hidden_size"]  # Flash 4096 / Pro 7168

    # chunk = 11264: flash_mla get_decoding_sched_meta smem = 4*(5b+1) <= ~227KB caps
    # per-forward tokens at ~11.6k (sparse attn treats each prefill token as a batch
    # entry). Longer docs are chunked; hidden pieces accumulate across chunks (patched).
    docs = [(d["id"], d["input_ids"]) for d in json.load(open(a.docs))][:8]
    print(f"docs: {[(d[0][:24], len(d[1])) for d in docs]}", flush=True)
    head_w = load_head_weight(a.model, hid)

    import sglang as sgl
    llm = sgl.Engine(
        model_path=a.model,
        tp_size=a.tp,
        enable_return_hidden_states=True,
        disable_radix_cache=True,      # prefix-cache hits would skip hidden computation
        # Hidden capture only covers the final prefill chunk, so the chunk must cover the
        # whole doc: chunk >= max doc len (16384). NOTE -1 (disable) makes the DeepGEMM
        # warmup sweep 65536 token-counts and crashes flash_mla get_decoding_sched_meta.
        chunked_prefill_size=a.chunk,
        mem_fraction_static=a.mem_frac,      # leave VRAM headroom for warmup + long-extend activations
        max_running_requests=a.max_running,      # default (derived from KV pool) exceeds ~11.6k -> flash_mla
                                       # get_decoding_sched_meta smem 4*(5b+1) > 227KB -> invalid argument
        disable_cuda_graph=True,       # prefill-only workload; faster startup
        context_length=32768,          # don't size the KV pool for 1M ctx
        moe_runner_backend="marlin",   # w4a16 path for fp4 experts on Hopper
        watchdog_timeout=1800,         # first batch serially JIT-compiles ~10+ kernels (>5 min)
        log_level="info",
        # threads=1 keeps loader in-flight commit bounded on strict-overcommit hosts
        **({"model_loader_extra_config": {"enable_multithread_load": True,
                                          "num_threads": a.load_threads}}
           if a.load_threads else {}),
    )
    sp = {"max_new_tokens": 1, "temperature": 0.0}

    # ---- A+B: single doc, reconstruct logprobs from hidden ----
    doc_id, ids = docs[0]
    out = llm.generate(
        input_ids=[ids], sampling_params=sp,
        return_hidden_states=True, return_logprob=True, logprob_start_len=0,
    )[0]
    h = get_hidden(out)
    print(f"[A] hidden: shape={tuple(h.shape)} (expect ~({len(ids)}, {hid}))", flush=True)
    assert h.shape[-1] == hid, f"got dim {h.shape[-1]} — patch not active? (pre-hc_head = 4*hid)"
    n = len(ids)
    h_prompt = h[:n]

    # engine's own prompt logprobs: input_token_logprobs[i] = (logprob, token_id, text)
    # for token at position i, predicted from position i-1; entry 0 has logprob None.
    itl = out["meta_info"]["input_token_logprobs"]
    assert len(itl) == n, (len(itl), n)
    tgt = torch.tensor(ids[1:])
    eng = torch.tensor([t[0] for t in itl[1:]], dtype=torch.float32)
    # fp32 head (reference-impl semantic, what distillation training will use)
    lp32 = torch.log_softmax(h_prompt[:-1] @ head_w.T, dim=-1)
    d32 = (lp32[torch.arange(n - 1), tgt] - eng).abs()
    # bf16 head (what the engine itself computes) -- this is the apples-to-apples check
    lp16 = torch.log_softmax((h_prompt[:-1].bfloat16() @ head_w.bfloat16().T).float(), dim=-1)
    d16 = (lp16[torch.arange(n - 1), tgt] - eng).abs()
    print(f"[B] reconstructed-vs-engine logprob: bf16-head max|d|={d16.max():.3e} "
          f"mean|d|={d16.mean():.3e} (argmax pos {d16.argmax().item()}) | "
          f"fp32-head max|d|={d32.max():.3e} mean|d|={d32.mean():.3e} (n={n-1})", flush=True)
    diff = d16

    # ---- C: batched submission must carry each doc's own hidden ----
    # Numerical control first: bs=1 repeated -> nondeterminism floor (MoE/marlin kernels
    # may not be bitwise deterministic across batch compositions).
    refs = []
    for did, dids in docs:
        o = llm.generate(input_ids=[dids], sampling_params=sp, return_hidden_states=True)[0]
        refs.append(get_hidden(o))
    o2 = llm.generate(input_ids=[docs[0][1]], sampling_params=sp, return_hidden_states=True)[0]
    h2 = get_hidden(o2)[:n]
    floor = (h2 - refs[0][:n]).abs()
    floor_mean = floor.mean().item()
    print(f"[C0] bs=1 repeat nondeterminism floor (doc0): max|d|={floor.max():.3e} "
          f"mean|d|={floor_mean:.3e}", flush=True)

    def top1(h, dids):
        """greedy next-token ids reconstructed from hidden (bf16 head, sampled positions)"""
        m = len(dids)
        idx = torch.arange(0, m - 1, max(1, (m - 1) // 512))
        logits = (h[idx].bfloat16() @ head_w.bfloat16().T).float()
        return logits.argmax(-1), idx

    outs = llm.generate(
        input_ids=[d[1] for d in docs], sampling_params=sp, return_hidden_states=True,
    )
    ok = True
    for j, ((did, dids), o) in enumerate(zip(docs, outs)):
        hb = get_hidden(o)
        m = len(dids)
        if hb.shape[0] not in (m, m + 1):
            print(f"[C] FAIL {did[:24]}: hidden len {hb.shape[0]} != doc len {m}")
            ok = False
            continue
        dmax = (hb[:m] - refs[j][:m]).abs().max()
        dmean = (hb[:m] - refs[j][:m]).abs().mean()
        gb, idx = top1(hb, dids)
        gr, _ = top1(refs[j], dids)
        agree = (gb == gr).float().mean().item()
        # Pass = batch effects no worse than the engine's own rerun nondeterminism:
        # mean hidden diff within 3x floor (floor is ~1.4e-2; misassignment would be O(1)).
        verdict = "ok" if dmean < 3 * floor_mean else "FAIL"
        if verdict == "FAIL":
            ok = False
        print(f"[C] {verdict} doc{j} (len {m}): hidden max|d|={dmax:.3e} mean|d|={dmean:.3e} "
              f"| batched-vs-ref greedy agreement {agree:.4f} ({len(idx)} pos)", flush=True)
    d0 = floor.max()  # pass criterion: batch effects no worse than rerun nondeterminism

    llm.shutdown()
    b_pass = diff.mean() < 1e-3  # bf16-head mean: exact-tensor proof (max has argmax
                                 # flips at near-tie positions; mass is what matters)
    print(f"\nRESULT: A=PASS B={'PASS' if b_pass else 'FAIL'} "
          f"C={'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
