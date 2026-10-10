#!/usr/bin/env python
"""OPD v2 training-image smoke — validates the B200 sink path INSIDE the container.

The one thing that de-risks B200: `attn_implementation="flash_attention_2"` must reproduce the
learnable attention sink that the (Hopper-only) patched FA3 kernel used to provide. We check FA2
against the repo's own `eager` sink reference (`eager_attention_forward_with_sink`) on a tiny
Olmo3Sink model — no 32B weights needed.

Checks:
  1. deps: torch 2.11 / cu12.8 / sm_100 visible; FSDP2 import; flash-attn importable.
  2. forward parity: fa2 last_hidden_state ~= eager (per-head sinks set to a nonzero value).
  3. sink-grad parity: the `self_attn.sinks` grad (the trainable sink) matches eager.
  4. q/k/v-path grad parity: the `q_proj.weight` grad matches eager. THIS is the check that earns
     "fine for training" — a naive post-correction sink (detached lse) would pass 2 & 3 but fail
     HERE, because the q/k/v gradient term flowing through lse would be missing/wrong.

Run inside the image on a GPU:  python /opt/opd/opd_v2_train_smoke.py
Exit 0 = pass. Skips (exit 0 with SKIP) if no GPU / no flash-attn (so it's safe in CI without a B200).
"""
from __future__ import annotations

import os
import sys

import torch

REPO = os.environ.get("OPD_REPO", "/opt/opd/repo")
for p in (f"{REPO}/training", f"{REPO}/training/olmo3_sink", f"{REPO}/training/opd_v2/src",
          f"{REPO}/training/stage1_v2/src", f"{REPO}/training/_common", f"{REPO}/training/_vendor_opd"):
    if p not in sys.path:
        sys.path.insert(0, p)

TOL_FWD = 3e-2      # bf16 tolerance for FULL-model logits (accumulates over embed->layers->lm_head)
TOL_GRAD = 3e-2
IGNORE = -100       # OPD label ignore index (matches opd_v2 trainer)


def _skip(msg: str) -> None:
    print(f"SKIP: {msg}")
    sys.exit(0)


def check_deps() -> None:
    print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
    # cu128 is the hard constraint (B200/AI2); torch minor (2.10/2.11) is not
    assert (torch.version.cuda or "").startswith("12.8"), f"expected cu12.8, got {torch.version.cuda}"
    if not torch.cuda.is_available():
        _skip("no CUDA device")
    cc = torch.cuda.get_device_capability()
    print(f"device {torch.cuda.get_device_name(0)} sm_{cc[0]}{cc[1]}")
    from torch.distributed.fsdp import fully_shard  # noqa: F401
    try:
        import flash_attn  # noqa: F401
        print("flash-attn", flash_attn.__version__)
    except Exception as e:  # noqa: BLE001
        _skip(f"flash-attn not importable ({e}) -> can't test FA2 sink path")


def build_pair():
    """Two tiny Olmo3Sink models sharing weights: one eager, one flash_attention_2, nonzero sinks."""
    from olmo3_sink import Olmo3SinkConfig, register_olmo3_sink
    from transformers import AutoModelForCausalLM
    register_olmo3_sink()
    kw = dict(
        vocab_size=1024, hidden_size=256, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
        head_dim=64, max_position_embeddings=1024, sink_init_value=0.0,
        rope_theta=10000.0,
        # NB: don't override tie_word_embeddings — transformers 5.x strict-validates its type and the
        # expected type flipped across versions (5.4=int, 5.13=bool); use the config's own default.
    )
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    m_eager = AutoModelForCausalLM.from_config(
        Olmo3SinkConfig(**kw), attn_implementation="eager").to(dev, dt).eval()
    # olmo3_sink_fa2 = the sink-correct FA2 backend (post-correction re-normalization). NOT stock
    # "flash_attention_2", which SILENTLY DROPS the sink (verified: stock FA2 has no sink arg).
    m_fa2 = AutoModelForCausalLM.from_config(
        Olmo3SinkConfig(**kw), attn_implementation="olmo3_sink_fa2").to(dev, dt).eval()
    m_fa2.load_state_dict(m_eager.state_dict())
    # nonzero per-head sinks (else the test barely exercises the sink term)
    with torch.no_grad():
        for m in (m_eager, m_fa2):
            for layer in m.model.layers:
                layer.self_attn.sinks.copy_(
                    torch.linspace(-1.0, 1.0, layer.self_attn.sinks.numel(), device=dev))
    return m_eager, m_fa2


def check_correction_fp64(dev: str = "cuda") -> None:
    """HIGH-PRECISION exactness: the FA2 sink post-correction (out / (1+exp(sink-lse))) reproduces the
    eager gpt-oss sink reference to fp64 machine precision — forward AND every gradient. This isolates
    the correction MATH from flash's bf16-only kernel, so any bf16 residual downstream is provably just
    flash rounding, not sink error. (Pure torch, no flash — runs on any device.)
    """
    torch.manual_seed(0)
    dt = torch.float64
    T, H, D = 64, 4, 32
    q, k, v = (torch.randn(T, H, D, dtype=dt, device=dev, requires_grad=True) for _ in range(3))
    sink = torch.randn(H, dtype=dt, device=dev, requires_grad=True)
    scale = D ** -0.5
    cmask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1)

    def scores(q, k):
        s = torch.einsum("thd,shd->hts", q, k) * scale
        return s.masked_fill(cmask[None], float("-inf"))

    def postcorr(q, k, v, sink):  # what olmo3_sink_fa2 implements: no-sink (out,lse) then rescale
        s = scores(q, k)
        lse = torch.logsumexp(s, -1)
        out = torch.einsum("hts,shd->thd", torch.softmax(s, -1), v)
        return out / (1.0 + torch.exp(sink.unsqueeze(1) - lse)).transpose(0, 1).unsqueeze(-1)

    def eager_sink(q, k, v, sink):  # gpt-oss reference: extra softmax column, dropped after norm
        s = scores(q, k)
        col = sink.view(H, 1, 1).expand(H, T, 1)
        p = torch.softmax(torch.cat([s, col], -1), -1)[..., :-1]
        return torch.einsum("hts,shd->thd", p, v)

    oa, ob = postcorr(q, k, v, sink), eager_sink(q, k, v, sink)
    ferr = (oa - ob).abs().max().item()
    ga = torch.autograd.grad(oa.sum(), [q, k, v, sink], retain_graph=True)
    gb = torch.autograd.grad(ob.sum(), [q, k, v, sink])
    gerr = {n: (a - b).abs().max().item() for n, (a, b) in zip(["dq", "dk", "dv", "dsink"], zip(ga, gb))}
    print("fp64 correction exactness: fwd=%.2e " % ferr + " ".join(f"{n}={e:.2e}" for n, e in gerr.items()))
    TOL = 1e-10
    assert ferr < TOL and all(e < TOL for e in gerr.values()), "FA2 sink correction NOT exact in fp64!"
    print("PASS: FA2 sink correction is algebraically EXACT (fp64 machine precision)")


def check_packed_isolation(m, dev: str = "cuda") -> bool:
    """Training packs trajectories varlen (cu_seqlens, no windowing). Verify the olmo3_sink_fa2 model
    isolates documents: packed[doc] equals that doc run ALONE, and perturbing one doc leaves the other's
    logits BIT-identical (zero cross-document attention leak)."""
    V = m.config.vocab_size
    L1, L2 = 40, 56
    torch.manual_seed(1)
    ids1 = torch.randint(0, V, (1, L1), device=dev)
    ids2 = torch.randint(0, V, (1, L2), device=dev)
    ids = torch.cat([ids1, ids2], 1)
    pos = torch.cat([torch.arange(L1), torch.arange(L2)]).view(1, -1).to(dev)  # per-doc RoPE reset
    ar = lambda L: torch.arange(L, device=dev).view(1, -1)  # noqa: E731
    with torch.no_grad():
        lp = m(input_ids=ids, position_ids=pos).logits.float()
        l1 = m(input_ids=ids1, position_ids=ar(L1)).logits.float()
        l2 = m(input_ids=ids2, position_ids=ar(L2)).logits.float()
        iso = max((lp[:, :L1] - l1).abs().max().item(), (lp[:, L1:] - l2).abs().max().item())
        ids_p = ids.clone()
        ids_p[0, :L1] = torch.randint(0, V, (L1,), device=dev)
        leak = (lp[:, L1:] - m(input_ids=ids_p, position_ids=pos).logits.float()[:, L1:]).abs().max().item()
    ok = iso < TOL_FWD and leak == 0.0
    print(f"packed doc-isolation      = {iso:.4g} (tol {TOL_FWD}); cross-doc leak = {leak:.2e} (must be 0)"
          f"  {'OK' if ok else 'FAIL'}")
    return ok


def check_torch_compile(dev: str = "cuda") -> bool:
    """The FA2 sink kernel is wrapped in torch.library.custom_op, so it must be OPAQUE to Dynamo —
    torch.compile(fullgraph=True) errors on any graph break. The OPD trainer uses grad-ckpt (+compile),
    so a graph break here would be a real perf/correctness footgun."""
    from olmo3_sink.olmo3_sink_fa2 import fa2_varlen_attn_with_sink_kernel as K
    T, H, D = 96, 8, 64
    mk = lambda: torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)  # noqa: E731
    q, k, v = mk(), mk(), mk()
    sink = torch.randn(H, device=dev, dtype=torch.float32)
    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)

    @torch.compile(fullgraph=True)
    def f(q, k, v, sink, cu):
        return K(q, k, v, sink, cu, cu, T, T)

    try:
        out = f(q, k, v, sink, cu)
        ok = tuple(out.shape) == (T, H, D)
        print(f"torch.compile(fullgraph=True) = {'OK (no graph break)' if ok else 'bad shape'}")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"torch.compile FAILED (graph break): {type(e).__name__}: {str(e)[:120]}")
        return False


def check_opd_loss(dev: str = "cuda") -> bool:
    """The OPD training objective — chunked fused-linear JSD(β) against teacher hidden — must equal a
    materialized fp32 reference in BOTH loss and gradients (student-hidden + lm_head). This is the loss
    the trainer actually optimizes; validate it independently of the model."""
    from jsd_kernel import fused_linear_jsd_fp32_softmax as JSD
    torch.manual_seed(0)
    dt = torch.float32
    N, Hd, V = 48, 64, 512
    sh = torch.randn(N, Hd, device=dev, dtype=dt, requires_grad=True)
    lm = torch.randn(V, Hd, device=dev, dtype=dt, requires_grad=True)
    th = torch.randn(N, Hd, device=dev, dtype=dt)
    wr = torch.randn(V, Hd, device=dev, dtype=dt)
    labels = torch.randint(0, V, (N,), device=dev)
    labels[::7] = IGNORE
    loss = JSD(sh, lm, th, wr, labels, weight_hard_loss=0.0, weight_soft_loss=1.0, beta=1.0,
               ignore_index=IGNORE, temperature=1.0, compiled=False, chunk_size=128, compute_ce_loss=False)
    loss.backward()
    gsh, glm = sh.grad.clone(), lm.grad.clone()
    sh2 = sh.detach().clone().requires_grad_(True)
    lm2 = lm.detach().clone().requires_grad_(True)
    s_lp = torch.log_softmax(sh2 @ lm2.t(), -1)
    t_lp = torch.log_softmax(th @ wr.t(), -1)
    per = (s_lp.exp() * (s_lp - t_lp)).sum(-1)   # KL(student‖teacher) = β=1 reverse-KL
    mask = labels != IGNORE
    ref = (per * mask.float()).sum() / mask.sum().clamp_min(1)
    ref.backward()
    rel = lambda a, b: (a - b).abs().max().item() / max(1e-6, a.abs().max().item())  # noqa: E731
    lerr = abs(loss.item() - ref.item())
    gerr = max(rel(gsh, sh2.grad), rel(glm, lm2.grad))
    ok = lerr < 1e-4 and gerr < 1e-4
    print(f"OPD JSD loss vs materialized fp32 ref: lossΔ={lerr:.2e} gradΔ={gerr:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def run() -> None:
    check_deps()
    check_correction_fp64()   # the math is exact in fp64; bf16 residual below is pure flash rounding
    opd_loss_ok = check_opd_loss()   # the training objective (chunked JSD) is bit-exact vs reference
    m_eager, m_fa2 = build_pair()
    dev = "cuda"
    ids = torch.randint(0, 1024, (1, 128), device=dev)

    def fwd_and_grads(m):
        m.zero_grad(set_to_none=True)
        for p in m.parameters():
            p.requires_grad_(True)
        out = m(input_ids=ids).logits.float()
        out.sum().backward()
        g_sink = torch.cat([l.self_attn.sinks.grad.flatten() for l in m.model.layers]).float()
        g_qkv = m.model.layers[0].self_attn.q_proj.weight.grad.flatten().float()  # q/k/v path
        return out.detach(), g_sink, g_qkv

    # --- eager sink path: arch-independent, always runs (validates model + sink on this transformers) ---
    o_e, gs_e, gq_e = fwd_and_grads(m_eager)
    assert torch.isfinite(o_e).all() and torch.isfinite(gs_e).all() and torch.isfinite(gq_e).all(), \
        "eager sink produced non-finite logits/grads"
    import transformers
    print(f"eager sink OK on transformers {transformers.__version__}: "
          f"logits+sink-grad+qkv-grad all finite (|sink-grad|max={gs_e.abs().max().item():.4g})")

    # --- FA2 kernel parity: needs a flash-attn kernel built for THIS GPU arch (B200/sm_100) ---
    try:
        o_f, gs_f, gq_f = fwd_and_grads(m_fa2)
    except RuntimeError as e:
        if "no kernel image" in str(e).lower() or "kernel image is available" in str(e).lower():
            cc = torch.cuda.get_device_capability()
            print(f"SKIP FA2 parity: flash-attn has no kernel for this GPU (sm_{cc[0]}{cc[1]}); "
                  f"the prebuilt flash-attn targets Blackwell/sm_100 -> run this on a B200.")
            print("PASS (eager sink validated; FA2 kernel parity deferred to B200)")
            sys.exit(0)
        raise

    def rel(a, b):
        return (a - b).abs().max().item() / max(1e-6, a.abs().max().item())

    fwd_err = (o_e - o_f).abs().max().item()
    sink_err = rel(gs_e, gs_f)
    qkv_err = rel(gq_e, gq_f)
    print(f"forward   max|Δlogit|   = {fwd_err:.4g} (tol {TOL_FWD})")
    print(f"sink-grad rel err       = {sink_err:.4g} (tol {TOL_GRAD})")
    print(f"q/k/v-grad rel err      = {qkv_err:.4g} (tol {TOL_GRAD})   <- the 'fine for training' check")

    compile_ok = check_torch_compile()        # custom_op kernel must not graph-break under torch.compile
    pack_ok = check_packed_isolation(m_fa2)   # varlen document isolation (the real training path)
    ok = (fwd_err < TOL_FWD and sink_err < TOL_GRAD and qkv_err < TOL_GRAD
          and pack_ok and compile_ok and opd_loss_ok)
    if ok:
        print("PASS: FA2 sink matches eager (fwd + sink-grad + q/k/v-grad) + packed doc-isolation -> fine for training")
    else:
        print("FAIL: FA2 sink diverges from eager. If ONLY q/k/v-grad fails -> transformers' s_aux does a "
              "naive (detached-lse) post-correction; wrap stock FA2 with the exact OLMo-core-style correction.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    run()
