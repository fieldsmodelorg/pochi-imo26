#!/usr/bin/env python3
"""DFlash unit tests (1 GPU). Run:
    PYTHONPATH=$ROOT:$ROOT/training/dflash uv run python training/dflash/tests/test_units.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)

from transformers import AutoConfig

from dflash import OnlineDFlashModel, create_dflash_block_mask
from draft_model_olmo3 import Olmo3DFlashDraftModel, apply_rotary_pos_emb
from data import L4Dataset, L4StripeSampler

PASS = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}", flush=True)
    PASS.append((name, bool(ok)))
    return ok


def dense_reference_mask(anchors, keep, S, bs, doc_ids=None, sliding_window=None, causal=False):
    """O(Q*KV) numpy reference of the 6 mask rules."""
    B, N = anchors.shape
    Q, KV = N * bs, S + N * bs
    out = np.zeros((B, Q, KV), dtype=bool)
    for b in range(B):
        for q in range(Q):
            qb, k_in = q // bs, q % bs
            a = int(anchors[b, qb])
            if not keep[b, qb]:
                continue
            for kv in range(KV):
                if kv < S:
                    ok = kv < a
                    if sliding_window is not None:
                        ok = ok and kv >= (a + k_in) - sliding_window
                    if doc_ids is not None:
                        ok = ok and doc_ids[b, min(a, S - 1)] == doc_ids[b, kv]
                    out[b, q, kv] = ok
                else:
                    kvb, kv_in = (kv - S) // bs, (kv - S) % bs
                    ok = kvb == qb
                    if causal:
                        ok = ok and kv_in <= k_in
                    out[b, q, kv] = ok
    return out


def test_block_mask():
    """Rules 1-6: flex_attention output with our BlockMask must equal eager
    attention under the dense numpy reference mask (token-exact semantics,
    including the block-sparsity layer)."""
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(0)
    S, bs, N = 256, 4, 8
    dev = "cuda"
    anchors = torch.sort(torch.randint(8, S - bs, (1, N), device=dev), dim=1).values
    keep = torch.ones(1, N, dtype=torch.bool, device=dev)
    keep[0, -2:] = False  # two invalid blocks
    anchors[0, -2:] = 0
    doc_ids = (torch.arange(S, device=dev) // 100).unsqueeze(0)

    Q_LEN, KV_LEN = N * bs, S + N * bs
    H, D = 2, 16
    q = torch.randn(1, H, Q_LEN, D, device=dev)
    k = torch.randn(1, H, KV_LEN, D, device=dev)
    v = torch.randn(1, H, KV_LEN, D, device=dev)

    for name, kwargs in [
        ("full", {}),
        ("swa64", {"sliding_window": 64}),
        ("causal", {"causal": True}),
        ("swa64+causal", {"sliding_window": 64, "causal": True}),
    ]:
        bm = create_dflash_block_mask(
            anchors, keep, S, bs, dev, context_doc_ids=doc_ids, **kwargs
        )
        out = flex_attention(q, k, v, block_mask=bm)

        ref_mask = dense_reference_mask(
            anchors.cpu().numpy(), keep.cpu().numpy(), S, bs,
            doc_ids=doc_ids.cpu().numpy(),
            sliding_window=kwargs.get("sliding_window"),
            causal=kwargs.get("causal", False),
        )
        bias = torch.where(
            torch.from_numpy(ref_mask).to(dev).unsqueeze(1),
            torch.zeros(1, device=dev),
            torch.full((1,), float("-inf"), device=dev),
        )
        scores = (q @ k.transpose(-1, -2)) * D**-0.5 + bias
        ref = torch.softmax(scores, dim=-1).nan_to_num(0.0) @ v

        d = (out - ref).abs().max().item()
        check(f"mask[{name}] flex == eager-with-reference", d < 1e-5, f"max|d|={d:.2e}")


def tiny_draft_config(use_sink=True, target_hidden=None, swa=32):
    cfg = AutoConfig.for_model(
        "olmo3",
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        vocab_size=512,
        max_position_embeddings=4096,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=swa,
        rope_parameters={
            "attention_factor": 1.2079441541679836,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 8.0,
            "original_max_position_embeddings": 512,
            "rope_theta": 500000,
            "rope_type": "yarn",
        },
        block_size=4,
        num_target_layers=8,
        target_hidden_size=target_hidden or 512,
        pad_token_id=2,
    )
    cfg.dflash_config = {
        "mask_token_id": 500,
        "target_layer_ids": [1, 3, 5, 7],
        "use_attention_sink": use_sink,
    }
    cfg._attn_implementation = "flex_attention"
    return cfg


def test_draft_forward_backward():
    """End-to-end OnlineDFlashModel: loss finite, grads flow everywhere."""
    torch.manual_seed(0)
    dev = "cuda"
    cfg = tiny_draft_config()
    draft = Olmo3DFlashDraftModel(cfg).to(dev, torch.bfloat16)

    vocab, hidden = cfg.vocab_size, 512
    embed = torch.nn.Embedding(vocab, hidden).to(dev, torch.bfloat16).requires_grad_(False)
    lm_head = torch.nn.Linear(hidden, vocab, bias=False).to(dev, torch.bfloat16).requires_grad_(False)
    with torch.no_grad():
        draft.mask_embed.data.copy_(embed.weight.mean(0))

    model = OnlineDFlashModel(
        draft_model=draft, target_lm_head=lm_head, target_embed_tokens=embed,
        mask_token_id=500, block_size=cfg.block_size, num_anchors=32,
        loss_decay_gamma=2.0, sliding_window=cfg.sliding_window,
    ).to(dev)

    B, S = 1, 512
    input_ids = torch.randint(0, 400, (B, S), device=dev)
    loss_mask = torch.zeros(B, S, dtype=torch.long, device=dev)
    loss_mask[:, 50:200] = 1
    loss_mask[:, 300:450] = 1
    doc_ids = (torch.arange(S, device=dev) // 256).unsqueeze(0)
    pos_ids = (torch.arange(S, device=dev) % 256).unsqueeze(0)
    target_hidden = torch.randn(B, S, 4 * 512, device=dev, dtype=torch.bfloat16)
    greedy = torch.randint(0, 400, (B, S), device=dev)
    # make ~70% of greedy match the data so prefix filtering passes some tokens
    m = torch.rand(B, S, device=dev) < 0.7
    greedy[m] = input_ids[m]

    loss, metrics = model(
        input_ids=input_ids, hidden_states=target_hidden, loss_mask=loss_mask,
        compute_accuracy=True, document_ids=doc_ids, context_position_ids=pos_ids,
        greedy_tokens=greedy,
    )
    loss.backward()
    check("draft loss finite", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    check("metrics present", metrics is not None and "greedy/mean_prefix_len" in metrics,
          f"prefix_len={metrics['greedy/mean_prefix_len']:.2f} match={metrics['greedy/match_rate']:.2f}")

    grads = {n: p.grad for n, p in draft.named_parameters()}
    no_grad = [n for n, g in grads.items() if g is None or not g.abs().sum() > 0]
    check("all draft params get grads", len(no_grad) == 0, f"missing={no_grad[:5]}")
    sink_g = [g.abs().sum().item() for n, g in grads.items() if "sinks" in n]
    check("sink grads nonzero", all(g > 0 for g in sink_g), f"n={len(sink_g)}")
    check("mask_embed grad nonzero", grads["mask_embed"].abs().sum().item() > 0)
    check("fc grad nonzero", grads["fc.weight"].abs().sum().item() > 0)


def test_greedy_prefix_semantics():
    """Hand-checkable greedy-prefix case: mismatch at k=2 -> loss at k=1,2 only;
    labels replaced by greedy."""
    dev = "cuda"
    cfg = tiny_draft_config(use_sink=False)
    draft = Olmo3DFlashDraftModel(cfg).to(dev, torch.bfloat16)
    vocab, hidden = cfg.vocab_size, 512
    embed = torch.nn.Embedding(vocab, hidden).to(dev, torch.bfloat16).requires_grad_(False)
    lm_head = torch.nn.Linear(hidden, vocab, bias=False).to(dev, torch.bfloat16).requires_grad_(False)
    with torch.no_grad():
        draft.mask_embed.data.copy_(embed.weight.mean(0))

    bs = cfg.block_size  # 4
    model = OnlineDFlashModel(
        draft_model=draft, target_lm_head=lm_head, target_embed_tokens=embed,
        mask_token_id=500, block_size=bs, num_anchors=32,
        sliding_window=None,
    ).to(dev)

    S = 64
    input_ids = torch.arange(1, S + 1, device=dev).unsqueeze(0) % 400
    loss_mask = torch.ones(1, S, dtype=torch.long, device=dev)
    target_hidden = torch.randn(1, S, 4 * 512, device=dev, dtype=torch.bfloat16)
    # greedy at pos p predicts input_ids[p+1] everywhere EXCEPT at one position
    greedy = torch.roll(input_ids, -1, dims=1)
    # Anchor will be sampled randomly; force determinism by monkeypatching
    anchors = torch.zeros(1, 32, dtype=torch.long, device=dev)
    anchors[0, 0], anchors[0, 1] = 10, 30
    keep = torch.zeros(1, 32, dtype=torch.bool, device=dev)
    keep[0, :2] = True
    model._sample_anchor_positions = lambda *a, **k: (anchors, keep)
    # break greedy at position anchor+1 (=11): the label check for block pos k=2
    # is greedy[anchor+1] vs data[anchor+2]
    greedy[0, 11] = 399

    loss, metrics = model(
        input_ids=input_ids, hidden_states=target_hidden, loss_mask=loss_mask,
        compute_accuracy=True, greedy_tokens=greedy,
    )
    # block 0 (anchor 10): k=1 trains (greedy match), k=2 trains (first mismatch,
    # corrected label 399), k=3 dropped. block 1 (anchor 30): k=1..3 all train.
    r1 = metrics["train_ratio/pos_1"].item()  # both blocks -> 2/2
    r2 = metrics["train_ratio/pos_2"].item()  # both blocks (one corrected) -> 2/2
    r3 = metrics["train_ratio/pos_3"].item()  # only block 1 -> 1/2
    ok = abs(r1 - 1.0) < 1e-3 and abs(r2 - 1.0) < 1e-3 and abs(r3 - 0.5) < 1e-3
    check("greedy prefix cut at first mismatch", ok, f"ratios={r1:.2f},{r2:.2f},{r3:.2f}")
    check("mean_prefix_len == 2.5", abs(metrics["greedy/mean_prefix_len"].item() - 2.5) < 1e-3,
          f"={metrics['greedy/mean_prefix_len']:.2f}")


def test_cross_doc_label_mask():
    """Anchor near a doc boundary: labels crossing into the next doc are dropped."""
    dev = "cuda"
    cfg = tiny_draft_config(use_sink=False)
    draft = Olmo3DFlashDraftModel(cfg).to(dev, torch.bfloat16)
    vocab, hidden = cfg.vocab_size, 512
    embed = torch.nn.Embedding(vocab, hidden).to(dev, torch.bfloat16).requires_grad_(False)
    lm_head = torch.nn.Linear(hidden, vocab, bias=False).to(dev, torch.bfloat16).requires_grad_(False)
    with torch.no_grad():
        draft.mask_embed.data.copy_(embed.weight.mean(0))

    bs = cfg.block_size
    model = OnlineDFlashModel(
        draft_model=draft, target_lm_head=lm_head, target_embed_tokens=embed,
        mask_token_id=500, block_size=bs, num_anchors=32, sliding_window=None,
    ).to(dev)

    S = 64
    input_ids = torch.randint(0, 400, (1, S), device=dev)
    loss_mask = torch.ones(1, S, dtype=torch.long, device=dev)
    doc_ids = torch.zeros(1, S, dtype=torch.long, device=dev)
    doc_ids[0, 32:] = 1  # boundary at 32
    target_hidden = torch.randn(1, S, 4 * 512, device=dev, dtype=torch.bfloat16)

    anchors = torch.zeros(1, 32, dtype=torch.long, device=dev)
    anchors[0, 0] = 30  # labels at 31,32,33 -> 32,33 cross
    keep = torch.zeros(1, 32, dtype=torch.bool, device=dev)
    keep[0, 0] = True
    model._sample_anchor_positions = lambda *a, **k: (anchors, keep)

    _, metrics = model(
        input_ids=input_ids, hidden_states=target_hidden, loss_mask=loss_mask,
        compute_accuracy=True, document_ids=doc_ids,
    )
    r1 = metrics["train_ratio/pos_1"].item()  # label at 31, same doc -> 1
    r2 = metrics["train_ratio/pos_2"].item()  # label at 32, next doc -> 0
    r3 = metrics["train_ratio/pos_3"].item()
    check("cross-doc labels masked",
          abs(r1 - 1.0) < 1e-3 and r2 == 0.0 and r3 == 0.0,
          f"ratios={r1:.4f},{r2},{r3}")


def test_rope_parity():
    """Draft per-layer rotary == olmo3_sink rotary on identical positions."""
    from olmo3_sink import register_olmo3_sink
    register_olmo3_sink()
    from olmo3_sink.modeling_olmo3_sink import Olmo3SinkRotaryEmbedding
    from olmo3_sink import Olmo3SinkConfig
    from draft_model_olmo3 import Olmo3DFlashRotaryEmbedding

    rp = {
        "attention_factor": 1.2079441541679836, "beta_fast": 32.0, "beta_slow": 1.0,
        "factor": 8.0, "original_max_position_embeddings": 8192,
        "rope_theta": 500000, "rope_type": "yarn",
    }
    sink_cfg = Olmo3SinkConfig(
        hidden_size=4096, num_attention_heads=32, num_hidden_layers=2,
        max_position_embeddings=65536, rope_parameters=dict(rp),
        layer_types=["sliding_attention", "full_attention"], sliding_window=4096,
    )
    draft_cfg = tiny_draft_config()
    draft_cfg.hidden_size = 4096
    draft_cfg.num_attention_heads = 32
    draft_cfg.head_dim = 128
    draft_cfg.max_position_embeddings = 65536
    draft_cfg.rope_parameters = dict(rp)

    x = torch.randn(1, 8, 4096, device="cuda")
    pos = torch.tensor([[0, 1, 5, 100, 8191, 8192, 20000, 65535]], device="cuda")
    for ltype, rope_type in [("sliding_attention", "default"), ("full_attention", None)]:
        ref = Olmo3SinkRotaryEmbedding(sink_cfg, device="cuda", rope_type=rope_type)
        got = Olmo3DFlashRotaryEmbedding(draft_cfg, device="cuda", rope_type=rope_type)
        rc, rs = ref(x, pos)
        gc, gs = got(x, pos)
        ok = torch.equal(rc, gc) and torch.equal(rs, gs)
        check(f"rope parity [{ltype}]", ok,
              f"max|dcos|={(rc-gc).abs().max().item():.2e}")

    # q shorter than k slicing
    cos = torch.randn(1, 8, 128)
    sin = torch.randn(1, 8, 128)
    q = torch.randn(1, 4, 3, 128)
    k = torch.randn(1, 2, 8, 128)
    qe, ke = apply_rotary_pos_emb(q, k, cos, sin)
    q_full = torch.cat([torch.zeros(1, 4, 5, 128), q], dim=2)
    qe_full, _ = apply_rotary_pos_emb(q_full, k, cos, sin)
    check("rope q-slice == last positions", torch.allclose(qe, qe_full[:, :, 5:], atol=1e-6))


def test_l4_reader():
    root = os.path.join(ROOT, "data/l4-g2r05-ml12288-mc65536")
    ds = L4Dataset(root, max_bins=64)
    item = ds[3]
    L = ds.micro_len
    ok_shapes = all(item[k].shape == (L,) for k in
                    ["input_ids", "loss_mask", "document_ids", "position_ids", "attention_mask"])
    check("L4 item shapes", ok_shapes)

    # positions reset exactly at doc starts
    pos, doc = item["position_ids"].numpy(), item["document_ids"].numpy()
    starts = np.flatnonzero(np.diff(np.concatenate([[-99], doc])) != 0)
    check("L4 positions reset per doc", bool((pos[starts] == 0).all()),
          f"docs={len(starts)}")
    # loss only inside non-pad docs
    lm = item["loss_mask"].numpy()
    check("L4 no loss on pad", bool((lm[doc == -1] == 0).all()))
    check("L4 loss tokens exist", int(lm.sum()) > 0, f"n={int(lm.sum())}")
    # mask token absent (spot)
    check("L4 no mask token (spot)", int((item["input_ids"] == 128000).sum()) == 0)

    # striping: 2 ranks, disjoint equal-size cover
    s0 = list(L4StripeSampler(64, 0, 2, seed=1))
    s1 = list(L4StripeSampler(64, 1, 2, seed=1))
    check("L4 stripe disjoint+equal", len(s0) == len(s1) == 32 and not set(s0) & set(s1))


def test_cce_vs_full_ce():
    torch.manual_seed(0)
    from cut_cross_entropy import linear_cross_entropy
    h = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    t = torch.randint(0, 512, (256,), device="cuda")
    t[::5] = -100
    l1 = linear_cross_entropy(h, w, t, reduction="none")
    logits = F.linear(h.float(), w.float())
    l2 = F.cross_entropy(logits, t.clamp(min=0), reduction="none")
    valid = t != -100
    d = (l1.float()[valid] - l2[valid]).abs().max().item()
    check("cce ~= full CE (bf16 tol)", d < 0.1, f"max|d|={d:.4f}")
    check("cce ignored targets = 0", bool((l1[~valid] == 0).all().item()))


def test_save_load_roundtrip():
    """save_pretrained -> from_pretrained must be value-exact for EVERY param.
    (Catches the tf5 _init_weights re-randomization class of bug.)"""
    import tempfile
    torch.manual_seed(7)
    cfg = tiny_draft_config()
    draft = Olmo3DFlashDraftModel(cfg).to(torch.bfloat16)
    with torch.no_grad():  # make every param distinctive
        for p in draft.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    with tempfile.TemporaryDirectory() as d:
        draft.save_pretrained(d)
        loaded = Olmo3DFlashDraftModel.from_pretrained(d, dtype=torch.bfloat16)
    sd0, sd1 = draft.state_dict(), loaded.state_dict()
    bad = [k for k in sd0 if not torch.equal(sd0[k], sd1[k])]
    check("save/load roundtrip value-exact", len(bad) == 0,
          f"mismatched={bad[:5]} ({len(bad)}/{len(sd0)})")


if __name__ == "__main__":
    test_save_load_roundtrip()
    test_block_mask()
    test_draft_forward_backward()
    test_greedy_prefix_semantics()
    test_cross_doc_label_mask()
    test_rope_parity()
    test_l4_reader()
    test_cce_vs_full_ce()
    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"\n{len(PASS) - n_fail}/{len(PASS)} passed")
    sys.exit(1 if n_fail else 0)
