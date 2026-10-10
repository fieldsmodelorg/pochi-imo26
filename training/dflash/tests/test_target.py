#!/usr/bin/env python3
"""Target-side integration tests with real stage1-7b-4n weights (1 GPU, no dist).

Checks:
  1. olmo3_sink loads with trained sinks intact (tf5 zeroing-bug regression check)
  2. hook capture == output_hidden_states on a real L4 bin
  3. _chunked_greedy == direct lm_head argmax
  4. TargetEmbeddingsAndHead rows == checkpoint rows
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)

TARGET = os.path.join(ROOT, "outputs/stage1-7b-4n")
L4 = os.path.join(ROOT, "data/l4-g2r05-ml12288-mc65536")

PASS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    PASS.append((name, bool(ok)))
    return ok


def main():
    from data import L4Dataset
    from target_model import FSDP2TargetModel
    from target_utils import TargetEmbeddingsAndHead
    from train import _chunked_greedy
    from draft_model_olmo3 import build_target_layer_ids

    target = FSDP2TargetModel.from_pretrained(TARGET, fsdp=False)
    layer_ids = [1, 5, 9, 13, 17, 21, 25, 29]
    assert layer_ids == build_target_layer_ids(32, 8), build_target_layer_ids(32, 8)
    target.set_capture_layers(layer_ids, capture_final_norm=True)

    # 1. trained sinks survived loading (regression check for tf5 zeroing bug)
    sink_max = max(
        layer.self_attn.sinks.abs().max().item()
        for layer in target.model.model.layers
    )
    check("trained sinks loaded (nonzero)", sink_max > 0, f"max|sink|={sink_max:.2e}")

    # Real L4 bin, sliced to 16k for the parity run
    ds = L4Dataset(L4, max_bins=8)
    item = ds[1]
    S = 16384
    input_ids = item["input_ids"][:S][None].cuda()
    loss_mask = item["loss_mask"][:S][None].cuda()
    position_ids = item["position_ids"][:S][None].cuda()
    n_docs = len(set(item["document_ids"][:S].tolist()))
    print(f"bin slice: {S} tokens, {n_docs} docs, {int(loss_mask.sum())} loss tokens")

    # 2. hook capture vs output_hidden_states
    out = target.generate_hidden_states(input_ids, loss_mask, position_ids=position_ids)
    check("capture shape", out.hidden_states.shape == (1, S, len(layer_ids) * 4096),
          f"{tuple(out.hidden_states.shape)}")
    check("last_hidden shape", out.last_hidden.shape == (1, S, 4096))

    with torch.no_grad():
        ref = target.model.model(
            input_ids=input_ids, position_ids=position_ids, use_cache=False,
            output_hidden_states=True,
        )
    ref_hs = ref.hidden_states  # tuple: [embed, layer0_out, ..., layer31_out]
    check("output_hidden_states tuple len", len(ref_hs) == 33, f"{len(ref_hs)}")
    max_d = 0.0
    for i, lid in enumerate(layer_ids):
        got = out.hidden_states[..., i * 4096 : (i + 1) * 4096]
        d = (got - ref_hs[lid + 1]).abs().max().item()
        max_d = max(max_d, d)
    check("hook capture == output_hidden_states", max_d == 0.0, f"max|d|={max_d}")
    d_norm = (out.last_hidden - ref.last_hidden_state).abs().max().item()
    check("norm capture == last_hidden_state", d_norm == 0.0, f"max|d|={d_norm}")

    # 3. chunked greedy == direct argmax (on a 4k slice)
    head = TargetEmbeddingsAndHead.from_pretrained(TARGET)
    w = head.lm_head.weight.data
    lh = out.last_hidden[:, :4096]
    greedy_chunked, _ = _chunked_greedy(w, lh)
    direct = torch.nn.functional.linear(lh, w).argmax(dim=-1)
    agree = (greedy_chunked == direct).float().mean().item()
    check("chunked greedy == direct argmax", agree == 1.0, f"agree={agree:.6f}")

    # sanity: greedy should agree with the data tokens often (target was
    # SFT-trained on this mix) — predicts NEXT token, compare shifted
    sl = lh.shape[1]
    nxt = input_ids[:, 1:sl]
    g = greedy_chunked[:, : sl - 1]
    lm_next = loss_mask[:, 1:sl].bool()
    match_rate = (g == nxt)[lm_next].float().mean().item()
    check("greedy matches data on loss tokens (sanity > 0.5)", match_rate > 0.5,
          f"match={match_rate:.3f}")

    # 4. embed/lm_head rows vs checkpoint
    from safetensors import safe_open
    import glob
    st = glob.glob(os.path.join(TARGET, "*.safetensors"))[0]
    with safe_open(st, framework="pt") as f:
        emb_ref = f.get_tensor("model.embed_tokens.weight")[:8].cuda()
        head_ref = f.get_tensor("lm_head.weight")[:8].cuda()
    check("embed rows bitwise", torch.equal(head.embed_tokens.weight[:8], emb_ref.to(head.embed_tokens.weight.dtype)))
    check("lm_head rows bitwise", torch.equal(head.lm_head.weight[:8], head_ref.to(head.lm_head.weight.dtype)))

    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"\n{len(PASS) - n_fail}/{len(PASS)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
