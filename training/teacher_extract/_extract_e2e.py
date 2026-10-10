# Copyright 2026 proof-pilot. Apache-2.0.
"""E2E extraction smoke: docs json -> sglang spool -> had+int6 shard.

Runs in the sglang container (run_in_container.sh). Per doc:
  spool pieces (bf16 [*,4096]) -> concat -> verify length -> select target rows
  -> hidden_codec.encode (rotate + int6 blk32 + pack) -> shard .pt

doc0 additionally keeps its raw bf16 target rows ("verify_h") so the training-side
smoke can check int6-reconstructed teacher logits against the exact reference.

  CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_in_container.sh _extract_e2e.py --tp 4
"""
import argparse
import json
import os
import sys
import time

SPOOL = "/tmp/hidden_spool_e2e"
os.environ["SGLANG_DSV4_HIDDEN_POST_NORM"] = "1"
os.environ["SGLANG_HIDDEN_SPOOL_DIR"] = SPOOL
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_FAST_WARMUP", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

import torch

MODEL = os.environ.get("DEEPSEEK_V4_FLASH", "/models/DeepSeek-V4-Flash")
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_docs_e2e.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shard_e2e.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(SPOOL, exist_ok=True)

    docs = json.load(open(DOCS))
    n_tok = sum(len(d["input_ids"]) for d in docs)
    print(f"{len(docs)} docs, {n_tok:,} tokens", flush=True)

    import sglang as sgl
    llm = sgl.Engine(
        model_path=MODEL,
        tp_size=a.tp,
        enable_return_hidden_states=True,
        disable_radix_cache=True,
        chunked_prefill_size=11264,    # flash_mla sched-meta smem cap (see README #4/#8)
        mem_fraction_static=0.80,
        max_running_requests=128,
        disable_cuda_graph=True,
        context_length=32768,
        moe_runner_backend="marlin",
        watchdog_timeout=1800,
        log_level="warning",
    )
    t0 = time.perf_counter()
    outs = llm.generate(
        input_ids=[d["input_ids"] for d in docs],
        sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        return_hidden_states=True,
    )
    dt = time.perf_counter() - t0
    print(f"extracted {n_tok:,} tok in {dt:.1f}s = {n_tok/dt:,.0f} tok/s", flush=True)
    llm.shutdown()
    torch.cuda.empty_cache()

    from hidden_codec import Rotator, encode
    rot = Rotator(device="cuda")
    shard = {"format": "had+int6_blk32", "rot_seed": 7, "docs": []}
    for j, (doc, out) in enumerate(zip(docs, outs)):
        paths = out["meta_info"]["hidden_states"]
        assert all(isinstance(p, str) for p in paths), "spool mode not active?"
        h = torch.cat([torch.load(p, weights_only=True) for p in paths], 0)
        m = len(doc["input_ids"])
        assert h.shape[0] in (m, m + 1), (doc["id"], h.shape[0], m)
        pos = torch.tensor(doc["positions"], dtype=torch.int32)
        h_sel = h[pos.long()].cuda()
        packed, scales = encode(h_sel, rot)
        entry = {
            "id": doc["id"],
            "input_ids": torch.tensor(doc["input_ids"], dtype=torch.int32),
            "positions": pos,
            "targets": torch.tensor(doc["targets"], dtype=torch.int32),
            "packed": packed.cpu(),
            "scales": scales.cpu(),
        }
        if j == 0:
            entry["verify_h"] = h_sel.cpu()   # bf16 reference rows for the train-side check
        shard["docs"].append(entry)
        print(f"  doc{j} {doc['id'][:13]}: {m} tok -> {len(pos)} target rows "
              f"({packed.numel() + scales.numel()*2:,} B)", flush=True)

    torch.save(shard, OUT)
    gb = os.path.getsize(OUT) / 1e9
    print(f"shard -> {OUT} ({gb:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
