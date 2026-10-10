"""Parity test: SGLang server logprobs vs HF olmo3_sink reference.

Teacher-forces the same token ids saved by _hf_reference.py through the SGLang
/generate endpoint (return_logprob, logprob_start_len=0) and reports per-token
logprob deltas and greedy top-1 agreement against the eager fp32-attention
reference.

Usage:
  uv run python training/stage1/sglang_deploy/_parity_test.py [--tag ref] [--port 30000]
"""

import argparse
import json
import urllib.request
from pathlib import Path

import torch

REF_DIR = Path(__file__).resolve().parent / "ref"


def post(url, payload):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ref", help="ref | ref_amp5")
    ap.add_argument("--port", type=int, default=30000)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    info = post if False else None  # noqa
    with urllib.request.urlopen(f"{base}/get_model_info", timeout=30) as r:
        print("model_info:", json.loads(r.read()))

    for f in sorted(REF_DIR.glob(f"{args.tag}_*.pt")):
        if f.name.endswith("summary.json"):
            continue
        ref = torch.load(f, weights_only=False)
        ids = ref["ids"]
        out = post(
            f"{base}/generate",
            {
                "input_ids": ids,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 1,
            },
        )
        meta = out["meta_info"]
        # input_token_logprobs: [(logprob, token_id, text?), ...]; first entry is None-logprob
        itl = meta["input_token_logprobs"]
        srv_lp, srv_ok = [], []
        for entry in itl:
            lp = entry[0]
            if lp is not None:
                srv_lp.append(lp)
        srv_lp = torch.tensor(srv_lp)
        ref_lp = ref["token_logprobs"]
        assert len(srv_lp) == len(ref_lp), f"{len(srv_lp)} vs {len(ref_lp)}"
        d = (srv_lp - ref_lp.float()).abs()

        # greedy top-1 agreement vs reference argmax
        top1_match = None
        if meta.get("input_top_logprobs"):
            srv_top1 = [e[0][1] for e in meta["input_top_logprobs"] if e]
            ref_top1 = ref["top1"].tolist()
            n = min(len(srv_top1), len(ref_top1))
            top1_match = sum(a == b for a, b in zip(srv_top1[-n:], ref_top1[-n:])) / n

        case = f.stem.replace(f"{args.tag}_", "")
        print(
            f"[{args.tag}/{case}] n={len(ref_lp)} "
            f"mean|dlp|={d.mean():.5f} p99|dlp|={d.quantile(0.99):.5f} max|dlp|={d.max():.5f} "
            f"mean_lp ref={ref_lp.mean():.5f} srv={srv_lp.mean():.5f}"
            + (f" top1_agree={top1_match:.4f}" if top1_match is not None else "")
        )


if __name__ == "__main__":
    main()
