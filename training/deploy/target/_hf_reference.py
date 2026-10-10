"""HF reference logprobs for the SGLang sink-deployment parity test.

Loads the ORIGINAL stage1-2node checkpoint (model_type=olmo3_sink) through the
olmo3_sink package with the sink-aware eager backend (fp32 reference attention),
teacher-forces fixed DeepSeek-format token sequences, and saves per-position
next-token logprobs as the parity baseline for the SGLang server.

--amplify S: scale all attention sinks to the constant S before the forward
pass (engagement test: proves the serving kernel actually applies sinks).

Usage:
  CUDA_VISIBLE_DEVICES=7 uv run python training/stage1/sglang_deploy/_hf_reference.py
  CUDA_VISIBLE_DEVICES=7 uv run python training/stage1/sglang_deploy/_hf_reference.py --amplify 5.0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]  # proof-pilot/ (file lives in deploy/target/)
sys.path.insert(0, str(ROOT / "training" / "stage1" / "src"))
sys.path.insert(0, str(ROOT / "olmo3_sink"))

from encoding_dsv4 import encode_messages  # noqa: E402

# Default 7B stage1; override with PP_REF_MODEL to build a reference for another
# olmo3_sink checkpoint (e.g. the 32B target, for its gptq serving-parity test).
MODEL = Path(os.environ.get("PP_REF_MODEL", str(ROOT / "outputs" / "stage1-2node")))
# ref/ tag derives from the model dir so 7B/32B references don't collide.
OUT_DIR = Path(__file__).resolve().parent / "ref"

LEMMA = (
    "Lemma: for every positive integer n, the sum of the first n odd numbers "
    "equals n squared. Proof sketch: induction on n; the base case n=1 gives 1=1, "
    "and the step adds the (n+1)-th odd number 2n+1 to n^2, giving (n+1)^2. "
)


def build_cases():
    """Fixed DeepSeek-format conversations rendered by the training ground truth."""
    cases = {}

    # 1) short math proof conversation, teacher-forced through the assistant turn
    cases["short_math"] = encode_messages(
        [
            {"role": "user", "content": "Prove that the sum of the first n odd positive integers equals n^2."},
            {
                "role": "assistant",
                "reasoning_content": "We can use induction. Base case n=1: the sum is 1 = 1^2. "
                "Inductive step: assume 1+3+...+(2n-1) = n^2. Adding the next odd number 2n+1 "
                "gives n^2 + 2n + 1 = (n+1)^2, completing the induction.",
                "content": "**Claim.** For every positive integer $n$, $\\sum_{k=1}^{n}(2k-1)=n^2$.\n\n"
                "**Proof.** We argue by induction on $n$. For $n=1$ the sum is $1=1^2$. "
                "Suppose the claim holds for $n$. Then adding the $(n+1)$-th odd number $2n+1$ yields "
                "$n^2+2n+1=(n+1)^2$. By induction the claim holds for all $n$. $\\blacksquare$",
            },
        ],
        thinking_mode="thinking",
        drop_thinking=False,
    )

    # 2) long prompt (>4096 tokens) so the sliding-window layers actually slide
    cases["long_swa"] = encode_messages(
        [
            {"role": "user", "content": "Here is a lemma repeated many times:\n" + LEMMA * 90
             + "\nState the lemma once, concisely."},
            {
                "role": "assistant",
                "reasoning_content": "The user pasted the same lemma many times; I just restate it once.",
                "content": "Lemma: for every positive integer n, the sum of the first n odd numbers equals n^2.",
            },
        ],
        thinking_mode="thinking",
        drop_thinking=False,
    )
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amplify", type=float, default=None,
                    help="overwrite all sinks with this constant before forward")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from olmo3_sink import register_olmo3_sink

    register_olmo3_sink()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda"
    )
    model.eval()

    sink_vals = []
    with torch.no_grad():
        for layer in model.model.layers:
            if args.amplify is not None:
                layer.self_attn.sinks.fill_(args.amplify)
            sink_vals.append(layer.self_attn.sinks.abs().max().item())
    print(f"sinks: n_layers={len(sink_vals)} max|v|={max(sink_vals):.6f}")

    # PP_REF_TAG keeps per-model references from colliding (7B "ref" vs 32B "ref32b").
    base_tag = os.environ.get("PP_REF_TAG", "ref")
    tag = base_tag if args.amplify is None else f"{base_tag}_amp{args.amplify:g}"
    OUT_DIR.mkdir(exist_ok=True)
    summary = {}
    for name, text in build_cases().items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        t = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            logits = model(t).logits.float()  # [1, L, V]
        logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
        nxt = t[0, 1:]
        tok_lp = logprobs.gather(-1, nxt[:, None]).squeeze(-1)  # logprob of actual next token
        top1 = logprobs.argmax(-1)  # greedy next-token prediction per position
        torch.save(
            {"ids": ids, "token_logprobs": tok_lp.cpu(), "top1": top1.cpu(), "text": text},
            OUT_DIR / f"{tag}_{name}.pt",
        )
        summary[name] = {
            "n_tokens": len(ids),
            "mean_logprob": tok_lp.mean().item(),
            "greedy_match_rate": (top1 == nxt).float().mean().item(),
        }
        print(f"{name}: {summary[name]}")

    (OUT_DIR / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print("saved ->", OUT_DIR / f"{tag}_*.pt")


if __name__ == "__main__":
    main()
