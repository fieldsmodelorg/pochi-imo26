"""GPU-only numerical check of SGLang's paged FA4 sink/window wrapper."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

import torch
from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func


@torch.inference_mode()
def check_case(query_tokens, key_tokens, query_heads, kv_heads, window, batch=1):
    device, dtype, dim, page_size = "cuda", torch.bfloat16, 128, 128
    pages = math.ceil(key_tokens / page_size)
    q = torch.randn(batch * query_tokens, query_heads, dim, device=device, dtype=dtype)
    shape = (batch * pages, page_size, kv_heads, dim)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    table = torch.randperm(batch * pages, device=device, dtype=torch.int32).reshape(batch, pages)
    sink = torch.linspace(-1, 8, query_heads, device=device, dtype=dtype)
    cu_q = torch.arange(0, (batch + 1) * query_tokens, query_tokens, device=device, dtype=torch.int32)
    used_k = torch.full((batch,), key_tokens, device=device, dtype=torch.int32)
    actual = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_q, seqused_k=used_k,
        max_seqlen_q=query_tokens, max_seqlen_k=key_tokens,
        page_table=table, softmax_scale=dim ** -0.5, causal=True,
        window_size=(window, 0), sinks=sink,
    )
    if isinstance(actual, tuple):
        actual = actual[0]
    reference = torch.empty_like(q, dtype=torch.float32)
    groups = query_heads // kv_heads
    for b in range(batch):
        kb = k[table[b].long()].flatten(0, 1)[:key_tokens].float()
        vb = v[table[b].long()].flatten(0, 1)[:key_tokens].float()
        for start in range(0, query_tokens, 32):
            end = min(start + 32, query_tokens)
            queries = q[b * query_tokens + start:b * query_tokens + end]
            queries = queries.float().reshape(end - start, kv_heads, groups, dim)
            logits = torch.einsum("qhgd,khd->hgqk", queries, kb) * dim ** -0.5
            qpos = torch.arange(start, end, device=device) + key_tokens - query_tokens
            kpos = torch.arange(key_tokens, device=device)
            allowed = kpos[None, :] <= qpos[:, None]
            if window >= 0:
                allowed &= kpos[None, :] >= qpos[:, None] - window
            logits.masked_fill_(~allowed[None, None, :, :], -torch.inf)
            sink_logits = sink.float().reshape(kv_heads, groups, 1, 1).expand(-1, -1, end - start, 1)
            weights = torch.softmax(torch.cat([logits, sink_logits], dim=-1), dim=-1)[..., :-1]
            output = torch.einsum("hgqk,khd->qhgd", weights, vb)
            reference[b * query_tokens + start:b * query_tokens + end] = output.reshape(-1, query_heads, dim)
    torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    error = actual.float() - reference
    return {
        "query_tokens": query_tokens, "key_tokens": key_tokens,
        "query_heads": query_heads, "kv_heads": kv_heads, "batch": batch,
        "window_left": window, "max_abs_error": error.abs().max().item(),
        "rmse": error.square().mean().sqrt().item(), "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "fa4": importlib.metadata.version("flash-attn-4"), "cases": [],
    }
    cases = [
        (1, 128, 20, 4, -1, 2), (8, 513, 20, 4, 511, 2),
        (128, 768, 20, 4, 511, 2), (2048, 8192, 20, 4, 4095, 1),
        (8, 1537, 40, 8, 511, 2), (1, 262144, 40, 8, -1, 1),
        (8, 262144, 40, 8, 4095, 1),
    ]
    try:
        for case in cases:
            result = check_case(*case)
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc), failed_case=case)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
