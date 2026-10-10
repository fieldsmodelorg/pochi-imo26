# Loss-head microbench: Liger CE vs Liger JSD vs repo chunk-JSD, at long context.
# Measures fwd+bwd time and peak HBM for student_hidden[BT,H] -> loss, sweeping BT up to 262144.
import os
import os, sys, time, math, json, gc
import torch
import torch.nn.functional as F

THIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_common"))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from jsd_kernel import fused_linear_jsd_fp32_softmax  # repo chunk-JSD
from liger_kernel.transformers.functional import (
    liger_fused_linear_cross_entropy as liger_ce,
    liger_fused_linear_jsd as liger_jsd,
)

DEV = "cuda"
H = 4096
V = 129280
DT = torch.bfloat16
IGNORE = -100
CHUNK = 4096
BETA = 0.0          # forward-KL (production)
TEMP = 1.0
WARMUP = 1
ITERS = 3
BTS = [32768, 65536, 131072, 262144]

g = torch.Generator(device=DEV).manual_seed(0)

def make_inputs(BT):
    s_in = torch.randn(BT, H, device=DEV, dtype=DT, generator=g, requires_grad=True)
    s_w  = (torch.randn(V, H, device=DEV, dtype=DT, generator=g) * 0.02).requires_grad_(True)
    t_in = torch.randn(BT, H, device=DEV, dtype=DT, generator=g)
    t_w  = (torch.randn(V, H, device=DEV, dtype=DT, generator=g) * 0.02)
    labels = torch.randint(0, V, (BT,), device=DEV, generator=g)
    return s_in, s_w, t_in, t_w, labels

def run_ce(s_in, s_w, t_in, t_w, labels):
    return liger_ce(s_in, s_w, labels, ignore_index=IGNORE, reduction="mean")

def run_liger_jsd(s_in, s_w, t_in, t_w, labels):
    return liger_jsd(s_in, s_w, t_in, t_w, shift_labels=labels, jsd_beta=BETA,
                     ignore_index=IGNORE, temperature=TEMP)

def run_chunk_jsd_soft(s_in, s_w, t_in, t_w, labels):
    return fused_linear_jsd_fp32_softmax(s_in, s_w, t_in, t_w, labels,
        weight_hard_loss=0.0, weight_soft_loss=1.0, beta=BETA, ignore_index=IGNORE,
        temperature=TEMP, compiled=False, chunk_size=CHUNK, compute_ce_loss=False)

def run_chunk_jsd_prod(s_in, s_w, t_in, t_w, labels):
    return fused_linear_jsd_fp32_softmax(s_in, s_w, t_in, t_w, labels,
        weight_hard_loss=0.5, weight_soft_loss=0.5, beta=BETA, ignore_index=IGNORE,
        temperature=TEMP, compiled=False, chunk_size=CHUNK, compute_ce_loss=True)

_compiled_fn = {}
def run_chunk_jsd_compiled(s_in, s_w, t_in, t_w, labels):
    return fused_linear_jsd_fp32_softmax(s_in, s_w, t_in, t_w, labels,
        weight_hard_loss=0.0, weight_soft_loss=1.0, beta=BETA, ignore_index=IGNORE,
        temperature=TEMP, compiled=True, chunk_size=CHUNK, compute_ce_loss=False)

CONFIGS = [
    ("liger_ce",          run_ce),
    ("liger_jsd",         run_liger_jsd),
    ("chunk_jsd_soft",    run_chunk_jsd_soft),
    ("chunk_jsd_prod",    run_chunk_jsd_prod),
    ("chunk_jsd_compiled",run_chunk_jsd_compiled),
]

def bench_one(name, fn, BT):
    gc.collect(); torch.cuda.empty_cache()
    try:
        s_in, s_w, t_in, t_w, labels = make_inputs(BT)
    except RuntimeError as e:
        return {"name": name, "BT": BT, "error": f"input-alloc OOM: {str(e)[:60]}"}
    inputs_gb = (s_in.numel()+t_in.numel())*2 + (s_w.numel()+t_w.numel())*2
    inputs_gb = inputs_gb/1e9
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    def step():
        if s_in.grad is not None: s_in.grad = None
        if s_w.grad is not None: s_w.grad = None
        loss = fn(s_in, s_w, t_in, t_w, labels)
        loss.backward()
        return loss
    try:
        for _ in range(WARMUP):
            l = step()
        torch.cuda.synchronize()
        ts = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            l = step()
            torch.cuda.synchronize()
            ts.append(time.perf_counter()-t0)
    except RuntimeError as e:
        del s_in, s_w, t_in, t_w, labels
        gc.collect(); torch.cuda.empty_cache()
        return {"name": name, "BT": BT, "error": f"OOM/err: {str(e)[:80]}"}
    ts.sort(); med = ts[len(ts)//2]
    peak = torch.cuda.max_memory_allocated()/1e9
    loss_val = float(l.detach().item())
    del s_in, s_w, t_in, t_w, labels, l
    gc.collect(); torch.cuda.empty_cache()
    return {"name": name, "BT": BT, "ms": round(med*1000,1),
            "tok_per_s": round(BT/med), "peak_gb": round(peak,2),
            "inputs_gb": round(inputs_gb,2), "loss": round(loss_val,4)}

print(f"# H={H} V={V} dtype={DT} chunk={CHUNK} beta={BETA} T={TEMP} warmup={WARMUP} iters={ITERS}")
print(f"# gpu: {torch.cuda.get_device_name(0)}  total={torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB")
rows=[]
for BT in BTS:
    for name, fn in CONFIGS:
        r = bench_one(name, fn, BT)
        rows.append(r)
        if "error" in r:
            print(f"BT={BT:>7} {name:20s} ERROR: {r['error']}")
        else:
            print(f"BT={BT:>7} {name:20s} {r['ms']:>8.1f} ms  {r['tok_per_s']:>9} tok/s  peak {r['peak_gb']:>6.2f} GB (in {r['inputs_gb']:.2f})  loss {r['loss']}")
    print("-"*100)
print("\nJSON:", json.dumps(rows))
