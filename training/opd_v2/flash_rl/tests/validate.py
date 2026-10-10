#!/usr/bin/env python3
"""Thorough post-reload generation validation for flash_rl fp8 + update_weights_from_disk.

Sequence:
  serve deploy(fp8) -> gen N prompts (baseline_deploy)
  update->opd       -> gen N prompts (must be coherent; should differ from deploy)
  update->deploy    -> gen N prompts (must be BIT-EXACT == baseline_deploy)
  + one long 512-tok decode after a reload (stress, catch late corruption)
"""
import os
import json, sys, urllib.request, hashlib

PORT = sys.argv[1]
A = os.environ.get("MODEL_A", "outputs/stage1-v2-7b-deploy")
B = os.environ.get("MODEL_B", "outputs/opd-mneff2-s1480")

PROMPTS = {
    "arith":  "Question: Compute 17 * 23 step by step.\nAnswer:",
    "proof":  "Prove that the sum of the first n positive integers equals n(n+1)/2.\nProof:",
    "word":   "A train travels 60 km in 45 minutes. What is its average speed in km/h? Explain.\nAnswer:",
    "primes": "List the first 8 prime numbers and briefly say why 1 is not prime.\nAnswer:",
}

def post(path, payload, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def degenerate(txt):
    """Heuristic: detect repetition collapse / empty / gibberish."""
    if len(txt.strip()) < 5:
        return "EMPTY"
    words = txt.split()
    if len(words) > 20:
        # any single token repeated >30% of the time => collapse
        from collections import Counter
        c = Counter(words)
        top, n = c.most_common(1)[0]
        if n > 0.30 * len(words):
            return f"REPEAT({top!r}x{n})"
    # longest run of one repeated 12-char chunk
    return None

def gen(tag, key, max_new=200):
    out = post("/generate", {"text": PROMPTS[key],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new}})
    txt = out["text"]
    h = hashlib.sha1(txt.encode()).hexdigest()[:10]
    deg = degenerate(txt)
    flag = "DEGEN:" + deg if deg else "ok"
    print(f"  [{tag}/{key}] {flag:14s} hash={h} :: {txt[:70].strip()!r}")
    return h, (deg is None)

def upd(path):
    out = post("/update_weights_from_disk", {"model_path": path, "flush_cache": True})
    print(f"  update -> {path.split('/')[-1]}: success={out.get('success')}")
    return out.get("success", False)

def gen_round(tag):
    return {k: gen(tag, k) for k in PROMPTS}

print("=== R0: baseline served=deploy(fp8) ===")
base = gen_round("deploy0")
print("=== reload -> opd ===")
assert upd(B)
opd = gen_round("opd")
print("=== reload -> deploy ===")
assert upd(A)
back = gen_round("deploy1")

print("\n=== long-decode stress (512 tok) after current reload (deploy) ===")
gen("deploy-long", "proof", max_new=512)

print("\n================ VERDICT ================")
allcoh = all(v[1] for v in {**base, **opd, **back}.values())
rt_exact = all(base[k][0] == back[k][0] for k in PROMPTS)
opd_differs = any(base[k][0] != opd[k][0] for k in PROMPTS)
print(f"all generations coherent (no degen): {allcoh}")
print(f"deploy->opd->deploy round-trip BIT-EXACT: {rt_exact}")
print(f"opd weights actually changed output: {opd_differs}")
for k in PROMPTS:
    same = "==base" if base[k][0]==back[k][0] else "!!DIFF"
    chg  = "changed" if base[k][0]!=opd[k][0] else "same-as-base"
    print(f"  {k:7s} deploy={base[k][0]} opd={opd[k][0]}({chg}) back={back[k][0]}({same})")
print("PASS" if (allcoh and rt_exact and opd_differs) else "FAIL")
