#!/usr/bin/env python3
"""Repeated reload stress test for flash_rl fp8 + update_weights_from_disk.
Usage: cycle.py <port> <ckptA> <ckptB>
Sequence: gen(A) -> [update B, gen] -> [update A, gen] -> [update B, gen] ...
"""
import json, sys, time, urllib.request, hashlib

def post(port, path, payload, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

PROMPT = "Question: What is 17 multiplied by 23? Show your reasoning briefly.\nAnswer:"

def gen(port, tag):
    out = post(port, "/generate", {"text": PROMPT,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 64}})
    txt = out["text"]
    h = hashlib.sha1(txt.encode()).hexdigest()[:8]
    has391 = "391" in txt
    print(f"  [{tag}] coherent={'YES' if has391 else 'NO?'} hash={h} :: {txt[:90].strip()!r}")
    return h

def upd(port, path, tag):
    t0 = time.time()
    out = post(port, "/update_weights_from_disk", {"model_path": path, "flush_cache": True})
    ok = out.get("success", False)
    print(f"  [{tag}] update_weights_from_disk success={ok} ({time.time()-t0:.1f}s) -> {path.split('/')[-1]}")
    return ok

if __name__ == "__main__":
    port, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    print("=== cycle 0: baseline (served A) ===")
    gen(port, "gen0")
    targets = [((f"->B"), B) if i % 2 else (("->A"), A) for i in range(n)]
    for i, (tag, path) in enumerate(targets, 1):
        print(f"=== cycle {i}: reload {tag} ===")
        if not upd(port, path, f"upd{i}"):
            print("  UPDATE RETURNED success=false"); break
        gen(port, f"gen{i}")
    print("=== final health ===")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
            print("  SERVER ALIVE after 4 reloads")
    except Exception as e:
        print("  SERVER DEAD:", e)
