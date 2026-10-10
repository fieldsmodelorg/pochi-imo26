#!/usr/bin/env python3
"""Print per-prompt output hashes for a server (to compare reload vs fresh-load)."""
import json, sys, urllib.request, hashlib
PORT = sys.argv[1]
PROMPTS = {
    "arith":  "Question: Compute 17 * 23 step by step.\nAnswer:",
    "proof":  "Prove that the sum of the first n positive integers equals n(n+1)/2.\nProof:",
    "primes": "List the first 8 prime numbers and briefly say why 1 is not prime.\nAnswer:",
    "logic":  "If all cats are mammals and some mammals are black, can we conclude some cats are black? Explain.\nAnswer:",
}
def post(p, payload, t=600):
    req=urllib.request.Request(f"http://127.0.0.1:{PORT}{p}", data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=t).read())
if len(sys.argv) > 2 and sys.argv[2] == "update":
    print("update:", post("/update_weights_from_disk", {"model_path": sys.argv[3], "flush_cache": True}).get("success"))
for k,pr in PROMPTS.items():
    out = post("/generate", {"text":pr, "sampling_params":{"temperature":0.0,"max_new_tokens":160}})
    h = hashlib.sha1(out["text"].encode()).hexdigest()[:12]
    print(f"{k}\t{h}\t{out['text'][:55].strip()!r}")
