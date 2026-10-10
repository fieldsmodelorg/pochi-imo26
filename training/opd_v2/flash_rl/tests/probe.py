#!/usr/bin/env python3
"""fp8 + update_weights_from_disk probe.
Usage: probe.py generate <port> | probe.py update <port> <model_path>
"""
import json, sys, time, urllib.request

def post(port, path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())

PROMPT = "Question: What is 17 multiplied by 23? Show your reasoning briefly.\nAnswer:"

def gen(port):
    t0 = time.time()
    out = post(port, "/generate", {
        "text": PROMPT,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 120},
    })
    dt = time.time() - t0
    txt = out["text"] if isinstance(out, dict) else out[0]["text"]
    meta = out.get("meta_info", {}) if isinstance(out, dict) else {}
    print(f"[gen] {dt:.2f}s  completion_tokens={meta.get('completion_tokens')}")
    print("----- OUTPUT -----")
    print(txt[:800])
    print("------------------")

def upd(port, path):
    t0 = time.time()
    out = post(port, "/update_weights_from_disk", {"model_path": path, "flush_cache": True})
    print(f"[update] {time.time()-t0:.2f}s  resp={out}")

if __name__ == "__main__":
    cmd = sys.argv[1]
    port = sys.argv[2]
    if cmd == "generate":
        gen(port)
    elif cmd == "update":
        upd(port, sys.argv[3])
