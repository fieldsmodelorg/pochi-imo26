#!/usr/bin/env python
"""Central launcher for EVERY attention-sink test across the OPD stack.

Runs the sink correctness tests that live in different repos / venvs / runners, and is
**skip-aware**: each target declares what it needs (GPU, FA2, FA3/Hopper, olmo_core, ≥2 GPUs, a live
sglang server) and is skipped with a reason when the environment can't satisfy it — so the same
command is safe on a laptop, one B200, a multi-GPU node, or inside the container.

Covers the three sink implementations + the serving path:
  * transformers FA2 sink (our B200 trainer path)     -> opd_v2_train_smoke.py
  * Yi-Chia's patched-FA3 in-kernel sink (Hopper)      -> olmo3_sink/tests/_fa3_*
  * OLMo-core post-correction sink (FA2/FA3 + CP)       -> OLMo-core src/test/nn/attention/*
  * sglang triton in-kernel sink (serving parity)       -> deploy/target/_parity_test.py

Usage:
  python test_attention_sink.py                # run the 'unit' group (default)
  python test_attention_sink.py --group all    # everything (adds multigpu + integration)
  python test_attention_sink.py --list         # list targets + why each would run/skip here
  python test_attention_sink.py -k fa2         # only targets whose key/desc match 'fa2'
  python test_attention_sink.py --with-server http://127.0.0.1:30000   # enable the sglang parity test
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

# --- roots: work both in-container (/opt/...) and in the dev workspace ---
IN_CONTAINER = os.path.isdir("/opt/opd/repo")
if IN_CONTAINER:
    REPO = "/opt/opd/repo"                                   # ycchen tree
    SMOKE = "/opt/opd/opd_v2_train_smoke.py"
    OLMO_CORE = os.environ.get("OLMO_CORE_DIR", "/opt/OLMo-core")
    HERE = "/opt/opd"
else:
    HERE = os.path.dirname(os.path.abspath(__file__))        # <repo>/training/docker/
    REPO = os.path.dirname(os.path.dirname(HERE))            # <repo>/
    SMOKE = f"{HERE}/opd_v2_train_smoke.py"
    OLMO_CORE = os.environ.get("OLMO_CORE_DIR", "/opt/OLMo-core")

# the FA3 (Hopper-only) olmo3_sink tests are not shipped in this tree; their targets fail fast if run
FA3_TESTS_DIR = f"{REPO}/training/olmo3_sink/tests"

# repo import paths (for the ycchen script-style tests that `import olmo3_sink` / opd_v2)
REPO_PYTHONPATH = os.pathsep.join([
    f"{REPO}/training", f"{REPO}/training/olmo3_sink", f"{REPO}/training/opd_v2/src",
    f"{REPO}/training/stage1_v2/src", f"{REPO}/training/_common", f"{REPO}/training/_vendor_opd",
])


@dataclass
class Target:
    key: str
    desc: str
    path: str                      # file (script) or test file (pytest)
    runner: str = "script"         # "script" | "pytest"
    group: str = "unit"            # "unit" | "multigpu" | "integration"
    needs: tuple = ()              # tokens: gpu fa2 fa3 olmo_core olmo3_sink sglang server gpus2
    args: list = field(default_factory=list)
    add_pythonpath: str = ""       # extra PYTHONPATH so repo imports resolve


TARGETS = [
    Target("opd_v2_fa2_smoke",
           "transformers FA2 sink == eager (forward + sink-grad + q/k/v-grad) — our B200 trainer path",
           SMOKE, "script", "unit", ("gpu", "fa2")),
    Target("ycchen_fa3_sink",
           "Yi-Chia patched-FA3 in-kernel sink fwd+bwd vs fp32 ref (Hopper only)",
           f"{FA3_TESTS_DIR}/_fa3_sink_test.py", "script", "unit", ("gpu", "fa3", "olmo3_sink"),
           add_pythonpath=REPO_PYTHONPATH),
    Target("ycchen_fa3_verify",
           "Yi-Chia FA3 sink numerical verify (Hopper only)",
           f"{FA3_TESTS_DIR}/_fa3_sink_verify.py", "script", "unit", ("gpu", "fa3", "olmo3_sink"),
           add_pythonpath=REPO_PYTHONPATH),
    Target("ycchen_fa3_compile",
           "Yi-Chia FA3 sink torch.compile (Hopper only)",
           f"{FA3_TESTS_DIR}/_fa3_compile_test.py", "script", "unit", ("gpu", "fa3", "olmo3_sink"),
           add_pythonpath=REPO_PYTHONPATH),
    Target("olmocore_sink_eager",
           "OLMo-core sink identity vs eager (no flash needed)",
           f"{OLMO_CORE}/src/test/nn/attention/attention_sink_test.py", "script", "unit",
           ("gpu", "olmo_core")),
    Target("olmocore_sink_flash",
           "OLMo-core real FA2/FA3 sink + in-kernel-vs-postproc equivalence",
           f"{OLMO_CORE}/src/test/nn/attention/attention_sink_flash_test.py", "pytest", "unit",
           ("gpu", "fa2", "olmo_core")),
    Target("olmocore_sink_ulysses",
           "OLMo-core Ulysses CP + sink correctness (≥2 GPU)",
           f"{OLMO_CORE}/src/test/nn/attention/attention_sink_ulysses_test.py", "pytest", "multigpu",
           ("gpus2", "olmo_core")),
    Target("olmocore_cp_intradoc",
           "OLMo-core CP intra-document masking, FA2/FA3 (≥2 GPU)",
           f"{OLMO_CORE}/src/test/nn/transformer/cp_intra_doc_test.py", "pytest", "multigpu",
           ("gpus2", "olmo_core")),
    Target("serve_triton_parity",
           "sglang triton in-kernel sink logprob parity vs HF olmo3_sink (needs a live server)",
           f"{REPO}/deploy/target/_parity_test.py", "script", "integration", ("sglang", "server"),
           add_pythonpath=REPO_PYTHONPATH),
]

_PROBE = r"""
import json
d = {}
try:
    import torch
    d["cuda"] = bool(torch.cuda.is_available()); d["gpus"] = torch.cuda.device_count() if d["cuda"] else 0
    d["cc"] = list(torch.cuda.get_device_capability()) if d["cuda"] else [0, 0]
except Exception:
    d["cuda"] = False; d["gpus"] = 0; d["cc"] = [0, 0]
for m in ("flash_attn", "flash_attn_interface", "olmo_core", "transformers", "sglang", "pytest"):
    try:
        __import__(m); d[m] = True
    except Exception:
        d[m] = False
print(json.dumps(d))
"""


def resolve_venvs() -> dict:
    v = {}
    for name, p in (("train", "/opt/venv/train/bin/python"), ("serve", "/opt/venv/serve/bin/python")):
        if os.path.exists(p):
            v[name] = p
    v["system"] = sys.executable
    return v


def probe(py: str) -> dict:
    try:
        out = subprocess.run([py, "-c", _PROBE], capture_output=True, text=True, timeout=120)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"cuda": False, "gpus": 0, "cc": [0, 0], "_err": str(e)}


def satisfies(pr: dict, tok: str, server: str | None) -> tuple[bool, str]:
    if tok == "gpu":
        return (pr.get("cuda", False), "no CUDA GPU")
    if tok == "gpus2":
        return (pr.get("gpus", 0) >= 2, f"needs ≥2 GPUs (have {pr.get('gpus', 0)})")
    if tok == "fa2":
        return (pr.get("flash_attn", False), "flash-attn (FA2) not installed")
    if tok == "fa3":
        hopper = tuple(pr.get("cc", [0, 0])) == (9, 0)
        return (pr.get("flash_attn_interface", False) and hopper,
                "FA3 needs Hopper sm_90 + flash_attn_interface")
    if tok in ("olmo_core", "sglang", "olmo3_sink"):
        if tok == "olmo3_sink":
            return (os.path.isdir(f"{REPO}/training/olmo3_sink"), "olmo3_sink source not found")
        return (pr.get(tok, False), f"{tok} not importable")
    if tok == "server":
        return (bool(server), "no --with-server URL given")
    return (True, "")


def pick_venv(t: Target, venvs: dict, probes: dict, server: str | None):
    """First venv whose probe satisfies all of t.needs; return (py, probe) or (None, reason)."""
    # Default: prefer "system" over "serve". The trainer venv is /opt/conda (cu128) and is reachable
    # only as "system" (there is no /opt/venv/train). "serve" is cu130 and must NOT run trainer-path
    # GPU tests (e.g. opd_v2_fa2_smoke, which hard-asserts cu12.8). On a NATIVE-cuda-13 B200 the serve
    # venv is CUDA-visible, so putting it before "system" wrongly grabbed it -> cu12.8 guard tripped.
    order = ["train", "system", "serve"]
    if "olmo_core" in t.needs:
        order = ["system", "train", "serve"]  # olmo_core usually lives in the SFT/system env
    if "sglang" in t.needs:
        order = ["serve", "system", "train"]
    reasons = []
    for name in order:
        if name not in venvs:
            continue
        pr = probes[name]
        ok = True
        for tok in t.needs:
            good, why = satisfies(pr, tok, server)
            if not good:
                ok = False
                reasons.append(f"{name}: {why}")
                break
        if ok:
            return venvs[name], pr, name
    return None, None, "; ".join(dict.fromkeys(reasons)) or "no suitable venv"


def run_target(t: Target, py: str, server: str | None) -> tuple[str, str]:
    if not os.path.exists(t.path):
        return "SKIP", f"test file not present ({t.path})"
    env = dict(os.environ)
    if t.add_pythonpath:
        env["PYTHONPATH"] = t.add_pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OPD_REPO", REPO)
    if t.runner == "pytest":
        cmd = [py, "-m", "pytest", "-q", t.path]
    else:
        cmd = [py, t.path, *t.args]
        if "server" in t.needs and server:
            url = server.rstrip("/")
            port = url.rsplit(":", 1)[-1]
            cmd += ["--port", port]
    print(f"  $ {' '.join(cmd)}")
    if t.runner == "pytest":
        # capture so we can tell a real pass from "exit 0 but every subtest skipped"
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()
        for ln in tail[-8:]:
            print("    " + ln)
        last = tail[-1] if tail else ""
        if r.returncode != 0:
            return "FAIL", f"exit {r.returncode}"
        if "passed" not in last and "skipped" in last:
            return "SKIP", "all subtests skipped (no FA2/FA3 or single GPU)"
        return "PASS", (last if "passed" in last else "")
    r = subprocess.run(cmd, env=env)
    return ("PASS", "") if r.returncode == 0 else ("FAIL", f"exit {r.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="unit", choices=["unit", "multigpu", "integration", "all"])
    ap.add_argument("-k", dest="filter", default="", help="substring filter on key/desc")
    ap.add_argument("--with-server", dest="server", default=os.environ.get("SGLANG_URL"),
                    help="sglang base URL to enable the serving parity test")
    ap.add_argument("--list", action="store_true", help="list targets + run/skip decision, don't run")
    args = ap.parse_args()

    venvs = resolve_venvs()
    probes = {name: probe(py) for name, py in venvs.items()}
    print(f"venvs: {', '.join(venvs)} | in_container={IN_CONTAINER}")
    for name, pr in probes.items():
        print(f"  [{name}] cuda={pr.get('cuda')} gpus={pr.get('gpus')} cc={tuple(pr.get('cc',[0,0]))} "
              f"fa2={pr.get('flash_attn')} fa3pkg={pr.get('flash_attn_interface')} "
              f"olmo_core={pr.get('olmo_core')} sglang={pr.get('sglang')}")

    sel = [t for t in TARGETS if (args.group == "all" or t.group == args.group)]
    if args.filter:
        f = args.filter.lower()
        sel = [t for t in sel if f in t.key.lower() or f in t.desc.lower()]

    results = []
    for t in sel:
        py, _pr, why = pick_venv(t, venvs, probes, args.server)
        print(f"\n### {t.key} [{t.group}] — {t.desc}")
        if py is None:
            print(f"  SKIP: {why}")
            results.append((t.key, "SKIP", why))
            continue
        if args.list:
            print(f"  would run in venv -> {py}")
            results.append((t.key, "RUN?", py))
            continue
        status, detail = run_target(t, py, args.server)
        print(f"  {status}{(' — ' + detail) if detail else ''}")
        results.append((t.key, status, detail))

    print("\n" + "=" * 72 + "\nSUMMARY")
    npass = sum(1 for _, s, _ in results if s == "PASS")
    nfail = sum(1 for _, s, _ in results if s == "FAIL")
    nskip = sum(1 for _, s, _ in results if s == "SKIP")
    for k, s, d in results:
        print(f"  {s:5} {k}{('  (' + d + ')') if d and s != 'PASS' else ''}")
    print(f"\n{npass} passed, {nfail} failed, {nskip} skipped")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
