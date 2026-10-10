#!/usr/bin/env python
"""Fail if any installed package pulls a CUDA newer than a ceiling (default 12.8).

Enforces the hard "everything on CUDA 12.8, nothing on 12.9 / 13.0" requirement at BUILD time:
run it as the last step of the image build so a contaminated transitive dep fails the build LOUDLY
instead of silently shipping.

Checks:
  1. torch.version.cuda must be <= the ceiling (e.g. "12.8").
  2. No installed distribution is a CUDA-13 wheel (name ends in "-cu13" / "-cu130").
  3. CUDA-versioned NVIDIA runtime libs (whose PyPI version IS the CUDA version, e.g.
     nvidia-cuda-runtime-cu12==12.8.90) must be <= the ceiling. Libs with independent versioning
     (cudnn=9.x, nccl=2.x) are exempt from the numeric check but still caught by the -cu13 rule.

Usage: python verify_cuda_max.py [--max 12.8]
"""
from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import distributions

# NVIDIA PyPI libs whose version string equals the CUDA toolkit version (X.Y.Z).
CUDA_VERSIONED = {
    "nvidia-cuda-runtime",
    "nvidia-cuda-cupti",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-sanitizer-api",
    "nvidia-cublas",
    "nvidia-cufft",
    "nvidia-curand",
    "nvidia-cusolver",
    "nvidia-cusparse",
    "nvidia-nvjitlink",
    "nvidia-nvtx",
    "nvidia-cufile",
}


def _ver_tuple(v: str) -> tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", v)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", default="12.8", help="max allowed CUDA minor, e.g. 12.8")
    args = ap.parse_args()
    ceil = _ver_tuple(args.max)

    violations: list[str] = []

    # 1. torch's CUDA
    try:
        import torch

        tv = torch.version.cuda  # e.g. "12.8"
        if tv is None:
            violations.append("torch has no CUDA build (torch.version.cuda is None)")
        elif _ver_tuple(tv) > ceil:
            violations.append(f"torch.version.cuda={tv} > {args.max}")
        else:
            print(f"[ok] torch.version.cuda={tv}")
    except Exception as e:  # noqa: BLE001
        violations.append(f"could not import torch: {e}")

    # 2 + 3. scan installed distributions
    for dist in distributions():
        name = (dist.metadata["Name"] or "").lower()
        ver = dist.version
        if not name:
            continue
        # CUDA-13 wheels (torch cu130 and its nvidia-*-cu13 deps, or any *-cu130)
        if ceil < (13, 0) and (name.endswith("-cu13") or name.endswith("-cu130") or "cu130" in name):
            violations.append(f"{name}=={ver} is a CUDA-13 package")
            continue
        # CUDA-versioned nvidia libs at > ceiling
        if name.endswith("-cu12"):
            base = name[: -len("-cu12")]
            if base in CUDA_VERSIONED and _ver_tuple(ver) > ceil:
                violations.append(f"{name}=={ver} > CUDA {args.max}")

    if violations:
        print(f"\n[FAIL] {len(violations)} CUDA-version violation(s) (ceiling {args.max}):", file=sys.stderr)
        for v in sorted(violations):
            print(f"   - {v}", file=sys.stderr)
        return 1

    print(f"[ok] no package exceeds CUDA {args.max}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
