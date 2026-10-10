#!/usr/bin/env bash
# Build a STOCK flash-attn 2 wheel for the ycchen-opd cu128 image, covering Ampere (sm_80),
# Hopper (sm_90, H200), and Blackwell (sm_100 B200 / sm_120), against torch 2.10.0+cu128 / cp312.
#
# WHY: the runtime base ships flash-attn built for Blackwell ONLY (sm_100/sm_120), so olmo3_sink_fa2
# hits "no kernel image" on H200 (sm_90). Official pre-built wheels carry all arches but top out at
# torch 2.9 (image is torch 2.10) and are cu13/aarch64 — no drop-in for cu128+torch2.10+x86_64. So we
# build the STOCK, UNMODIFIED kernel ourselves (the sink is a post-correction outside the kernel, so no
# patching). The base is a runtime image (no nvcc) -> build in a cu128 DEVEL container. ~30 min.
#
#   OUT=training/docker/wheels bash training/docker/build_fa2_wheel.sh
# then rebuild the image (Dockerfile picks the wheel up from training/docker/wheels/).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/wheels}"
BUILDER="${BUILDER:-pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel}"   # cu128 nvcc + py3.12 + cxx11abi True
FA_VER="${FA_VER:-2.8.2}"
# Targets ONLY: Hopper sm_90 (H200) + Blackwell sm_100 (B200). No sm_80/sm_120 (not deployment targets;
# fewer archs = less compile memory/time). Add archs here if you ever need A100/consumer-Blackwell.
ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90;100}"
JOBS="${MAX_JOBS:-3}"    # each nvcc compiles all archs at once (memory-heavy) -> keep modest to avoid OOM-kill
mkdir -p "$OUT"
echo ">>> building stock flash-attn $FA_VER for arches [$ARCHS] against torch 2.10.0+cu128 (builder=$BUILDER, MAX_JOBS=$JOBS)"
docker run --rm -v "$OUT":/out "$BUILDER" bash -c "
  set -eux
  export PIP_BREAK_SYSTEM_PACKAGES=1
  pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 --force-reinstall
  pip install --no-cache-dir ninja packaging wheel setuptools psutil
  python -c 'import torch; print(\"builder torch\", torch.__version__, torch.version.cuda, \"cxx11abi\", torch._C._GLIBCXX_USE_CXX11_ABI)'
  export FLASH_ATTN_CUDA_ARCHS='$ARCHS' MAX_JOBS='$JOBS' NVCC_THREADS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE
  pip wheel --no-build-isolation --no-deps 'flash-attn==$FA_VER' -w /out
  ls -la /out
"
echo ">>> done -> $OUT"
