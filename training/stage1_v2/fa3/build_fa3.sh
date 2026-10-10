#!/usr/bin/env bash
# Copyright 2026 proof-pilot. Apache-2.0.
# Build the PATCHED FlashAttention-3 (in-kernel attention sink) for sm90 (H100/H200),
# producing the `flash_attn_interface` module that olmo3_sink imports for
# attn_implementation="olmo3_sink_fa3". Called from the Singularity %post.
#
# Recipe mirrors docs/fa3_build.md: build ONLY bf16 + hdim128 (Olmo3-7B head_dim=128) +
# sm90, so the kernel matrix is small (~2-3 min). The 6-file sink patch is backward
# compatible (sink=None == stock FA3).
#
set -euo pipefail

FA_SRC="${FA_SRC:-/opt/flash-attention}"
PATCH="${PATCH:-/opt/fa3/fa3_attention_sink.patch}"
# PINNED to the flash-attention commit the sink patch was authored + validated against
# (fp32 kernel verify 9/9; 2026-05-27 main). cutlass submodule resolves to 7127592 (v4.3.0).
FA_COMMIT="${FA_COMMIT:-0bbb25a3a5ad3c58c029b3d287d6c9af56a5cad5}"

git clone https://github.com/Dao-AILab/flash-attention.git "$FA_SRC"
cd "$FA_SRC"
git checkout "$FA_COMMIT"
git submodule update --init --recursive
git apply "$PATCH"

# The sink patch modifies hopper/flash_api.cpp (the non-stable Torch API source). For
# torch >= 2.9 the hopper setup.py auto-selects flash_api_stable.cpp instead, which is
# UNPATCHED -> the `sink` arg is missing from the registered flash_attn_3::fwd op and the
# in-kernel sink call fails at runtime ("expected at most 34 arguments but received 35").
# We build from source pinned to the runtime torch, so the non-stable ABI is fine: force
# flash_api.cpp regardless of torch version.
sed -i 's/if torch_version >= target_version:/if False and torch_version >= target_version:/' hopper/setup.py

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS="${MAX_JOBS:-12}" NVCC_THREADS="${NVCC_THREADS:-2}"
# keep only what Olmo3-7B training needs: bf16, hdim128, sm90, fwd+bwd, varlen, sliding
export FLASH_ATTENTION_DISABLE_FP8=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE
export FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE
export FLASH_ATTENTION_DISABLE_HDIM192=TRUE FLASH_ATTENTION_DISABLE_HDIM256=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
export FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE FLASH_ATTENTION_DISABLE_PACKGQA=TRUE

pip install --no-build-isolation --no-cache-dir ./hopper
python -c "from flash_attn_interface import flash_attn_varlen_func; print('FA3 sink build OK')"
