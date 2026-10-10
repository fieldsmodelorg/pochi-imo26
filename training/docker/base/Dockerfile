# syntax=docker/dockerfile:1.7
# =============================================================================
#  Base image for the OPD image: CUDA 12.8 / torch 2.10 / FlashAttention / OLMo-core (sink fork)
# =============================================================================
# Recipe for chankhavu/olmo3-olmocore:cu128-fa2-sink, which Dockerfile.ycchen-opd builds FROM (pinned there by
# digest, sha256:e67d3894…). It follows OLMo-core's official image (a build stage with the CUDA toolchain that
# compiles the kernels into /opt/conda, and a slim release stage that copies /opt/conda), with:
#   * CUDA 12.8.1, torch 2.10.0+cu128 (everything <= CUDA 12.8; verify_cuda_max.py enforces it),
#   * flash-attn 2.8.2 built from source for sm_86/90/100/120, FA3 (Hopper) at a pinned commit, stock kernels
#     (the attention sink is applied as an exact post-correction, so no kernel patch is needed),
#   * OLMo-core's extras minus fa4, and its source from the hav4ik/OLMo-core fork at the commit in the image.
# Python package versions are pinned by constraints-base.txt (`pip list` of the published base image).
#
# Build from the REPO ROOT (FA2 for 4 archs + FA3 + grouped_gemm is a long compile; needs ~64GB RAM):
#   DOCKER_BUILDKIT=1 docker build -f training/docker/base/Dockerfile \
#     -t olmo3-olmocore:cu128-fa2-sink training/docker/base
# then build the OPD image with --build-arg BASE=olmo3-olmocore:cu128-fa2-sink.
# =============================================================================
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12
ARG MINIFORGE_VERSION=26.5.3-0
ARG OLMO_CORE_REPO=https://github.com/hav4ik/OLMo-core.git
ARG OLMO_CORE_COMMIT=830908352554ab3c24020b28d82b9f27868d576f

#########################################################################
# Build stage: toolchain + compiled kernels into /opt/conda
#########################################################################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS build
WORKDIR /app/build

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl wget libxml2-dev libjpeg-dev libpng-dev gcc git && \
    rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION
ARG MINIFORGE_VERSION
ENV PATH=/opt/conda/bin:$PATH
RUN curl -fsSL -o ~/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" && \
    bash ~/miniforge.sh -b -p /opt/conda && rm ~/miniforge.sh && \
    conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja && \
    conda clean -ya

COPY constraints-base.txt /tmp/constraints-base.txt
ENV PIP_CONSTRAINT=/tmp/constraints-base.txt

# torch (cu128). torchaudio is not installed: the official image's unpinned torchaudio resolved to a build
# for a newer torch and was removed downstream.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0+cu128 torchvision==0.25.0+cu128

# grouped-gemm with CUTLASS (batch sizes on GPU).
ARG GROUPED_GEMM_SHA=f1429a3c44c98f7912aa4b00125144cdf4e7fdb2
RUN TORCH_CUDA_ARCH_LIST="9.0 10.0" GROUPED_GEMM_CUTLASS=1 pip install --no-build-isolation --no-cache-dir \
        "grouped_gemm @ git+https://github.com/tgale96/grouped_gemm.git@${GROUPED_GEMM_SHA}"

# flash-attn 2 (stock), built for Ampere sm_86 + Hopper sm_90 + Blackwell sm_100/sm_120.
ARG FLASH_ATTN_VERSION=2.8.2
ARG FLASH_ATTN_CUDA_ARCHS="86;90;100;120"
RUN FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS}" pip install --no-build-isolation --no-cache-dir \
        flash-attn==${FLASH_ATTN_VERSION}

# flash-attn 3 (Hopper), stock, at a pinned commit. nvcc must be present or setup.py installs it uncompiled.
ARG FLASH_ATTN_3_SHA=92ca9da8d66f7b34ff50dc080ec0fef9661260d6
ARG FA3_MAX_JOBS=16
RUN command -v nvcc >/dev/null && \
    git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && git fetch --depth 1 origin ${FLASH_ATTN_3_SHA} && git checkout ${FLASH_ATTN_3_SHA} && \
    git submodule update --init --depth 1 && cd hopper && \
    FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=${FA3_MAX_JOBS} python setup.py install && \
    cd /app/build && rm -rf flash-attention
# FA3 installs as an egg importable only as `flash_attn_interface`; TransformerEngine expects
# `flash_attn_3.flash_attn_interface`.
RUN cd /opt/conda/lib/python${PYTHON_VERSION}/site-packages/ && \
    mkdir -p flash_attn_3 && \
    mv flash_attn_3-*.egg/flash_attn_3/* flash_attn_3/ && \
    mv flash_attn_3-*.egg/flash_attn_interface.py flash_attn_3/ && \
    touch flash_attn_3/__init__.py && rm -rf flash_attn_3-*

RUN pip install --no-build-isolation --no-cache-dir "transformer-engine[pytorch]==2.9.0"
RUN pip install --no-build-isolation --no-cache-dir ring-flash-attn==0.1.8
RUN pip install --no-build-isolation --no-cache-dir liger-kernel==0.6.4

# OLMo-core's dependencies (all extras except fa4), without the package itself (its source is added below).
ARG OLMO_CORE_REPO
ARG OLMO_CORE_COMMIT
RUN git clone "${OLMO_CORE_REPO}" /tmp/olmo-core && cd /tmp/olmo-core && git checkout "${OLMO_CORE_COMMIT}" && \
    pip install --no-cache-dir ".[dev,beaker,comet,dion,eval,fla,torchao,transformers,wandb]" && \
    pip uninstall -y ai2-olmo-core && cd / && rm -rf /tmp/olmo-core

#########################################################################
# Release stage
#########################################################################
FROM ubuntu:${UBUNTU_VERSION} AS release
ARG UBUNTU_VERSION
ARG PYTHON_VERSION

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl wget unzip libxml2-dev libjpeg-dev libpng-dev gnupg jq gcc git \
        libibverbs1 librdmacm1 libibumad3 ibverbs-providers rdma-core ibverbs-utils && \
    rm -rf /var/lib/apt/lists/*

# uv (Dockerfile.ycchen-opd creates its sglang venv with it).
ARG UV_VERSION=0.11.28
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh

# Mellanox OFED user-space drivers (InfiniBand for NCCL).
ENV MOFED_VER=24.01-0.3.3.1
RUN wget --quiet https://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-x86_64.tgz && \
    tar -xf MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-x86_64.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-x86_64/mlnxofedinstall --basic --user-space-only --without-fw-update -q && \
    rm -rf MLNX_OFED_LINUX-${MOFED_VER}-ubuntu${UBUNTU_VERSION}-x86_64*

COPY --from=build /opt/conda /opt/conda
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/root/.local/bin:/root/.cargo/bin:/opt/conda/bin:$PATH

# OLMo-core source (sink fork), editable, at the pinned commit.
ARG OLMO_CORE_REPO
ARG OLMO_CORE_COMMIT
RUN git clone "${OLMO_CORE_REPO}" /workspace/OLMo-core && cd /workspace/OLMo-core && \
    git checkout "${OLMO_CORE_COMMIT}" && pip install --no-cache-dir -e . --no-deps

# TransformerEngine finds libnvrtc via `ldconfig -p`; the cu12 wheels put it off the default linker path.
RUN set -e; d="$(dirname "$(find /opt/conda -name 'libnvrtc.so*' | head -1)")"; \
    echo "$d" > /etc/ld.so.conf.d/zz-nvrtc.conf && ldconfig && ldconfig -p | grep -q 'libnvrtc\.so'

COPY constraints-base.txt /tmp/constraints-base.txt
# ring-flash-attn's HF adapter imports a symbol transformers 5.x moved to transformers.utils.
RUN F="$(python -c 'import transformers.modeling_flash_attention_utils as m; print(m.__file__)')" && \
    echo "from transformers.utils import is_flash_attn_greater_or_equal_2_10  # shim: ring_flash_attn hf_adapter on transformers 5.x" >> "$F" && \
    python -c "import ring_flash_attn; print('ring-flash-attn OK')" && \
    pip install --no-cache-dir -c /tmp/constraints-base.txt bitsandbytes

# Hard ceiling: fail if any installed package pulls CUDA newer than 12.8.
COPY verify_cuda_max.py /usr/local/bin/verify_cuda_max.py
RUN python /usr/local/bin/verify_cuda_max.py --max 12.8

ENV HF_HOME=/data/training/hf_cache \
    NCCL_IB_DISABLE=0 \
    NCCL_SOCKET_IFNAME=^docker0,lo,veth \
    NCCL_BUFFSIZE=2097152 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_BLOCKING_WAIT=1
WORKDIR /workspace
CMD ["/bin/bash"]
