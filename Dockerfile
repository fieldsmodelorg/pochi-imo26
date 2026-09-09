# syntax=docker/dockerfile:1.7

# Global meta-arg: declared before the first FROM so BuildKit substitutes it in
# `FROM ${RUNTIME_BASE_IMAGE}` below (see the runtime stage + runtime/README.md).
ARG RUNTIME_BASE_IMAGE=ghcr.io/fieldsmodelorg/aimo-proof-pilot:sha-463682b

FROM ghcr.io/astral-sh/uv:0.11.19 AS uv

# ---------------------------------------------------------------------------
# Stage: the pinned SGLang runtime (/opt/pp). It is REUSED from the existing
# published release image -- which already bakes /opt/pp -- so the build depends
# only on the org container registry, not an external dataset, and nothing new
# has to be built or published. The `COPY --from=runtime /opt/pp` in the final
# stage lifts ONLY /opt/pp into the fresh image; everything else here is dropped.
#
# RUNTIME_BASE_IMAGE may be any image that carries a ready /opt/pp; pin it by
# @sha256 digest for strict reproducibility (see runtime/README.md). The base
# image must stay published and readable at build time (keep the GHCR package
# public). The SGLang attention-sink + DFlash patches live in-repo
# (sglang_patches/) and are applied on top at container boot, so the runtime is
# still modifiable -- fork the repo, edit the patches/config, rebuild.
# ---------------------------------------------------------------------------
FROM ${RUNTIME_BASE_IMAGE} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY --from=uv /uv /usr/local/bin/uv
COPY evaluation/requirements.txt /tmp/requirements.txt
COPY docker/validate_cutlass_install.py /tmp/validate_cutlass_install.py

# /opt/pp comes prebuilt + relocated from the base image. Re-assert the pinned
# PyPI deps so evaluation/requirements.txt stays authoritative, then verify the
# runtime imports.
# The base/CUDA-13 CUTLASS wheels overlap. Restore CUDA-13 last so Python
# helpers and MLIR bindings come from one wheel (required by FA4 on B200).
RUN set -Eeuo pipefail; \
    test -x /opt/pp/venv/bin/python; \
    test -x /opt/pp/pybase/bin/python3; \
    UV_LINK_MODE=copy /usr/local/bin/uv pip install \
        --python /opt/pp/venv/bin/python -r /tmp/requirements.txt; \
    UV_LINK_MODE=copy /usr/local/bin/uv pip install \
        --python /opt/pp/venv/bin/python --no-deps --reinstall \
        nvidia-cutlass-dsl-libs-cu13==4.5.2; \
    LD_LIBRARY_PATH=/opt/pp/pybase/lib /opt/pp/venv/bin/python \
        /tmp/validate_cutlass_install.py; \
    touch "/opt/pp/.proof-pilot-deps-$(sha256sum /tmp/requirements.txt | awk '{print $1}')"; \
    LD_LIBRARY_PATH=/opt/pp/pybase/lib /opt/pp/venv/bin/python -c \
        "import sglang, torch, flash_attn; print('baked runtime:', sglang.__version__, torch.__version__)"

# ---------------------------------------------------------------------------
# Final image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG VCS_REF=unknown

LABEL org.opencontainers.image.source="https://github.com/fieldsmodelorg/AIMO-Proof-Pilot"
LABEL org.opencontainers.image.revision="$VCS_REF"
LABEL org.opencontainers.image.title="AIMO Proof Pilot Inference"
LABEL org.opencontainers.image.description="OPD-32B generate-verify-refine inference; SGLang runtime baked in"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        git-lfs \
        jq \
        libaio1t64 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libibverbs1 \
        libnuma1 \
        numactl \
        ninja-build \
        pciutils \
        procps \
        python3 \
        python3-venv \
        rsync \
        tar \
        tini \
        unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    HF_HOME=/workspace/.hf_home \
    HF_XET_HIGH_PERFORMANCE=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    REPO=/opt/aimo-proof-pilot-inference \
    RUNTIME_ROOT=/opt/pp \
    VENV=/opt/pp/venv \
    IMAGE_REVISION=$VCS_REF

RUN uv tool install --python /usr/bin/python3 "huggingface-hub==1.18.0"

# The pinned, relocated, deps-complete SGLang runtime (a non-/workspace path so
# a runtime -v mount over /workspace never hides it).
COPY --from=runtime /opt/pp /opt/pp

WORKDIR /opt/aimo-proof-pilot-inference
COPY . /opt/aimo-proof-pilot-inference

RUN chmod 0755 docker/entrypoint.sh run_submission.sh

VOLUME ["/workspace"]
STOPSIGNAL SIGTERM

HEALTHCHECK --start-period=45m --interval=30s --timeout=10s --retries=5 \
    CMD test -n "$CONFIG" \
        && test -f /workspace/.proof-pilot/server-ready \
        && URL=$("$VENV/bin/python" \
            "$REPO/docker/inspect_config.py" "$CONFIG" \
            | jq -er .server_url) \
        && curl -fsS "$URL/health" >/dev/null \
        || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/aimo-proof-pilot-inference/docker/entrypoint.sh"]
CMD ["serve"]
