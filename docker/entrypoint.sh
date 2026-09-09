#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/opt/aimo-proof-pilot-inference}"
WORKSPACE=/workspace
# The SGLang runtime is baked into the image at /opt/pp; overridable for the
# legacy download-at-boot path (RUNTIME_ROOT=/workspace/pp).
RUNTIME_ROOT="${RUNTIME_ROOT:-/opt/pp}"
HF_HOME="${HF_HOME:-/workspace/.hf_home}"
VENV="${VENV:-$RUNTIME_ROOT/venv}"
STATE_DIR=/workspace/.proof-pilot
MODEL_ROOT=/workspace/models
DEFAULT_TARGET_MODEL="$MODEL_ROOT/opd-32b-bf16-step-225"
DEFAULT_DRAFT_MODEL="$MODEL_ROOT/dflash-32b-draft-v2test-phaseL"
TARGET_MODEL=
DRAFT_MODEL=
# Default target = the step-225 checkpoint (best IMO result), served from the
# FM-Pochi-32B-IMO26 repo. The shared DFlash draft ships ONLY in the
# FM-Pochi-32B-ProofPilot repo, so target and draft come from two different
# repos (see ensure_models).
# Override any of these four to serve a different checkpoint without editing
# this file.
MODEL_REPO="${MODEL_REPO:-fieldsmodelorg/FM-Pochi-32B-IMO26}"
MODEL_REVISION="${MODEL_REVISION:-f14030d3c65e1ed59e4e70477297053fc9a75151}"
DRAFT_REPO="${DRAFT_REPO:-fieldsmodelorg/FM-Pochi-32B-ProofPilot}"
DRAFT_REVISION="${DRAFT_REVISION:-87707b8030800b1e531b78c9823cb80a63d66e5e}"
# Runtime venv (patched SGLang + kernels). The image BAKES this at /opt/pp from
# RUNTIME_BASE_IMAGE at build time (see runtime/README.md), so the normal boot
# path uses the baked runtime and downloads nothing.
# sha256 of proof-pilot-env.bin (the gzip'd venv tar). Boot dies on mismatch.
# Set empty to disable the check (e.g. when deliberately using a different archive).
RUNTIME_ARCHIVE_SHA256="${RUNTIME_ARCHIVE_SHA256:-71190f4f2554c29ec6b99ae6bda7af64f1348876b85cfbdfa1d102f9dfa8c831}"
RUNTIME_BIN="${RUNTIME_BIN:-/workspace/proof-pilot-env.bin}"
# Optional override: set to a Kaggle dataset to fetch the runtime archive instead
# of using the baked /opt/pp. NOT revision-pinned; the sha256 check still applies,
# so a changed upstream archive fails loud rather than drifting.
RUNTIME_DATASET="${RUNTIME_DATASET:-}"
RUNTIME_ARCHIVE="${RUNTIME_ARCHIVE:-/workspace/proof-pilot-env.zip}"
CONFIG_SOURCE="${CONFIG:-}"
SERVER_HOST=
SERVER_PORT=
SERVER_URL=
SERVER_LOG="${EVAL_SERVER_LOG:-/workspace/opd32b-eval.log}"
SERVER_VALIDATION="${SERVER_VALIDATION:-$STATE_DIR/server-validation.json}"
SERVER_STARTUP_TIMEOUT_SECONDS="${SERVER_STARTUP_TIMEOUT_SECONDS:-2700}"
EXPECTED_GPU_COUNT=
REQUIRE_H200="${REQUIRE_H200:-1}"
# Input CSV: an explicit $INPUT_CSV wins; else an operator-supplied /workspace/test.csv
# if present; else the committed IMO-2026 set -- the exact 6-problem CSV validated on
# NII. This makes the container reproduce the NII input by DEFAULT (the harness keys all
# deterministic RNG on CSV row index, so the exact set+order matters). Mount your own at
# /workspace/test.csv (or set INPUT_CSV) to run different problems.
if [[ -z "${INPUT_CSV:-}" ]]; then
    if [[ -f /workspace/test.csv ]]; then
        INPUT_CSV=/workspace/test.csv
    else
        INPUT_CSV="$REPO/evaluation/data/imo2026-latex-test.csv"
    fi
fi
OUTPUT_CSV="${OUTPUT_CSV:-/workspace/submission.csv}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/workspace/submission_artifacts}"
SERVER_PID=
TEMP_PATHS=()

log() {
    printf '[proof-pilot] %s\n' "$*"
}

die() {
    printf '[proof-pilot] ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup_temp_paths() {
    local path
    for path in "${TEMP_PATHS[@]:-}"; do
        if [[ "$path" == /workspace/.proof-pilot-* ]]; then
            rm -rf -- "$path"
        fi
    done
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "stopping server pid=$SERVER_PID"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
}

trap cleanup_temp_paths EXIT

load_workspace_env() {
    if [[ -f /workspace/.env ]]; then
        log "loading /workspace/.env"
        set -a
        # shellcheck disable=SC1091
        source /workspace/.env
        set +a
    fi
}

validate_gpus() {
    command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable; launch with NVIDIA GPUs enabled"

    local names=()
    mapfile -t names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
    [[ "${#names[@]}" -eq "$EXPECTED_GPU_COUNT" ]] || {
        printf '[proof-pilot] visible GPUs:\n%s\n' "${names[*]:-none}" >&2
        die "expected $EXPECTED_GPU_COUNT visible GPUs, found ${#names[@]}"
    }

    if [[ "$REQUIRE_H200" == "1" ]]; then
        local name
        for name in "${names[@]}"; do
            [[ "$name" == *H200* ]] || die "expected H200 GPUs, found: $name"
        done
    fi
    log "validated ${#names[@]} GPUs: ${names[0]}"
}


verify_runtime_sha256() {
    # Pins the SGLang runtime by content: proof-pilot-env.bin must hash to the
    # pinned sha256, or we refuse to boot. This is what guarantees an identical
    # SGLang across deployments regardless of where the archive came from.
    local file="$1"
    if [[ -z "$RUNTIME_ARCHIVE_SHA256" ]]; then
        log "runtime sha256 check disabled (RUNTIME_ARCHIVE_SHA256 empty)"
        return
    fi
    log "verifying runtime archive sha256"
    local actual
    actual="$(sha256sum "$file" | awk '{print $1}')"
    [[ "$actual" == "$RUNTIME_ARCHIVE_SHA256" ]] || die \
        "runtime archive sha256 mismatch: got $actual, expected $RUNTIME_ARCHIVE_SHA256 "\
"(the runtime archive changed; update RUNTIME_ARCHIVE_SHA256, or clear it to override)"
}

fetch_runtime_payload() {
    # Produces the pinned proof-pilot-env.bin at $RUNTIME_BIN.
    if [[ -f "$RUNTIME_BIN" ]]; then
        log "using cached runtime payload $RUNTIME_BIN"
        return
    fi

    if [[ -n "$RUNTIME_DATASET" ]]; then
        # Override: unpinned Kaggle dataset -> pull its zip and extract the .bin.
        log "downloading runtime dataset $RUNTIME_DATASET (Kaggle override; not revision-pinned)"
        [[ -f "$RUNTIME_ARCHIVE" ]] || kaggle datasets download "$RUNTIME_DATASET" --path "$WORKSPACE"
        unzip -tq "$RUNTIME_ARCHIVE" >/dev/null || die "runtime archive failed ZIP integrity validation"
        local extract_root
        extract_root="$(mktemp -d /workspace/.proof-pilot-archive.XXXXXX)"
        TEMP_PATHS+=("$extract_root")
        unzip -q "$RUNTIME_ARCHIVE" -d "$extract_root"
        local extracted
        extracted="$(find "$extract_root" -type f -name proof-pilot-env.bin -print -quit)"
        [[ -n "$extracted" ]] || die "proof-pilot-env.bin is missing from $RUNTIME_ARCHIVE"
        mv "$extracted" "$RUNTIME_BIN"
        rm -rf "$extract_root"
    else
        die "runtime not baked at $RUNTIME_ROOT and no RUNTIME_DATASET override set. "\
"The image bakes /opt/pp from RUNTIME_BASE_IMAGE at build time (see runtime/README.md); "\
"rebuild the image, mount a prebuilt /opt/pp, or set RUNTIME_DATASET to a runtime archive."
    fi
}

ensure_runtime() {
    if [[ -x "$VENV/bin/python" && -x "$RUNTIME_ROOT/pybase/bin/python3" ]]; then
        log "using existing runtime at $RUNTIME_ROOT"
        return
    fi
    [[ ! -e "$RUNTIME_ROOT" ]] || die "$RUNTIME_ROOT exists but is incomplete; move or remove it before retrying"

    fetch_runtime_payload
    verify_runtime_sha256 "$RUNTIME_BIN"

    local stage_root
    stage_root="$(mktemp -d /workspace/.proof-pilot-runtime.XXXXXX)"
    TEMP_PATHS+=("$stage_root")

    log "extracting the relocatable runtime"
    tar -xzf "$RUNTIME_BIN" -C "$stage_root" --strip-components=1
    [[ -x "$stage_root/venv/bin/python" ]] || die "extracted runtime has no venv Python"
    [[ -x "$stage_root/pybase/bin/python3" ]] || die "extracted runtime has no base Python"

    sed -i "s|^home = .*|home = $RUNTIME_ROOT/pybase/bin|" "$stage_root/venv/pyvenv.cfg"
    mv "$stage_root" "$RUNTIME_ROOT"
    log "runtime installed at $RUNTIME_ROOT"
}

prepare_caches() {
    mkdir -p /root/.cache/flashinfer /root/.humming/cache
    if [[ -d "$RUNTIME_ROOT/flashinfer_cache" ]]; then
        cp -rn "$RUNTIME_ROOT/flashinfer_cache/." /root/.cache/flashinfer/
    fi
    if [[ -d "$RUNTIME_ROOT/humming_cache" ]]; then
        cp -rn "$RUNTIME_ROOT/humming_cache/." /root/.humming/cache/
    fi
}

install_dependencies_and_patches() {
    local requirements_hash
    local marker
    requirements_hash="$(sha256sum "$REPO/evaluation/requirements.txt" | awk '{print $1}')"
    marker="$RUNTIME_ROOT/.proof-pilot-deps-$requirements_hash"

    if [[ ! -f "$marker" ]]; then
        log "installing pinned evaluation dependencies"
        uv pip install --python "$VENV/bin/python" -r "$REPO/evaluation/requirements.txt"
        touch "$marker"
    else
        log "using previously installed evaluation dependencies"
    fi

    log "applying the checked-in SGLang patch set"
    bash "$REPO/sglang_patches/apply_patches.sh" "$VENV" \
        "$RUNTIME_ROOT/proof-pilot/deploy/w4a8/humming_w4a8.py"
}

require_config_file() {
    [[ -n "$CONFIG_SOURCE" ]] || die "CONFIG is required and must point to config.yaml"
    [[ -f "$CONFIG_SOURCE" ]] || die "configuration does not exist: $CONFIG_SOURCE"
}

load_runtime_config() {
    local inspected
    inspected="$("$VENV/bin/python" "$REPO/docker/inspect_config.py" "$CONFIG_SOURCE")" \
        || die "configuration validation failed: $CONFIG_SOURCE"

    SERVER_HOST="$(jq -er ".server_host | strings" <<<"$inspected")"
    SERVER_PORT="$(jq -er ".server_port | numbers" <<<"$inspected")"
    SERVER_URL="$(jq -er ".server_url | strings" <<<"$inspected")"
    EXPECTED_GPU_COUNT="$(jq -er ".expected_gpu_count | numbers" <<<"$inspected")"
    TARGET_MODEL="$(jq -er ".target_model | strings" <<<"$inspected")"
    DRAFT_MODEL="$(jq -r '.draft_model // ""' <<<"$inspected")"
    log "using authoritative config unchanged: $CONFIG_SOURCE"
}

model_complete() {
    local path="$1"
    local index="$1/model.safetensors.index.json"
    local shard
    local shard_list
    local shards=()

    [[ -d "$path" && -f "$path/config.json" ]] || return 1
    if [[ -f "$index" ]]; then
        shard_list="$(jq -er ".weight_map | values[]" "$index" | sort -u)" \
            || return 1
        mapfile -t shards <<<"$shard_list"
        [[ "${#shards[@]}" -gt 0 ]] || return 1
        for shard in "${shards[@]}"; do
            [[ -f "$path/$shard" ]] || return 1
        done
        return 0
    fi
    [[ -n "$(find "$path" -maxdepth 1 -type f -name "*.safetensors" -print -quit)" ]]
}

uses_default_models() {
    [[ "$TARGET_MODEL" == "$DEFAULT_TARGET_MODEL" ]] \
        && { [[ -z "$DRAFT_MODEL" ]] || [[ "$DRAFT_MODEL" == "$DEFAULT_DRAFT_MODEL" ]]; }
}

ensure_models() {
    # The default target and the shared draft live in different repos, so the
    # state key records both -- changing either forces a reconcile.
    local expected_source="$MODEL_REPO@$MODEL_REVISION+$DRAFT_REPO@$DRAFT_REVISION"
    local recorded_source=
    if [[ -f "$STATE_DIR/model-revision" ]]; then
        recorded_source="$(<"$STATE_DIR/model-revision")"
    fi

    if uses_default_models; then
        if ! model_complete "$TARGET_MODEL" \
            || { [[ -n "$DRAFT_MODEL" ]] && ! model_complete "$DRAFT_MODEL"; } \
            || [[ "$recorded_source" != "$expected_source" ]]; then
            mkdir -p "$MODEL_ROOT"
            log "reconciling default model assets with $expected_source"
            hf download "$MODEL_REPO" \
                --revision "$MODEL_REVISION" \
                --include "$(basename "$DEFAULT_TARGET_MODEL")/*" \
                --local-dir "$MODEL_ROOT"
            if [[ -n "$DRAFT_MODEL" ]]; then
                hf download "$DRAFT_REPO" \
                    --revision "$DRAFT_REVISION" \
                    --include "$(basename "$DEFAULT_DRAFT_MODEL")/*" \
                    --local-dir "$MODEL_ROOT"
            fi
        fi
        printf "%s\n" "$expected_source" > "$STATE_DIR/model-revision"
    fi

    model_complete "$TARGET_MODEL" \
        || die "configured target model is incomplete: $TARGET_MODEL"
    if [[ -n "$DRAFT_MODEL" ]]; then
        model_complete "$DRAFT_MODEL" \
            || die "configured draft model is incomplete: $DRAFT_MODEL"
    fi
    log "using configured target=$TARGET_MODEL draft=${DRAFT_MODEL:-disabled}"
}

prepare() {
    require_config_file
    mkdir -p "$STATE_DIR" "$HF_HOME"
    ensure_runtime
    prepare_caches
    install_dependencies_and_patches
    load_runtime_config
    ensure_models
}

start_server() {
    rm -f "$STATE_DIR/server-ready"
    : > "$SERVER_LOG"
    log "starting production server on $SERVER_HOST:$SERVER_PORT"
    # Point the humming (W4A8) helpers at the runtime root (baked at /opt/pp),
    # not launch_server.py's /workspace/pp default. Inert for the BF16 path.
    export HUMMING_PATH="${HUMMING_PATH:-$RUNTIME_ROOT}"
    export W4A8_HELPER_DIR="${W4A8_HELPER_DIR:-$RUNTIME_ROOT/proof-pilot/deploy/w4a8}"
    "$VENV/bin/python" "$REPO/evaluation/harness/launch_server.py" \
        --config "$CONFIG_SOURCE" \
        > >(tee -a "$SERVER_LOG") 2>&1 &
    SERVER_PID=$!
}

wait_for_server() {
    local deadline=$((SECONDS + SERVER_STARTUP_TIMEOUT_SECONDS))
    while (( SECONDS < deadline )); do
        if curl -fsS "$SERVER_URL/health" >/dev/null 2>&1; then
            return
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || {
            wait "$SERVER_PID" || true
            die "server exited before becoming healthy; inspect $SERVER_LOG"
        }
        sleep 5
    done
    die "server did not become healthy within $SERVER_STARTUP_TIMEOUT_SECONDS seconds"
}

validate_server() {
    log "validating live server and startup markers"
    "$VENV/bin/python" "$REPO/evaluation/harness/validate_server.py" \
        --url "$SERVER_URL" \
        --config "$CONFIG_SOURCE" \
        --output "$SERVER_VALIDATION" \
        --server-log "$SERVER_LOG"
    touch "$STATE_DIR/server-ready"
    log "server validation passed"
}

run_server() {
    prepare

    validate_gpus
    trap stop_server TERM INT
    start_server
    wait_for_server
    validate_server
    log "server ready; log=$SERVER_LOG validation=$SERVER_VALIDATION"
    local status=0
    wait "$SERVER_PID" || status=$?
    SERVER_PID=
    return "$status"
}

run_submission() {
    [[ -f "$INPUT_CSV" ]] || die "input CSV does not exist: $INPUT_CSV"
    prepare
    validate_gpus
    trap stop_server TERM INT
    trap 'stop_server; cleanup_temp_paths' EXIT
    start_server
    wait_for_server
    validate_server
    log "running submission: input=$INPUT_CSV output=$OUTPUT_CSV"
    CONFIG="$CONFIG_SOURCE" ARTIFACTS_DIR="$ARTIFACTS_DIR" \
        bash "$REPO/run_submission.sh" "$INPUT_CSV" "$OUTPUT_CSV"
    log "submission complete: $OUTPUT_CSV"
}

usage() {
    cat <<'EOF'
Usage: entrypoint.sh COMMAND [ARGS...]

Commands:
  serve       Bootstrap, validate the configured GPU topology, and run the production SGLang server.
  submission  Bootstrap, run the server, and generate /workspace/submission.csv.
  bootstrap   Download and prepare the runtime and models without requiring GPUs.
  validate    Validate an already running server against the supplied config.
  shell       Bootstrap and open a shell.
  help        Show this message.

CONFIG must point to an existing YAML file for every command except help.
Any other command is executed after bootstrap. Persistent model, cache, log, and
result data live under /workspace (the runtime is baked into the image at /opt/pp).
EOF
}

load_workspace_env
cd "$REPO"

case "${1:-serve}" in
    serve)
        run_server
        ;;
    submission)
        run_submission
        ;;
    bootstrap)
        prepare
        ;;
    validate)
        prepare
        validate_gpus
        validate_server
        ;;
    shell)
        prepare
        exec /bin/bash
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        prepare
        exec "$@"
        ;;
esac
