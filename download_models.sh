#!/usr/bin/env bash
# download_models.sh -- fetch the proof-pilot model weights into a models folder.
#
# Usage: ./download_models.sh [WHICH] [MODELS_DIR]
#   WHICH       all (default) | step225 | deploy   -- which target checkpoint(s)
#   MODELS_DIR  destination directory (default /workspace/models)
#
# The source repos are PUBLIC, so no HuggingFace token is needed. The shared
# DFlash draft is always fetched (every config uses it). Budget on disk: roughly
# ~64 GB per BF16 target checkpoint plus ~5 GB for the draft.
#
# The download layout matches the config model paths, e.g. after running this the
# step-225 config finds its weights at <MODELS_DIR>/opd-32b-bf16-step-225.
set -Eeuo pipefail

WHICH="${1:-all}"
MODELS_DIR="${2:-/workspace/models}"

# Pinned, public source repos + revisions (immutable for reproducibility).
PROOFPILOT_REPO="fieldsmodelorg/FM-Pochi-32B-ProofPilot"  # deploy + draft
PROOFPILOT_REV="87707b8030800b1e531b78c9823cb80a63d66e5e"
IMO_REPO="fieldsmodelorg/FM-Pochi-32B-IMO26"              # step checkpoints
IMO_REV="f14030d3c65e1ed59e4e70477297053fc9a75151"
DRAFT="dflash-32b-draft-v2test-phaseL"

log() { printf '[download-models] %s\n' "$*"; }
die() { printf '[download-models] ERROR: %s\n' "$*" >&2; exit 1; }

# huggingface-hub 1.x ships `hf`; older ships `huggingface-cli`.
HF="$(command -v hf || command -v huggingface-cli || true)"
[[ -n "$HF" ]] || die "the HuggingFace CLI is not on PATH (inside the container it is; otherwise: pip install huggingface-hub)"

pull() {  # repo revision subfolder
    local repo="$1" rev="$2" sub="$3"
    log "fetching ${sub}  <-  ${repo}@${rev:0:12}"
    "$HF" download "$repo" --revision "$rev" --include "${sub}/*" --local-dir "$MODELS_DIR"
}

mkdir -p "$MODELS_DIR"
log "target(s)=${WHICH}  dest=${MODELS_DIR}  (public repos, no token required)"

pull "$PROOFPILOT_REPO" "$PROOFPILOT_REV" "$DRAFT"          # shared DFlash draft, always
case "$WHICH" in
    step225) pull "$IMO_REPO" "$IMO_REV" "opd-32b-bf16-step-225" ;;
    deploy)  pull "$PROOFPILOT_REPO" "$PROOFPILOT_REV" "opd-32b-deploy" ;;
    all)     pull "$PROOFPILOT_REPO" "$PROOFPILOT_REV" "opd-32b-deploy"
             pull "$IMO_REPO" "$IMO_REV" "opd-32b-bf16-step-225" ;;
    *) die "unknown target '${WHICH}' (use: step225 | deploy | all)" ;;
esac

log "done. contents of ${MODELS_DIR}:"
ls -1 "$MODELS_DIR"
