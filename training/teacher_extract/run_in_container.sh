#!/usr/bin/env bash
# Run a teacher_extract script inside the lmsysorg/sglang SIF with the proof-pilot
# hidden-extraction patches bind-mounted over the image's sglang files.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_in_container.sh _validate_hidden.py --tp 4
#
# NOT part of the FMI submission; local extraction tooling only.
set -euo pipefail
SIF=${SIF:-/images/sglang.sif}
HERE="$(cd "$(dirname "$0")" && pwd)"
PDIR="$HERE/_patched"

# Locate the in-image sglang and keep patched copies alongside this script.
SGL=$(apptainer exec "$SIF" python3 -c "import sglang; print(sglang.__file__.rsplit('/',1)[0])")
if [ ! -d "$PDIR" ] || [ "${REPATCH:-0}" = "1" ]; then
  rm -rf "$PDIR"; mkdir -p "$PDIR/orig"
  apptainer exec "$SIF" cat "$SGL/srt/models/deepseek_v4.py" > "$PDIR/orig/deepseek_v4.py"
  apptainer exec "$SIF" cat "$SGL/srt/managers/scheduler.py" > "$PDIR/orig/scheduler.py"
  apptainer exec "$SIF" cat "$SGL/srt/managers/scheduler_output_processor_mixin.py" \
    > "$PDIR/orig/scheduler_output_processor_mixin.py"
  apptainer exec "$SIF" cat "$SGL/srt/entrypoints/http_server.py" \
    > "$PDIR/orig/http_server.py"
  python3 "$HERE/_patch_sglang.py" "$PDIR/orig" "$PDIR"
fi

SCRIPT=$1; shift
exec apptainer exec --nv \
  --bind /work \
  --bind "$PDIR/deepseek_v4.py:$SGL/srt/models/deepseek_v4.py" \
  --bind "$PDIR/scheduler.py:$SGL/srt/managers/scheduler.py" \
  --bind "$PDIR/scheduler_output_processor_mixin.py:$SGL/srt/managers/scheduler_output_processor_mixin.py" \
  --env SGLANG_DSV4_HIDDEN_POST_NORM=1 \
  --env MALLOC_ARENA_MAX=4 \
  --env SGLANG_HIDDEN_SPOOL_DIR="${SGLANG_HIDDEN_SPOOL_DIR:-}" \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  "$SIF" python3 "$HERE/$SCRIPT" "$@"
