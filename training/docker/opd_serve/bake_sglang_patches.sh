#!/usr/bin/env bash
# Bake Yi-Chia's OPD serve patches into the serve venv's sglang AT BUILD TIME (replaces the
# runtime --bind overlays in her apptainer scripts). Run inside the image after sglang is installed.
#
#   SERVE_PY=/opt/venv/serve/bin/python OPD_REPO=/opt/opd/repo ./bake_sglang_patches.sh
#
# ⚠️ VERSION-SENSITIVE. The committed rollout patches (loader.py, model_config.py) are whole-file
#    replacements content-pinned to sglang 0.5.12.post1. If the serve venv ships a different sglang,
#    set REGEN=1 to re-derive them from the installed source via her apply scripts (anchored edits).
#    The teacher patches are always generated from the installed source by _patch_sglang.py.
set -euo pipefail
SERVE_PY="${SERVE_PY:-/opt/venv/serve/bin/python}"
OPD_REPO="${OPD_REPO:-/opt/opd/repo}"
REGEN="${REGEN:-0}"

SGL="$("$SERVE_PY" -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')"
echo "[bake] sglang @ $SGL  (version: $("$SERVE_PY" -c 'import sglang;print(getattr(sglang,"__version__","?"))'))"
FR="$OPD_REPO/training/opd_v2/flash_rl"
TE="$OPD_REPO/training/teacher_extract"

# ---- ROLLOUT (student) patches ----
# This fork carries native OLMo3-sink, FlashInfer, and FlashRL reload support.
# Do not replace those files with the 0.5.12 whole-file overlays when it is installed.
if grep -q "_run_flashinfer_paged_with_sinks" "$SGL/srt/layers/attention/flashinfer_backend.py" \
   && grep -q "Olmo3SinkForCausalLM" "$SGL/srt/models/olmo2.py" \
   && grep -q "_validate_attention_sink_checkpoint" "$SGL/srt/model_loader/loader.py"; then
  echo "[bake] native OLMo3 FlashInfer/FlashRL support detected; skipping rollout overlays"
else
  # 1) Legacy OLMo sink model overlay.
  cp -v "$OPD_REPO/deploy/target/olmo2_sink.py" "$SGL/srt/models/olmo2.py"

  # 2) Legacy flash_rl fp8 loader + SWA model_config overlays.
  if [ "$REGEN" = "1" ]; then
    echo "[bake] REGEN loader/model_config from installed sglang"
    cp "$SGL/srt/model_loader/loader.py" /tmp/_loader_stock.py
    "$SERVE_PY" "$FR/apply_patch.py"            /tmp/_loader_stock.py "$SGL/srt/model_loader/loader.py"
    cp "$SGL/srt/configs/model_config.py" /tmp/_mcfg_stock.py
    "$SERVE_PY" "$FR/patches/apply_swa_patch.py" /tmp/_mcfg_stock.py "$SGL/srt/configs/model_config.py"
  else
    cp -v "$FR/patches/loader.py"       "$SGL/srt/model_loader/loader.py"
    cp -v "$FR/patches/model_config.py" "$SGL/srt/configs/model_config.py"
  fi
fi

# ---- TEACHER (DeepSeek-V4-Flash) hidden-extract patches ----
# Generated from the installed sglang source (anchored edits): deepseek_v4 / scheduler /
# output_processor / http_server(out_path FS-write).
SRC=/tmp/_teacher_stock; OUT=/tmp/_teacher_patched
mkdir -p "$SRC" "$OUT"
cp "$SGL/srt/models/deepseek_v4.py" "$SRC/deepseek_v4.py"
cp "$SGL/srt/managers/scheduler.py" "$SRC/scheduler.py"
cp "$SGL/srt/managers/scheduler_output_processor_mixin.py" \
  "$SRC/scheduler_output_processor_mixin.py"
cp "$SGL/srt/entrypoints/http_server.py" "$SRC/http_server.py"
"$SERVE_PY" "$TE/_patch_sglang.py" "$SRC" "$OUT"
for f in deepseek_v4.py:srt/models scheduler.py:srt/managers \
         scheduler_output_processor_mixin.py:srt/managers http_server.py:srt/entrypoints; do
  name="${f%%:*}"; sub="${f##*:}"
  [ -f "$OUT/$name" ] && cp -v "$OUT/$name" "$SGL/$sub/$name" || echo "[bake] WARN: $name not generated"
done

echo "[bake] verifying sglang still imports with patches applied"
"$SERVE_PY" -c "import sglang.srt.models.olmo2, sglang.srt.model_loader.loader; print('[bake] OK')"
