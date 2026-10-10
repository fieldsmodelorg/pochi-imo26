# Copyright 2026 proof-pilot. Apache-2.0.
"""Shared training-pipeline modules (single source of truth).

Canonical home for stage-agnostic training code:

- `encoding_dsv4`: vendored DeepSeek-V4 official chat encoder (`encode_messages`).
- `l3_render`: L2 messages -> tokenized (input_ids, labels) with offset-based
  assistant-only loss masking. Round-trip validated against `encode_messages`.

Consumed by `training/stage1_v2/` (and later stages) via direct import — the FMI
container gets a copy materialized at packaging time by each stage's
`make_pkg.py` (see `training/stage1_v2/README.md` for the layout rationale).

NOTE: `training/stage1/src/{encoding_dsv4,l3_render}.py` are the ARCHIVED
stage-1 copies (frozen with that package); this directory is where they live on.
"""
