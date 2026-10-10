# Copyright 2026 proof-pilot. Apache-2.0.
"""agentic semi-on-policy OPD — pool-based multi-role distillation (only loaded when producer="agentic").

Pushes single-round prover OPD to the whole math_3r loop (prove/verify/refine/select). This package is only
imported inside the orchestrator process and only when cfg.producer=="agentic" -> the single-round path never
touches it (not even loaded at import time).

Modules:
- pool.py       PoolStore: per-problem graph + append-only JSONL + in-memory index (pure data/persistence)
- roles.py      reconstruct math_3r dataclasses + rank/bundle + render XML templates -> Prompt
- sampler.py    fill-fraction role sampling + within-role deficit item selection
- writeback.py  PoolIngestor: rollout -> decode/parse(answer-only) -> validity-gate -> pool.admit
- seed.py       cold-start: DeepSeek r3_hard2000 nested data -> seed.jsonl (full fill)
"""
