"""Load the DSMV2-Simple-3R prompt templates and render them.

The template files live in the repo's `evaluation/prompts/ycchen_math_3r/` (shared with the inference
harness, which loads the same files), so training and serving read byte-identical prompts.

All four templates are in the lean t3_dsmv2_lite style: a ===SYSTEM===/===USER=== split, sent as
a system + user message pair. There are NO per-call strategy/mode variants — the 6 provers, the
verifiers, and the refiners each share one prompt, and diversity comes purely from the backend's
sampling randomness (reasoning=high uses the backend default sampling).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# training/distill_gen/math_3r/prompts.py -> parents[3] = repo root
PROMPT_DIR = Path(__file__).resolve().parents[3] / "evaluation" / "prompts" / "ycchen_math_3r"
SYS_DELIM = "===SYSTEM==="
USR_DELIM = "===USER==="


@lru_cache(maxsize=None)
def _load(name: str) -> str:
    return (PROMPT_DIR / name).read_text()


def to_messages(text: str) -> list[dict]:
    """Split a rendered ===SYSTEM===/===USER=== template into system+user messages."""
    if SYS_DELIM in text and USR_DELIM in text:
        sys_part = text.split(SYS_DELIM, 1)[1].split(USR_DELIM, 1)[0].strip()
        usr_part = text.split(USR_DELIM, 1)[1].strip()
        return [{"role": "system", "content": sys_part},
                {"role": "user", "content": usr_part}]
    return [{"role": "user", "content": text}]


def render_prover_prompt(problem: str) -> str:
    return _load("prover.txt").replace("{problem}", problem)


def render_verifier_prompt(problem: str, candidate_solution: str, candidate_self_eval: str) -> str:
    return (_load("verifier.txt")
            .replace("{problem}", problem)
            .replace("{candidate_solution}", candidate_solution)
            .replace("{candidate_self_eval}", candidate_self_eval))


def render_refiner_prompt(problem: str, candidate_bundle: str) -> str:
    return (_load("refiner.txt")
            .replace("{problem}", problem)
            .replace("{candidate_bundle}", candidate_bundle))


def render_selector_prompt(problem: str, selection_bundle: str) -> str:
    return (_load("selector.txt")
            .replace("{problem}", problem)
            .replace("{selection_bundle}", selection_bundle))
