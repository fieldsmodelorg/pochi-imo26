"""Deterministic cleaning of the final output, and fallbacks.

The final proof we return is a candidate's <solution> content (looked up by the selector's chosen
id), so it is already tag-free. `deterministic_clean` is a safety net: if any wrapper/self-eval/
score tags survive (e.g. on the fallback path that uses raw call content), strip them. Math inside
the proof ($...$, \\boxed{...}) is preserved untouched.
"""
from __future__ import annotations

import re

from parser import ProofPackage, RefinedPackage
from prompts import fallback_preamble

_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)
_SELFEVAL_BLOCK = re.compile(r"<self_evaluation>.*?</self_evaluation>", re.DOTALL | re.IGNORECASE)
_SCORE_BLOCK = re.compile(r"<score>.*?</score>", re.DOTALL | re.IGNORECASE)
_STRUCT_TAG = re.compile(r"</?(?:solution|self_evaluation|score|evaluation|suggestions|selected_id)\s*>",
                         re.IGNORECASE)
MIN_FINAL_CHARS = 200


def deterministic_clean(text: str) -> str:
    text = text or ""
    # if the whole tagged block is present, keep only the <solution> body
    m = _SOLUTION_RE.search(text)
    if m:
        text = m.group(1)
    # otherwise strip any stray self-eval / score blocks and structural tags
    text = _SELFEVAL_BLOCK.sub("", text)
    text = _SCORE_BLOCK.sub("", text)
    text = _STRUCT_TAG.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text) >= MIN_FINAL_CHARS else ""


def fallback_best_available(ranked: list[ProofPackage], refined: list[RefinedPackage]) -> str:
    for r in refined:
        if r.valid:
            cleaned = deterministic_clean(r.proof)
            if cleaned:
                return cleaned
    for p in ranked:
        if p.valid:
            cleaned = deterministic_clean(p.proof)
            if cleaned:
                return cleaned
    return fallback_preamble()


def fallback_from_raw(proofs: list[ProofPackage]) -> str:
    """No valid proof at all: return the strongest cleaned raw solution, else the preamble."""
    candidates = sorted(proofs, key=lambda p: len(p.proof or ""), reverse=True)
    for p in candidates:
        cleaned = deterministic_clean(p.proof or p.call.get("content", ""))
        if cleaned:
            return cleaned
    return fallback_preamble()
