"""Parse model outputs into typed packages and decide validity.

Outputs use XML tags (NOT markdown / \\boxed) so parsing is unambiguous and never collides with
`\\boxed{answer}` inside a proof:
  - prover / refiner: <solution>...</solution> <self_evaluation>...</self_evaluation> <score>0|0.5|1</score>
  - verifier:         <evaluation>...</evaluation> <suggestions>...</suggestions> <score>0|0.5|1</score>
  - selector:         <selected_id>P#|R#</selected_id>

Parsing is pure and never raises: a malformed generation yields valid=False / score=None, which
the pipeline filters out (the sample is dropped, others are used).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_SOLUTION_CHARS = 500

_SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)
_SELFEVAL_RE = re.compile(r"<self_evaluation>(.*?)</self_evaluation>", re.DOTALL | re.IGNORECASE)
_SCORE_RE = re.compile(r"<score>\s*(0(?:\.5)?|1)\s*</score>", re.IGNORECASE)
_ID_RE = re.compile(r"<selected_id>\s*([PR]\d+)\s*</selected_id>", re.IGNORECASE)


@dataclass
class ProofPackage:
    candidate_id: str          # P0, P1, ...
    proof: str                 # <solution> content
    self_eval: str             # <self_evaluation> content
    self_score: float | None   # <score>
    valid: bool
    call: dict = field(repr=False)


@dataclass
class VerificationPackage:
    candidate_id: str          # which proof this verifies
    verifier_idx: int
    text: str                  # full verifier output (evaluation + suggestions)
    score: float | None
    call: dict = field(repr=False)


@dataclass
class RefinedPackage:
    refiner_id: str            # R0, R1, ...
    proof: str
    self_eval: str
    self_score: float | None
    valid: bool
    call: dict = field(repr=False)


def _first(regex: re.Pattern, text: str) -> str:
    m = regex.search(text or "")
    return m.group(1).strip() if m else ""


def parse_score(text: str) -> float | None:
    m = _SCORE_RE.search(text or "")
    return float(m.group(1)) if m else None


def parse_selected_id(text: str) -> str | None:
    matches = _ID_RE.findall(text or "")
    return matches[-1] if matches else None


def _two_section(text: str, finish_reason: str | None, error) -> tuple[str, str, float | None, bool]:
    """Shared parsing/validity for prover & refiner outputs."""
    solution = _first(_SOLUTION_RE, text)
    self_eval = _first(_SELFEVAL_RE, text)
    score = parse_score(text)
    valid = (
        error is None
        and finish_reason != "length"
        and bool(solution)
        and score in {0.0, 0.5, 1.0}
        and len(solution) > MIN_SOLUTION_CHARS
    )
    return solution, self_eval, score, valid


def parse_proof_package(call: dict, candidate_id: str) -> ProofPackage:
    sol, se, score, valid = _two_section(call.get("content", ""),
                                         call.get("finish_reason"), call.get("error"))
    return ProofPackage(candidate_id=candidate_id, proof=sol, self_eval=se,
                        self_score=score, valid=valid, call=call)


def parse_refined_package(call: dict, refiner_id: str) -> RefinedPackage:
    sol, se, score, valid = _two_section(call.get("content", ""),
                                         call.get("finish_reason"), call.get("error"))
    return RefinedPackage(refiner_id=refiner_id, proof=sol, self_eval=se,
                          self_score=score, valid=valid, call=call)


def parse_verification(call: dict, candidate_id: str, verifier_idx: int) -> VerificationPackage:
    text = call.get("content", "") or ""
    score = None if call.get("error") else parse_score(text)
    return VerificationPackage(candidate_id=candidate_id, verifier_idx=verifier_idx,
                               text=text, score=score, call=call)
