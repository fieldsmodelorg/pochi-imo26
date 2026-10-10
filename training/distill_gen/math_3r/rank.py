"""Rank valid proofs by verifier scores, then self-score, then length (spec section 6).

A proof with no verifications ranks below any verified proof (mean/min set to -1).
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean

from parser import ProofPackage, VerificationPackage


def length_score(proof: str) -> int:
    return len(proof or "")


def group_by_candidate(verifs: list[VerificationPackage]) -> dict[str, list[VerificationPackage]]:
    by: dict[str, list[VerificationPackage]] = defaultdict(list)
    for v in verifs:
        by[v.candidate_id].append(v)
    return by


def rank_proofs(proofs: list[ProofPackage],
                verifs: list[VerificationPackage]) -> list[ProofPackage]:
    by_cand = group_by_candidate(verifs)
    scored = []
    for p in proofs:
        vs = [v.score for v in by_cand.get(p.candidate_id, []) if v.score is not None]
        if vs:
            mean_v, min_v = mean(vs), min(vs)
        else:
            mean_v = min_v = -1.0
        scored.append((mean_v, min_v, p.self_score if p.self_score is not None else -1.0,
                       length_score(p.proof), p))
    scored.sort(key=lambda t: t[:4], reverse=True)
    return [t[-1] for t in scored]
