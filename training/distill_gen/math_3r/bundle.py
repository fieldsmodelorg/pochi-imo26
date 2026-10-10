"""Build the refiner and selector input bundles as XML (no markdown, no rank labels).

Each candidate is one <candidate id="..."> block holding its <proof>, its <verifier_review> (s),
and (for refine) its <self_evaluation>. Ranking is used internally to pick the top-4 and to order
the blocks, but is NOT shown to the model. Refiners all get the same bundle; the selector bundle
also returns an id -> proof-text map so the selector picks an ID and the proof is looked up
deterministically.

Token counting is APPROXIMATE: len(text)//CHARS_PER_TOK (no DeepSeek tokenizer shipped).
"""
from __future__ import annotations

from rank import group_by_candidate
from parser import ProofPackage, RefinedPackage, VerificationPackage

CHARS_PER_TOK = 4
REFINE_CAP_TOKENS = 90_000
SELECT_CAP_TOKENS = 120_000


def est_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOK


def _assemble(segments: list[tuple[int, str]], cap_tokens: int) -> str:
    """segments = (priority, text); lower priority kept first. Whole blocks kept until the cap;
    the highest-priority overflowing block is head-truncated rather than dropped."""
    budget = cap_tokens
    kept: list[tuple[int, str]] = []
    truncated_once = False
    for i in sorted(range(len(segments)), key=lambda j: (segments[j][0], j)):
        _, text = segments[i]
        need = est_tokens(text)
        if need <= budget:
            kept.append((i, text))
            budget -= need
        elif not kept and not truncated_once:
            kept.append((i, text[: budget * CHARS_PER_TOK] + "\n...[truncated]"))
            budget = 0
            truncated_once = True
    kept.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in kept)


def _verifs_sorted(cand_id: str, by_cand):
    # most informative first: lowest score first (None last)
    return sorted(by_cand.get(cand_id, []),
                  key=lambda v: (v.score is None, v.score if v.score is not None else 9))


def _fmt_score(s) -> str:
    return "?" if s is None else ("%g" % s)  # 1.0->"1", 0.5->"0.5", 0.0->"0"


def _candidate_block(cid: str, proof: str, verifs, self_eval: str | None) -> str:
    parts = [f'<candidate id="{cid}">', "<proof>", proof or "", "</proof>"]
    for v in verifs:
        parts.append(f'<verifier_review score="{_fmt_score(v.score)}">')
        parts.append(v.text or "")
        parts.append("</verifier_review>")
    if self_eval:
        parts += ["<self_evaluation>", self_eval, "</self_evaluation>"]
    parts.append("</candidate>")
    return "\n".join(parts)


def build_refine_bundle(ranked: list[ProofPackage], verifs: list[VerificationPackage],
                        cap_tokens: int = REFINE_CAP_TOKENS) -> str:
    """top-4 candidates only, each as an XML block with proof + verifier reviews + self-eval."""
    by_cand = group_by_candidate(verifs)
    segs = [(r, _candidate_block(p.candidate_id, p.proof, _verifs_sorted(p.candidate_id, by_cand),
                                 p.self_eval))
            for r, p in enumerate(ranked[:4])]
    return _assemble(segs, cap_tokens)


def build_select_bundle(refined: list[RefinedPackage],
                        cap_tokens: int = SELECT_CAP_TOKENS) -> tuple[str, dict[str, str]]:
    """Returns (bundle_xml, id->proof_text) over the REFINED candidates only — the selector picks
    among the refined proofs (originals already fed the refiners). Returns ("", {}) if no refined,
    in which case the pipeline falls back to the best verified original."""
    id_map: dict[str, str] = {}
    segs: list[tuple[int, str]] = []
    for j, rp in enumerate(refined):
        id_map[rp.refiner_id] = rp.proof
        segs.append((j, _candidate_block(rp.refiner_id, rp.proof, [], None)))
    return _assemble(segs, cap_tokens), id_map
