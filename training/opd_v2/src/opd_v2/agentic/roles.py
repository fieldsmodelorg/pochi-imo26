# Copyright 2026 proof-pilot. Apache-2.0.
"""role -> Prompt: use **math_3r's XML templates + rank/bundle** to assemble pool artifacts into a role
context, then render into input_ids with the **student tokenizer** (token-in-token-out).

Four roles:
- prove  : problem only -> render_prover_prompt
- verify : problem + 1 proof -> render_verifier_prompt (feeds the proof's <solution>+<self_eval>, no think)
- refine : problem + that problem's proofs(+verifies) -> rank_proofs -> build_refine_bundle(top-4) -> render_refiner_prompt
- select : problem + that problem's refined -> build_select_bundle -> render_selector_prompt

Reuses math_3r's existing pure code (parser dataclasses / rank / bundle / prompts), **no re-parsing** — the
pool stores the parsed fields, so we reconstruct the dataclasses directly. candidate_id/refiner_id are
**ephemeral labels per bundle** (P0../R0..), unrelated to pool node ids. on-policy: when
prefer_student_context, prefer student-source artifacts.

The math_3r modules are imported by bare name (parser/rank/bundle/prompts) — following the repo's existing
pattern (pipeline.py/run.py/trainer/core.py all sys.path.insert + bare import); this module is only loaded
when producer=="agentic".
"""
from __future__ import annotations

import os
import sys

from opd_v2.config import OPDConfig
from opd_v2.data_plane.produce import Prompt

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
_M3R = os.path.join(_REPO, "training", "distill_gen", "math_3r")
if _M3R not in sys.path:
    sys.path.insert(0, _M3R)

# math_3r pure modules (bare import; see module docstring)
from bundle import build_refine_bundle, build_select_bundle          # noqa: E402
from parser import ProofPackage, RefinedPackage, VerificationPackage  # noqa: E402
from prompts import (render_prover_prompt, render_refiner_prompt,      # noqa: E402
                     render_selector_prompt, render_verifier_prompt, to_messages)
from rank import rank_proofs                                          # noqa: E402


class RolePromptBuilder:
    """Holds the student tokenizer + cfg, assembles (role, pool context) into Prompt(ids, meta)."""

    def __init__(self, student_path: str, cfg: OPDConfig):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
        self.cfg = cfg

    # ---- render: math_3r rendered text (===SYSTEM===/===USER===) -> student chat template -> ids ----
    def _render(self, text: str) -> list[int] | None:
        msgs = to_messages(text)                          # [{system}, {user}] or [{user}]
        rendered = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = self.tok.encode(rendered, add_special_tokens=False)
        if len(ids) > self.cfg.agentic.max_prompt_tokens:
            return None
        return ids

    def _prompt(self, ids: list[int] | None, stage: str, problem_id: str,
                refs: list, ctx_wvs: list) -> Prompt | None:
        if ids is None:
            return None
        return Prompt(ids=ids, meta={"stage": stage, "problem_id": problem_id,
                                     "refs": refs, "ctx_wvs": ctx_wvs})

    # ---- proofs/refined selection (prefer student-source to drive on-policy transfer) ----
    def _proofs_for_refine(self, prob):
        ag = self.cfg.agentic
        student = [p for p in prob.proofs if p.source == "student"]
        # only narrow to student when "there are student proofs and at least one has been verified" (otherwise refine has no verifier review)
        if ag.prefer_student_context and any(p.n_verifies() > 0 for p in student):
            return [p for p in student]
        return list(prob.proofs)

    def _refined_for_select(self, prob):
        ag = self.cfg.agentic
        student = [r for r in prob.refined if r.source == "student"]
        if ag.prefer_student_context and len(student) >= 2:
            return student
        return list(prob.refined)

    # ---- the four roles ----
    def build_prove(self, prob) -> Prompt | None:
        ids = self._render(render_prover_prompt(prob.text))
        return self._prompt(ids, "prove", prob.problem_id, [], [])

    def build_verify(self, prob, proof) -> Prompt | None:
        text = render_verifier_prompt(prob.text, proof.content, proof.self_eval)
        ids = self._render(text)
        return self._prompt(ids, "verify", prob.problem_id, [proof.id], [proof.wv])

    def build_refine(self, prob) -> Prompt | None:
        proofs = self._proofs_for_refine(prob)
        if not proofs:
            return None
        # reconstruct ProofPackage[] (ephemeral P0..) + VerificationPackage[] (same candidate_id)
        pkgs, verifs, node_by_label = [], [], {}
        for i, p in enumerate(proofs):
            cid = f"P{i}"
            node_by_label[cid] = p
            pkgs.append(ProofPackage(candidate_id=cid, proof=p.content, self_eval=p.self_eval,
                                     self_score=p.self_score, valid=True, call={}))
            for j, v in enumerate(p.verifies):
                verifs.append(VerificationPackage(candidate_id=cid, verifier_idx=j, text=v.text,
                                                  score=v.score, call={}))
        ranked = rank_proofs(pkgs, verifs)
        bundle = build_refine_bundle(ranked, verifs, cap_tokens=self.cfg.agentic.refine_bundle_cap_tokens)
        ids = self._render(render_refiner_prompt(prob.text, bundle))
        top = [node_by_label[p.candidate_id] for p in ranked[:4]]   # refs/ctx_wvs aligned to the top-4 that actually entered the bundle
        return self._prompt(ids, "refine", prob.problem_id, [n.id for n in top], [n.wv for n in top])

    def build_select(self, prob) -> Prompt | None:
        refined = self._refined_for_select(prob)
        if len(refined) < 2:
            return None
        pkgs, id_by_label = [], {}
        for i, r in enumerate(refined):
            rid = f"R{i}"
            id_by_label[rid] = r.id
            pkgs.append(RefinedPackage(refiner_id=rid, proof=r.content, self_eval=r.self_eval,
                                       self_score=r.self_score, valid=True, call={}))
        bundle, _id_map = build_select_bundle(pkgs, cap_tokens=self.cfg.agentic.select_bundle_cap_tokens)
        ids = self._render(render_selector_prompt(prob.text, bundle))
        return self._prompt(ids, "select", prob.problem_id, [r.id for r in refined],
                            [r.wv for r in refined])
