# Copyright 2026 proof-pilot. Apache-2.0.
"""PoolStore — the per-problem artifact pool for agentic OPD (pure data structure + persistence).

graph: `problem → proofs → verifies`, `problem → refined`. select stores **no node** (nothing downstream
consumes its output, it's just a training sample) -> it only increments a counter. Each node carries
provenance (id/wv/source/step) for later tracking of vintage and on-policy share.

Design (PLAN agentic section):
- **The index is the in-memory truth** (`dict[problem_id → ProblemNode]`); the sampler reads and admit
  writes, **both on the orchestrator's single event loop** -> lock-free (same reasoning as buffer.py).
  This module **does not touch the tokenizer / parsing** (that's in writeback.py); admit_* receives
  "already-parsed artifacts".
- **Persistence = append-only JSONL**: `seed.jsonl` (cold-start, written by seed.py, immutable) +
  `artifacts.jsonl` (student admit append). `load()` replays both at startup -> resume-safe (echoes the
  collect.py / rollout_store convention).
- **on-policy transfer**: depth/fill counts **only count student-source** (seed provides context but does
  not satisfy student depth) -> the student keeps generating -> the pool naturally drifts from
  seed-dominated to student-dominated; the sampler prefers student-source when assembling context.

ids: `p{n}/v{n}/r{n}` globally monotonic (per kind); on replay the counter is restored from the existing
max value (no collision, resumable).
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
from dataclasses import dataclass, field

log = logging.getLogger("opd_v2.agentic.pool")

ROLES = ("prove", "verify", "refine", "select")


@dataclass
class VerifyNode:
    id: str
    problem_id: str
    proof_id: str
    score: float | None        # verifier <score> ∈ {0, .5, 1} (used for rank / refine bundle review)
    text: str                  # verifier <evaluation>/<suggestions> (used by refine bundle)
    wv: int
    source: str                # "deepseek_seed" | "student"


@dataclass
class ProofNode:
    id: str
    problem_id: str
    content: str               # parsed <solution> (answer-only, no think)
    self_eval: str
    self_score: float | None
    wv: int
    source: str
    verifies: list = field(default_factory=list)   # list[VerifyNode]

    def n_verifies(self, student_only: bool = False) -> int:
        if student_only:
            return sum(1 for v in self.verifies if v.source == "student")
        return len(self.verifies)


@dataclass
class RefinedNode:
    id: str
    problem_id: str
    parent_proof_ids: list
    content: str
    self_eval: str
    self_score: float | None
    wv: int
    source: str


@dataclass
class ProblemNode:
    problem_id: str
    text: str
    meta: dict = field(default_factory=dict)
    proofs: list = field(default_factory=list)     # list[ProofNode]
    refined: list = field(default_factory=list)    # list[RefinedNode]

    def has_verified_proof(self, student_only: bool = False) -> bool:
        return any(p.n_verifies(student_only) > 0 for p in self.proofs)


class PoolStore:
    """per-problem artifact pool. admit_* = in-memory index insert (+ append-only persistence)."""

    def __init__(self, pool_dir: str, *, seed: int = 0, max_artifact_chars: int = 200000):
        self.dir = pool_dir
        os.makedirs(self.dir, exist_ok=True)
        self.max_artifact_chars = max_artifact_chars
        self.problems: dict[str, ProblemNode] = {}
        self._proof_by_id: dict[str, ProofNode] = {}
        self._ctr = {"p": 0, "v": 0, "r": 0}
        # incremental student counts (O(1) student_counts; select has no node but is counted here too)
        # -> fill_fraction need not scan the whole pool every atom
        self._sc = {"prove": 0, "verify": 0, "refine": 0, "select": 0}
        # persistence: student admits accumulate into _wal; the flusher / persist() writes artifacts.jsonl (append-only)
        self._wal: list[dict] = []
        self._persisted_n = 0
        self._persist_lock = threading.Lock()   # persist is idempotent across threads (flusher executor vs close on loop)
        self._flusher = None
        self._loop = None

    # ---- paths ----
    @property
    def seed_path(self) -> str:
        return os.path.join(self.dir, "seed.jsonl")

    @property
    def artifacts_path(self) -> str:
        return os.path.join(self.dir, "artifacts.jsonl")

    # ---- id ----
    def _new_id(self, kind: str) -> str:
        self._ctr[kind] += 1
        return f"{kind}{self._ctr[kind]}"

    def _bump_ctr_from_id(self, _id: str) -> None:
        kind, n = _id[0], _id[1:]
        if kind in self._ctr and n.isdigit():
            self._ctr[kind] = max(self._ctr[kind], int(n))

    # ---- problems ----
    def add_problem(self, problem_id: str, text: str, meta: dict | None = None) -> ProblemNode:
        p = self.problems.get(problem_id)
        if p is None:
            p = ProblemNode(problem_id=problem_id, text=text, meta=dict(meta or {}))
            self.problems[problem_id] = p
        elif text and not p.text:
            p.text = text
        return p

    # ---- admit (live, student; build node + record wal) ----
    def admit_proof(self, problem_id: str, content: str, self_eval: str, self_score: float | None,
                    *, wv: int, source: str = "student") -> ProofNode | None:
        prob = self.problems.get(problem_id)
        if prob is None or len(content or "") > self.max_artifact_chars:
            return None                      # pathologically long proof does not enter the pool (else render blows up -> role starve)
        node = ProofNode(id=self._new_id("p"), problem_id=problem_id, content=content,
                         self_eval=self_eval, self_score=self_score, wv=wv, source=source)
        prob.proofs.append(node)
        self._proof_by_id[node.id] = node
        if source == "student":
            self._sc["prove"] += 1
        self._log({"kind": "proof", "id": node.id, "problem_id": problem_id, "content": content,
                   "self_eval": self_eval, "self_score": self_score, "wv": wv, "source": source})
        return node

    def admit_verify(self, problem_id: str, proof_id: str, score: float | None, text: str,
                     *, wv: int, source: str = "student") -> VerifyNode | None:
        proof = self._proof_by_id.get(proof_id)
        if proof is None or proof.problem_id != problem_id:
            return None
        node = VerifyNode(id=self._new_id("v"), problem_id=problem_id, proof_id=proof_id,
                          score=score, text=text, wv=wv, source=source)
        proof.verifies.append(node)
        if source == "student":
            self._sc["verify"] += 1
        self._log({"kind": "verify", "id": node.id, "problem_id": problem_id, "proof_id": proof_id,
                   "score": score, "text": text, "wv": wv, "source": source})
        return node

    def admit_refined(self, problem_id: str, parent_proof_ids: list, content: str, self_eval: str,
                      self_score: float | None, *, wv: int, source: str = "student") -> RefinedNode | None:
        prob = self.problems.get(problem_id)
        if prob is None or len(content or "") > self.max_artifact_chars:
            return None
        node = RefinedNode(id=self._new_id("r"), problem_id=problem_id,
                           parent_proof_ids=list(parent_proof_ids or []), content=content,
                           self_eval=self_eval, self_score=self_score, wv=wv, source=source)
        prob.refined.append(node)
        if source == "student":
            self._sc["refine"] += 1
        self._log({"kind": "refined", "id": node.id, "problem_id": problem_id,
                   "parent_proof_ids": node.parent_proof_ids, "content": content,
                   "self_eval": self_eval, "self_score": self_score, "wv": wv, "source": source})
        return node

    def admit_select(self, problem_id: str, *, wv: int, source: str = "student") -> None:
        """select has no node (nothing downstream consumes it) -> only count + record one entry for later tracking."""
        if source == "student":
            self._sc["select"] += 1
        self._log({"kind": "select", "problem_id": problem_id, "wv": wv, "source": source})

    # ---- persistence ----
    def _log(self, rec: dict) -> None:
        """Record a student admit into the wal (later the flusher / persist() appends to artifacts.jsonl)."""
        self._wal.append(rec)

    def persist(self) -> int:
        """Append the not-yet-flushed wal records to artifacts.jsonl (synchronous, append-only). Returns the number newly written.

        ★ Concurrency-safe: persist runs on an executor thread while `_log`(append) runs on the event-loop
        thread. First **snapshot `end = len(_wal)`**, write only `[persisted_n:end]`, and advance
        `_persisted_n` only to end — anything the loop appends afterwards lands in the next persist
        (list.append does not invalidate an existing slice; the GIL protects a single append). If instead
        we did `_persisted_n = len(_wal)` (re-reading len after writing), records appended in between would
        be skipped -> data loss.
        """
        with self._persist_lock:        # flusher(executor) and close(loop) won't concurrently write duplicate lines
            end = len(self._wal)
            new = self._wal[self._persisted_n:end]
            if not new:
                return 0
            with open(self.artifacts_path, "a") as f:
                for rec in new:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            self._persisted_n = end
            return len(new)

    def _apply_record(self, rec: dict) -> None:
        """Replay one persisted record into the index (shared by load / seed; the id comes from rec, not newly generated).

        A valid-JSON but corrupt line missing required fields (id/problem_id) -> skip (don't crash load;
        the load side also has a try/except backstop).
        """
        kind = rec.get("kind")
        if kind == "problem":
            if rec.get("problem_id"):
                self.add_problem(rec["problem_id"], rec.get("text") or "", rec.get("meta"))
            return
        if kind in ("proof", "verify", "refined") and (not rec.get("id") or not rec.get("problem_id")):
            return
        student = rec.get("source") == "student"
        if kind == "proof":
            if len(rec.get("content") or "") > self.max_artifact_chars:
                return
            prob = self.problems.get(rec["problem_id"]) or self.add_problem(rec["problem_id"], "")
            node = ProofNode(id=rec["id"], problem_id=rec["problem_id"], content=rec.get("content") or "",
                             self_eval=rec.get("self_eval") or "", self_score=rec.get("self_score"),
                             wv=int(rec.get("wv", -1)), source=rec.get("source") or "deepseek_seed")
            prob.proofs.append(node)
            self._proof_by_id[node.id] = node
            self._bump_ctr_from_id(node.id)
            if student:
                self._sc["prove"] += 1
        elif kind == "verify":
            proof = self._proof_by_id.get(rec.get("proof_id"))
            if proof is not None:
                node = VerifyNode(id=rec["id"], problem_id=rec["problem_id"], proof_id=rec["proof_id"],
                                  score=rec.get("score"), text=rec.get("text") or "",
                                  wv=int(rec.get("wv", -1)), source=rec.get("source") or "deepseek_seed")
                proof.verifies.append(node)
                self._bump_ctr_from_id(node.id)
                if student:
                    self._sc["verify"] += 1
        elif kind == "refined":
            if len(rec.get("content") or "") > self.max_artifact_chars:
                return
            prob = self.problems.get(rec["problem_id"]) or self.add_problem(rec["problem_id"], "")
            node = RefinedNode(id=rec["id"], problem_id=rec["problem_id"],
                               parent_proof_ids=rec.get("parent_proof_ids") or [],
                               content=rec.get("content") or "", self_eval=rec.get("self_eval") or "",
                               self_score=rec.get("self_score"), wv=int(rec.get("wv", -1)),
                               source=rec.get("source") or "deepseek_seed")
            prob.refined.append(node)
            self._bump_ctr_from_id(node.id)
            if student:
                self._sc["refine"] += 1
        elif kind == "select":
            if student:
                self._sc["select"] += 1

    def load(self) -> dict:
        """Startup replay: seed.jsonl (cold-start, immutable) + artifacts.jsonl (student, resumable). Returns stats."""
        for path in (self.seed_path, self.artifacts_path):
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._apply_record(json.loads(line))
                    except Exception:   # noqa: BLE001 — tolerate half lines / bad fields, skip that line without crashing load (resume resilience)
                        continue
        # artifacts.jsonl is now all on disk -> mark as persisted (the wal only accumulates new student admits from here)
        self._wal.clear()
        self._persisted_n = 0
        st = self.stats()
        log.info("pool loaded: %d problems, proofs=%d verifies=%d refined=%d (student p/v/r/s=%d/%d/%d/%d)",
                 st["n_problems"], st["n_proofs"], st["n_verifies"], st["n_refined"],
                 st["student"]["prove"], st["student"]["verify"], st["student"]["refine"],
                 st["student"]["select"])
        return st

    # ---- background flusher (live; non-blocking periodic persist) ----
    def start(self, flush_interval_s: float = 30.0) -> None:
        import asyncio
        self._loop = asyncio.get_running_loop()
        self._flusher = asyncio.create_task(self._run_flusher(flush_interval_s), name="pool_flusher")

    async def _run_flusher(self, interval: float) -> None:
        import asyncio
        try:
            while True:
                await asyncio.sleep(interval)
                n = await self._loop.run_in_executor(None, self.persist)
                if n:
                    log.debug("pool persisted %d records", n)
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        import asyncio
        if self._flusher is not None:
            self._flusher.cancel()
            try:
                await self._flusher
            except (asyncio.CancelledError, Exception):
                pass
            self._flusher = None
        self.persist()

    # ---- counts ----
    def student_counts(self) -> dict:
        """O(1): incrementally maintained (updated on admit/_apply_record) -> next_prompt need not scan the whole pool each time (P1)."""
        return dict(self._sc)

    def stats(self) -> dict:
        n_proofs = n_verifies = n_refined = 0
        for prob in self.problems.values():
            n_proofs += len(prob.proofs)
            n_refined += len(prob.refined)
            for p in prob.proofs:
                n_verifies += len(p.verifies)
        return {"n_problems": len(self.problems), "n_proofs": n_proofs,
                "n_verifies": n_verifies, "n_refined": n_refined,
                "student": self.student_counts(), "wal_pending": len(self._wal) - self._persisted_n}

    # ---- availability / item-selection (used by the sampler; simple scan, cheap because atoms are sparse under long-CoT) ----
    def available_roles(self, cfg) -> set:
        """Which roles are currently samplable (have valid context). prove is always available (any problem suffices).

        verify's "samplable" matches pick_verify_target: there exists a proof with **student-verify count < cap**
        (the cap only counts student verifies -> a seed proof's seed verify does not consume the cap, so
        cold-start can still do on-policy verify on seed proofs; as student proofs accumulate, prefer-student
        naturally transfers).
        """
        cap_v = cfg.agentic.max_verifies_per_proof
        roles = set()
        if self.problems:
            roles.add("prove")
        for prob in self.problems.values():
            if any(p.n_verifies(student_only=True) < cap_v for p in prob.proofs):
                roles.add("verify")
            if prob.has_verified_proof():
                roles.add("refine")
            if len(prob.refined) >= 2:
                roles.add("select")
            if {"verify", "refine", "select"} <= roles:
                break
        return roles

    def pick_prove_problem(self, cfg, rng: random.Random) -> ProblemNode | None:
        """Spread: pick randomly among the problems with the fewest student proofs (< max_proofs_per_problem preferred)."""
        if not self.problems:
            return None
        cap = cfg.agentic.max_proofs_per_problem
        probs = list(self.problems.values())
        under = [p for p in probs if sum(1 for x in p.proofs if x.source == "student") < cap]
        pool = under or probs
        m = min(sum(1 for x in p.proofs if x.source == "student") for p in pool)
        cands = [p for p in pool if sum(1 for x in p.proofs if x.source == "student") == m]
        return rng.choice(cands)

    def pick_verify_target(self, cfg, rng: random.Random) -> tuple[ProblemNode, ProofNode] | None:
        """Pick the proof with the fewest student-verifies (spread, to avoid piling up un-verified proofs); prefer student-source proof.

        The cap only counts **student** verifies (consistent with available_roles) -> a seed proof (0 student
        verifies) is samplable at cold-start for on-policy verify; as student proofs appear (with a smaller
        not_student=0 key) they are preferentially verified.
        """
        cap = cfg.agentic.max_verifies_per_proof
        prefer = cfg.agentic.prefer_student_context
        best = None  # (key, proof, problem)
        n_tie = 0    # reservoir: uniformly random among the min-key ties (avoid deterministically always picking the same proof -> verify concentration)
        for prob in self.problems.values():
            for p in prob.proofs:
                nv = p.n_verifies(student_only=True)
                if nv >= cap:
                    continue
                key = (nv, 0 if (p.source == "student" or not prefer) else 1)
                if best is None or key < best[0]:
                    best = (key, p, prob); n_tie = 1
                elif key == best[0]:
                    n_tie += 1
                    if rng.random() < 1.0 / n_tie:
                        best = (key, p, prob)
        if best is None:
            return None
        return best[2], best[1]

    def pick_refine_problem(self, cfg, rng: random.Random) -> ProblemNode | None:
        """Pick a problem that has a verified proof and the fewest student-refined (< max_refined_per_problem preferred)."""
        cap = cfg.agentic.max_refined_per_problem
        cands = [p for p in self.problems.values() if p.has_verified_proof()]
        if not cands:
            return None
        under = [p for p in cands if sum(1 for r in p.refined if r.source == "student") < cap]
        pool = under or cands
        m = min(sum(1 for r in p.refined if r.source == "student") for p in pool)
        return rng.choice([p for p in pool if sum(1 for r in p.refined if r.source == "student") == m])

    def pick_select_problem(self, cfg, rng: random.Random) -> ProblemNode | None:
        """Pick a problem with ≥2 refined, **randomly** among eligible ones (diversity; avoid concentrating select training on a few problems)."""
        cands = [p for p in self.problems.values() if len(p.refined) >= 2]
        return rng.choice(cands) if cands else None
