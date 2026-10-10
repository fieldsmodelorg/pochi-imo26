# Copyright 2026 proof-pilot. Apache-2.0.
"""cold-start seed — fill the pool with all DeepSeek **r3_hard2000 nested data** (proofs->verifies, refined).

Written to `<pool_dir>/seed.jsonl` (immutable; replayed by PoolStore.load() at startup). This lets all four
roles sample context from step 0 (otherwise only prove is samplable early on). The seed is DeepSeek-source
(off-policy context); as the student generates + the ring-free prefer-student selection kicks in, the pool
naturally drifts to student-dominated (depth/fill only count student -> seed does not satisfy student depth).

Sources (cfg.agentic.seed_format):
- "hf_per_problem": the HF dataset's per_problem config (one row per problem, nested `proofs_json`/`refined_json`,
  each artifact's `content` is raw XML -> parse solution/score with the math_3r parser). **Already cached locally, no download.**
- "records_jsonl": math_3r run.py's records.jsonl (one row per problem, `stages.{prove,verify,refine}`).

gate (same as writeback): **only parse-passing artifacts enter the seed** (truncated / malformed DeepSeek
generations are not used as context). select does not enter the pool (nothing downstream consumes it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys

log = logging.getLogger("opd_v2.agentic.seed")

_M3R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "distill_gen", "math_3r"))
if _M3R not in sys.path:
    sys.path.insert(0, _M3R)


def _pid_of(problem: str) -> str:
    return hashlib.blake2b(problem.encode(), digest_size=8).hexdigest()


class _IdGen:
    def __init__(self):
        self.c = {"p": 0, "v": 0, "r": 0}

    def new(self, kind: str) -> str:
        self.c[kind] += 1
        return f"{kind}{self.c[kind]}"


def _proof_artifact(content: str):
    """raw proof XML -> (solution, self_eval, self_score) or None (parse-gate)."""
    from parser import parse_proof_package
    pkg = parse_proof_package({"content": content or "", "finish_reason": "stop", "error": None}, "P0")
    if not pkg.valid:
        return None
    return pkg.proof, pkg.self_eval, pkg.self_score


def _refined_artifact(content: str):
    from parser import parse_refined_package
    pkg = parse_refined_package({"content": content or "", "finish_reason": "stop", "error": None}, "R0")
    if not pkg.valid:
        return None
    return pkg.proof, pkg.self_eval, pkg.self_score


def _emit_problem(rows_out: list, ids: _IdGen, problem_id: str, text: str,
                  proofs: list, refined: list, *, wv: int = -1, source: str = "deepseek_seed") -> dict:
    """Flatten one problem's (proofs[{content, verifications:[{score,content}]}], refined[{content}]) into records.

    Returns that problem's counts."""
    rows_out.append({"kind": "problem", "problem_id": problem_id, "text": text})
    n_p = n_v = n_r = 0
    for p in proofs:
        art = _proof_artifact(p.get("content"))
        if art is None:
            continue
        pid = ids.new("p")
        rows_out.append({"kind": "proof", "id": pid, "problem_id": problem_id, "content": art[0],
                         "self_eval": art[1], "self_score": art[2], "wv": wv, "source": source})
        n_p += 1
        for v in p.get("verifications") or []:
            score = v.get("score")
            if score is None:
                continue
            rows_out.append({"kind": "verify", "id": ids.new("v"), "problem_id": problem_id,
                             "proof_id": pid, "score": score, "text": v.get("content") or "",
                             "wv": wv, "source": source})
            n_v += 1
    for r in refined or []:
        art = _refined_artifact(r.get("content"))
        if art is None:
            continue
        rows_out.append({"kind": "refined", "id": ids.new("r"), "problem_id": problem_id,
                         "parent_proof_ids": [], "content": art[0], "self_eval": art[1],
                         "self_score": art[2], "wv": wv, "source": source})
        n_r += 1
    return {"proofs": n_p, "verifies": n_v, "refined": n_r}


def _iter_hf_per_problem(repo: str, config: str):
    from datasets import load_dataset
    ds = load_dataset(repo, config, split="train")
    for row in ds:
        problem = row.get("problem")
        if not problem:
            continue
        pid = row.get("problem_id") or _pid_of(problem)
        proofs = json.loads(row["proofs_json"]) if row.get("proofs_json") else []
        refined = json.loads(row["refined_json"]) if row.get("refined_json") else []
        yield pid, problem, proofs, refined


def _iter_records_jsonl(path: str):
    """math_3r records.jsonl: each row has `stages.{prove,verify,refine}`, verify links back to a proof via candidate_id."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem = rec.get("problem")
            if not problem:
                continue
            st = rec.get("stages") or {}
            by_cand: dict = {}
            for v in st.get("verify", []):
                by_cand.setdefault(v.get("candidate_id"), []).append(
                    {"score": v.get("score"), "content": v.get("content") or ""})
            proofs = [{"content": p.get("content") or "",
                       "verifications": by_cand.get(p.get("candidate_id"), [])}
                      for p in st.get("prove", [])]
            refined = [{"content": r.get("content") or ""} for r in st.get("refine", [])]
            yield (rec.get("problem_id") or _pid_of(problem)), problem, proofs, refined


def build_seed(cfg, *, force: bool = False) -> str:
    """Build <pool_dir>/seed.jsonl (skip if it exists and is non-empty and not force). Returns the path."""
    pool_dir = cfg.pool_dir
    os.makedirs(pool_dir, exist_ok=True)
    out = os.path.join(pool_dir, "seed.jsonl")
    if os.path.exists(out) and os.path.getsize(out) > 0 and not force:
        log.info("seed.jsonl already exists (%s) — skip (use force to rebuild)", out)
        return out

    ag = cfg.agentic
    if ag.seed_format == "hf_per_problem":
        src = _iter_hf_per_problem(ag.seed_source, ag.seed_hf_config)
    elif ag.seed_format == "records_jsonl":
        src = _iter_records_jsonl(ag.seed_source)
    else:
        raise ValueError(f"unknown seed_format {ag.seed_format!r}")

    ids = _IdGen()
    rows: list = []
    n_prob = 0
    tot = {"proofs": 0, "verifies": 0, "refined": 0}
    for pid, problem, proofs, refined in src:
        c = _emit_problem(rows, ids, pid, problem, proofs, refined)
        n_prob += 1
        for k in tot:
            tot[k] += c[k]

    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
    os.replace(tmp, out)
    log.info("seed built: %d problems, proofs=%d verifies=%d refined=%d -> %s",
             n_prob, tot["proofs"], tot["verifies"], tot["refined"], out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="build agentic OPD pool seed.jsonl from DeepSeek nested data")
    ap.add_argument("--run-dir", default=os.environ.get("OPD_RUN_DIR", ""))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not a.run_dir:
        raise SystemExit("--run-dir (or OPD_RUN_DIR) required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    from opd_v2.config import OPDConfig
    # use config.json if it exists (seed settings consistent with the run); otherwise use defaults (for
    # **pre-building** seed.jsonl on the login node — avoids the headless run's auth/block when connecting to
    # a private HF repo; default seed_source=canonical HF repo).
    cfg_path = os.path.join(a.run_dir, "config.json")
    cfg = OPDConfig.load(a.run_dir) if os.path.exists(cfg_path) else OPDConfig(run_dir=a.run_dir).resolve()
    build_seed(cfg, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
