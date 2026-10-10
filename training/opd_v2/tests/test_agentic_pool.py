# Copyright 2026 proof-pilot. Apache-2.0.
"""agentic OPD — no-GPU full-chain logic test (no rollout/teacher; pure pool/sampler/roles/writeback/seed).

Verifies:
  1. seed (synthetic records_jsonl) -> build_seed -> PoolStore.load -> correct graph, all deepseek_seed.
  2. parse_artifact gate: valid / truncated / garbage.
  3. sampler: all four roles are samplable (seed provides enough depth); render round-trip (real student tokenizer, the generation prompt ends at <think>).
  4. write-back: admit a student proof -> student_counts updates -> fill_fraction drifts.
  5. persist + replay: after reload the seed+student counts are consistent.

Run: PYTHONPATH=src python tests/test_agentic_pool.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "..", "src"))

from opd_v2.config import OPDConfig

STUDENT = os.environ.get("STUDENT_PATH", "outputs/opd-v2-lc128k-softdistill-v2test")


# ---- synthetic valid XML (passes math_3r validity: not-truncated ∧ has <solution> ∧ score∈{0,.5,1} ∧ len(sol)>500) ----
def _sol(tag: str) -> str:
    body = (f"We prove the claim ({tag}). " * 30)   # > 500 chars
    return (f"<solution>\n{body}\nHence the result follows. $\\boxed{{42}}$\n</solution>\n"
            f"<self_evaluation>\nKey steps are justified; no gaps.\n</self_evaluation>\n<score>1</score>")


def _ver(score: str) -> str:
    return (f"<evaluation>\nThe proof is rigorous; every nontrivial claim is justified.\n</evaluation>\n"
            f"<suggestions>\nMinor: clarify the lemma statement.\n</suggestions>\n<score>{score}</score>")


def _write_synthetic_records(path: str, n_problems: int = 3) -> None:
    with open(path, "w") as f:
        for i in range(n_problems):
            stages = {
                "prove": [{"candidate_id": f"P{j}", "content": _sol(f"prob{i}-P{j}"),
                           "self_score": 1.0, "valid": True} for j in range(3)],
                "verify": [{"candidate_id": f"P{j}", "verifier_idx": k, "score": (1.0 if k == 0 else 0.5),
                            "content": _ver("1" if k == 0 else "0.5")}
                           for j in range(3) for k in range(2)],
                "refine": [{"refiner_id": f"R{j}", "content": _sol(f"prob{i}-R{j}"),
                            "self_score": 1.0, "valid": True} for j in range(2)],
                "select": [{"content": "<selected_id>R0</selected_id>"}],
            }
            f.write(json.dumps({"problem": f"Prove statement number {i}: for all n, P({i},n) holds.",
                                "stages": stages}) + "\n")


def _mk_cfg(run_dir: str, records: str) -> OPDConfig:
    cfg = OPDConfig(run_name="agentic_test", run_dir=run_dir).resolve()
    cfg.producer = "agentic"
    cfg.trainer.student_path = STUDENT
    cfg.agentic.seed_format = "records_jsonl"
    cfg.agentic.seed_source = records
    return cfg


def test_seed_and_load():
    from opd_v2.agentic.pool import PoolStore
    from opd_v2.agentic.seed import build_seed
    with tempfile.TemporaryDirectory() as td:
        rec = os.path.join(td, "records.jsonl")
        _write_synthetic_records(rec, n_problems=3)
        cfg = _mk_cfg(td, rec)
        build_seed(cfg)
        assert os.path.exists(os.path.join(cfg.pool_dir, "seed.jsonl"))
        pool = PoolStore(cfg.pool_dir)
        st = pool.load()
        assert st["n_problems"] == 3, st
        assert st["n_proofs"] == 9, st            # 3 prob × 3 proof
        assert st["n_verifies"] == 18, st         # 9 proof × 2 verify
        assert st["n_refined"] == 6, st           # 3 prob × 2 refined
        # seed does not count as student depth (used for on-policy transfer)
        assert st["student"] == {"prove": 0, "verify": 0, "refine": 0, "select": 0}, st["student"]
        # all four roles samplable
        roles = pool.available_roles(cfg)
        assert roles == {"prove", "verify", "refine", "select"}, roles
    print("  [1] seed + load + availability OK")


def test_parse_gate():
    from opd_v2.agentic.writeback import parse_artifact
    # valid proof
    a = parse_artifact(_sol("x"), "prove", "stop")
    assert a is not None and len(a["content"]) > 500 and a["self_score"] == 1.0
    # truncated -> None (even if the content looks like a proof)
    assert parse_artifact(_sol("x"), "prove", "length") is None
    # garbage -> None
    assert parse_artifact("no tags here, just rambling", "prove", "stop") is None
    # verify: has score -> ok; truncated -> None
    v = parse_artifact(_ver("0.5"), "verify", "stop")
    assert v is not None and v["score"] == 0.5
    assert parse_artifact(_ver("1"), "verify", "length") is None
    # refine goes through the same proof gate
    assert parse_artifact(_sol("r"), "refine", "stop") is not None
    print("  [2] parse_artifact gate (valid/truncated/garbage) OK")


def test_sampler_and_render():
    from opd_v2.agentic.pool import PoolStore
    from opd_v2.agentic.roles import RolePromptBuilder
    from opd_v2.agentic.sampler import PoolSampler
    from opd_v2.agentic.seed import build_seed
    with tempfile.TemporaryDirectory() as td:
        rec = os.path.join(td, "records.jsonl")
        _write_synthetic_records(rec, n_problems=4)
        cfg = _mk_cfg(td, rec)
        build_seed(cfg)
        pool = PoolStore(cfg.pool_dir); pool.load()
        builder = RolePromptBuilder(STUDENT, cfg)
        sampler = PoolSampler(cfg, pool, builder)
        seen = {"prove": 0, "verify": 0, "refine": 0, "select": 0}
        last = {}
        for _ in range(400):
            p = sampler.next_prompt()
            assert p is not None and p.ids, "sampler must always yield a Prompt"
            stage = p.meta["stage"]
            seen[stage] += 1
            last[stage] = p
            # the generation prompt ends at the student's <think> (token-in-token-out then continues generating reasoning)
            tail = builder.tok.decode(p.ids[-8:])
            assert tail.endswith("<think>"), f"{stage} prompt tail={tail!r}"
        for r in ("prove", "verify", "refine", "select"):
            assert seen[r] > 0, f"role {r} never sampled: {seen}"
        # role-specific content checks
        assert "verifier" in builder.tok.decode(last["verify"].ids).lower()
        assert "<candidate" in builder.tok.decode(last["refine"].ids)        # refine bundle
        assert "<candidate" in builder.tok.decode(last["select"].ids)        # select bundle
        # refine/select refs point to real node ids
        assert last["verify"].meta["refs"] and last["verify"].meta["refs"][0].startswith("p")
        print(f"  [3] sampler 4-role + render round-trip OK (seen={seen})")
        return td  # unused (temp already cleaned)


def test_writeback_and_replay():
    from opd_v2.agentic.pool import PoolStore
    from opd_v2.agentic.writeback import parse_artifact
    from opd_v2.agentic.seed import build_seed
    with tempfile.TemporaryDirectory() as td:
        rec = os.path.join(td, "records.jsonl")
        _write_synthetic_records(rec, n_problems=2)
        cfg = _mk_cfg(td, rec)
        build_seed(cfg)
        pool = PoolStore(cfg.pool_dir); pool.load()
        pid = next(iter(pool.problems))
        # simulate write-back: admit a student proof
        art = parse_artifact(_sol("student"), "prove", "stop")
        node = pool.admit_proof(pid, art["content"], art["self_eval"], art["self_score"],
                                wv=5, source="student")
        assert node is not None and node.source == "student"
        # student verify on that proof
        vart = parse_artifact(_ver("1"), "verify", "stop")
        v = pool.admit_verify(pid, node.id, vart["score"], vart["text"], wv=5, source="student")
        assert v is not None
        sc = pool.student_counts()
        assert sc["prove"] == 1 and sc["verify"] == 1, sc
        # fill_fraction drift: verify now has a student artifact
        n_persist = pool.persist()
        assert n_persist == 2, n_persist     # proof + verify wal
        assert os.path.exists(pool.artifacts_path)
        # replay: a new PoolStore loads seed + artifacts -> student counts are still there
        pool2 = PoolStore(cfg.pool_dir); pool2.load()
        sc2 = pool2.student_counts()
        assert sc2["prove"] == 1 and sc2["verify"] == 1, sc2
        # id counter continues (no collision with seed): admit one more -> new id
        n2 = pool2.admit_proof(pid, art["content"], art["self_eval"], art["self_score"],
                               wv=6, source="student")
        assert n2.id != node.id, (n2.id, node.id)
        print(f"  [4] write-back + persist + replay OK (student after replay={sc2})")


if __name__ == "__main__":
    print("agentic OPD no-GPU tests:")
    test_seed_and_load()
    test_parse_gate()
    test_sampler_and_render()
    test_writeback_and_replay()
    print("ALL AGENTIC NO-GPU TESTS PASSED")
