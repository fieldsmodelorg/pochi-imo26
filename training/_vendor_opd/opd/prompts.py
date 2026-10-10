# Copyright 2026 proof-pilot. Apache-2.0.
"""A "prompt-only" loader.

OPD rollouts need the **prompt token_ids** (token-in-token-out): read the OpenAI-style
`messages` from the L2 parquet, take everything up to the first assistant (system + user turns), and
render it into input_ids for the rollout server using the **student's chat template**
(`add_generation_prompt=True`). All answer/assistant content is discarded — OPD learns on the
teacher's distribution and does not need a reference solution.

Domain selection reuses data_mix's partition scan; by default it takes math/proof/science (the
evaluation domains) and excludes agentic (multi-turn tool use, unrelated to the proof objective).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Iterator, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_mix import partitions  # noqa: E402  (reuse the partition scan)

# Domains taken by default (partition key = "<dataset>/<domain>"); math/proof/science.
DEFAULT_INCLUDE = ["*/math_cot", "*/math_tir", "*/math*", "*/proof*", "*/*verification*", "*/science*"]
DEFAULT_EXCLUDE = ["*/agent*", "*/swe*", "*/tool*"]


@dataclass
class Prompt:
    id: str
    input_ids: list[int]          # the rendered prompt (with generation prompt), token-in-token-out
    domain: str
    meta: dict


def _select_shards(roots: list[Path], include: list[str], exclude: list[str]) -> list[tuple[str, Path]]:
    """Return [(partition_key, shard_path)], filtering partitions by the include/exclude globs."""
    out: list[tuple[str, Path]] = []
    for root in roots:
        for pk, shards in sorted(partitions(root).items()):
            if not any(fnmatchcase(pk, g) for g in include):
                continue
            if any(fnmatchcase(pk, g) for g in exclude):
                continue
            out.extend((pk, s) for s in shards)
    if not out:
        raise FileNotFoundError(f"no partitions matched include={include} exclude={exclude}")
    return out


def _prompt_messages(messages: list[dict]) -> Optional[list[dict]]:
    """Take the messages up to the first assistant (system + user turns); None if there is no user."""
    pre: list[dict] = []
    for m in messages:
        if m.get("role") == "assistant":
            break
        pre.append({"role": m["role"], "content": m.get("content") or ""})
    if not any(m["role"] == "user" for m in pre):
        return None
    return pre


class PromptLoader:
    """Stream L2 prompts, rendering them into student-tokenizer input_ids."""

    def __init__(self, student_path: str, roots: list[str],
                 include: Optional[list[str]] = None, exclude: Optional[list[str]] = None,
                 max_prompt_tokens: int = 8192, seed: int = 0):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
        self.roots = [Path(r) for r in roots]
        self.include = include or DEFAULT_INCLUDE
        self.exclude = exclude or DEFAULT_EXCLUDE
        self.max_prompt_tokens = max_prompt_tokens
        self.seed = seed
        self.shards = _select_shards(self.roots, self.include, self.exclude)

    def render(self, messages: list[dict]) -> Optional[list[int]]:
        pre = _prompt_messages(messages)
        if pre is None:
            return None
        # Render to a string first, then encode (add_special_tokens=False, since the DeepSeek chat
        # template carries its own <｜begin▁of▁sentence｜>), guaranteeing a list[int] and avoiding the
        # BatchEncoding from tokenize=True.
        text = self.tok.apply_chat_template(pre, add_generation_prompt=True, tokenize=False)
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) > self.max_prompt_tokens:
            return None   # drop over-long prompts (simple Phase-0 handling; can switch to capping later)
        return ids

    def iter_prompts(self, shuffle: bool = True) -> Iterator[Prompt]:
        import random

        import pyarrow.parquet as pq

        shards = list(self.shards)
        if shuffle:
            random.Random(self.seed).shuffle(shards)
        for pk, path in shards:
            # Read the single file directly (no dataset/partitioning inference, to avoid a hive partition
            # column colliding with a real column of the same name inside the file)
            tbl = pq.ParquetFile(str(path)).read(columns=["id", "domain", "messages", "meta"])
            for row in tbl.to_pylist():
                try:
                    msgs = json.loads(row["messages"])
                except (TypeError, json.JSONDecodeError):
                    continue
                ids = self.render(msgs)
                if ids is None:
                    continue
                meta = {}
                try:
                    meta = json.loads(row["meta"]) if row.get("meta") else {}
                except json.JSONDecodeError:
                    pass
                yield Prompt(id=row["id"], input_ids=ids, domain=row.get("domain") or pk, meta=meta)


# ============================================================================
# Problem-pool loader (distill_gen/problems/problems.parquet, 9,834 deduplicated olympiad proof problems)
# ============================================================================
# The prompt corpus used for real OPD training: apply a prover template to the problem text -> student
# chat template -> input_ids. Rotates through the same 3 prover styles as distill_gen (diversity).
# A template with {problem} -> a single user message; without it (imo25) -> template as system, problem
# as the user turn. **Use .replace, not .format** (dsmv2 contains literal braces like \boxed{...}).

_PROVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..", "..", "distill_gen", "prompts"))
DEFAULT_TEMPLATE_POOL = ("proofbench_generator", "dsmv2_a1", "imo25_prover")


class ProblemPromptLoader:
    """Stream problems from problems.parquet, apply a prover template, render into student input_ids (token-in-token-out)."""

    def __init__(self, student_path: str, problems_parquet: str,
                 template_pool: Optional[list[str]] = None, template_dir: str = _PROVER_DIR,
                 max_prompt_tokens: int = 4096, seed: int = 0):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
        self.parquet = problems_parquet
        self.max_prompt_tokens = max_prompt_tokens
        self.seed = seed
        self.pool = list(template_pool or DEFAULT_TEMPLATE_POOL)
        self.templates = {}
        for name in self.pool:
            with open(os.path.join(template_dir, f"{name}.txt"), encoding="utf-8") as f:
                self.templates[name] = f.read()

    def render(self, problem: str, template_name: str) -> Optional[list[int]]:
        text = self.templates[template_name]
        if "{problem}" in text:
            msgs = [{"role": "user", "content": text.replace("{problem}", problem)}]
        else:                                   # imo25: template as system, problem as the user turn
            msgs = [{"role": "system", "content": text}, {"role": "user", "content": problem}]
        rendered = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = self.tok.encode(rendered, add_special_tokens=False)
        if len(ids) > self.max_prompt_tokens:
            return None
        return ids

    def iter_prompts(self, shuffle: bool = True) -> Iterator[Prompt]:
        import random

        import pyarrow.parquet as pq

        probs = pq.ParquetFile(self.parquet).read(columns=["problem", "origin", "category"]).to_pylist()
        order = list(range(len(probs)))
        if shuffle:
            random.Random(self.seed).shuffle(order)
        for j, i in enumerate(order):
            row = probs[i]
            problem = row.get("problem")
            if not problem:
                continue
            tname = self.pool[(i + self.seed) % len(self.pool)]   # deterministic template rotation
            ids = self.render(problem, tname)
            if ids is None:
                continue
            yield Prompt(id=f"prob-{i}", input_ids=ids,
                         domain=row.get("category") or row.get("origin") or "proof",
                         meta={"template": tname, "origin": row.get("origin")})


def make_prompt_loader(cfg):
    """Return the loader matching cfg.prompt_source ('problems' = distill_gen problem bank; 'l2' = L2 messages)."""
    src = getattr(cfg, "prompt_source", "problems")
    if src == "problems":
        return ProblemPromptLoader(cfg.rollout.model_path, cfg.problems_parquet,
                                   template_pool=list(cfg.prover_template_pool),
                                   seed=int(getattr(cfg, "prompt_seed", 0)))
    return PromptLoader(cfg.rollout.model_path, list(cfg.dataset_roots),
                        seed=int(getattr(cfg, "prompt_seed", 0)))


def _selftest():
    """Render one row on real data, printing the token count and the decoded prompt tail (to confirm the generation prompt)."""
    from opd.config import STUDENT_PATH, OPDConfig

    cfg = OPDConfig()
    ld = PromptLoader(STUDENT_PATH, list(cfg.dataset_roots))
    print(f"selected partitions/shards: {len(ld.shards)}")
    it = ld.iter_prompts()
    for i, p in enumerate(it):
        tail = ld.tok.decode(p.input_ids[-40:])
        print(f"\n[{p.domain}] id={p.id[:12]} prompt_tokens={len(p.input_ids)}")
        print(f"  tail decode: ...{tail!r}")
        if i >= 2:
            break
    print("\nprompts selftest OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    _selftest()
