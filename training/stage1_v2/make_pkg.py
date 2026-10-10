#!/usr/bin/env python3
# Copyright 2026 proof-pilot. Apache-2.0.
"""Materialize the FMI container payload for stage1_v2 (single-source layout).

The repo keeps exactly ONE hand-edited copy of every shared module (top-level
`olmo3_sink/`, `train_core/`). The container needs a flat self-contained /app, so
the copy is made HERE, at packaging time: `--write` regenerates `build/`
(gitignored) from `pkg.manifest` and stamps `build/PROVENANCE.json` (git commit +
dirty flag + per-file sha256) so any .sif can be traced back to exact sources.

Usage:
    python make_pkg.py --write     # regenerate build/ from pkg.manifest
    python make_pkg.py --check     # verify build/ matches current sources; exit 1 if stale

ALWAYS run --write immediately before `apptainer build` (the .def copies ./build/*).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent        # training/stage1_v2
ROOT = HERE.parent.parent                     # repo root
MANIFEST = HERE / "pkg.manifest"
BUILD = HERE / "build"


def parse_manifest() -> list[tuple[Path, Path, str, str]]:
    entries = []
    for raw in MANIFEST.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        src, sep, dst = line.partition("->")
        src, dst = src.strip(), dst.strip()
        if not (sep and src and dst):
            raise ValueError(f"bad manifest line: {raw!r}")
        sp = ROOT / src
        if not sp.is_file():
            raise FileNotFoundError(f"manifest source missing: {sp}")
        entries.append((sp, BUILD / dst, src, dst))
    if not entries:
        raise ValueError(f"empty manifest: {MANIFEST}")
    return entries


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def write() -> None:
    entries = parse_manifest()
    if BUILD.exists():
        shutil.rmtree(BUILD)
    prov = {
        "package": "stage1_v2",
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "files": {},
    }
    for sp, dp, src, dst in entries:
        dp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, dp)
        prov["files"][dst] = {"source": src, "sha256": sha256(sp)}
    (BUILD / "PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n")
    dirty = " (DIRTY working tree!)" if prov["git_dirty"] else ""
    print(f"[make_pkg] wrote {len(entries)} files -> {BUILD}  @ {prov['git_commit'][:12]}{dirty}")


def check() -> None:
    if not BUILD.exists():
        sys.exit("[make_pkg] FAIL: build/ does not exist -- run `make_pkg.py --write` first")
    problems = []
    for sp, dp, src, dst in parse_manifest():
        if not dp.is_file():
            problems.append(f"missing in build/: {dst}")
        elif sha256(sp) != sha256(dp):
            problems.append(f"STALE: build/{dst} != {src}")
    if problems:
        print("\n".join(f"[make_pkg] {p}" for p in problems))
        sys.exit(f"[make_pkg] FAIL: build/ is stale -- run `make_pkg.py --write`")
    print(f"[make_pkg] OK: build/ matches all {len(parse_manifest())} manifest sources")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="regenerate build/")
    g.add_argument("--check", action="store_true", help="verify build/ against sources")
    args = ap.parse_args()
    write() if args.write else check()


if __name__ == "__main__":
    main()
