#!/usr/bin/env python3
"""Verify downloaded model folders against HuggingFace's recorded hashes.

For every file HF has for a model, this compares the LOCAL file's SHA-256 against
the sha256 HF stores (the LFS `oid`, present even on Xet-backed repos). Small
non-LFS files (config.json, tokenizer_config.json, ...) have no sha256 on the
API, so they are checked by exact byte size instead. This catches truncated
downloads, silently corrupted shards, and missing files -- things a plain
`hf download` re-run will NOT re-verify once it has recorded a file as complete.

Usage (on the node, inside the venv so `python` is the bundled interpreter):

    source /tmp/chankhavu/venvs/infervenv/.runtime/activate-env.sh
    python install/verify_models.py                       # checks /tmp/chankhavu/models
    python install/verify_models.py /path/to/models       # custom root
    python install/verify_models.py --only opd-32b-deploy # one folder

Exit status is nonzero if anything fails to verify, so it is safe to gate a
launch on it.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Map a local subfolder name -> (hf repo, path-in-repo). Every bf16 target
# checkpoint (step-* and merged-*) lives in the FM-Pochi-32B-IMO26 repo under an
# identically named subfolder (handled below); a few specials are pinned in KNOWN.
KNOWN = {
    "opd-32b-deploy": ("fieldsmodelorg/FM-Pochi-32B-ProofPilot", "opd-32b-deploy"),
    "dflash-32b-draft-v2test-phaseL": (
        "fieldsmodelorg/FM-Pochi-32B-ProofPilot",
        "dflash-32b-draft-v2test-phaseL",
    ),
}
IMO_REPO = "fieldsmodelorg/FM-Pochi-32B-IMO26"

DEFAULT_ROOT = "/tmp/chankhavu/models"
CHUNK = 16 * 1024 * 1024


def resolve_source(subfolder: str):
    """Return (repo, path_in_repo) for a local subfolder, or None if unknown."""
    if subfolder in KNOWN:
        return KNOWN[subfolder]
    if subfolder.startswith("opd-32b-bf16-"):  # step-* and merged-* checkpoints
        return (IMO_REPO, subfolder)
    return None


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or unit == "TiB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{n} B"


def fetch_manifest(repo: str, path_in_repo: str, revision: str) -> dict:
    """{relpath_within_subfolder: {'size': int, 'sha256': str|None}} from HF."""
    out: dict[str, dict] = {}
    base = f"https://huggingface.co/api/models/{repo}/tree/{revision}/{path_in_repo}"
    url = f"{base}?recursive=true&expand=true"
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "verify-models/1"})
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
            link = resp.headers.get("Link", "")
        for e in data:
            if e.get("type") != "file":
                continue
            rel = e["path"]
            prefix = path_in_repo.rstrip("/") + "/"
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
            lfs = e.get("lfs") or {}
            out[rel] = {"size": e.get("size"), "sha256": lfs.get("oid")}
        # follow pagination if present (won't trigger for a flat model folder)
        url = ""
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.find("<") + 1 : part.find(">")]
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_folder(root: Path, subfolder: str, revision: str, workers: int) -> bool:
    src = resolve_source(subfolder)
    folder = root / subfolder
    print(f"\n=== {subfolder} ===")
    if src is None:
        print("  SKIP  unknown folder (no HF mapping); pass --repo to check it")
        return True
    repo, path_in_repo = src
    print(f"  source  {repo}@{revision}:{path_in_repo}")
    try:
        manifest = fetch_manifest(repo, path_in_repo, revision)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  ERROR   could not fetch manifest: {e}")
        return False
    if not manifest:
        print("  ERROR   HF returned an empty manifest (bad repo/path/revision?)")
        return False

    local_files = {
        str(p.relative_to(folder)): p
        for p in folder.rglob("*")
        if p.is_file() and ".cache/" not in str(p.relative_to(folder))
    }

    missing = [r for r in manifest if r not in local_files]
    extra = [r for r in local_files if r not in manifest]

    # size gate first (cheap) -- only hash files whose size already matches
    to_hash, size_bad, size_ok_nolfs = [], [], []
    for rel, meta in manifest.items():
        if rel in missing:
            continue
        actual = local_files[rel].stat().st_size
        if meta["size"] is not None and actual != meta["size"]:
            size_bad.append((rel, actual, meta["size"]))
        elif meta["sha256"]:
            to_hash.append(rel)
        else:
            size_ok_nolfs.append(rel)  # non-LFS: size match is all we can check

    hashed_bad, total_bytes = [], 0
    if to_hash:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_sha256, str(local_files[r])): r for r in to_hash}
            for fut in cf.as_completed(futs):
                rel = futs[fut]
                got = fut.result()
                want = manifest[rel]["sha256"]
                total_bytes += manifest[rel]["size"] or 0
                if got != want:
                    hashed_bad.append((rel, got, want))

    ok = not (missing or size_bad or hashed_bad)
    good = len(to_hash) - len(hashed_bad) + len(size_ok_nolfs)
    print(
        f"  {'OK   ' if ok else 'FAIL '} "
        f"{good}/{len(manifest)} files verified "
        f"({len(to_hash) - len(hashed_bad)} by sha256 = {human(total_bytes)}, "
        f"{len(size_ok_nolfs)} by size)"
    )
    for rel in missing:
        print(f"    MISSING   {rel}")
    for rel, a, w in size_bad:
        print(f"    SIZE      {rel}: {a} B on disk, expected {w} B")
    for rel, got, want in hashed_bad:
        print(f"    SHA256    {rel}: {got[:12]}… != {want[:12]}…")
    for rel in extra:
        print(f"    extra     {rel} (present locally, not in manifest -- ignored)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default=DEFAULT_ROOT, help="models dir")
    ap.add_argument("--only", action="append", help="check only this subfolder (repeatable)")
    ap.add_argument("--revision", default="main", help="HF revision (default: main)")
    ap.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 4),
        help="parallel hash workers",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"no such directory: {root}", file=sys.stderr)
        return 2

    subs = args.only or sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not subs:
        print(f"no model subfolders under {root}", file=sys.stderr)
        return 2

    print(f"verifying {len(subs)} folder(s) under {root} against HF (rev {args.revision})")
    results = {s: verify_folder(root, s, args.revision, args.workers) for s in subs}

    print("\n=== summary ===")
    for s, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {s}")
    n = len(results)
    n_ok = sum(1 for ok in results.values() if ok)
    checked = ", ".join(results)
    # One self-contained final line: survives a truncated `opd-status` tail and
    # names every folder that was checked, so "did it do all of them?" is answered.
    if n_ok == n:
        print(f"\nRESULT: {n_ok}/{n} folders verified OK  [{checked}]")
    else:
        bad = ", ".join(s for s, ok in results.items() if not ok)
        print(f"\nRESULT: FAILED — {n - n_ok}/{n} bad: {bad}  (checked: {checked})")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
