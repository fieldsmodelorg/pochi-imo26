#!/usr/bin/env python3
"""Upload the unified L2 dataset to a HuggingFace dataset repo, then verify integrity.

Run `hf auth login` (or set HF_TOKEN) first. This pushes:
  - datasets/training_l2/  (parquet shards + README.md dataset card)
  - the reproduction scripts -> scripts/ in the repo

Integrity: because this host can flip bytes in RAM under load, the upload path itself is
not fully trusted. After upload we compare, for a sample of shards, the LOCAL sha256
(computed twice, must agree -> guards the local read) against the REMOTE LFS sha256
reported by the Hub (-> guards the upload transfer). Any mismatch is reported loudly.

Usage:
  python scripts/td_upload_hf.py --repo <user>/<name> [--private] [--sample 8] [--no-upload]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys

from huggingface_hub import HfApi


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. yourname/unified-deepseek-sft")
    ap.add_argument("--data", default="datasets/training_l2")
    ap.add_argument("--scripts", nargs="*",
                    default=["scripts/td_normalize.py", "scripts/td_build_l2.py"])
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--sample", type=int, default=8, help="shards to integrity-check")
    ap.add_argument("--no-upload", action="store_true", help="only run integrity check")
    args = ap.parse_args()

    api = HfApi()
    repo_type = "dataset"

    if not args.no_upload:
        api.create_repo(args.repo, repo_type=repo_type, private=args.private, exist_ok=True)
        print(f"[upload] data folder {args.data} -> {args.repo}", flush=True)
        api.upload_large_folder(
            repo_id=args.repo, repo_type=repo_type, folder_path=args.data,
        )
        for sp in args.scripts:
            if os.path.exists(sp):
                api.upload_file(
                    path_or_fileobj=sp, path_in_repo=f"scripts/{os.path.basename(sp)}",
                    repo_id=args.repo, repo_type=repo_type)
                print(f"[upload] {sp} -> scripts/{os.path.basename(sp)}", flush=True)

    # ---- integrity check: local sha256 (x2) vs remote LFS sha256 ----
    print("[verify] fetching remote file metadata ...", flush=True)
    info = api.repo_info(args.repo, repo_type=repo_type, files_metadata=True)
    remote_sha = {}
    for s in info.siblings:
        if s.rfilename.endswith(".parquet") and s.lfs is not None:
            remote_sha[s.rfilename] = s.lfs.sha256

    local_parquets = []
    for dp, _, fns in os.walk(args.data):
        for fn in fns:
            if fn.endswith(".parquet"):
                p = os.path.join(dp, fn)
                local_parquets.append((os.path.relpath(p, args.data), p))

    random.seed(0)
    sample = random.sample(local_parquets, min(args.sample, len(local_parquets)))
    bad = 0
    missing = 0
    for rel, path in sample:
        if rel not in remote_sha:
            print(f"  MISSING on hub: {rel}")
            missing += 1
            continue
        s1 = sha256_file(path)
        s2 = sha256_file(path)
        if s1 != s2:
            print(f"  LOCAL READ UNSTABLE (flaky RAM?) re-run: {rel}")
            bad += 1
            continue
        if s1 != remote_sha[rel]:
            print(f"  SHA MISMATCH (upload corrupted?): {rel}\n    local={s1}\n    remote={remote_sha[rel]}")
            bad += 1
        else:
            print(f"  ok  {rel}")

    n_remote = len(remote_sha)
    print(f"\n[verify] sampled {len(sample)} shards | local parquet files={len(local_parquets)} "
          f"remote parquet files={n_remote} | mismatches={bad} missing={missing}")
    if bad or missing or n_remote != len(local_parquets):
        print("VERIFICATION FAILED — re-upload the affected files.")
        sys.exit(1)
    print("PASS — sampled shards byte-exact on the Hub; file counts match.")


if __name__ == "__main__":
    main()
