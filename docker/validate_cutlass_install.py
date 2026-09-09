"""Reject a mixed base/CUDA-13 CUTLASS installation, even if versions agree."""

import base64
import hashlib
from importlib.metadata import distribution


def main() -> None:
    packages = [
        distribution(name) for name in (
            "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base",
            "nvidia-cutlass-dsl-libs-cu13",
        )
    ]
    versions = {package.version for package in packages}
    if len(versions) != 1:
        raise RuntimeError(f"CUTLASS package versions differ: {versions}")
    # Both wheels own overlapping paths; CUDA-13 must be the last writer.
    cuda13 = packages[-1]
    checked, mismatches = 0, []
    for record in cuda13.files or []:
        if record.hash is None or record.hash.mode != "sha256":
            continue
        path = record.locate()
        checked += 1
        if not path.is_file():
            mismatches.append(str(record))
            continue
        actual = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
        if actual.decode().rstrip("=") != record.hash.value:
            mismatches.append(str(record))
    if not checked or mismatches:
        raise RuntimeError(
            f"Mixed or incomplete CUDA-13 CUTLASS files: {mismatches}. "
            "Reinstall nvidia-cutlass-dsl-libs-cu13 after all dependency installs."
        )
    print(f"CUTLASS {cuda13.version}: verified {checked} CUDA-13 wheel files")


if __name__ == "__main__":
    main()
