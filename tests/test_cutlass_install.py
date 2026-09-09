import base64
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "docker/validate_cutlass_install.py"
spec = importlib.util.spec_from_file_location("validate_cutlass_install", SOURCE)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class CutlassInstallTests(unittest.TestCase):
    def packages(self, path, expected=b"CUDA13", version="4.5.2"):
        digest = base64.urlsafe_b64encode(hashlib.sha256(expected).digest()).decode().rstrip("=")
        record = SimpleNamespace(
            hash=SimpleNamespace(mode="sha256", value=digest), locate=lambda: path,
        )
        return [
            SimpleNamespace(version=version),
            SimpleNamespace(version=version),
            SimpleNamespace(version=version, files=[record]),
        ]

    def test_accepts_consistent_cuda13_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "binding.py"
            path.write_bytes(b"CUDA13")
            with patch.object(validator, "distribution", side_effect=self.packages(path)):
                validator.main()

    def test_rejects_overwritten_binding_despite_matching_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "binding.py"
            path.write_bytes(b"BASE")
            with patch.object(validator, "distribution", side_effect=self.packages(path)):
                with self.assertRaisesRegex(RuntimeError, "Mixed or incomplete"):
                    validator.main()

    def test_rejects_missing_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.py"
            with patch.object(validator, "distribution", side_effect=self.packages(path)):
                with self.assertRaisesRegex(RuntimeError, "Mixed or incomplete"):
                    validator.main()

    def test_rejects_mixed_versions(self):
        packages = self.packages(Path("unused"))
        packages[0].version = "4.6.2"
        with patch.object(validator, "distribution", side_effect=packages):
            with self.assertRaisesRegex(RuntimeError, "versions differ"):
                validator.main()

    def test_rejects_unverifiable_wheel(self):
        packages = self.packages(Path("unused"))
        packages[-1].files = []
        with patch.object(validator, "distribution", side_effect=packages):
            with self.assertRaisesRegex(RuntimeError, "Mixed or incomplete"):
                validator.main()


if __name__ == "__main__":
    unittest.main()
