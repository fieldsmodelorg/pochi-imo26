from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

from eval_config import active_model, load_config  # noqa: E402
from launch_server import attention_arguments, decode_graph_batches  # noqa: E402

class RuntimeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = REPO / "config.yaml"
        cls.config = load_config(cls.path)

    def test_root_config_is_valid(self):
        self.assertIsInstance(self.config, dict)

    def test_b200_profile_preserves_production_limits(self):
        config = load_config(REPO / "config-b200.yaml")
        production = load_config(REPO / "config-model-step225-budget-xhigh.yaml")
        self.assertEqual(config["search"], production["search"])
        expected_server = dict(production["server"], attention_backend="fa4", page_size=128)
        self.assertEqual(config["server"], expected_server)
        self.assertFalse(config["model"]["dflash"])
        self.assertFalse(config["model"]["quantized"])
        self.assertEqual(config["model"]["kv_cache_dtype"], "auto")
        for gpus, expected_dp in (("2", 1), ("8", 4)):
            with self.subTest(gpus=gpus), mock.patch.dict(os.environ, {"PP_GPU_COUNT": gpus}):
                model = active_model(config)
                self.assertEqual(model.tensor_parallel_size, 2)
                self.assertEqual(model.data_parallel_size, expected_dp)
                self.assertIsNone(model.draft)

    def write_config(self, directory, configure, name="config.yaml"):
        config = copy.deepcopy(self.config)
        configure(config)
        path = Path(directory) / name
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        return path

    def test_quantization_and_dflash_are_independent(self):
        configured_paths = self.config["models"]
        expected = {
            (False, False): ("bf16", "bf16_target", None),
            (True, False): ("humming_w4a8", "quantized_target", None),
            (False, True): ("bf16", "bf16_target", "bf16_draft"),
            (True, True): (
                "humming_w4a8", "quantized_target", "quantized_draft"
            ),
        }
        for flags, values in expected.items():
            with self.subTest(flags=flags):
                config = copy.deepcopy(self.config)
                config["model"]["quantized"], config["model"]["dflash"] = flags
                model = active_model(config)
                self.assertEqual(model.mode, values[0])
                self.assertEqual(model.target, Path(configured_paths[values[1]]))
                expected_draft = (
                    Path(configured_paths[values[2]]) if values[2] else None
                )
                self.assertEqual(model.draft, expected_draft)

    def test_docker_requires_explicit_config(self):
        env = os.environ.copy()
        env["REPO"] = str(REPO)
        env.pop("CONFIG", None)
        result = subprocess.run(
            ["bash", "docker/entrypoint.sh", "bootstrap"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CONFIG is required", result.stderr)

    def test_docker_inspector_preserves_config(self):
        def configure(config):
            config["models"]["bf16_target"] = "/workspace/models/custom-target"
            config["models"]["bf16_draft"] = "/workspace/models/custom-draft"
            config["model"].update(
                tensor_parallel_size=2,
                data_parallel_size=4,
                quantized=False,
                dflash=True,
                kv_cache_dtype="fp8_e4m3",
            )
            config["server"].update(
                host="0.0.0.0", port=31000, context_length=1048576
            )

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, configure)
            source = path.read_text()
            configured = load_config(path)
            self.assertEqual(configured["server"]["context_length"], 1048576)
            self.assertEqual(configured["model"]["kv_cache_dtype"], "fp8_e4m3")
            output = subprocess.check_output(
                [sys.executable, str(REPO / "docker/inspect_config.py"), str(path)],
                cwd=REPO,
                text=True,
            )
            self.assertEqual(path.read_text(), source)

        inspected = json.loads(output)
        self.assertEqual(inspected["server_host"], "0.0.0.0")
        self.assertEqual(inspected["server_port"], 31000)
        self.assertEqual(inspected["server_url"], "http://127.0.0.1:31000")
        self.assertEqual(inspected["expected_gpu_count"], 8)
        self.assertEqual(
            inspected["target_model"], "/workspace/models/custom-target"
        )
        self.assertEqual(
            inspected["draft_model"], "/workspace/models/custom-draft"
        )

    def test_tp_width_is_not_artificially_capped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(
                directory,
                lambda config: config["model"].__setitem__(
                    "tensor_parallel_size", 4
                ),
            )
            model = active_model(load_config(path))
        self.assertEqual(model.tensor_parallel_size, 4)

    def test_dp_width_is_not_artificially_capped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(
                directory,
                lambda config: config["model"].__setitem__(
                    "data_parallel_size", 7
                ),
            )
            model = active_model(load_config(path))
        self.assertEqual(model.data_parallel_size, 7)

    def test_data_parallel_size_auto_derives_from_gpu_count(self):
        def configure(config):
            config["model"]["tensor_parallel_size"] = 2
            config["model"]["data_parallel_size"] = "auto"

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, configure)
            config = load_config(path)  # 'auto' passes validation unresolved
            with mock.patch.dict(os.environ, {"PP_GPU_COUNT": "4"}):
                model = active_model(config)
        self.assertEqual(model.data_parallel_size, 2)  # 4 GPUs / tp 2

    def test_data_parallel_size_auto_requires_divisible_gpu_count(self):
        def configure(config):
            config["model"]["tensor_parallel_size"] = 4
            config["model"]["data_parallel_size"] = "auto"

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, configure)
            config = load_config(path)
            with mock.patch.dict(os.environ, {"PP_GPU_COUNT": "6"}):
                with self.assertRaisesRegex(ValueError, "not divisible"):
                    active_model(config)

    def test_dynamic_profile_config_validates_and_resolves(self):
        # config-dynamic.yaml is the shipped auto-dp + fp8-KV profile; it must
        # validate against the same strict schema and resolve across node sizes.
        path = REPO / "config-dynamic.yaml"
        config = load_config(path)
        self.assertEqual(config["model"]["kv_cache_dtype"], "fp8_e4m3")
        self.assertEqual(config["model"]["data_parallel_size"], "auto")
        for gpus, expected_dp in (("2", 1), ("4", 2), ("8", 4)):
            with self.subTest(gpus=gpus):
                with mock.patch.dict(os.environ, {"PP_GPU_COUNT": gpus}):
                    model = active_model(config)
                self.assertEqual(model.tensor_parallel_size, 2)
                self.assertEqual(model.data_parallel_size, expected_dp)
                self.assertEqual(model.kv_cache_dtype, "fp8_e4m3")

    def test_search_completion_budget_is_not_coupled_to_server_context(self):
        def configure(config):
            config["server"]["context_length"] = 1048576
            config["search"].update(
                proofs_per_round=32,
                verifications_per_proof=8,
                top_proofs=8,
                refine_parents=4,
                reviews_per_refine_parent=3,
                max_completion_tokens=32768,
                solution_continuation_tokens=8192,
                verifier_continuation_tokens=4096,
                min_valid_verifications=5,
            )

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, configure)
            config = load_config(path)
        self.assertEqual(config["server"]["context_length"], 1048576)
        self.assertEqual(config["search"]["max_completion_tokens"], 32768)
        self.assertEqual(config["search"]["solution_continuation_tokens"], 8192)
        self.assertEqual(config["search"]["verifier_continuation_tokens"], 4096)
        self.assertEqual(config["search"]["min_valid_verifications"], 5)

    def test_search_round_limit_is_configurable(self):
        for rounds in (16, 32):
            with (
                self.subTest(rounds=rounds),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = self.write_config(
                    directory,
                    lambda config: config["search"].__setitem__(
                        "max_rounds", rounds
                    ),
                )
                config = load_config(path)
                self.assertEqual(config["search"]["max_rounds"], rounds)

    def test_search_temperature_is_configurable(self):
        for temperature in (0, 0.6, 1.5):
            with (
                self.subTest(temperature=temperature),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = self.write_config(
                    directory,
                    lambda config: config["search"].__setitem__(
                        "temperature", temperature
                    ),
                )
                config = load_config(path)
                self.assertEqual(config["search"]["temperature"], temperature)

    def test_search_temperature_rejects_invalid_values(self):
        invalid_values = (-0.1, float("nan"), "invalid")
        for temperature in invalid_values:
            with (
                self.subTest(temperature=temperature),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = self.write_config(
                    directory,
                    lambda config: config["search"].__setitem__(
                        "temperature", temperature
                    ),
                    name="invalid.yaml",
                )
                with self.assertRaisesRegex(
                    ValueError, "search.temperature must be"
                ):
                    load_config(path)

    def test_search_top_p_is_configurable(self):
        for top_p in (0.5, 0.95, 1):
            with (
                self.subTest(top_p=top_p),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = self.write_config(
                    directory,
                    lambda config: config["search"].__setitem__(
                        "top_p", top_p
                    ),
                )
                config = load_config(path)
                self.assertEqual(config["search"]["top_p"], top_p)

    def test_search_top_p_rejects_invalid_values(self):
        invalid_values = (0, -0.1, 1.1, float("nan"), "invalid")
        for top_p in invalid_values:
            with (
                self.subTest(top_p=top_p),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = self.write_config(
                    directory,
                    lambda config: config["search"].__setitem__(
                        "top_p", top_p
                    ),
                    name="invalid.yaml",
                )
                with self.assertRaisesRegex(ValueError, "search.top_p must be"):
                    load_config(path)

    def test_search_shape_validation_rejects_inconsistent_profiles(self):
        invalid_values = (
            ("top_proofs", 33),                 # > proofs_per_round
            ("refine_parents", 9),              # > top_proofs
            ("reviews_per_refine_parent", 17),  # > verifications_per_proof
            ("min_valid_verifications", 17),    # > verifications_per_proof
        )
        for key, value in invalid_values:
            with (
                self.subTest(key=key, value=value),
                tempfile.TemporaryDirectory() as directory,
            ):
                def configure(config, key=key, value=value):
                    config["search"].update(
                        proofs_per_round=32,
                        verifications_per_proof=16,
                        top_proofs=8,
                        refine_parents=4,
                        reviews_per_refine_parent=3,
                        min_valid_verifications=4,
                    )
                    config["search"][key] = value

                path = self.write_config(
                    directory, configure, name="invalid.yaml"
                )
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_decode_graphs_cover_configured_ceiling(self):
        batches = decode_graph_batches(96)
        self.assertEqual(batches[:16], list(range(1, 17)))
        self.assertEqual(batches[-1], 96)

    def test_submission_wrapper_requires_explicit_config(self):
        launcher = (REPO / "run_submission.sh").read_text()
        self.assertIn("CONFIG is required", launcher)
        self.assertIn("run_submission.py", launcher)
        self.assertNotIn("MODEL_MODE", launcher)
        self.assertNotIn("DFLASH=", launcher)

    def test_launcher_selects_fa3_or_fa4_strictly_from_yaml(self):
        fa3 = {
            "attention_backend": "fa3",
            "page_size": 1,
            "deterministic_inference": True,
        }
        self.assertEqual(
            attention_arguments(fa3),
            [
                "--attention-backend", "fa3", "--page-size", "1",
                "--enable-deterministic-inference",
            ],
        )

        def configure(config):
            config["server"].update(
                attention_backend="fa4",
                page_size=128,
                deterministic_inference=False,
            )

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, configure, name="fa4.yaml")
            fa4 = load_config(path)["server"]
        self.assertEqual(
            attention_arguments(fa4),
            ["--attention-backend", "fa4", "--page-size", "128"],
        )

        launcher = (HARNESS / "launch_server.py").read_text()
        self.assertIn("str(server[\"attention_backend\"])", launcher)
        # triton is now supported (Blackwell sm120): the launcher adds its kv-splits.
        self.assertIn("--triton-attention-num-kv-splits", launcher)
        # DFlash's ring worker still restricts the DRAFT backend to fa3/fa4, so
        # DFlash is unavailable with a triton draft (Blackwell) -- documented.
        worker = (REPO / "sglang_patches/dflash_worker_v2_ring.py").read_text()
        self.assertIn("draft_backend not in {\"fa3\", \"fa4\"}", worker)

    def test_attention_backend_validation_rejects_invalid_profiles(self):
        invalid_values = (
            ("attention_backend", "flashinfer"),  # not one of fa3/fa4/triton
            ("page_size", 128),                    # fa3 requires page_size=1
        )
        for key, value in invalid_values:
            with (
                self.subTest(key=key, value=value),
                tempfile.TemporaryDirectory() as directory,
            ):
                def configure(config, key=key, value=value):
                    config["server"].update(
                        attention_backend="fa3",
                        page_size=1,
                        deterministic_inference=True,
                    )
                    config["server"][key] = value

                path = self.write_config(
                    directory, configure, name="invalid.yaml"
                )
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_fa3_allows_either_determinism(self):
        # fa3 no longer forces deterministic_inference: Yi-Chia's DFlash rollouts run
        # fa3 nondeterministic, so both true and false are valid (page_size stays 1).
        for det in (True, False):
            def configure(config, det=det):
                config["server"].update(
                    attention_backend="fa3", page_size=1, deterministic_inference=det
                )
            with tempfile.TemporaryDirectory() as directory:
                path = self.write_config(directory, configure, name="fa3.yaml")
                server = load_config(path)["server"]
            self.assertEqual(server["deterministic_inference"], det)

    def test_triton_backend_profile_is_accepted(self):
        # Blackwell: triton is valid with page_size=1; deterministic optional.
        def configure(config):
            config["server"].update(attention_backend="triton", page_size=1)

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, configure, name="triton.yaml")
            server = load_config(path)["server"]
        self.assertEqual(server["attention_backend"], "triton")
        # triton with a non-1 page_size is rejected
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(
                directory,
                lambda c: c["server"].update(
                    attention_backend="triton", page_size=128
                ),
                name="triton_bad.yaml",
            )
            with self.assertRaises(ValueError):
                load_config(path)

    def test_flashinfer_arch_mapping(self):
        from launch_server import flashinfer_cuda_arch
        from unittest import mock
        import subprocess as sp

        def fake(cap):
            m = mock.Mock()
            m.stdout = cap + "\n"
            return m

        with mock.patch.object(sp, "run", return_value=fake("9.0")):
            self.assertEqual(flashinfer_cuda_arch(), "9.0a")   # Hopper
        with mock.patch.object(sp, "run", return_value=fake("12.0")):
            self.assertEqual(flashinfer_cuda_arch(), "12.0f")  # Blackwell sm120
        with mock.patch.object(sp, "run", side_effect=OSError):
            self.assertEqual(flashinfer_cuda_arch(), "9.0a")   # fallback


if __name__ == "__main__":
    unittest.main()
