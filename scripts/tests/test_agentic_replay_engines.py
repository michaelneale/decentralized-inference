from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[2]
EVALS = REPO / "evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

import agentic_replay_engines as ENGINES  # noqa: E402


class AgenticReplayEnginesTest(unittest.TestCase):
    def arm(self, engine: str, executable: str = "/engine") -> ENGINES.EngineArm:
        return ENGINES.EngineArm(
            label=engine.replace(".", "-"),
            engine=engine,
            executable=executable,
            model="org/model.gguf" if engine == "llama.cpp" else "org/model",
            served_model="hf://org/model",
            context_size=131072,
            max_concurrency=8,
            tokenizer="org/tokenizer",
            hf_config="org/config.json",
            prefix_cache=True,
            batch_size=2048,
            ubatch_size=128,
            extra_args=("--extra", "value"),
            cwd=REPO,
        )

    def test_load_config_records_paths_without_file_hash_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "engines.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "comparison": {"model": "hf://org/model"},
                        "arms": [
                            {
                                "label": "llama",
                                "engine": "llama.cpp",
                                "executable": "llama-server",
                                "model": "org/model.gguf",
                                "context_size": 131072,
                                "max_concurrency": 8,
                            },
                            {
                                "label": "vllm",
                                "engine": "vllm",
                                "executable": "vllm",
                                "model": "org/model",
                                "context_size": 131072,
                                "max_concurrency": 8,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = ENGINES.load_engine_config(config_path)

        self.assertEqual(config.comparison.model, "hf://org/model")
        self.assertEqual([arm.engine for arm in config.arms], ["llama.cpp", "vllm"])
        self.assertEqual(config.arms[0].model, "org/model.gguf")
        self.assertEqual(len(config.sha256), 64)

    def test_engine_commands_preserve_capacity_and_cache_controls(self) -> None:
        llama = ENGINES.engine_server_command(self.arm("llama.cpp"))
        vllm = ENGINES.engine_server_command(self.arm("vllm"))
        sglang = ENGINES.engine_server_command(self.arm("sglang"))

        self.assertIn("--parallel", llama)
        self.assertIn("--kv-unified", llama)
        self.assertIn("--max-num-seqs", vllm)
        self.assertIn("--enable-prefix-caching", vllm)
        self.assertIn("sglang.launch_server", sglang)
        self.assertIn("--max-running-requests", sglang)
        for command in (llama, vllm, sglang):
            self.assertEqual(command[-2:], ["--extra", "value"])

    def test_external_identity_hashes_reported_version_only(self) -> None:
        arm = self.arm("vllm", executable=sys.executable)
        version = "vLLM 0.10.1+cuda"

        with mock.patch.object(ENGINES, "engine_version", return_value=version):
            build = ENGINES.verify_engine_arm(arm)

        expected = hashlib.sha256(version.encode()).hexdigest()
        self.assertEqual(build["commit"], expected)
        self.assertEqual(build["version_sha256"], expected)
        self.assertEqual(build["provenance"]["version"], version)
        self.assertNotIn("binary_sha256", build)
        self.assertNotIn("model_sha256", build["provenance"])
        self.assertNotIn("executable_sha256", build["provenance"])

    def test_version_command_failure_is_a_preflight_error(self) -> None:
        failed = mock.Mock(returncode=1, stdout="", stderr="unknown option")

        with mock.patch.object(ENGINES.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "version command failed"):
                ENGINES.engine_version(self.arm("vllm"))

    def test_duplicate_external_labels_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.json"
            arm = {
                "label": "same",
                "engine": "vllm",
                "executable": "vllm",
                "model": "org/model",
                "context_size": 4096,
                "max_concurrency": 2,
            }
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "comparison": {"model": "org/model"},
                        "arms": [arm, arm],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "labels must be unique"):
                ENGINES.load_engine_config(path)


if __name__ == "__main__":
    unittest.main()
