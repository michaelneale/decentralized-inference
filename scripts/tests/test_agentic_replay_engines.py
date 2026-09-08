from __future__ import annotations

import hashlib
import json
import subprocess
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
    def arm(
        self, engine: str, executable: str = "/engine", cwd: Path | None = None
    ) -> ENGINES.EngineArm:
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
            cwd=cwd or REPO,
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

    def test_config_digest_and_arms_come_from_one_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "engines.json"
            document = {
                "schema_version": 1,
                "comparison": {"model": "hf://org/model"},
                "arms": [
                    {
                        "label": "vllm",
                        "engine": "vllm",
                        "executable": "vllm",
                        "model": "org/model",
                        "context_size": 4096,
                        "max_concurrency": 2,
                    }
                ],
            }
            config_path.write_text(json.dumps(document), encoding="utf-8")

            def one_snapshot_read(self: Path) -> bytes:
                # Simulate the file changing between reads: every read returns
                # the same mutated snapshot, so a digest from a second read
                # would not match the digest of the parsed document.
                document["comparison"]["model"] = "hf://org/other-model"
                return json.dumps(document).encode("utf-8")

            with mock.patch.object(
                Path, "read_bytes", one_snapshot_read
            ), mock.patch.object(
                Path, "read_text", autospec=True, side_effect=AssertionError(
                    "load_engine_config must read the file once via read_bytes"
                ),
            ):
                config = ENGINES.load_engine_config(config_path)

            self.assertEqual(
                config.sha256,
                hashlib.sha256(json.dumps(document).encode("utf-8")).hexdigest(),
            )

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
                ENGINES.engine_version(self.arm("vllm"), "/engine")

    def test_version_command_timeout_is_a_preflight_error(self) -> None:
        with mock.patch.object(
            ENGINES.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="vllm --version", timeout=30),
        ):
            with self.assertRaisesRegex(RuntimeError, "version command timed out"):
                ENGINES.engine_version(self.arm("vllm"), "/engine")

    def test_engine_version_passes_timeout_and_resolved_executable(self) -> None:
        arm = self.arm("vllm", executable="/opt/vllm/bin/vllm")
        completed = mock.Mock(returncode=0, stdout="vLLM 0.10.1", stderr="")

        with mock.patch.object(
            ENGINES.subprocess, "run", return_value=completed
        ) as run:
            version = ENGINES.engine_version(arm, "/resolved/vllm")

        self.assertEqual(version, "vLLM 0.10.1")
        run.assert_called_once()
        self.assertEqual(
            run.call_args.kwargs.get("timeout"),
            ENGINES.VERSION_COMMAND_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            run.call_args.args[0], ["/resolved/vllm", "--version"]
        )

    def test_relative_executable_resolves_against_arm_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine_dir = root / "engine"
            engine_dir.mkdir()
            engine_bin = engine_dir / "server"
            engine_bin.write_text("#!/bin/sh\necho engine-version\n")
            engine_bin.chmod(0o755)
            runner_bin = root / "server"
            runner_bin.write_text("#!/bin/sh\necho runner-version\n")
            runner_bin.chmod(0o755)

            arm = self.arm(
                "llama.cpp", executable="./server", cwd=engine_dir
            )

            with mock.patch.dict(
                "os.environ", {"PATH": str(root)}, clear=False
            ):
                resolved = ENGINES.resolve_engine_executable(arm)

            self.assertEqual(resolved, str(engine_bin.resolve()))

    def test_bare_executable_name_resolves_on_runner_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine_dir = root / "engine"
            engine_dir.mkdir()
            decoy = engine_dir / "llama-server"
            decoy.write_text("#!/bin/sh\n")
            decoy.chmod(0o755)
            path_bin = root / "llama-server"
            path_bin.write_text("#!/bin/sh\n")
            path_bin.chmod(0o755)

            arm = self.arm("llama.cpp", executable="llama-server", cwd=engine_dir)
            with mock.patch.dict(
                "os.environ", {"PATH": str(root)}, clear=False
            ):
                resolved = ENGINES.resolve_engine_executable(arm)

            self.assertEqual(resolved, str(path_bin))

    def test_missing_relative_executable_fails_preflight_with_cwd_context(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine_dir = root / "engine"
            engine_dir.mkdir()

            arm = self.arm("llama.cpp", executable="./server", cwd=engine_dir)

            with self.assertRaisesRegex(FileNotFoundError, "engine"):
                ENGINES.verify_engine_arm(arm)

    def test_preflight_records_the_resolved_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine_dir = root / "engine"
            engine_dir.mkdir()
            engine_bin = engine_dir / "server"
            engine_bin.write_text("#!/bin/sh\n")
            engine_bin.chmod(0o755)

            arm = self.arm("llama.cpp", executable="./server", cwd=engine_dir)

            with mock.patch.object(
                ENGINES,
                "engine_version",
                return_value="llama.cpp b1234",
            ) as version:
                build = ENGINES.verify_engine_arm(arm)

            self.assertEqual(
                build["provenance"]["resolved_executable"],
                str(engine_bin.resolve()),
            )
            self.assertEqual(
                version.call_args.args[1],
                str(engine_bin.resolve()),
            )

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

    def config(self, directory: str) -> ENGINES.EngineConfig:
        path = Path(directory) / "engines.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "comparison": {"model": "org/model"},
                    "arms": [
                        {
                            "label": "llama",
                            "engine": "llama.cpp",
                            "executable": "llama-server",
                            "model": "org/model.gguf",
                            "context_size": 4096,
                            "max_concurrency": 2,
                        },
                        {
                            "label": "vllm",
                            "engine": "vllm",
                            "executable": "vllm",
                            "model": "org/model",
                            "context_size": 4096,
                            "max_concurrency": 2,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return ENGINES.load_engine_config(path)

    def test_engine_order_specs_use_verified_identity_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)

            pending = ENGINES.engine_order_specs(config)
            verified = ENGINES.engine_order_specs(
                config, {"llama": "a" * 64, "vllm": "b" * 64}
            )

        self.assertEqual(
            [spec["commit"] for spec in pending],
            [ENGINES.PENDING_VERSION_SHA256, ENGINES.PENDING_VERSION_SHA256],
        )
        self.assertEqual(
            [spec["commit"] for spec in verified], ["a" * 64, "b" * 64]
        )


if __name__ == "__main__":
    unittest.main()
