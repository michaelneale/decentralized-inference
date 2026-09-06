"""Contract tests for scripts/skippy-canary-live-matrix.sh."""
from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "skippy-canary-live-matrix.sh"
MANIFEST = ROOT / "docs/skippy/llama-parity-candidates.json"


class LiveMatrixScriptTests(unittest.TestCase):
    def test_script_is_executable_and_parses(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dry_run_lists_every_runnable_model_pin_row(self):
        rows = [
            row
            for row in json.loads(MANIFEST.read_text())["candidates"]
            if row.get("model_pin")
            and row.get("status") in ("certified", "candidate", "candidate_stateful")
        ]
        self.assertGreaterEqual(len(rows), 5, "minimum live matrix rows missing")
        result = subprocess.run(
            [str(SCRIPT), "--dry-run"],
            capture_output=True,
            text=True,
            env={**os.environ, "MESH_LLM_BIN": "/nonexistent-mesh-llm"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for row in rows:
            self.assertIn(row["llama_model"], result.stdout)
        self.assertIn("dry run: no rows executed", result.stdout)

    def test_script_uses_immutable_pin_and_two_node_smoke(self):
        text = SCRIPT.read_text()
        # Pinned revision downloads, integrity checks, package-v2 write and
        # independent verification, and the two-node split smoke.
        self.assertIn("--revision", text)
        self.assertIn("shasum -a 256", text)
        self.assertIn("write-package", text)
        self.assertIn("verify-package-v2", text)
        self.assertIn("ci-two-node-split-smoke.sh", text)
        # Fail closed: any failed row fails the matrix.
        self.assertIn("exit 1", text)
        # Recurrent rows expect the recurrent payload kind.
        self.assertIn("kv-recurrent", text)
        self.assertIn("kv-dense", text)

    def test_one_mocked_row_passes_beyond_dry_run(self):
        """Execute one row through the field-extraction/download seam.

        Mocks hf download (serves a fixture GGUF), the package tool, and the
        two-node smoke; runs the real script with a one-row manifest so the
        per-row field extraction, pin verification, and pipeline actually
        execute (a dry run cannot reach these).
        """
        import tempfile

        payload = b"fake-gguf-bytes-for-matrix-row"
        import hashlib

        blob = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            gguf = tmp / "model-Q4_K_M.gguf"
            gguf.write_bytes(payload)
            pkg_dir = tmp / "pkg"
            pkg_dir.mkdir()

            hf_mock = tmp / "hf-mock"
            hf_mock.write_text(
                "#!/usr/bin/env bash\n"
                'if [[ "$1" == "download" && "$4" == "--revision" ]]; then\n'
                '  printf "%s\\n" "$5" >/dev/null # arg shape sanity\n'
                '  printf "%s\\n" "' + str(gguf) + '"\n'
                "  exit 0\n"
                "fi\n"
                "exit 2\n"
            )
            hf_mock.chmod(0o755)

            pkg_tool = tmp / "pkg-tool-mock"
            pkg_tool.write_text(
                "#!/usr/bin/env bash\n"
                'args=("$@")\n'
                'if [[ "${args[0]}" == "write-package" ]]; then\n'
                "  # Positional input must be the local GGUF path and the\n"
                "  # coordinate must arrive via --model-id (resolve_package_input\n"
                "  # rejects identity flags on a non-local positional input).\n"
                '  [[ "${args[1]}" == "' + str(gguf) + '" ]] || { echo "expected local path positional, got ${args[1]}" >&2; exit 3; }\n'
                "  out=\"\"; model_id=\"\"; prev=\"\"\n"
                '  for a in "$@"; do\n'
                '    case "$prev" in\n'
                '      --out-dir) out="$a" ;;\n'
                '      --model-id) model_id="$a" ;;\n'
                "    esac\n"
                "    prev=\"$a\"\n"
                "  done\n"
                '  [[ "$model_id" == "org/mock-gguf:Q4_K_M" ]] || { echo "bad --model-id: $model_id" >&2; exit 3; }\n'
                '  mkdir -p "$out/artifacts"\n'
                '  printf "{}" >"$out/model-package.json"\n'
                "  exit 0\n"
                "fi\n"
                'if [[ "${args[0]}" == "verify-package-v2" ]]; then exit 0; fi\n'
                "exit 2\n"
            )
            pkg_tool.chmod(0o755)

            smoke_mock = tmp / "smoke-mock.sh"
            smoke_mock.write_text(
                "#!/usr/bin/env bash\n"
                'printf "smoke ran with model %s payload-kind %s\\n" "$MESH_TWO_NODE_SPLIT_MODEL" "$MESH_TWO_NODE_SPLIT_EXPECTED_EXACT_PAYLOAD_KIND"\n'
            )
            smoke_mock.chmod(0o755)

            manifest = tmp / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "llama_model": "mockfamily",
                                "status": "certified",
                                "recurrent": "all",
                                "model_pin": {
                                    "repo": "org/mock-gguf",
                                    "revision": "1" * 40,
                                    "file": "model-Q4_K_M.gguf",
                                    "size_bytes": len(payload),
                                    "blob_sha256": blob,
                                    "selector": "Q4_K_M",
                                },
                            }
                        ]
                    }
                )
            )

            result = subprocess.run(
                [str(SCRIPT)],
                capture_output=True,
                text=True,
                cwd=tmp,
                env={
                    **os.environ,
                    "SKIPPY_PARITY_MANIFEST": str(manifest),
                    "SKIPPY_CANARY_LIVE_MATRIX_PKG_TOOL": str(pkg_tool),
                    "SKIPPY_CANARY_LIVE_MATRIX_SPLIT_SMOKE": str(smoke_mock),
                    "SKIPPY_CANARY_LIVE_MATRIX_HF_DOWNLOAD": str(hf_mock),
                    "SKIPPY_CANARY_LIVE_MATRIX_ROOT": str(tmp / "evidence"),
                    "SKIPPY_CANARY_LIVE_MATRIX_WORK_ROOT": str(tmp / "work"),
                    "STAGE_SERVER_BIN": "/usr/bin/true",
                    "MESH_LLM_BIN": "/usr/bin/true",
                },
            )
            self.assertEqual(
                result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
            self.assertIn("row mockfamily: PASS", result.stdout)
            self.assertIn("live matrix passed: 1/1 rows", result.stdout)
            # The mocked smoke received the package dir and payload kind.
            smoke_log = (tmp / "evidence" / "live-matrix" / "mockfamily" / "two-node-split.log")
            self.assertIn("payload-kind kv-recurrent", smoke_log.read_text())

    def test_one_mocked_row_sha_mismatch_fails_closed(self):
        """A pinned sha mismatch must fail the matrix beyond dry-run."""
        import tempfile

        payload = b"tampered-bytes"
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            gguf = tmp / "model-Q4_K_M.gguf"
            gguf.write_bytes(payload)
            hf_mock = tmp / "hf-mock"
            hf_mock.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "' + str(gguf) + '"\n'
            )
            hf_mock.chmod(0o755)
            manifest = tmp / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "llama_model": "mockfamily",
                                "status": "certified",
                                "model_pin": {
                                    "repo": "org/mock-gguf",
                                    "revision": "1" * 40,
                                    "file": "model-Q4_K_M.gguf",
                                    "size_bytes": len(payload),
                                    "blob_sha256": "0" * 64,
                                    "selector": "Q4_K_M",
                                },
                            }
                        ]
                    }
                )
            )
            result = subprocess.run(
                [str(SCRIPT)],
                capture_output=True,
                text=True,
                cwd=tmp,
                env={
                    **os.environ,
                    "SKIPPY_PARITY_MANIFEST": str(manifest),
                    # Hermetic: never invoke a real cargo build in the test
                    # tree — the sha mismatch must be reached and recorded
                    # without any producer build.
                    "SKIPPY_CANARY_LIVE_MATRIX_PKG_TOOL": str(hf_mock),
                    "SKIPPY_CANARY_LIVE_MATRIX_HF_DOWNLOAD": str(hf_mock),
                    "SKIPPY_CANARY_LIVE_MATRIX_ROOT": str(tmp / "evidence"),
                    "SKIPPY_CANARY_LIVE_MATRIX_WORK_ROOT": str(tmp / "work"),
                    "STAGE_SERVER_BIN": "/usr/bin/true",
                    "MESH_LLM_BIN": "/usr/bin/true",
                },
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("sha256 mismatch", result.stdout)
            self.assertIn("mockfamily:sha256", result.stdout)

    def test_one_mocked_row_accepts_all_hf_output_forms_and_symlink(self):
        """The hf CLI reports the final artifact path in several shapes, and
        the HF cache stores snapshot entries as symlinks. The download gate
        must accept `path=/...`, `path: /...`, and a bare path, resolve the
        symlink to the real blob, and still pass the size and SHA gates
        (stat on a symlink reports the link length, not the blob size)."""
        import hashlib
        import tempfile

        payload = b"symlinked-gguf-bytes-for-matrix-row"
        blob = hashlib.sha256(payload).hexdigest()
        forms = ["path={}", "path: {}", "{}"]
        for form in forms:
            with self.subTest(form=form.split("{}")[0].strip() or "bare"):
                with tempfile.TemporaryDirectory() as tmp_name:
                    tmp = Path(tmp_name)
                    # HF-cache shape: blobs/<sha> real file, snapshots/<rev>/
                    # symlink pointing at it.
                    real = tmp / "blobs" / blob
                    real.parent.mkdir()
                    real.write_bytes(payload)
                    snapshot_dir = tmp / "snapshots" / ("1" * 40)
                    snapshot_dir.mkdir(parents=True)
                    snapshot = snapshot_dir / "model-Q4_K_M.gguf"
                    snapshot.symlink_to(real)

                    hf_mock = tmp / "hf-mock"
                    hf_mock.write_text(
                        "#!/usr/bin/env bash\n"
                        'printf "%s\\n" "' + form.format(snapshot) + '"\n'
                    )
                    hf_mock.chmod(0o755)

                    pkg_tool = tmp / "pkg-tool-mock"
                    pkg_tool.write_text("#!/usr/bin/env bash\nexit 0\n")
                    pkg_tool.chmod(0o755)
                    smoke_mock = tmp / "smoke-mock.sh"
                    smoke_mock.write_text("#!/usr/bin/env bash\nexit 0\n")
                    smoke_mock.chmod(0o755)

                    manifest = tmp / "manifest.json"
                    manifest.write_text(
                        json.dumps(
                            {
                                "candidates": [
                                    {
                                        "llama_model": "mockfamily",
                                        "status": "certified",
                                        "model_pin": {
                                            "repo": "org/mock-gguf",
                                            "revision": "1" * 40,
                                            "file": "model-Q4_K_M.gguf",
                                            "size_bytes": len(payload),
                                            "blob_sha256": blob,
                                            "selector": "Q4_K_M",
                                        },
                                    }
                                ]
                            }
                        )
                    )
                    result = subprocess.run(
                        [str(SCRIPT)],
                        capture_output=True,
                        text=True,
                        cwd=tmp,
                        env={
                            **os.environ,
                            "SKIPPY_PARITY_MANIFEST": str(manifest),
                            "SKIPPY_CANARY_LIVE_MATRIX_PKG_TOOL": str(pkg_tool),
                            "SKIPPY_CANARY_LIVE_MATRIX_SPLIT_SMOKE": str(smoke_mock),
                            "SKIPPY_CANARY_LIVE_MATRIX_HF_DOWNLOAD": str(hf_mock),
                            "SKIPPY_CANARY_LIVE_MATRIX_ROOT": str(tmp / "evidence"),
                            "SKIPPY_CANARY_LIVE_MATRIX_WORK_ROOT": str(tmp / "work"),
                            "STAGE_SERVER_BIN": "/usr/bin/true",
                            "MESH_LLM_BIN": "/usr/bin/true",
                        },
                    )
                    self.assertEqual(
                        result.returncode, 0,
                        f"form={form!r} stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                    )
                    self.assertIn("verified pinned source", result.stdout)

    def test_prepare_builds_exact_producers_and_records_provenance(self):
        """--prepare must build this run's host binary and patched native
        runtime and write producer-provenance.json; the provenance writer
        must fail closed without an adjacent runtime manifest."""
        import os
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            result = subprocess.run(
                [str(SCRIPT), "--prepare"],
                capture_output=True,
                text=True,
                cwd=tmp,
                env={
                    **os.environ,
                    "SKIPPY_CANARY_LIVE_MATRIX_ROOT": str(tmp / "evidence"),
                    "SKIPPY_CANARY_LIVE_MATRIX_WORK_ROOT": str(tmp / "work"),
                    # Point at a binary with NO adjacent native-runtimes
                    # bundle: provenance must fail closed before the build
                    # and packaging producers run (their failure would be a
                    # different defect).
                    "MESH_LLM_BIN": "/usr/bin/true",
                    # Skip the real producers; this test pins the
                    # provenance fail-closed path, not a full build.
                    "SKIPPY_CANARY_LIVE_MATRIX_PREPARE_SKIP_BUILD": "1",
                },
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "no manifest for meshllm-native-runtime-darwin-aarch64-metal",
                result.stdout + result.stderr,
            )

            # Decoy: a WRONG/stale runtime manifest under the bundle root
            # must not be attested — provenance reads only the exact path
            # meshllm-native-runtime-<os>-<arch>-<backend>/manifest.json.
            decoy_root = tmp / "work" / "native-runtimes"
            decoy_root.mkdir(parents=True, exist_ok=True)
            decoy_dir = decoy_root / "meshllm-native-runtime-darwin-aarch64-vulkan"
            decoy_dir.mkdir()
            (decoy_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "runtime": {"id": "decoy", "skippy_abi": "0"},
                        "build": {
                            "backend": "vulkan",
                            "platform": "darwin-aarch64",
                            "llama_patch_digest": "decoy",
                            "llama_upstream_sha": "0" * 40,
                            "llama_patched_sha": "0" * 40,
                        },
                    }
                )
            )
            result = subprocess.run(
                [str(SCRIPT), "--prepare"],
                capture_output=True,
                text=True,
                cwd=tmp,
                env={
                    **os.environ,
                    "SKIPPY_CANARY_LIVE_MATRIX_ROOT": str(tmp / "evidence"),
                    "SKIPPY_CANARY_LIVE_MATRIX_WORK_ROOT": str(tmp / "work"),
                    "MESH_LLM_BIN": "/usr/bin/true",
                    "SKIPPY_CANARY_LIVE_MATRIX_PREPARE_SKIP_BUILD": "1",
                    "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR": str(decoy_root),
                },
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "no manifest for meshllm-native-runtime-darwin-aarch64-metal",
                result.stdout + result.stderr,
            )
            self.assertNotIn("decoy", (tmp / "evidence" / "live-matrix" / "producer-provenance.json").read_text() if (tmp / "evidence" / "live-matrix" / "producer-provenance.json").exists() else "")

        text = SCRIPT.read_text()
        self.assertIn("cargo build -p mesh-llm", text)
        self.assertIn("package-native-runtime.sh", text)
        self.assertIn("--build --backend", text)
        self.assertIn("producer-provenance.json", text)
        self.assertIn("llama_upstream_sha", text)
        self.assertIn("llama_patched_sha", text)
        self.assertIn("skippy_abi", text)


if __name__ == "__main__":
    unittest.main()
