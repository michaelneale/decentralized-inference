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
            gguf = tmp / "model.gguf"
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
                'if [[ "$1" == "write-package" ]]; then\n'
                '  out=""; prev=""\n'
                '  for a in "$@"; do\n'
                '    if [[ "$prev" == "--out-dir" ]]; then out="$a"; fi\n'
                "    prev=\"$a\"\n"
                "  done\n"
                '  mkdir -p "$out/artifacts"\n'
                '  printf "{}" >"$out/model-package.json"\n'
                "  exit 0\n"
                "fi\n"
                'if [[ "$1" == "verify-package-v2" ]]; then exit 0; fi\n'
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
                                    "file": "model.gguf",
                                    "size_bytes": len(payload),
                                    "blob_sha256": blob,
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
            gguf = tmp / "model.gguf"
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
                                    "file": "model.gguf",
                                    "size_bytes": len(payload),
                                    "blob_sha256": "0" * 64,
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


if __name__ == "__main__":
    unittest.main()
