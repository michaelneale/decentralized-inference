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


if __name__ == "__main__":
    unittest.main()
