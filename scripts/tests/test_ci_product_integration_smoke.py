from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PRODUCT_SCRIPT = ROOT / "scripts" / "ci-product-integration-smoke.sh"
REQUIRED_PHASES = [
    "dense-standalone",
    "dense-openai-sdk",
    "dense-constrained-tokio-restart",
    "dense-split-kv",
    "recurrent-split-kv",
]


class ProductIntegrationSmokeTests(unittest.TestCase):
    def write_stub(self, path: Path) -> None:
        path.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
phase="${MESH_PRODUCT_INTEGRATION_PHASE:?missing phase}"
for log_path in "${MESH_CI_LOG:-}" "${MESH_CI_HEADLESS_LOG:-}" "${MESH_COMPAT_LOG:-}"; do
    if [[ -n "$log_path" ]]; then
        mkdir -p "$(dirname "$log_path")"
        : >"$log_path"
    fi
done
if [[ "${STUB_FAIL_PHASE:-}" == "$phase" ]]; then
    exit 42
fi
""",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def run_suite(self, failure_phase: str | None = None) -> tuple[subprocess.CompletedProcess[str], dict]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy2(PRODUCT_SCRIPT, scripts / PRODUCT_SCRIPT.name)
            for name in (
                "ci-smoke-test.sh",
                "ci-compat-smoke.sh",
                "ci-two-node-split-smoke.sh",
            ):
                self.write_stub(scripts / name)

            binary = root / "mesh-llm"
            binary.touch(mode=0o755)
            artifact_dir = root / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "product-manifest.json").write_text(
                '{"backend":"cpu"}\n', encoding="utf-8"
            )
            dense_model = root / "dense.gguf"
            recurrent_model = root / "recurrent.gguf"
            dense_model.touch()
            recurrent_model.touch()
            phase_root = root / "phase-evidence"
            env = {
                **os.environ,
                "PATH": "/usr/bin:/bin",
                "MESH_PRODUCT_INTEGRATION_PHASE_ROOT": str(phase_root),
            }
            if failure_phase is not None:
                env["STUB_FAIL_PHASE"] = failure_phase

            result = subprocess.run(
                [
                    "bash",
                    "scripts/ci-product-integration-smoke.sh",
                    str(binary),
                    str(artifact_dir),
                    str(dense_model),
                    str(recurrent_model),
                    "linux",
                    "cpu",
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            manifest = json.loads((phase_root / "phase-results.json").read_text())
            return result, manifest

    def test_success_reconciles_the_exact_five_phases(self) -> None:
        result, manifest = self.run_suite()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(manifest["suite_status"], "passed")
        self.assertEqual(
            manifest["provenance"],
            {
                "platform": "linux",
                "backend": "cpu",
                "device": "CPU",
                "product_backend": "cpu",
            },
        )
        self.assertEqual(manifest["required_phases"], REQUIRED_PHASES)
        self.assertEqual(
            [phase["phase"] for phase in manifest["phases"]], REQUIRED_PHASES
        )
        self.assertEqual(manifest["reconciliation"], {
            "status": "passed",
            "finalized": True,
            "failure_phase": None,
            "missing_phases": [],
            "errors": [],
        })
        for phase in manifest["phases"]:
            with self.subTest(phase=phase["phase"]):
                self.assertEqual(phase["status"], "passed")
                self.assertEqual(phase["exit_code"], 0)
                self.assertTrue(phase["model"]["label"])
                self.assertTrue(phase["model"]["identity"])
                self.assertTrue(phase["workdir"])
                self.assertTrue(phase["log_paths"])
                self.assertLessEqual(
                    phase["started_at_unix_ns"], phase["ended_at_unix_ns"]
                )

    def test_failed_phase_is_recorded_and_reconciliation_fails_closed(self) -> None:
        result, manifest = self.run_suite("dense-openai-sdk")

        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual(manifest["suite_status"], "failed")
        self.assertEqual(manifest["reconciliation"]["status"], "failed")
        self.assertTrue(manifest["reconciliation"]["finalized"])
        self.assertEqual(manifest["reconciliation"]["failure_phase"], "dense-openai-sdk")
        self.assertEqual(
            [phase["phase"] for phase in manifest["phases"]],
            ["dense-standalone", "dense-openai-sdk"],
        )
        failed = manifest["phases"][-1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["exit_code"], 42)
        self.assertIn("missing required phase: dense-constrained-tokio-restart", manifest["reconciliation"]["errors"])
        self.assertIn("phase did not pass: dense-openai-sdk", manifest["reconciliation"]["errors"])


if __name__ == "__main__":
    unittest.main()
