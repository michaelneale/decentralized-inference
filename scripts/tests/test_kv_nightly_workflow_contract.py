from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
OWNERSHIP = ROOT / ".github" / "workflows" / "nightly-kv-coverage.yml"
STABILITY = ROOT / ".github" / "workflows" / "nightly-stability-run.yml"


class KvNightlyWorkflowContractTests(unittest.TestCase):
    def test_ownership_schedule_is_trusted_hosted_and_reproducible(self) -> None:
        workflow = OWNERSHIP.read_text(encoding="utf-8")
        self.assertIn('    - cron: "23 6 * * *"', workflow)
        self.assertIn("  workflow_dispatch:", workflow)
        self.assertNotIn("\n  pull_request:", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertIn("runs-on: ubuntu-24.04", workflow)
        self.assertIn("ref: main", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("verify-runner-image public cpu", workflow)
        self.assertIn("cargo test --locked -p skippy-cache", workflow)
        self.assertIn("SKIPPY_CACHE_STATE_MACHINE_SEEDS", workflow)
        self.assertIn("SKIPPY_CACHE_STATE_MACHINE_STEPS", workflow)
        self.assertNotIn("secrets.", workflow)

    def test_live_nightly_runs_both_harnesses_and_preserves_failures(self) -> None:
        workflow = STABILITY.read_text(encoding="utf-8")
        self.assertIn("scripts/qa-nightly-stability.py", workflow)
        self.assertIn("scripts/qa-kv-tool-loop-stability.py", workflow)
        self.assertIn("continue-on-error: true", workflow)
        self.assertIn("Fail on stability regression", workflow)
        self.assertIn("steps.stability_harness.outcome == 'failure'", workflow)
        self.assertIn("steps.kv_tool_loop.outcome == 'failure'", workflow)
        self.assertIn("runs-on: ubuntu-24.04", workflow)
        self.assertNotIn("runner_label", workflow)

if __name__ == "__main__":
    unittest.main()
