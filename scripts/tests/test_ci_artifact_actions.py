from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ROOT / ".github" / "actions"


class CiArtifactActionTests(unittest.TestCase):
    def read_action(self, name: str) -> str:
        return (ACTIONS / name / "action.yml").read_text(encoding="utf-8")

    def test_external_actions_have_sha_and_release_provenance(self) -> None:
        action_files = sorted(ACTIONS.glob("*/action.yml"))
        workflow_files = sorted(
            (ROOT / ".github" / "workflows").glob("*.yml"),
        )
        exact_pin = re.compile(
            r"^[^@\s]+@[0-9a-f]{40}\s+#\s+\S",
        )
        protected_pr_lanes = {
            f"Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main": f"pr_{lane}.yml"
            for lane in ("quality", "website", "linux", "macos", "windows")
        }
        protected_pre_checkout_action = (
            "Mesh-LLM/mesh-llm/.github/actions/"
            "audit-depot-pr-isolation@ed07043b84d720aab30e75ed2f038f7042576f16"
        )

        for path in (*action_files, *workflow_files):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if "uses:" not in line:
                    continue
                value = line.split("uses:", maxsplit=1)[1].strip()
                if value.startswith("./"):
                    continue
                if value in protected_pr_lanes:
                    self.assertEqual(protected_pr_lanes[value], path.name)
                    continue
                if value == protected_pre_checkout_action:
                    self.assertIn(path.name, {
                        "ci-linux-host-slice.yml",
                        "ci-linux-product-slice.yml",
                        "ci-linux-runtime-slice.yml",
                        "ci-macos-host-slice.yml",
                        "ci-macos-product-slice.yml",
                        "ci-macos-runtime-slice.yml",
                        "ci-platform-checks-slice.yml",
                        "ci-quality-slice.yml",
                        "ci-rust-tests-slice.yml",
                        "ci-ui-artifact-slice.yml",
                        "ci-web-slice.yml",
                        "ci-windows-host-slice.yml",
                        "ci-windows-product-slice.yml",
                        "ci-windows-runtime-slice.yml",
                        "native-sdk-artifact.yml",
                        "static-abi-artifact.yml",
                        "swift-sdk-artifact.yml",
                    })
                    continue
                with self.subTest(
                    path=path.relative_to(ROOT),
                    line=line_number,
                ):
                    self.assertRegex(value, exact_pin)

    def test_workflow_status_gates_do_not_resist_cancellation(self) -> None:
        workflow_files = sorted(
            (ROOT / ".github" / "workflows").glob("*.yml"),
        )

        for path in workflow_files:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn(
                    "always()",
                    path.read_text(encoding="utf-8"),
                )

    def test_quality_slice_requires_ci_contract_validation(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "ci-quality-slice.yml"
        ).read_text(encoding="utf-8")
        contract_start = workflow.index("  quality_contracts:")
        contract_end = workflow.index("\n  rust_fmt:", contract_start)
        contract = workflow[contract_start:contract_end]

        self.assertIn(
            "uses: ./.github/actions/install-actionlint",
            contract,
        )
        self.assertNotIn("tool: actionlint@", contract)
        self.assertIn(
            "actionlint -config-file .github/actionlint.yaml",
            contract,
        )
        self.assertIn(
            "python3 -m pip install --disable-pip-version-check --no-input "
            "-r ci/requirements-ci-python.txt",
            contract,
        )
        self.assertIn(
            "python3 -m unittest discover -s scripts/tests -p 'test_*.py'",
            contract,
        )
        requirements = (
            ROOT / "ci" / "requirements-ci-python.txt"
        ).read_text(encoding="utf-8")
        self.assertRegex(requirements, r"(?m)^PyYAML>=6\.0$")
        self.assertIn(
            "cargo run -p xtask -- repo-consistency release-targets",
            contract,
        )
        self.assertIn("cargo tree -p mesh-llm-client", contract)
        self.assertIn("quality_contracts", workflow)


if __name__ == "__main__":
    unittest.main()
