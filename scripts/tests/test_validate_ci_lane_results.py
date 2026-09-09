from __future__ import annotations

from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
from unittest import mock
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "validate_ci_lane_results", ROOT / "scripts" / "validate-ci-lane-results.py"
)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def state(result: str) -> dict[str, str]:
    return {"result": result}


class ValidateCiLaneResultsTests(unittest.TestCase):
    def test_linux_lane_requires_each_selected_producer_and_consumer(self) -> None:
        plan = {
            "lane": "linux",
            "required": True,
            "required_slices": ["static-abi", "rust-tests", "runtime-product", "sdk"],
            "matrices": {
                "rust_tests": [{"id": "batch-0"}],
                "hosts": [{"id": "linux-amd64"}],
                "runtime_products": [{"id": "linux-cpu"}],
                "smoke": [],
                "sdk": [{"id": "kotlin"}],
            },
        }
        needs = {
            job: state("success")
            for job in {
                "ui_artifact",
                "static_abi",
                "rust_tests",
                "hosts",
                "native_runtimes",
                "runtime_product",
                "kotlin_sdk_input",
                "sdk",
            }
        }
        VALIDATOR.validate(plan, needs)
        needs["runtime_product"] = state("skipped")
        with self.assertRaisesRegex(VALIDATOR.LaneResultError, "runtime_product"):
            VALIDATOR.validate(plan, needs)

    def test_macos_lane_maps_swift_sdk_and_platform_jobs(self) -> None:
        plan = {
            "lane": "macos",
            "required": True,
            "required_slices": ["runtime-product", "platform-checks", "sdk"],
            "matrices": {
                "hosts": [{"id": "macos-arm64"}],
                "runtime_products": [{"id": "macos-metal"}],
                "platform_checks": [{"id": "macos-portable"}],
                "sdk": [{"id": "swift"}],
                "smoke": [],
            },
        }
        expected = {
            "validate_plan",
            "ui_artifact",
            "hosts",
            "native_runtimes",
            "runtime_product",
            "platform_checks",
            "swift_sdk_input",
            "sdk",
        }

        self.assertEqual(VALIDATOR._required_jobs(plan), expected)
        VALIDATOR.validate(plan, {job: state("success") for job in expected})

    def test_windows_runtime_products_require_selected_product_smoke(self) -> None:
        plan = {
            "lane": "windows",
            "required": True,
            "required_slices": ["runtime-product"],
            "matrices": {
                "hosts": [],
                "runtime_products": [{"id": "windows-cpu"}],
                "platform_checks": [],
                "smoke": [{"id": "product-integration-windows-cpu"}],
            },
        }
        expected = {"native_runtimes", "runtime_product", "product_smoke"}

        self.assertEqual(VALIDATOR._required_jobs(plan), expected)
        VALIDATOR.validate(plan, {job: state("success") for job in expected})

    def test_malformed_planned_need_uses_lane_error_exit(self) -> None:
        plan = {
            "lane": "quality",
            "required": True,
            "required_slices": ["quality"],
            "matrices": {},
        }
        with self.assertRaisesRegex(VALIDATOR.LaneResultError, "None"):
            VALIDATOR.validate(plan, {"quality": None})

        argv = [
            "validate-ci-lane-results.py",
            "--lane-plan",
            json.dumps(plan),
            "--needs",
            json.dumps({"quality": None}),
        ]
        with mock.patch("sys.argv", argv), redirect_stderr(io.StringIO()):
            self.assertEqual(VALIDATOR.main(), 2)

    def test_unplanned_skip_is_allowed_but_failure_is_not(self) -> None:
        plan = {
            "lane": "quality",
            "required": True,
            "required_slices": ["quality"],
            "matrices": {"clippy": []},
        }
        VALIDATOR.validate(
            plan,
            {"quality": state("success"), "runner_contract": state("skipped")},
        )
        with self.assertRaisesRegex(VALIDATOR.LaneResultError, "failure"):
            VALIDATOR.validate(plan, {"quality": state("failure")})
        with self.assertRaisesRegex(VALIDATOR.LaneResultError, "runner_contract"):
            VALIDATOR.validate(
                plan,
                {"quality": state("success"), "runner_contract": state("success")},
            )

    def test_inactive_lane_must_not_claim_required_without_jobs(self) -> None:
        plan = {
            "lane": "windows",
            "required": True,
            "required_slices": [],
            "matrices": {"hosts": [], "runtime_products": [], "platform_checks": []},
        }
        with self.assertRaisesRegex(VALIDATOR.LaneResultError, "no planned jobs"):
            VALIDATOR.validate(plan, {})


if __name__ == "__main__":
    unittest.main()
