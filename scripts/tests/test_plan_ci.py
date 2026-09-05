from __future__ import annotations

import copy
import importlib.util
import json
from fnmatch import fnmatchcase
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PLANNER_PATH = ROOT / "scripts" / "plan-ci.py"
FIXTURE_ROOT = ROOT / "scripts" / "tests" / "fixtures" / "ci-plan"
SPEC = importlib.util.spec_from_file_location("plan_ci_under_test", PLANNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to import {PLANNER_PATH}")
PLANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANNER)


def fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


class PlanCiTests(unittest.TestCase):
    def test_manifest_root_is_independent_from_workspace_root(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace_directory,
            tempfile.TemporaryDirectory() as manifest_directory,
        ):
            workspace_root = Path(workspace_directory)
            manifest_root = Path(manifest_directory)
            manifest_ci = manifest_root / "ci"
            manifest_ci.mkdir()
            ownership = json.loads((ROOT / "ci" / "ownership.yml").read_text())
            ownership["path_rules"].append(
                {"domain": "docs", "patterns": ["source-only/**"]}
            )
            (manifest_ci / "ownership.yml").write_text(json.dumps(ownership))
            shutil.copyfile(ROOT / "ci" / "slices.yml", manifest_ci / "slices.yml")
            payload = fixture("docs-only.json")
            payload.update(
                {
                    "changed_files": ["source-only/guide.md"],
                    "workspace_packages": [
                        {"name": "workspace-crate", "path": "crates/workspace-crate"}
                    ],
                    "affected_crates": ["workspace-crate"],
                }
            )

            plan = PLANNER.build_plan(
                payload,
                root=workspace_root,
                manifest_root=manifest_root,
            )

            self.assertEqual(plan["domains"], ["docs"])

    def test_workspace_operations_remain_rooted_at_workspace_root(self) -> None:
        with (
            tempfile.TemporaryDirectory() as workspace_directory,
            tempfile.TemporaryDirectory() as manifest_directory,
        ):
            workspace_root = Path(workspace_directory)
            manifest_root = Path(manifest_directory)
            manifest_ci = manifest_root / "ci"
            manifest_ci.mkdir()
            shutil.copyfile(ROOT / "ci" / "ownership.yml", manifest_ci / "ownership.yml")
            shutil.copyfile(ROOT / "ci" / "slices.yml", manifest_ci / "slices.yml")
            payload = fixture("runtime.json")
            del payload["workspace_packages"]
            payload["affected_crates"] = []
            package_ids = {
                "mesh-llm-host-runtime": "host-id",
                "mesh-llm": "binary-id",
            }
            metadata = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {
                        "workspace_members": list(package_ids.values()),
                        "workspace_root": str(workspace_root),
                        "packages": [
                            {
                                "id": package_id,
                                "name": package_name,
                                "manifest_path": str(
                                    workspace_root
                                    / "crates"
                                    / package_name
                                    / "Cargo.toml"
                                ),
                            }
                            for package_name, package_id in package_ids.items()
                        ],
                    }
                ),
                stderr="",
            )
            affected = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout='{"affected":["mesh-llm-host-runtime","mesh-llm"]}',
                stderr="",
            )

            with mock.patch.object(
                PLANNER.subprocess,
                "run",
                side_effect=(metadata, affected),
            ) as run:
                PLANNER.build_plan(
                    payload,
                    root=workspace_root,
                    manifest_root=manifest_root,
                )

            self.assertEqual(run.call_count, 2)
            metadata_call, affected_call = run.call_args_list
            self.assertEqual(metadata_call.kwargs["cwd"], workspace_root)
            self.assertEqual(
                metadata_call.args[0],
                ["cargo", "metadata", "--format-version=1", "--no-deps"],
            )
            self.assertEqual(affected_call.kwargs["cwd"], workspace_root)
            self.assertEqual(
                affected_call.args[0],
                [
                    "bash",
                    str(workspace_root / "scripts" / "affected-crates.sh"),
                    "--stdin",
                ],
            )

    def test_missing_source_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as manifest_directory:
            manifest_root = Path(manifest_directory)
            manifest_ci = manifest_root / "ci"
            manifest_ci.mkdir()
            shutil.copyfile(ROOT / "ci" / "ownership.yml", manifest_ci / "ownership.yml")

            with self.assertRaisesRegex(
                PLANNER.PlanError,
                r"unable to load .*ci/slices\.yml",
            ):
                PLANNER.build_plan(
                    fixture("docs-only.json"),
                    root=ROOT,
                    manifest_root=manifest_root,
                )

    def test_manifest_root_defaults_to_workspace_root(self) -> None:
        payload = fixture("docs-only.json")

        default_plan = PLANNER.build_plan(payload, root=ROOT)
        explicit_plan = PLANNER.build_plan(payload, root=ROOT, manifest_root=ROOT)

        self.assertEqual(default_plan, explicit_plan)

    def test_all_tracked_paths_have_ownership(self) -> None:
        ownership = json.loads((ROOT / "ci" / "ownership.yml").read_text())
        patterns = [
            pattern
            for rule in ownership["path_rules"]
            for pattern in rule["patterns"]
        ]
        tracked = subprocess.check_output(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
        ).splitlines()

        unmatched = [
            path
            for path in tracked
            if not any(fnmatchcase(path, pattern) for pattern in patterns)
        ]

        self.assertEqual([], unmatched)

    def test_imported_justfiles_use_existing_rust_ownership(self) -> None:
        payload = fixture("docs-only.json")
        payload["changed_files"] = ["just/build.just"]

        plan = PLANNER.build_plan(payload, root=ROOT)

        self.assertEqual(plan["domains"], ["rust"])

    def test_docs_only_selects_the_fast_quality_slice(self) -> None:
        plan = PLANNER.build_plan(fixture("docs-only.json"), root=ROOT)

        self.assertEqual(plan["domains"], ["docs"])
        self.assertEqual(plan["direct_crates"], [])
        self.assertEqual(plan["affected_crates"], [])
        self.assertEqual(plan["required_slices"], ["quality"])
        self.assertEqual(plan["matrices"]["clippy"], [])
        self.assertEqual(plan["matrices"]["rust_tests"], [])
        self.assertEqual(plan["matrices"]["runtime_products"], [])
        self.assertEqual(
            plan["signals"],
            {
                "rust_changed": False,
                "ui_changed": False,
                "website_changed": False,
                "website_docs_changed": False,
                "cli_surface_changed": False,
                "docs_only": True,
                "backend_changed": False,
                "runner_contract_required": False,
            },
        )

    def test_draft_profile_skips_build_slices_for_regular_changes(self) -> None:
        payload = fixture("runtime.json")
        payload.update({"profile": "pr-draft", "event_name": "pull_request"})

        plan = PLANNER.build_plan(payload, root=ROOT)

        self.assertEqual(plan["required_slices"], [])
        self.assertTrue(all(not rows for rows in plan["matrices"].values()))

    def test_draft_docs_and_ci_control_keeps_runner_contract_without_rows(self) -> None:
        payload = fixture("docs-only.json")
        payload.update(
            {
                "profile": "pr-draft",
                "changed_files": ["CONTRIBUTING.md", ".github/README.md"],
            }
        )

        plan = PLANNER.build_plan(payload, root=ROOT)

        self.assertEqual(plan["domains"], ["ci-control", "docs"])
        self.assertTrue(plan["signals"]["docs_only"])
        self.assertTrue(plan["signals"]["runner_contract_required"])
        self.assertEqual(plan["required_slices"], ["runner-contract"])
        self.assertEqual(
            plan["reasons"]["runner-contract"],
            ["domain:ci-control"],
        )
        self.assertTrue(all(not rows for rows in plan["matrices"].values()))

    def test_runtime_change_separates_direct_and_affected_crates(self) -> None:
        plan = PLANNER.build_plan(fixture("runtime.json"), root=ROOT)

        self.assertEqual(plan["direct_crates"], ["mesh-llm-host-runtime"])
        self.assertEqual(
            plan["affected_crates"],
            ["mesh-llm-host-runtime", "mesh-llm"],
        )
        self.assertEqual(plan["domains"], ["rust", "runtime-product"])
        self.assertEqual(
            plan["required_slices"],
            ["quality", "ui-artifact", "static-abi", "rust-tests", "runtime-product", "product-smoke"],
        )
        self.assertEqual(
            [row["id"] for row in plan["matrices"]["runtime_products"]],
            ["linux-cpu"],
        )
        self.assertEqual(
            plan["dependencies"]["runtime-product"],
            ["ui-artifact"],
        )
        self.assertIn("dependency:runtime-product", plan["reasons"]["ui-artifact"])
        self.assertTrue(plan["signals"]["rust_changed"])
        self.assertTrue(plan["signals"]["backend_changed"])

    def test_cli_surface_change_selects_inventory_validation(self) -> None:
        payload = fixture("runtime.json")
        payload["changed_files"] = ["crates/mesh-llm-cli/src/parser/commands.rs"]
        payload["workspace_packages"] = [
            {"name": "mesh-llm-cli", "path": "crates/mesh-llm-cli"}
        ]
        payload["affected_crates"] = ["mesh-llm-cli"]

        plan = PLANNER.build_plan(payload, root=ROOT)

        self.assertEqual(plan["domains"], ["cli", "rust"])
        self.assertIn("quality", plan["required_slices"])
        self.assertNotIn("web", plan["required_slices"])
        self.assertTrue(plan["signals"]["cli_surface_changed"])
        self.assertFalse(plan["signals"]["website_docs_changed"])

    def test_backend_change_adds_only_the_owned_backend_rows(self) -> None:
        payload = fixture("runtime.json")
        payload["changed_files"] = ["scripts/build-linux-rocm.sh"]
        payload["affected_crates"] = ["mesh-llm"]

        plan = PLANNER.build_plan(payload, root=ROOT)

        self.assertEqual(
            plan["domains"],
            ["tooling", "runtime-product", "backend-rocm"],
        )
        self.assertEqual(
            [row["id"] for row in plan["matrices"]["runtime_products"]],
            ["linux-cpu", "linux-rocm", "windows-rocm"],
        )
        self.assertEqual(
            [row["id"] for row in plan["matrices"]["smoke"]],
            ["product-integration-cpu", "qwen-recurrent-gate"],
        )

    def test_cuda_change_selects_the_gpu_smoke_row(self) -> None:
        payload = fixture("runtime.json")
        payload["changed_files"] = ["scripts/detect-cuda-arch.sh"]
        payload["affected_crates"] = ["mesh-llm"]

        plan = PLANNER.build_plan(payload, root=ROOT)

        self.assertEqual(
            [row["id"] for row in plan["matrices"]["smoke"]],
            ["core-cuda"],
        )

    def test_macos_platform_change_selects_portable_and_unit_rows(self) -> None:
        payload = fixture("runtime.json")
        payload["changed_files"] = ["scripts/build-mac.sh"]
        payload["affected_crates"] = ["mesh-llm"]

        plan = PLANNER.build_plan(payload, root=ROOT)

        self.assertEqual(
            [row["id"] for row in plan["matrices"]["platform_checks"]],
            ["macos-portable", "macos-unit"],
        )

    def test_log_store_selects_only_the_windows_storage_privacy_row(self) -> None:
        payload = fixture("runtime.json")
        payload["changed_files"] = ["crates/mesh-llm-log-store/src/lib.rs"]
        payload["workspace_packages"] = [
            {"name": "mesh-llm-log-store", "path": "crates/mesh-llm-log-store"}
        ]
        payload["affected_crates"] = ["mesh-llm-log-store"]

        plan = PLANNER.build_plan(payload, root=ROOT)

        self.assertEqual(
            [row["id"] for row in plan["matrices"]["platform_checks"]],
            ["windows-log-store"],
        )
        self.assertEqual(plan["matrices"]["runtime_products"], [])
        self.assertEqual(plan["matrices"]["smoke"], [])

    def test_main_covers_every_workspace_crate_once(self) -> None:
        plan = PLANNER.build_plan(fixture("main.json"), root=ROOT)

        workspace = {"mesh-llm", "mesh-llm-host-runtime", "mesh-llm-config"}
        tested = {
            crate
            for batch in plan["matrices"]["rust_tests"]
            for crate in batch["crates"]
        }
        self.assertEqual(tested, workspace)
        self.assertEqual(len(plan["matrices"]["runtime_products"]), 9)
        self.assertEqual(
            {row["id"] for row in plan["matrices"]["smoke"]},
            {
                "product-integration-cpu",
                "qwen-recurrent-gate",
                "core-cuda",
                "model-download",
                "metal-model-load",
            },
        )
        self.assertIn(
            "macos-unit",
            {row["id"] for row in plan["matrices"]["platform_checks"]},
        )
        self.assertEqual(
            {row["id"] for row in plan["matrices"]["sdk"]},
            {"rust", "kotlin", "swift"},
        )
        self.assertEqual(plan["budgets"]["total_max_workers"], 18)
        self.assertEqual(
            plan["cache_modes"]["runtime-product"],
            "trusted-readwrite",
        )

    def test_full_profiles_ignore_partial_affected_crate_input(self) -> None:
        workspace = {"mesh-llm", "mesh-llm-host-runtime", "mesh-llm-config"}
        for profile, event_name in (
            ("main", "push"),
            ("manual-full", "workflow_dispatch"),
        ):
            with self.subTest(profile=profile):
                payload = fixture("main.json")
                payload.update(
                    {
                        "profile": profile,
                        "event_name": event_name,
                        "affected_crates": ["mesh-llm"],
                    }
                )

                plan = PLANNER.build_plan(payload, root=ROOT)

                tested = {
                    crate
                    for batch in plan["matrices"]["rust_tests"]
                    for crate in batch["crates"]
                }
                self.assertEqual(set(plan["affected_crates"]), workspace)
                self.assertEqual(tested, workspace)

    def test_control_plane_changes_fail_open_to_the_profile_rows(self) -> None:
        payload = fixture("runtime.json")
        payload["changed_files"] = [".github/workflows/pr_linux.yml"]
        payload["affected_crates"] = []

        plan = PLANNER.build_plan(payload, root=ROOT)

        self.assertIn("ci-control", plan["domains"])
        self.assertIn("runner-contract", plan["required_slices"])
        self.assertEqual(len(plan["matrices"]["runtime_products"]), 9)
        self.assertIn(
            "control-plane:fail-open",
            plan["reasons"]["runner-contract"],
        )

    def test_control_plane_changes_select_static_abi_for_its_matrix_gated_consumers(
        self,
    ) -> None:
        # static_abi is the only lane job gated on required_slices while its
        # consumers (rust_tests, kotlin_sdk_input) are gated on the matrices
        # force_all_rows populates. If either matrix comes back non-empty,
        # static-abi must be selected too, or the consumer can never run.
        cases = (
            ("pr-draft", "pull_request", "runtime.json"),
            ("pr-ready", "pull_request", "runtime.json"),
            ("main", "push", "main.json"),
            ("manual-full", "workflow_dispatch", "main.json"),
        )
        for profile, event_name, fixture_name in cases:
            with self.subTest(profile=profile):
                payload = fixture(fixture_name)
                payload.update(
                    {
                        "profile": profile,
                        "event_name": event_name,
                        "changed_files": [".github/workflows/pr_linux.yml"],
                        "affected_crates": [],
                    }
                )

                plan = PLANNER.build_plan(payload, root=ROOT)

                kotlin_planned = "kotlin" in {
                    row["id"] for row in plan["matrices"]["sdk"]
                }
                rust_tests_planned = bool(plan["matrices"]["rust_tests"])
                # This fixture always plans kotlin via force_all_rows, so
                # assert the concrete case directly rather than relying on
                # readers to trace that through -- the conditional below
                # keeps the invariant general for any future payload.
                self.assertIn("static-abi", plan["required_slices"])
                if kotlin_planned or rust_tests_planned:
                    self.assertIn("static-abi", plan["required_slices"])

    def test_macos_consumer_rows_reject_multiple_architectures(self) -> None:
        slices = json.loads((ROOT / "ci" / "slices.yml").read_text())
        extra = copy.deepcopy(
            next(row for row in slices["runtime_rows"] if row["id"] == "macos-metal")
        )
        extra.update({"id": "macos-metal-amd64", "architecture": "amd64"})
        slices["runtime_rows"].append(extra)

        with self.assertRaisesRegex(
            PLANNER.PlanError,
            "macOS runtime_products must use one architecture",
        ):
            PLANNER._select_rows(
                slices,
                profile="main",
                domains=[],
                selected=set(),
                force_all_rows=True,
            )

    def test_markdown_ci_control_change_does_not_force_all_rows(self) -> None:
        for path in (".github/README.md", ".omo/specs/ci-note.md"):
            with self.subTest(path=path):
                payload = fixture("docs-only.json")
                payload["changed_files"] = [path]
                payload["affected_crates"] = ["mesh-llm"]

                plan = PLANNER.build_plan(payload, root=ROOT)

                self.assertTrue(plan["signals"]["docs_only"])
                self.assertIn("runner-contract", plan["required_slices"])
                self.assertEqual(plan["matrices"]["runtime_products"], [])
                self.assertNotIn(
                    "control-plane:fail-open",
                    plan["reasons"]["runner-contract"],
                )

    def test_website_owned_generated_path_is_not_docs_only(self) -> None:
        payload = fixture("docs-only.json")
        payload["changed_files"] = ["docs/index.html"]
        payload["affected_crates"] = ["mesh-llm"]

        plan = PLANNER.build_plan(payload, root=ROOT)

        self.assertIn("website", plan["domains"])
        self.assertIn("web", plan["required_slices"])
        self.assertFalse(plan["signals"]["docs_only"])

    def test_empty_affected_crates_uses_fallback_computation(self) -> None:
        payload = fixture("runtime.json")
        payload["affected_crates"] = []
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"affected":["mesh-llm-host-runtime","mesh-llm"]}',
            stderr="",
        )

        with mock.patch.object(PLANNER.subprocess, "run", return_value=completed) as run:
            plan = PLANNER.build_plan(payload, root=ROOT)

        run.assert_called_once()
        self.assertEqual(
            plan["affected_crates"],
            ["mesh-llm-host-runtime", "mesh-llm"],
        )

    def test_falsey_malformed_affected_crates_are_rejected(self) -> None:
        for value in (False, ""):
            with self.subTest(value=value):
                payload = fixture("runtime.json")
                payload["affected_crates"] = value
                with self.assertRaisesRegex(
                    PLANNER.PlanError,
                    "affected_crates must be an array of non-empty strings",
                ):
                    PLANNER.build_plan(payload, root=ROOT)

    def test_semantic_validation_rejects_unselected_map_keys_and_duplicate_ids(self) -> None:
        payload = fixture("runtime.json")
        plan = PLANNER.build_plan(payload, root=ROOT)
        slices = json.loads((ROOT / "ci" / "slices.yml").read_text())
        packages = payload["workspace_packages"]

        for field in ("reasons", "dependencies", "runner_roles", "cache_modes"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(plan)
                invalid[field]["not-selected"] = (
                    [] if field == "dependencies" else "invalid"
                )
                with self.assertRaisesRegex(
                    PLANNER.PlanError,
                    f"plan {field} keys must equal required_slices",
                ):
                    PLANNER._validate_plan(invalid, slices, packages)

        invalid = copy.deepcopy(plan)
        invalid["matrices"]["runtime_products"].append(
            copy.deepcopy(invalid["matrices"]["runtime_products"][0])
        )
        with self.assertRaisesRegex(
            PLANNER.PlanError,
            "plan matrix runtime_products contains duplicate rows",
        ):
            PLANNER._validate_plan(invalid, slices, packages)

    def test_manifest_platform_budgets_must_fit_total_worker_ceiling(self) -> None:
        ownership = json.loads((ROOT / "ci" / "ownership.yml").read_text())
        slices = json.loads((ROOT / "ci" / "slices.yml").read_text())
        slices["profiles"]["pr-ready"]["budgets"]["total_max_workers"] = 9

        with self.assertRaisesRegex(
            PLANNER.PlanError,
            "profile pr-ready.budgets platform ceilings exceed total_max_workers",
        ):
            PLANNER._validate_manifests(ownership, slices)

    def test_manifest_domain_row_mapping_rejects_unknown_ids(self) -> None:
        ownership = json.loads((ROOT / "ci" / "ownership.yml").read_text())
        slices = json.loads((ROOT / "ci" / "slices.yml").read_text())
        slices["platform_domain_rows"]["platform-windows"].append("missing-row")

        with self.assertRaisesRegex(
            PLANNER.PlanError,
            "platform_domain_rows.platform-windows references unknown rows",
        ):
            PLANNER._validate_manifests(ownership, slices)

    def test_manifest_runtime_row_requires_runner_role(self) -> None:
        ownership = json.loads((ROOT / "ci" / "ownership.yml").read_text())
        slices = json.loads((ROOT / "ci" / "slices.yml").read_text())
        del slices["runtime_rows"][0]["runner_role"]

        with self.assertRaisesRegex(
            PLANNER.PlanError,
            r"runtime_rows\[0\]\.runner_role must be a non-empty string",
        ):
            PLANNER._validate_manifests(ownership, slices)

    def test_unknown_paths_and_unknown_input_fields_fail_closed(self) -> None:
        payload = fixture("docs-only.json")
        payload["changed_files"] = ["new-owner/unknown.dat"]
        with self.assertRaisesRegex(
            PLANNER.PlanError,
            "ownership has no rule for changed paths: new-owner/unknown.dat",
        ):
            PLANNER.build_plan(payload, root=ROOT)

        invalid = fixture("docs-only.json")
        invalid["unexpected"] = True
        with self.assertRaisesRegex(
            PLANNER.PlanError,
            "planner input has unknown fields: unexpected",
        ):
            PLANNER.build_plan(invalid, root=ROOT)

    def test_profile_event_contract_is_strict(self) -> None:
        payload = fixture("docs-only.json")
        payload["profile"] = "main"
        with self.assertRaisesRegex(
            PLANNER.PlanError,
            "profile main requires push or workflow_dispatch",
        ):
            PLANNER.build_plan(payload, root=ROOT)

        payload = fixture("docs-only.json")
        payload["source_sha"] = "not-a-sha"
        with self.assertRaisesRegex(
            PLANNER.PlanError,
            "source_sha must be a lowercase 40-character SHA",
        ):
            PLANNER.build_plan(payload, root=ROOT)

    def test_cli_emits_one_versioned_json_plan(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(PLANNER_PATH),
                "--manifest-root",
                str(ROOT),
            ],
            cwd=ROOT,
            input=json.dumps(fixture("runtime.json")),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(plan["profile"], "pr-ready")

        invalid = subprocess.run(
            [sys.executable, str(PLANNER_PATH)],
            cwd=ROOT,
            input="{}",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("ERROR: unable to build CI plan", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
