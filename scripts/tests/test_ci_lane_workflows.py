from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


class CiLaneWorkflowTests(unittest.TestCase):
    def workflow(self, name: str) -> str:
        return (WORKFLOWS / name).read_text(encoding="utf-8")

    def test_manual_controller_plans_once_and_dispatches_native_lane_inputs(self) -> None:
        workflow = self.workflow("ci-control.yml")
        self.assertEqual(1, workflow.count("uses: ./.github/actions/plan-ci"))
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("workflow_run:", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("\n  push:\n", workflow)
        self.assertIn("actions: write", workflow)
        self.assertIn("checks: write", workflow)
        self.assertIn("github.rest.actions.createWorkflowDispatch", workflow)
        self.assertIn("PLAN_DIGEST", workflow)
        self.assertIn("planner output digest mismatch", workflow)
        self.assertIn(
            "ref: ${{ github.event.repository.default_branch }}",
            workflow,
        )
        self.assertIn("ref: process.env.DEFAULT_BRANCH", workflow)
        self.assertNotIn("ref: process.env.SOURCE_REF", workflow)
        self.assertNotIn("ref: process.env.SOURCE_SHA", workflow)
        self.assertNotIn("download-artifact", workflow)
        for name in ("quality", "website", "linux", "macos", "windows"):
            self.assertIn(f"ci-{name}-lane.yml", workflow)

    def test_prs_use_protected_native_reusable_lanes(self) -> None:
        workflow = self.workflow("ci-control.yml")
        self.assertNotIn("workflow_run:", workflow)

        for lane in ("quality", "website", "linux", "macos", "windows"):
            pr_workflow = self.workflow(f"pr_{lane}.yml")
            self.assertIn("pull_request:", pr_workflow)
            self.assertEqual(1, pr_workflow.count("uses: ./.github/actions/plan-ci"))
            self.assertIn(
                f"uses: Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main",
                pr_workflow,
            )
            self.assertIn(f"name: PR / {'macOS' if lane == 'macos' else lane.title()}", pr_workflow)
            self.assertNotIn("github.rest.actions.createWorkflowDispatch", pr_workflow)

    def test_main_uses_five_same_commit_native_reusable_lanes(self) -> None:
        labels = {
            "quality": "Quality",
            "website": "Website",
            "linux": "Linux",
            "macos": "macOS",
            "windows": "Windows",
        }
        for lane, label in labels.items():
            workflow = self.workflow(f"main_{lane}.yml")
            self.assertIn("\n  push:\n", workflow)
            self.assertIn("branches: [main]", workflow)
            self.assertEqual(1, workflow.count("uses: ./.github/actions/plan-ci"))
            self.assertIn(
                f"uses: ./.github/workflows/ci-{lane}-lane.yml",
                workflow,
            )
            self.assertNotIn(
                f"uses: Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main",
                workflow,
            )
            self.assertIn(f"name: Main / {label}", workflow)
            self.assertIn('if [[ "$BASE_SHA" =~ ^0+$ ]]', workflow)
            self.assertEqual(3, workflow.count("base_sha: ${{ steps.identity.outputs.base_sha }}"))
            self.assertIn('[[ "$LANE_RESULT" == "skipped" ]]', workflow)
            self.assertNotIn("concurrency:", workflow)
            self.assertNotIn("createWorkflowDispatch", workflow)

    def test_controller_summary_tolerates_omitted_optional_matrices(self) -> None:
        workflow = self.workflow("ci-control.yml")
        for matrix in (
            "hosts",
            "runtime_products",
            "rust_tests",
            "smoke",
            "sdk",
            "platform_checks",
        ):
            with self.subTest(matrix=matrix):
                self.assertIn(f"(.matrices.{matrix} // [])[]", workflow)

    def test_pr_routes_cancel_independently(self) -> None:
        self.assertNotIn("concurrency:", self.workflow("ci.yml"))
        self.assertIn("concurrency:", self.workflow("ci-control.yml"))
        for lane in ("quality", "website", "linux", "macos", "windows"):
            workflow = self.workflow(f"pr_{lane}.yml")
            self.assertIn(f"group: pr-{lane}-", workflow)
            self.assertIn("cancel-in-progress: true", workflow)

    def test_lane_workflows_are_reusable_and_dispatchable(self) -> None:
        checks = {
            "quality": "CI / Quality",
            "website": "CI / Website",
            "linux": "CI / Linux",
            "macos": "CI / macOS",
            "windows": "CI / Windows",
        }
        for lane, check in checks.items():
            with self.subTest(lane=lane):
                workflow = self.workflow(f"ci-{lane}-lane.yml")
                self.assertIn("workflow_dispatch:", workflow)
                self.assertIn("workflow_call:", workflow)
                self.assertIn("lane_plan_json:", workflow)
                self.assertIn(f"name: {check}", workflow)
                self.assertIn("uses: ./.github/actions/report-ci-lane", workflow)
                self.assertIn(
                    "ref: ${{ github.event.repository.default_branch }}",
                    workflow,
                )

    def test_dispatched_lanes_pass_source_sha_only_to_product_workflows(
        self,
    ) -> None:
        lane_workflows = {
            "ci-quality-lane.yml": 3,
            "ci-website-lane.yml": 2,
            "ci-linux-lane.yml": 10,
            "ci-macos-lane.yml": 9,
            "ci-windows-lane.yml": 6,
        }
        for workflow_name, expected_calls in lane_workflows.items():
            with self.subTest(workflow=workflow_name):
                workflow = self.workflow(workflow_name)
                self.assertEqual(
                    expected_calls,
                    workflow.count("source_sha: ${{ inputs.source_sha }}"),
                )

        product_workflows = (
            "ci-quality-slice.yml",
            "ci-runner-contract-slice.yml",
            "ci-web-slice.yml",
            "ci-ui-artifact-slice.yml",
            "ci-rust-tests-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-macos-host-slice.yml",
            "ci-windows-host-slice.yml",
            "ci-linux-runtime-slice.yml",
            "ci-linux-product-slice.yml",
            "ci-macos-runtime-slice.yml",
            "ci-macos-product-slice.yml",
            "ci-windows-runtime-slice.yml",
            "ci-windows-product-slice.yml",
            "ci-platform-checks-slice.yml",
            "static-abi-artifact.yml",
            "native-sdk-artifact.yml",
            "swift-sdk-artifact.yml",
            "smoke.yml",
            "scripted-binary-smoke.yml",
            "product-integration-smoke.yml",
            "sdk-smoke.yml",
            "hf-download-smoke.yml",
        )
        for workflow_name in product_workflows:
            with self.subTest(workflow=workflow_name):
                workflow = self.workflow(workflow_name)
                self.assertIn("source_sha:", workflow)
                checkout_ref = (
                    "ref: ${{ inputs.source_sha }}"
                    if workflow_name in (
                        "ci-windows-runtime-slice.yml",
                        "product-integration-smoke.yml",
                    )
                    else "ref: ${{ inputs.source_sha || github.sha }}"
                )
                self.assertIn(
                    checkout_ref,
                    workflow,
                )

        windows_runtime = self.workflow("ci-windows-runtime-slice.yml")
        self.assertIn("Validate immutable source SHA", windows_runtime)
        self.assertIn("SOURCE_SHA: ${{ inputs.source_sha }}", windows_runtime)
        self.assertIn(
            "if ($env:SOURCE_SHA -notmatch '^[0-9a-f]{40}$')",
            windows_runtime,
        )
        self.assertNotIn("inputs.source_sha || github.sha", windows_runtime)

    def test_lane_plans_are_bounded_platform_projections(self) -> None:
        action = (ROOT / ".github/actions/plan-ci/action.yml").read_text(
            encoding="utf-8"
        )
        for output in (
            "quality_lane_plan",
            "website_lane_plan",
            "linux_lane_plan",
            "macos_lane_plan",
            "windows_lane_plan",
        ):
            self.assertIn(f"{output}:", action)
            self.assertIn(f'echo "{output}=$', action)
        for platform in ("linux", "macos", "windows"):
            self.assertIn(f'select(.platform == "{platform}")', action)

    def test_pr_planner_uses_only_immutable_source_manifests(self) -> None:
        action = (ROOT / ".github/actions/plan-ci/action.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn('[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]]', action)
        self.assertIn('mktemp -d "$RUNNER_TEMP/mesh-ci-manifests.XXXXXX"', action)
        self.assertIn(
            'git archive --format=tar "$SOURCE_SHA" -- ci/ownership.yml ci/slices.yml',
            action,
        )
        self.assertIn(
            'git show "$SOURCE_SHA:ci/slices.yml" | cmp -s - ci/slices.yml',
            action,
        )
        self.assertIn(
            'git show "$SOURCE_SHA:ci/ownership.yml" | cmp -s - ci/ownership.yml',
            action,
        )
        self.assertIn(
            'python3 scripts/plan-ci.py --manifest-root "$manifest_root"',
            action,
        )
        self.assertNotIn("git checkout", action)
        self.assertNotIn('git archive --format=tar "$SOURCE_SHA" -- .', action)

    def test_topic_lane_projections_are_valid_jq(self) -> None:
        action = (ROOT / ".github/actions/plan-ci/action.yml").read_text(
            encoding="utf-8"
        )
        plans = (
            {
                "profile": "pr-ready",
                "domains": ["ci-control"],
                "required_slices": ["quality", "runner-contract", "web"],
                "signals": {"rust_changed": True},
                "budgets": {"total_max_workers": 10},
                "matrices": {"clippy": [{"id": "batch-0"}]},
            },
            {
                "profile": "pr-ready",
                "domains": ["docs"],
                "required_slices": [],
                "signals": {"rust_changed": False},
                "budgets": {"total_max_workers": 2},
                "matrices": {"clippy": []},
            },
        )
        for output, lane in (
            ("quality_lane_plan", "quality"),
            ("website_lane_plan", "website"),
        ):
            match = re.search(
                rf"{output}=\$\(jq -ce '(.*?)' ci-plan\.json\)",
                action,
                re.DOTALL,
            )
            self.assertIsNotNone(match)
            for required, plan in zip((True, False), plans, strict=True):
                with self.subTest(output=output, required=required):
                    result = subprocess.run(
                        ["jq", "-ce", match.group(1)],
                        input=json.dumps(plan),
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    projection = json.loads(result.stdout)
                    self.assertEqual(lane, projection["lane"])
                    self.assertIs(required, projection["required"])
                    self.assertEqual(
                        plan["required_slices"], projection["required_slices"]
                    )
                    self.assertEqual(plan["signals"], projection["signals"])
                    self.assertEqual(plan["budgets"], projection["budgets"])
                    expected_matrices = (
                        {"clippy": plan["matrices"]["clippy"]}
                        if lane == "quality"
                        else {}
                    )
                    self.assertEqual(expected_matrices, projection["matrices"])

    def test_dispatched_pr_lanes_receive_no_hugging_face_credential(self) -> None:
        controller = self.workflow("ci-control.yml")
        self.assertNotIn("HF_TOKEN", controller)
        for name in ("ci-linux-lane.yml", "ci-macos-lane.yml"):
            workflow = self.workflow(name)
            self.assertIn("inputs.original_event_name == 'push'", workflow)

    def test_product_smoke_jobs_parse_formatted_matrix_json(self) -> None:
        smoke_workflows = {
            "ci-linux-product-smoke-slice.yml": (
                "product-integration-cpu",
                "product-integration-cuda",
                "qwen-recurrent-gate",
                "core-cuda",
                "model-download",
            ),
            "ci-macos-product-smoke-slice.yml": (
                "metal-model-load",
                "product-integration-metal",
            ),
        }
        for workflow_name, smoke_ids in smoke_workflows.items():
            workflow = self.workflow(workflow_name)
            for smoke_id in smoke_ids:
                with self.subTest(workflow=workflow_name, smoke_id=smoke_id):
                    self.assertIn(
                        f"contains(fromJson(inputs.smoke_matrix).*.id, '{smoke_id}')",
                        workflow,
                    )
            self.assertNotIn("contains(inputs.smoke_matrix,", workflow)

    def test_runtime_and_product_artifact_ids_preserve_architecture(self) -> None:
        for platform in ("linux", "macos", "windows"):
            with self.subTest(platform=platform):
                runtime = self.workflow(f"ci-{platform}-runtime-slice.yml")
                product = self.workflow(f"ci-{platform}-product-slice.yml")
                runtime_id = (
                    f"ci-runtime-{platform}-${{{{ matrix.runtime.architecture }}}}-"
                    "${{ matrix.runtime.backend }}"
                )
                product_id = (
                    f"ci-product-{platform}-${{{{ matrix.runtime.architecture }}}}-"
                    "${{ matrix.runtime.backend }}"
                )
                self.assertIn(runtime_id, runtime)
                self.assertIn(runtime_id, product)
                self.assertIn(product_id, product)

        macos_lane = self.workflow("ci-macos-lane.yml")
        self.assertEqual(
            2,
            macos_lane.count(
                "architecture: ${{ fromJson(inputs.lane_plan_json).matrices.runtime_products[0].architecture }}"
            ),
        )
        for name in (
            "ci-macos-product-smoke-slice.yml",
            "ci-macos-sdk-slice.yml",
        ):
            with self.subTest(workflow=name):
                self.assertIn(
                    "ci-product-macos-${{ inputs.architecture }}-metal",
                    self.workflow(name),
                )

        self.assertIn("  validate_plan:", macos_lane)
        self.assertIn("[.matrices.runtime_products[].architecture] | unique | length", macos_lane)
        self.assertIn("needs: [validate_plan, runtime_product, swift_sdk_input]", macos_lane)
        self.assertIn("needs: [validate_plan, runtime_product]", macos_lane)

    def test_windows_vulkan_cache_is_restore_only_for_pr_dispatches(self) -> None:
        lane = self.workflow("ci-windows-lane.yml")
        runtime = self.workflow("ci-windows-runtime-slice.yml")
        self.assertIn("original_event_name: ${{ inputs.original_event_name }}", lane)
        self.assertIn("cache: true", runtime)
        self.assertIn(
            "cache_save_if: ${{ inputs.original_event_name != 'pull_request' && inputs.original_event_name != 'pull_request_target' }}",
            runtime,
        )

    def test_dispatched_main_preserves_trusted_runner_policy(self) -> None:
        selector = (ROOT / ".github/actions/select-ci-runners/action.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("DISPATCH_ORIGINAL_EVENT_NAME", selector)
        self.assertIn("pull_request|pull_request_target)", selector)
        self.assertIn("is_dispatched_pull_request=true", selector)
        for name in ("static-abi-artifact.yml", "native-sdk-artifact.yml"):
            workflow = self.workflow(name)
            self.assertIn(
                "original_event_name: ${{ inputs.original_event_name }}",
                workflow,
            )
            self.assertIn("repository: ${{ github.repository }}", workflow)
            self.assertIn(
                "head_repository: ${{ github.event.pull_request.head.repo.full_name }}",
                workflow,
            )
            self.assertIn(
                "depot_pr_enabled: ${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
                workflow,
            )

    def test_runner_contract_changes_force_all_pr_build_slices_hosted(self) -> None:
        signal = (
            "force_hosted: "
            "${{ fromJson(inputs.lane_plan_json).signals.runner_contract_required }}"
        )
        minimum_calls = {
            "ci-quality-lane.yml": 1,
            "ci-website-lane.yml": 1,
            "ci-linux-lane.yml": 7,
            "ci-macos-lane.yml": 6,
            "ci-windows-lane.yml": 5,
        }
        for name, minimum in minimum_calls.items():
            with self.subTest(workflow=name):
                self.assertGreaterEqual(self.workflow(name).count(signal), minimum)
        selector = (ROOT / ".github/actions/select-ci-runners/action.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('"$INPUT_FORCE_HOSTED" == "false"', selector)

    def test_reporter_completes_only_correlated_checks(self) -> None:
        action = (ROOT / ".github/actions/report-ci-lane/action.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("check.external_id === process.env.CORRELATION_ID", action)
        self.assertIn("check.head_sha !== process.env.SOURCE_SHA", action)
        self.assertIn("if (process.env.OVERALL_CHECK_ID)", action)
        self.assertIn("github.paginate(github.rest.checks.listForRef", action)
        self.assertNotIn("response.data.check_runs", action)
        self.assertIn("lanes.length === expected.length", action)
        self.assertNotIn("correlated lane checks did not converge", action)

    def test_reporter_allows_protected_workflow_sha_to_differ(self) -> None:
        action = (ROOT / ".github/actions/report-ci-lane/action.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("context.sha", action)
        self.assertIn(
            "!/^[0-9a-f]{40}$/.test(process.env.SOURCE_SHA)",
            action,
        )
        self.assertIn("check.head_sha !== process.env.SOURCE_SHA", action)

    def test_manual_depot_input_is_explicitly_forwarded(self) -> None:
        main = self.workflow("ci.yml")
        controller = self.workflow("ci-control.yml")
        prs = "\n".join(
            self.workflow(f"pr_{lane}.yml")
            for lane in ("quality", "website", "linux", "macos", "windows")
        )

        self.assertNotIn("workflow_dispatch:", main)
        self.assertIn("workflow_dispatch:", controller)
        self.assertIn("context.payload.inputs?.use_depot", controller)
        self.assertIn("inputs.use_depot = process.env.USE_DEPOT", controller)
        self.assertNotIn("use_depot: true", prs)
        for name in (
            "ci-quality-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-linux-runtime-slice.yml",
            "static-abi-artifact.yml",
            "native-sdk-artifact.yml",
        ):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                self.assertIn("use_depot:", workflow)
                self.assertIn("${{ inputs.use_depot }}", workflow)

    def test_superseded_pr_runs_cancel_by_pull_request_identity(self) -> None:
        for lane in ("quality", "website", "linux", "macos", "windows"):
            pr_workflow = self.workflow(f"pr_{lane}.yml")
            self.assertIn(
                f"group: pr-{lane}-${{{{ github.event.pull_request.number }}}}",
                pr_workflow,
            )
            self.assertIn("cancel-in-progress: true", pr_workflow)
            workflow = self.workflow(f"ci-{lane}-lane.yml")
            self.assertIn("inputs.supersession_key || inputs.source_sha", workflow)


if __name__ == "__main__":
    unittest.main()
