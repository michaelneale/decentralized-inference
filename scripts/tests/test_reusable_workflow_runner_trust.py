from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


class ReusableWorkflowRunnerTrustTests(unittest.TestCase):
    def workflow(self, name: str) -> str:
        return (WORKFLOWS / name).read_text(encoding="utf-8")

    def test_reusable_workflows_accept_semantic_inputs_not_raw_runners(self) -> None:
        offenders = []
        for path in sorted(WORKFLOWS.glob("*.yml")):
            workflow = path.read_text(encoding="utf-8")
            if "workflow_call:" not in workflow:
                continue
            if "fromJson(inputs.runs_on)" in workflow or "runs_on:" in workflow:
                offenders.append(path.name)
        self.assertEqual([], offenders)

    def test_credential_smokes_remain_fixed_github_hosted(self) -> None:
        for name in (
            "hf-download-smoke.yml",
            "smoke.yml",
            "scripted-binary-smoke.yml",
            "product-integration-smoke.yml",
            "sdk-smoke.yml",
        ):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                self.assertIn("ubuntu-24.04", workflow)
                self.assertNotIn("depot-ubuntu", workflow)
                checkout_count = workflow.count("uses: actions/checkout@")
                persisted_credentials_disabled = workflow.count(
                    "persist-credentials: false"
                )
                self.assertEqual(
                    checkout_count,
                    persisted_credentials_disabled,
                )

    def test_main_entry_and_orchestrator_do_not_use_pull_request_target(self) -> None:
        for name in (
            "ci.yml",
            "ci-control.yml",
            "pr_quality.yml",
            "pr_website.yml",
            "pr_linux.yml",
            "pr_macos.yml",
            "pr_windows.yml",
        ):
            with self.subTest(workflow=name):
                self.assertNotIn("pull_request_target", self.workflow(name))

    def test_macos_and_windows_slices_use_central_platform_policy(self) -> None:
        platform_slices = {
            "macos": (
                "macos-15",
                (
                    "ci-macos-host-slice.yml",
                    "ci-macos-runtime-slice.yml",
                    "ci-macos-product-slice.yml",
                ),
            ),
            "windows": (
                "windows-2022",
                (
                    "ci-windows-host-slice.yml",
                    "ci-windows-runtime-slice.yml",
                    "ci-windows-product-slice.yml",
                ),
            ),
        }
        for platform, (runner, names) in platform_slices.items():
            for name in names:
                with self.subTest(platform=platform, workflow=name):
                    workflow = self.workflow(name)
                    output = "runner_macos" if platform == "macos" else "runner_windows"
                    self.assertIn(
                        f"{output}: ${{{{ steps.policy.outputs.{output} }}}}",
                        workflow,
                    )
                    self.assertIn(
                        f"runs-on: ${{{{ needs.runner_policy.outputs.{output} }}}}",
                        workflow,
                    )
                    self.assertIn(
                        "depot_pr_enabled: ${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
                        workflow,
                    )
                    self.assertIn(
                        "head_repository: ${{ github.event.pull_request.head.repo.full_name }}",
                        workflow,
                    )
                    selector = (ROOT / ".github" / "actions" / "select-ci-runners" / "action.yml").read_text(encoding="utf-8")
                    self.assertIn(f"depot-{runner}", selector)
                    self.assertNotIn("macos-latest", workflow)
                    self.assertNotIn("windows-latest", workflow)

    def test_all_eligible_pr_slices_bind_runner_and_cache_policy(self) -> None:
        """Every ordinary PR producer must consume the same guarded policy.

        The protected reusable lanes are deliberately split across several
        workflows, so a future slice can otherwise accidentally omit the
        bounded global gate while still looking like it uses the central
        selector.  Keep this census explicit: credential-bearing smoke
        consumers and the GPU exception are intentionally outside it.
        """

        eligible = (
            "ci-quality-slice.yml",
            "ci-web-slice.yml",
            "ci-ui-artifact-slice.yml",
            "ci-rust-tests-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-linux-runtime-slice.yml",
            "ci-linux-product-slice.yml",
            "static-abi-artifact.yml",
            "ci-macos-host-slice.yml",
            "ci-macos-runtime-slice.yml",
            "ci-macos-product-slice.yml",
            "swift-sdk-artifact.yml",
            "ci-platform-checks-slice.yml",
            "ci-windows-host-slice.yml",
            "ci-windows-runtime-slice.yml",
            "ci-windows-product-slice.yml",
            "native-sdk-artifact.yml",
        )

        selector = (
            ROOT / ".github" / "actions" / "select-ci-runners" / "action.yml"
        ).read_text(encoding="utf-8")
        self.assertNotIn("INPUT_PR_APPROVED_REF", selector)
        self.assertNotIn("INPUT_PR_APPROVED_SHA", selector)
        self.assertIn('depot_pr_exception_expires="2026-09-14"', selector)

        for name in eligible:
            workflow = self.workflow(name)
            with self.subTest(workflow=name):
                # A raw provider label would bypass the protected policy.
                self.assertNotIn("runs-on: depot-", workflow)
                self.assertNotIn("secrets: inherit", workflow)
                self.assertIn("allow_depot_remote_cache:", workflow)
                self.assertIn("allow_native_github_cache:", workflow)
                self.assertIn(
                    "head_repository: ${{ github.event.pull_request.head.repo.full_name }}",
                    workflow,
                )
                self.assertIn(
                    "head_sha: ${{ github.event.pull_request.head.sha || github.sha }}",
                    workflow,
                )
                self.assertIn("ref: ${{ github.ref }}", workflow)
                self.assertIn(
                    "depot_pr_enabled: ${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
                    workflow,
                )
                self.assertIn(
                    "pr_approved_ref: ${{ vars.DEPOT_PR_APPROVED_REF }}",
                    workflow,
                )
                self.assertIn(
                    "pr_approved_sha: ${{ vars.DEPOT_PR_APPROVED_SHA }}",
                    workflow,
                )
                self.assertIn("force_hosted: ${{ inputs.force_hosted }}", workflow)

                # Validate every selector invocation, including Quality's
                # separate no-checkout authority-sentinel selector.  The
                # sentinel intentionally supplies empty approval values, but
                # it must still receive the same bounded inputs and fail-closed
                # policy implementation.
                lines = workflow.splitlines()
                selector_indices = [
                    index
                    for index, line in enumerate(lines)
                    if "uses: ./.github/actions/select-ci-runners" in line
                ]
                self.assertGreaterEqual(len(selector_indices), 1)
                for index in selector_indices:
                    block_lines = []
                    for line in lines[index + 1 :]:
                        if line.startswith("      - "):
                            break
                        block_lines.append(line)
                    block = "\n".join(block_lines)
                    common_values = (
                        "event_name: ${{ github.event_name }}",
                        "original_event_name: ${{ inputs.original_event_name }}",
                        "repository: ${{ github.repository }}",
                        "head_repository: ${{ github.event.pull_request.head.repo.full_name }}",
                        "head_sha: ${{ github.event.pull_request.head.sha || github.sha }}",
                        "ref: ${{ github.ref }}",
                        "force_hosted: ${{ inputs.force_hosted }}",
                    )
                    for value in common_values:
                        self.assertIn(value, block, value)

                    is_sentinel = any(
                        "id: sentinel_policy" in line
                        for line in lines[max(0, index - 2) : index]
                    )
                    if is_sentinel:
                        bounded_values = (
                            "depot_main_enabled: 'false'",
                            "depot_pr_enabled: 'false'",
                            "pr_canary_ref: ${{ vars.DEPOT_PR_SENTINEL_REF }}",
                            "pr_approved_ref: ''",
                            "pr_approved_sha: ''",
                            "manual_use_depot: 'false'",
                        )
                    else:
                        bounded_values = (
                            "depot_pr_enabled: ${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
                            "pr_canary_ref: ${{ vars.DEPOT_PR_CANARY_REF }}",
                            "pr_approved_ref: ${{ vars.DEPOT_PR_APPROVED_REF }}",
                            "pr_approved_sha: ${{ vars.DEPOT_PR_APPROVED_SHA }}",
                        )
                        self.assertTrue(
                            "depot_main_enabled: 'false'" in block
                            or "depot_main_enabled: ${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}"
                            in block,
                            "depot_main_enabled must be a bounded constant or repository gate",
                        )
                        if "manual_use_depot:" in block:
                            self.assertIn(
                                "manual_use_depot: ${{ inputs.use_depot }}",
                                block,
                            )
                    for value in bounded_values:
                        self.assertIn(value, block, value)

                # The runner-owning job must consume a selector output.  The
                # platform-check slice maps two semantic OS outputs through a
                # bounded JSON object; all other producers select one output
                # directly (native SDK resolves its size in a policy helper).
                if name == "ci-platform-checks-slice.yml":
                    self.assertIn(
                        "runner_by_platform: ${{ steps.platform_runners.outputs.runner_by_platform }}",
                        workflow,
                    )
                    self.assertIn(
                        "runs-on: ${{ fromJSON(needs.runner_policy.outputs.runner_by_platform)[matrix.check.platform] }}",
                        workflow,
                    )
                else:
                    self.assertIn("runs-on: ${{ needs.runner_policy.outputs.", workflow)

    def test_main_runner_contract_preserves_image_checks(self) -> None:
        workflow = self.workflow("ci-runner-contract-slice.yml")
        self.assertIn("inputs.profile == 'main'", workflow)
        self.assertIn("runner: mesh-llm-amd64", workflow)
        self.assertIn("runner: mesh-llm-arm64", workflow)
        self.assertIn("verify-runner-image self-hosted", workflow)

    def test_runner_contract_rejects_malformed_roles_and_missing_workflows(self) -> None:
        workflow = self.workflow("ci-runner-contract-slice.yml")
        self.assertIn('.runner_roles | type == "object"', workflow)
        self.assertIn('.value | type == "string"', workflow)
        self.assertIn('test("depot"; "i")', workflow)
        self.assertIn('[[ -f "$workflow" ]]', workflow)
        self.assertIn('workflow_scan_targets+=("$workflow")', workflow)
        self.assertIn(
            '[[ "$workflow" != ".github/workflows/ci-runner-contract-slice.yml" ]]',
            workflow,
        )
        self.assertIn('"${workflow_scan_targets[@]}"', workflow)
        self.assertIn(
            "grep -nE '^[[:space:]]+pull_request_target:'",
            workflow,
        )
        self.assertNotIn("2>/dev/null", workflow)

    def test_runner_contract_scans_only_pr_validation_entrypoints(self) -> None:
        workflow = self.workflow("ci-runner-contract-slice.yml")
        for name in (
            "pr_quality.yml",
            "pr_website.yml",
            "pr_linux.yml",
            "pr_macos.yml",
            "pr_windows.yml",
        ):
            self.assertIn(f".github/workflows/{name}", workflow)
        self.assertNotIn(".github/workflows/pr_*.yml", workflow)
        self.assertNotIn(".github/workflows/pr_auto_assign.yml", workflow)
        self.assertNotIn(".github/workflows/pr_cleanup.yml", workflow)

    def test_sdk_slice_matches_parsed_row_ids(self) -> None:
        linux = self.workflow("ci-linux-sdk-slice.yml")
        macos = self.workflow("ci-macos-sdk-slice.yml")
        self.assertEqual(
            linux.count("contains(fromJson(inputs.sdk_matrix).*.id"),
            2,
        )
        self.assertEqual(
            macos.count("contains(fromJson(inputs.sdk_matrix).*.id"),
            1,
        )
        self.assertNotIn("contains(inputs.sdk_matrix", linux + macos)

    def test_depot_allowlist_excludes_credential_smokes(self) -> None:
        migration = (ROOT / "ci" / "DEPOT_MIGRATION.md").read_text(
            encoding="utf-8",
        )
        allowlist_start = migration.index("The current main allowlist is:")
        allowlist_end = migration.index("```", allowlist_start)
        allowlist_end = migration.index("```", allowlist_end + 3)
        allowlist = migration[allowlist_start:allowlist_end]
        self.assertIn("ci-linux-lane.yml@refs/heads/main", allowlist)
        self.assertIn("ci-linux-runtime-slice.yml@refs/heads/main", allowlist)
        self.assertIn("depot-canary.yml@refs/heads/main", allowlist)
        self.assertIn("release.yml@refs/heads/main", allowlist)
        for name in (
            "ci-quality-slice.yml",
            "ci-web-slice.yml",
            "ci-ui-artifact-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-linux-runtime-slice.yml",
            "ci-linux-product-slice.yml",
            "ci-rust-tests-slice.yml",
            "ci-macos-host-slice.yml",
            "ci-macos-runtime-slice.yml",
            "ci-macos-product-slice.yml",
            "ci-windows-host-slice.yml",
            "ci-windows-runtime-slice.yml",
            "ci-windows-product-slice.yml",
            "ci-platform-checks-slice.yml",
            "native-sdk-artifact.yml",
            "static-abi-artifact.yml",
            "swift-sdk-artifact.yml",
        ):
            with self.subTest(workflow=name):
                self.assertIn(f"{name}@refs/heads/main", allowlist)
        for name in (
            "hf-download-smoke.yml",
            "smoke.yml",
            "scripted-binary-smoke.yml",
            "product-integration-smoke.yml",
            "sdk-smoke.yml",
        ):
            with self.subTest(workflow=name):
                self.assertNotIn(name, allowlist)
        self.assertIn(
            "swift-sdk-artifact.yml@refs/heads/main",
            allowlist,
        )
        swift = self.workflow("swift-sdk-artifact.yml")
        self.assertIn("depot_main_enabled: 'false'", swift)
        self.assertIn(
            "depot_pr_enabled: ${{ vars.DEPOT_PR_RUNNERS_ENABLED == 'true' }}",
            swift,
        )

    def test_pr_entrypoint_maps_no_repository_secret(self) -> None:
        for lane in ("quality", "website", "linux", "macos", "windows"):
            workflow = self.workflow(f"pr_{lane}.yml")
            self.assertNotIn("secrets:", workflow)
            self.assertNotIn("HF_TOKEN", workflow)

    def test_pr_facing_checkouts_disable_persisted_credentials(self) -> None:
        names = [
            "docker-precheck.yml",
            "ci-control.yml",
            "static-abi-artifact.yml",
            *sorted(path.name for path in WORKFLOWS.glob("main_*.yml")),
            *sorted(path.name for path in WORKFLOWS.glob("pr_*.yml")),
            *sorted(
                path.name
                for pattern in ("ci-*-slice.yml", "ci-*-lane.yml")
                for path in WORKFLOWS.glob(pattern)
            ),
        ]
        for name in names:
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                checkout_count = workflow.count("uses: actions/checkout@")
                if checkout_count:
                    self.assertEqual(
                        checkout_count,
                        workflow.count("persist-credentials: false"),
                    )

    def test_future_pr_depot_executor_requires_least_privilege_token_handling(
        self,
    ) -> None:
        migration = (ROOT / "ci" / "DEPOT_MIGRATION.md").read_text(
            encoding="utf-8",
        )
        section_start = migration.index("## Future protected PR Depot executor")
        section_end = migration.index("\n## ", section_start + 4)
        section = migration[section_start:section_end]
        self.assertIn("`permissions: contents: read`", section)
        self.assertIn("`persist-credentials: false`", section)


if __name__ == "__main__":
    unittest.main()
