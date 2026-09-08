from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

try:
    from scripts.tests.ci_test_helpers import RunnerSelectorMixin
except ModuleNotFoundError:
    from ci_test_helpers import RunnerSelectorMixin


ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ROOT / ".github" / "actions"


class CiRunnerSelectionAndCachePolicyTests(RunnerSelectorMixin, unittest.TestCase):
    def read_action(self, name: str) -> str:
        return (ACTIONS / name / "action.yml").read_text(encoding="utf-8")

    def test_actionlint_installer_verifies_pinned_release_archives(
        self,
    ) -> None:
        action = self.read_action("install-actionlint")

        self.assertIn('ACTIONLINT_VERSION: "1.7.12"', action)
        self.assertIn(
            "8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8",
            action,
        )
        self.assertIn(
            "325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6",
            action,
        )
        self.assertIn("actionlint archive checksum mismatch", action)
        self.assertIn("scripts/safe-extract-tar.py", action)
        self.assertNotIn("tar -x", action)

    def test_sccache_prefers_depot_webdav_with_disk_fallback(self) -> None:
        action = self.read_action("configure-sccache-gha")

        self.assertIn("allow_depot_remote_cache", action)
        self.assertIn('default: "false"', action)
        self.assertIn("allow_native_github_cache", action)
        self.assertIn('default: "false"', action)
        self.assertIn("SCCACHE_WEBDAV_ENDPOINT", action)
        self.assertIn("DEPOT_CACHE_TOKEN", action)
        self.assertIn("process.env.SCCACHE_DIR", action)
        self.assertIn("process.env.RUNNER_TEMP", action)
        self.assertIn("await io.mkdirP(diskCacheDirectory)", action)
        self.assertIn(
            "core.exportVariable('SCCACHE_DIR', diskCacheDirectory)",
            action,
        )
        self.assertIn("SCCACHE_DIR: diskCacheDirectory", action)
        self.assertIn("'disk,webdav'", action)
        self.assertIn("'disk'", action)
        self.assertIn(
            "Depot cache is present but disabled for this trust context",
            action,
        )
        self.assertIn(
            "Native GitHub and Depot cache disabled for this trust context",
            action,
        )
        self.assertIn(
            "'Unable to start baked sccache with its trust-isolated disk cache.'",
            action,
        )
        self.assertIn("env: diskOnlyEnvironment()", action)
        self.assertNotIn(
            "core.exportVariable('ACTIONS_RUNTIME_TOKEN', '')",
            action,
        )

    def test_runner_selection_uses_event_repository_and_ref_policy(self) -> None:
        action = self.read_action("select-ci-runners")

        self.assertIn("depot_main_enabled", action)
        self.assertIn("depot_pr_enabled", action)
        self.assertIn("\n  repository:", action)
        self.assertIn("INPUT_REPOSITORY", action)
        self.assertIn("\n  head_repository:", action)
        self.assertIn("INPUT_HEAD_REPOSITORY", action)
        self.assertIn("INPUT_DEPOT_PR_ENABLED", action)
        self.assertIn("\n  ref:", action)
        self.assertIn("refs/pull/[0-9]+/merge", action)
        self.assertIn("pull_request_target)", action)
        self.assertIn("allow_depot_remote_cache=false", action)
        self.assertIn("depot-ubuntu-24.04-16", action)
        self.assertIn("depot-ubuntu-24.04-arm-16", action)
        self.assertIn("depot-macos-15", action)
        self.assertIn("depot-windows-2022", action)
        self.assertIn('depot_pr_exception_expires="2026-09-14"', action)
        self.assertNotIn("INPUT_PR_APPROVED_REF", action)
        self.assertNotIn("INPUT_PR_APPROVED_SHA", action)

        selector_calls = 0
        approved_policy_calls = 0
        for workflow_path in sorted(
            (ROOT / ".github" / "workflows").glob("*.yml")
        ):
            lines = workflow_path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if "uses: ./.github/actions/select-ci-runners" not in line:
                    continue
                selector_calls += 1
                block = "\n".join(lines[index : index + 20])
                with self.subTest(selector_caller=workflow_path.name):
                    self.assertIn("head_sha:", block)
                if "pr_approved_ref:" in block:
                    approved_policy_calls += 1
                    self.assertIn("pr_approved_sha:", block)
        self.assertEqual(selector_calls, 19)
        # The release workflow selects a runner for a non-PR ref and does not
        # pass the pull-request approval inputs.
        self.assertEqual(approved_policy_calls, 18)

        cases = (
            (
                "pull_request",
                "refs/pull/12/merge",
                "false",
                "false",
                "true",
                "true",
                "depot-ubuntu-24.04",
                "false",
            ),
            (
                "pull_request",
                "refs/pull/12/merge",
                "false",
                "false",
                "true",
                "false",
                "ubuntu-24.04",
                "false",
            ),
            (
                "pull_request_target",
                "refs/pull/12/merge",
                "true",
                "true",
                "true",
                "false",
                "ubuntu-24.04",
                "false",
            ),
            (
                "workflow_dispatch",
                "refs/heads/main",
                "true",
                "false",
                "",
                "false",
                "depot-ubuntu-24.04",
                "false",
            ),
            (
                "workflow_dispatch",
                "refs/heads/main",
                "false",
                "true",
                "",
                "false",
                "depot-ubuntu-24.04",
                "false",
            ),
            (
                "workflow_dispatch",
                "refs/heads/main",
                "true",
                "true",
                "",
                "false",
                "depot-ubuntu-24.04",
                "false",
            ),
            (
                "workflow_dispatch",
                "refs/heads/main",
                "true",
                "true",
                "push",
                "false",
                "depot-ubuntu-24.04",
                "false",
            ),
            (
                "workflow_dispatch",
                "refs/heads/feature",
                "true",
                "true",
                "",
                "true",
                "ubuntu-24.04",
                "false",
            ),
            (
                "push",
                "refs/heads/main",
                "true",
                "false",
                "",
                "false",
                "depot-ubuntu-24.04",
                "false",
            ),
            (
                "push",
                "refs/heads/feature",
                "true",
                "false",
                "",
                "false",
                "ubuntu-24.04",
                "false",
            ),
            (
                "push",
                "refs/tags/v1.2.3",
                "true",
                "false",
                "",
                "false",
                "ubuntu-24.04",
                "false",
            ),
            (
                "push",
                "refs/heads/main",
                "false",
                "false",
                "",
                "false",
                "ubuntu-24.04",
                "false",
            ),
            (
                "schedule",
                "refs/heads/main",
                "true",
                "true",
                "",
                "false",
                "ubuntu-24.04",
                "false",
            ),
        )
        for (
            event_name,
            ref,
            main,
            manual,
            original_event_name,
            pr_enabled,
            runner,
            cache_enabled,
        ) in cases:
            with self.subTest(event_name=event_name, ref=ref):
                outputs = self.run_runner_selector(
                    event_name=event_name,
                    ref=ref,
                    main_enabled=main,
                    manual_enabled=manual,
                    original_event_name=original_event_name,
                    pr_enabled=pr_enabled,
                    pr_approved_ref=(
                        ref
                        if event_name == "pull_request"
                        and pr_enabled == "true"
                        and runner.startswith("depot-")
                        else ""
                    ),
                    pr_approved_sha=(
                        "0123456789abcdef0123456789abcdef01234567"
                        if event_name == "pull_request"
                        and pr_enabled == "true"
                        and runner.startswith("depot-")
                        else ""
                    ),
                )
                enabled = "true" if runner.startswith("depot-") else "false"
                self.assertEqual(outputs["depot_enabled"], enabled)
                self.assertEqual(
                    outputs["allow_depot_remote_cache"],
                    cache_enabled,
                )
                expected_native_cache = (
                    "true"
                    if event_name == "pull_request" and enabled == "true"
                    else "false" if enabled == "true" else "true"
                )
                self.assertEqual(
                    outputs["allow_native_github_cache"],
                    expected_native_cache,
                )
                self.assertEqual(
                    outputs["allow_trusted_sccache_seed"],
                    "false" if enabled == "true" else "true",
                )
                self.assertEqual(outputs["runner"], runner)
                expected_arm = (
                    "depot-ubuntu-24.04-arm"
                    if enabled == "true"
                    else "ubuntu-24.04-arm"
                )
                self.assertEqual(outputs["runner_arm"], expected_arm)
                for size in ("4", "8", "16"):
                    expected_sized_arm = (
                        f"depot-ubuntu-24.04-arm-{size}"
                        if enabled == "true"
                        else "ubuntu-24.04-arm"
                    )
                    self.assertEqual(
                        outputs[f"runner_arm_{size}"],
                        expected_sized_arm,
                    )
                expected_macos = (
                    "depot-macos-15"
                    if enabled == "true"
                    else "macos-15"
                )
                expected_windows = (
                    "depot-windows-2022"
                    if enabled == "true"
                    else "windows-2022"
                )
                self.assertEqual(outputs["runner_macos"], expected_macos)
                self.assertEqual(outputs["runner_windows"], expected_windows)

        untrusted_repository = self.run_runner_selector(
            event_name="pull_request",
            ref="refs/pull/12/merge",
            main_enabled="true",
            manual_enabled="true",
            pr_enabled="true",
            pr_approved_ref="refs/pull/12/merge",
            pr_approved_sha="0123456789abcdef0123456789abcdef01234567",
            repository="attacker/mesh-llm",
        )
        self.assertEqual(untrusted_repository["depot_enabled"], "false")
        self.assertEqual(untrusted_repository["runner"], "ubuntu-24.04")
        self.assertEqual(
            untrusted_repository["allow_depot_remote_cache"],
            "false",
        )
        self.assertEqual(
            untrusted_repository["allow_native_github_cache"],
            "true",
        )

        fork_head_repository = self.run_runner_selector(
            event_name="pull_request",
            ref="refs/pull/12/merge",
            main_enabled="true",
            manual_enabled="true",
            pr_enabled="true",
            pr_approved_ref="refs/pull/12/merge",
            pr_approved_sha="0123456789abcdef0123456789abcdef01234567",
            head_repository="attacker/mesh-llm",
        )
        self.assertEqual(fork_head_repository["depot_enabled"], "false")
        self.assertEqual(fork_head_repository["runner"], "ubuntu-24.04")
        self.assertEqual(
            fork_head_repository["allow_native_github_cache"],
            "true",
        )

        runner_contract_change = self.run_runner_selector(
            event_name="pull_request",
            ref="refs/pull/12/merge",
            main_enabled="true",
            manual_enabled="true",
            pr_enabled="true",
            pr_approved_ref="refs/pull/12/merge",
            pr_approved_sha="0123456789abcdef0123456789abcdef01234567",
            force_hosted="true",
        )
        self.assertEqual(runner_contract_change["depot_enabled"], "false")
        self.assertEqual(runner_contract_change["runner"], "ubuntu-24.04")
        self.assertEqual(
            runner_contract_change["allow_native_github_cache"],
            "true",
        )

        non_merge_ref = self.run_runner_selector(
            event_name="pull_request",
            ref="refs/pull/12/head",
            main_enabled="true",
            manual_enabled="true",
            pr_enabled="true",
            pr_approved_ref="refs/pull/12/merge",
            pr_approved_sha="0123456789abcdef0123456789abcdef01234567",
        )
        self.assertEqual(non_merge_ref["depot_enabled"], "false")
        self.assertEqual(non_merge_ref["runner"], "ubuntu-24.04")

        untrusted_dispatch = self.run_runner_selector(
            event_name="workflow_dispatch",
            ref="refs/heads/main",
            main_enabled="true",
            manual_enabled="true",
            original_event_name="pull_request_target",
            pr_enabled="true",
        )
        self.assertEqual(untrusted_dispatch["depot_enabled"], "false")
        self.assertEqual(untrusted_dispatch["runner"], "ubuntu-24.04")
        self.assertEqual(
            untrusted_dispatch["allow_depot_remote_cache"],
            "false",
        )
        self.assertEqual(
            untrusted_dispatch["allow_native_github_cache"],
            "true",
        )

        canary_pr = self.run_runner_selector(
            event_name="pull_request",
            ref="refs/pull/12/merge",
            main_enabled="false",
            manual_enabled="false",
            pr_enabled="false",
            pr_canary_ref="refs/pull/12/merge",
        )
        self.assertEqual(canary_pr["depot_enabled"], "true")
        self.assertEqual(canary_pr["runner"], "depot-ubuntu-24.04")
        self.assertEqual(canary_pr["allow_depot_remote_cache"], "false")
        self.assertEqual(canary_pr["allow_native_github_cache"], "false")
        self.assertEqual(canary_pr["allow_trusted_sccache_seed"], "false")

        globally_enabled_pr = self.run_runner_selector(
            event_name="pull_request",
            ref="refs/pull/12/merge",
            main_enabled="false",
            manual_enabled="false",
            pr_enabled="true",
        )
        self.assertEqual(globally_enabled_pr["depot_enabled"], "true")
        self.assertEqual(globally_enabled_pr["runner"], "depot-ubuntu-24.04")
        self.assertEqual(
            globally_enabled_pr["allow_native_github_cache"],
            "true",
        )

        stale_approval = self.run_runner_selector(
            event_name="pull_request",
            ref="refs/pull/12/merge",
            main_enabled="false",
            manual_enabled="false",
            pr_enabled="true",
            pr_approved_ref="refs/pull/12/merge",
            pr_approved_sha="fedcba9876543210fedcba9876543210fedcba98",
        )
        self.assertEqual(stale_approval["depot_enabled"], "true")

        stale_ref_approval = self.run_runner_selector(
            event_name="pull_request",
            ref="refs/pull/12/merge",
            main_enabled="false",
            manual_enabled="false",
            pr_enabled="true",
            pr_approved_ref="refs/pull/13/merge",
            pr_approved_sha="0123456789abcdef0123456789abcdef01234567",
        )
        self.assertEqual(stale_ref_approval["depot_enabled"], "true")
        self.assertEqual(stale_ref_approval["runner"], "depot-ubuntu-24.04")
        self.assertEqual(
            stale_ref_approval["allow_native_github_cache"],
            "true",
        )
        self.assertEqual(
            stale_ref_approval["allow_depot_remote_cache"],
            "false",
        )

        expired_approval = self.run_runner_selector(
            event_name="pull_request",
            ref="refs/pull/12/merge",
            main_enabled="false",
            manual_enabled="false",
            pr_enabled="true",
            pr_approved_ref="refs/pull/12/merge",
            pr_approved_sha="0123456789abcdef0123456789abcdef01234567",
            current_date="2026-09-14",
        )
        self.assertEqual(expired_approval["depot_enabled"], "false")
        self.assertEqual(expired_approval["allow_native_github_cache"], "true")

        trusted_main_cross_branch_cache = self.run_runner_selector(
            event_name="push",
            ref="refs/heads/main",
            main_enabled="true",
            manual_enabled="false",
            pr_enabled="true",
        )
        self.assertEqual(trusted_main_cross_branch_cache["depot_enabled"], "true")
        self.assertEqual(
            trusted_main_cross_branch_cache["allow_native_github_cache"],
            "true",
        )
        self.assertEqual(
            trusted_main_cross_branch_cache["allow_trusted_sccache_seed"],
            "false",
        )

        for name, kwargs in (
            (
                "empty canary ref",
                {"pr_canary_ref": ""},
            ),
            (
                "different pull-request ref",
                {"pr_canary_ref": "refs/pull/13/merge"},
            ),
            (
                "fork head",
                {
                    "pr_canary_ref": "refs/pull/12/merge",
                    "head_repository": "attacker/mesh-llm",
                },
            ),
            (
                "pull_request_target",
                {
                    "pr_canary_ref": "refs/pull/12/merge",
                    "event_name": "pull_request_target",
                },
            ),
            (
                "forced hosted",
                {
                    "pr_canary_ref": "refs/pull/12/merge",
                    "force_hosted": "true",
                },
            ),
            (
                "dispatch source",
                {
                    "pr_canary_ref": "refs/pull/12/merge",
                    "event_name": "workflow_dispatch",
                    "ref": "refs/heads/main",
                },
            ),
        ):
            with self.subTest(canary_case=name):
                case = {
                    "event_name": "pull_request",
                    "ref": "refs/pull/12/merge",
                    "main_enabled": "false",
                    "manual_enabled": "false",
                    "pr_enabled": "false",
                    **kwargs,
                }
                selected = self.run_runner_selector(**case)
                self.assertEqual(selected["depot_enabled"], "false")
                self.assertEqual(selected["runner"], "ubuntu-24.04")
                self.assertEqual(
                    selected["allow_depot_remote_cache"],
                    "false",
                )

        action = self.read_action("select-ci-runners")
        run_block = action.split("      run: |\n", maxsplit=1)[1]
        script = "\n".join(
            line[8:] if line.startswith("        ") else line
            for line in run_block.splitlines()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "github-output"
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=ROOT,
                env={
                    **os.environ,
                    "GITHUB_OUTPUT": str(output),
                    "INPUT_EVENT_NAME": "pull_request",
                    "INPUT_REPOSITORY": "Mesh-LLM/mesh-llm",
                    "INPUT_HEAD_REPOSITORY": "Mesh-LLM/mesh-llm",
                    "INPUT_HEAD_SHA": "0123456789abcdef0123456789abcdef01234567",
                    "INPUT_REF": "refs/pull/12/merge",
                    "INPUT_DEPOT_MAIN_ENABLED": "false",
                    "INPUT_DEPOT_PR_ENABLED": "false",
                    "INPUT_PR_CANARY_REF": "refs/heads/main",
                    "INPUT_PR_APPROVED_REF": "",
                    "INPUT_PR_APPROVED_SHA": "",
                    "INPUT_FORCE_HOSTED": "false",
                    "INPUT_MANUAL_USE_DEPOT": "false",
                },
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exact pull-request merge ref", result.stderr)

    def test_dispatched_pr_cache_writes_remain_blocked_with_depot(
        self,
    ) -> None:
        workflow_names = (
            "ci-quality-slice.yml",
            "ci-rust-tests-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-linux-runtime-slice.yml",
            "static-abi-artifact.yml",
        )
        for workflow_name in workflow_names:
            workflow = (
                ROOT / ".github" / "workflows" / workflow_name
            ).read_text(encoding="utf-8")
            with self.subTest(workflow=workflow_name):
                self.assertIn("CACHE_NAMESPACE: mesh-llm", workflow)
                self.assertNotIn("CACHE_NAMESPACE: mesh-llm-pr", workflow)
                self.assertNotIn("'mesh-llm-pr'", workflow)
                if workflow_name in {
                    "ci-quality-slice.yml",
                    "ci-rust-tests-slice.yml",
                    "ci-linux-host-slice.yml",
                    "ci-linux-runtime-slice.yml",
                }:
                    self.assertIn('SCCACHE_GHA_ENABLED: "false"', workflow)
                else:
                    self.assertNotIn('SCCACHE_GHA_ENABLED: "false"', workflow)
                if "uses: Swatinem/rust-cache@" in workflow:
                    self.assertIn(
                        "save-if: ${{ github.ref == 'refs/heads/main' && ",
                        workflow,
                    )
                    self.assertIn(
                        "github.event.inputs.original_event_name != 'pull_request'",
                        workflow,
                    )
                if "uses: ./.github/actions/configure-sccache-gha" in workflow:
                    self.assertIn("allow_depot_remote_cache", workflow)

        for pr_path in (ROOT / ".github" / "workflows").glob("pr_*.yml"):
            pr = pr_path.read_text(encoding="utf-8")
            self.assertNotIn("depot-ubuntu", pr)
            self.assertNotIn("SCCACHE_GHA_ENABLED: \"false\"", pr)

    def test_depot_pr_native_cache_consumers_obey_central_policy(self) -> None:
        eligible_consumers = {
            "ci-quality-slice.yml": ("uses: ./.github/actions/restore-sccache-seed",),
            # ci-web-slice.yml and ci-ui-artifact-slice.yml have no native
            # GitHub cache consumers left: their pnpm jobs (ui_quality,
            # ui_e2e, ui_artifact) point store-dir at the runner image's
            # baked store instead of the Actions cache (#1392), and
            # `website` was already deleted-outright rather than gated (see
            # the comment on that job's entry below).
            "ci-linux-host-slice.yml": ("uses: ./.github/actions/restore-sccache-seed",),
            "ci-linux-runtime-slice.yml": ("uses: ./.github/actions/restore-sccache-seed",),
            "ci-rust-tests-slice.yml": ("uses: ./.github/actions/restore-sccache-seed",),
            "ci-macos-host-slice.yml": ("Swatinem/rust-cache@",),
            "ci-platform-checks-slice.yml": (
                "uses: actions/cache/restore@",
                "uses: actions/cache/save@",
            ),
            "ci-windows-host-slice.yml": ("Swatinem/rust-cache@",),
            "ci-windows-runtime-slice.yml": (
                "uses: ./.github/actions/restore-windows-abi-cache",
                "use-github-cache:",
                "uses: jakoch/install-vulkan-sdk-action@",
                "uses: ./.github/actions/setup-windows-rocm-sdk",
                "uses: actions/cache/save@",
            ),
            "static-abi-artifact.yml": ("uses: actions/cache@",),
            "swift-sdk-artifact.yml": (
                "cache: ${{ needs.runner_policy.outputs.allow_native_github_cache",
                "Swatinem/rust-cache@",
                "uses: actions/cache@",
            ),
        }
        expected_jobs = {
            "ci-quality-slice.yml": {
                "runner_policy", "quality_contracts", "rust_fmt", "rust_clippy", "cli_docs_sync", "authority_sentinel",
            },
            "ci-web-slice.yml": {"runner_policy", "ui_quality", "ui_e2e", "website"},
            "ci-ui-artifact-slice.yml": {"runner_policy", "ui_artifact"},
            "ci-linux-host-slice.yml": {"runner_policy", "linux_host"},
            "ci-linux-runtime-slice.yml": {"runner_policy", "linux_runtime"},
            "ci-rust-tests-slice.yml": {
                "runner_policy",
                "rust_tests",
                "safetensors_runtime_smoke",
            },
            "ci-macos-host-slice.yml": {"runner_policy", "macos_host"},
            "ci-platform-checks-slice.yml": {"runner_policy", "platform_checks"},
            "ci-windows-host-slice.yml": {"runner_policy", "windows_host"},
            "ci-windows-runtime-slice.yml": {"runner_policy", "windows_runtime"},
            "static-abi-artifact.yml": {"runner_policy", "static_abi_artifact"},
            "swift-sdk-artifact.yml": {
                "runner_policy",
                "swift_sdk_target",
                "swift_sdk_artifact",
            },
        }

        def step_block(workflow: str, marker: str) -> str:
            lines = workflow.splitlines()
            for index, line in enumerate(lines):
                if marker not in line:
                    continue
                indent = len(line) - len(line.lstrip())
                step_indent = indent if line.lstrip().startswith("-") else indent - 2
                start = index
                while start > 0:
                    candidate = lines[start - 1]
                    candidate_indent = len(candidate) - len(candidate.lstrip())
                    if candidate_indent == step_indent and candidate.lstrip().startswith("-"):
                        start -= 1
                        break
                    if candidate_indent < step_indent:
                        break
                    start -= 1
                end = index + 1
                while end < len(lines):
                    candidate = lines[end]
                    candidate_indent = len(candidate) - len(candidate.lstrip())
                    if candidate_indent == step_indent and candidate.lstrip().startswith("-"):
                        break
                    end += 1
                return "\n".join(lines[start:end])
            self.fail(f"missing cache consumer marker: {marker}")

        for filename, markers in eligible_consumers.items():
            workflow = (
                ROOT / ".github" / "workflows" / filename
            ).read_text(encoding="utf-8")
            with self.subTest(workflow=filename):
                self.assertIn(
                    "allow_native_github_cache: ${{ steps.policy.outputs.allow_native_github_cache }}",
                    workflow,
                )
                for marker in markers:
                    block = step_block(workflow, marker)
                    with self.subTest(consumer=marker):
                        if "restore-sccache-seed" in marker:
                            self.assertIn("allow_trusted_sccache_seed", block)
                        else:
                            self.assertIn("allow_native_github_cache", block)

        for filename, jobs in expected_jobs.items():
            workflow = (
                ROOT / ".github" / "workflows" / filename
            ).read_text(encoding="utf-8")
            job_section = workflow.split("\njobs:\n", maxsplit=1)[1]
            actual_jobs = set(re.findall(r"^  ([A-Za-z0-9_]+):", job_section, re.MULTILINE))
            with self.subTest(workflow=filename):
                self.assertEqual(jobs, actual_jobs)
                self.assertNotRegex(
                    job_section,
                    r"^  [A-Za-z0-9_]+:\n(?:    [^\n]*\n){0,4}    if:.*allow_native_github_cache",
                )

        swift = (
            ROOT / ".github" / "workflows" / "swift-sdk-artifact.yml"
        ).read_text(encoding="utf-8")
        windows = (
            ROOT / ".github" / "workflows" / "ci-windows-runtime-slice.yml"
        ).read_text(encoding="utf-8")
        release = (
            ROOT / ".github" / "workflows" / "release.yml"
        ).read_text(encoding="utf-8")
        warmer = (
            ROOT / ".github" / "workflows" / "windows-warm-caches.yml"
        ).read_text(encoding="utf-8")
        native_cache_expression = (
            "needs.runner_policy.outputs.allow_native_github_cache == 'true'"
        )
        # ci-web-slice.yml's `website` job runs in the prebuilt public-web
        # image (no bare-metal row), so its setup-node native-cache
        # consumer was deleted outright rather than gated -- there is
        # nothing left in that job for the depot/native cache policy to
        # govern. The other jobs in that file (ui_quality, ui_e2e) and in
        # ci-ui-artifact-slice.yml (ui_artifact) have no native-cache
        # consumer left either now that they point at the runner image's
        # baked pnpm store instead (#1392); see the comment on
        # `eligible_consumers` above.
        self.assertIn(
            f"cache: ${{{{ inputs.ui_artifact_name == '' && {native_cache_expression} && 'pnpm' || '' }}}}",
            swift,
        )
        self.assertIn(
            f"package-manager-cache: ${{{{ inputs.ui_artifact_name == '' && {native_cache_expression} }}}}",
            swift,
        )
        self.assertIn(
            f"use-github-cache: ${{{{ {native_cache_expression} }}}}",
            windows,
        )
        self.assertIn(
            f"cache: ${{{{ {native_cache_expression} }}}}",
            windows,
        )

        for action_name in ("restore-windows-abi-cache", "setup-windows-rocm-sdk"):
            action = self.read_action(action_name)
            self.assertIn("inputs.allow-native-github-cache == 'true'", action)

        # Trusted Depot release selections must leave native cache consumers
        # inert, while the hosted release/cache-warmer paths retain their
        # existing GitHub cache opt-in.
        self.assertIn(
            "allow_native_github_cache: ${{ steps.runners.outputs.allow_native_github_cache }}",
            release,
        )
        for cache_name in (
            "Cache native runtime ROCm backend build",
            "Cache native runtime Vulkan backend build",
        ):
            cache_start = release.index(f"name: {cache_name}")
            cache_block = release[cache_start : release.find("\n      - ", cache_start + 1)]
            self.assertIn(
                "!startsWith(needs.metadata.outputs.runner_16, 'depot-')",
                cache_block,
            )
        self.assertGreaterEqual(
            warmer.count('allow-native-github-cache: "true"'),
            2,
        )

    def test_authority_sentinel_is_explicit_cache_gate_exemption(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "ci-quality-slice.yml"
        ).read_text(encoding="utf-8")
        jobs = workflow.split("\njobs:\n", maxsplit=1)[1]
        match = re.search(
            r"^  authority_sentinel:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
            jobs,
            re.MULTILINE | re.DOTALL,
        )
        if match is None:
            self.fail("authority sentinel job was not found")
        sentinel = match.group("body")
        self.assertIn(
            "# Explicit diagnostic exception: this no-checkout job attests the",
            workflow,
        )
        self.assertIn("authority_sentinel", jobs)
        self.assertIn("Attest provider-injected cache backend", sentinel)
        self.assertIn("actions/cache/restore@", sentinel)
        self.assertIn("actions/cache/save@", sentinel)
        self.assertNotIn("allow_native_github_cache", sentinel)
        self.assertNotIn("allow_depot_remote_cache", sentinel)
        self.assertNotIn("audit-depot-pr-isolation@", sentinel)

    def test_depot_sccache_consumers_receive_both_central_cache_outputs(self) -> None:
        provider_workflows = (
            "ci-quality-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-linux-runtime-slice.yml",
            "ci-rust-tests-slice.yml",
            "ci-windows-host-slice.yml",
            "ci-windows-runtime-slice.yml",
            "static-abi-artifact.yml",
            "native-sdk-artifact.yml",
            "swift-sdk-artifact.yml",
        )
        for filename in provider_workflows:
            workflow = (
                ROOT / ".github" / "workflows" / filename
            ).read_text(encoding="utf-8")
            self.assertIn(
                "allow_depot_remote_cache: ${{ needs.runner_policy.outputs.allow_depot_remote_cache }}",
                workflow,
                filename,
            )
            self.assertIn(
                "allow_native_github_cache: ${{ needs.runner_policy.outputs.allow_native_github_cache }}",
                workflow,
                filename,
            )
        release = (
            ROOT / ".github" / "workflows" / "release.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "allow_native_github_cache: ${{ ((matrix.target == 'x86_64-unknown-linux-gnu' && startsWith(needs.metadata.outputs.runner_8, 'depot-')) || (matrix.target == 'aarch64-unknown-linux-gnu' && startsWith(needs.metadata.outputs.runner_arm_8, 'depot-'))) && 'false' || 'true' }}",
            release,
        )


if __name__ == "__main__":
    unittest.main()
