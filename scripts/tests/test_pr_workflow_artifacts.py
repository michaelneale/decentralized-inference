from pathlib import Path
import re
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
PLATFORM = WORKFLOWS / "ci-platform-checks-slice.yml"
NON_VALIDATION_PR_WORKFLOWS = {"pr_auto_assign.yml", "pr_cleanup.yml"}


def workflow_triggers(path: Path) -> set[str]:
    document = yaml.safe_load(path.read_text()) or {}
    raw = document.get("on", document.get(True, {}))
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(trigger) for trigger in raw}
    if isinstance(raw, dict):
        return {str(trigger) for trigger in raw}
    return set()


class PrWorkflowArtifactTests(unittest.TestCase):
    def workflow(self, name: str) -> str:
        return (WORKFLOWS / name).read_text()

    def job_blocks(self, workflow: str) -> dict[str, str]:
        """Split a workflow's raw text into {job_name: block_text} by
        top-level (2-space-indented) job keys under `jobs:`."""
        pattern = re.compile(r"^  ([A-Za-z_][\w-]*):$", re.MULTILINE)
        matches = list(pattern.finditer(workflow))
        blocks = {}
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(workflow)
            blocks[match.group(1)] = workflow[start:end]
        return blocks

    def test_windows_log_store_privacy_checks_are_platform_owned(self):
        workflow = PLATFORM.read_text()
        self.assertIn("name: Test Windows log artifact privacy ACL", workflow)
        self.assertIn(
            "windows_artifact_paths_have_current_owner_and_exact_user_only_dacl",
            workflow,
        )
        self.assertIn("name: Test Windows log SQLite storage ACL", workflow)
        self.assertIn(
            "sqlite_root_database_and_sidecars_have_only_current_user_acl",
            workflow,
        )

    def test_pr_entrypoints_call_protected_native_lanes(self):
        for lane in ("quality", "website", "linux", "macos", "windows"):
            workflow = self.workflow(f"pr_{lane}.yml")
            self.assertIn("pull_request:", workflow)
            self.assertIn(
                f"uses: Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main",
                workflow,
            )
            self.assertNotIn("actions.createWorkflowDispatch", workflow)
            self.assertNotIn("prepare-host-input", workflow)

    def test_retained_ci_shim_is_inert(self):
        workflow = self.workflow("ci.yml")
        self.assertEqual({"workflow_call"}, workflow_triggers(WORKFLOWS / "ci.yml"))
        self.assertNotIn("uses: ./.github/workflows/ci-", workflow)
        self.assertNotIn("Mesh-LLM/mesh-llm/.github/workflows/ci-", workflow)

    def test_pr_validation_has_exactly_five_focused_entrypoints(self):
        expected = {
            "pr_quality.yml": "quality",
            "pr_website.yml": "website",
            "pr_linux.yml": "linux",
            "pr_macos.yml": "macos",
            "pr_windows.yml": "windows",
        }
        workflow_paths = sorted(
            [*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]
        )
        pr_attached = {
            path.name
            for path in workflow_paths
            if {"pull_request", "pull_request_target"} & workflow_triggers(path)
        }
        self.assertTrue(NON_VALIDATION_PR_WORKFLOWS <= pr_attached)
        actual = pr_attached - NON_VALIDATION_PR_WORKFLOWS
        self.assertEqual(set(expected), actual)
        for filename, lane in expected.items():
            workflow = self.workflow(filename)
            protected_calls = [
                line.strip()
                for line in workflow.splitlines()
                if line.strip().startswith("uses: Mesh-LLM/mesh-llm/.github/workflows/ci-")
            ]
            self.assertEqual(
                [f"uses: Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main"],
                protected_calls,
            )
            self.assertNotIn("paths:", workflow)
            self.assertNotIn("createWorkflowDispatch", workflow)

    def test_main_validation_has_exactly_five_focused_entrypoints(self):
        expected = {
            "main_quality.yml": "quality",
            "main_website.yml": "website",
            "main_linux.yml": "linux",
            "main_macos.yml": "macos",
            "main_windows.yml": "windows",
        }
        main_paths = sorted(
            [*WORKFLOWS.glob("main_*.yml"), *WORKFLOWS.glob("main_*.yaml")]
        )
        actual = {
            path.name for path in main_paths if "push" in workflow_triggers(path)
        }
        self.assertEqual(set(expected), actual)

        for filename, lane in expected.items():
            workflow = self.workflow(filename)
            local_calls = [
                line.strip()
                for line in workflow.splitlines()
                if line.strip().startswith("uses: ./.github/workflows/ci-")
            ]
            self.assertEqual(
                [f"uses: ./.github/workflows/ci-{lane}-lane.yml"],
                local_calls,
            )
            self.assertNotIn("paths:", workflow)
            self.assertNotIn("concurrency:", workflow)
            self.assertNotIn("createWorkflowDispatch", workflow)

        controller = self.workflow("ci-control.yml")
        self.assertIn("workflow_dispatch:", controller)
        self.assertNotIn("workflow_run:", controller)
        self.assertNotIn("\n  push:\n", controller)

    def test_ci_docs_forbid_monolithic_or_dispatch_only_pr_visibility(self):
        docs = (ROOT / "ci" / "ci.md").read_text()
        self.assertIn("The five-way split is a hard CI architecture invariant", docs)
        self.assertIn("`dispatched`, with the real work detached", docs)
        self.assertIn("Do not add another all-lanes PR", docs)
        self.assertIn("Do not funnel main pushes through `ci-control.yml`", docs)

    def test_controller_dispatches_only_named_topic_and_platform_workflows(self):
        workflow = self.workflow("ci-control.yml")
        for lane in ("quality", "website", "linux", "macos", "windows"):
            self.assertIn(f"'ci-{lane}-lane.yml'", workflow)
        self.assertEqual(1, workflow.count("uses: ./.github/actions/plan-ci"))
        self.assertIn("name: 'CI Required'", workflow)
        compatibility = self.workflow("ci.yml")
        self.assertNotIn("createWorkflowDispatch", compatibility)

    def test_platform_consumers_require_only_matching_producers(self):
        for platform in ("linux", "macos", "windows"):
            with self.subTest(platform=platform):
                lane = self.workflow(f"ci-{platform}-lane.yml")
                self.assertIn("needs: [hosts, native_runtimes]", lane)
                self.assertIn("needs.hosts.result == 'success'", lane)
                self.assertIn("needs.native_runtimes.result == 'success'", lane)
                native_start = lane.index("  native_runtimes:")
                product_start = lane.index("  runtime_product:")
                self.assertLess(native_start, product_start)
                native = lane[native_start:product_start]
                self.assertNotIn("needs.hosts", native)

    def test_host_slices_are_platform_pure(self):
        expected = {
            "linux": ("Linux", "ci-host-linux-", "macOS host", "Windows host"),
            "macos": ("macOS", "ci-host-macos-", "Linux host", "Windows host"),
            "windows": ("Windows", "ci-host-windows-", "Linux host", "macOS host"),
        }
        for platform, (label, artifact, other_a, other_b) in expected.items():
            with self.subTest(platform=platform):
                workflow = self.workflow(f"ci-{platform}-host-slice.yml")
                self.assertIn(f"name: {label} host (", workflow)
                self.assertIn(f"name: {artifact}", workflow)
                self.assertIn("name: ${{ inputs.ui_artifact_name }}", workflow)
                self.assertNotIn(other_a, workflow)
                self.assertNotIn(other_b, workflow)
        windows = self.workflow("ci-windows-host-slice.yml")
        self.assertIn("prepare-windows-host-input", windows)
        self.assertNotIn("build-windows.ps1", windows)

    def test_runtime_producers_and_product_composers_are_separate(self):
        for platform in ("linux", "macos", "windows"):
            with self.subTest(platform=platform):
                runtime = self.workflow(f"ci-{platform}-runtime-slice.yml")
                product = self.workflow(f"ci-{platform}-product-slice.yml")
                self.assertIn("prepare-native-runtime-input", runtime)
                self.assertNotIn("compose-product-input", runtime)
                self.assertIn("compose-product-input", product)
                self.assertNotIn("prepare-native-runtime-input", product)
                self.assertNotIn("cargo build", product)
                self.assertNotIn("compose_products", runtime + product)

    def test_control_plane_fail_open_executes_both_web_rows(self):
        workflow = self.workflow("ci-website-lane.yml")
        control_domain = "contains(fromJson(inputs.lane_plan_json).domains, 'ci-control')"
        self.assertIn(
            "ui_changed: ${{ fromJson(inputs.lane_plan_json).signals.ui_changed || "
            f"{control_domain} }}}}",
            workflow,
        )
        self.assertIn(
            "website_changed: ${{ fromJson(inputs.lane_plan_json).signals.website_changed || "
            f"{control_domain} }}}}",
            workflow,
        )

    def test_rust_test_batches_isolate_cargo_feature_resolution(self):
        workflow = self.workflow("ci-rust-tests-slice.yml")
        self.assertIn('cargo test --locked -p "$crate"', workflow)
        self.assertNotIn('args+=("-p" "$crate")', workflow)

    def test_rust_tests_restore_only_verified_trusted_model_cache(self):
        workflow = self.workflow("ci-rust-tests-slice.yml")
        self.assertIn(
            "ci/model-artifacts/manifests/skippy-correctness.json",
            workflow,
        )
        self.assertIn(
            "scripts/resolve-test-model-manifest.py",
            workflow,
        )
        self.assertIn("Restore Skippy correctness model cache", workflow)
        self.assertIn("uses: actions/cache/restore@", workflow)
        self.assertIn(
            "needs.runner_policy.outputs.allow_native_github_cache == 'true'",
            workflow,
        )
        self.assertIn(
            '--revision "${{ steps.skippy_correctness_model.outputs.revision }}"',
            workflow,
        )
        self.assertIn('--verify-root "$RUNNER_TEMP/skippy-correctness-model"', workflow)
        self.assertIn("Save trusted Skippy correctness model cache", workflow)
        self.assertIn("uses: actions/cache/save@", workflow)
        self.assertIn(
            "if: ${{ contains(toJson(matrix.batch.crates), 'skippy-runtime') && "
            "needs.runner_policy.outputs.allow_native_github_cache == 'true'",
            workflow,
        )
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn(
            "inputs.original_event_name != 'pull_request_target'",
            workflow,
        )
        self.assertIn(
            "steps.skippy_correctness_model_cache.outputs.cache-hit != 'true'",
            workflow,
        )
        restore_start = workflow.index("- name: Restore Skippy correctness model cache")
        download_start = workflow.index("- name: Download Skippy correctness model")
        verify_start = workflow.index("- name: Verify Skippy correctness model")
        save_start = workflow.index("- name: Save trusted Skippy correctness model cache")
        restore_block = workflow[restore_start:download_start]
        verify_block = workflow[verify_start:save_start]
        save_block = workflow[save_start : workflow.index("- name: Run isolated Cargo tests")]
        self.assertNotIn("restore-keys:", restore_block)
        self.assertNotIn("cache-hit", verify_block)
        self.assertIn("github.ref == 'refs/heads/main'", save_block)
        self.assertLess(restore_start, download_start)
        self.assertLess(download_start, verify_start)
        self.assertLess(verify_start, save_start)

    def test_full_swift_sdk_has_a_cold_native_build_budget(self):
        workflow = self.workflow("ci-macos-lane.yml")
        self.assertIn("timeout_minutes: 90", workflow)

    def test_pr_platform_critical_matrices_fail_fast_by_profile(self):
        slices = (
            "ci-rust-tests-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-linux-runtime-slice.yml",
            "ci-linux-product-slice.yml",
            "ci-macos-host-slice.yml",
            "ci-macos-runtime-slice.yml",
            "ci-macos-product-slice.yml",
            "ci-windows-host-slice.yml",
            "ci-windows-runtime-slice.yml",
            "ci-windows-product-slice.yml",
            "ci-platform-checks-slice.yml",
        )
        for filename in slices:
            with self.subTest(filename=filename):
                workflow = self.workflow(filename)
                self.assertIn("fail_fast:", workflow)
                self.assertIn("fail-fast: ${{ inputs.fail_fast }}", workflow)

        for platform in ("linux", "macos", "windows"):
            lane = self.workflow(f"ci-{platform}-lane.yml")
            self.assertIn(
                "fail_fast: ${{ inputs.original_event_name == 'pull_request' }}",
                lane,
            )

        quality = self.workflow("ci-quality-slice.yml")
        self.assertIn("fail-fast: false", quality)
        self.assertNotIn("fail-fast: ${{ inputs.fail_fast }}", quality)

    def test_pr_cache_publishers_are_exact_and_bounded(self):
        ui_artifact = self.workflow("ci-ui-artifact-slice.yml")
        website = self.workflow("ci-web-slice.yml")
        # UI installs no longer round-trip through the Actions cache at all
        # (#1392): the runner image bakes a warm pnpm store and every pnpm
        # job in these two files points store-dir at it directly, so there
        # is nothing here to save or restore.
        self.assertNotIn("name: Save pnpm store", ui_artifact)
        self.assertNotIn("name: Restore pnpm store", ui_artifact)
        self.assertNotIn("actions/cache", ui_artifact)
        self.assertNotIn("name: Save pnpm store", website)
        self.assertNotIn("name: Restore pnpm store", website)
        self.assertNotIn("actions/cache", website)
        # Every pnpm job in the two files points store-dir at the image's
        # baked store directly: once for ui_artifact, and once each for
        # ui_quality and ui_e2e.
        store_dir_config = "run: pnpm config set store-dir /home/runner/.local/share/pnpm/store"
        self.assertEqual(1, ui_artifact.count(store_dir_config))
        website_jobs = self.job_blocks(website)
        for job in ("ui_quality", "ui_e2e"):
            self.assertEqual(
                1,
                website_jobs[job].count(store_dir_config),
                f"expected exactly one baked-store config in {job}",
            )
        # The `website` job itself runs in the prebuilt public-web image with
        # no bare-metal row, so it has no native-cache-gated npm consumer
        # left to publish or bound -- setup-node's own cache was deleted
        # outright, not gated.

        windows = self.workflow("ci-windows-runtime-slice.yml")
        self.assertIn("name: Save exact PR-scoped Windows ABI build", windows)
        self.assertIn("key: ${{ steps.llama_cache.outputs.cache-primary-key }}", windows)
        self.assertNotIn("restore-keys:", windows)

        platform = self.workflow("ci-platform-checks-slice.yml")
        self.assertIn("inputs.original_event_name == 'pull_request'", platform)
        self.assertIn("key: ${{ steps.llama_cache.outputs.cache-primary-key }}", platform)

        rust_tests = self.workflow("ci-rust-tests-slice.yml")
        self.assertNotIn("Swatinem/rust-cache@", rust_tests)
        self.assertIn("uses: ./.github/actions/restore-sccache-seed", rust_tests)
        self.assertIn("allow_trusted_sccache_seed", rust_tests)


if __name__ == "__main__":
    unittest.main()
