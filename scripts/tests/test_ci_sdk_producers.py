from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import yaml

try:
    from scripts.tests.ci_test_helpers import RunnerSelectorMixin
except ModuleNotFoundError:
    from ci_test_helpers import RunnerSelectorMixin


ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ROOT / ".github" / "actions"


class CiSdkProducersTests(RunnerSelectorMixin, unittest.TestCase):
    def read_action(self, name: str) -> str:
        return (ACTIONS / name / "action.yml").read_text(encoding="utf-8")

    def read_compute_changes(self) -> str:
        return self.read_action("compute-changes") + "\n" + (
            ACTIONS / "compute-changes" / "derive-outputs.sh"
        ).read_text(encoding="utf-8")

    def run_reusable_runner_policy(
        self,
        workflow_name: str,
        *,
        repository: str,
        event_name: str,
        ref: str,
        depot_enabled: str,
        target: str,
        runner_size: str,
        manual_use_depot: str = "false",
        pr_enabled: str = "false",
        pr_approved_ref: str = "",
        pr_approved_sha: str = "",
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
        workflow = (
            ROOT / ".github" / "workflows" / workflow_name
        ).read_text(encoding="utf-8")
        selected = self.run_runner_selector(
            event_name=event_name,
            ref=ref,
            main_enabled=depot_enabled,
            manual_enabled=manual_use_depot,
            repository=repository,
            pr_enabled=pr_enabled,
            pr_approved_ref=pr_approved_ref,
            pr_approved_sha=pr_approved_sha,
        )
        policy = workflow.split(
            "      - name: Resolve runner size and target\n",
            maxsplit=1,
        )[1]
        run_block = policy.split("        run: |\n", maxsplit=1)[1]
        script_lines: list[str] = []
        for line in run_block.splitlines():
            if line.startswith("          "):
                script_lines.append(line[10:])
            elif not line:
                script_lines.append("")
            else:
                break
        script = "\n".join(script_lines)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "github-output"
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=ROOT,
                env={
                    **os.environ,
                    "GITHUB_OUTPUT": str(output),
                    "TARGET": target,
                    "RUNNER_SIZE": runner_size,
                    "POLICY_EVENT_NAME": event_name,
                    "ALLOW_DEPOT_REMOTE_CACHE": selected[
                        "allow_depot_remote_cache"
                    ],
                    "ALLOW_NATIVE_GITHUB_CACHE": selected[
                        "allow_native_github_cache"
                    ],
                    "RUNNER_DEFAULT": selected["runner"],
                    "RUNNER_4": selected["runner_4"],
                    "RUNNER_8": selected["runner_8"],
                    "RUNNER_16": selected["runner_16"],
                    "RUNNER_ARM": selected["runner_arm"],
                    "RUNNER_ARM_4": selected["runner_arm_4"],
                    "RUNNER_ARM_8": selected["runner_arm_8"],
                    "RUNNER_ARM_16": selected["runner_arm_16"],
                    "RUNNER_MACOS": selected["runner_macos"],
                },
                check=False,
                capture_output=True,
                text=True,
            )
            outputs = {}
            if output.exists():
                outputs = dict(
                    line.split("=", maxsplit=1)
                    for line in output.read_text(
                        encoding="utf-8",
                    ).splitlines()
                )
            outputs.setdefault(
                "allow_depot_remote_cache",
                selected["allow_depot_remote_cache"],
            )
            outputs.setdefault(
                "allow_native_github_cache",
                selected["allow_native_github_cache"],
            )
            return result, outputs

    def test_sdk_routing_covers_every_direct_smoke_script(self) -> None:
        action = self.read_compute_changes()
        match = re.search(
            r"DIRECT_SDK_INPUTS=.*?grep -E '([^']+)'",
            action,
        )
        if match is None:
            self.fail("direct SDK routing pattern was not found")
        direct_sdk_pattern = re.compile(match.group(1))
        self.assertRegex(
            ".github/actions/restore-smoke-inputs/action.yml",
            direct_sdk_pattern,
        )
        for contract_path in (
            ".github/actions/compute-changes/action.yml",
            ".github/actions/compute-changes/derive-outputs.sh",
            ".github/workflows/ci.yml",
            ".github/workflows/release.yml",
        ):
            with self.subTest(contract_path=contract_path):
                self.assertRegex(contract_path, direct_sdk_pattern)
        self.assertIn("pr_[a-z]+", action)
        self.assertIn("main_[a-z]+", action)
        smoke_scripts = (
            ROOT / "scripts" / "ci-rust-sdk-smoke.sh",
            ROOT / "scripts" / "ci-kotlin-sdk-smoke.sh",
            ROOT / "scripts" / "ci-swift-sdk-smoke.sh",
        )

        direct_calls: set[str] = set()
        for smoke_script in smoke_scripts:
            direct_calls.update(
                f"scripts/{name}"
                for name in re.findall(
                    r"(?m)^\s*(?:retry_transient\s+)?"
                    r"scripts/([A-Za-z0-9_.-]+\.sh)",
                    smoke_script.read_text(encoding="utf-8"),
                )
            )

        for script in sorted(direct_calls):
            with self.subTest(script=script):
                self.assertRegex(script, direct_sdk_pattern)

    def test_native_sdk_build_is_a_shared_immutable_producer(self) -> None:
        producer = (
            ROOT / ".github" / "workflows" / "native-sdk-artifact.yml"
        ).read_text(encoding="utf-8")
        producer_action = self.read_action("prepare-native-sdk-input")
        consumer_workflow = (
            ROOT / ".github" / "workflows" / "sdk-smoke.yml"
        ).read_text(encoding="utf-8")
        consumer_script = (
            ROOT / "scripts" / "ci-kotlin-sdk-smoke.sh"
        ).read_text(encoding="utf-8")
        restore_script = (
            ROOT / "scripts" / "restore-native-sdk-input.sh"
        ).read_text(encoding="utf-8")
        routing = self.read_compute_changes()

        self.assertIn(
            "uses: ./.github/actions/prepare-native-sdk-input",
            producer,
        )
        self.assertIn(
            "uses: ./.github/workflows/static-abi-artifact.yml",
            producer,
        )
        self.assertIn(
            "scripts/restore-static-abi-input.sh",
            producer,
        )
        self.assertIn(
            "LLAMA_STAGE_BUILD_DIR: "
            ".deps/llama.cpp/build-stage-abi-static",
            producer,
        )
        self.assertIn("persist-credentials: false", producer)
        self.assertIn("actions/upload-artifact@", producer)
        self.assertIn("inputs.include_runtime_crate", producer)
        self.assertIn("RUSTC_WRAPPER: sccache", producer)
        self.assertEqual(
            producer.count(
                "uses: ./.github/actions/capture-sccache-stats",
            ),
            2,
        )
        self.assertIn(
            "sccache-native-sdk-${{ inputs.target }}-"
            "${{ inputs.backend }}-${{ inputs.profile }}-"
            "${{ github.run_attempt }}",
            producer,
        )
        self.assertIn(
            "require_prebuilt_static_abi: "
            "${{ inputs.static_abi_artifact_name != '' }}",
            producer,
        )
        linux_start = producer.index("  linux_native_sdk_artifact:")
        linux_end = producer.index("  macos_native_sdk_artifact:")
        linux_producer = producer[linux_start:linux_end]
        trust_step = (
            'name: Trust checkout directory\n'
            '        run: git config --global --add safe.directory '
            '"$GITHUB_WORKSPACE"'
        )
        self.assertIn(trust_step, linux_producer)
        self.assertLess(
            linux_producer.index("uses: actions/checkout@"),
            linux_producer.index(trust_step),
        )
        self.assertLess(
            linux_producer.index(trust_step),
            linux_producer.index("name: Prepare dispatched release version"),
        )
        self.assertIn("scripts/package-native-sdk.sh", producer_action)
        self.assertIn("--build", producer_action)
        self.assertIn("--require-prebuilt-llama", producer_action)
        self.assertIn(
            "scripts/verify-native-sdk-package.sh",
            producer_action,
        )
        self.assertIn(
            "scripts/package-native-sdk-crate.sh",
            producer_action,
        )
        self.assertIn(
            "native SDK release asset basename collision",
            producer_action,
        )

        self.assertIn(
            "name: ${{ inputs.kotlin_artifact_name }}",
            consumer_workflow,
        )
        self.assertIn(
            "actions/download-artifact@"
            "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
            consumer_workflow,
        )
        self.assertIn(
            "scripts/restore-native-sdk-input.sh",
            consumer_script,
        )
        for forbidden in (
            "cargo ",
            "prepare-llama.sh",
            "build-llama.sh",
            "package-native-sdk.sh",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, consumer_script)
        self.assertIn("scripts/safe-extract-tar.py", restore_script)
        self.assertIn("prepare-native-sdk-input", routing)
        self.assertIn("native-sdk-artifact", routing)
        self.assertIn("restore-native-sdk-input", routing)

    def test_static_abi_artifact_is_typed_and_safely_reused(self) -> None:
        producer = (
            ROOT / ".github" / "workflows" / "static-abi-artifact.yml"
        ).read_text(encoding="utf-8")
        producer_action = self.read_action("prepare-static-abi-input")
        restore_script = (
            ROOT / "scripts" / "restore-static-abi-input.sh"
        ).read_text(encoding="utf-8")
        native_sdk_producer = (
            ROOT / ".github" / "workflows" / "native-sdk-artifact.yml"
        ).read_text(encoding="utf-8")
        routing = self.read_compute_changes()

        self.assertIn("CACHE_NAMESPACE: mesh-llm", producer)
        self.assertIn(
            "inputs.backend, inputs.target, "
            "steps.native_toolchain.outputs.epoch, hashFiles(",
            producer,
        )
        self.assertIn("'Justfile', 'just/**'", producer)
        self.assertIn(
            "uses: ./.github/actions/resolve-native-toolchain-epoch",
            producer,
        )
        self.assertIn(
            "uses: ./.github/actions/resolve-native-toolchain-epoch",
            native_sdk_producer,
        )
        self.assertIn(
            'include_tool_versions: "true"',
            native_sdk_producer,
        )
        self.assertIn("path: static-abi-artifact-output", producer)
        self.assertNotIn(
            "path: .deps/llama.cpp/build-stage-abi-static",
            producer,
        )
        self.assertIn(
            "mesh-llm-cuda-runner-sha256-"
            "8d93de6ba30173e825a16fdecf011f9c632edc6e1259df7289e491b0a05f829d",
            producer,
        )
        epoch = (
            "mesh-llm-cuda-runner-sha256-"
            "8d93de6ba30173e825a16fdecf011f9c632edc6e1259df7289e491b0a05f829d"
        )
        for consumer in (native_sdk_producer,):
            self.assertIn(epoch, consumer)
        linux_lane = (
            ROOT / ".github" / "workflows" / "ci-linux-lane.yml"
        ).read_text(encoding="utf-8")
        rust_tests = (
            ROOT / ".github" / "workflows" / "ci-rust-tests-slice.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("ci-static-abi-${{ github.run_id }}", linux_lane)
        self.assertIn("static_abi_artifact_name", rust_tests)
        self.assertIn("static_abi_artifact_name", linux_lane)
        self.assertIn(
            "uses: ./.github/actions/prepare-static-abi-input",
            producer,
        )
        prepare_index = producer.index(
            "name: Prepare patched llama.cpp checkout",
        )
        cache_index = producer.index(
            "name: Cache portable static ABI input",
        )
        self.assertLess(prepare_index, cache_index)
        for cache_input in (
            "scripts/prepare-llama.sh",
            "scripts/restore-static-abi-input.sh",
            "scripts/safe-extract-tar.py",
            "scripts/verify-checksum-sidecar.py",
            "scripts/verify-static-abi-build-stamp.py",
            ".github/actions/prepare-static-abi-input/action.yml",
        ):
            self.assertIn(cache_input, producer)
        self.assertIn("name: ${{ inputs.artifact_name }}", producer)
        self.assertIn(
            "scripts/restore-static-abi-input.sh",
            producer,
        )
        self.assertIn(
            "artifact_name: sccache-static-abi-"
            "${{ inputs.target }}-${{ inputs.backend }}-"
            "${{ github.run_attempt }}",
            producer,
        )
        self.assertIn("target/runner architecture mismatch", producer_action)
        self.assertIn("verify-static-abi-build-stamp.py", producer_action)
        self.assertIn("--patched-sha", producer_action)
        self.assertIn("Portable MeshLLM static ABI link metadata", producer_action)
        self.assertIn("retained producer-local path", producer_action)
        self.assertNotIn(
            'tar -C "$(dirname "$LLAMA_STAGE_BUILD_DIR")"',
            producer_action,
        )
        for archive in (
            "libllama-common-base.a",
            "libggml.a",
            "libggml-base.a",
            "libggml-cpu.a",
            "libvendor-hash.a",
        ):
            self.assertIn(archive, producer_action)
        self.assertIn(
            ".mesh-llm-static-abi-input.json",
            producer_action,
        )
        self.assertIn("verify-checksum-sidecar.py", producer_action)

        self.assertIn("scripts/safe-extract-tar.py", restore_script)
        self.assertIn("mesh-llm-static-abi-v3", restore_script)
        self.assertIn("toolchain_epoch", restore_script)
        self.assertIn("verify-checksum-sidecar.py", restore_script)
        self.assertIn("verify-static-abi-build-stamp.py", restore_script)
        self.assertIn("target/runner architecture mismatch", restore_script)
        self.assertNotIn("tar -x", restore_script)
        self.assertIn("prepare-static-abi-input", routing)
        self.assertIn("restore-static-abi-input", routing)
        self.assertIn("static-abi-artifact", routing)

    def test_protected_reusable_producers_own_runner_and_cache_policy(
        self,
    ) -> None:
        workflow_names = (
            "native-sdk-artifact.yml",
            "static-abi-artifact.yml",
        )
        for workflow_name in workflow_names:
            workflow = (
                ROOT / ".github" / "workflows" / workflow_name
            ).read_text(encoding="utf-8")
            inputs = workflow[: workflow.index("\njobs:\n")]
            with self.subTest(workflow=workflow_name):
                self.assertIn("runner_size:", inputs)
                self.assertIn("default: '8'", inputs)
                self.assertNotIn("runs_on:", inputs)
                self.assertNotIn("allow_depot_remote_cache:", inputs)
                self.assertNotIn("inputs.runs_on", workflow)
                self.assertNotIn(
                    "inputs.allow_depot_remote_cache",
                    workflow,
                )
                self.assertIn(
                    "runs-on: ${{ needs.runner_policy.outputs.runner }}",
                    workflow,
                )
                self.assertIn(
                    "allow_depot_remote_cache: "
                    "${{ needs.runner_policy.outputs."
                    "allow_depot_remote_cache }}",
                    workflow,
                )
                self.assertIn(
                    "depot_main_enabled: "
                    "${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
                    workflow,
                )
                self.assertIn(
                    "manual_use_depot: "
                    "${{ inputs.use_depot }}",
                    workflow,
                )
                self.assertIn("repository: ${{ github.repository }}", workflow)
                self.assertIn(
                    "head_repository: ${{ github.event.pull_request.head.repo.full_name }}",
                    workflow,
                )
                self.assertIn(
                    "head_sha: ${{ github.event.pull_request.head.sha || github.sha }}",
                    workflow,
                )
                self.assertIn("ref: ${{ github.ref }}", workflow)
                self.assertIn("depot_pr_enabled:", workflow)
                self.assertIn(
                    "pr_approved_ref: ${{ vars.DEPOT_PR_APPROVED_REF }}",
                    workflow,
                )
                self.assertIn(
                    "pr_approved_sha: ${{ vars.DEPOT_PR_APPROVED_SHA }}",
                    workflow,
                )
                self.assertIn("default) runner=", workflow)
                self.assertIn("RUNNER_ARM", workflow)

    def test_protected_reusable_runner_policy_is_fail_closed(self) -> None:
        hosted_cases = (
            (
                "pull_request",
                "refs/pull/12/merge",
                "Mesh-LLM/mesh-llm",
            ),
            (
                "pull_request_target",
                "refs/heads/main",
                "Mesh-LLM/mesh-llm",
            ),
            (
                "push",
                "refs/tags/v1.2.3",
                "Mesh-LLM/mesh-llm",
            ),
            (
                "workflow_dispatch",
                "refs/heads/feature",
                "Mesh-LLM/mesh-llm",
            ),
            (
                "push",
                "refs/heads/main",
                "attacker/mesh-llm",
            ),
        )
        for workflow_name in (
            "native-sdk-artifact.yml",
            "static-abi-artifact.yml",
        ):
            for event_name, ref, repository in hosted_cases:
                with self.subTest(
                    workflow=workflow_name,
                    event_name=event_name,
                    ref=ref,
                    repository=repository,
                ):
                    result, outputs = self.run_reusable_runner_policy(
                        workflow_name,
                        repository=repository,
                        event_name=event_name,
                        ref=ref,
                        depot_enabled="true",
                        target="x86_64-unknown-linux-gnu",
                        runner_size="16",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(outputs["runner"], "ubuntu-24.04")
                    self.assertEqual(
                        outputs["allow_depot_remote_cache"],
                        "false",
                    )

            trusted_cases = (
                (
                    "x86_64-unknown-linux-gnu",
                    "8",
                    "depot-ubuntu-24.04-8",
                ),
                (
                    "aarch64-unknown-linux-gnu",
                    "4",
                    "depot-ubuntu-24.04-arm-4",
                ),
                (
                    "x86_64-unknown-linux-gnu",
                    "default",
                    "depot-ubuntu-24.04",
                ),
            )
            for target, size, expected_runner in trusted_cases:
                with self.subTest(
                    workflow=workflow_name,
                    target=target,
                    size=size,
                ):
                    result, outputs = self.run_reusable_runner_policy(
                        workflow_name,
                        repository="Mesh-LLM/mesh-llm",
                        event_name="push",
                        ref="refs/heads/main",
                        depot_enabled="true",
                        target=target,
                        runner_size=size,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(outputs["runner"], expected_runner)
                    self.assertEqual(
                        outputs["allow_native_github_cache"],
                        "false",
                    )
                    self.assertEqual(
                        outputs["allow_depot_remote_cache"],
                        "false",
                    )

            result, outputs = self.run_reusable_runner_policy(
                workflow_name,
                repository="Mesh-LLM/mesh-llm",
                event_name="push",
                ref="refs/heads/main",
                depot_enabled="false",
                target="aarch64-unknown-linux-gnu",
                runner_size="8",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(outputs["runner"], "ubuntu-24.04-arm")
            self.assertEqual(
                outputs["allow_native_github_cache"],
                "true",
            )
            self.assertEqual(
                outputs["allow_depot_remote_cache"],
                "false",
            )

            if workflow_name == "native-sdk-artifact.yml":
                approved_pr_sha = (
                    "0123456789abcdef0123456789abcdef01234567"
                )
                result, outputs = self.run_reusable_runner_policy(
                    workflow_name,
                    repository="Mesh-LLM/mesh-llm",
                    event_name="pull_request",
                    ref="refs/pull/12/merge",
                    depot_enabled="false",
                    target="x86_64-unknown-linux-gnu",
                    runner_size="8",
                    pr_enabled="true",
                    pr_approved_ref="refs/pull/12/merge",
                    pr_approved_sha=approved_pr_sha,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(outputs["runner"], "depot-ubuntu-24.04-8")
                self.assertEqual(
                    outputs["allow_native_github_cache"],
                    "true",
                )

                result, outputs = self.run_reusable_runner_policy(
                    workflow_name,
                    repository="Mesh-LLM/mesh-llm",
                    event_name="push",
                    ref="refs/heads/main",
                    depot_enabled="true",
                    target="aarch64-apple-darwin",
                    runner_size="8",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(outputs["runner"], "macos-15")
                self.assertEqual(outputs["allow_native_github_cache"], "true")
                self.assertEqual(outputs["allow_depot_remote_cache"], "false")

            result, outputs = self.run_reusable_runner_policy(
                workflow_name,
                repository="Mesh-LLM/mesh-llm",
                event_name="push",
                ref="refs/heads/main",
                depot_enabled="true",
                target="x86_64-unknown-linux-gnu",
                runner_size="unbounded",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "runner_size must be one of",
                result.stderr,
            )
            self.assertNotIn("runner", outputs)

            result, outputs = self.run_reusable_runner_policy(
                workflow_name,
                repository="Mesh-LLM/mesh-llm",
                event_name="workflow_dispatch",
                ref="refs/heads/main",
                depot_enabled="false",
                manual_use_depot="true",
                target="x86_64-unknown-linux-gnu",
                runner_size="8",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                outputs["runner"],
                "depot-ubuntu-24.04-8",
            )
            self.assertEqual(
                outputs["allow_depot_remote_cache"],
                "false",
            )

            for event_name, ref, manual_use_depot in (
                ("workflow_dispatch", "refs/heads/main", "false"),
                ("workflow_dispatch", "refs/heads/feature", "true"),
                ("pull_request", "refs/pull/12/merge", "true"),
                ("push", "refs/heads/main", "true"),
            ):
                with self.subTest(
                    workflow=workflow_name,
                    event_name=event_name,
                    ref=ref,
                    manual_use_depot=manual_use_depot,
                ):
                    result, outputs = self.run_reusable_runner_policy(
                        workflow_name,
                        repository="Mesh-LLM/mesh-llm",
                        event_name=event_name,
                        ref=ref,
                        depot_enabled="false",
                        manual_use_depot=manual_use_depot,
                        target="x86_64-unknown-linux-gnu",
                        runner_size="8",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(outputs["runner"], "ubuntu-24.04")
                    self.assertEqual(
                        outputs["allow_depot_remote_cache"],
                        "false",
                    )

        result, outputs = self.run_reusable_runner_policy(
            "native-sdk-artifact.yml",
            repository="Mesh-LLM/mesh-llm",
            event_name="push",
            ref="refs/heads/main",
            depot_enabled="true",
            target="aarch64-apple-darwin",
            runner_size="8",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs["runner"], "macos-15")
        self.assertEqual(
            outputs["allow_depot_remote_cache"],
            "false",
        )

    def test_swift_sdk_build_is_a_shared_immutable_producer(self) -> None:
        producer = (
            ROOT / ".github" / "workflows" / "swift-sdk-artifact.yml"
        ).read_text(encoding="utf-8")
        consumer_workflow = (
            ROOT / ".github" / "workflows" / "sdk-smoke.yml"
        ).read_text(encoding="utf-8")
        consumer_script = (
            ROOT / "scripts" / "ci-swift-sdk-smoke.sh"
        ).read_text(encoding="utf-8")
        routing = self.read_compute_changes()

        self.assertIn("type: string", producer)
        self.assertIn("host-only|full", producer)
        self.assertIn(
            "sdk/swift/scripts/build-host-macos-xcframework.sh",
            producer,
        )
        self.assertIn("sdk/swift/scripts/build-xcframework.sh", producer)
        self.assertIn("max-parallel: ${{ inputs.max_parallel }}", producer)
        self.assertEqual(producer.count("- aarch64-apple-ios\n"), 1)
        self.assertIn(
            'build-xcframework.sh --target "${{ matrix.target }}"',
            producer,
        )
        self.assertIn(
            "name: swift-sdk-target-${{ matrix.target }}-"
            "${{ github.run_attempt }}",
            producer,
        )
        self.assertIn(
            "pattern: swift-sdk-target-*",
            producer,
        )
        self.assertNotIn(
            "pattern: swift-sdk-target-*-${{ github.run_attempt }}",
            producer,
        )
        self.assertIn(
            "build-xcframework.sh --assemble-from dist/swift-targets",
            producer,
        )
        self.assertIn(
            "scripts/verify-swift-release-artifact.sh",
            producer,
        )
        self.assertIn(
            "scripts/verify-swift-xcframework.py",
            (
                ROOT / "scripts" / "verify-swift-release-artifact.sh"
            ).read_text(encoding="utf-8"),
        )
        self.assertIn("persist-credentials: false", producer)
        self.assertIn("actions/upload-artifact@", producer)
        self.assertIn(
            "runs-on: ${{ needs.runner_policy.outputs.runner_macos }}",
            producer,
        )
        self.assertIn("depot-macos-15", self.read_action("select-ci-runners"))
        self.assertIn("RUSTC_WRAPPER: sccache", producer)
        self.assertIn("SCCACHE_GHA_RW_MODE:", producer)
        self.assertIn(
            "uses: ./.github/actions/configure-sccache-gha",
            producer,
        )
        self.assertIn("shared-key: swift-sdk", producer)
        self.assertIn(
            "save-if: ${{ github.event_name == 'push' "
            "&& github.ref == 'refs/heads/main' }}",
            producer,
        )
        self.assertNotIn("macos_runner:", producer)
        self.assertIn(
            "name: generated-swift-binding-${{ inputs.artifact_name }}",
            producer,
        )
        self.assertIn(
            "git diff --exit-code -- \"$generated_binding\"",
            producer,
        )
        self.assertIn(
            "name: Verify committed Swift binding source is current\n"
            "        if: ${{ !inputs.prepare_release_version }}",
            producer,
        )
        self.assertNotIn(
            "!inputs.prepare_release_version && (github.ref ==",
            producer,
        )
        self.assertIn("EVENT_NAME: ${{ github.event_name }}", producer)
        self.assertIn(
            "release source preparation requires workflow_dispatch",
            producer,
        )
        self.assertIn(
            "uses: ./.github/actions/capture-sccache-stats",
            producer,
        )
        self.assertIn("if: ${{ !cancelled() }}", producer)
        self.assertIn(
            "artifact_name: sccache-swift-sdk-"
            "${{ inputs.mode }}-${{ github.run_attempt }}",
            producer,
        )

        targets = (
            "aarch64-apple-ios",
            "aarch64-apple-ios-sim",
            "x86_64-apple-ios",
            "aarch64-apple-ios-macabi",
            "x86_64-apple-ios-macabi",
            "aarch64-apple-darwin",
            "x86_64-apple-darwin",
        )
        workflow = yaml.safe_load(producer)
        target_job = workflow["jobs"]["swift_sdk_target"]
        self.assertEqual(
            target_job["strategy"]["matrix"]["target"],
            list(targets),
        )
        upload_steps = [
            step
            for step in target_job["steps"]
            if step.get("name") == "Upload immutable Swift target"
        ]
        self.assertEqual(len(upload_steps), 1)
        self.assertEqual(
            upload_steps[0]["with"]["name"],
            "swift-sdk-target-${{ matrix.target }}-${{ github.run_attempt }}",
        )
        assembly_job = workflow["jobs"]["swift_sdk_artifact"]
        download_steps = [
            step
            for step in assembly_job["steps"]
            if step.get("name") == "Download immutable Swift targets"
        ]
        self.assertEqual(len(download_steps), 1)
        self.assertEqual(
            download_steps[0]["with"]["pattern"],
            "swift-sdk-target-*",
        )

        self.assertIn(
            "name: ${{ inputs.swift_artifact_name }}",
            consumer_workflow,
        )
        self.assertIn(
            "name: generated-swift-binding-"
            "${{ inputs.swift_artifact_name }}",
            consumer_workflow,
        )
        self.assertIn(
            "actions/download-artifact@"
            "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
            consumer_workflow,
        )
        self.assertIn("persist-credentials: false", consumer_workflow)
        self.assertIn(
            "if: ${{ inputs.sdk_kind == 'rust' }}",
            consumer_workflow,
        )
        self.assertNotIn("pnpm/action-setup@", consumer_workflow)
        self.assertNotIn("actions/setup-node@", consumer_workflow)
        self.assertIn(
            'LEGACY_PNPM_STORE="$(pnpm store path --silent)"',
            consumer_script,
        )
        self.assertIn('mkdir -p "$LEGACY_PNPM_STORE"', consumer_script)

        for forbidden in (
            "cargo ",
            "build-llama.sh",
            "package-native-sdk.sh",
            "build-xcframework.sh",
            "build-host-macos-xcframework.sh",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, consumer_script)
        self.assertIn(
            'scripts/safe-extract-zip.py "$SWIFT_INPUT_ARCHIVE"',
            consumer_script,
        )
        self.assertIn(
            'install -m 0644 "$SWIFT_INPUT_BINDING" '
            '"$SWIFT_TRACKED_BINDING"',
            consumer_script,
        )
        self.assertIn(
            '[[ -L "$SWIFT_GENERATED_DIR" ]]',
            consumer_script,
        )
        self.assertIn(
            '[[ -e "$SWIFT_GENERATED_DIR" '
            '&& ! -d "$SWIFT_GENERATED_DIR" ]]',
            consumer_script,
        )
        mkdir_index = consumer_script.index(
            'mkdir -p "$SWIFT_GENERATED_DIR"',
        )
        move_index = consumer_script.index(
            'mv "$SWIFT_EXTRACT_DIR/MeshLLMFFI.xcframework" '
            '"$SWIFT_XCFRAMEWORK"',
        )
        self.assertLess(mkdir_index, move_index)
        self.assertIn("safe-extract-(tar|zip)", routing)
        self.assertIn("verify-swift-xcframework", routing)
        for workflow in (
            "native-sdk-artifact",
            "sdk-smoke",
            "static-abi-artifact",
            "swift-sdk-artifact",
        ):
            self.assertIn(workflow, routing)

    def test_swift_sdk_cache_is_mode_independent_and_target_specific(
        self,
    ) -> None:
        producer = (
            ROOT / ".github" / "workflows" / "swift-sdk-artifact.yml"
        ).read_text(encoding="utf-8")
        host_builder = (
            ROOT / "sdk" / "swift" / "scripts"
            / "build-host-macos-xcframework.sh"
        ).read_text(encoding="utf-8")

        exact_key = "format('mesh-llm-swift-sdk-target-{0}-{1}-{2}-{3}-{4}'"
        self.assertEqual(producer.count(exact_key), 2)
        self.assertIn(
            "shared-key: swift-sdk-${{ matrix.target }}",
            producer,
        )
        self.assertIn(
            "shared-key: ${{ format('swift-sdk-{0}', runner.arch == 'ARM64' "
            "&& 'aarch64-apple-darwin' || 'x86_64-apple-darwin') }}",
            producer,
        )
        self.assertIn(
            "path: ${{ format('.deps/llama-build/build-stage-abi-{0}-metal'",
            producer,
        )
        self.assertNotIn("runner.arch, inputs.mode, hashFiles(", producer)
        self.assertIn(
            "uses: ./.github/actions/resolve-native-toolchain-epoch",
            producer,
        )
        self.assertIn('include_tool_versions: "true"', producer)
        self.assertNotIn("SWIFT_NATIVE_XCODE_CACHE_EPOCH", producer)
        self.assertIn("Trusted main can", producer)
        self.assertNotIn("build-stage-abi-host-metal", producer)
        self.assertIn(
            ".deps/llama-build/build-stage-abi-$RUST_TARGET-metal",
            host_builder,
        )
        self.assertIn("-DCMAKE_OSX_SYSROOT=macosx", host_builder)
        self.assertIn('-DCMAKE_OSX_ARCHITECTURES="$CMAKE_ARCH"', host_builder)
        self.assertIn(
            '-DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET"',
            host_builder,
        )

    def test_runtime_action_never_builds_the_host(self) -> None:
        action = self.read_action("prepare-native-runtime-input")

        self.assertIn('scripts/package-native-runtime.sh "${args[@]}"', action)
        self.assertIn("scripts/verify-native-runtime-package.sh", action)
        self.assertNotIn("build-host.sh", action)
        self.assertNotIn("build-release.sh", action)


if __name__ == "__main__":
    unittest.main()
