from __future__ import annotations

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def job_block(workflow: str, job_name: str, next_job_name: str) -> str:
    start = workflow.index(f"  {job_name}:")
    end = workflow.index(f"  {next_job_name}:", start)
    return workflow[start:end]


class ReleaseWorkflowArtifactTests(unittest.TestCase):
    def test_release_ui_version_preparation_handles_container_ownership(self) -> None:
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/ci-ui-artifact-slice.yml").read_text()
        )
        step = next(
            step for step in workflow["jobs"]["ui_artifact"]["steps"]
            if step.get("name") == "Prepare release UI version"
        )
        for source in ("matching", "mismatching", "malformed"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                repo = root / "container checkout"
                repo.mkdir()
                home = root / "home"
                home.mkdir()
                env = {
                    key: value for key, value in os.environ.items()
                    if not key.startswith("GIT_")
                }
                env.update(
                    HOME=str(home), XDG_CONFIG_HOME=str(home),
                    GIT_CONFIG_GLOBAL=str(home / ".gitconfig"),
                    GIT_CONFIG_NOSYSTEM="1",
                )

                def git(*args: str) -> subprocess.CompletedProcess[str]:
                    return subprocess.run(
                        ["git", *args], cwd=repo, env=env,
                        text=True, capture_output=True,
                    )

                self.assertEqual(git("init", "--quiet").returncode, 0)
                commit = git(
                    "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "fixture",
                )
                self.assertEqual(commit.returncode, 0, commit.stderr)
                sha = git("rev-parse", "HEAD").stdout.strip()
                scripts = repo / "scripts"
                scripts.mkdir()
                version = scripts / "release-version.sh"
                version.write_text(
                    '#!/usr/bin/env bash\nset -euo pipefail\n'
                    'git rev-parse HEAD > version-source\nprintf "%s\\n" "$1" > version-tag\n'
                )
                version.chmod(0o755)
                env.update(
                    GIT_TEST_ASSUME_DIFFERENT_OWNER="1",
                    GITHUB_WORKSPACE=str(repo), GITHUB_ENV=str(root / "github-env"),
                    UI_SOURCE_SHA=sha if source == "matching" else (
                        "0" * 40 if source == "mismatching" else "invalid"
                    ),
                    RELEASE_TAG="v0.76.0-rc9",
                )
                untrusted = git("rev-parse", "HEAD")
                self.assertNotEqual(untrusted.returncode, 0)
                self.assertIn("dubious ownership", untrusted.stderr)
                result = subprocess.run(
                    ["bash", "-c", step["run"]], cwd=repo, env=env,
                    text=True, capture_output=True,
                )
                if source != "matching":
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((repo / "version-tag").exists())
                    self.assertFalse((root / "github-env").exists())
                    continue
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((repo / "version-source").read_text().strip(), sha)
                self.assertEqual((repo / "version-tag").read_text().strip(), env["RELEASE_TAG"])
                self.assertEqual(
                    (root / "github-env").read_text(), "VITE_MESH_LLM_DEBUG_UI=false\n"
                )
                self.assertEqual(
                    git("config", "--global", "--get-all", "safe.directory").stdout.splitlines(),
                    [str(repo)],
                )

    def test_release_ui_is_one_verified_producer_for_hosts_and_sdks(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        producer = job_block(workflow, "release_ui", "build")
        self.assertIn("uses: ./.github/workflows/ci-ui-artifact-slice.yml", producer)
        self.assertIn("source_sha: ${{ needs.metadata.outputs.source_sha }}", producer)
        self.assertIn("release_tag: ${{ needs.metadata.outputs.tag }}", producer)
        artifact = "prepared-release-ui-${{ needs.metadata.outputs.tag }}-${{ needs.metadata.outputs.source_sha }}"
        self.assertIn(f"artifact_name: {artifact}", producer)
        ui_workflow = (ROOT / ".github/workflows/ci-ui-artifact-slice.yml").read_text()
        self.assertIn('echo "VITE_MESH_LLM_DEBUG_UI=false" >> "$GITHUB_ENV"', ui_workflow)
        self.assertLess(ui_workflow.index("Prepare release UI version"), ui_workflow.index("Install UI dependencies"))
        self.assertIn("python3 scripts/ui-distribution.py stamp", ui_workflow)
        restore = (ROOT / ".github/actions/restore-release-ui/action.yml").read_text()
        self.assertIn('"$python_bin" scripts/ui-distribution.py verify', restore)
        for name, next_name in (("build", "compose_cpu_products"),
                                ("build_linux_arm64", "compose_linux_arm64_cpu"),
                                ("windows_host_input", "compose_windows_gpu")):
            host = job_block(workflow, name, next_name)
            with self.subTest(host=name):
                self.assertIn("needs: [metadata, release_ui]", host)
                self.assertIn(f"artifact_name: {artifact}", host)
                self.assertIn('skip_ui: "true"', host)
                self.assertNotIn("pnpm/action-setup", host)
                self.assertLess(host.index("restore-release-ui"), host.index("Build and attest"))
        swift = job_block(workflow, "build_swift_sdk_artifact", "build_linux_arm64")
        self.assertIn("needs: [metadata, release_ui]", swift)
        self.assertIn(f"ui_artifact_name: {artifact}", swift)
        swift_workflow = (ROOT / ".github/workflows/swift-sdk-artifact.yml").read_text()
        self.assertIn("if: ${{ inputs.ui_artifact_name == '' }}", swift_workflow)
        self.assertIn("scripts/package-sdk-console-assets.sh --sdk swift --skip-build", swift_workflow)
        publish = job_block(workflow, "publish", "dispatch_packaging_release")
        self.assertIn("      - release_ui", publish)
        self.assertIn("--sdk all --skip-build", publish)
        self.assertNotIn("pnpm/action-setup", publish)
        self.assertFalse(artifact.startswith("release-"), "UI must not match published release-* assets")

    def test_linux_release_composers_use_cpu_tools_and_keep_readiness(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        cpu = "ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:8d93de6ba30173e825a16fdecf011f9c632edc6e1259df7289e491b0a05f829d"
        for name, next_name in (("compose_linux_aarch64_cuda", "compose_linux_cuda"),
                                ("compose_linux_cuda", "compose_linux_rocm"),
                                ("compose_linux_rocm", "compose_linux_vulkan"),
                                ("compose_linux_vulkan", "windows_host_input")):
            compose = job_block(workflow, name, next_name)
            with self.subTest(composer=name):
                self.assertIn(f"image: {cpu}", compose)
                self.assertIn("verify-runner-image public cpu", compose)
                self.assertIn('readiness_smoke: "true"', compose)
                self.assertIn("MESH_RELEASE_HOST_PRESTAMPED", compose)
                self.assertNotIn("prepare-native-runtime-input", compose)
        cuda = job_block(workflow, "build_native_runtime_linux_x86_64_cuda", "build_native_runtime_linux_x86_64_rocm")
        self.assertIn("vars.USE_SELF_HOSTED == 'true'", cuda)
        self.assertIn('"amd64","gpu-nvidia"', cuda)
        self.assertIn("image: ${{ matrix.runner_image }}", cuda)
        self.assertIn("verify-runner-image public cuda", cuda)

    def test_release_entrypoint_rejects_untrusted_refs(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        metadata = job_block(workflow, "metadata", "build")

        manual_guard = metadata.index("Require the trusted release ref")
        checkout = metadata.index("uses: actions/checkout@")
        selector = metadata.index(
            "uses: ./.github/actions/select-ci-runners",
        )
        self.assertLess(manual_guard, checkout)
        self.assertLess(checkout, selector)
        self.assertIn(
            '"$GITHUB_EVENT_NAME" == "workflow_dispatch"',
            metadata,
        )
        self.assertIn(
            '"$GITHUB_REF" != "refs/heads/main"',
            metadata,
        )
        self.assertIn(
            'git merge-base --is-ancestor "$GITHUB_SHA" '
            "refs/remotes/origin/main",
            metadata,
        )
        self.assertIn("Reject an existing manual release tag", metadata)
        self.assertIn(
            "Manual release tag already exists and is immutable",
            metadata,
        )

    def test_release_source_version_is_synchronized_before_builds(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        metadata = job_block(workflow, "metadata", "build")
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )

        source = metadata.index("name: Prepare canonical release source")
        selector = metadata.index("uses: ./.github/actions/select-ci-runners")
        self.assertLess(source, selector)
        self.assertIn("source_sha: ${{ steps.source.outputs.sha }}", metadata)
        self.assertIn(
            "release_notes_base: "
            "${{ steps.source.outputs.release_notes_base }}",
            metadata,
        )
        self.assertIn('scripts/release-version.sh "$RELEASE_TAG"', metadata)
        self.assertIn("Canary release: leaving main unchanged", metadata)
        self.assertIn(
            'git push "$release_remote" "$source_sha:refs/heads/main"',
            metadata,
        )
        self.assertIn(
            "does not contain its complete version update",
            metadata,
        )
        self.assertIn(
            "ref: ${{ needs.metadata.outputs.source_sha }}",
            publish,
        )
        self.assertIn("generate_release_notes: true", publish)
        self.assertIn(
            "previous_tag: "
            "${{ needs.metadata.outputs.release_notes_base }}",
            publish,
        )
        self.assertIn(
            'python3 scripts/select-release-notes-base.py "$RELEASE_TAG"',
            metadata,
        )
        manual_version_update = metadata.index(
            'else\n            scripts/release-version.sh "$RELEASE_TAG"',
        )
        format_check = metadata.index(
            "cargo fmt --all -- --check",
            manual_version_update,
        )
        whitespace_check = metadata.index("git diff --check", format_check)
        stage_release_source = metadata.index("git add --update", whitespace_check)
        push_release_source = metadata.index(
            'git push "$release_remote" "$source_sha:refs/heads/main"',
        )
        self.assertLess(manual_version_update, format_check)
        self.assertLess(format_check, whitespace_check)
        self.assertLess(whitespace_check, stage_release_source)
        self.assertLess(stage_release_source, push_release_source)

    def test_release_depot_policy_is_main_ref_only_and_selected_once(
        self,
    ) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        metadata = job_block(workflow, "metadata", "build")

        self.assertIn("use_depot:", workflow[: workflow.index("\njobs:\n")])
        self.assertIn(
            "uses: ./.github/actions/select-ci-runners",
            metadata,
        )
        self.assertIn("ref: ${{ github.ref }}", metadata)
        self.assertIn(
            "depot_main_enabled: ${{ vars.DEPOT_RUNNERS_ENABLED == 'true' }}",
            metadata,
        )
        self.assertIn(
            "manual_use_depot: ${{ inputs.use_depot == true }}",
            metadata,
        )
        self.assertEqual(
            workflow.count("uses: ./.github/actions/select-ci-runners"),
            1,
        )

    def test_release_routes_only_initial_non_secret_linux_lanes(
        self,
    ) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        host = job_block(workflow, "build", "compose_cpu_products")
        sdk_runtime = job_block(
            workflow,
            "build_native_sdk_runtime",
            "build_native_runtime",
        )
        native_runtime = job_block(
            workflow,
            "build_native_runtime",
            "build_native_runtime_linux_aarch64_cuda",
        )
        arm_host = job_block(
            workflow,
            "build_linux_arm64",
            "compose_linux_arm64_cpu",
        )
        arm_compose = job_block(
            workflow,
            "compose_linux_arm64_cpu",
            "smoke_linux_arm64_artifact",
        )
        arm_smoke = job_block(
            workflow,
            "smoke_linux_arm64_artifact",
            "compose_linux_aarch64_cuda",
        )
        rocm = job_block(
            workflow,
            "build_native_runtime_linux_x86_64_rocm",
            "build_native_runtime_linux_x86_64_vulkan",
        )
        vulkan = job_block(
            workflow,
            "build_native_runtime_linux_x86_64_vulkan",
            "build_swift_sdk_artifact",
        )
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )

        self.assertIn("runs-on: ${{ matrix.os }}", host)
        self.assertIn("RELEASE_ATTESTATION_SIGNING_KEY", host)
        self.assertIn("runner_size: '8'", sdk_runtime)
        self.assertNotIn("runs_on:", sdk_runtime)
        self.assertNotIn("allow_depot_remote_cache:", sdk_runtime)
        self.assertNotIn("needs.metadata.outputs.runner", sdk_runtime)
        for producer in (native_runtime,):
            self.assertIn(
                "matrix.target == 'x86_64-unknown-linux-gnu'",
                producer,
            )
            self.assertIn(
                "needs.metadata.outputs.runner_8",
                producer,
            )
            self.assertIn(
                "needs.metadata.outputs.runner_arm_8",
                producer,
            )
        for producer in (rocm, vulkan):
            self.assertIn(
                "runs-on: ${{ needs.metadata.outputs.runner_16 }}",
                producer,
            )
            self.assertIn(
                "allow_depot_remote_cache: "
                "${{ needs.metadata.outputs.allow_depot_remote_cache }}",
                producer,
            )
        self.assertIn("runs-on: ubuntu-24.04", publish)
        self.assertNotIn("needs.metadata.outputs.runner", publish)
        self.assertIn("RELEASE_ATTESTATION_SIGNING_KEY", arm_host)
        self.assertIn("runs-on: ubuntu-24.04-arm", arm_host)
        self.assertNotIn("needs.metadata.outputs.runner_arm", arm_host)
        self.assertIn(
            "runs-on: ${{ needs.metadata.outputs.runner_arm_4 }}",
            arm_compose,
        )
        self.assertIn(
            "runs-on: ${{ needs.metadata.outputs.runner_arm }}",
            arm_smoke,
        )
        self.assertNotIn("USE_SELF_HOSTED", arm_smoke)

    def test_native_runtime_cache_policy_tracks_effective_runner(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        native_runtime = job_block(
            workflow,
            "build_native_runtime",
            "build_native_runtime_linux_aarch64_cuda",
        )
        rocm = job_block(
            workflow,
            "build_native_runtime_linux_x86_64_rocm",
            "build_native_runtime_linux_x86_64_vulkan",
        )
        vulkan = job_block(
            workflow,
            "build_native_runtime_linux_x86_64_vulkan",
            "build_swift_sdk_artifact",
        )
        self.assertIn(
            "allow_native_github_cache: ${{ ((matrix.target == 'x86_64-unknown-linux-gnu' && startsWith(needs.metadata.outputs.runner_8, 'depot-')) || (matrix.target == 'aarch64-unknown-linux-gnu' && startsWith(needs.metadata.outputs.runner_arm_8, 'depot-'))) && 'false' || 'true' }}",
            native_runtime,
        )
        for target, runner_8, runner_arm_8, expected in (
            ("x86_64-unknown-linux-gnu", "depot-ubuntu-24.04-8", "ubuntu-24.04-arm", "false"),
            ("x86_64-unknown-linux-gnu", "ubuntu-24.04", "ubuntu-24.04-arm", "true"),
            ("aarch64-unknown-linux-gnu", "ubuntu-24.04", "depot-ubuntu-24.04-arm-8", "false"),
            ("aarch64-apple-darwin", "depot-ubuntu-24.04-8", "depot-ubuntu-24.04-arm-8", "true"),
        ):
            depot = (
                target == "x86_64-unknown-linux-gnu" and runner_8.startswith("depot-")
            ) or (
                target == "aarch64-unknown-linux-gnu"
                and runner_arm_8.startswith("depot-")
            )
            self.assertEqual("false" if depot else "true", expected)

        for runner_16, expected in (
            ("depot-ubuntu-24.04-16", "false"),
            ("ubuntu-24.04", "true"),
        ):
            self.assertEqual(
                "false" if runner_16.startswith("depot-") else "true",
                expected,
            )

        effective_runner_16_cache_gate = (
            "if: ${{ !startsWith(needs.metadata.outputs.runner_16, 'depot-') }}"
        )
        provider_level_cache_gate = (
            "if: ${{ needs.metadata.outputs.allow_native_github_cache == 'true' }}"
        )
        for producer in (rocm, vulkan):
            self.assertIn(effective_runner_16_cache_gate, producer)
            self.assertNotIn(provider_level_cache_gate, producer)

    def test_inference_smoke_consumes_composed_product(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

        self.assertEqual(
            workflow.count("ci-release-linux-inference-product"),
            2,
        )
        self.assertNotIn("release-linux-inference-binary", workflow)
        self.assertIn(
            "uses: ./.github/actions/compose-product-input",
            workflow,
        )
        self.assertIn("output_dir: product-input", workflow)
        self.assertIn(
            "path: ${{ steps.compose.outputs.archive_path }}",
            workflow,
        )
        inference = job_block(
            workflow,
            "inference_smoke_tests",
            "build_native_sdk_runtime",
        )
        self.assertNotIn("runs_on:", inference)
        smoke = (
            ROOT / ".github" / "workflows" / "smoke.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("default: 'ubuntu-24.04'", smoke)
        self.assertIn("|| 'ubuntu-24.04'", smoke)

    def test_swift_release_reuses_full_typed_producer(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        reusable = (
            ROOT / ".github" / "workflows" / "swift-sdk-artifact.yml"
        ).read_text(encoding="utf-8")
        producer = job_block(
            workflow,
            "build_swift_sdk_artifact",
            "build_linux_arm64",
        )

        self.assertIn(
            "uses: ./.github/workflows/swift-sdk-artifact.yml",
            producer,
        )
        self.assertIn("mode: full", producer)
        self.assertIn("artifact_name: release-swift-sdk", producer)
        self.assertIn("max_parallel: 4", producer)
        self.assertIn("timeout_minutes: 180", producer)
        self.assertNotIn("macos_runner:", producer)
        self.assertIn(
            "release_tag: ${{ needs.metadata.outputs.tag }}",
            producer,
        )
        self.assertIn(
            "prepare_release_version: "
            "${{ github.event_name == 'workflow_dispatch' }}",
            producer,
        )
        self.assertNotIn("build-xcframework.sh", producer)
        self.assertNotIn("cargo ", producer)
        self.assertIn("name: swift-package-manifest", reusable)
        self.assertIn(
            "name: generated-swift-binding-${{ inputs.artifact_name }}",
            reusable,
        )
        self.assertIn("name: ${{ inputs.artifact_name }}", reusable)
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )
        self.assertIn(
            "name: generated-swift-binding-release-swift-sdk",
            publish,
        )
        self.assertIn(
            'install -m 0644 "$generated_binding" "$tracked_binding"',
            publish,
        )

    def test_release_permissions_are_least_privilege(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        header = workflow[: workflow.index("\njobs:\n")]
        metadata = job_block(
            workflow,
            "metadata",
            "build",
        )
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )

        self.assertIn(
            "permissions:\n  contents: read\n  packages: read",
            header,
        )
        self.assertNotIn("contents: write", header)
        self.assertNotIn("packages: write", header)
        self.assertIn(
            "    permissions:\n      contents: write",
            metadata,
        )
        self.assertNotIn("packages: write", metadata)
        self.assertIn(
            "    permissions:\n      contents: write",
            publish,
        )
        self.assertNotIn("packages: write", publish)

    def test_publish_fan_in_stops_when_release_is_cancelled(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )

        self.assertIn("if: ${{ !cancelled()", publish)
        self.assertNotIn("always()", publish)

    def test_prereleases_never_dispatch_downstream_publication(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        metadata = job_block(
            workflow,
            "metadata",
            "build",
        )
        dispatch = job_block(
            workflow,
            "dispatch_packaging_release",
            "publish_crates_preflight",
        )

        self.assertIn('if [[ "$version" == *-* ]]', metadata)
        self.assertIn("prerelease=true", metadata)
        self.assertIn(
            "needs.metadata.outputs.prerelease != 'true'",
            dispatch,
        )
        self.assertIn(
            "needs.metadata.outputs.skip_gpu_bundles != 'true'",
            dispatch,
        )
        self.assertIn("dry_run: false", dispatch)
        self.assertIn("publish_images: true", dispatch)
        self.assertIn("publish_release_assets: true", dispatch)
        self.assertIn("publish_npm: true", dispatch)

    def test_release_assets_and_manual_tags_are_immutable(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )

        self.assertIn("Release tag already exists and cannot be reused", publish)
        self.assertNotIn("reusing it", publish)
        self.assertIn("overwrite_files: false", publish)
        self.assertIn("persist-credentials: false", publish)
        self.assertNotIn("persist-credentials: true", publish)
        self.assertIn(
            'git push "$release_remote" "refs/tags/$RELEASE_TAG"',
            publish,
        )
        self.assertNotIn(
            'git push origin "refs/tags/$RELEASE_TAG"',
            publish,
        )

    def test_release_push_token_is_isolated_to_the_push_step(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )
        generate_start = publish.index(
            "- name: Generate native runtime release manifest",
        )
        prepare_start = publish.index(
            "- name: Prepare dispatched release tag",
        )
        push_start = publish.index(
            "- name: Push dispatched release tag",
        )
        release_start = publish.index(
            "- name: Publish GitHub release",
        )
        generate = publish[generate_start:prepare_start]
        prepare = publish[prepare_start:push_start]
        push = publish[push_start:release_start]

        token_binding = "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}"
        self.assertNotIn(token_binding, generate)
        self.assertNotIn(token_binding, prepare)
        self.assertIn(token_binding, push)
        self.assertEqual(publish.count(token_binding), 1)
        self.assertNotIn("release_remote=", prepare)
        self.assertNotIn("git push", prepare)
        self.assertIn(
            'git push "$release_remote" "refs/tags/$RELEASE_TAG"',
            push,
        )

    def test_arm64_smoke_requires_integrity_and_safe_extraction(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        smoke = job_block(
            workflow,
            "smoke_linux_arm64_artifact",
            "compose_linux_aarch64_cuda",
        )

        self.assertIn("scripts/verify-checksum-sidecar.py", smoke)
        self.assertIn("scripts/safe-extract-tar.py", smoke)
        self.assertIn(
            'scripts/ci-hf-xet-portability-smoke.sh "$binary"',
            smoke,
        )
        self.assertNotIn("tar -xzf", smoke)
        self.assertNotIn("command -v sha256sum", smoke)

    def test_native_sdk_assets_are_staged_flat_for_publishing(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        reusable = (
            ROOT / ".github" / "workflows" / "native-sdk-artifact.yml"
        ).read_text(encoding="utf-8")
        producer_action = (
            ROOT / ".github" / "actions"
            / "prepare-native-sdk-input" / "action.yml"
        ).read_text(encoding="utf-8")
        caller = job_block(
            workflow,
            "build_native_sdk_runtime",
            "build_native_runtime",
        )
        publish = job_block(
            workflow,
            "publish",
            "dispatch_packaging_release",
        )

        self.assertIn(
            "uses: ./.github/workflows/native-sdk-artifact.yml",
            caller,
        )
        self.assertIn("profile: release", caller)
        self.assertIn("include_runtime_crate: true", caller)
        self.assertIn(
            "static_abi_artifact_name: "
            "ci-release-native-sdk-static-abi-${{ matrix.artifact_suffix }}",
            caller,
        )
        self.assertIn(
            "produce_static_abi: "
            "${{ endsWith(matrix.target, '-unknown-linux-gnu') }}",
            caller,
        )
        self.assertIn(
            "artifact_name: "
            "release-native-sdk-${{ matrix.artifact_suffix }}",
            caller,
        )
        self.assertIn("runner_size: '8'", caller)
        self.assertNotIn("runs_on:", caller)
        self.assertNotIn("allow_depot_remote_cache:", caller)
        self.assertNotIn("scripts/package-native-sdk.sh", caller)
        self.assertIn(
            "uses: ./.github/actions/prepare-native-sdk-input",
            reusable,
        )
        self.assertIn(
            "uses: ./.github/workflows/static-abi-artifact.yml",
            reusable,
        )
        self.assertIn(
            "scripts/restore-static-abi-input.sh",
            reusable,
        )
        self.assertIn("name: ${{ inputs.artifact_name }}", reusable)
        self.assertIn(
            "path: ${{ steps.native-sdk.outputs.upload_path }}",
            reusable,
        )
        self.assertIn(
            "scripts/package-native-sdk-crate.sh",
            producer_action,
        )
        self.assertIn(
            "native SDK release asset basename collision",
            producer_action,
        )
        self.assertIn('upload_sources=("$archive_path" "$checksum_path")', producer_action)
        self.assertIn('upload_sources+=("${runtime_crates[0]}")', producer_action)
        self.assertIn("files: release-artifacts/*", publish)

    def test_windows_host_publishes_prebuilt_attestation_verifier(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        producer = job_block(
            workflow,
            "windows_host_input",
            "compose_windows_gpu",
        )

        self.assertIn(
            "uses: ./.github/actions/prepare-windows-host-input",
            producer,
        )
        self.assertIn("profile: release", producer)
        self.assertIn(
            "attestation_signing_key_file:",
            producer,
        )
        self.assertIn(
            "attestation_public_key_file:",
            producer,
        )
        self.assertIn("path: host-input/*", producer)
        self.assertNotIn("prepare-native-runtime-input", producer)
        self.assertNotIn("compose-product-input", producer)

    def test_windows_composers_use_shared_verified_product_action(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        jobs = (
            ("compose_windows_gpu", "compose_windows_cpu"),
            ("compose_windows_cpu", "build_native_runtime_windows_cpu"),
        )

        for job_name, next_job_name in jobs:
            with self.subTest(job=job_name):
                job = job_block(workflow, job_name, next_job_name)
                composition = job.index(
                    "uses: ./.github/actions/compose-product-input",
                )
                packaging = job.index(
                    "- name: Package verified Windows",
                )
                self.assertLess(composition, packaging)
                self.assertIn(
                    "binary_name: mesh-llm.exe",
                    job,
                )
                self.assertIn(
                    "attestation_verifier: host-input/release-attestation-verifier.exe",
                    job,
                )
                self.assertIn(
                    "version: ${{ needs.metadata.outputs.tag }}",
                    job,
                )
                expected_backend = (
                    "backend: ${{ matrix.backend }}"
                    if job_name == "compose_windows_gpu"
                    else "backend: cpu"
                )
                self.assertIn(expected_backend, job)
                self.assertIn('readiness_smoke: "true"', job)
                self.assertIn(
                    "MESH_LLM_PRECOMPOSED_PRODUCT_DIR: ${{ steps.compose.outputs.product_dir }}",
                    job,
                )
                self.assertIn(
                    'MESH_RELEASE_ATTESTATION_PREVERIFIED: "1"',
                    job,
                )
                self.assertNotIn("Verify immutable runtime archive", job)
                self.assertNotIn("tar -xzf", job)
                self.assertNotIn("ci-client-readiness-smoke.sh", job)
                self.assertNotIn("cargo run", job)
                self.assertNotIn("dtolnay/rust-toolchain", job)
                self.assertNotIn("sccache-action", job)

    def test_windows_cuda12_label_rejects_other_toolkit_majors(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        producer = job_block(
            workflow,
            "build_native_runtime_windows_gpu",
            "publish",
        )

        validation = producer.index("- name: Validate CUDA 12 artifact contract")
        installation = producer.index("- name: Install CUDA toolkit")
        self.assertLess(validation, installation)
        self.assertIn("$cudaMajor -ne '12'", producer)
        self.assertIn(
            "release-native-runtime-windows-x86_64-cuda12",
            producer,
        )

    def test_cuda_runtime_producers_validate_matrix_version_against_compiler(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        aarch64 = job_block(
            workflow,
            "build_native_runtime_linux_aarch64_cuda",
            "build_native_runtime_linux_x86_64_cuda",
        )
        x86_64 = job_block(
            workflow,
            "build_native_runtime_linux_x86_64_cuda",
            "build_native_runtime_linux_x86_64_rocm",
        )
        windows = job_block(
            workflow,
            "build_native_runtime_windows_gpu",
            "publish",
        )

        for producer in (aarch64, x86_64):
            self.assertIn(
                "MESH_CUDA_VERSION: ${{ matrix.cuda_version }}",
                producer,
            )
            self.assertIn(
                "MESH_LLM_CUDA_TOOLKIT_MAJOR: ${{ matrix.cuda_major }}",
                producer,
            )
        self.assertIn(
            "MESH_CUDA_VERSION: ${{ vars.CUDA_VERSION || '12.9.2' }}",
            windows,
        )
        self.assertIn("MESH_LLM_CUDA_TOOLKIT_MAJOR: '12'", windows)

    def test_cuda12_release_runtime_includes_pascal_sm61(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        linux_cuda = job_block(
            workflow,
            "build_native_runtime_linux_x86_64_cuda",
            "build_native_runtime_linux_x86_64_rocm",
        )
        windows_cuda = job_block(
            workflow,
            "build_native_runtime_windows_gpu",
            "publish",
        )

        self.assertIn(
            "cuda_architectures: '61;75;80;86;87;89;90'",
            linux_cuda,
        )
        self.assertIn(
            "cuda_architectures_cache: '61_75_80_86_87_89_90'",
            linux_cuda,
        )
        self.assertIn(
            "cuda_architectures: '75;80;86;87;89;90;100;103;120;121'",
            linux_cuda,
        )
        self.assertIn(
            "cuda_architectures_cache: '75_80_86_87_89_90_100_103_120_121'",
            linux_cuda,
        )
        self.assertIn(
            "cuda_architectures: '61;75;80;86;87;89;90'",
            windows_cuda,
        )

    def test_linux_cuda_composition_uses_hosted_runner(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        composer = job_block(
            workflow,
            "compose_linux_cuda",
            "compose_linux_rocm",
        )
        job_header = composer[: composer.index("    steps:")]

        self.assertIn(
            "    runs-on: ${{ needs.metadata.outputs.runner_4 }}",
            job_header,
        )
        self.assertNotIn("self-hosted", job_header)
        self.assertNotIn("USE_SELF_HOSTED", job_header)

    def test_release_uses_shared_host_and_runtime_producers(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

        self.assertEqual(
            workflow.count("uses: ./.github/actions/prepare-host-input"),
            2,
        )
        self.assertEqual(
            workflow.count(
                "uses: ./.github/actions/prepare-windows-host-input",
            ),
            1,
        )
        self.assertGreaterEqual(
            workflow.count("uses: ./.github/actions/prepare-native-runtime-input"),
            5,
        )
        self.assertEqual(
            workflow.count("uses: ./.github/actions/compose-product-input"),
            8,
        )
        self.assertNotIn(
            "scripts/ci-client-readiness-smoke.sh host-input/mesh-llm runtime-root",
            workflow,
        )

    def test_release_product_jobs_do_not_restore_compiler_caches(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        product_jobs = (
            "compose_linux_aarch64_cuda",
            "compose_linux_cuda",
            "compose_linux_rocm",
            "compose_linux_vulkan",
        )

        for index, job_name in enumerate(product_jobs):
            start = workflow.index(f"  {job_name}:")
            next_starts = [
                workflow.find(f"  {other_job}:", start + 1)
                for other_job in product_jobs[index + 1 :]
            ]
            next_starts = [position for position in next_starts if position >= 0]
            end = min(next_starts) if next_starts else len(workflow)
            job = workflow[start:end]
            self.assertIn(
                "uses: ./.github/actions/compose-product-input",
                job,
            )
            self.assertNotIn(
                "uses: ./.github/actions/configure-sccache-gha",
                job,
            )
            self.assertNotIn("uses: actions/cache@", job)


if __name__ == "__main__":
    unittest.main()
