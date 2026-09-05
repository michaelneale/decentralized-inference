from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ROOT / ".github" / "actions"


class CiNativeToolchainRoutingTests(unittest.TestCase):
    def read_action(self, name: str) -> str:
        return (ACTIONS / name / "action.yml").read_text(encoding="utf-8")

    def read_workflow(self, name: str) -> dict:
        path = ROOT / ".github" / "workflows" / name
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def read_compute_changes(self) -> str:
        return self.read_action("compute-changes") + "\n" + (
            ACTIONS / "compute-changes" / "derive-outputs.sh"
        ).read_text(encoding="utf-8")

    def test_native_toolchain_epoch_is_exact_and_shared_with_build_stamp(
        self,
    ) -> None:
        resolver = self.read_action("resolve-native-toolchain-epoch")
        runtime_workflow = (
            ROOT / ".github" / "workflows" / "ci-linux-runtime-slice.yml"
        ).read_text(encoding="utf-8")
        release_workflow = (
            ROOT / ".github" / "workflows" / "release.yml"
        ).read_text(encoding="utf-8")
        warmer = (
            ROOT / ".github" / "workflows" / "windows-warm-caches.yml"
        ).read_text(encoding="utf-8")

        for contract in (
            'image_os="${ImageOS:-}"',
            'image_version="${ImageVersion:-}"',
            'epoch="runner-${RUNNER_OS_VALUE}-${RUNNER_ARCH_VALUE}"',
            'INPUT_PINNED_EPOCH: ${{ inputs.pinned_epoch }}',
            'echo "epoch=$epoch" >> "$GITHUB_OUTPUT"',
            'echo "MESH_LLM_LLAMA_TOOLCHAIN_EPOCH=$epoch" >> "$GITHUB_ENV"',
            "sw_vers -productVersion",
            "xcodebuild -version",
            "cmake --version",
            "ninja --version",
        ):
            with self.subTest(contract=contract):
                self.assertIn(contract, resolver)

        static_workflow = (
            ROOT / ".github" / "workflows" / "static-abi-artifact.yml"
        ).read_text(encoding="utf-8")
        native_sdk_workflow = (
            ROOT / ".github" / "workflows" / "native-sdk-artifact.yml"
        ).read_text(encoding="utf-8")
        swift_workflow = (
            ROOT / ".github" / "workflows" / "swift-sdk-artifact.yml"
        ).read_text(encoding="utf-8")

        for workflow in (
            runtime_workflow,
            static_workflow,
            native_sdk_workflow,
            swift_workflow,
            release_workflow,
            warmer,
        ):
            self.assertIn(
                "uses: ./.github/actions/resolve-native-toolchain-epoch",
                workflow,
            )

        # Inspect the parsed step objects so a long `with:` block cannot be
        # truncated by a line-count regex, and so each cache is matched to
        # the epoch step in its own job rather than the first matching block
        # in an entire workflow.
        for workflow_name in (
            "ci-linux-runtime-slice.yml",
            "static-abi-artifact.yml",
            "native-sdk-artifact.yml",
            "swift-sdk-artifact.yml",
            "release.yml",
        ):
            workflow = self.read_workflow(workflow_name)
            for job_name, job in (workflow.get("jobs") or {}).items():
                if not isinstance(job, dict):
                    continue
                steps = job.get("steps") or []
                epoch_steps = [
                    step
                    for step in steps
                    if isinstance(step, dict)
                    and step.get("uses") == "./.github/actions/resolve-native-toolchain-epoch"
                    and step.get("id") == "native_toolchain"
                ]
                for step in steps:
                    if not isinstance(step, dict) or not str(step.get("uses", "")).startswith(
                        "actions/cache@"
                    ):
                        continue
                    with_args = step.get("with") or {}
                    cache_key = str(with_args.get("key", ""))
                    cache_path = str(with_args.get("path", ""))
                    with self.subTest(workflow=workflow_name, job=job_name, cache=step.get("name")):
                        self.assertEqual(
                            len(epoch_steps),
                            1,
                            "every native build cache must have one stable epoch step in its job",
                        )
                        self.assertIn("steps.native_toolchain.outputs.epoch", cache_key)
                        if "${{ env.LLAMA_STAGE_BUILD_DIR }}" in cache_path:
                            self.assertIn("LLAMA_STAGE_BUILD_DIR", job.get("env", {}))

    def test_native_toolchain_epoch_fingerprints_depot_macos_without_image_vars(
        self,
    ) -> None:
        action_document = yaml.safe_load(self.read_action("resolve-native-toolchain-epoch")) or {}
        epoch_steps = [
            step
            for step in action_document.get("runs", {}).get("steps", [])
            if isinstance(step, dict)
            and step.get("id") == "resolve"
            and 'echo "epoch=$epoch"' in str(step.get("run", ""))
        ]
        self.assertEqual(len(epoch_steps), 1)
        script = epoch_steps[0]["run"]

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            bin_dir = workspace / "bin"
            bin_dir.mkdir()
            for command in ("sw_vers", "xcodebuild", "clang", "cmake", "ninja"):
                executable = bin_dir / command
                executable.write_text(
                    "#!/bin/sh\nprintf 'fixture-%s-1\\n' \"${0##*/}\"\n",
                    encoding="utf-8",
                )
                executable.chmod(0o755)

            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
                "GITHUB_OUTPUT": str(workspace / "github-output"),
                "GITHUB_ENV": str(workspace / "github-env"),
                "INPUT_PINNED_EPOCH": "",
                "INPUT_INCLUDE_TOOL_VERSIONS": "true",
                "RUNNER_OS_VALUE": "macOS",
                "RUNNER_ARCH_VALUE": "ARM64",
            }
            environment.pop("ImageOS", None)
            environment.pop("ImageVersion", None)
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            output = (workspace / "github-output").read_text(encoding="utf-8")
            self.assertRegex(
                output,
                r"^epoch=runner-macOS-ARM64-native-[0-9a-f]{64}\n$",
            )

            environment["INPUT_INCLUDE_TOOL_VERSIONS"] = "false"
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "ImageOS and ImageVersion are required unless exact native tool versions are included",
                result.stderr,
            )

    def test_push_routing_diffs_the_complete_event_range(self) -> None:
        action = self.read_compute_changes()
        push_start = action.index(
            'elif [[ "${{ inputs.event_name }}" == "push" ]]',
        )
        push_end = action.index(
            'elif [[ "${{ inputs.event_name }}" == "workflow_dispatch" ]]',
            push_start,
        )
        push = action[push_start:push_end]

        self.assertIn('base_sha="${{ inputs.base_sha }}"', push)
        self.assertIn('head_sha="${{ inputs.head_sha }}"', push)
        self.assertIn('git diff --name-only "$base_sha" "$head_sha"', push)
        self.assertIn('"$base_sha" =~ ^0+$', push)
        self.assertIn('"__force_all__"', push)
        self.assertNotIn("HEAD^", action)
        self.assertIn('if [[ "$FORCE_ALL" == "true" ]]', action)
        force_windows = action[
            action.index('if [[ "$FORCE_ALL" == "true" ]]')
            : action.index(
                "# SDK smokes are consumer tests",
            )
        ]
        self.assertIn('WINDOWS_CPU_BUILD_REQUIRED="true"', force_windows)
        self.assertIn('WINDOWS_GPU_BUILD_REQUIRED="true"', force_windows)

    def test_runner_contract_routing_covers_cache_evidence_actions(
        self,
    ) -> None:
        action = self.read_compute_changes()
        routing = action[
            action.index("RUNNER_CONTRACT_INPUTS=")
            : action.index("# Determine docs_only")
        ]

        for local_action in (
            "capture-sccache-stats",
            "configure-sccache-gha",
            "resolve-native-toolchain-epoch",
            "select-ci-runners",
        ):
            with self.subTest(local_action=local_action):
                self.assertIn(local_action, routing)

        epoch_resolver = "resolve-native-toolchain-epoch"
        for route_start, route_end in (
            ("BACKEND_INPUTS=", "WINDOWS_CPU_BUILD_REQUIRED="),
            ("WINDOWS_CPU_INPUTS=", "# SDK smokes are consumer tests"),
            ("DIRECT_SDK_INPUTS=", "# Inference artifacts are needed"),
        ):
            with self.subTest(route=route_start):
                route = action[
                    action.index(route_start) : action.index(
                        route_end,
                        action.index(route_start),
                    )
                ]
                self.assertIn(epoch_resolver, route)

    def test_justfile_release_primitives_route_backend_builds(self) -> None:
        action = self.read_compute_changes()
        match = re.search(
            r"function is_backend_recipe\(name\).*?"
            r"return name ~ /\^\((.*?)\)\$/",
            action,
            re.DOTALL,
        )
        if match is None:
            self.fail("backend recipe allowlist was not found")
        recipe_names = set(match.group(1).split("|"))

        for recipe in (
            "release-host-build",
            "release-runtime-build",
            "release-host-build-windows",
        ):
            with self.subTest(recipe=recipe):
                self.assertIn(recipe, recipe_names)

    def test_imported_justfile_sources_are_classified_on_both_diff_sides(self) -> None:
        action = self.read_compute_changes()

        self.assertIn("grep -E '^Justfile$|^just/.+\\.just$'", action)
        self.assertIn("$JUSTFILE_SOURCE_BASE_SHA:$JUSTFILE_SOURCE", action)
        self.assertIn("$HEAD_SHA:$JUSTFILE_SOURCE", action)
        self.assertIn(
            'JUSTFILE_SOURCE_BASE_SHA=$(git merge-base "$BASE_SHA" "$HEAD_SHA")',
            action,
        )
        self.assertIn("git diff --name-status --no-renames", action)
        self.assertIn("A$'\\t'*)", action)
        self.assertIn("D$'\\t'*)", action)
        self.assertIn("M$'\\t'*)", action)
        self.assertIn("justfile_has_recipe \"$JUSTFILE_SOURCE_BASE\"", action)
        self.assertIn("justfile_has_recipe \"$JUSTFILE_SOURCE_HEAD\"", action)
        self.assertIn(
            'changed_range_touches_lines "$JUSTFILE_SOURCE_BACKEND_LINES_HEAD" '
            '"$JUSTFILE_SOURCE_CHANGED_LINES" new',
            action,
        )
        self.assertIn(
            'changed_range_touches_lines "$JUSTFILE_SOURCE_BACKEND_LINES_BASE" '
            '"$JUSTFILE_SOURCE_CHANGED_LINES" old',
            action,
        )

    def test_top_level_justfile_inputs_of_backend_recipes_route_backend_builds(
        self,
    ) -> None:
        action = self.read_compute_changes()

        self.assertIn(
            'justfile_backend_recipe_tokens "$JUSTFILE_SOURCE_BASE_SHA" '
            '> "$JUSTFILE_BACKEND_TOKENS_BASE"',
            action,
        )
        self.assertIn(
            'justfile_backend_recipe_tokens "$HEAD_SHA" '
            '> "$JUSTFILE_BACKEND_TOKENS_HEAD"',
            action,
        )
        self.assertIn(
            'justfile_backend_input_lines "$JUSTFILE_SOURCE_BASE" '
            '"$JUSTFILE_BACKEND_TOKENS_BASE"',
            action,
        )
        self.assertIn(
            'justfile_backend_input_lines "$JUSTFILE_SOURCE_HEAD" '
            '"$JUSTFILE_BACKEND_TOKENS_HEAD"',
            action,
        )
        self.assertIn('>> "$JUSTFILE_SOURCE_BACKEND_LINES_BASE"', action)
        self.assertIn('>> "$JUSTFILE_SOURCE_BACKEND_LINES_HEAD"', action)

    def test_backend_recipe_attributes_route_backend_builds(self) -> None:
        action = self.read_compute_changes()

        self.assertIn("pending_attribute_lines[++pending_attribute_count] = NR", action)
        self.assertIn("print pending_attribute_lines[pending_index]", action)
        self.assertIn("delete pending_attribute_lines", action)

    def test_sccache_seed_keys_include_imported_just_sources(self) -> None:
        workflow_dir = ROOT / ".github" / "workflows"
        workflows = (
            "cache-warm-sccache.yml",
            "ci-quality-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-rust-tests-slice.yml",
            "ci-linux-runtime-slice.yml",
        )

        for workflow in workflows:
            with self.subTest(workflow=workflow):
                source = (workflow_dir / workflow).read_text(encoding="utf-8")
                self.assertIn(
                    "hashFiles('Cargo.lock', '.github/cache-version.txt', 'Justfile', 'just/**')",
                    source,
                )

    def test_root_justfile_import_graph_changes_fail_open_to_backend_builds(self) -> None:
        action = self.read_compute_changes()

        self.assertIn("JUSTFILE_SOURCE_DIFF=$(git diff -U0", action)
        self.assertIn(
            'printf \'%s\\n\' "$JUSTFILE_SOURCE_DIFF" | justfile_changed_import',
            action,
        )
        self.assertIn("if (line ~ /^import[?]?[[:space:]]+/)", action)
        self.assertIn(
            '[[ "$JUSTFILE_SOURCE_BASE_AVAILABLE" == "false" '
            '&& "$JUSTFILE_SOURCE_HEAD_AVAILABLE" == "false" ]]',
            action,
        )


if __name__ == "__main__":
    unittest.main()
