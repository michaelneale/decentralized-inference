from __future__ import annotations

from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ROOT / ".github" / "actions"
RELEASE_FOOTER_MANIFEST = ROOT / "crates" / "mesh-llm-release-footer" / "Cargo.toml"
XTASK_MANIFEST = ROOT / "tools" / "xtask" / "Cargo.toml"


class CiWindowsCompositionTests(unittest.TestCase):
    def read_action(self, name: str) -> str:
        return (ACTIONS / name / "action.yml").read_text(encoding="utf-8")

    def read_compute_changes(self) -> str:
        return self.read_action("compute-changes") + "\n" + (
            ACTIONS / "compute-changes" / "derive-outputs.sh"
        ).read_text(encoding="utf-8")

    def test_host_action_uses_canonical_dynamic_host_builder(self) -> None:
        action = self.read_action("prepare-host-input")

        self.assertIn('scripts/build-host.sh --profile "$INPUT_PROFILE"', action)
        self.assertIn("scripts/verify-host-dependencies.py", action)
        self.assertNotIn("package-native-runtime.sh", action)

    def test_windows_host_action_owns_the_neutral_host_integrity_contract(
        self,
    ) -> None:
        action = self.read_action("prepare-windows-host-input")

        self.assertIn(
            "& .\\scripts\\build-windows.ps1 -BuildProfile $profile -HostOnly",
            action,
        )
        self.assertIn("scripts\\verify-host-dependencies.py", action)
        self.assertIn("mesh-llm.exe.sha256", action)
        self.assertIn("cargo build -q -p xtask --bin xtask", action)
        self.assertIn("release-attestation stamp", action)
        self.assertIn("release-attestation inspect", action)
        self.assertIn('"$attestationVerifierPath.sha256"', action)
        self.assertIn(
            '"$verifierHash  release-attestation-verifier.exe"',
            action,
        )
        self.assertNotIn("package-native-runtime.sh", action)
        self.assertNotIn("compose-product", action)

    def test_windows_attestation_verifier_stays_native_abi_free(self) -> None:
        xtask = tomllib.loads(XTASK_MANIFEST.read_text(encoding="utf-8"))
        xtask_dependencies = xtask["dependencies"]
        self.assertEqual(
            xtask_dependencies["mesh-llm-release-footer"],
            {"workspace": True},
        )
        self.assertNotIn("mesh-llm-system", xtask_dependencies)
        self.assertNotIn("skippy-ffi", xtask_dependencies)

        footer = tomllib.loads(RELEASE_FOOTER_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(set(footer["dependencies"]), {"hex", "sha2"})

    def test_windows_debug_host_uses_the_package_version_for_composition(
        self,
    ) -> None:
        action = self.read_action("prepare-windows-host-input")

        debug = action[
            action.index('if ($profile -eq "debug")')
            : action.index('if ($env:INPUT_SKIP_UI -eq "true")')
        ]
        self.assertIn("cargo pkgid -p mesh-llm", debug)
        self.assertIn("$env:MESH_LLM_BUILD_VERSION", debug)
        self.assertNotIn("git ", debug)

    def test_windows_routes_cover_every_shared_product_primitive(self) -> None:
        action = self.read_compute_changes()
        routing = action[
            action.index("WINDOWS_CPU_INPUTS=")
            : action.index("# SDK smokes are consumer tests")
        ]
        cpu_routing = routing[: routing.index("WINDOWS_GPU_INPUTS=")]
        gpu_routing = routing[routing.index("WINDOWS_GPU_INPUTS=") :]

        self.assertIn("^crates/mesh-llm-release-footer/", cpu_routing)
        self.assertNotIn("^crates/mesh-llm-release-footer/", gpu_routing)
        self.assertIn("package-release", cpu_routing)
        self.assertIn("package-release", gpu_routing)
        for workflow in (
            "ci",
            "main_[a-z]+",
            "pr_[a-z]+",
            "release",
            "windows-warm-caches",
        ):
            with self.subTest(workflow=workflow):
                self.assertIn(workflow, cpu_routing)
                self.assertIn(workflow, gpu_routing)

        for input_name, route in (
            ("WINDOWS_CPU_INPUTS", cpu_routing),
            ("WINDOWS_GPU_INPUTS", gpu_routing),
        ):
            with self.subTest(input_name=input_name):
                match = re.search(
                    rf"{input_name}=.*?grep -E '([^']+)'",
                    route,
                )
                self.assertIsNotNone(
                    match,
                    f"{input_name} classifier pattern was not found",
                )
                classifier = re.compile(match.group(1))
                for action_path in (
                    ".github/actions/compute-changes/action.yml",
                    ".github/actions/compute-changes/derive-outputs.sh",
                ):
                    with self.subTest(action_path=action_path):
                        self.assertRegex(action_path, classifier)

        for primitive in (
            "prepare-windows-host-input",
            "prepare-native-runtime-input",
            "compose-product-input",
            "save-and-verify-actions-cache",
            "package-native-runtime",
            "verify-native-runtime-package",
            "verify-checksum-sidecar",
            "safe-extract-tar",
            "compose-product-bundle",
            "ci-compose-product-input",
            "ci-client-readiness-smoke",
        ):
            with self.subTest(primitive=primitive):
                self.assertIn(primitive, routing)

    def test_windows_abi_cache_action_keys_every_compatibility_boundary(
        self,
    ) -> None:
        action = self.read_action("restore-windows-abi-cache")

        for action_input in (
            "backend:",
            "build_dir:",
            "toolchain_epoch:",
            "architecture_set:",
            "cuda_toolchain_version:",
            "vulkan_toolchain_version:",
            "rocm_toolchain_version:",
        ):
            with self.subTest(action_input=action_input):
                self.assertIn(action_input, action)

        self.assertIn(
            '$backend -notin @("cpu", "cuda", "rocm", "vulkan")',
            action,
        )
        self.assertIn(
            '$backend -in @("cuda", "rocm") -and -not $architectureSet',
            action,
        )
        self.assertIn(
            "build_dir must resolve inside GITHUB_WORKSPACE",
            action,
        )
        self.assertIn(
            "build_dir must remain outside the replaceable llama.cpp ",
            action,
        )
        self.assertIn(
            "worktree: $resolvedBuildDir",
            action,
        )
        for toolchain_boundary in (
            "cuda-$version-Jimver-v0.2.35",
            "vulkan-$version-jakoch-v1.5.2",
            "rocm-$version",
        ):
            with self.subTest(toolchain_boundary=toolchain_boundary):
                self.assertIn(toolchain_boundary, action)

        expected_hash = (
            "${{ hashFiles("
            "'.github/actions/restore-windows-abi-cache/action.yml', "
            "'.github/actions/save-and-verify-actions-cache/action.yml', "
            "'.github/actions/resolve-native-toolchain-epoch/action.yml', "
            "'.github/actions/prepare-native-runtime-input/action.yml', "
            "'.github/actions/setup-windows-rocm-sdk/action.yml', "
            "'scripts/build-llama.sh', 'scripts/prepare-llama.sh', "
            "'scripts/package-native-runtime.sh', "
            "'third_party/llama.cpp/upstream.txt', "
            "'third_party/llama.cpp/patches/**', "
            "'.github/cache-version.txt') }}"
        )
        self.assertIn(expected_hash, action)
        self.assertIn(
            '"mesh-llm-windows-2022-skippy-abi-'
            '$backend-$architectureSet-$toolchain-$toolchainEpoch-$inputHash"',
            action,
        )
        self.assertIn(
            "toolchain_epoch must match MESH_LLM_LLAMA_TOOLCHAIN_EPOCH",
            action,
        )
        self.assertIn(
            "actions/cache/restore@"
            "caa296126883cff596d87d8935842f9db880ef25 # v5.1.0",
            action,
        )
        self.assertNotIn("restore-keys:", action)
        self.assertIn(
            "value: ${{ steps.restore.outputs.cache-hit }}",
            action,
        )
        self.assertIn(
            "value: ${{ steps.restore.outputs.cache-primary-key }}",
            action,
        )
        self.assertIn(
            "value: ${{ steps.identity.outputs.build-dir }}",
            action,
        )

    def test_windows_native_cache_inputs_fail_closed_and_callers_opt_in(
        self,
    ) -> None:
        for action_name in (
            "restore-windows-abi-cache",
            "setup-windows-rocm-sdk",
        ):
            with self.subTest(action=action_name):
                action = self.read_action(action_name)
                input_start = action.index("  allow-native-github-cache:")
                input_end = action.find("\n\n", input_start)
                input_block = action[input_start:input_end]
                self.assertIn('required: false', input_block)
                self.assertIn('default: "false"', input_block)
                self.assertNotIn('default: "true"', input_block)

        expected_callers = {
            "restore-windows-abi-cache": {
                "ci-windows-runtime-slice.yml": 1,
                "release.yml": 2,
                "windows-warm-caches.yml": 2,
            },
            "setup-windows-rocm-sdk": {
                "ci-windows-runtime-slice.yml": 1,
                "release.yml": 1,
                "windows-warm-caches.yml": 1,
            },
        }
        policy_value = (
            "allow-native-github-cache: "
            "${{ needs.runner_policy.outputs.allow_native_github_cache }}"
        )
        for action_name, expected_counts in expected_callers.items():
            calls: list[tuple[str, str]] = []
            for workflow_path in sorted(
                (ROOT / ".github" / "workflows").glob("*.yml")
            ):
                lines = workflow_path.read_text(encoding="utf-8").splitlines()
                for index, line in enumerate(lines):
                    marker = f"uses: ./.github/actions/{action_name}"
                    if marker not in line:
                        continue
                    line_indent = len(line) - len(line.lstrip())
                    step_indent = line_indent
                    for candidate in reversed(lines[:index]):
                        candidate_indent = len(candidate) - len(candidate.lstrip())
                        if candidate_indent <= line_indent and candidate.lstrip().startswith("-"):
                            step_indent = candidate_indent
                            break
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
                    calls.append((workflow_path.name, "\n".join(lines[start:end])))

            actual_counts: dict[str, int] = {}
            for workflow_name, block in calls:
                actual_counts[workflow_name] = actual_counts.get(workflow_name, 0) + 1
                with self.subTest(action=action_name, workflow=workflow_name):
                    if workflow_name == "ci-windows-runtime-slice.yml":
                        self.assertIn(policy_value, block)
                    else:
                        self.assertIn(
                            'allow-native-github-cache: "true"',
                            block,
                        )
            self.assertEqual(expected_counts, actual_counts)


if __name__ == "__main__":
    unittest.main()
