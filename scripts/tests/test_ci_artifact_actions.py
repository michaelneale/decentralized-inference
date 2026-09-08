from __future__ import annotations

import hashlib
import json
import os
from fnmatch import fnmatch
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ROOT / ".github" / "actions"
COMPOSE_SCRIPT = ROOT / "scripts" / "ci-compose-product-input.sh"
RELEASE_FOOTER_MANIFEST = ROOT / "crates" / "mesh-llm-release-footer" / "Cargo.toml"
XTASK_MANIFEST = ROOT / "tools" / "xtask" / "Cargo.toml"


class CiArtifactActionTests(unittest.TestCase):
    def read_action(self, name: str) -> str:
        return (ACTIONS / name / "action.yml").read_text(encoding="utf-8")

    def read_compute_changes(self) -> str:
        return self.read_action("compute-changes") + "\n" + (
            ACTIONS / "compute-changes" / "derive-outputs.sh"
        ).read_text(encoding="utf-8")

    def test_external_actions_have_sha_and_release_provenance(self) -> None:
        action_files = sorted(ACTIONS.glob("*/action.yml"))
        workflow_files = sorted(
            (ROOT / ".github" / "workflows").glob("*.yml"),
        )
        exact_pin = re.compile(
            r"^[^@\s]+@[0-9a-f]{40}\s+#\s+\S",
        )
        protected_pr_lanes = {
            f"Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main": f"pr_{lane}.yml"
            for lane in ("quality", "website", "linux", "macos", "windows")
        }
        protected_pre_checkout_action = (
            "Mesh-LLM/mesh-llm/.github/actions/"
            "audit-depot-pr-isolation@ed07043b84d720aab30e75ed2f038f7042576f16"
        )

        for path in (*action_files, *workflow_files):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if "uses:" not in line:
                    continue
                value = line.split("uses:", maxsplit=1)[1].strip()
                if value.startswith("./"):
                    continue
                if value in protected_pr_lanes:
                    self.assertEqual(protected_pr_lanes[value], path.name)
                    continue
                if value == protected_pre_checkout_action:
                    self.assertIn(path.name, {
                        "ci-linux-host-slice.yml",
                        "ci-linux-product-slice.yml",
                        "ci-linux-runtime-slice.yml",
                        "ci-macos-host-slice.yml",
                        "ci-macos-product-slice.yml",
                        "ci-macos-runtime-slice.yml",
                        "ci-platform-checks-slice.yml",
                        "ci-quality-slice.yml",
                        "ci-rust-tests-slice.yml",
                        "ci-ui-artifact-slice.yml",
                        "ci-web-slice.yml",
                        "ci-windows-host-slice.yml",
                        "ci-windows-product-slice.yml",
                        "ci-windows-runtime-slice.yml",
                        "native-sdk-artifact.yml",
                        "static-abi-artifact.yml",
                        "swift-sdk-artifact.yml",
                    })
                    continue
                with self.subTest(
                    path=path.relative_to(ROOT),
                    line=line_number,
                ):
                    self.assertRegex(value, exact_pin)

    def test_workflow_status_gates_do_not_resist_cancellation(self) -> None:
        workflow_files = sorted(
            (ROOT / ".github" / "workflows").glob("*.yml"),
        )

        for path in workflow_files:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn(
                    "always()",
                    path.read_text(encoding="utf-8"),
                )

    def test_quality_slice_requires_ci_contract_validation(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "ci-quality-slice.yml"
        ).read_text(encoding="utf-8")
        contract_start = workflow.index("  quality_contracts:")
        contract_end = workflow.index("\n  rust_fmt:", contract_start)
        contract = workflow[contract_start:contract_end]

        self.assertIn(
            "uses: ./.github/actions/install-actionlint",
            contract,
        )
        self.assertNotIn("tool: actionlint@", contract)
        self.assertIn(
            "actionlint -config-file .github/actionlint.yaml",
            contract,
        )
        self.assertIn(
            "python3 -m pip install --disable-pip-version-check --no-input "
            "-r ci/requirements-ci-python.txt",
            contract,
        )
        self.assertIn(
            "python3 -m unittest discover -s scripts/tests -p 'test_*.py'",
            contract,
        )
        requirements = (
            ROOT / "ci" / "requirements-ci-python.txt"
        ).read_text(encoding="utf-8")
        self.assertRegex(requirements, r"(?m)^PyYAML>=6\.0$")
        self.assertIn(
            "cargo run -p xtask -- repo-consistency release-targets",
            contract,
        )
        self.assertIn("cargo tree -p mesh-llm-client", contract)
        self.assertIn("quality_contracts", workflow)

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

    def write_fake_product_inputs(
        self,
        workspace: Path,
        *,
        host_version: str = "1.2.3",
    ) -> tuple[Path, Path]:
        host_input = workspace / "host-input"
        runtime_input = workspace / "runtime-input"
        host_input.mkdir()
        runtime_input.mkdir()

        host = host_input / "mesh-llm"
        host.write_text(
            "#!/usr/bin/env bash\n"
            f"printf 'mesh-llm {host_version}\\n'\n",
            encoding="utf-8",
        )
        host.chmod(0o755)
        host_digest = hashlib.sha256(host.read_bytes()).hexdigest()
        (host_input / "mesh-llm.sha256").write_text(
            f"{host_digest}  mesh-llm\n",
            encoding="utf-8",
        )
        (host_input / "host-imports.json").write_text(
            "{}\n",
            encoding="utf-8",
        )

        runtime_id = "meshllm-native-runtime-darwin-x86_64-cpu"
        runtime = runtime_input / runtime_id
        (runtime / "lib").mkdir(parents=True)
        (runtime / "tools").mkdir()
        library = runtime / "lib" / "libmesh_fake.a"
        library.write_bytes(b"fake static library")
        tool = runtime / "tools" / "mesh-runtime-bench"
        tool.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        tool.chmod(0o755)
        library_digest = hashlib.sha256(library.read_bytes()).hexdigest()
        tool_digest = hashlib.sha256(tool.read_bytes()).hexdigest()
        manifest = {
            "runtime": {
                "id": runtime_id,
                "mesh_version": "1.2.3",
                "skippy_abi": "1.0.0",
                "platform": {
                    "os": "macos",
                    "arch": "x86_64",
                    "target": "x86_64-apple-darwin",
                },
                "backend": {"kind": "cpu"},
                "libraries": ["lib/libmesh_fake.a"],
                "files": {
                    "lib/libmesh_fake.a": library_digest,
                },
                "tools": {"tools/mesh-runtime-bench": tool_digest},
            },
            "build": {
                "backend": "cpu",
                "primary_library": "lib/libmesh_fake.a",
                "library_sha256": library_digest,
            },
        }
        (runtime / "manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        return host_input, runtime_input

    def write_noncanonical_sidecar(
        self,
        artifact: Path,
        mode: str,
    ) -> None:
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if mode == "wrong-name":
            contents = f"{digest}  unexpected-name\n"
        elif mode == "multiline":
            contents = (
                f"{digest}  {artifact.name}\n"
                f"{digest}  {artifact.name}\n"
            )
        else:
            raise ValueError(f"unsupported sidecar mode: {mode}")
        artifact.with_name(f"{artifact.name}.sha256").write_text(
            contents,
            encoding="utf-8",
        )

    def run_product_composer(
        self,
        workspace: Path,
        *,
        host_version: str = "1.2.3",
        runtime_archive: str | None = None,
        host_sidecar: str | None = None,
        attestation_sidecar: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        host_input, runtime_input = self.write_fake_product_inputs(
            workspace,
            host_version=host_version,
        )
        if host_sidecar is not None:
            self.write_noncanonical_sidecar(
                host_input / "mesh-llm",
                host_sidecar,
            )
        if runtime_archive is not None:
            runtime_dir = next(
                path
                for path in runtime_input.iterdir()
                if path.is_dir()
            )
            archive = runtime_input / f"{runtime_dir.name}.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(runtime_dir, arcname=runtime_dir.name)
            shutil.rmtree(runtime_dir)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if runtime_archive != "missing":
                sidecar_digest = (
                    "0" * 64
                    if runtime_archive == "corrupt"
                    else digest
                )
                archive.with_name(f"{archive.name}.sha256").write_text(
                    f"{sidecar_digest}  {archive.name}\n",
                    encoding="utf-8",
                )
            if runtime_archive == "duplicate":
                (runtime_input / "unrelated.tar.gz.sha256").write_text(
                    f"{digest}  unrelated.tar.gz\n",
                    encoding="utf-8",
                )
        environment = {
            **os.environ,
            "GITHUB_WORKSPACE": str(workspace),
            "GITHUB_OUTPUT": str(workspace / "github-output"),
            "INPUT_HOST_INPUT_DIR": str(host_input),
            "INPUT_RUNTIME_INPUT_DIR": str(runtime_input),
            "INPUT_OUTPUT_DIR": str(workspace / "product-input"),
            "INPUT_BACKEND": "cpu",
            "INPUT_VERSION": "1.2.3",
            "INPUT_BINARY_NAME": "mesh-llm",
            "INPUT_READINESS_SMOKE": "false",
        }
        if attestation_sidecar is not None:
            verifier = host_input / "release-attestation-verifier"
            verifier.write_bytes(b"test verifier")
            self.write_noncanonical_sidecar(
                verifier,
                attestation_sidecar,
            )
            public_key = workspace / "release-attestation-public-key.json"
            public_key.write_text("{}\n", encoding="utf-8")
            environment["INPUT_ATTESTATION_PUBLIC_KEY_FILE"] = str(
                public_key,
            )
        return subprocess.run(
            [str(COMPOSE_SCRIPT)],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def run_runner_selector(
        self,
        *,
        event_name: str,
        ref: str,
        main_enabled: str,
        manual_enabled: str,
        original_event_name: str = "",
        repository: str = "Mesh-LLM/mesh-llm",
        head_repository: str | None = None,
        head_sha: str = "0123456789abcdef0123456789abcdef01234567",
        pr_enabled: str = "false",
        pr_canary_ref: str = "",
        pr_approved_ref: str = "",
        pr_approved_sha: str = "",
        force_hosted: str = "false",
        current_date: str = "2026-08-14",
    ) -> dict[str, str]:
        action = self.read_action("select-ci-runners")
        run_block = action.split("      run: |\n", maxsplit=1)[1]
        script = "\n".join(
            line[8:] if line.startswith("        ") else line
            for line in run_block.splitlines()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "github-output"
            bin_dir = Path(temp_dir) / "bin"
            bin_dir.mkdir()
            date = bin_dir / "date"
            date.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$SELECTOR_TEST_DATE\"\n",
                encoding="utf-8",
            )
            date.chmod(0o755)
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
                    "SELECTOR_TEST_DATE": current_date,
                    "GITHUB_OUTPUT": str(output),
                    "INPUT_EVENT_NAME": event_name,
                    "INPUT_ORIGINAL_EVENT_NAME": original_event_name,
                    "GITHUB_EVENT_NAME": event_name,
                    "INPUT_REPOSITORY": repository,
                    "INPUT_HEAD_REPOSITORY": head_repository or repository,
                    "INPUT_HEAD_SHA": head_sha,
                    "GITHUB_REPOSITORY": repository,
                    "INPUT_REF": ref,
                    "GITHUB_REF": ref,
                    "INPUT_DEPOT_MAIN_ENABLED": main_enabled,
                    "INPUT_DEPOT_PR_ENABLED": pr_enabled,
                    "INPUT_PR_CANARY_REF": pr_canary_ref,
                    "INPUT_PR_APPROVED_REF": pr_approved_ref,
                    "INPUT_PR_APPROVED_SHA": pr_approved_sha,
                    "INPUT_FORCE_HOSTED": force_hosted,
                    "INPUT_MANUAL_USE_DEPOT": manual_enabled,
                    "DISPATCH_ORIGINAL_EVENT_NAME": original_event_name,
                },
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return dict(
                line.split("=", maxsplit=1)
                for line in output.read_text(encoding="utf-8").splitlines()
            )

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
        for workflow in (
            runtime_workflow,
            static_workflow,
            native_sdk_workflow,
            swift_workflow,
            release_workflow,
        ):
            for cache_block in re.findall(
                r"uses: actions/cache@[^\n]+\n"
                r"(?:[ \t]+[^\n]*\n){1,8}",
                workflow,
            ):
                if "LLAMA_STAGE_BUILD_DIR" in cache_block:
                    self.assertIn(
                        "native_toolchain.outputs.epoch",
                        cache_block,
                    )

    def test_native_toolchain_epoch_fingerprints_depot_macos_without_image_vars(
        self,
    ) -> None:
        action = self.read_action("resolve-native-toolchain-epoch")
        run_block = action.split("      run: |\n", maxsplit=1)[1]
        script = "\n".join(
            line[8:] if line.startswith("        ") else line
            for line in run_block.splitlines()
        )

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
        assembly_only_retry = [
            f"swift-sdk-target-{target}-1" for target in targets
        ]
        self.assertEqual(
            {name.rsplit("-", maxsplit=1)[0] for name in assembly_only_retry
             if fnmatch(name, "swift-sdk-target-*")},
            {f"swift-sdk-target-{target}" for target in targets},
        )
        partial_matrix_retry = [
            f"swift-sdk-target-{target}-{2 if index == 3 else 1}"
            for index, target in enumerate(targets)
        ]
        self.assertEqual(
            {name.rsplit("-", maxsplit=1)[0] for name in partial_matrix_retry
             if fnmatch(name, "swift-sdk-target-*")},
            {f"swift-sdk-target-{target}" for target in targets},
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

    def test_product_action_only_composes_verified_inputs(self) -> None:
        action = self.read_action("compose-product-input")

        self.assertIn("scripts/ci-compose-product-input.sh", action)
        self.assertNotIn("cargo build", action)
        self.assertNotIn("package-native-runtime.sh", action)
        script = COMPOSE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("scripts/compose-product-bundle.py", script)
        self.assertIn("scripts/verify-native-runtime-package.sh", script)
        self.assertIn("scripts/verify-checksum-sidecar.py", script)
        self.assertIn("scripts/safe-extract-tar.py", script)
        self.assertIn("scripts/ci-client-readiness-smoke.sh", script)
        self.assertIn('archive_path="$product_dir.tar.gz"', script)
        self.assertIn('tar -C "$product_dir" -czf "$archive_path" .', script)

    def test_product_composer_normalizes_windows_shell_boundaries(self) -> None:
        script = COMPOSE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("local path=\"${1%$'\\r'}\"", script)
        self.assertIn('cygpath -u "$path"', script)
        self.assertIn('cygpath -m "$path"', script)
        self.assertIn(
            'canonical_paths+=("$(to_shell_path "$path")")',
            script,
        )
        self.assertIn(
            'GITHUB_OUTPUT="$(to_shell_path "$GITHUB_OUTPUT")"',
            script,
        )
        self.assertIn('require_file "immutable host" "$host"', script)
        self.assertNotIn('test -f "$host"', script)

    def test_product_archive_preserves_verified_executable_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            result = self.run_product_composer(workspace)

            self.assertEqual(result.returncode, 0, result.stderr)
            archive = workspace / "product-input.tar.gz"
            self.assertTrue(archive.is_file())
            with tarfile.open(archive, "r:gz") as bundle:
                host = next(
                    member
                    for member in bundle.getmembers()
                    if member.name.endswith("/mesh-llm")
                )
                tool = next(
                    member
                    for member in bundle.getmembers()
                    if member.name.endswith(
                        "/tools/mesh-runtime-bench"
                    )
                )
                self.assertNotEqual(host.mode & 0o111, 0)
                self.assertNotEqual(tool.mode & 0o111, 0)
            output = (workspace / "github-output").read_text(encoding="utf-8")
            self.assertIn(f"archive_path={archive.resolve()}", output)

    def test_product_composer_rejects_host_version_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_product_composer(
                Path(temp_dir),
                host_version="9.9.9",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("composed host version mismatch", result.stderr)

    def test_product_composer_accepts_host_build_metadata(self) -> None:
        """Non-release hosts carry `+g<sha>[.dirty]` build metadata.

        Semver build metadata is not part of version identity, so a debug host
        stamped with its commit SHA still composes against the runtime's
        release version.
        """
        for host_version in ("1.2.3+gABC123", "1.2.3+gABC123.dirty"):
            with self.subTest(host_version=host_version):
                with tempfile.TemporaryDirectory() as temp_dir:
                    result = self.run_product_composer(
                        Path(temp_dir),
                        host_version=host_version,
                    )

                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_product_composer_rejects_drift_despite_build_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_product_composer(
                Path(temp_dir),
                host_version="9.9.9+gABC123",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("composed host version mismatch", result.stderr)

    def test_product_composer_accepts_one_checksums_runtime_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_product_composer(
                Path(temp_dir),
                runtime_archive="valid",
            )

            self.assertEqual(result.returncode, 0, result.stderr)

    def test_product_composer_requires_exact_runtime_archive_sidecar(
        self,
    ) -> None:
        for mode in ("missing", "duplicate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp_dir:
                result = self.run_product_composer(
                    Path(temp_dir),
                    runtime_archive=mode,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn("expected exactly one checksum sidecar", result.stderr)

    def test_product_composer_rejects_corrupt_runtime_archive_sidecar(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_product_composer(
                Path(temp_dir),
                runtime_archive="corrupt",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("archive checksum mismatch", result.stderr)

    def test_product_composer_rejects_noncanonical_host_sidecar(self) -> None:
        expected_errors = {
            "wrong-name": "checksum sidecar names",
            "multiline": "exactly one canonical line",
        }
        for mode, expected_error in expected_errors.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp_dir:
                result = self.run_product_composer(
                    Path(temp_dir),
                    host_sidecar=mode,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)

    def test_product_composer_rejects_noncanonical_verifier_sidecar(
        self,
    ) -> None:
        expected_errors = {
            "wrong-name": "checksum sidecar names",
            "multiline": "exactly one canonical line",
        }
        for mode, expected_error in expected_errors.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp_dir:
                result = self.run_product_composer(
                    Path(temp_dir),
                    attestation_sidecar=mode,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)

    def test_release_attestation_is_verified_without_compiling_in_composer(
        self,
    ) -> None:
        host_action = self.read_action("prepare-host-input")
        product_action = self.read_action("compose-product-input")
        product_script = COMPOSE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("cargo build -q -p xtask --bin xtask", host_action)
        self.assertIn("release-attestation-verifier.sha256", host_action)
        self.assertNotIn("cargo ", product_action)
        self.assertIn(
            '"$attestation_verifier" release-attestation inspect',
            product_script,
        )
        self.assertIn(
            '"$python_bin" scripts/verify-checksum-sidecar.py \\\n'
            '        "$attestation_verifier"',
            product_script,
        )

    def test_smoke_restore_rechecks_the_archived_product(self) -> None:
        action = self.read_action("restore-smoke-inputs")

        self.assertIn("expected exactly one composed product archive", action)
        self.assertIn("scripts/safe-extract-tar.py", action)
        self.assertNotIn("tar -xzf", action)
        self.assertIn("product host path must be", action)
        self.assertIn(
            "product runtime must be one direct child of native-runtimes",
            action,
        )
        self.assertIn("product top-level contents are not canonical", action)
        self.assertIn(
            "product must contain exactly its manifest-selected runtime",
            action,
        )
        self.assertIn("scripts/verify-native-runtime-package.sh", action)
        self.assertIn("--check", action)

    def test_smoke_restore_model_is_optional(self) -> None:
        action = self.read_action("restore-smoke-inputs")
        model_inputs_present = (
            "steps.resolve-model.outputs.url != '' && "
            "steps.resolve-model.outputs.file != ''"
        )

        self.assertEqual(action.count(model_inputs_present), 4)
        self.assertIn("model_manifest:", action)
        self.assertIn("model_cadence:", action)
        self.assertIn("scripts/resolve-test-model-manifest.py", action)
        self.assertIn('--cadence "$MODEL_CADENCE"', action)
        self.assertIn("--require-single-file", action)
        self.assertIn('^[A-Za-z0-9][A-Za-z0-9._-]*$', action)
        self.assertIn("--verify-root \"$HOME/.models\"", action)
        self.assertIn("MODEL_MANIFEST: ${{ inputs.model_manifest }}", action)
        self.assertIn("MODEL_CADENCE: ${{ inputs.model_cadence }}", action)
        self.assertIn("MODEL_URL: ${{ steps.resolve-model.outputs.url }}", action)
        self.assertIn("MODEL_FILE: ${{ steps.resolve-model.outputs.file }}", action)
        self.assertNotIn('"${{ inputs.model_manifest }}"', action)
        self.assertNotIn('"${{ inputs.model_cadence }}"', action)
        self.assertNotIn('"${{ steps.resolve-model.outputs.url }}"', action)
        self.assertNotIn('"${{ steps.resolve-model.outputs.file }}"', action)
        self.assertIn(
            f"if: ${{{{ {model_inputs_present} }}}}\n"
            "      id: cache-model",
            action,
        )
        self.assertIn(
            f"if: ${{{{ {model_inputs_present} }}}}\n"
            "      id: model-file",
            action,
        )

    def test_product_action_rejects_destructive_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            host_input = workspace / "inputs" / "host"
            runtime_input = workspace / "inputs" / "runtime"
            host_input.mkdir(parents=True)
            runtime_input.mkdir(parents=True)
            sentinel = workspace / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            outside = workspace.parent / f"{workspace.name}-outside"
            dangerous_outputs = (
                ".",
                "./",
                "product/..",
                str(workspace),
                str(outside),
                str(host_input),
                str(host_input / "product"),
                str(workspace / "inputs"),
            )

            for output in dangerous_outputs:
                with self.subTest(output=output):
                    result = subprocess.run(
                        [str(COMPOSE_SCRIPT)],
                        cwd=workspace,
                        env={
                            **os.environ,
                            "GITHUB_WORKSPACE": str(workspace),
                            "GITHUB_OUTPUT": str(workspace / "github-output"),
                            "INPUT_HOST_INPUT_DIR": str(host_input),
                            "INPUT_RUNTIME_INPUT_DIR": str(runtime_input),
                            "INPUT_OUTPUT_DIR": output,
                            "INPUT_BACKEND": "cpu",
                            "INPUT_VERSION": "",
                            "INPUT_BINARY_NAME": "mesh-llm",
                            "INPUT_READINESS_SMOKE": "false",
                        },
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

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
