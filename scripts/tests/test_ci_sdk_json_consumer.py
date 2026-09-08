"""The #1675 CLI JSON boundary must run even without full SDK selection."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.tests import test_ci_artifact_actions as artifacts
from scripts.tests import test_ci_prepare_native_runtime as runtime_tests
from scripts.tests import test_plan_ci as planner_tests


ROOT = Path(__file__).resolve().parents[2]


class SdkJsonConsumerTests(unittest.TestCase):
    def test_original_cli_change_selects_the_consumer_and_producers(self) -> None:
        # Exact changed paths at #1675 head 14ec8d0, not this CI repair's
        # control-plane paths (which would mask the miss by selecting everything).
        payload = planner_tests.fixture("runtime-catalog-pr-1675.json")
        for paths in (
            payload["changed_files"],
            ["crates/mesh-llm-commands/src/runtime_native/formatters.rs"],
        ):
            with self.subTest(paths=paths):
                plan = planner_tests.PLANNER.build_plan(
                    {**payload, "changed_files": paths}, root=ROOT
                )
                self.assertNotIn("ci-control", plan["domains"])
                self.assertEqual(plan["matrices"]["sdk"], [])
                self.assertIn("ui-artifact", plan["required_slices"])
                self.assertIn("runtime-product", plan["required_slices"])
                self.assertEqual(
                    [row["id"] for row in plan["matrices"]["runtime_products"]],
                    ["linux-cpu"],
                )
                self.assertEqual(
                    [row["id"] for row in plan["matrices"]["hosts"]],
                    ["linux-amd64-host"],
                )

        lane = (ROOT / ".github/workflows/ci-linux-lane.yml").read_text()
        product = lane.split("  runtime_product:\n", 1)[1].split(
            "  kotlin_sdk_input:\n", 1
        )[0]
        self.assertIn("needs: [hosts, native_runtimes]", product)
        self.assertIn("uses: ./.github/workflows/ci-linux-product-slice.yml", product)
        workflow = (ROOT / ".github/workflows/ci-linux-product-slice.yml").read_text()
        self.assertIn("ref: ${{ inputs.source_sha || github.sha }}", workflow)
        self.assertIn("uses: ./.github/actions/compose-product-input", workflow)
        self.assertNotIn("readiness_smoke:", workflow)
        action = (ROOT / ".github/actions/compose-product-input/action.yml").read_text()
        readiness = action.split("  readiness_smoke:\n", 1)[1].split(
            "  attestation_public_key_file:\n", 1
        )[0]
        self.assertIn('default: "true"', readiness)
        self.assertIn("scripts/ci-compose-product-input.sh", action)

    def run_composer(self, workspace: Path, report_kind: str) -> subprocess.CompletedProcess:
        scripts = workspace / "scripts"
        scripts.mkdir()
        for name in (
            "ci-compose-product-input.sh",
            "ci-prepare-native-runtime.sh",
            "compose-product-bundle.py",
            "verify-native-runtime-package.sh",
            "verify-checksum-sidecar.py",
            "safe-extract-tar.py",
        ):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        abi = workspace / "crates/skippy-ffi/src/lib.rs"
        abi.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "crates/skippy-ffi/src/lib.rs", abi)
        # Only readiness/network startup is stubbed; execute the real composer,
        # SDK reader and package verification against checksum-bound fixture bytes.
        (scripts / "ci-client-readiness-smoke.sh").write_text(
            '#!/bin/sh\ntouch "$GITHUB_WORKSPACE/readiness-ran"\n'
        )
        (scripts / "ci-client-readiness-smoke.sh").chmod(0o755)
        (scripts / "package-native-runtime.sh").write_text(
            '#!/bin/sh\ntouch "$GITHUB_WORKSPACE/fallback-ran"\nexit 99\n'
        )
        (scripts / "package-native-runtime.sh").chmod(0o755)
        # Exercise the Linux-only gate on development hosts too, without changing
        # the immutable package fixture into an actual loadable Linux runtime.
        tools = workspace / "bin"
        tools.mkdir()
        (tools / "uname").write_text("#!/bin/sh\nprintf 'Linux\\n'\n")
        (tools / "uname").chmod(0o755)
        host_input, runtime_input = artifacts.CiArtifactActionTests().write_fake_product_inputs(
            workspace
        )
        manifest_path = next(runtime_input.glob("*/manifest.json"))
        manifest = json.loads(manifest_path.read_text())
        manifest["runtime"]["skippy_abi"] = runtime_tests.current_skippy_abi()
        manifest_path.write_text(json.dumps(manifest))
        rows = [{"id": manifest["runtime"]["id"], "backend": "cpu", "supported": True}]
        reports = {
            "catalog": {"catalogs": {}, "runtimes": rows},
            "legacy": rows,
            "malformed": {"runtimes": {}},
            "unsupported": {"runtimes": [{**rows[0], "supported": False}]},
        }
        host = host_input / "mesh-llm"
        host.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'if [[ "$*" == *"runtime list --available"* ]]; then\n'
            '  [[ "$*" == *"--log-format json"* && "$*" == *"--json"* ]]\n'
            '  [[ "$MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK" == "0" ]]\n'
            '  [[ -z "${MESH_LLM_CONFIG:-}" && -z "${MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR:-}" ]]\n'
            '  touch "$GITHUB_WORKSPACE/sdk-reader-ran"\n'
            f"  printf '%s\\n' '{json.dumps(reports[report_kind])}'\n"
            'else\n  printf "mesh-llm 1.2.3\\n"\nfi\n'
        )
        digest = hashlib.sha256(host.read_bytes()).hexdigest()
        (host_input / "mesh-llm.sha256").write_text(f"{digest}  mesh-llm\n")
        return subprocess.run(
            [str(scripts / "ci-compose-product-input.sh")],
            cwd=workspace,
            env={
                **os.environ,
                "PATH": f"{tools}:{os.environ['PATH']}",
                "GITHUB_WORKSPACE": str(workspace),
                "GITHUB_OUTPUT": str(workspace / "github-output"),
                "INPUT_HOST_INPUT_DIR": str(host_input),
                "INPUT_RUNTIME_INPUT_DIR": str(runtime_input),
                "INPUT_OUTPUT_DIR": str(workspace / "product-input"),
                "INPUT_BACKEND": "cpu",
                "INPUT_VERSION": "1.2.3",
                "INPUT_BINARY_NAME": "mesh-llm",
                "INPUT_READINESS_SMOKE": "true",
                "INPUT_ATTESTATION_PUBLIC_KEY_FILE": "",
                "INPUT_ATTESTATION_VERIFIER": "",
                "MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK": "1",
                "MESH_LLM_CONFIG": str(workspace / "ambient-config"),
            },
            capture_output=True, text=True, check=False,
        )

    def test_composition_runs_the_sdk_reader_before_publishing(self) -> None:
        for report in ("catalog", "legacy"):
            with self.subTest(report=report), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                result = self.run_composer(workspace, report)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((workspace / "sdk-reader-ran").exists())
                self.assertTrue((workspace / "readiness-ran").exists())
                self.assertTrue((workspace / "product-input.tar.gz").exists())
                self.assertFalse((workspace / "fallback-ran").exists())
                self.assertIn("Reusing compatible native runtime", result.stderr)

    def test_bad_cli_json_blocks_product_publication_without_building(self) -> None:
        for report, error in (
            ("malformed", "native runtime compatibility output must be"),
            ("unsupported", "expected exactly one compatible adjacent native runtime"),
        ):
            with self.subTest(report=report), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                result = self.run_composer(workspace, report)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertTrue((workspace / "sdk-reader-ran").exists())
                self.assertFalse((workspace / "readiness-ran").exists())
                self.assertFalse((workspace / "product-input.tar.gz").exists())
                self.assertFalse((workspace / "github-output").exists())
                self.assertFalse((workspace / "fallback-ran").exists())


if __name__ == "__main__":
    unittest.main()
