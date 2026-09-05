from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ROOT / ".github" / "actions"
COMPOSE_SCRIPT = ROOT / "scripts" / "ci-compose-product-input.sh"


class CiProductComposerTests(unittest.TestCase):
    def read_action(self, name: str) -> str:
        return (ACTIONS / name / "action.yml").read_text(encoding="utf-8")

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


if __name__ == "__main__":
    unittest.main()
