from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "package-native-runtime.sh"


def write_failing_nvcc(path: Path) -> None:
    path.write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
    path.chmod(0o755)


class PackageNativeRuntimeTests(unittest.TestCase):
    def test_macos_model_package_tool_uses_portable_linker_flags(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        start = script.index("build_model_package_tool() {")
        end = script.index("collect_runtime_libraries() {", start)
        function = script[start:end]
        self.assertIn('command -v ld64.lld', function)
        self.assertIn(
            'CARGO_ENCODED_RUSTFLAGS=-Clink-arg=-fuse-ld=lld',
            function,
        )
        self.assertIn('cargo_env+=("CARGO_ENCODED_RUSTFLAGS=")', function)
        self.assertNotIn(
            "LLVM ld64.lld is required to build the macOS model package tool",
            function,
        )

    def test_windows_package_skips_dynamic_model_package_tool(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        start = script.index("build_model_package_tool() {")
        end = script.index("collect_runtime_libraries() {", start)
        harness = (
            "set -euo pipefail\n"
            + script[start:end]
            + 'runtime_os="windows"\n'
            + 'stage_dir="${TMPDIR:-/tmp}/mesh-test-no-windows-tool"\n'
            + "build_model_package_tool\n"
            + '[[ ! -e "$stage_dir/tools/skippy-model-package" ]]\n'
        )
        result = subprocess.run(
            ["/bin/bash", "-s"],
            input=harness,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpu_package_includes_model_package_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dir = root / "build"
            build_dir.mkdir()
            (build_dir / "libllama.so").write_bytes(b"test native runtime")

            tool_dir = root / "tools"
            tool_dir.mkdir()
            patchelf = tool_dir / "patchelf"
            patchelf.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            patchelf.chmod(0o755)

            env = os.environ.copy()
            env["LLAMA_STAGE_BUILD_DIR"] = str(build_dir)
            env["MESH_NATIVE_RUNTIME_MODEL_PACKAGE_TOOL"] = "/usr/bin/true"
            env["PATH"] = f"{tool_dir}{os.pathsep}{env['PATH']}"
            result = subprocess.run(
                [
                    "/bin/bash",
                    str(SCRIPT),
                    "--backend",
                    "cpu",
                    "--target",
                    "x86_64-unknown-linux-gnu",
                    "--out",
                    str(root / "output"),
                ],
                env=env,
                text=True,
                capture_output=True,
            )

            result.check_returncode()
            manifest = json.loads(
                (
                    root
                    / "output"
                    / "meshllm-native-runtime-linux-x86_64-cpu"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(manifest["runtime"]["tools"]),
                {"tools/skippy-model-package"},
            )
            archive = (
                root
                / "output"
                / "meshllm-native-runtime-linux-x86_64-cpu.tar.gz"
            )
            with tarfile.open(archive, "r:gz") as handle:
                self.assertFalse(
                    any(Path(member.name).name.startswith("._") for member in handle),
                    "native runtime archive must not contain macOS AppleDouble entries",
                )

    def test_rocm_benchmark_tool_uses_configured_offload_arches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dir = root / "build"
            build_dir.mkdir()
            (build_dir / "libllama.so").write_bytes(b"test native runtime")

            tool_dir = root / "bin"
            tool_dir.mkdir()
            hipcc = tool_dir / "hipcc"
            hipcc.write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                "printf '%s\\n' \"$@\" > \"$HIPCC_ARGS_LOG\"\n"
                "previous=''\n"
                "for argument in \"$@\"; do\n"
                "  if [ \"$previous\" = '-o' ]; then output=\"$argument\"; fi\n"
                "  previous=\"$argument\"\n"
                "done\n"
                ": > \"$output\"\n",
                encoding="utf-8",
            )
            hipcc.chmod(0o755)
            patchelf = tool_dir / "patchelf"
            patchelf.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            patchelf.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "HIPCC": str(hipcc),
                    "HIPCC_ARGS_LOG": str(root / "hipcc-args.log"),
                    "LLAMA_STAGE_AMDGPU_TARGETS": "gfx90a;gfx942, gfx1151",
                    "LLAMA_STAGE_BUILD_DIR": str(build_dir),
                    "MESH_NATIVE_RUNTIME_MODEL_PACKAGE_TOOL": "/usr/bin/true",
                    "PATH": f"{tool_dir}{os.pathsep}{env['PATH']}",
                }
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    str(SCRIPT),
                    "--backend",
                    "rocm",
                    "--target",
                    "x86_64-unknown-linux-gnu",
                    "--out",
                    str(root / "output"),
                ],
                env=env,
                text=True,
                capture_output=True,
            )

            result.check_returncode()
            arguments = (
                (root / "hipcc-args.log").read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(
                [
                    argument
                    for argument in arguments
                    if argument.startswith("--offload-arch=")
                ],
                [
                    "--offload-arch=gfx90a",
                    "--offload-arch=gfx942",
                    "--offload-arch=gfx1151",
                ],
            )
            manifest = json.loads(
                (
                    root
                    / "output"
                    / "meshllm-native-runtime-linux-x86_64-rocm"
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["runtime"]["backend"]["rocm"]["gpu_arches"],
                ["gfx90a", "gfx942", "gfx1151"],
            )

    def test_cuda_flavor_uses_mesh_cuda_version_major(self) -> None:
        self.assertEqual(
            self.backend_flavor(
                "cuda", mesh_cuda_version="13.1.2", compiler_version="13.1"
            ),
            "cuda13",
        )

    def test_cuda_flavor_accepts_major_only_mesh_cuda_version(self) -> None:
        self.assertEqual(
            self.backend_flavor(
                "cuda", mesh_cuda_version="13", compiler_version="13.0"
            ),
            "cuda13",
        )

    def test_cuda_flavor_rejects_mesh_cuda_version_minor_mismatch(self) -> None:
        result = self.backend_flavor_process(
            "cuda", mesh_cuda_version="13.1.2", compiler_version="13.0"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "does not match the selected CUDA compiler/toolkit version 13.0",
            result.stderr,
        )

    def test_explicit_cuda_toolkit_major_wins(self) -> None:
        result = self.backend_flavor_process(
            "cuda",
            toolkit_major="12",
            compiler_version="13.0",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match the selected CUDA compiler", result.stderr)

    def test_cuda_flavor_requires_compiler_or_toolkit_evidence(self) -> None:
        result = self.backend_flavor_process("cuda")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA toolkit version could not be detected", result.stderr)

    def test_cuda_blackwell_requires_compiler_or_toolkit_evidence(self) -> None:
        result = self.backend_flavor_process("cuda-blackwell")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA toolkit version could not be detected", result.stderr)

    def test_explicit_cuda_toolkit_major_rejects_non_digits(self) -> None:
        for toolkit_major in ("12.1", "cuda12"):
            with self.subTest(toolkit_major=toolkit_major):
                result = self.backend_flavor_process(
                    "cuda",
                    toolkit_major=toolkit_major,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "MESH_LLM_CUDA_TOOLKIT_MAJOR must be digits-only",
                    result.stderr,
                )

    def test_cuda13_compiler_cannot_be_labeled_cuda12(self) -> None:
        result, _ = self.package_cuda_fixture(
            compiler_version="13.0",
            toolkit_major="12",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match the selected CUDA compiler", result.stderr)

    def test_explicit_cuda_toolkit_major_is_accepted_when_compiler_matches(self) -> None:
        result, manifest = self.package_cuda_fixture(
            compiler_version="12.9",
            toolkit_major="12",
        )
        result.check_returncode()
        self.assertEqual(manifest["runtime"]["backend"]["cuda"]["toolkit_major"], 12)

    def test_cuda13_compiler_derives_manifest_major_without_declaration(self) -> None:
        result, manifest = self.package_cuda_fixture(compiler_version="13.0")
        result.check_returncode()
        self.assertEqual(manifest["runtime"]["backend"]["cuda"]["toolkit_major"], 13)

    def test_cuda_version_declaration_must_match_compiler_major(self) -> None:
        result, _ = self.package_cuda_fixture(
            compiler_version="13.0",
            mesh_cuda_version="12.9.2",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match the selected CUDA compiler", result.stderr)

    def test_cuda_version_declaration_is_validated_against_compiler(self) -> None:
        result, manifest = self.package_cuda_fixture(
            compiler_version="13.1",
            mesh_cuda_version="13.1.2",
        )
        result.check_returncode()
        self.assertEqual(manifest["runtime"]["backend"]["cuda"]["toolkit_major"], 13)

    def test_cuda_major_only_version_declaration_is_accepted(self) -> None:
        result, manifest = self.package_cuda_fixture(
            compiler_version="13.0",
            mesh_cuda_version="13",
        )
        result.check_returncode()
        self.assertEqual(manifest["runtime"]["backend"]["cuda"]["toolkit_major"], 13)

    def test_cuda_version_declaration_rejects_compiler_minor_mismatch(self) -> None:
        result, _ = self.package_cuda_fixture(
            compiler_version="13.0",
            mesh_cuda_version="13.1.2",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "does not match the selected CUDA compiler/toolkit version 13.0",
            result.stderr,
        )

    def test_cuda_packaging_without_toolkit_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dir = root / "build"
            build_dir.mkdir()
            (build_dir / "libllama.so").write_bytes(b"test native runtime")
            tool_dir = root / "bin"
            tool_dir.mkdir()
            write_failing_nvcc(tool_dir / "nvcc")
            patchelf = tool_dir / "patchelf"
            patchelf.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            patchelf.chmod(0o755)
            env = self.clean_package_env(tool_dir)
            env["LLAMA_STAGE_BUILD_DIR"] = str(build_dir)
            env["MESH_NATIVE_RUNTIME_MODEL_PACKAGE_TOOL"] = "/usr/bin/true"
            result = subprocess.run(
                [
                    "/bin/bash",
                    str(SCRIPT),
                    "--backend",
                    "cuda",
                    "--target",
                    "x86_64-unknown-linux-gnu",
                    "--out",
                    str(root / "output"),
                ],
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CUDA toolkit version could not be detected", result.stderr)

    def backend_flavor(
        self,
        backend: str,
        *,
        mesh_cuda_version: str | None = None,
        toolkit_major: str | None = None,
        compiler_version: str | None = None,
    ) -> str:
        result = self.backend_flavor_process(
            backend,
            mesh_cuda_version=mesh_cuda_version,
            toolkit_major=toolkit_major,
            compiler_version=compiler_version,
        )
        result.check_returncode()
        return result.stdout.strip()

    def backend_flavor_process(
        self,
        backend: str,
        *,
        mesh_cuda_version: str | None = None,
        toolkit_major: str | None = None,
        compiler_version: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        script = SCRIPT.read_text(encoding="utf-8")
        start = script.index("backend_flavor()")
        end = script.index("build_backend()", start)
        helpers = script[start:end]
        env = os.environ.copy()
        env["BACKEND"] = backend
        for name in (
            "CUDACXX",
            "CMAKE_CUDA_COMPILER",
            "NVCC",
            "CUDAToolkit_ROOT",
            "CUDA_HOME",
            "CUDA_PATH",
        ):
            env.pop(name, None)
        for name, value in (
            ("MESH_CUDA_VERSION", mesh_cuda_version),
            ("MESH_LLM_CUDA_TOOLKIT_MAJOR", toolkit_major),
        ):
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        with tempfile.TemporaryDirectory() as directory:
            if compiler_version is not None:
                compiler = Path(directory) / "nvcc"
                compiler.write_text(
                    "#!/bin/bash\n"
                    'if [[ "${1:-}" == "--version" ]]; then\n'
                    f'  printf "Cuda compilation tools, release {compiler_version}, V{compiler_version}.0\\n"\n'
                    "  exit 0\n"
                    "fi\n",
                    encoding="utf-8",
                )
                compiler.chmod(0o755)
            else:
                write_failing_nvcc(Path(directory) / "nvcc")
            env["PATH"] = f"{directory}{os.pathsep}{env['PATH']}"
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    "set -euo pipefail\n"
                    f"source {SCRIPT.parent / 'lib' / 'cuda-toolkit.sh'}\n"
                    f"{helpers}\nbackend_flavor",
                ],
                env=env,
                text=True,
                capture_output=True,
            )
        return result

    def package_cuda_fixture(
        self,
        *,
        compiler_version: str,
        mesh_cuda_version: str | None = None,
        toolkit_major: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        build_dir = root / "build"
        build_dir.mkdir()
        (build_dir / "libllama.so").write_bytes(b"test native runtime")

        tool_dir = root / "bin"
        tool_dir.mkdir()
        nvcc = tool_dir / "nvcc"
        nvcc.write_text(
            "#!/bin/bash\n"
            'if [[ "${1:-}" == "--version" ]]; then\n'
            f'  printf "Cuda compilation tools, release {compiler_version}, V{compiler_version}.0\\n"\n'
            "  exit 0\n"
            "fi\n"
            "output=\n"
            "previous=\n"
            "for argument in \"$@\"; do\n"
            '  if [[ "$previous" == "-o" ]]; then output="$argument"; fi\n'
            '  previous="$argument"\n'
            "done\n"
            '[[ -n "$output" ]] && : > "$output"\n',
            encoding="utf-8",
        )
        nvcc.chmod(0o755)
        patchelf = tool_dir / "patchelf"
        patchelf.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        patchelf.chmod(0o755)

        inherited_env = os.environ.copy()
        inherited_env["CMAKE_CUDA_COMPILER"] = str(root / "inherited-cmake-nvcc")
        env = self.clean_package_env(tool_dir, inherited_env)
        env["LLAMA_STAGE_BUILD_DIR"] = str(build_dir)
        env["MESH_NATIVE_RUNTIME_MODEL_PACKAGE_TOOL"] = "/usr/bin/true"
        if mesh_cuda_version is not None:
            env["MESH_CUDA_VERSION"] = mesh_cuda_version
        if toolkit_major is not None:
            env["MESH_LLM_CUDA_TOOLKIT_MAJOR"] = toolkit_major
        result = subprocess.run(
            [
                "/bin/bash",
                str(SCRIPT),
                "--backend",
                "cuda",
                "--target",
                "x86_64-unknown-linux-gnu",
                "--out",
                str(root / "output"),
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        manifest_path = (
            root
            / "output"
            / f"meshllm-native-runtime-linux-x86_64-cuda{compiler_version.split('.', 1)[0]}"
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        return result, manifest

    @staticmethod
    def clean_package_env(
        tool_dir: Path, base_env: dict[str, str] | None = None
    ) -> dict[str, str]:
        env = (base_env if base_env is not None else os.environ).copy()
        for name in (
            "CUDACXX",
            "CMAKE_CUDA_COMPILER",
            "CUDAToolkit_ROOT",
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_LIBRARY_PATH",
            "LIBRARY_PATH",
            "LD_LIBRARY_PATH",
            "NVCC",
            "MESH_CUDA_VERSION",
            "MESH_LLM_CUDA_TOOLKIT_MAJOR",
        ):
            env.pop(name, None)
        env["PATH"] = f"{tool_dir}{os.pathsep}{env['PATH']}"
        return env


if __name__ == "__main__":
    unittest.main()
