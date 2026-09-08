import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci-prepare-native-runtime.sh"
SDK_SMOKE_SCRIPTS = (
    ROOT / "scripts/ci-rust-sdk-smoke.sh",
    ROOT / "scripts/ci-kotlin-sdk-smoke.sh",
    ROOT / "scripts/ci-swift-sdk-smoke.sh",
)


def host_os() -> str:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform in {"win32", "cygwin"}:
        return "windows"
    raise RuntimeError(f"unsupported test platform: {sys.platform}")


def host_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "aarch64"
    if machine in {"amd64", "x86_64"}:
        return "x86_64"
    return machine


def host_target() -> str:
    targets = {
        ("macos", "aarch64"): "aarch64-apple-darwin",
        ("macos", "x86_64"): "x86_64-apple-darwin",
        ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
        ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
        ("windows", "x86_64"): "x86_64-pc-windows-msvc",
    }
    return targets[(host_os(), host_arch())]


def accelerated_backend() -> str:
    return "metal" if host_os() == "macos" else "vulkan"


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def current_skippy_abi() -> str:
    values = {}
    constants = ROOT / "crates/skippy-ffi/src/lib.rs"
    for line in constants.read_text(encoding="utf-8").splitlines():
        for part in ("MAJOR", "MINOR", "PATCH"):
            prefix = f"pub const ABI_VERSION_{part}: u32 = "
            if line.startswith(prefix):
                values[part] = line.removeprefix(prefix).removesuffix(";")
    return f"{values['MAJOR']}.{values['MINOR']}.{values['PATCH']}"


class CiPrepareNativeRuntimeTests(unittest.TestCase):
    def write_runtime(
        self,
        root: Path,
        runtime_id: str = "meshllm-native-runtime-test",
        backend: str = "cpu",
        skippy_abi: str | None = None,
    ) -> Path:
        artifact = root / runtime_id
        library = artifact / "lib/runtime.bin"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"verified runtime fixture")
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        manifest = {
            "runtime": {
                "id": runtime_id,
                "mesh_version": "0.72.1",
                "skippy_abi": skippy_abi or current_skippy_abi(),
                "platform": {
                    "os": host_os(),
                    "arch": host_arch(),
                    "target": host_target(),
                },
                "backend": {"kind": backend},
                "libraries": ["lib/runtime.bin"],
                "files": {"lib/runtime.bin": digest},
            },
            "build": {
                "primary_library": "lib/runtime.bin",
                "library_sha256": digest,
            },
        }
        (artifact / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return artifact

    def write_fake_binary(
        self,
        product: Path,
        runtime_id: str,
        backend: str,
        supported: bool,
    ) -> Path:
        binary = product / "mesh-llm"
        binary.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "id": runtime_id,
                "mesh_version": "0.72.1",
                "skippy_abi": current_skippy_abi(),
                "backend": backend,
                "os": host_os(),
                "arch": host_arch(),
                "supported": supported,
                "rejection_reasons": [] if supported else ["mesh version mismatch"],
                "url": None,
            }
        ]
        rows_path = product / "runtime-rows.json"
        rows_path.write_text(
            json.dumps({"catalogs": {}, "runtimes": rows}), encoding="utf-8"
        )
        binary.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" != *"runtime list --available"* ]]; then
    echo "unexpected fake mesh-llm invocation: $*" >&2
    exit 2
fi
cat "$(dirname "$0")/runtime-rows.json"
""",
            encoding="utf-8",
        )
        make_executable(binary)
        return binary

    def run_script(
        self,
        out: Path,
        binary: Path,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["CI"] = "true"
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [
                str(SCRIPT),
                str(out),
                "cpu",
                "--reuse-from-binary",
                str(binary),
            ],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_reuses_compatible_runtime_beside_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            product = Path(directory) / "product"
            runtime = self.write_runtime(product / "native-runtimes")
            binary = self.write_fake_binary(product, runtime.name, "cpu", True)
            out = Path(directory) / "fallback"

            result = self.run_script(out, binary)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), runtime.resolve())
            self.assertIn("verified native runtime artifact", result.stderr)
            self.assertIn("Reusing compatible native runtime", result.stderr)
            self.assertFalse(out.exists())

    def test_reuses_sole_compatible_product_backend_before_cpu_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            product = Path(directory) / "product"
            runtime = self.write_runtime(
                product / "native-runtimes",
                backend=accelerated_backend(),
            )
            binary = self.write_fake_binary(
                product,
                runtime.name,
                accelerated_backend(),
                True,
            )
            out = Path(directory) / "fallback"

            result = self.run_script(out, binary)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), runtime.resolve())
            self.assertFalse(out.exists())

    def test_rejects_incompatible_staged_runtime_without_building(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            product = Path(directory) / "product"
            runtime = self.write_runtime(product / "native-runtimes")
            binary = self.write_fake_binary(product, runtime.name, "cpu", False)
            out = Path(directory) / "fallback"

            result = self.run_script(
                out,
                binary,
                extra_env={"MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK": "1"},
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected exactly one compatible", result.stderr)
            self.assertFalse(out.exists())

    def test_rejects_staged_runtime_with_wrong_sdk_abi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            product = Path(directory) / "product"
            runtime = self.write_runtime(
                product / "native-runtimes",
                skippy_abi="99.99.99",
            )
            binary = self.write_fake_binary(product, runtime.name, "cpu", True)
            out = Path(directory) / "fallback"

            result = self.run_script(
                out,
                binary,
                extra_env={"MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK": "1"},
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("has Skippy ABI 99.99.99", result.stderr)
            self.assertFalse(out.exists())

    def test_ci_requires_adjacent_runtime_without_explicit_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            product = Path(directory) / "product"
            binary = self.write_fake_binary(product, "unused", "cpu", True)
            out = Path(directory) / "fallback"

            result = self.run_script(out, binary)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Adjacent native runtime bundle is required in CI", result.stderr)
            self.assertFalse(out.exists())

    def test_explicit_ci_fallback_preserves_standalone_build_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_root = Path(directory) / "repo"
            scripts = fake_root / "scripts"
            scripts.mkdir(parents=True)
            helper = scripts / SCRIPT.name
            shutil.copy2(SCRIPT, helper)
            make_executable(helper)

            package_script = scripts / "package-native-runtime.sh"
            package_script.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
out=""
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--out" ]]; then
        out="$2"
        shift 2
    else
        shift
    fi
done
runtime="$out/meshllm-native-runtime-fallback"
mkdir -p "$runtime"
printf '{}\n' > "$runtime/manifest.json"
printf 'archive\n' > "$out/meshllm-native-runtime-fallback.tar.gz"
""",
                encoding="utf-8",
            )
            make_executable(package_script)
            verify_script = scripts / "verify-native-runtime-package.sh"
            verify_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
            make_executable(verify_script)

            product = Path(directory) / "product"
            binary = self.write_fake_binary(product, "unused", "cpu", True)
            out = Path(directory) / "fallback"
            env = os.environ.copy()
            env.update(
                {
                    "CI": "true",
                    "MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK": "1",
                }
            )

            result = subprocess.run(
                [
                    str(helper),
                    str(out),
                    "cpu",
                    "--reuse-from-binary",
                    str(binary),
                ],
                cwd=fake_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("building the standalone fallback", result.stderr)
            self.assertEqual(
                Path(result.stdout.strip()),
                out / "meshllm-native-runtime-fallback",
            )

    def test_all_sdk_smokes_prefer_the_binary_runtime(self) -> None:
        for script in SDK_SMOKE_SCRIPTS:
            with self.subTest(script=script.name):
                contents = script.read_text(encoding="utf-8")
                self.assertIn("--reuse-from-binary", contents)
                self.assertIn('"$1"', contents)


if __name__ == "__main__":
    unittest.main()
