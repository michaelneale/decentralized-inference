from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SkippyDynamicLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary = Path(cls.directory.name) / (
            "build-script.exe" if os.name == "nt" else "build-script"
        )
        # Compile a standalone test fixture with the host linker. The contracts
        # job has Rust and cc but does not install LLD.
        result = subprocess.run(
            ["rustc", "--edition=2024",
             str(ROOT / "crates/skippy-ffi/build.rs"), "-o", str(cls.binary)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"build-script fixture compilation failed:\n{result.stderr}")

    def run_build_script(self, target: str, backend: str, *, runtime_loader: bool = False,
                         legacy: bool = False) -> tuple[list[str], Path]:
        fixture = tempfile.TemporaryDirectory()
        self.addCleanup(fixture.cleanup)
        root = Path(fixture.name).resolve()
        build = root / "native build"
        build.mkdir()
        stubs = root / "selected toolkit" / "stubs"
        stubs.mkdir(parents=True)
        driver = stubs / "libcuda.so"
        driver.touch()
        (build / "CMakeCache.txt").write_text(
            f"CUDA_cuda_driver_LIBRARY:FILEPATH={driver}\n"
        )
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("LLAMA_STAGE_", "SKIPPY_LLAMA_", "CARGO_FEATURE_"))
        }
        prefix = "SKIPPY_LLAMA" if legacy else "LLAMA_STAGE"
        env.update({
            "CARGO_MANIFEST_DIR": str(ROOT / "crates/skippy-ffi"),
            "TARGET": target,
            f"{prefix}_LINK_MODE": "dynamic",
            f"{prefix}_BUILD_DIR": str(build),
            f"{prefix}_LIB_DIR": str(root / "staged libraries"),
            f"{prefix}_BACKEND": backend,
        })
        if runtime_loader:
            env["CARGO_FEATURE_DYNAMIC_RUNTIME"] = "1"
        result = subprocess.run(
            [str(self.binary)], cwd=root, env=env, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.splitlines(), stubs

    def test_linux_cuda_links_selected_driver_without_runtime_search_path(self) -> None:
        for target in ("aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"):
            with self.subTest(target=target):
                lines, stubs = self.run_build_script(target, "cuda")
                self.assertIn(f"cargo:rustc-link-search=native={stubs}", lines)
                self.assertIn("cargo:rustc-link-lib=dylib=cuda", lines)
                self.assertIn("cargo:rustc-link-lib=dylib=llama", lines)
                self.assertIn(
                    f"cargo:rerun-if-changed={stubs.parent.parent / 'native build/CMakeCache.txt'}",
                    lines,
                )
                self.assertFalse(any("rpath" in line or "rustc-link-arg" in line for line in lines))

    def test_legacy_cuda_selectors_resolve_the_same_driver(self) -> None:
        lines, stubs = self.run_build_script("aarch64-unknown-linux-gnu", "cuda", legacy=True)
        self.assertIn(f"cargo:rustc-link-search=native={stubs}", lines)
        self.assertIn("cargo:rustc-link-lib=dylib=cuda", lines)

    def test_other_dynamic_targets_do_not_acquire_cuda_dependency(self) -> None:
        for target, backend in (
            ("aarch64-unknown-linux-gnu", "cpu"),
            ("x86_64-unknown-linux-gnu", "rocm"),
            ("x86_64-unknown-linux-gnu", "vulkan"),
            ("aarch64-apple-darwin", "metal"),
            ("x86_64-pc-windows-msvc", "cuda"),
        ):
            with self.subTest(target=target, backend=backend):
                lines, stubs = self.run_build_script(target, backend)
                self.assertNotIn("cargo:rustc-link-lib=dylib=cuda", lines)
                self.assertNotIn(f"cargo:rustc-link-search=native={stubs}", lines)
                self.assertIn("cargo:rustc-link-lib=dylib=llama", lines)

    def test_runtime_loader_remains_free_of_native_link_dependencies(self) -> None:
        lines, _ = self.run_build_script("aarch64-unknown-linux-gnu", "cuda", runtime_loader=True)
        self.assertFalse(any(line.startswith("cargo:rustc-link-") for line in lines))

    @unittest.skipUnless(sys.platform == "linux", "requires the ELF linker")
    def test_driverless_elf_link_resolves_indirect_driver_without_bundling_stub(self) -> None:
        self.assertIsNotNone(shutil.which("cc"))
        self.assertIsNotNone(shutil.which("readelf"))
        self.assertIsNotNone(shutil.which("ld.bfd"))
        lines, stubs = self.run_build_script("aarch64-unknown-linux-gnu", "cuda")
        root = stubs.parent.parent
        staged = root / "staged libraries"
        staged.mkdir()

        def run(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(args, cwd=root, check=True, capture_output=True, text=True)

        driver = root / "driver.c"
        driver.write_text("int cuGetErrorString(void) { return 0; }\n")
        run("cc", "-shared", "-fPIC", str(driver), "-Wl,-soname,libcuda.so.1",
            "-o", str(stubs / "libcuda.so"))
        runtime = root / "runtime.c"
        runtime.write_text("extern int cuGetErrorString(void);\n"
                           "int llama_entry(void) { return cuGetErrorString(); }\n")
        run("cc", "-shared", "-fPIC", str(runtime), "-L", str(stubs), "-lcuda",
            "-Wl,-soname,libllama.so", "-o", str(staged / "libllama.so"))
        empty = root / "empty.c"
        empty.write_text("void unused(void) {}\n")
        for name in ("mtmd", "llama-common"):
            run("cc", "-shared", "-fPIC", str(empty), "-o", str(staged / f"lib{name}.so"))
        source = root / "probe.rs"
        source.write_text('unsafe extern "C" { fn llama_entry() -> i32; }\n'
                          'fn main() { std::process::exit(unsafe { llama_entry() }); }\n')
        args = []
        for line in lines:
            if line.startswith("cargo:rustc-link-search="):
                args.extend(["-L", line.split("=", 1)[1]])
            elif line.startswith("cargo:rustc-link-lib="):
                args.extend(["-l", line.split("=", 1)[1]])
        self.assertEqual(args[-2:], ["-l", "dylib=cuda"])
        # Reproduce the ARM64 release's GNU linker. Other Rust targets may
        # default to LLD, which accepts this unresolved indirect dependency.
        command = ["rustc", "--edition=2024", "-C", "link-arg=-fuse-ld=bfd",
                   str(source), "-o", str(root / "probe")]
        broken = subprocess.run(command + args[:-2], cwd=root, capture_output=True, text=True)
        self.assertNotEqual(broken.returncode, 0)
        self.assertIn("cuGetErrorString", broken.stderr)
        run(*(command + args))
        dynamic = run("readelf", "-d", str(root / "probe")).stdout
        runtime_dynamic = run("readelf", "-d", str(staged / "libllama.so")).stdout
        self.assertIn("[libcuda.so.1]", runtime_dynamic)
        self.assertNotIn(str(stubs), dynamic)
        self.assertNotIn("RPATH", dynamic)
        self.assertNotIn("RUNPATH", dynamic)
        self.assertFalse((staged / "libcuda.so").exists())
        self.assertFalse((staged / "libcuda.so.1").exists())
