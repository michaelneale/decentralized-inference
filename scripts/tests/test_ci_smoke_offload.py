"""Execute the offload gate with native-log fixtures, without starting a GPU job."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci-verify-smoke-offload.py"
LAYERS = "load_tensors: offloaded 25/25 layers to GPU\n"
BUFFER = "load_tensors:        CUDA0 model buffer size =   256.50 MiB\n"


class SmokeOffloadTests(unittest.TestCase):
    def run_gate(self, log, device="CUDA0"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skippy-native.log"
            if log is not None:
                path.write_text(log)
            return subprocess.run([sys.executable, str(SCRIPT), "--device", device,
                                   "--native-log", str(path)], text=True, capture_output=True)

    def test_accepts_positive_offload_and_cuda_weights(self):
        result = self.run_gate(LAYERS + BUFFER)
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(result.stdout)
        self.assertEqual(evidence["device"], "CUDA0")
        self.assertEqual(evidence["evidence"], [LAYERS.strip(), BUFFER.strip()])

    def test_rejects_cpu_fallback_and_non_evidence(self):
        for log in (None, "", LAYERS, BUFFER,
                    "ggml_cuda_init: found 1 CUDA devices\n",
                    "load_tensors: CPU_Mapped model buffer size = 256.50 MiB\n",
                    LAYERS.replace("25/25", "0/25") + BUFFER,
                    LAYERS.replace("25/25", "26/25") + BUFFER,
                    LAYERS + BUFFER.replace("256.50", "0.00"),
                    LAYERS + BUFFER.replace("CUDA0", "CUDA_Host"),
                    LAYERS + BUFFER.replace("CUDA0", "CUDA1"),
                    LAYERS + BUFFER.replace("model buffer", "compute buffer"),
                    "load_tensors: offloading 25 repeating layers to GPU\n" + BUFFER,
                    LAYERS + BUFFER + LAYERS.replace("25/25", "0/25")):
            with self.subTest(log=log):
                self.assertNotEqual(self.run_gate(log).returncode, 0)

    def test_cpu_row_does_not_require_cuda_log(self):
        result = self.run_gate(None, "CPU")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not requested", result.stdout)

    def test_unknown_device_cannot_disable_the_gate(self):
        for device in ("auto", "cuda", "", "CUDA1"):
            with self.subTest(device=device):
                self.assertNotEqual(self.run_gate(LAYERS + BUFFER, device).returncode, 0)

    def test_smoke_scripts_reject_unknown_device_before_launch(self):
        for script in ("ci-smoke-test.sh", "ci-compat-smoke.sh"):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                launched = root / "launched"
                binary = root / "mesh-llm"
                binary.write_text('#!/bin/sh\nprintf "launch\\n" >> "$LAUNCH_PROBE"\nexit 1\n')
                binary.chmod(0o755)
                model = root / "model.gguf"
                model.touch()
                (root / "native-runtimes").mkdir()
                # Isolate the SDK preflight too: a misplaced guard must be able
                # to reach the launch probe, never a real runtime or endpoint.
                (root / "scripts").mkdir()
                helper = root / "scripts/ci-prepare-native-runtime.sh"
                helper.write_text('#!/bin/sh\nexit 0\n')
                helper.chmod(0o755)
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(("MESH_", "BASH_ENV"))}
                result = subprocess.run(
                    ["bash", str(ROOT / "scripts" / script), str(binary), str(root), str(model)],
                    cwd=root, env={**env, "MESH_CI_DEVICE": "auto",
                                   "LAUNCH_PROBE": str(launched),
                                   "MESH_CI_LOG": str(root / "smoke.log"),
                                   "MESH_COMPAT_LOG": str(root / "compat.log")},
                    capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Unsupported MESH_CI_DEVICE", result.stderr)
                self.assertFalse(launched.exists(), "invalid device launched mesh-llm")


    def test_other_product_backends_are_preserved_without_cuda_claim(self):
        for device in ("Vulkan0", "ROCm0", "MTL0"):
            with self.subTest(device=device):
                result = self.run_gate(None, device)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"{device} smoke: CUDA offload qualification not requested", result.stdout)

    def test_compat_device_override_and_common_fallback(self):
        script = (ROOT / "scripts/ci-compat-smoke.sh").read_text()
        setup = script[:script.index('MESH_LLM=')]
        for common, compat, expected in (("CUDA0", None, "CUDA0"),
                                          (None, "CUDA0", "CUDA0"),
                                          (None, "MTL0", "MTL0"),
                                          (None, "Vulkan0", "Vulkan0"),
                                          (None, "ROCm0", "ROCm0"),
                                          ("CPU", "CUDA0", "CUDA0"),
                                          (None, None, "CPU")):
            with self.subTest(common=common, compat=compat):
                env = {key: value for key, value in os.environ.items()
                       if key not in ("MESH_CI_DEVICE", "MESH_COMPAT_DEVICE")}
                if common is not None:
                    env["MESH_CI_DEVICE"] = common
                if compat is not None:
                    env["MESH_COMPAT_DEVICE"] = compat
                result = subprocess.run(["bash", "-c", setup + '\nprintf "%s" "$SMOKE_DEVICE"'],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)
