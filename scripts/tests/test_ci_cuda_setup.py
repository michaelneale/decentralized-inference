"""Execute both CUDA workflow preflights with isolated tool/library probes."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
LIBRARIES = ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12")


class CudaSetupTests(unittest.TestCase):
    def test_both_cuda_preflights_fail_closed_without_installing(self):
        for workflow, job, condition in (
            ("smoke.yml", "smoke_tests", "inputs.runner == 'gpu-nvidia'"),
            ("product-integration-smoke.yml", "product_integration",
             "inputs.platform == 'linux' && inputs.backend == 'cuda'"),
        ):
            with self.subTest(workflow=workflow):
                data = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
                steps = [step for step in data["jobs"][job]["steps"]
                         if step.get("if") == condition and "run" in step]
                script = "\n".join(step["run"] for step in steps)
                self.assertNotIn("sudo", script)
                self.assertNotIn("apt-get", script)
                for missing in (None, "curl", "jq", "lsof", *LIBRARIES, "nvidia-smi"):
                    with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        calls = root / "calls"
                        tools = root / "bin"
                        tools.mkdir()
                        for tool in ("curl", "jq", "lsof", "nvidia-smi"):
                            if tool == missing and tool != "nvidia-smi":
                                continue
                            path = tools / tool
                            path.write_text('#!/bin/sh\nprintf "%s\\n" "' + tool +
                                            ' $*" >> "$CALLS"\n' +
                                            ('exit 1\n' if tool == missing else 'exit 0\n'))
                            path.chmod(0o755)
                        (tools / "python3").symlink_to(sys.executable)
                        (root / "ctypes.py").write_text(
                            'import os\n'
                            'def CDLL(library):\n'
                            '    with open(os.environ["CALLS"], "a") as calls:\n'
                            '        calls.write(library + "\\n")\n'
                            '    if library == os.environ.get("MISSING"):\n'
                            '        raise OSError("missing library")\n')
                        result = subprocess.run(
                            [shutil.which("bash"), "-c", script], cwd=root,
                            env={**os.environ, "PATH": str(tools), "PYTHONPATH": str(root),
                                 "CALLS": str(calls), "MISSING": missing or ""},
                            capture_output=True, text=True, timeout=10)
                        recorded = calls.read_text().splitlines() if calls.exists() else []
                        if missing is None:
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertEqual(recorded, [*LIBRARIES,
                                "nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader"])
                        else:
                            self.assertNotEqual(result.returncode, 0, result.stdout)
                            if missing in ("curl", "jq", "lsof"):
                                self.assertEqual(recorded, [])
                            elif missing in LIBRARIES:
                                self.assertEqual(recorded, list(LIBRARIES[:LIBRARIES.index(missing) + 1]))
