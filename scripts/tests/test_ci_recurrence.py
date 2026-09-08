"""Execution-level regressions for the PR-green/main-red preparation boundaries."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]


class CiRecurrenceTests(unittest.TestCase):
    def test_ui_candidate_step_executes_checkout_trust_before_version_preparation(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/ci-ui-artifact-slice.yml").read_text())
        steps = workflow["jobs"]["ui_artifact"]["steps"]
        step = next(item for item in steps if "ci-prepare-release-ui.sh" in item.get("run", ""))
        self.assertNotIn("if", step, "ordinary PR/main must exercise release preparation")
        # Git's own ownership-test mode reproduces the container ownership error
        # without root/chown. Only the expensive version rewrite is a test double.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy2(ROOT / "scripts/ci-prepare-release-ui.sh", scripts)
            version = scripts / "release-version.sh"
            version.write_text('#!/usr/bin/env bash\nset -euo pipefail\ngit ls-files > prepared-files\nprintf "%s" "$1" > prepared-version\n')
            version.chmod(0o755)
            env = {**os.environ, "GIT_CONFIG_GLOBAL": str(root / "empty-config"),
                   "GIT_CONFIG_NOSYSTEM": "1", "GITHUB_ENV": str(root / "github-env"),
                   "RUNNER_TEMP": directory}
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, env=env, text=True).strip()
            git("init", "--quiet")
            git("add", "scripts")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture")
            sha = git("rev-parse", "HEAD")
            env.update(UI_SOURCE_SHA=sha, RELEASE_TAG="v99.99.99-ci-rehearsal",
                       GIT_TEST_ASSUME_DIFFERENT_OWNER="1")
            rejected = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, env=env, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0, "ownership reproduction must fail first")
            result = subprocess.run(["bash", "-e", "-c", step["run"]], cwd=root, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("scripts/release-version.sh", (root / "prepared-files").read_text())
            self.assertEqual((root / "prepared-version").read_text(), env["RELEASE_TAG"])
            (root / "prepared-files").unlink()
            (root / "prepared-version").unlink()
            (root / "github-env").unlink()
            env["RELEASE_TAG"] = ""
            result = subprocess.run(["bash", "-e", "-c", step["run"]], cwd=root, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "prepared-version").exists())
            self.assertFalse((root / "github-env").exists(), "rehearsal must not change the real UI mode")
            self.assertEqual(list(root.glob("mesh-release-ui.*")), [])

    def test_product_smoke_exercises_sdk_reader_before_serving(self):
        script = (ROOT / "scripts/ci-smoke-test.sh").read_text()
        self.assertIn("MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK=0 scripts/ci-prepare-native-runtime.sh", script)
        self.assertLess(script.index("scripts/ci-prepare-native-runtime.sh"), script.index("MESH_PID=$!"))
        self.assertIn('--reuse-from-binary "$MESH_LLM"', script)

    def test_gpu_setup_checks_existing_libraries_instead_of_sudo(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/smoke.yml").read_text())
        steps = workflow["jobs"]["smoke_tests"]["steps"]
        runs = "\n".join(step.get("run", "") for step in steps)
        self.assertNotIn("sudo", runs)
        for library in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"):
            self.assertIn(library, runs)
        self.assertIn("ctypes.CDLL(library)", runs)


class CudaSmokeContractTests(unittest.TestCase):
    def test_cuda_row_selects_device_for_all_smoke_steps(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/smoke.yml").read_text())
        job = workflow["jobs"]["smoke_tests"]
        self.assertEqual(job["env"].get("MESH_CI_DEVICE"),
                         "${{ inputs.runner == 'gpu-nvidia' && 'CUDA0' || 'CPU' }}")
        for step in job["steps"]:
            if "ci-smoke-test.sh" in step.get("run", "") or "ci-compat-smoke.sh" in step.get("run", ""):
                self.assertNotIn("MESH_CI_DEVICE", step.get("env", {}))

    def test_every_serve_checks_its_own_native_offload_log(self):
        for filename, pids in (("ci-smoke-test.sh", ("MESH_PID", "HEADLESS_PID")),
                               ("ci-compat-smoke.sh", ("MESH_PID",))):
            with self.subTest(filename=filename):
                script = (ROOT / "scripts" / filename).read_text()
                self.assertNotIn("--device CPU", script)
                self.assertEqual(script.count('    --device "$SMOKE_DEVICE"'), len(pids))
                for pid in pids:
                    self.assertIn(f'"$SMOKE_RUNTIME_ROOT/${pid}/logs/skippy-native.log"', script)
                self.assertEqual(script.count("scripts/ci-verify-smoke-offload.py"), len(pids))


    def test_headless_completion_succeeds_before_its_offload_check(self):
        script = (ROOT / "scripts/ci-smoke-test.sh").read_text()
        # Execute the real final headless block, including set -e behavior,
        # curl flags and jq's nonempty-content gate. Only HTTP and the offload
        # subprocess are doubles; no model/server or GPU is started here.
        block = script[script.index('HEADLESS_ATTESTATION_STATUS='):]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = root / "bin"
            commands.mkdir()
            trace = root / "trace"
            curl = commands / "curl"
            curl.write_text('''#!/usr/bin/env bash
set -eu
case "$*" in
    *'/api/status'*) printf '%s' '{"release_attestation":{"status":"missing"}}' ;;
    *'/v1/chat/completions'*)
        printf 'chat:%s\\n' "$*" >> "$TRACE"
        printf '%s' "$CHAT_RESPONSE"
        exit "$CHAT_EXIT"
        ;;
    *) exit 90 ;;
esac
''')
            python = commands / "python3"
            python.write_text('''#!/usr/bin/env bash
set -eu
if [[ "$1" == scripts/ci-verify-smoke-offload.py ]]; then
    printf 'offload:%s\\n' "$*" >> "$TRACE"
else
    exec "$REAL_PYTHON" "$@"
fi
''')
            curl.chmod(0o755)
            python.chmod(0o755)
            env = {**os.environ, "PATH": f"{commands}:{os.environ['PATH']}",
                   "TRACE": str(trace), "REAL_PYTHON": sys.executable,
                   "HEADLESS_CONSOLE_PORT": "3132", "HEADLESS_API_PORT": "9338",
                   "ATTESTATION_PUBLIC_KEY_FILE": "", "SMOKE_DEVICE": "CUDA0",
                   "SMOKE_RUNTIME_ROOT": str(root / "runtime"), "HEADLESS_PID": "12345",
                   "AUTO_PAYLOAD": '{"model":"auto"}'}
            for response, status, succeeds in (
                ('{"choices":[{"message":{"content":"hello"}}]}', "0", True),
                ('{"choices":[{"message":{"content":""}}]}', "0", False),
                ('{"choices":[]}', "0", False),
                ('not json', "0", False),
                ('{"choices":[{"message":{"content":"hello"}}]}', "22", False),
            ):
                with self.subTest(response=response, status=status):
                    trace.write_text("")
                    result = subprocess.run(["bash", "-euo", "pipefail", "-c", block],
                                            cwd=root, env={**env, "CHAT_RESPONSE": response,
                                                          "CHAT_EXIT": status},
                                            capture_output=True, text=True)
                    events = trace.read_text().splitlines()
                    self.assertTrue(events and events[0].startswith("chat:"), events)
                    self.assertIn("-fsS --max-time 60", events[0])
                    self.assertIn("http://127.0.0.1:9338/v1/chat/completions", events[0])
                    if succeeds:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(len(events), 2, events)
                        self.assertEqual(events[1], "offload:scripts/ci-verify-smoke-offload.py "
                                         f"--device CUDA0 --native-log {root}/runtime/12345/logs/skippy-native.log")
                    else:
                        self.assertNotEqual(result.returncode, 0, result.stdout)
                        self.assertEqual(len(events), 1, "offload must not run after failed inference")
