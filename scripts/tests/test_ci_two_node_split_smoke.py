import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = ROOT / "scripts/ci-two-node-split-smoke.sh"
PROMPT_COUNTS = [644, 644, 788, 788, 916, 916]


class WorkDirCreationTests(unittest.TestCase):
    def test_external_work_dir_is_created_before_logs_are_opened(self) -> None:
        """An externally supplied MESH_TWO_NODE_SPLIT_WORK_DIR may be a fresh
        nested evidence path; the smoke must mkdir -p it before opening any
        log file (the mktemp default self-creates, the override does not)."""
        script = SMOKE_SCRIPT.read_text(encoding="utf-8")
        work_dir_line = script.index('WORK_DIR="${MESH_TWO_NODE_SPLIT_WORK_DIR:-')
        mkdir_line = script.index('mkdir -p "$WORK_DIR"')
        seed_log_line = script.index('SEED_LOG="${WORK_DIR}/')
        worker_log_line = script.index('WORKER_LOG="${WORK_DIR}/')
        self.assertLess(work_dir_line, mkdir_line)
        self.assertLess(mkdir_line, seed_log_line)
        self.assertLess(mkdir_line, worker_log_line)
        # And behaviorally: run the script far enough to observe the directory
        # exists even when nothing else is mocked (script exits early on a
        # missing binary, but only after creating WORK_DIR).
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory) / "nested" / "evidence" / "split-work"
            self.assertFalse(work_dir.exists())
            env = {
                **os.environ,
                "MESH_TWO_NODE_SPLIT_WORK_DIR": str(work_dir),
                "MESH_LLM_BIN": str(Path(directory) / "nonexistent-mesh-llm"),
            }
            subprocess.run(
                ["bash", str(SMOKE_SCRIPT), str(Path(directory) / "nonexistent-mesh-llm"),
                 "/bin", str(Path(directory) / "nonexistent-model.gguf")],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertTrue(
                work_dir.is_dir(),
                "smoke must create an externally supplied WORK_DIR before use",
            )


def prefix_validator_source() -> str:
    script = SMOKE_SCRIPT.read_text()
    function = script[script.index("validate_prefix_responses() {") :]
    match = re.search(r"<<'PY'\n(?P<source>.*?)\nPY\n}", function, re.DOTALL)
    if match is None:
        raise AssertionError("could not extract split-prefix response validator")
    return match.group("source")


def shell_function_block(first: str, following: str) -> str:
    script = SMOKE_SCRIPT.read_text(encoding="utf-8")
    return script[
        script.index(f"{first}() {{") : script.index(f"{following}() {{")
    ]


def write_runtime_manifest(
    runtime_bundle: Path,
    runtime_id: str = "runtime-id",
    *,
    tool_rel: str = "tools/skippy-model-package",
    tool_contents: bytes = b"#!/usr/bin/env bash\nexit 0\n",
    executable: bool = True,
    manifest_tool_rel: str | None = None,
) -> tuple[Path, Path]:
    runtime_dir = runtime_bundle / runtime_id
    tool = runtime_dir / tool_rel
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_bytes(tool_contents)
    if executable:
        tool.chmod(0o755)
    declared_rel = manifest_tool_rel or tool_rel
    manifest = {
        "runtime": {
            "tools": {
                declared_rel: hashlib.sha256(tool_contents).hexdigest(),
            }
        }
    }
    (runtime_dir / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return runtime_dir, tool


class SnapshotHandler(BaseHTTPRequestHandler):
    response: dict | None = None
    delay_seconds = 0.0

    def do_GET(self) -> None:
        time.sleep(self.delay_seconds)
        payload = json.dumps(self.response or {}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format: str, *_args: object) -> None:
        pass


class TwoNodeSplitSmokeTests(unittest.TestCase):
    def run_validator(
        self, cached_counts: list[int]
    ) -> subprocess.CompletedProcess[str]:
        self.assertEqual(len(cached_counts), len(PROMPT_COUNTS))
        with tempfile.TemporaryDirectory() as directory:
            response_dir = Path(directory)
            for index, (prompt_tokens, cached_tokens) in enumerate(
                zip(PROMPT_COUNTS, cached_counts), start=1
            ):
                response = {
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    },
                }
                (response_dir / f"response-{index}.json").write_text(
                    json.dumps(response)
                )

            return subprocess.run(
                [sys.executable, "-", directory, "6", "kv-recurrent"],
                input=prefix_validator_source(),
                text=True,
                capture_output=True,
                check=False,
            )

    def test_prefix_validator_exit_behavior(self):
        cases = {
            "second request misses": ([0, 0, 512, 640, 640, 768], 1),
            "recurrent checkpoint reuse grows": ([0, 512, 512, 640, 640, 768], 0),
            "recurrent longer first sights may be cold": (
                [0, 512, 0, 640, 0, 768],
                0,
            ),
            "later repeated request misses": ([0, 512, 512, 640, 640, 0], 1),
            "all follow-up requests miss": ([0, 0, 0, 0, 0, 0], 75),
        }

        for name, (cached_counts, expected_exit) in cases.items():
            with self.subTest(name=name):
                result = self.run_validator(cached_counts)
                self.assertEqual(
                    result.returncode,
                    expected_exit,
                    msg=f"stdout={result.stdout!r}\nstderr={result.stderr!r}",
                )

    def test_readiness_reconciles_persisted_snapshots_from_both_observers(self):
        script = SMOKE_SCRIPT.read_text(encoding="utf-8")

        self.assertEqual(script.count('wait_for_split_topology "'), 2)
        for observer in ("seed", "worker"):
            for snapshot in ("status", "stages", "models"):
                self.assertIn(f"{observer}-{snapshot}.json", script)
        self.assertIn('"mesh_id": payload.get("mesh_id")', script)
        self.assertNotIn('payload["token"] = "[redacted]"', script)
        self.assertIn("scripts/reconcile-two-node-split-evidence.py", script)
        self.assertIn(
            'SPLIT_EVIDENCE_PATH="${WORK_DIR}/${prefix}split-evidence.json"', script
        )
        self.assertIn("report_split_readiness_failure", script)
        self.assertIn("seed log tail at timeout", script)
        self.assertIn("worker log tail at timeout", script)
        self.assertNotIn('seq 1 "$MAX_WAIT"', script)
        self.assertIn('deadline=$((started_at + READINESS_TIMEOUT_SECONDS))', script)
        workflow = (
            ROOT / ".github/workflows/product-integration-smoke.yml"
        ).read_text()
        self.assertIn("if: success() || failure()", workflow)

    def test_raw_gguf_is_prepared_as_verified_package_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Fixture-Q4_K_M.gguf"
            source.write_bytes(b"immutable-gguf-fixture")
            calls = root / "calls.log"
            package_tool = root / "skippy-model-package"
            package_tool.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$CALLS_LOG"
if [[ "$1" == write-package ]]; then
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == --out-dir ]]; then
      mkdir -p "$2"
      printf '{"schema_version":2}\\n' > "$2/model-package.json"
      exit 0
    fi
    shift
  done
fi
[[ "$1" == verify-package-v2 ]]
""",
                encoding="utf-8",
            )
            package_tool.chmod(0o755)
            harness = (
                "set -euo pipefail\n"
                + shell_function_block("sha256_file", "quant_selector_from_gguf_file")
                + shell_function_block("quant_selector_from_gguf_file", "resolve_package_tool")
                + shell_function_block("prepare_split_package", "descendant_pids")
                + f'WORK_DIR="{root / "work"}"\n'
                + f'export CALLS_LOG="{calls}"\n'
                + f'prepare_split_package dense "{source}" "{package_tool}"\n'
            )
            result = subprocess.run(
                ["bash", "-s"],
                input=harness,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            package_dir = root / "work/prepared-packages/dense"
            self.assertEqual(result.stdout.strip(), str(package_dir))
            invocation = calls.read_text(encoding="utf-8")
            self.assertIn(f"write-package {source}", invocation)
            self.assertIn("--model-id ci/dense-", invocation)
            self.assertIn(":Q4_K_M", invocation)
            self.assertIn("--source-revision", invocation)
            self.assertIn(f"verify-package-v2 {package_dir}", invocation)

    def test_existing_package_v2_is_passed_through(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package_dir = Path(directory) / "package"
            package_dir.mkdir()
            (package_dir / "model-package.json").write_text(
                '{"schema_version":2}\n', encoding="utf-8"
            )
            harness = (
                "set -euo pipefail\n"
                + shell_function_block("prepare_split_package", "descendant_pids")
                + f'WORK_DIR="{Path(directory) / "work"}"\n'
                + f'prepare_split_package dense "{package_dir}" /missing/tool\n'
            )
            result = subprocess.run(
                ["bash", "-s"],
                input=harness,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(package_dir))

    def test_package_tool_is_resolved_from_verified_runtime_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_bundle = root / "runtime"
            _, tool = write_runtime_manifest(runtime_bundle)
            harness = (
                "set -euo pipefail\n"
                + shell_function_block("resolve_package_tool", "prepare_split_package")
                + f'RUNTIME_BUNDLE="{runtime_bundle}"\n'
                + "resolve_package_tool\n"
            )
            result = subprocess.run(
                ["bash", "-s"],
                input=harness,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(tool.resolve()))

    def test_package_tool_must_be_declared_at_exact_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_bundle = Path(directory) / "runtime"
            write_runtime_manifest(
                runtime_bundle,
                tool_rel="other/skippy-model-package",
                manifest_tool_rel="other/skippy-model-package",
            )
            harness = (
                "set -euo pipefail\n"
                + shell_function_block("resolve_package_tool", "prepare_split_package")
                + f'RUNTIME_BUNDLE="{runtime_bundle}"\n'
                + "resolve_package_tool\n"
            )
            result = subprocess.run(
                ["bash", "-s"],
                input=harness,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not declare tools/skippy-model-package", result.stderr)

    def test_package_tool_checksum_is_verified_before_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_bundle = Path(directory) / "runtime"
            _, tool = write_runtime_manifest(runtime_bundle)
            tool.write_bytes(b"tampered\n")
            tool.chmod(0o755)
            harness = (
                "set -euo pipefail\n"
                + shell_function_block("resolve_package_tool", "prepare_split_package")
                + f'RUNTIME_BUNDLE="{runtime_bundle}"\n'
                + "resolve_package_tool\n"
            )
            result = subprocess.run(
                ["bash", "-s"],
                input=harness,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checksum mismatch", result.stderr)

    def test_package_tool_rejects_ambiguous_runtime_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_bundle = Path(directory) / "runtime"
            write_runtime_manifest(runtime_bundle, "runtime-a")
            write_runtime_manifest(runtime_bundle, "runtime-b")
            harness = (
                "set -euo pipefail\n"
                + shell_function_block("resolve_package_tool", "prepare_split_package")
                + f'RUNTIME_BUNDLE="{runtime_bundle}"\n'
                + "resolve_package_tool\n"
            )
            result = subprocess.run(
                ["bash", "-s"],
                input=harness,
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected exactly one native runtime manifest", result.stderr)

    def test_package_tool_has_no_consumer_build_fallback(self) -> None:
        script = SMOKE_SCRIPT.read_text(encoding="utf-8")
        start = script.index("resolve_package_tool()")
        end = script.index("prepare_split_package()", start)
        resolver = script[start:end]
        self.assertNotIn("cargo build", resolver)
        self.assertIn("$RUNTIME_BUNDLE", resolver)

    def test_status_snapshot_is_strict_identity_whitelist(self) -> None:
        SnapshotHandler.delay_seconds = 0
        SnapshotHandler.response = {
            "node_id": "seed-node",
            "mesh_id": "mesh-a",
            "token": "top-level-secret",
            "storage_path": "/private/model/path",
            "nested": {"api_key": "nested-secret", "path": "/nested/path"},
            "peers": [
                {
                    "id": "worker-node",
                    "token": "peer-secret",
                    "materialized_path": "/peer/path",
                }
            ],
        }
        server = ThreadingHTTPServer(("127.0.0.1", 0), SnapshotHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "status.json"
                harness = (
                    "set -euo pipefail\n"
                    + shell_function_block(
                        "capture_json_snapshot", "capture_split_snapshots"
                    )
                    + (
                        "capture_json_snapshot status "
                        f'"http://127.0.0.1:{server.server_port}/api/status" '
                        f'"{output}" 2\n'
                    )
                )
                result = subprocess.run(
                    ["bash", "-s"],
                    input=harness,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                raw = output.read_text(encoding="utf-8")
                self.assertEqual(
                    json.loads(raw),
                    {
                        "mesh_id": "mesh-a",
                        "node_id": "seed-node",
                        "peers": [{"id": "worker-node"}],
                    },
                )
                for forbidden in (
                    "top-level-secret",
                    "nested-secret",
                    "peer-secret",
                    "/private/model/path",
                    "/nested/path",
                    "/peer/path",
                ):
                    self.assertNotIn(forbidden, raw)
        finally:
            server.shutdown()
            server.server_close()

    def test_hung_snapshot_endpoints_obey_wall_clock_deadline(self) -> None:
        SnapshotHandler.delay_seconds = 10
        SnapshotHandler.response = {}
        server = ThreadingHTTPServer(("127.0.0.1", 0), SnapshotHandler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                harness = f"""set -euo pipefail
{shell_function_block("configure_split_evidence_paths", "start_node")}
WORK_DIR={root!s}
PRIMARY_MODEL_LABEL=dense
MODEL_LABEL=dense
READINESS_TIMEOUT_SECONDS=2
SNAPSHOT_REQUEST_TIMEOUT_SECONDS=1
SEED_API_PORT={server.server_port}
SEED_CONSOLE_PORT={server.server_port}
WORKER_API_PORT={server.server_port}
WORKER_CONSOLE_PORT={server.server_port}
SEED_PID=$$
WORKER_PID=$$
SEED_LOG="$WORK_DIR/seed.log"
WORKER_LOG="$WORK_DIR/worker.log"
: >"$SEED_LOG"
: >"$WORKER_LOG"
wait_for_split_topology ""
"""
                started = time.monotonic()
                result = subprocess.run(
                    ["bash", "-s"],
                    input=harness,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=5,
                )
                elapsed = time.monotonic() - started
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertLess(elapsed, 4)
                evidence = json.loads(
                    (root / "split-evidence.json").read_text(encoding="utf-8")
                )
                self.assertEqual(evidence["status"], "failed")
                self.assertIn("timed out after 2s", result.stderr)
                self.assertEqual(
                    len(list((root / "split-evidence-snapshots").glob("*.json"))),
                    6,
                )
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
