import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
