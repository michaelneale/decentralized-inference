from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PRODUCT_SCRIPT = ROOT / "scripts" / "ci-product-integration-smoke.sh"
REQUIRED_PHASES = [
    "dense-standalone",
    "dense-openai-sdk",
    "dense-constrained-tokio-restart",
    "dense-split-kv",
    "recurrent-split-kv",
]
DENSE_ARTIFACT_ID = "smollm2-q8-inference"
RECURRENT_ARTIFACT_ID = "family-granite-hybrid"
DENSE_FIXTURE = b"dense fixture"
RECURRENT_FIXTURE = b"recurrent fixture"


class ProductIntegrationSmokeTests(unittest.TestCase):
    def write_stub(self, path: Path) -> None:
        if path.name == "ci-two-node-split-smoke.sh":
            path.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
phase="${MESH_PRODUCT_INTEGRATION_PHASE:?missing phase}"
if [[ "${STUB_FAIL_PHASE:-}" == "$phase" ]]; then
    exit 42
fi
evidence_root="${MESH_TWO_NODE_SPLIT_WORK_DIR:?missing evidence root}"
snapshot_root="$evidence_root/split-evidence-snapshots"
mkdir -p "$snapshot_root"
python3 - "$snapshot_root" <<'PY'
import copy
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
stages = [
    {"stage_id": "stage-0", "stage_index": 0, "node_id": "seed-node-0001", "layer_start": 0, "layer_end": 12, "endpoint": {"bind_addr": "127.0.0.1:5501"}},
    {"stage_id": "stage-1", "stage_index": 1, "node_id": "worker-node-0002", "layer_start": 12, "layer_end": 24, "endpoint": {"bind_addr": "127.0.0.1:5502"}},
]
topology = {"topology_id": "topology-a", "run_id": "run-a", "model_id": "model-a", "package_ref": "hf:test/model@revision", "manifest_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "stages": stages}
statuses = [
    {"topology_id": "topology-a", "run_id": "run-a", "model_id": "model-a", "package_ref": "hf:test/model@revision", "manifest_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "stage_id": stage["stage_id"], "stage_index": stage["stage_index"], "node_id": stage["node_id"], "layer_start": stage["layer_start"], "layer_end": stage["layer_end"], "bind_addr": stage["endpoint"]["bind_addr"], "state": "ready"}
    for stage in stages
]
stage_snapshot = {"stages": statuses, "topologies": [topology], "statuses": statuses}
snapshots = {
    "seed-status.json": {"node_id": "seed-node", "mesh_id": "mesh-a", "peers": [{"id": "worker-node"}]},
    "seed-stages.json": stage_snapshot,
    "seed-models.json": {"data": [{"id": "model-a"}]},
    "worker-status.json": {"node_id": "worker-node", "mesh_id": "mesh-a", "peers": [{"id": "seed-node"}]},
    "worker-stages.json": copy.deepcopy(stage_snapshot),
    "worker-models.json": {"data": [{"id": "model-a"}]},
}
for name, payload in snapshots.items():
    (root / name).write_text(json.dumps(payload, sort_keys=True) + "\\n", encoding="utf-8")
PY
python3 scripts/reconcile-two-node-split-evidence.py \
    --seed-status "$snapshot_root/seed-status.json" \
    --seed-stages "$snapshot_root/seed-stages.json" \
    --seed-models "$snapshot_root/seed-models.json" \
    --worker-status "$snapshot_root/worker-status.json" \
    --worker-stages "$snapshot_root/worker-stages.json" \
    --worker-models "$snapshot_root/worker-models.json" \
    --model-label "${MESH_TWO_NODE_SPLIT_MODEL_LABEL:?missing model label}" \
    --output "$evidence_root/split-evidence.json"
case "${STUB_SPLIT_EVIDENCE_MODE:-valid}" in
    valid) ;;
    missing|"missing-$phase") rm -f "$evidence_root/split-evidence.json" ;;
    tampered|"tampered-$phase") printf '{"status":"ready"}\n' >"$evidence_root/split-evidence.json" ;;
    missing-*|tampered-*) ;;
    *) exit 64 ;;
esac
""",
                encoding="utf-8",
            )
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
            return
        # Execute the actual consumer's device selection/guard before the
        # harmless phase stub, so typed-call environment drift cannot hide
        # behind the suite manifest's independently recorded device.
        consumer = (ROOT / "scripts" / path.name).read_text()
        device_setup = consumer[:consumer.index("MESH_LLM=")]
        # Fail explicitly: errexit can be disabled by the shell invocation.
        path.write_text(
            device_setup + '\n[[ "$SMOKE_DEVICE" == "$STUB_EXPECTED_DEVICE" ]] || exit 97\n' +
            """
set -euo pipefail
phase="${MESH_PRODUCT_INTEGRATION_PHASE:?missing phase}"
for log_path in "${MESH_CI_LOG:-}" "${MESH_CI_HEADLESS_LOG:-}" "${MESH_COMPAT_LOG:-}"; do
    if [[ -n "$log_path" ]]; then
        mkdir -p "$(dirname "$log_path")"
        : >"$log_path"
    fi
done
if [[ "${STUB_FAIL_PHASE:-}" == "$phase" ]]; then
    exit 42
fi
""",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def run_suite(
        self,
        failure_phase: str | None = None,
        *,
        platform: str = "linux",
        backend: str = "cpu",
        dense_artifact_id: str = DENSE_ARTIFACT_ID,
        dense_sha256: str | None = None,
        recurrent_artifact_id: str = RECURRENT_ARTIFACT_ID,
        recurrent_sha256: str | None = None,
        split_evidence_mode: str = "valid",
        script_mutation: tuple[str, str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict | None]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy2(PRODUCT_SCRIPT, scripts / PRODUCT_SCRIPT.name)
            shutil.copy2(
                ROOT / "scripts" / "reconcile-two-node-split-evidence.py",
                scripts / "reconcile-two-node-split-evidence.py",
            )
            for name in (
                "ci-smoke-test.sh",
                "ci-compat-smoke.sh",
                "ci-two-node-split-smoke.sh",
            ):
                self.write_stub(scripts / name)

            if script_mutation is not None:
                name, original, replacement = script_mutation
                script = scripts / name
                source = script.read_text()
                self.assertIn(original, source, f"mutation target missing in {name}")
                script.write_text(source.replace(original, replacement, 1))

            binary = root / "mesh-llm"
            binary.touch(mode=0o755)
            artifact_dir = root / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "product-manifest.json").write_text(
                json.dumps({"backend": backend}) + "\n", encoding="utf-8"
            )
            dense_model = root / "dense.gguf"
            recurrent_model = root / "recurrent.gguf"
            dense_model.write_bytes(DENSE_FIXTURE)
            recurrent_model.write_bytes(RECURRENT_FIXTURE)
            if dense_sha256 is None:
                dense_sha256 = hashlib.sha256(DENSE_FIXTURE).hexdigest()
            if recurrent_sha256 is None:
                recurrent_sha256 = hashlib.sha256(RECURRENT_FIXTURE).hexdigest()
            phase_root = root / "phase-evidence"
            env = {
                **os.environ,
                "PATH": "/usr/bin:/bin",
                "MESH_PRODUCT_INTEGRATION_PHASE_ROOT": str(phase_root),
                "STUB_SPLIT_EVIDENCE_MODE": split_evidence_mode,
                "STUB_EXPECTED_DEVICE": {"cpu": "CPU", "cuda": "CUDA0", "metal": "MTL0",
                                         "vulkan": "Vulkan0", "rocm": "ROCm0"}.get(backend, "invalid"),
                # A typed compatibility call must override the generic CPU
                # fallback via MESH_COMPAT_DEVICE; the standalone caller must
                # override this common default via MESH_CI_DEVICE.
                "MESH_CI_DEVICE": "CPU",
                "MESH_COMPAT_DEVICE": "CPU",
            }
            if failure_phase is not None:
                env["STUB_FAIL_PHASE"] = failure_phase

            result = subprocess.run(
                [
                    "bash",
                    "scripts/ci-product-integration-smoke.sh",
                    str(binary),
                    str(artifact_dir),
                    str(dense_model),
                    str(recurrent_model),
                    platform,
                    backend,
                    dense_artifact_id,
                    dense_sha256,
                    recurrent_artifact_id,
                    recurrent_sha256,
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            manifest_path = phase_root / "phase-results.json"
            manifest = (
                json.loads(manifest_path.read_text()) if manifest_path.exists() else None
            )
            return result, manifest

    def test_success_reconciles_the_exact_five_phases(self) -> None:
        result, manifest = self.run_suite()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(manifest["suite_status"], "passed")
        self.assertEqual(
            manifest["provenance"],
            {
                "platform": "linux",
                "backend": "cpu",
                "device": "CPU",
                "product_backend": "cpu",
            },
        )
        self.assertEqual(manifest["required_phases"], REQUIRED_PHASES)
        self.assertEqual(
            [phase["phase"] for phase in manifest["phases"]], REQUIRED_PHASES
        )
        self.assertEqual(
            manifest["reconciliation"],
            {
                "status": "passed",
                "finalized": True,
                "failure_phase": None,
                "missing_phases": [],
                "errors": [],
            },
        )
        for phase in manifest["phases"]:
            with self.subTest(phase=phase["phase"]):
                self.assertEqual(phase["status"], "passed")
                self.assertEqual(phase["exit_code"], 0)
                expected_recurrent = phase["phase"] == "recurrent-split-kv"
                self.assertEqual(
                    phase["model"]["label"],
                    "recurrent" if expected_recurrent else "dense",
                )
                self.assertEqual(
                    phase["model"]["artifact_id"],
                    RECURRENT_ARTIFACT_ID if expected_recurrent else DENSE_ARTIFACT_ID,
                )
                self.assertEqual(
                    phase["model"]["sha256"],
                    hashlib.sha256(
                        RECURRENT_FIXTURE if expected_recurrent else DENSE_FIXTURE
                    ).hexdigest(),
                )
                self.assertEqual(
                    Path(phase["model"]["path"]).name,
                    "recurrent.gguf" if expected_recurrent else "dense.gguf",
                )
                self.assertTrue(phase["workdir"])
                self.assertTrue(phase["log_paths"])
                if phase["phase"] in {"dense-split-kv", "recurrent-split-kv"}:
                    self.assertRegex(
                        phase["split_evidence"]["sha256"], r"^[0-9a-f]{64}$"
                    )
                    self.assertTrue(
                        phase["split_evidence"]["path"].endswith(
                            "split-evidence.json"
                        )
                    )
                else:
                    self.assertIsNone(phase["split_evidence"])
                self.assertLessEqual(
                    phase["started_at_unix_ns"], phase["ended_at_unix_ns"]
                )

    def test_typed_backend_selectors_drive_the_expected_device(self) -> None:
        cases = {
            ("linux", "cpu"): "CPU",
            ("linux", "cuda"): "CUDA0",
            ("linux", "vulkan"): "Vulkan0",
            ("linux", "rocm"): "ROCm0",
            ("macos", "metal"): "MTL0",
        }
        for (platform, backend), expected_device in cases.items():
            with self.subTest(platform=platform, backend=backend):
                result, manifest = self.run_suite(platform=platform, backend=backend)
                self.assertEqual(result.returncode, 0, result.stderr)
                assert manifest is not None
                self.assertEqual(manifest["provenance"]["device"], expected_device)

    def test_device_drift_mutations_fail_the_owning_phase(self) -> None:
        # Mutate only disposable scripts, retaining the real caller and device
        # guards. Exact status/phase assertions exclude unrelated shell failures.
        cases = (
            (
                "standalone caller override",
                (PRODUCT_SCRIPT.name, 'MESH_CI_DEVICE="$DEVICE"', 'MESH_CI_DEVICE="CPU"'),
                "dense-standalone",
            ),
            (
                "compatibility caller override",
                (PRODUCT_SCRIPT.name, 'MESH_COMPAT_DEVICE="$DEVICE"', 'MESH_COMPAT_DEVICE="CPU"'),
                "dense-openai-sdk",
            ),
            (
                "compatibility selector precedence",
                (
                    "ci-compat-smoke.sh",
                    'SMOKE_DEVICE="${MESH_COMPAT_DEVICE:-${MESH_CI_DEVICE:-CPU}}"',
                    'SMOKE_DEVICE="${MESH_CI_DEVICE:-CPU}"',
                ),
                "dense-openai-sdk",
            ),
        )
        for platform, backend in (
            ("linux", "cuda"), ("linux", "vulkan"),
            ("linux", "rocm"), ("macos", "metal"),
        ):
            for name, mutation, failure_phase in cases:
                with self.subTest(backend=backend, mutation=name):
                    result, manifest = self.run_suite(
                        platform=platform, backend=backend, script_mutation=mutation,
                    )
                    self.assertEqual(result.returncode, 97, result.stderr)
                    self.assertIsNotNone(manifest)
                    assert manifest is not None
                    self.assertEqual(manifest["suite_status"], "failed")
                    self.assertEqual(manifest["reconciliation"]["status"], "failed")
                    self.assertEqual(
                        manifest["reconciliation"]["failure_phase"], failure_phase,
                    )
                    self.assertEqual(manifest["phases"][-1]["phase"], failure_phase)
                    self.assertEqual(manifest["phases"][-1]["exit_code"], 97)

    def test_failed_phase_is_recorded_and_reconciliation_fails_closed(self) -> None:
        result, manifest = self.run_suite("dense-openai-sdk")

        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(manifest["suite_status"], "failed")
        self.assertEqual(manifest["reconciliation"]["status"], "failed")
        self.assertTrue(manifest["reconciliation"]["finalized"])
        self.assertEqual(
            manifest["reconciliation"]["failure_phase"], "dense-openai-sdk"
        )
        self.assertEqual(
            [phase["phase"] for phase in manifest["phases"]],
            ["dense-standalone", "dense-openai-sdk"],
        )
        failed = manifest["phases"][-1]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["exit_code"], 42)
        self.assertIn(
            "missing required phase: dense-constrained-tokio-restart",
            manifest["reconciliation"]["errors"],
        )
        self.assertIn(
            "phase did not pass: dense-openai-sdk",
            manifest["reconciliation"]["errors"],
        )

    def test_fixture_identity_rejects_missing_malformed_swapped_and_unverified_inputs(self) -> None:
        cases = {
            "missing": ({"dense_sha256": ""}, "Usage:"),
            "malformed": (
                {"dense_sha256": "not-a-sha"},
                "invalid dense fixture SHA-256",
            ),
            "swapped": (
                {"dense_artifact_id": RECURRENT_ARTIFACT_ID},
                "unexpected dense fixture artifact id",
            ),
            "unverified": ({"dense_sha256": "0" * 64}, "dense fixture digest mismatch"),
        }

        for name, (kwargs, expected_error) in cases.items():
            with self.subTest(name=name):
                result, manifest = self.run_suite(**kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)
                self.assertIsNone(manifest)

    def test_missing_or_tampered_split_evidence_fails_the_owning_phase(self) -> None:
        cases = {
            "missing-dense-split-kv": "dense-split-kv",
            "tampered-recurrent-split-kv": "recurrent-split-kv",
        }
        for mode, failure_phase in cases.items():
            with self.subTest(mode=mode):
                result, manifest = self.run_suite(split_evidence_mode=mode)
                self.assertEqual(result.returncode, 71, result.stderr)
                self.assertIsNotNone(manifest)
                assert manifest is not None
                self.assertEqual(manifest["suite_status"], "failed")
                self.assertEqual(
                    manifest["reconciliation"]["failure_phase"], failure_phase
                )
                self.assertEqual(manifest["phases"][-1]["status"], "failed")
                self.assertIsNone(manifest["phases"][-1]["split_evidence"])


if __name__ == "__main__":
    unittest.main()
