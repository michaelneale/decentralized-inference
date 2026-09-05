from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RECONCILER = ROOT / "scripts" / "reconcile-two-node-split-evidence.py"
MANIFEST_SHA256 = "a" * 64


def valid_snapshots() -> dict[str, dict]:
    stages = [
        {
            "stage_id": "stage-0",
            "stage_index": 0,
            "node_id": "seed-node-0001",
            "layer_start": 0,
            "layer_end": 12,
            "endpoint": {"bind_addr": "127.0.0.1:5501"},
        },
        {
            "stage_id": "stage-1",
            "stage_index": 1,
            "node_id": "worker-node-0002",
            "layer_start": 12,
            "layer_end": 24,
            "endpoint": {"bind_addr": "127.0.0.1:5502"},
        },
    ]
    topology = {
        "topology_id": "topology-a",
        "run_id": "run-a",
        "model_id": "model-a",
        "package_ref": "hf:test/model@revision",
        "manifest_sha256": MANIFEST_SHA256,
        "stages": stages,
    }
    statuses = [
        {
            "topology_id": "topology-a",
            "run_id": "run-a",
            "model_id": "model-a",
            "package_ref": "hf:test/model@revision",
            "manifest_sha256": MANIFEST_SHA256,
            "stage_id": stage["stage_id"],
            "stage_index": stage["stage_index"],
            "node_id": stage["node_id"],
            "layer_start": stage["layer_start"],
            "layer_end": stage["layer_end"],
            "bind_addr": stage["endpoint"]["bind_addr"],
            "state": "ready",
        }
        for stage in stages
    ]
    stage_snapshot = {
        "stages": statuses,
        "topologies": [topology],
        "statuses": statuses,
    }
    worker_stage_snapshot = copy.deepcopy(stage_snapshot)
    worker_stage_snapshot["topologies"][0]["manifest_sha256"] = (
        MANIFEST_SHA256.upper()
    )
    for status in worker_stage_snapshot["statuses"]:
        status["manifest_sha256"] = MANIFEST_SHA256.upper()
    return {
        "seed_status": {
            "node_id": "seed-node",
            "mesh_id": "mesh-a",
            "peers": [{"id": "worker-node"}],
        },
        "seed_stages": copy.deepcopy(stage_snapshot),
        "seed_models": {"object": "list", "data": [{"id": "model-a"}]},
        "worker_status": {
            "node_id": "worker-node",
            "mesh_id": "mesh-a",
            "peers": [{"id": "seed-node"}],
        },
        "worker_stages": worker_stage_snapshot,
        "worker_models": {"object": "list", "data": [{"id": "model-a"}]},
    }


def update_both_statuses(
    snapshots: dict[str, dict], stage_index: int, **values: object
) -> None:
    for observer in ("seed_stages", "worker_stages"):
        snapshots[observer]["statuses"][stage_index].update(values)


class ReconcileTwoNodeSplitEvidenceTests(unittest.TestCase):
    def write_snapshots(
        self, root: Path, snapshots: dict[str, dict]
    ) -> dict[str, Path]:
        paths = {}
        for name, payload in snapshots.items():
            path = root / f"{name.replace('_', '-')}.json"
            path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            paths[name] = path
        return paths

    def command(self, paths: dict[str, Path], mode: str, target: Path) -> list[str]:
        command = [sys.executable, str(RECONCILER)]
        for name, path in paths.items():
            command.extend((f"--{name.replace('_', '-')}", str(path)))
        command.extend(("--model-label", "dense", mode, str(target)))
        return command

    def test_valid_snapshots_emit_atomic_ready_evidence_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_snapshots(root, valid_snapshots())
            evidence_path = root / "split-evidence.json"

            result = subprocess.run(
                self.command(paths, "--output", evidence_path),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "ready")
            self.assertEqual(evidence["model_id"], "model-a")
            self.assertEqual(evidence["topology"]["run_id"], "run-a")
            self.assertEqual(
                evidence["topology"]["package_ref"], "hf:test/model@revision"
            )
            self.assertEqual(
                evidence["topology"]["manifest_sha256"], MANIFEST_SHA256
            )
            self.assertEqual(evidence["topology"]["layer_start"], 0)
            self.assertEqual(evidence["topology"]["layer_end"], 24)
            self.assertEqual(len(evidence["topology"]["stages"]), 2)
            self.assertEqual(
                evidence["topology"]["stages"][0]["endpoint"]["bind_addr"],
                "127.0.0.1:5501",
            )
            self.assertEqual(evidence["observers"]["mesh_id"], "mesh-a")
            self.assertEqual(evidence["observers"]["seed"]["node_id"], "seed-node")
            self.assertFalse(list(root.glob(".split-evidence.json.*.tmp")))

            verified = subprocess.run(
                self.command(paths, "--verify", evidence_path),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_mismatches_fail_closed_with_persisted_failure_evidence(self) -> None:
        cases = {
            "topology run mismatch": lambda data: data["worker_stages"][
                "topologies"
            ][0].update(run_id="run-b"),
            "more than two stages": lambda data: data["seed_stages"][
                "topologies"
            ][0]["stages"].append(
                copy.deepcopy(
                    data["seed_stages"]["topologies"][0]["stages"][1]
                )
            ),
            "noncontiguous ranges": lambda data: data["seed_stages"]["topologies"][
                0
            ]["stages"][1].update(layer_start=13),
            "same observer": lambda data: data["worker_status"].update(
                node_id="seed-node"
            ),
            "nonready status": lambda data: data["worker_stages"]["statuses"][
                1
            ].update(state="starting"),
            "model mismatch": lambda data: data["worker_models"]["data"][0].update(
                id="model-b"
            ),
            "package mismatch between observers": lambda data: data[
                "worker_stages"
            ]["topologies"][0].update(package_ref="hf:test/other@revision"),
            "endpoint mismatch between observers": lambda data: data[
                "worker_stages"
            ]["topologies"][0]["stages"][1]["endpoint"].update(
                bind_addr="127.0.0.1:5599"
            ),
            "manifest mismatch between topology and agreed statuses": lambda data: (
                update_both_statuses(data, 0, manifest_sha256="b" * 64)
            ),
            "package mismatch between topology and agreed statuses": lambda data: (
                update_both_statuses(data, 0, package_ref="hf:test/other@revision")
            ),
            "bind mismatch between topology and agreed statuses": lambda data: (
                update_both_statuses(data, 1, bind_addr="127.0.0.1:5599")
            ),
            "mesh mismatch between observers": lambda data: data[
                "worker_status"
            ].update(mesh_id="mesh-b"),
            "missing status package": lambda data: data["worker_stages"][
                "statuses"
            ][0].pop("package_ref"),
        }

        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                snapshots = valid_snapshots()
                mutate(snapshots)
                paths = self.write_snapshots(root, snapshots)
                evidence_path = root / "split-evidence.json"
                result = subprocess.run(
                    self.command(paths, "--output", evidence_path),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 1)
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                self.assertEqual(evidence["status"], "failed")
                self.assertTrue(evidence["errors"])

    def test_missing_snapshot_and_tampered_snapshot_cannot_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_snapshots(root, valid_snapshots())
            evidence_path = root / "split-evidence.json"
            created = subprocess.run(
                self.command(paths, "--output", evidence_path),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(created.returncode, 0, created.stderr)

            paths["worker_models"].write_text(
                '{"data":[{"id":"model-b"}]}\n', encoding="utf-8"
            )
            tampered = subprocess.run(
                self.command(paths, "--verify", evidence_path),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(tampered.returncode, 1)

            paths["worker_models"].unlink()
            missing = subprocess.run(
                self.command(paths, "--output", root / "failed-evidence.json"),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(missing.returncode, 1)
            failed = json.loads((root / "failed-evidence.json").read_text())
            self.assertEqual(failed["status"], "failed")


if __name__ == "__main__":
    unittest.main()
