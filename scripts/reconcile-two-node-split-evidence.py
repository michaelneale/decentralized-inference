#!/usr/bin/env python3
"""Reconcile persisted two-node split-serving readiness snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


KIND = "mesh-llm-two-node-split-readiness"
SNAPSHOT_ARGUMENTS = (
    "seed_status",
    "seed_stages",
    "seed_models",
    "worker_status",
    "worker_stages",
    "worker_models",
)


class ReconciliationError(ValueError):
    pass


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReconciliationError(f"{label} must be a JSON object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReconciliationError(f"{label} must be a JSON array")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReconciliationError(f"{label} must be a non-empty string")
    return value


def require_sha256(value: Any, label: str) -> str:
    digest = require_string(value, label).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ReconciliationError(f"{label} must be a 64-character SHA-256")
    return digest


def require_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ReconciliationError(f"{label} must be a non-negative integer")
    return value


def load_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReconciliationError(
            f"cannot read {label} snapshot {path}: {error}"
        ) from error
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReconciliationError(
            f"invalid JSON in {label} snapshot {path}: {error}"
        ) from error
    return require_object(payload, label), digest


def canonical_stage(stage: Any, label: str) -> dict[str, Any]:
    value = require_object(stage, label)
    endpoint = require_object(value.get("endpoint"), f"{label}.endpoint")
    return {
        "stage_id": require_string(value.get("stage_id"), f"{label}.stage_id"),
        "stage_index": require_nonnegative_int(
            value.get("stage_index"), f"{label}.stage_index"
        ),
        "node_id": require_string(value.get("node_id"), f"{label}.node_id"),
        "layer_start": require_nonnegative_int(
            value.get("layer_start"), f"{label}.layer_start"
        ),
        "layer_end": require_nonnegative_int(
            value.get("layer_end"), f"{label}.layer_end"
        ),
        "endpoint": {
            "bind_addr": require_string(
                endpoint.get("bind_addr"), f"{label}.endpoint.bind_addr"
            )
        },
    }


def canonical_topology(payload: dict[str, Any], label: str) -> dict[str, Any]:
    topologies = require_list(payload.get("topologies"), f"{label}.topologies")
    if len(topologies) != 1:
        raise ReconciliationError(
            f"{label}.topologies must contain exactly one topology, got {len(topologies)}"
        )
    topology = require_object(topologies[0], f"{label}.topologies[0]")
    stages = require_list(topology.get("stages"), f"{label}.topologies[0].stages")
    if len(stages) != 2:
        raise ReconciliationError(
            f"{label} topology must contain exactly two stages, got {len(stages)}"
        )
    canonical_stages = sorted(
        (
            canonical_stage(stage, f"{label}.topologies[0].stages[{index}]")
            for index, stage in enumerate(stages)
        ),
        key=lambda stage: stage["stage_index"],
    )
    if [stage["stage_index"] for stage in canonical_stages] != [0, 1]:
        raise ReconciliationError(
            f"{label} topology stage indexes must be exactly [0, 1]"
        )
    if canonical_stages[0]["layer_start"] != 0:
        raise ReconciliationError(f"{label} topology must start at layer 0")
    if any(stage["layer_start"] >= stage["layer_end"] for stage in canonical_stages):
        raise ReconciliationError(
            f"{label} topology stages must have non-empty layer ranges"
        )
    if canonical_stages[0]["layer_end"] != canonical_stages[1]["layer_start"]:
        raise ReconciliationError(f"{label} topology layer ranges must be contiguous")
    if len({stage["stage_id"] for stage in canonical_stages}) != 2:
        raise ReconciliationError(f"{label} topology stage IDs must be distinct")
    if len({stage["node_id"] for stage in canonical_stages}) != 2:
        raise ReconciliationError(f"{label} topology stage nodes must be distinct")
    return {
        "topology_id": require_string(
            topology.get("topology_id"), f"{label}.topologies[0].topology_id"
        ),
        "run_id": require_string(
            topology.get("run_id"), f"{label}.topologies[0].run_id"
        ),
        "model_id": require_string(
            topology.get("model_id"), f"{label}.topologies[0].model_id"
        ),
        "package_ref": require_string(
            topology.get("package_ref"), f"{label}.topologies[0].package_ref"
        ),
        "manifest_sha256": require_sha256(
            topology.get("manifest_sha256"),
            f"{label}.topologies[0].manifest_sha256",
        ),
        "stages": canonical_stages,
    }


def canonical_statuses(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    statuses = require_list(payload.get("statuses"), f"{label}.statuses")
    if len(statuses) != 2:
        raise ReconciliationError(
            f"{label}.statuses must contain exactly two stage statuses, got {len(statuses)}"
        )
    canonical = []
    for index, status in enumerate(statuses):
        item_label = f"{label}.statuses[{index}]"
        value = require_object(status, item_label)
        canonical.append(
            {
                "topology_id": require_string(
                    value.get("topology_id"), f"{item_label}.topology_id"
                ),
                "run_id": require_string(value.get("run_id"), f"{item_label}.run_id"),
                "model_id": require_string(
                    value.get("model_id"), f"{item_label}.model_id"
                ),
                "package_ref": require_string(
                    value.get("package_ref"), f"{item_label}.package_ref"
                ),
                "manifest_sha256": require_sha256(
                    value.get("manifest_sha256"),
                    f"{item_label}.manifest_sha256",
                ),
                "stage_id": require_string(
                    value.get("stage_id"), f"{item_label}.stage_id"
                ),
                "stage_index": require_nonnegative_int(
                    value.get("stage_index"), f"{item_label}.stage_index"
                ),
                "node_id": require_string(
                    value.get("node_id"), f"{item_label}.node_id"
                ),
                "layer_start": require_nonnegative_int(
                    value.get("layer_start"), f"{item_label}.layer_start"
                ),
                "layer_end": require_nonnegative_int(
                    value.get("layer_end"), f"{item_label}.layer_end"
                ),
                "bind_addr": require_string(
                    value.get("bind_addr"), f"{item_label}.bind_addr"
                ),
                "state": require_string(value.get("state"), f"{item_label}.state"),
            }
        )
    canonical.sort(key=lambda status: status["stage_index"])
    if [status["stage_index"] for status in canonical] != [0, 1]:
        raise ReconciliationError(f"{label} status indexes must be exactly [0, 1]")
    if any(status["state"] != "ready" for status in canonical):
        states = [status["state"] for status in canonical]
        raise ReconciliationError(
            f"{label} stage statuses must both be ready, got {states}"
        )
    return canonical


def canonical_model_id(payload: dict[str, Any], label: str) -> str:
    models = require_list(payload.get("data"), f"{label}.data")
    concrete_models = []
    for index, model in enumerate(models):
        value = require_object(model, f"{label}.data[{index}]")
        model_id = require_string(value.get("id"), f"{label}.data[{index}].id")
        if model_id != "mesh":
            concrete_models.append(model_id)
    if len(concrete_models) != 1:
        raise ReconciliationError(
            f"{label}.data must contain exactly one concrete model, "
            f"got {len(concrete_models)}"
        )
    return concrete_models[0]


def observer_identity(status: dict[str, Any], label: str) -> tuple[str, str]:
    return (
        require_string(status.get("node_id"), f"{label}.node_id"),
        require_string(status.get("mesh_id"), f"{label}.mesh_id"),
    )


def peer_node_ids(status: dict[str, Any], label: str) -> list[str]:
    peers = require_list(status.get("peers"), f"{label}.peers")
    peer_ids = [
        require_string(
            require_object(peer, f"{label}.peers[{index}]").get("id"),
            f"{label}.peers[{index}].id",
        )
        for index, peer in enumerate(peers)
    ]
    if len(peer_ids) != 1:
        raise ReconciliationError(
            f"{label}.peers must contain exactly one peer, got {len(peer_ids)}"
        )
    return peer_ids


def short_id_matches(full_id: str, short_id: str) -> bool:
    return full_id == short_id or full_id.startswith(short_id)


def reconcile(
    snapshots: dict[str, dict[str, Any]],
    digests: dict[str, str],
    paths: dict[str, Path],
    model_label: str,
) -> dict[str, Any]:
    seed_node, seed_mesh = observer_identity(
        snapshots["seed_status"], "seed_status"
    )
    worker_node, worker_mesh = observer_identity(
        snapshots["worker_status"], "worker_status"
    )
    if seed_node == worker_node:
        raise ReconciliationError("seed and worker observers must have distinct node IDs")
    if seed_mesh != worker_mesh:
        raise ReconciliationError("seed and worker observers must share the same mesh ID")

    seed_peers = peer_node_ids(snapshots["seed_status"], "seed_status")
    worker_peers = peer_node_ids(snapshots["worker_status"], "worker_status")
    if seed_peers != [worker_node] or worker_peers != [seed_node]:
        raise ReconciliationError(
            "seed and worker status snapshots must identify each other as their sole peer"
        )

    seed_topology = canonical_topology(snapshots["seed_stages"], "seed_stages")
    worker_topology = canonical_topology(snapshots["worker_stages"], "worker_stages")
    if seed_topology != worker_topology:
        raise ReconciliationError("seed and worker topology snapshots do not match exactly")

    seed_statuses = canonical_statuses(snapshots["seed_stages"], "seed_stages")
    worker_statuses = canonical_statuses(snapshots["worker_stages"], "worker_stages")
    if seed_statuses != worker_statuses:
        raise ReconciliationError(
            "seed and worker stage status snapshots do not match exactly"
        )

    topology_key = (
        seed_topology["topology_id"],
        seed_topology["run_id"],
        seed_topology["model_id"],
        seed_topology["package_ref"],
        seed_topology["manifest_sha256"],
    )
    topology_stages = []
    for stage, status in zip(seed_topology["stages"], seed_statuses):
        status_key = (
            status["topology_id"],
            status["run_id"],
            status["model_id"],
            status["package_ref"],
            status["manifest_sha256"],
        )
        if status_key != topology_key:
            raise ReconciliationError(
                f"stage status {status['stage_id']} does not match the common "
                "topology/run/model/package/manifest"
            )
        for field in ("stage_id", "stage_index", "node_id", "layer_start", "layer_end"):
            if status[field] != stage[field]:
                raise ReconciliationError(
                    f"stage status {status['stage_id']} does not match topology field {field}"
                )
        if status["bind_addr"] != stage["endpoint"]["bind_addr"]:
            raise ReconciliationError(
                f"stage status {status['stage_id']} does not match topology bind_addr"
            )
        topology_stages.append({**stage, "state": status["state"]})

    stage_node_ids = [stage["node_id"] for stage in seed_topology["stages"]]
    for observer, node_id in (("seed", seed_node), ("worker", worker_node)):
        matches = [
            full_id
            for full_id in stage_node_ids
            if short_id_matches(full_id, node_id)
        ]
        if len(matches) != 1:
            raise ReconciliationError(
                f"{observer} observer node ID must match exactly one topology stage node"
            )

    seed_model = canonical_model_id(snapshots["seed_models"], "seed_models")
    worker_model = canonical_model_id(snapshots["worker_models"], "worker_models")
    if seed_model != worker_model or seed_model != seed_topology["model_id"]:
        raise ReconciliationError(
            "seed, worker, and topology model IDs must match exactly"
        )

    return {
        "schema_version": 1,
        "kind": KIND,
        "status": "ready",
        "model_label": model_label,
        "model_id": seed_model,
        "topology": {
            **{
                key: seed_topology[key]
                for key in (
                    "topology_id",
                    "run_id",
                    "model_id",
                    "package_ref",
                    "manifest_sha256",
                )
            },
            "layer_start": topology_stages[0]["layer_start"],
            "layer_end": topology_stages[-1]["layer_end"],
            "stages": topology_stages,
        },
        "observers": {
            "mesh_id": seed_mesh,
            "seed": {"node_id": seed_node, "peer_node_ids": seed_peers},
            "worker": {"node_id": worker_node, "peer_node_ids": worker_peers},
        },
        "snapshots": {
            name: {"path": paths[name].name, "sha256": digests[name]}
            for name in SNAPSHOT_ARGUMENTS
        },
        "errors": [],
    }


def failed_evidence(model_label: str, error: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": KIND,
        "status": "failed",
        "model_label": model_label,
        "errors": [error],
    }


def write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SNAPSHOT_ARGUMENTS:
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--model-label", required=True)
    output_mode = parser.add_mutually_exclusive_group(required=True)
    output_mode.add_argument("--output", type=Path)
    output_mode.add_argument("--verify", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_label = args.model_label.strip()
    if not model_label:
        print("model label must be non-empty", file=sys.stderr)
        return 2
    paths = {name: getattr(args, name) for name in SNAPSHOT_ARGUMENTS}
    try:
        loaded = {name: load_snapshot(path, name) for name, path in paths.items()}
        evidence = reconcile(
            {name: value[0] for name, value in loaded.items()},
            {name: value[1] for name, value in loaded.items()},
            paths,
            model_label,
        )
    except (OSError, ReconciliationError) as error:
        if args.output is not None:
            write_atomic_json(args.output, failed_evidence(model_label, str(error)))
        print(f"two-node split evidence reconciliation failed: {error}", file=sys.stderr)
        return 1

    if args.verify is not None:
        try:
            actual = require_object(
                json.loads(args.verify.read_text(encoding="utf-8")), "split evidence"
            )
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ReconciliationError,
        ) as error:
            print(
                f"cannot verify two-node split evidence {args.verify}: {error}",
                file=sys.stderr,
            )
            return 1
        if actual != evidence:
            print(
                f"two-node split evidence does not match persisted snapshots: {args.verify}",
                file=sys.stderr,
            )
            return 1
        print(f"Verified two-node split evidence: {args.verify}")
        return 0

    assert args.output is not None
    write_atomic_json(args.output, evidence)
    print(
        "ready=true "
        f"topology={evidence['topology']['topology_id']} "
        f"run={evidence['topology']['run_id']} model={evidence['model_id']} "
        "stages=2 observers=2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
