#!/usr/bin/env python3
"""Validate and plan the supported-family certification battery.

The checked-in policy manifest is JSON so the trusted self-hosted runner needs
no package installation before it can reject a malformed policy or a missing
immutable cache entry.  The emitted plan is deterministic: it records exact
artifact revisions, resolves profile-owned lanes, and produces bounded GitHub
matrix shards without depending on runner availability or wall-clock state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "ci" / "llama-canary" / "family-certified.json"
CORE_LANES = ("single-step", "chain", "state-handoff")
PROFILE_NAMES = ("full", "package-oracle", "graph-only")
CERTIFIED_PROFILES = ("full", "package-oracle")
CERTIFICATION_STATUSES = ("certified", "provisional")
ORACLE_KINDS = ("local-monolithic", "independent-trace", "none")
CADENCES = ("llama-bump", "manual-full", "nightly", "rotating")
CACHE_POLICIES = ("immutable-local",)
RUNNER_ROLES = ("family-certify",)
SPECULATIVE_POLICIES = ("mtp-if-present", "disabled")
SHA_RE = re.compile(r"^[0-9a-f]{40,64}$")
FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class PlanError(ValueError):
    """Raised when the policy, selection, or immutable cache is invalid."""


class _GgufReader:
    """Minimal GGUF metadata reader; tensor payloads are never touched."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("rb")

    def close(self) -> None:
        self.handle.close()

    def read(self, size: int) -> bytes:
        data = self.handle.read(size)
        if len(data) != size:
            raise PlanError(f"short GGUF metadata read: {self.path}")
        return data

    def unpack(self, pattern: str) -> object:
        return struct.unpack(pattern, self.read(struct.calcsize(pattern)))[0]

    def u32(self) -> int:
        return int(self.unpack("<I"))

    def u64(self) -> int:
        return int(self.unpack("<Q"))

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", errors="replace")

    def value(self, kind: int) -> object:
        scalar_formats = {
            0: "<B",
            1: "<b",
            2: "<H",
            3: "<h",
            4: "<I",
            5: "<i",
            6: "<f",
            7: "<?",
            10: "<Q",
            11: "<q",
            12: "<d",
        }
        if kind in scalar_formats:
            return self.unpack(scalar_formats[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            item_kind = self.u32()
            return [self.value(item_kind) for _ in range(self.u64())]
        raise PlanError(f"unsupported GGUF metadata type {kind}: {self.path}")


def _gguf_dimensions(path: Path) -> tuple[int, int] | None:
    reader = _GgufReader(path)
    try:
        if reader.read(4) != b"GGUF":
            raise PlanError(f"invalid GGUF magic: {path}")
        version = reader.u32()
        if version < 2:
            raise PlanError(f"unsupported GGUF version {version}: {path}")
        reader.u64()  # tensor count
        kv_count = reader.u64()
        block_counts: list[int] = []
        embedding_lengths: list[int] = []
        architecture: str | None = None
        hyper_connection_counts: list[int] = []
        embedding_lengths_out: list[int] = []
        for _ in range(kv_count):
            key = reader.string()
            value = reader.value(reader.u32())
            if key == "general.architecture" and isinstance(value, str):
                architecture = value
            if key.endswith(".block_count") and type(value) is int:
                block_counts.append(value)
            if key.endswith(".embedding_length") and type(value) is int:
                embedding_lengths.append(value)
            if key.endswith(".hyper_connection.count") and type(value) is int:
                hyper_connection_counts.append(value)
            if key.endswith(".embedding_length_out") and type(value) is int:
                embedding_lengths_out.append(value)
        if not block_counts and not embedding_lengths:
            return None
        if len(block_counts) != 1 or block_counts[0] < 1:
            raise PlanError(f"GGUF must contain exactly one positive *.block_count: {path}")
        if len(embedding_lengths) != 1 or embedding_lengths[0] < 1:
            raise PlanError(
                f"GGUF must contain exactly one positive *.embedding_length: {path}"
            )
        activation_width = embedding_lengths[0]
        if architecture == "qwen4exp" or (
            architecture == "dflash" and hyper_connection_counts
        ):
            if len(hyper_connection_counts) != 1 or hyper_connection_counts[0] < 1:
                raise PlanError(
                    f"{architecture} GGUF must contain exactly one positive "
                    f"*.hyper_connection.count: {path}"
                )
            activation_width *= hyper_connection_counts[0]
            if activation_width > 0x7FFFFFFF:
                raise PlanError(f"{architecture} activation width exceeds i32: {path}")
            if len(embedding_lengths_out) > 1 or (
                embedding_lengths_out and embedding_lengths_out[0] != activation_width
            ):
                raise PlanError(
                    f"{architecture} *.embedding_length_out disagrees with "
                    f"hyper-connected activation width {activation_width}: {path}"
                )
        return block_counts[0], activation_width
    finally:
        reader.close()


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanError(f"{field} must be an object")
    return value


def _exact_keys(value: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise PlanError(f"{field} contains unknown fields: {', '.join(unknown)}")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or any(char in value for char in "\r\n\t|"):
        raise PlanError(f"{field} must be a non-empty single-line string")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise PlanError(f"{field} must be an array")
    result = [_string(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise PlanError(f"{field} must not contain duplicates")
    return result


def _integer(
    value: object, field: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        bounds = (
            f">= {minimum}"
            if maximum is None
            else f"between {minimum} and {maximum}"
        )
        raise PlanError(f"{field} must be an integer {bounds}")
    return value


def _enum(value: object, field: str, allowed: tuple[str, ...]) -> str:
    result = _string(value, field)
    if result not in allowed:
        raise PlanError(f"{field} must be one of: {', '.join(allowed)}")
    return result


def _artifact(value: object, field: str) -> dict[str, Any]:
    artifact = _object(value, field)
    _exact_keys(
        artifact,
        {"repo", "revision", "files", "file_integrity", "selector"},
        field,
    )
    repo = _string(artifact.get("repo"), f"{field}.repo")
    if repo.count("/") != 1 or repo.startswith("/") or repo.endswith("/"):
        raise PlanError(f"{field}.repo must be an owner/repository coordinate")
    revision = _string(artifact.get("revision"), f"{field}.revision")
    if not SHA_RE.fullmatch(revision):
        raise PlanError(f"{field}.revision must be a lowercase immutable SHA")
    files = _string_list(artifact.get("files"), f"{field}.files")
    if not files:
        raise PlanError(f"{field}.files must not be empty")
    for file in files:
        path = PurePosixPath(file)
        if path.is_absolute() or ".." in path.parts or file.endswith("/"):
            raise PlanError(f"{field}.files contains an unsafe path: {file!r}")
    integrity = _object(artifact.get("file_integrity"), f"{field}.file_integrity")
    if set(integrity) != set(files):
        raise PlanError(f"{field}.file_integrity must exactly cover {field}.files")
    normalized_integrity: dict[str, dict[str, Any]] = {}
    for file in files:
        record = _object(integrity[file], f"{field}.file_integrity[{file!r}]")
        _exact_keys(
            record,
            {"size_bytes", "blob_id"},
            f"{field}.file_integrity[{file!r}]",
        )
        size_bytes = _integer(
            record.get("size_bytes"),
            f"{field}.file_integrity[{file!r}].size_bytes",
            1,
        )
        blob_id = _string(
            record.get("blob_id"),
            f"{field}.file_integrity[{file!r}].blob_id",
        )
        if not re.fullmatch(r"[0-9a-f]{64}", blob_id):
            raise PlanError(
                f"{field}.file_integrity[{file!r}].blob_id must be a lowercase SHA-256"
            )
        normalized_integrity[file] = {
            "size_bytes": size_bytes,
            "blob_id": blob_id,
        }
    selector = _string(artifact.get("selector"), f"{field}.selector")
    return {
        "repo": repo,
        "revision": revision,
        "files": files,
        "file_integrity": normalized_integrity,
        "selector": selector,
    }


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"unable to load {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise PlanError(f"{path} must contain an object")
    if manifest.get("schema_version") != 1:
        raise PlanError(f"{path} has unsupported schema_version")
    return manifest, hashlib.sha256(raw).hexdigest()


def _validate_policy(value: object) -> dict[str, Any]:
    policy = _object(value, "policy")
    _exact_keys(policy, {"profiles", "cadences"}, "policy")
    cadences = _string_list(policy.get("cadences"), "policy.cadences")
    if tuple(cadences) != CADENCES:
        raise PlanError("policy.cadences must list the complete supported cadence order")

    profiles = _object(policy.get("profiles"), "policy.profiles")
    if set(profiles) != set(PROFILE_NAMES):
        raise PlanError("policy.profiles must define full, package-oracle, and graph-only")
    normalized: dict[str, Any] = {}
    for name in PROFILE_NAMES:
        profile = _object(profiles[name], f"policy.profiles.{name}")
        _exact_keys(profile, {"status", "oracle", "required_lanes"}, f"policy.profiles.{name}")
        status = _enum(
            profile.get("status"),
            f"policy.profiles.{name}.status",
            CERTIFICATION_STATUSES,
        )
        oracle = _enum(
            profile.get("oracle"), f"policy.profiles.{name}.oracle", ORACLE_KINDS
        )
        lanes = _string_list(
            profile.get("required_lanes"),
            f"policy.profiles.{name}.required_lanes",
        )
        if name in CERTIFIED_PROFILES:
            if status != "certified" or tuple(lanes) != CORE_LANES:
                raise PlanError(
                    f"certified profile {name} must require exactly the three core lanes"
                )
        elif status != "provisional" or oracle != "none":
            raise PlanError("graph-only must remain provisional and oracle-free")
        normalized[name] = {
            "status": status,
            "oracle": oracle,
            "required_lanes": lanes,
        }
    if normalized["full"]["oracle"] != "local-monolithic":
        raise PlanError("full profile must use the local-monolithic oracle")
    if normalized["package-oracle"]["oracle"] != "independent-trace":
        raise PlanError("package-oracle must use an independent trace")
    return {"profiles": normalized, "cadences": cadences}


def _normalize_models(value: object, policy: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PlanError("models must be a non-empty array")
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_model in enumerate(value):
        field = f"models[{index}]"
        model = _object(raw_model, field)
        _exact_keys(
            model,
            {
                "family",
                "profile",
                "cadences",
                "artifact",
                "draft_artifact",
                "mmproj_artifact",
                "execution",
                "resources",
                "notes",
            },
            field,
        )
        family = _string(model.get("family"), f"{field}.family")
        if not FAMILY_RE.fullmatch(family):
            raise PlanError(f"{field}.family has an invalid label: {family!r}")
        if family in seen:
            raise PlanError(f"duplicate family: {family}")
        seen.add(family)
        profile = _enum(model.get("profile"), f"{field}.profile", PROFILE_NAMES)
        cadences = _string_list(model.get("cadences"), f"{field}.cadences")
        if not cadences or any(item not in policy["cadences"] for item in cadences):
            raise PlanError(f"{field}.cadences contains an unsupported cadence")
        artifact = _artifact(model.get("artifact"), f"{field}.artifact")
        draft = None
        if "draft_artifact" in model:
            draft = _artifact(model["draft_artifact"], f"{field}.draft_artifact")
        mmproj = None
        if "mmproj_artifact" in model:
            mmproj = _artifact(model["mmproj_artifact"], f"{field}.mmproj_artifact")
        if mmproj is not None and len(mmproj["files"]) != 1:
            raise PlanError(f"{field}.mmproj_artifact.files must name exactly one projector GGUF")

        execution = _object(model.get("execution"), f"{field}.execution")
        _exact_keys(
            execution,
            {
                "trunk_layers",
                "mtp_layers",
                "activation_width",
                "boundary_sweep_period",
                "speculative_policy",
            },
            f"{field}.execution",
        )
        trunk_layers = _integer(
            execution.get("trunk_layers"), f"{field}.execution.trunk_layers", 1
        )
        mtp_layers = _integer(
            execution.get("mtp_layers"), f"{field}.execution.mtp_layers"
        )
        activation_width = _integer(
            execution.get("activation_width"),
            f"{field}.execution.activation_width",
            1,
        )
        sweep_period = _integer(
            execution.get("boundary_sweep_period"),
            f"{field}.execution.boundary_sweep_period",
        )
        layer_end = trunk_layers + mtp_layers
        if sweep_period > layer_end:
            raise PlanError(f"{field}.execution.boundary_sweep_period exceeds layer range")
        speculative_policy = _enum(
            execution.get("speculative_policy"),
            f"{field}.execution.speculative_policy",
            SPECULATIVE_POLICIES,
        )

        resources = _object(model.get("resources"), f"{field}.resources")
        _exact_keys(
            resources,
            {
                "runner_role",
                "cache_policy",
                "estimated_model_bytes",
                "startup_timeout_secs",
            },
            f"{field}.resources",
        )
        runner_role = _enum(
            resources.get("runner_role"),
            f"{field}.resources.runner_role",
            RUNNER_ROLES,
        )
        cache_policy = _enum(
            resources.get("cache_policy"),
            f"{field}.resources.cache_policy",
            CACHE_POLICIES,
        )
        estimated_model_bytes = _integer(
            resources.get("estimated_model_bytes"),
            f"{field}.resources.estimated_model_bytes",
            1,
        )
        startup_timeout_secs = None
        if "startup_timeout_secs" in resources:
            startup_timeout_secs = _integer(
                resources["startup_timeout_secs"],
                f"{field}.resources.startup_timeout_secs",
                180,
                900,
            )
        notes = _string(model.get("notes"), f"{field}.notes")
        profile_policy = policy["profiles"][profile]
        models.append(
            {
                "family": family,
                "profile": profile,
                "certification_status": profile_policy["status"],
                "oracle": profile_policy["oracle"],
                "certification_lanes": profile_policy["required_lanes"],
                "cadences": cadences,
                "artifact": artifact,
                "draft_artifact": draft,
                "mmproj_artifact": mmproj,
                "execution": {
                    "trunk_layers": trunk_layers,
                    "mtp_layers": mtp_layers,
                    "activation_width": activation_width,
                    "layer_end": layer_end,
                    "boundary_sweep_period": sweep_period,
                    "speculative_policy": speculative_policy,
                },
                "resources": {
                    "runner_role": runner_role,
                    "cache_policy": cache_policy,
                    "estimated_model_bytes": estimated_model_bytes,
                    "startup_timeout_secs": startup_timeout_secs,
                },
                "notes": notes,
                "manifest_index": index,
            }
        )
    return models


def _select_models(
    models: list[dict[str, Any]], families: str, cadence: str
) -> list[dict[str, Any]]:
    selected = models
    if cadence:
        if cadence not in CADENCES:
            raise PlanError(f"--cadence must be one of: {', '.join(CADENCES)}")
        selected = [model for model in selected if cadence in model["cadences"]]
    if not families:
        return selected
    requested = families.split(",")
    if any(not FAMILY_RE.fullmatch(item) for item in requested) or len(
        set(requested)
    ) != len(requested):
        raise PlanError("--families must contain unique comma-separated family labels")
    by_family = {model["family"]: model for model in models}
    unknown = [family for family in requested if family not in by_family]
    if unknown:
        raise PlanError(f"unknown selected families: {', '.join(unknown)}")
    requested_set = set(requested)
    return [model for model in selected if model["family"] in requested_set]


def _work_weight(model: dict[str, Any]) -> int:
    period = model["execution"]["boundary_sweep_period"]
    certifications = 1 + (period * 3 if period else 0)
    return model["resources"]["estimated_model_bytes"] * certifications


def _shards(models: list[dict[str, Any]], requested_count: int) -> list[dict[str, Any]]:
    if requested_count < 1:
        raise PlanError("--shard-count must be positive")
    count = min(requested_count, len(models))
    shards = [{"shard_index": index, "models": [], "estimated_work_bytes": 0} for index in range(count)]
    for model in sorted(models, key=lambda item: (-_work_weight(item), item["family"])):
        shard = min(shards, key=lambda item: (item["estimated_work_bytes"], item["shard_index"]))
        shard["models"].append(model)
        shard["estimated_work_bytes"] += _work_weight(model)
    for shard in shards:
        shard["models"].sort(key=lambda item: item["manifest_index"])
        shard["families"] = [model["family"] for model in shard.pop("models")]
        shard["id"] = f"family-battery-{shard['shard_index'] + 1:02d}"
    return shards


def _cache_hub(cache_root: Path) -> Path:
    return cache_root if cache_root.name == "hub" else cache_root / "hub"


def _artifact_cache_paths(cache_root: Path, artifact: dict[str, Any]) -> list[Path]:
    repo_dir = "models--" + artifact["repo"].replace("/", "--")
    base = _cache_hub(cache_root) / repo_dir / "snapshots" / artifact["revision"]
    return [base.joinpath(*PurePosixPath(file).parts) for file in artifact["files"]]


def _verify_cache(models: list[dict[str, Any]], cache_root: Path) -> None:
    missing: list[str] = []
    for model in models:
        artifacts = [("target", model["artifact"])]
        if model["draft_artifact"] is not None:
            artifacts.append(("draft", model["draft_artifact"]))
        if model["mmproj_artifact"] is not None:
            artifacts.append(("mmproj", model["mmproj_artifact"]))
        for kind, artifact in artifacts:
            repo_dir = "models--" + artifact["repo"].replace("/", "--")
            blob_dir = (_cache_hub(cache_root) / repo_dir / "blobs").resolve()
            for relative, path in zip(
                artifact["files"], _artifact_cache_paths(cache_root, artifact), strict=True
            ):
                if not path.is_file():
                    missing.append(f"{model['family']} {kind}: {path}")
                    continue
                integrity = artifact["file_integrity"][relative]
                if not path.is_symlink():
                    missing.append(
                        f"{model['family']} {kind}: snapshot entry is not a content-addressed symlink: {path}"
                    )
                    continue
                resolved = path.resolve()
                if resolved.parent != blob_dir or resolved.name != integrity["blob_id"]:
                    missing.append(
                        f"{model['family']} {kind}: expected blob {integrity['blob_id']} but snapshot resolves to {resolved}"
                    )
                    continue
                actual_size = path.stat().st_size
                if actual_size != integrity["size_bytes"]:
                    missing.append(
                        f"{model['family']} {kind}: expected {integrity['size_bytes']} bytes for {relative} but found {actual_size}"
                    )
    if missing:
        joined = "\n  ".join(missing)
        raise PlanError(f"immutable family cache is incomplete:\n  {joined}")
    for model in models:
        artifacts = [("target", model["artifact"])]
        if model["draft_artifact"] is not None:
            artifacts.append(("draft", model["draft_artifact"]))
        # A projector GGUF (mmproj_artifact) is deliberately excluded from
        # this pass: it is a sidecar encoder, not a trunk model, so it carries
        # no *.block_count / *.embedding_length trunk dimensions to check.
        for kind, artifact in artifacts:
            found_dimensions = False
            for target in _artifact_cache_paths(cache_root, artifact):
                dimensions = _gguf_dimensions(target)
                if dimensions is None:
                    continue
                found_dimensions = True
                block_count, embedding_length = dimensions
                if kind != "target":
                    continue
                planned = model["execution"]["layer_end"]
                if block_count != planned:
                    raise PlanError(
                        f"{model['family']} plans {planned} runtime layers but immutable "
                        f"GGUF metadata in {target.name} declares {block_count}"
                    )
                planned_width = model["execution"]["activation_width"]
                if embedding_length != planned_width:
                    raise PlanError(
                        f"{model['family']} plans activation width {planned_width} but immutable "
                        f"GGUF metadata in {target.name} declares {embedding_length}"
                    )
            if not found_dimensions:
                raise PlanError(
                    f"{model['family']} {kind} artifact has no GGUF shard with "
                    "positive *.block_count and *.embedding_length metadata"
                )


def build_plan(
    manifest_path: Path,
    families: str = "",
    cadence: str = "",
    shard_count: int = 1,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    manifest, manifest_sha256 = _load_manifest(manifest_path)
    _exact_keys(manifest, {"schema_version", "policy", "models"}, "manifest")
    policy = _validate_policy(manifest.get("policy"))
    models = _select_models(
        _normalize_models(manifest.get("models"), policy), families, cadence
    )
    if not models:
        raise PlanError("family selection produced no models")
    if cache_root is not None:
        _verify_cache(models, cache_root)
    shards = _shards(models, shard_count)
    matrix = {
        "include": [
            {
                "id": shard["id"],
                "shard_index": shard["shard_index"],
                "families": ",".join(shard["families"]),
                "estimated_work_bytes": shard["estimated_work_bytes"],
            }
            for shard in shards
        ]
    }
    try:
        manifest_source = manifest_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        manifest_source = manifest_path.name
    return {
        "schema_version": 1,
        "generated_by": "scripts/plan-family-battery.py",
        "manifest": manifest_source,
        "manifest_sha256": manifest_sha256,
        "required_certification_lanes": list(CORE_LANES),
        "selected_cadence": cadence or None,
        "selected_family_count": len(models),
        "selected_models": models,
        "shards": shards,
        "github_matrix": matrix,
    }


def _write_github_output(path: Path, plan: dict[str, Any], plan_path: Path) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"plan_path={plan_path.resolve()}\n")
        handle.write(f"manifest_sha256={plan['manifest_sha256']}\n")
        handle.write(f"family_count={plan['selected_family_count']}\n")
        handle.write(f"matrix={json.dumps(plan['github_matrix'], separators=(',', ':'))}\n")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--families", default="")
    parser.add_argument("--cadence", default="", choices=("", *CADENCES))
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--github-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    cache_root = args.cache_root
    if args.check_cache and cache_root is None:
        env_cache = os.environ.get("HF_CACHE") or os.environ.get("HF_HOME")
        if not env_cache:
            raise PlanError("--check-cache requires --cache-root or HF_CACHE/HF_HOME")
        cache_root = Path(env_cache)
    if not args.check_cache:
        cache_root = None
    plan = build_plan(
        args.manifest,
        families=args.families,
        cadence=args.cadence,
        shard_count=args.shard_count,
        cache_root=cache_root,
    )
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    if args.github_output:
        if not args.output:
            raise PlanError("--github-output requires --output")
        _write_github_output(args.github_output, plan, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlanError as error:
        print(f"family battery plan failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
