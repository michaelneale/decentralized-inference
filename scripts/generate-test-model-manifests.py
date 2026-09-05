#!/usr/bin/env python3
"""Validate the test-artifact registry and generate suite-owned manifests.

The registry is the only source of model download identity for the suites
listed in it. Generated files deliberately retain the input shape expected by
existing consumers (the family battery keeps its established policy schema),
while adding the common repository/revision/file-integrity contract to every
suite manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "ci" / "model-artifacts" / "registry.json"
MANIFEST_DIR = ROOT / "ci" / "model-artifacts" / "manifests"
SHA_RE = re.compile(r"^[0-9a-f]{40,64}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

SUITE_OUTPUTS = {
    "product-smoke": MANIFEST_DIR / "product-smoke.json",
    "product-integration-smoke": MANIFEST_DIR / "product-integration-smoke.json",
    "scripted-binary-smoke": MANIFEST_DIR / "scripted-binary-smoke.json",
    "sdk-smoke": MANIFEST_DIR / "sdk-smoke.json",
    "hf-download-smoke": MANIFEST_DIR / "hf-download-smoke.json",
    "openai-smoke": MANIFEST_DIR / "openai-smoke.json",
    "skippy-correctness": MANIFEST_DIR / "skippy-correctness.json",
    "safetensors-runtime-smoke": MANIFEST_DIR / "safetensors-runtime-smoke.json",
    "skippy-ci-smoke": MANIFEST_DIR / "skippy-ci-smoke.json",
    "skippy-parity": MANIFEST_DIR / "skippy-parity.json",
    "competitive-benchmark": MANIFEST_DIR / "competitive-benchmark.json",
    "radix-cache": MANIFEST_DIR / "radix-cache.json",
}
FAMILY_MANIFEST = ROOT / "ci" / "llama-canary" / "family-certified.json"


class RegistryError(ValueError):
    """Raised for an invalid registry or generated source."""


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RegistryError(f"{field} must be an object")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{field} must be a non-empty string")
    if any(char in value for char in "\0\r\n\t|"):
        raise RegistryError(f"{field} must be a single-line value")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RegistryError(f"{field} must be a non-empty array")
    result = [_string(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise RegistryError(f"{field} must not contain duplicates")
    return result


def _exact_keys(value: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RegistryError(f"{field} contains unknown fields: {', '.join(unknown)}")


def _artifact(value: Any, field: str) -> dict[str, Any]:
    artifact = _object(value, field)
    _exact_keys(artifact, {"repo", "revision", "selector", "files"}, field)
    repo = _string(artifact.get("repo"), f"{field}.repo")
    if repo.count("/") != 1 or repo.startswith("/") or repo.endswith("/"):
        raise RegistryError(f"{field}.repo must be an owner/repository coordinate")
    revision = _string(artifact.get("revision"), f"{field}.revision")
    if not SHA_RE.fullmatch(revision):
        raise RegistryError(f"{field}.revision must be a lowercase immutable SHA")
    selector = _string(artifact.get("selector"), f"{field}.selector")
    files = artifact.get("files")
    if not isinstance(files, list) or not files:
        raise RegistryError(f"{field}.files must be a non-empty array")
    paths: list[str] = []
    for index, raw_file in enumerate(files):
        file = _object(raw_file, f"{field}.files[{index}]")
        _exact_keys(file, {"path", "size_bytes", "sha256"}, f"{field}.files[{index}]")
        path = _string(file.get("path"), f"{field}.files[{index}].path")
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.endswith("/")
            or "\\" in path
        ):
            raise RegistryError(f"{field}.files[{index}].path is unsafe: {path!r}")
        if path in paths:
            raise RegistryError(f"{field}.files contains duplicate path: {path}")
        paths.append(path)
        size = file.get("size_bytes")
        if type(size) is not int or size <= 0:
            raise RegistryError(f"{field}.files[{index}].size_bytes must be positive")
        digest = _string(file.get("sha256"), f"{field}.files[{index}].sha256")
        if not HASH_RE.fullmatch(digest):
            raise RegistryError(f"{field}.files[{index}].sha256 must be a SHA-256")
    return artifact


def _validate_registry(raw: Any) -> dict[str, Any]:
    registry = _object(raw, "registry")
    _exact_keys(
        registry,
        {"schema_version", "cadences", "suites", "family_policy", "artifacts", "unverified_consumers"},
        "registry",
    )
    if registry.get("schema_version") != 1:
        raise RegistryError("registry.schema_version must be 1")
    cadences = _string_list(registry.get("cadences"), "registry.cadences")
    suites = _string_list(registry.get("suites"), "registry.suites")
    policy = _object(registry.get("family_policy"), "registry.family_policy")
    _exact_keys(policy, {"profiles", "cadences"}, "registry.family_policy")
    if policy.get("cadences") != ["llama-bump", "manual-full", "nightly", "rotating"]:
        raise RegistryError("registry.family_policy.cadences must preserve family cadence order")
    profiles = _object(policy.get("profiles"), "registry.family_policy.profiles")
    expected_profiles = {"full", "package-oracle", "graph-only"}
    if set(profiles) != expected_profiles:
        raise RegistryError("registry.family_policy.profiles must contain the three family profiles")
    for profile_name, profile in profiles.items():
        profile = _object(profile, f"registry.family_policy.profiles.{profile_name}")
        _exact_keys(profile, {"status", "oracle", "required_lanes"}, f"profile {profile_name}")
        _string(profile.get("status"), f"profile {profile_name}.status")
        _string(profile.get("oracle"), f"profile {profile_name}.oracle")
        _string_list(profile.get("required_lanes"), f"profile {profile_name}.required_lanes")
    if not isinstance(registry.get("artifacts"), list) or not registry["artifacts"]:
        raise RegistryError("registry.artifacts must be a non-empty array")
    seen_ids: set[str] = set()
    for index, raw_row in enumerate(registry["artifacts"]):
        field = f"registry.artifacts[{index}]"
        row = _object(raw_row, field)
        _exact_keys(
            row,
            {"id", "family", "suites", "cadences", "capability_tags", "artifact", "certification", "quantizations", "notes"},
            field,
        )
        row_id = _string(row.get("id"), f"{field}.id")
        if not ID_RE.fullmatch(row_id):
            raise RegistryError(f"{field}.id has invalid characters")
        if row_id in seen_ids:
            raise RegistryError(f"duplicate artifact id: {row_id}")
        seen_ids.add(row_id)
        _string(row.get("family"), f"{field}.family")
        row_suites = _string_list(row.get("suites"), f"{field}.suites")
        row_cadences = _string_list(row.get("cadences"), f"{field}.cadences")
        if any(suite not in suites for suite in row_suites):
            raise RegistryError(f"{field}.suites contains an undeclared suite")
        if any(cadence not in cadences for cadence in row_cadences):
            raise RegistryError(f"{field}.cadences contains an undeclared cadence")
        tags = _string_list(row.get("capability_tags"), f"{field}.capability_tags")
        if any(not ID_RE.fullmatch(tag) for tag in tags):
            raise RegistryError(f"{field}.capability_tags contains an invalid tag")
        _artifact(row.get("artifact"), f"{field}.artifact")
        if "quantizations" in row:
            _string_list(row["quantizations"], f"{field}.quantizations")
        if "notes" in row:
            _string(row["notes"], f"{field}.notes")
        if "llama-family-certification" in row["suites"]:
            certification = _object(row.get("certification"), f"{field}.certification")
            _exact_keys(
                certification,
                {"profile", "cadences", "execution", "resources", "notes", "draft_artifact", "mmproj_artifact"},
                f"{field}.certification",
            )
            profile = _string(certification.get("profile"), f"{field}.certification.profile")
            if profile not in profiles:
                raise RegistryError(f"{field}.certification.profile is not a family profile")
            if "cadences" in certification:
                certification_cadences = _string_list(
                    certification["cadences"], f"{field}.certification.cadences"
                )
                if any(cadence not in policy["cadences"] for cadence in certification_cadences):
                    raise RegistryError(
                        f"{field}.certification.cadences contains a non-family cadence"
                    )
            _object(certification.get("execution"), f"{field}.certification.execution")
            _object(certification.get("resources"), f"{field}.certification.resources")
            _string(certification.get("notes"), f"{field}.certification.notes")
            for optional in ("draft_artifact", "mmproj_artifact"):
                if optional in certification:
                    _artifact(certification[optional], f"{field}.certification.{optional}")
    unverified = registry.get("unverified_consumers", [])
    if not isinstance(unverified, list):
        raise RegistryError("registry.unverified_consumers must be an array")
    for index, raw_item in enumerate(unverified):
        item = _object(raw_item, f"registry.unverified_consumers[{index}]")
        _exact_keys(item, {"path", "reason"}, f"registry.unverified_consumers[{index}]")
        _string(item.get("path"), f"registry.unverified_consumers[{index}].path")
        _string(item.get("reason"), f"registry.unverified_consumers[{index}].reason")
    return registry


def _integrity(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        file["path"]: {"size_bytes": file["size_bytes"], "blob_id": file["sha256"]}
        for file in artifact["files"]
    }


def _family_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo": artifact["repo"],
        "revision": artifact["revision"],
        "files": [file["path"] for file in artifact["files"]],
        "file_integrity": _integrity(artifact),
        "selector": artifact["selector"],
    }


def _family_manifest(registry: dict[str, Any]) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for row in registry["artifacts"]:
        if "llama-family-certification" not in row["suites"]:
            continue
        certification = row["certification"]
        model: dict[str, Any] = {
            "family": row["family"],
            "profile": certification["profile"],
            "cadences": certification.get("cadences", row["cadences"]),
            "artifact": _family_artifact(row["artifact"]),
        }
        for optional in ("draft_artifact", "mmproj_artifact"):
            if optional in certification:
                model[optional] = _family_artifact(certification[optional])
        model.update(
            execution=certification["execution"],
            resources=certification["resources"],
            notes=certification["notes"],
        )
        models.append(model)
    if not models:
        raise RegistryError("suite llama-family-certification has no registered artifacts")
    return {
        "schema_version": 1,
        "policy": registry["family_policy"],
        "models": models,
    }


def _suite_row(row: dict[str, Any]) -> dict[str, Any]:
    artifact = row["artifact"]
    files = artifact["files"]
    result: dict[str, Any] = {
        "id": row["id"],
        "family": row["family"],
        "capability_tags": row["capability_tags"],
        "suites": row["suites"],
        "cadences": row["cadences"],
        "repo": artifact["repo"],
        "revision": artifact["revision"],
        "selector": artifact["selector"],
        "files": [file["path"] for file in files],
        "file_integrity": _integrity(artifact),
        "model_ref": f"{artifact['repo']}:{artifact['selector']}",
        "urls": [
            f"https://huggingface.co/{artifact['repo']}/resolve/{artifact['revision']}/{file['path']}"
            for file in files
        ],
    }
    if len(files) == 1:
        result["file"] = files[0]["path"]
        result["size_bytes"] = files[0]["size_bytes"]
        result["sha256"] = files[0]["sha256"]
        result["url"] = result["urls"][0]
    else:
        result["size_bytes"] = sum(file["size_bytes"] for file in files)
    if "quantizations" in row:
        result["quantizations"] = row["quantizations"]
    if "notes" in row:
        result["notes"] = row["notes"]
    return result


def _dump(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _dump_family(value: dict[str, Any]) -> bytes:
    """Keep the generated family policy compact enough for human review."""

    def compact(item: Any) -> str:
        return json.dumps(item, ensure_ascii=False)

    lines = ["{", '  "schema_version": 1,', '  "policy": {', '    "profiles": {']
    profile_items = list(value["policy"]["profiles"].items())
    for profile_index, (name, profile) in enumerate(profile_items):
        lines.extend(
            [
                f'      {compact(name)}: {{',
                f'        "status": {compact(profile["status"])},',
                f'        "oracle": {compact(profile["oracle"])},',
                f'        "required_lanes": {compact(profile["required_lanes"])}',
                "      }" + ("," if profile_index + 1 < len(profile_items) else ""),
            ]
        )
    lines.extend(
        [
            "    },",
            f'    "cadences": {compact(value["policy"]["cadences"])}',
            "  },",
            '  "models": [',
        ]
    )
    for model_index, model in enumerate(value["models"]):
        lines.extend(
            [
                "    {",
                f'      "family": {compact(model["family"])},',
                f'      "profile": {compact(model["profile"])},',
                f'      "cadences": {compact(model["cadences"])},',
                f'      "artifact": {compact(model["artifact"])},',
            ]
        )
        for optional in ("draft_artifact", "mmproj_artifact"):
            if optional in model:
                lines.append(f'      {compact(optional)}: {compact(model[optional])},')
        lines.extend(
            [
                f'      "execution": {compact(model["execution"])},',
                f'      "resources": {compact(model["resources"])},',
                f'      "notes": {compact(model["notes"])}',
                "    }" + ("," if model_index + 1 < len(value["models"]) else ""),
            ]
        )
    lines.extend(["  ]", "}"])
    return ("\n".join(lines) + "\n").encode("utf-8")


def _expected_outputs(registry: dict[str, Any], registry_path: Path) -> dict[Path, bytes]:
    outputs: dict[Path, bytes] = {
        FAMILY_MANIFEST: _dump_family(_family_manifest(registry))
    }
    for suite, destination in SUITE_OUTPUTS.items():
        rows = [_suite_row(row) for row in registry["artifacts"] if suite in row["suites"]]
        if not rows:
            raise RegistryError(f"suite {suite} has no registered artifacts")
        outputs[destination] = _dump(
            {
                "schema_version": 1,
                "manifest_kind": "test-model-artifacts",
                "suite": suite,
                "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                "artifacts": rows,
            }
        )
    return outputs


def _write_or_check(outputs: dict[Path, bytes], check: bool) -> int:
    failures: list[str] = []
    for path, expected in outputs.items():
        actual = path.read_bytes() if path.exists() else None
        if check:
            if actual != expected:
                failures.append(path.relative_to(ROOT).as_posix())
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
    if failures:
        print("generated test-model manifests are stale:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--check", action="store_true", help="fail when generated outputs are stale")
    args = parser.parse_args()
    try:
        raw = json.loads(args.registry.read_text(encoding="utf-8"))
        registry = _validate_registry(raw)
        outputs = _expected_outputs(registry, args.registry)
    except (OSError, json.JSONDecodeError, RegistryError) as error:
        print(f"test-model registry error: {error}", file=sys.stderr)
        return 2
    return _write_or_check(outputs, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
