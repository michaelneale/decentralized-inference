#!/usr/bin/env python3
"""Resolve and verify an artifact from a generated test-model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when a generated manifest violates the consumer contract."""


def _load(path: Path, artifact_id: str | None, cadence: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("manifest_kind") != "test-model-artifacts":
        raise ManifestError("manifest_kind must be test-model-artifacts")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ManifestError("manifest must contain artifacts")
    matches = [row for row in artifacts if isinstance(row, dict) and row.get("id") == artifact_id]
    if artifact_id is None:
        if len(artifacts) != 1 or not isinstance(artifacts[0], dict):
            raise ManifestError("manifest contains multiple artifacts; pass --artifact-id")
        artifact = artifacts[0]
    elif len(matches) == 1:
        artifact = matches[0]
    elif not matches:
        raise ManifestError(f"artifact id is absent from manifest: {artifact_id}")
    else:
        raise ManifestError(f"artifact id is duplicated in manifest: {artifact_id}")

    required_strings = ("id", "repo", "revision", "selector", "model_ref")
    for field in required_strings:
        if not isinstance(artifact.get(field), str) or not artifact[field]:
            raise ManifestError(f"artifact.{field} must be a non-empty string")
    cadences = artifact.get("cadences")
    if not isinstance(cadences, list) or not cadences or not all(
        isinstance(value, str) and value for value in cadences
    ):
        raise ManifestError("artifact.cadences must be a non-empty string list")
    if cadence not in cadences:
        raise ManifestError(
            f"artifact {artifact['id']} is not allowed at cadence {cadence!r}"
        )
    files = artifact.get("files")
    integrity = artifact.get("file_integrity")
    urls = artifact.get("urls")
    if not isinstance(files, list) or not files or not isinstance(integrity, dict):
        raise ManifestError("artifact files and file_integrity are required")
    if not isinstance(urls, list) or len(urls) != len(files):
        raise ManifestError("artifact urls must exactly cover files")
    for index, name in enumerate(files):
        if not isinstance(name, str) or not name:
            raise ManifestError(f"artifact.files[{index}] must be a non-empty string")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ManifestError(f"artifact.files[{index}] is unsafe: {name!r}")
        record = integrity.get(name)
        if not isinstance(record, dict):
            raise ManifestError(f"artifact.file_integrity is missing {name}")
        size = record.get("size_bytes")
        digest = record.get("blob_id")
        if type(size) is not int or size <= 0:
            raise ManifestError(f"artifact size is invalid for {name}")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ManifestError(f"artifact SHA-256 is invalid for {name}")
        if not isinstance(urls[index], str) or not urls[index]:
            raise ManifestError(f"artifact URL is invalid for {name}")
    return artifact


def _summary(artifact: dict[str, Any]) -> dict[str, str]:
    files = artifact["files"]
    result = {
        "artifact_id": artifact["id"],
        "repo": artifact["repo"],
        "revision": artifact["revision"],
        "selector": artifact["selector"],
        "model_ref": artifact["model_ref"],
        "files_json": json.dumps(files, separators=(",", ":")),
    }
    if len(files) == 1:
        name = files[0]
        integrity = artifact["file_integrity"][name]
        result.update(
            file=name,
            url=artifact["urls"][0],
            sha256=integrity["blob_id"],
            size_bytes=str(integrity["size_bytes"]),
        )
    quantizations = artifact.get("quantizations")
    if isinstance(quantizations, list) and all(
        isinstance(value, str) and value for value in quantizations
    ):
        result["quantizations_json"] = json.dumps(quantizations, separators=(",", ":"))
    return result


def _verify(artifact: dict[str, Any], root: Path) -> None:
    for name in artifact["files"]:
        path = root / name
        if not path.is_file():
            raise ManifestError(f"artifact file is missing: {path}")
        record = artifact["file_integrity"][name]
        actual_size = path.stat().st_size
        if actual_size != record["size_bytes"]:
            raise ManifestError(
                f"artifact size mismatch for {path}: expected {record['size_bytes']}, got {actual_size}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != record["blob_id"]:
            raise ManifestError(
                f"artifact SHA-256 mismatch for {path}: expected {record['blob_id']}, got {actual_sha}"
            )
        print(f"verified immutable test artifact: {path} ({actual_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--artifact-id")
    parser.add_argument("--cadence", required=True)
    parser.add_argument("--require-single-file", action="store_true")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument(
        "--github-output-prefix",
        help="safe prefix prepended to every GitHub output name",
    )
    parser.add_argument("--verify-root", type=Path)
    args = parser.parse_args()
    try:
        artifact = _load(args.manifest, args.artifact_id, args.cadence)
        if args.require_single_file and len(artifact["files"]) != 1:
            raise ManifestError(
                f"artifact {artifact['id']} must contain exactly one file"
            )
        if args.verify_root is not None:
            _verify(artifact, args.verify_root)
        summary = _summary(artifact)
        if args.github_output is not None:
            prefix = args.github_output_prefix or ""
            if prefix and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*_", prefix):
                raise ManifestError(
                    "github output prefix must end in '_' and contain only "
                    "ASCII letters, digits, and underscores"
                )
            with args.github_output.open("a", encoding="utf-8") as output:
                for name, value in summary.items():
                    if "\n" in value or "\r" in value:
                        raise ManifestError(f"output {name} must be single-line")
                    output.write(f"{prefix}{name}={value}\n")
        elif args.verify_root is None:
            print(json.dumps(summary, sort_keys=True))
    except (OSError, json.JSONDecodeError, ManifestError) as error:
        print(f"test-model manifest error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
