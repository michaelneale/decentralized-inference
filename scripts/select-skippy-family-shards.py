#!/usr/bin/env python3
"""Select certified family tests from generated shards or upstream paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SENTINELS = {"qwen3-dense", "qwen3-moe", "mamba", "lfm2-vl"}


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("shards"), list):
        raise ValueError(f"invalid generated family series: {path}")
    return payload


def keyed(payload: dict) -> dict[tuple[str, ...], dict]:
    result = {}
    for shard in payload["shards"]:
        families = shard.get("families")
        sources = shard.get("sources")
        digest = shard.get("sha256")
        if not isinstance(families, list) or not isinstance(sources, list) or not isinstance(digest, str):
            raise ValueError("generated family series contains an invalid shard record")
        key = tuple(sources)
        if not key or key in result:
            raise ValueError("generated family series contains duplicate or empty source sets")
        result[key] = shard
    return result


def select(base: dict, current: dict, include_sentinels: bool) -> dict:
    if base.get("generator_version") != current.get("generator_version"):
        return {"mode": "full", "families": [], "reason": "generator-version-changed"}
    old = keyed(base)
    new = keyed(current)
    changed = {
        key for key in set(old) | set(new)
        if key not in old or key not in new or old[key]["sha256"] != new[key]["sha256"]
    }
    if not changed:
        return {"mode": "none", "families": [], "reason": "no-shard-changes"}
    families: set[str] = set()
    for key in changed:
        for record in (old.get(key), new.get(key)):
            if record is None:
                continue
            owners = record["families"]
            if not owners:
                return {"mode": "full", "families": [], "reason": "unmapped-shard-changed"}
            families.update(owners)
    if include_sentinels:
        families.update(SENTINELS)
    return {"mode": "targeted", "families": sorted(families), "reason": "mapped-shards-changed"}


def load_family_map(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    families = payload.get("families")
    if payload.get("schema_version") != 1 or not isinstance(families, dict):
        raise ValueError(f"invalid generated family source map: {path}")
    result: dict[str, list[str]] = {}
    for family, sources in families.items():
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(sources, list)
            or not sources
            or any(not isinstance(source, str) or not source for source in sources)
        ):
            raise ValueError("generated family source map contains an invalid record")
        result[family] = sources
    return result


def select_upstream_paths(
    changed_paths: list[str], family_map: dict[str, list[str]], include_sentinels: bool
) -> dict:
    paths = {path.strip() for path in changed_paths if path.strip()}
    if not paths:
        return {"mode": "none", "families": [], "reason": "no-upstream-changes"}

    owners_by_source: dict[str, set[str]] = {}
    for family, sources in family_map.items():
        for source in sources:
            owners_by_source.setdefault(source, set()).add(family)

    families: set[str] = set()
    for path in paths:
        owners = owners_by_source.get(path)
        if owners is None:
            reason = (
                "unmapped-model-source-changed"
                if path.startswith("src/models/") and path.endswith(".cpp")
                else "shared-upstream-source-changed"
            )
            return {"mode": "full", "families": [], "reason": reason}
        families.update(owners)

    if include_sentinels:
        families.update(SENTINELS)
    return {
        "mode": "targeted",
        "families": sorted(families),
        "reason": "mapped-upstream-model-sources-changed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--changed-paths", type=Path)
    parser.add_argument("--family-map", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-sentinels", action="store_true")
    args = parser.parse_args(argv)
    try:
        shard_mode = args.base is not None or args.current is not None
        path_mode = args.changed_paths is not None or args.family_map is not None
        if shard_mode == path_mode:
            raise ValueError(
                "provide either --base/--current or --changed-paths/--family-map"
            )
        if shard_mode:
            if args.base is None or args.current is None:
                raise ValueError("--base and --current must be provided together")
            result = select(load(args.base), load(args.current), args.include_sentinels)
        else:
            if args.changed_paths is None or args.family_map is None:
                raise ValueError("--changed-paths and --family-map must be provided together")
            result = select_upstream_paths(
                args.changed_paths.read_text(encoding="utf-8").splitlines(),
                load_family_map(args.family_map),
                args.include_sentinels,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
