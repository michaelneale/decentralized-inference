#!/usr/bin/env python3
"""Bind a prepared release UI to its source revision and verify its exact bytes."""

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re


MANIFEST = ".mesh-llm-ui-release.json"


class ModuleScripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and attrs.get("type") == "module":
            self.sources.append(attrs.get("src", ""))


def describe(root: Path, source_sha: str, release_tag: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source SHA must be 40 lowercase hexadecimal characters")
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", release_tag):
        raise ValueError("release tag must be a versioned v-prefixed tag")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("UI distribution must be a real directory")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("UI distribution must not contain symbolic links")
        if path.is_file() and path.relative_to(root).as_posix() != MANIFEST:
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = root / "index.html"
    if "index.html" not in files:
        raise ValueError("UI distribution is missing index.html")
    parser = ModuleScripts()
    parser.feed(entry.read_text(encoding="utf-8"))
    if not any(
        src.removeprefix("/") in files and src.endswith(".js")
        for src in parser.sources
    ):
        raise ValueError("UI index must reference a built local JavaScript module")
    return {"schema": 1, "source_sha": source_sha, "release_tag": release_tag, "files": files}


def stamp(root: Path, source_sha: str, release_tag: str) -> None:
    manifest = describe(root, source_sha, release_tag)
    (root / MANIFEST).write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def verify(root: Path, source_sha: str, release_tag: str) -> None:
    expected = describe(root, source_sha, release_tag)
    manifest = root / MANIFEST
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("UI distribution is missing its release manifest")
    if json.loads(manifest.read_text(encoding="utf-8")) != expected:
        raise ValueError("UI release identity or file checksums do not match")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stamp", "verify"))
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args()
    try:
        {"stamp": stamp, "verify": verify}[args.operation](args.dist, args.source_sha, args.release_tag)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
