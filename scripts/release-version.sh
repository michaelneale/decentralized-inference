#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: scripts/release-version.sh <version|vversion>" >&2
    exit 1
fi

raw_version="$1"
version="${raw_version#v}"

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "invalid version: $raw_version" >&2
    echo "expected semantic version like 0.49.0 or v0.49.0" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

files=(
    "$REPO_ROOT/mesh-llm/src/main.rs"
    "$REPO_ROOT/mesh-llm/Cargo.toml"
    "$REPO_ROOT/mesh-llm/plugin/Cargo.toml"
    "$REPO_ROOT/mesh-llm/src/plugins/example/Cargo.toml"
)

perl -0pi -e 's/pub const VERSION: &str = "\K[^"]+(?=";)/'"$version"'/g' \
    "$REPO_ROOT/mesh-llm/src/main.rs"

for manifest in \
    "$REPO_ROOT/mesh-llm/Cargo.toml" \
    "$REPO_ROOT/mesh-llm/plugin/Cargo.toml" \
    "$REPO_ROOT/mesh-llm/src/plugins/example/Cargo.toml"
do
    perl -0pi -e 's/^version = "[^"]+"/version = "'"$version"'"/m' "$manifest"
done

echo "Refreshing Cargo.lock workspace package versions..."
(cd "$REPO_ROOT" && cargo metadata --format-version 1 >/dev/null)

files+=("$REPO_ROOT/Cargo.lock")

echo "Updated release version to $version:"
for file in "${files[@]}"; do
    echo "  $file"
done
