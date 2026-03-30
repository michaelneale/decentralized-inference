#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || -z "${1:-}" ]]; then
    echo "usage: scripts/package-release.sh <version> [output_dir]" >&2
    exit 1
fi

VERSION="$1"
OUTPUT_DIR="${2:-dist}"
RELEASE_FLAVOR="${MESH_RELEASE_FLAVOR:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_BIN_DIR="$REPO_ROOT/llama.cpp/build/bin"
RELEASE_BIN_DIR="$REPO_ROOT/target/release"

python_bin() {
    if command -v python3 >/dev/null 2>&1; then
        echo python3
    elif command -v python >/dev/null 2>&1; then
        echo python
    else
        echo "python3 or python is required for packaging" >&2
        exit 1
    fi
}

copy_runtime_libs() {
    local bundle_dir="$1"
    shopt -s nullglob
    case "$(uname -s)" in
        Darwin)
            for lib in "$BUILD_BIN_DIR"/*.dylib; do
                cp "$lib" "$bundle_dir/"
            done
            ;;
        Linux)
            for lib in "$BUILD_BIN_DIR"/*.so "$BUILD_BIN_DIR"/*.so.*; do
                cp "$lib" "$bundle_dir/"
            done
            ;;
    esac
    shopt -u nullglob
}

bundle_bin_name() {
    local name="$1"
    if [[ "$name" == "mesh-llm" ]]; then
        echo "$name"
        return
    fi

    local binary_flavor="$RELEASE_FLAVOR"
    if [[ -z "$binary_flavor" ]]; then
        case "$(uname -s)" in
            Darwin) binary_flavor="metal" ;;
            Linux) binary_flavor="cpu" ;;
        esac
    fi

    if [[ -n "$binary_flavor" ]]; then
        echo "${name}-${binary_flavor}"
    else
        echo "$name"
    fi
}

create_archive() {
    local source_dir="$1"
    local archive_path="$2"
    local archive_kind="$3"
    local py
    py="$(python_bin)"

    rm -f "$archive_path"
    mkdir -p "$(dirname "$archive_path")"

    "$py" - "$source_dir" "$archive_path" "$archive_kind" <<'PY'
import os
import sys
import tarfile
import zipfile

source_dir, archive_path, archive_kind = sys.argv[1:4]
base = os.path.basename(os.path.normpath(source_dir))
root = os.path.dirname(os.path.normpath(source_dir))

if archive_kind == "tar.gz":
    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(source_dir, arcname=base)
elif archive_kind == "zip":
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for current_root, dirs, files in os.walk(source_dir):
            dirs.sort()
            files.sort()
            rel_root = os.path.relpath(current_root, root)
            if rel_root != ".":
                zf.write(current_root, rel_root)
            for filename in files:
                path = os.path.join(current_root, filename)
                rel = os.path.relpath(path, root)
                zf.write(path, rel)
else:
    raise SystemExit(f"unsupported archive kind: {archive_kind}")
PY
}

os_name="$(uname -s)"
case "$os_name" in
    Darwin)
        TARGET_TRIPLE="aarch64-apple-darwin"
        BIN_EXT=""
        ARCHIVE_EXT="tar.gz"
        STABLE_ASSET="mesh-llm-${TARGET_TRIPLE}.tar.gz"
        LEGACY_ASSET="mesh-bundle.tar.gz"
        ;;
    Linux)
        TARGET_TRIPLE="x86_64-unknown-linux-gnu"
        BIN_EXT=""
        ARCHIVE_EXT="tar.gz"
        STABLE_ASSET="mesh-llm-${TARGET_TRIPLE}.tar.gz"
        LEGACY_ASSET=""
        ;;
    *)
        echo "Unsupported OS for packaging: $os_name" >&2
        exit 1
        ;;
esac

if [[ -n "$RELEASE_FLAVOR" ]]; then
    TARGET_TRIPLE="${TARGET_TRIPLE}-${RELEASE_FLAVOR}"
    STABLE_ASSET="mesh-llm-${TARGET_TRIPLE}.${ARCHIVE_EXT}"
fi

VERSIONED_ASSET="mesh-llm-${VERSION}-${TARGET_TRIPLE}.${ARCHIVE_EXT}"

mkdir -p "$OUTPUT_DIR"
staging_dir="$(mktemp -d)"
trap 'rm -rf "$staging_dir"' EXIT

bundle_dir="$staging_dir/mesh-bundle"
mkdir -p "$bundle_dir"

cp "$RELEASE_BIN_DIR/mesh-llm${BIN_EXT}" "$bundle_dir/$(bundle_bin_name mesh-llm)"
cp "$BUILD_BIN_DIR/rpc-server${BIN_EXT}" "$bundle_dir/$(bundle_bin_name rpc-server)"
cp "$BUILD_BIN_DIR/llama-server${BIN_EXT}" "$bundle_dir/$(bundle_bin_name llama-server)"
cp "$BUILD_BIN_DIR/llama-moe-split${BIN_EXT}" "$bundle_dir/llama-moe-split"
copy_runtime_libs "$bundle_dir"

if [[ "$os_name" == "Darwin" ]]; then
    for bin in "$bundle_dir/mesh-llm" "$bundle_dir/rpc-server" "$bundle_dir/llama-server" "$bundle_dir/llama-moe-split"; do
        [[ -f "$bin" ]] || continue
        install_name_tool -add_rpath @executable_path/ "$bin" 2>/dev/null || true
    done
fi

create_archive "$bundle_dir" "$OUTPUT_DIR/$VERSIONED_ASSET" "$ARCHIVE_EXT"
create_archive "$bundle_dir" "$OUTPUT_DIR/$STABLE_ASSET" "$ARCHIVE_EXT"

if [[ -n "$LEGACY_ASSET" ]]; then
    cp "$OUTPUT_DIR/$STABLE_ASSET" "$OUTPUT_DIR/$LEGACY_ASSET"
fi

echo "Created release archives:"
find "$OUTPUT_DIR" -maxdepth 1 -type f -print | sort
