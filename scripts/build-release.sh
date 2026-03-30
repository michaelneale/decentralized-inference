#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="$REPO_ROOT/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
UI_DIR="$REPO_ROOT/mesh-llm/ui"

detect_jobs() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
    else
        echo 4
    fi
}

compiler_launcher_flags=()

configure_compiler_cache() {
    local cache_bin=""
    if command -v sccache >/dev/null 2>&1; then
        cache_bin="sccache"
    elif command -v ccache >/dev/null 2>&1; then
        cache_bin="ccache"
    else
        return
    fi

    echo "Using compiler cache: $cache_bin"
    compiler_launcher_flags=(
        -DCMAKE_C_COMPILER_LAUNCHER="$cache_bin"
        -DCMAKE_CXX_COMPILER_LAUNCHER="$cache_bin"
    )
}

clone_or_update_llama() {
    if [[ ! -d "$LLAMA_DIR" ]]; then
        echo "Cloning michaelneale/llama.cpp (rebase-upstream-master)..."
        git clone -b rebase-upstream-master \
            https://github.com/michaelneale/llama.cpp.git "$LLAMA_DIR"
        return
    fi

    pushd "$LLAMA_DIR" >/dev/null
    current_branch="$(git branch --show-current)"
    if [[ "$current_branch" != "rebase-upstream-master" ]]; then
        echo "Switching llama.cpp from '$current_branch' to 'rebase-upstream-master'..."
        git checkout rebase-upstream-master
    fi
    git pull --ff-only origin rebase-upstream-master
    popd >/dev/null
}

os_name="$(uname -s)"
cmake_flags=(
    -B "$BUILD_DIR"
    -S "$LLAMA_DIR"
    -DGGML_RPC=ON
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_OPENSSL=OFF
)

case "$os_name" in
    Darwin)
        cmake_flags+=(
            -DGGML_METAL=ON
        )
        ;;
    Linux)
        cmake_flags+=(
            -DGGML_CUDA=OFF
            -DGGML_HIP=OFF
            -DGGML_METAL=OFF
            -DGGML_VULKAN=OFF
        )
        ;;
    *)
        echo "Unsupported OS for release build: $os_name" >&2
        exit 1
        ;;
esac

if command -v ninja >/dev/null 2>&1; then
    cmake_flags=(-G Ninja "${cmake_flags[@]}")
fi

configure_compiler_cache
cmake_flags+=("${compiler_launcher_flags[@]}")

clone_or_update_llama

echo "Configuring llama.cpp for $os_name..."
cmake "${cmake_flags[@]}"

echo "Building llama.cpp..."
cmake --build "$BUILD_DIR" --config Release --parallel "$(detect_jobs)"

echo "Building UI..."
(
    cd "$UI_DIR"
    npm ci
    npm run build
)

echo "Building mesh-llm..."
(
    cd "$REPO_ROOT"
    cargo build --release --locked -p mesh-llm
)
