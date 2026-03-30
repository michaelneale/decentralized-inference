#!/usr/bin/env zsh
# build-mac.sh — build llama.cpp + mesh-llm on macOS Apple Silicon
#
# Usage:
#   scripts/build-mac.sh

setopt errexit nounset pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"

LLAMA_DIR="$REPO_ROOT/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
MESH_DIR="$REPO_ROOT/mesh-llm"
UI_DIR="$MESH_DIR/ui"

compiler_launcher_flags=()
rustc_wrapper=""

detect_jobs() {
    sysctl -n hw.ncpu 2>/dev/null || echo 4
}

configure_compiler_cache() {
    local cache_bin=""
    if (( $+commands[sccache] )); then
        cache_bin="sccache"
        rustc_wrapper="$cache_bin"
    elif (( $+commands[ccache] )); then
        cache_bin="ccache"
    else
        return
    fi

    echo "Using compiler cache: $cache_bin"
    compiler_launcher_flags=(
        -DCMAKE_C_COMPILER_LAUNCHER="$cache_bin"
        -DCMAKE_CXX_COMPILER_LAUNCHER="$cache_bin"
    )

    if [[ -n "$rustc_wrapper" ]]; then
        echo "Using Rust compiler wrapper: $rustc_wrapper"
    fi
}

clone_or_update_llama() {
    if [[ ! -d "$LLAMA_DIR" ]]; then
        echo "Cloning michaelneale/llama.cpp (rebase-upstream-master branch)..."
        git clone -b rebase-upstream-master https://github.com/michaelneale/llama.cpp.git "$LLAMA_DIR"
        return
    fi

    pushd "$LLAMA_DIR" >/dev/null
    local current_branch
    current_branch="$(git branch --show-current)"
    if [[ "$current_branch" != "rebase-upstream-master" ]]; then
        echo "⚠️  llama.cpp is on branch '$current_branch', switching to rebase-upstream-master..."
        git checkout rebase-upstream-master
    fi
    echo "Pulling latest rebase-upstream-master from origin..."
    git pull --ff-only origin rebase-upstream-master
    popd >/dev/null
}

clone_or_update_llama
configure_compiler_cache

cmake_flags=(
    -B "$BUILD_DIR"
    -S "$LLAMA_DIR"
    -DGGML_METAL=ON
    -DGGML_RPC=ON
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_OPENSSL=OFF
)

if (( $+commands[ninja] )); then
    cmake_flags=(-G Ninja "${cmake_flags[@]}")
fi

cmake_flags+=("${compiler_launcher_flags[@]}")

echo "Configuring llama.cpp for macOS..."
cmake "${cmake_flags[@]}"

echo "Building llama.cpp..."
cmake --build "$BUILD_DIR" --config Release --parallel "$(detect_jobs)"
echo "Build complete: $BUILD_DIR/bin/"

if [[ -d "$MESH_DIR" ]]; then
    echo "Building mesh-llm..."
    if [[ -d "$UI_DIR" ]]; then
        echo "Building mesh-llm UI..."
        (
            cd "$UI_DIR"
            npm ci
            npm run build
        )
    fi

    if [[ -n "$rustc_wrapper" ]]; then
        (
            cd "$REPO_ROOT"
            RUSTC_WRAPPER="$rustc_wrapper" cargo build --release
        )
    else
        (
            cd "$REPO_ROOT"
            cargo build --release
        )
    fi

    echo "Mesh binary: target/release/mesh-llm"
fi
