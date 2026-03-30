#!/usr/bin/env bash
# build-linux.sh — build llama.cpp + mesh-llm on Linux
#
# Usage:
#   scripts/build-linux.sh [--clean] [--backend cpu|cuda|rocm|vulkan] [--cuda-arch SM_LIST] [--rocm-arch GFX_LIST]
#
# Examples:
#   scripts/build-linux.sh
#   scripts/build-linux.sh --backend cpu
#   scripts/build-linux.sh --backend cuda --cuda-arch '120;86'
#   scripts/build-linux.sh --backend rocm --rocm-arch 'gfx942;gfx90a'
#   scripts/build-linux.sh --backend vulkan
#
# Must be run from the repository root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_DIR="$REPO_ROOT/llama.cpp"
BUILD_DIR="$LLAMA_DIR/build"
MESH_DIR="$REPO_ROOT/mesh-llm"
UI_DIR="$MESH_DIR/ui"

CLEAN=0
BACKEND=""
CUDA_ARCH=""
ROCM_ARCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
            shift
            ;;
        --backend)
            BACKEND="${2:-}"
            shift 2
            ;;
        --cuda-arch)
            CUDA_ARCH="${2:-}"
            shift 2
            ;;
        --rocm-arch)
            ROCM_ARCH="${2:-}"
            shift 2
            ;;
        *)
            # Backward compatibility: treat a bare arg as cuda_arch.
            [[ -z "$CUDA_ARCH" ]] && CUDA_ARCH="$1"
            shift
            ;;
    esac
done

detect_backend() {
    if command -v nvidia-smi &>/dev/null; then
        echo cuda
        return 0
    fi
    if command -v tegrastats &>/dev/null; then
        echo cuda
        return 0
    fi
    if command -v nvcc &>/dev/null; then
        echo cuda
        return 0
    fi
    if command -v rocm-smi &>/dev/null; then
        echo rocm
        return 0
    fi
    if command -v rocminfo &>/dev/null; then
        echo rocm
        return 0
    fi
    if command -v hipcc &>/dev/null; then
        echo rocm
        return 0
    fi
    if [[ -x /opt/rocm/bin/hipcc ]]; then
        echo rocm
        return 0
    fi
    if command -v glslc &>/dev/null; then
        if command -v vulkaninfo &>/dev/null && vulkaninfo --summary >/dev/null 2>&1; then
            echo vulkan
            return 0
        fi
        if pkg-config --exists vulkan 2>/dev/null; then
            echo vulkan
            return 0
        fi
        if [[ -n "${VULKAN_SDK:-}" ]]; then
            echo vulkan
            return 0
        fi
    fi
    echo cpu
}

locate_nvcc() {
    if command -v nvcc &>/dev/null; then
        return 0
    fi
    for CANDIDATE in /usr/local/cuda/bin /opt/cuda/bin /usr/cuda/bin; do
        if [[ -x "$CANDIDATE/nvcc" ]]; then
            export PATH="$CANDIDATE:$PATH"
            return 0
        fi
    done
    return 1
}

locate_hip_toolchain() {
    if command -v hipcc &>/dev/null; then
        return 0
    fi
    for CANDIDATE in /opt/rocm/bin /usr/lib/rocm/bin /usr/local/rocm/bin; do
        if [[ -x "$CANDIDATE/hipcc" ]]; then
            export PATH="$CANDIDATE:$PATH"
            return 0
        fi
    done
    return 1
}

locate_vulkan_toolchain() {
    if ! command -v glslc &>/dev/null; then
        if [[ -n "${VULKAN_SDK:-}" && -x "$VULKAN_SDK/bin/glslc" ]]; then
            export PATH="$VULKAN_SDK/bin:$PATH"
        else
            return 1
        fi
    fi

    if pkg-config --exists vulkan 2>/dev/null; then
        return 0
    fi

    if [[ -f /usr/include/vulkan/vulkan.h || -f /usr/local/include/vulkan/vulkan.h ]]; then
        return 0
    fi

    if [[ -n "${VULKAN_SDK:-}" ]]; then
        export CMAKE_PREFIX_PATH="${VULKAN_SDK}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
        if [[ -f "$VULKAN_SDK/include/vulkan/vulkan.h" ]]; then
            return 0
        fi
    fi

    return 1
}

compiler_launcher_flags=()

configure_compiler_cache() {
    local backend="$1"
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

    case "$backend" in
        cuda)
            compiler_launcher_flags+=(-DCMAKE_CUDA_COMPILER_LAUNCHER="$cache_bin")
            ;;
        rocm)
            compiler_launcher_flags+=(-DCMAKE_HIP_COMPILER_LAUNCHER="$cache_bin")
            ;;
    esac
}

if [[ -z "$BACKEND" ]]; then
    BACKEND="$(detect_backend)"
fi

case "$BACKEND" in
    cuda)
        locate_nvcc || {
            echo "Error: nvcc not found. Install the CUDA toolkit and ensure nvcc is in your PATH." >&2
            echo "  Arch Linux:    sudo pacman -S cuda" >&2
            echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit" >&2
            exit 1
        }
        if [[ -z "$CUDA_ARCH" ]]; then
            echo "No cuda_arch specified — running auto-detection..."
            CUDA_ARCH="$("$SCRIPT_DIR/detect-cuda-arch.sh")"
            echo "Using SM ${CUDA_ARCH}"
        fi
        echo "Building Linux backend: CUDA"
        echo "Using nvcc: $(command -v nvcc) ($(nvcc --version | grep release | awk '{print $5}' | tr -d ','))"
        ;;
    rocm)
        locate_hip_toolchain || {
            echo "Error: hipcc not found. Install ROCm and ensure hipcc is in your PATH." >&2
            echo "  Typical location: /opt/rocm/bin/hipcc" >&2
            exit 1
        }
        if [[ -z "$ROCM_ARCH" ]]; then
            echo "No rocm_arch specified — running auto-detection..."
            ROCM_ARCH="$("$SCRIPT_DIR/detect-rocm-arch.sh")"
            echo "Using AMDGPU_TARGETS ${ROCM_ARCH}"
        fi
        echo "Building Linux backend: ROCm/HIP"
        echo "Using hipcc: $(command -v hipcc)"
        ;;
    vulkan)
        locate_vulkan_toolchain || {
            echo "Error: Vulkan SDK/development files not found." >&2
            echo "  Need both the Vulkan headers/loader and 'glslc' in your PATH." >&2
            echo "  Ubuntu/Debian: sudo apt install libvulkan-dev glslc" >&2
            echo "  Arch Linux:    sudo pacman -S vulkan-headers shaderc" >&2
            exit 1
        }
        echo "Building Linux backend: Vulkan"
        echo "Using glslc: $(command -v glslc)"
        ;;
    cpu)
        echo "Building Linux backend: CPU only (no GPU acceleration)"
        ;;
    *)
        echo "Error: unsupported backend '$BACKEND' (expected 'cpu', 'cuda', 'rocm', or 'vulkan')." >&2
        exit 1
        ;;
esac

if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "Cloning michaelneale/llama.cpp (rebase-upstream-master)..."
    git clone -b rebase-upstream-master \
        https://github.com/michaelneale/llama.cpp.git "$LLAMA_DIR"
else
    cd "$LLAMA_DIR"
    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != "rebase-upstream-master" ]]; then
        echo "⚠️  llama.cpp is on branch '$CURRENT_BRANCH', switching to rebase-upstream-master..."
        git checkout rebase-upstream-master
    fi
    echo "Pulling latest rebase-upstream-master from origin..."
    git pull --ff-only origin rebase-upstream-master
    cd "$REPO_ROOT"
fi

if [[ "$CLEAN" -eq 1 && -d "$BUILD_DIR" ]]; then
    echo "Cleaning build dir..."
    rm -rf "$BUILD_DIR"
fi

configure_compiler_cache "$BACKEND"

cmake_flags=(
    -B "$BUILD_DIR"
    -S "$LLAMA_DIR"
    -DGGML_RPC=ON
    -DBUILD_SHARED_LIBS=OFF
    -DLLAMA_OPENSSL=OFF
)

if [[ "$BACKEND" == "cpu" ]]; then
    cmake_flags+=(
        -DGGML_CUDA=OFF
        -DGGML_HIP=OFF
        -DGGML_VULKAN=OFF
        -DGGML_METAL=OFF
    )
elif [[ "$BACKEND" == "cuda" ]]; then
    cmake_flags+=(
        -DGGML_CUDA=ON
        -DGGML_HIP=OFF
        -DGGML_VULKAN=OFF
        -DGGML_METAL=OFF
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    )
elif [[ "$BACKEND" == "rocm" ]]; then
    if command -v hipconfig &>/dev/null; then
        export HIPCXX="$(hipconfig -l)/clang"
        export HIP_PATH="$(hipconfig -R)"
    fi
    cmake_flags+=(
        -DGGML_CUDA=OFF
        -DGGML_HIP=ON
        -DGGML_VULKAN=OFF
        -DGGML_METAL=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DAMDGPU_TARGETS="$ROCM_ARCH"
    )
else
    cmake_flags+=(
        -DGGML_CUDA=OFF
        -DGGML_HIP=OFF
        -DGGML_VULKAN=ON
        -DGGML_METAL=OFF
    )
fi

cmake_flags+=("${compiler_launcher_flags[@]}")

cmake "${cmake_flags[@]}"
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"
echo "llama.cpp build complete: $BUILD_DIR/bin/"

if [[ -d "$MESH_DIR" ]]; then
    if [[ -d "$UI_DIR" ]]; then
        echo "Building mesh-llm UI..."
        (cd "$UI_DIR" && npm ci && npm run build)
    fi
    echo "Building mesh-llm..."
    (cd "$MESH_DIR" && cargo build --release)
    echo "Mesh binary: target/release/mesh-llm"
fi
