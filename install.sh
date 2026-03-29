#!/usr/bin/env bash

set -euo pipefail

REPO="${MESH_LLM_INSTALL_REPO:-michaelneale/mesh-llm}"
INSTALL_DIR="${MESH_LLM_INSTALL_DIR:-$HOME/.local/bin}"
INSTALL_FLAVOR="${MESH_LLM_INSTALL_FLAVOR:-}"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command not found: $1" >&2
        exit 1
    fi
}

path_contains_install_dir() {
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) return 0 ;;
        *) return 1 ;;
    esac
}

platform_id() {
    printf "%s/%s\n" "$(uname -s)" "$(uname -m)"
}

probe_nvidia() {
    command -v nvidia-smi >/dev/null 2>&1 ||
        command -v nvcc >/dev/null 2>&1 ||
        [[ -e /dev/nvidiactl ]] ||
        [[ -d /proc/driver/nvidia/gpus ]]
}

probe_rocm() {
    command -v rocm-smi >/dev/null 2>&1 ||
        command -v rocminfo >/dev/null 2>&1 ||
        command -v hipcc >/dev/null 2>&1 ||
        [[ -x /opt/rocm/bin/hipcc ]]
}

probe_vulkan() {
    if command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary >/dev/null 2>&1; then
        return 0
    fi
    if command -v glslc >/dev/null 2>&1; then
        if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists vulkan 2>/dev/null; then
            return 0
        fi
        if [[ -f /usr/include/vulkan/vulkan.h || -f /usr/local/include/vulkan/vulkan.h ]]; then
            return 0
        fi
        if [[ -n "${VULKAN_SDK:-}" ]]; then
            return 0
        fi
    fi
    return 1
}

supported_flavors() {
    case "$(platform_id)" in
        Darwin/arm64)
            echo "metal"
            ;;
        Linux/x86_64)
            echo "cpu cuda rocm vulkan"
            ;;
        *)
            echo "error: unsupported platform: $(platform_id)" >&2
            exit 1
            ;;
    esac
}

recommended_flavor() {
    case "$(platform_id)" in
        Darwin/arm64)
            echo "metal"
            ;;
        Linux/x86_64)
            if probe_nvidia; then
                echo "cuda"
            elif probe_rocm; then
                echo "rocm"
            elif probe_vulkan; then
                echo "vulkan"
            else
                echo "cpu"
            fi
            ;;
        *)
            echo "error: unsupported platform: $(platform_id)" >&2
            exit 1
            ;;
    esac
}

recommendation_reason() {
    case "$(recommended_flavor)" in
        metal)
            echo "Apple Silicon host detected."
            ;;
        cuda)
            echo "NVIDIA tooling or devices were detected."
            ;;
        rocm)
            echo "ROCm/HIP tooling was detected."
            ;;
        vulkan)
            echo "Vulkan tooling was detected."
            ;;
        cpu)
            echo "No supported GPU runtime was detected."
            ;;
    esac
}

validate_flavor() {
    local flavor="$1"
    local supported
    for supported in $(supported_flavors); do
        if [[ "$supported" == "$flavor" ]]; then
            return 0
        fi
    done
    echo "error: unsupported flavor '$flavor' for $(platform_id)" >&2
    exit 1
}

choose_flavor() {
    local recommended
    recommended="$(recommended_flavor)"

    if [[ -n "$INSTALL_FLAVOR" ]]; then
        validate_flavor "$INSTALL_FLAVOR"
        echo "$INSTALL_FLAVOR"
        return 0
    fi

    if [[ ! -t 0 || ! -t 1 ]]; then
        echo "$recommended"
        return 0
    fi

    local flavors
    flavors=($(supported_flavors))

    if [[ ${#flavors[@]} -eq 1 ]]; then
        echo "$recommended"
        return 0
    fi

    echo "Mesh LLM installer"
    echo "Platform: $(platform_id)"
    echo "Recommended flavor: $recommended"
    echo "Reason: $(recommendation_reason)"
    echo
    echo "Available flavors:"

    local index=1
    local flavor
    for flavor in "${flavors[@]}"; do
        if [[ "$flavor" == "$recommended" ]]; then
            echo "  $index. $flavor (recommended)"
        else
            echo "  $index. $flavor"
        fi
        index=$((index + 1))
    done

    echo
    local reply
    read -r -p "Install which flavor? [$recommended] " reply
    reply="${reply:-$recommended}"

    if [[ "$reply" =~ ^[0-9]+$ ]]; then
        local selection=$((reply - 1))
        if (( selection >= 0 && selection < ${#flavors[@]} )); then
            reply="${flavors[$selection]}"
        fi
    fi

    validate_flavor "$reply"
    echo "$reply"
}

asset_name() {
    local flavor="$1"
    case "$(platform_id)" in
        Darwin/arm64)
            echo "mesh-llm-aarch64-apple-darwin.tar.gz"
            ;;
        Linux/x86_64)
            case "$flavor" in
                cpu) echo "mesh-llm-x86_64-unknown-linux-gnu.tar.gz" ;;
                cuda) echo "mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz" ;;
                rocm) echo "mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz" ;;
                vulkan) echo "mesh-llm-x86_64-unknown-linux-gnu-vulkan.tar.gz" ;;
                *)
                    echo "error: unsupported Linux flavor '$flavor'" >&2
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "error: unsupported platform: $(platform_id)" >&2
            exit 1
            ;;
    esac
}

stale_binary_names() {
    cat <<'EOF'
mesh-llm
rpc-server
llama-server
llama-moe-split
rpc-server-cpu
llama-server-cpu
rpc-server-cuda
llama-server-cuda
rpc-server-rocm
llama-server-rocm
rpc-server-vulkan
llama-server-vulkan
rpc-server-metal
llama-server-metal
EOF
}

remove_stale_binaries() {
    mkdir -p "$INSTALL_DIR"
    local name
    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        rm -f "$INSTALL_DIR/$name"
    done < <(stale_binary_names)
}

install_bundle() {
    local bundle_dir="$1"
    remove_stale_binaries

    local file
    for file in "$bundle_dir"/*; do
        mv -f "$file" "$INSTALL_DIR/"
    done
}

main() {
    need_cmd curl
    need_cmd tar
    need_cmd mktemp

    local flavor
    flavor="$(choose_flavor)"
    local asset
    asset="$(asset_name "$flavor")"
    local url="https://github.com/${REPO}/releases/latest/download/${asset}"

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT

    local archive="$tmp_dir/$asset"
    echo "Installing flavor: $flavor"
    echo "Downloading $url"
    curl -fsSL "$url" -o "$archive"

    tar -xzf "$archive" -C "$tmp_dir"

    if [[ ! -d "$tmp_dir/mesh-bundle" ]]; then
        echo "error: release archive did not contain mesh-bundle/" >&2
        exit 1
    fi

    install_bundle "$tmp_dir/mesh-bundle"

    echo "Installed $asset to $INSTALL_DIR"

    if ! path_contains_install_dir; then
        echo
        echo "$INSTALL_DIR is not on your PATH."
        echo "Add it with one of these commands:"
        echo
        echo "bash:"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc"
        echo "  source ~/.bashrc"
        echo
        echo "zsh:"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.zshrc"
        echo "  source ~/.zshrc"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
