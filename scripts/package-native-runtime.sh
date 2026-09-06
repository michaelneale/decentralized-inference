#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/cuda-toolkit.sh"

BUILD=0
OUT_DIR="$REPO_ROOT/dist/native-runtimes"
BACKEND="${LLAMA_STAGE_BACKEND:-${SKIPPY_LLAMA_BACKEND:-cpu}}"
TARGET_TRIPLE="${MESH_NATIVE_RUNTIME_TARGET:-}"
LLAMA_WORKDIR="${LLAMA_WORKDIR:-$REPO_ROOT/.deps/llama.cpp}"

usage() {
    cat >&2 <<'EOF'
Usage: scripts/package-native-runtime.sh [options]

Package a MeshLLM native runtime artifact containing the patched llama/Skippy
shared libraries selected by `mesh-llm runtime install`.

Options:
  --build             Build patched llama.cpp shared libraries before packaging.
  --backend NAME      cpu, metal, cuda, rocm, hip, vulkan, or cuda-blackwell.
  --target TRIPLE     Runtime target triple. Defaults to the host target.
  --out DIR           Output directory. Defaults to dist/native-runtimes.
  -h, --help          Show this help.

Environment:
  LLAMA_STAGE_CUDA_ARCHITECTURES / SKIPPY_CUDA_ARCHITECTURES
  LLAMA_STAGE_AMDGPU_TARGETS / SKIPPY_AMDGPU_TARGETS
  LLAMA_STAGE_BUILD_DIR
  MESH_NATIVE_RUNTIME_TARGET
  MESH_NATIVE_RUNTIME_MODEL_PACKAGE_TOOL (non-Windows test/packaging override)
  MESH_CUDA_VERSION / MESH_LLM_CUDA_TOOLKIT_MAJOR (validated against the selected compiler)
  MESH_LLM_LLAMA_PIN_SHA
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --build)
            BUILD=1
            shift
            ;;
        --backend)
            BACKEND="${2:?missing backend}"
            shift 2
            ;;
        --target)
            TARGET_TRIPLE="${2:?missing target triple}"
            shift 2
            ;;
        --out)
            OUT_DIR="${2:?missing output directory}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

case "$BACKEND" in
    cpu|metal|cuda|cuda-blackwell|rocm|hip|vulkan) ;;
    *)
        echo "unsupported native runtime backend: $BACKEND" >&2
        exit 1
        ;;
esac

host_os() {
    case "$(uname -s)" in
        Darwin) printf 'darwin\n' ;;
        Linux) printf 'linux\n' ;;
        MINGW*|MSYS*|CYGWIN*) printf 'windows\n' ;;
        *) uname -s | tr '[:upper:]' '[:lower:]' ;;
    esac
}

host_arch() {
    case "$(uname -m)" in
        arm64|aarch64) printf 'aarch64\n' ;;
        x86_64|amd64) printf 'x86_64\n' ;;
        *) uname -m ;;
    esac
}

default_target_triple() {
    case "$(host_os)/$(host_arch)" in
        darwin/aarch64) printf 'aarch64-apple-darwin\n' ;;
        darwin/x86_64) printf 'x86_64-apple-darwin\n' ;;
        linux/x86_64) printf 'x86_64-unknown-linux-gnu\n' ;;
        linux/aarch64) printf 'aarch64-unknown-linux-gnu\n' ;;
        windows/x86_64) printf 'x86_64-pc-windows-msvc\n' ;;
        *) printf '\n' ;;
    esac
}

target_platform() {
    case "$1" in
        aarch64-apple-darwin) printf 'darwin-aarch64\n' ;;
        x86_64-apple-darwin) printf 'darwin-x86_64\n' ;;
        x86_64-unknown-linux-gnu) printf 'linux-x86_64\n' ;;
        aarch64-unknown-linux-gnu) printf 'linux-aarch64\n' ;;
        x86_64-pc-windows-msvc) printf 'windows-x86_64\n' ;;
        *) printf '%s\n' "$1" | tr '_' '-' ;;
    esac
}

target_runtime_os() {
    case "$1" in
        *apple-darwin) printf 'macos\n' ;;
        *linux*) printf 'linux\n' ;;
        *windows*) printf 'windows\n' ;;
        *) echo "cannot infer runtime os for target: $1" >&2; exit 1 ;;
    esac
}

target_runtime_arch() {
    case "$1" in
        aarch64-*) printf 'aarch64\n' ;;
        x86_64-*) printf 'x86_64\n' ;;
        armv7-*) printf 'arm\n' ;;
        *) echo "cannot infer runtime arch for target: $1" >&2; exit 1 ;;
    esac
}

sanitize_component() {
    printf '%s' "$1" | tr ';, /:' '_____' | tr -cd 'A-Za-z0-9_.-'
}

backend_flavor() {
    local cuda_major
    case "$BACKEND" in
        cuda)
            if ! cuda_major="$(cuda_toolkit_major)"; then
                return 1
            fi
            printf 'cuda%s\n' "$cuda_major"
            ;;
        cuda-blackwell)
            if ! cuda_major="$(cuda_toolkit_major)"; then
                return 1
            fi
            printf 'cuda%s-sm120\n' "$cuda_major"
            ;;
        rocm|hip) printf 'rocm\n' ;;
        *) printf '%s\n' "$BACKEND" ;;
    esac
}

cuda_toolkit_major() {
    if [[ -z "$_mesh_cuda_toolkit_manifest_major_cache" ]]; then
        if ! _mesh_cuda_toolkit_manifest_major_cache="$(cuda_toolkit_manifest_major)"; then
            return 1
        fi
    fi
    printf '%s\n' "$_mesh_cuda_toolkit_manifest_major_cache"
}

build_backend() {
    case "$BACKEND" in
        cuda-blackwell) printf 'cuda\n' ;;
        hip) printf 'rocm\n' ;;
        *) printf '%s\n' "$BACKEND" ;;
    esac
}

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        echo "shasum or sha256sum is required" >&2
        exit 1
    fi
}

python_bin() {
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    echo "Python 3.9 or newer is required to package native runtimes" >&2
    exit 1
}

workspace_version() {
    "$(python_bin)" - "$REPO_ROOT/Cargo.toml" <<'PY'
import re
import sys

in_workspace_package = False
for line in open(sys.argv[1], encoding="utf-8"):
    stripped = line.strip()
    if stripped == "[workspace.package]":
        in_workspace_package = True
        continue
    if stripped.startswith("[") and stripped != "[workspace.package]":
        in_workspace_package = False
    if in_workspace_package:
        match = re.match(r'version\s*=\s*"([^"]+)"', stripped)
        if match:
            print(match.group(1))
            raise SystemExit(0)
raise SystemExit("workspace package version not found")
PY
}

skippy_abi_version() {
    "$(python_bin)" - "$REPO_ROOT/crates/skippy-ffi/src/lib.rs" <<'PY'
import re
import sys

values = {}
for line in open(sys.argv[1], encoding="utf-8"):
    match = re.match(r"pub const ABI_VERSION_(MAJOR|MINOR|PATCH): u32 = ([0-9]+);", line.strip())
    if match:
        values[match.group(1)] = match.group(2)
print("{}.{}.{}".format(values["MAJOR"], values["MINOR"], values["PATCH"]))
PY
}

library_pattern() {
    case "$TARGET_TRIPLE" in
        *apple-darwin) printf '*.dylib\n' ;;
        *windows*) printf '*.dll\n' ;;
        *) printf '*.so*\n' ;;
    esac
}

primary_library_names() {
    case "$TARGET_TRIPLE" in
        *apple-darwin) printf 'libllama.dylib\n' ;;
        *windows*) printf 'llama.dll\nlibllama.dll\n' ;;
        *) printf 'libllama.so\n' ;;
    esac
}

gpu_benchmark_tool_path() {
    if [[ "$runtime_os" == "windows" ]]; then
        printf 'tools/mesh-llm-gpu-benchmark.exe\n'
    else
        printf 'tools/mesh-llm-gpu-benchmark\n'
    fi
}

model_package_tool_path() {
    printf 'tools/skippy-model-package\n'
}

hip_offload_arch_args() {
    local raw arch
    local -a arches=()
    raw="${LLAMA_STAGE_AMDGPU_TARGETS:-${SKIPPY_AMDGPU_TARGETS:-}}"
    [[ -n "$raw" ]] || return 0

    raw="${raw//;/ }"
    raw="${raw//,/ }"
    read -r -a arches <<< "$raw"
    for arch in "${arches[@]}"; do
        [[ -n "$arch" ]] && printf -- '--offload-arch=%s\n' "$arch"
    done
}

build_gpu_benchmark_tool() {
    local tool_rel tool_path source_root compiler arch_arg
    local -a hip_arch_args=()
    case "$BACKEND" in
        cuda|cuda-blackwell|rocm|hip|metal) ;;
        *) return 0 ;;
    esac

    tool_rel="$(gpu_benchmark_tool_path)"
    tool_path="$stage_dir/$tool_rel"
    source_root="$REPO_ROOT/crates/mesh-llm-gpu-bench/native"
    mkdir -p "$(dirname "$tool_path")"

    case "$BACKEND" in
        cuda|cuda-blackwell)
            compiler="$(cuda_selected_compiler)"
            "$compiler" -O3 -std=c++17 "$source_root/cuda/membench-fingerprint.cu" -o "$tool_path"
            ;;
        rocm|hip)
            compiler="${HIPCC:-hipcc}"
            while IFS= read -r arch_arg; do
                hip_arch_args+=("$arch_arg")
            done < <(hip_offload_arch_args)
            "$compiler" -O3 -std=c++17 "${hip_arch_args[@]}" \
                "$source_root/hip/membench-fingerprint.hip" -o "$tool_path"
            ;;
        metal)
            compiler="${CC:-clang}"
            "$compiler" -O3 -fobjc-arc \
                "$source_root/metal/membench_metal.m" \
                "$source_root/metal/membench_main.m" \
                -framework Foundation -framework Metal -o "$tool_path"
            ;;
    esac

    case "$runtime_os" in
        linux) patchelf --set-rpath "\$ORIGIN/../lib" "$tool_path" ;;
        macos) install_name_tool -add_rpath '@loader_path/../lib' "$tool_path" ;;
    esac
    chmod +x "$tool_path"
    tool_paths+=("$tool_rel")
}

build_model_package_tool() {
    # The package tool links against the staged Skippy DLLs. Windows native
    # runtime producers do not have a robust import-library path for that
    # dynamic link, and the current consumers do not need this offline tool.
    # Keep Windows runtime artifacts limited to the established DLL producer
    # path until an import-library mechanism is available.
    if [[ "$runtime_os" == "windows" ]]; then
        return 0
    fi

    local tool_rel tool_path source_path configured cargo_target_dir
    local -a cargo_env=(
        "LLAMA_STAGE_LINK_MODE=dynamic"
        "LLAMA_STAGE_LIB_DIR=$stage_dir/lib"
        "LLAMA_STAGE_BUILD_DIR=$LLAMA_STAGE_BUILD_DIR"
        "LLAMA_STAGE_BACKEND=$(build_backend)"
    )
    tool_rel="$(model_package_tool_path)"
    tool_path="$stage_dir/$tool_rel"
    configured="${MESH_NATIVE_RUNTIME_MODEL_PACKAGE_TOOL:-}"
    mkdir -p "$(dirname "$tool_path")"

    if [[ -n "$configured" ]]; then
        if [[ ! -x "$configured" ]]; then
            echo "configured model package tool is not executable: $configured" >&2
            exit 1
        fi
        source_path="$configured"
    else
        if [[ "$runtime_os" == "macos" ]]; then
            if command -v ld64.lld >/dev/null 2>&1; then
                # Cargo's encoded flags override the checked-in target
                # rustflags, whose absolute `-fuse-ld=/path/to/ld64.lld`
                # form is rejected by Apple clang. Prefer the portable LLD
                # driver name when the producer installed it.
                cargo_env+=("CARGO_ENCODED_RUSTFLAGS=-Clink-arg=-fuse-ld=lld")
            else
                # Protected reusable workflows may not include the repository
                # setup action. An explicitly empty encoded flag set still
                # overrides the non-portable checked-in target rustflags and
                # lets Apple clang use the system linker.
                cargo_env+=("CARGO_ENCODED_RUSTFLAGS=")
            fi
        fi
        env "${cargo_env[@]}" \
            cargo build --release --locked --target "$TARGET_TRIPLE" \
                -p skippy-model-package
        cargo_target_dir="$(
            cargo metadata --no-deps --format-version 1 |
                "$(python_bin)" -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])'
        )"
        source_path="$cargo_target_dir/$TARGET_TRIPLE/release/$(basename "$tool_rel")"
        if [[ ! -x "$source_path" ]]; then
            echo "model package tool build did not produce $source_path" >&2
            exit 1
        fi
    fi

    cp "$source_path" "$tool_path"
    case "$runtime_os" in
        linux)
            patchelf --set-rpath "\$ORIGIN/../lib" "$tool_path"
            ;;
        macos)
            if ! otool -l "$tool_path" | awk '
                $1 == "cmd" && $2 == "LC_RPATH" { in_rpath = 1; next }
                in_rpath && $1 == "path" { print $2; in_rpath = 0 }
            ' | grep -qx '@loader_path/../lib'; then
                install_name_tool -add_rpath '@loader_path/../lib' "$tool_path"
            fi
            ;;
    esac
    chmod +x "$tool_path"
    tool_paths+=("$tool_rel")
}

collect_runtime_libraries() {
    local pattern primary_names
    pattern="$(library_pattern)"
    primary_names="$(primary_library_names | tr '\n' ' ')"
    find "$LLAMA_STAGE_BUILD_DIR" \( -type f -o -type l \) -name "$pattern" \
        ! -path '*/CMakeFiles/*' \
        | sort \
        | awk -v primary_names="$primary_names" '
            BEGIN {
                primary_count = split(primary_names, names, " ")
                for (idx = 1; idx <= primary_count; idx++) {
                    if (names[idx] != "") primary[names[idx]] = idx
                }
            }
            {
                name = $0
                sub(/^.*\//, "", name)
                paths[++path_count] = $0
                if (name in primary) primary_paths[name] = $0
            }
            END {
                chosen_primary = ""
                for (idx = 1; idx <= primary_count; idx++) {
                    if (names[idx] in primary_paths) {
                        chosen_primary = primary_paths[names[idx]]
                        break
                    }
                }
                for (idx = 1; idx <= path_count; idx++) {
                    if (paths[idx] != chosen_primary) print paths[idx]
                }
                if (chosen_primary != "") print chosen_primary
            }
        '
}

rewrite_macos_runtime_paths() {
    case "$TARGET_TRIPLE" in
        *apple-darwin) ;;
        *) return 0 ;;
    esac
    if ! command -v install_name_tool >/dev/null 2>&1; then
        echo "install_name_tool is required to package macOS native runtimes" >&2
        exit 1
    fi
    if ! command -v otool >/dev/null 2>&1; then
        echo "otool is required to package macOS native runtimes" >&2
        exit 1
    fi

    local rel_path library name dep dep_name candidate candidate_name
    for rel_path in "${library_paths[@]}"; do
        library="$stage_dir/$rel_path"
        name="$(basename "$library")"
        install_name_tool -id "@rpath/$name" "$library"
        if ! otool -l "$library" | awk '
            $1 == "cmd" && $2 == "LC_RPATH" { in_rpath = 1; next }
            in_rpath && $1 == "path" { print $2; in_rpath = 0 }
        ' | grep -qx '@loader_path'; then
            install_name_tool -add_rpath "@loader_path" "$library"
        fi
    done

    for rel_path in "${library_paths[@]}"; do
        library="$stage_dir/$rel_path"
        while IFS= read -r dep; do
            dep_name="$(basename "$dep")"
            for candidate in "${library_paths[@]}"; do
                candidate_name="$(basename "$candidate")"
                if [[ "$dep_name" == "$candidate_name" && "$dep" != "@rpath/$candidate_name" ]]; then
                    install_name_tool -change "$dep" "@rpath/$candidate_name" "$library"
                fi
            done
        done < <(otool -L "$library" | awk 'NR > 1 { print $1 }')
    done
}

rewrite_linux_runtime_paths() {
    case "$TARGET_TRIPLE" in
        *linux*) ;;
        *) return 0 ;;
    esac
    if ! command -v patchelf >/dev/null 2>&1; then
        echo "patchelf is required to package Linux native runtimes" >&2
        exit 1
    fi

    local rel_path library
    for rel_path in "${library_paths[@]}"; do
        library="$stage_dir/$rel_path"
        patchelf --set-rpath "\$ORIGIN" "$library"
    done
}

if [[ -z "$TARGET_TRIPLE" ]]; then
    TARGET_TRIPLE="$(default_target_triple)"
fi
if [[ -z "$TARGET_TRIPLE" ]]; then
    echo "could not infer target triple; pass --target" >&2
    exit 1
fi

if [[ -z "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
    LLAMA_STAGE_BUILD_DIR="$(LLAMA_STAGE_LINK_MODE=dynamic LLAMA_STAGE_BACKEND="$(build_backend)" "$SCRIPT_DIR/build-llama.sh" --print-build-dir)"
fi

if [[ "$BUILD" == "1" ]]; then
    "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"
    env \
        LLAMA_STAGE_LINK_MODE=dynamic \
        LLAMA_STAGE_BACKEND="$(build_backend)" \
        LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
        LLAMA_STAGE_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
        "$SCRIPT_DIR/build-llama.sh"
fi

platform="$(target_platform "$TARGET_TRIPLE")"
runtime_os="$(target_runtime_os "$TARGET_TRIPLE")"
runtime_arch="$(target_runtime_arch "$TARGET_TRIPLE")"
if ! flavor="$(backend_flavor)"; then
    exit 1
fi
artifact_id="meshllm-native-runtime-${platform}-${flavor}"
stage_dir="$OUT_DIR/$artifact_id"

runtime_libraries=()
while IFS= read -r library; do
    runtime_libraries+=("$library")
done < <(collect_runtime_libraries)
if [[ "${#runtime_libraries[@]}" -eq 0 ]]; then
    echo "no native runtime libraries found under $LLAMA_STAGE_BUILD_DIR" >&2
    echo "rerun with --build or build patched llama.cpp with LLAMA_STAGE_LINK_MODE=dynamic" >&2
    exit 1
fi

last_index=$((${#runtime_libraries[@]} - 1))
primary_name="$(basename "${runtime_libraries[$last_index]}")"
if ! primary_library_names | grep -Fxq "$primary_name"; then
    echo "primary native runtime library not found; expected one of:" >&2
    primary_library_names | sed 's/^/  /' >&2
    exit 1
fi

rm -rf "$stage_dir"
mkdir -p "$stage_dir/lib"

tool_paths=()

library_paths=()
for library in "${runtime_libraries[@]}"; do
    name="$(basename "$library")"
    cp "$library" "$stage_dir/lib/$name"
    library_paths+=("lib/$name")
done

rewrite_macos_runtime_paths
rewrite_linux_runtime_paths

build_gpu_benchmark_tool
build_model_package_tool

if [[ "$runtime_os" == "windows" ]]; then
    dependency_args=()
    for library in "${runtime_libraries[@]}"; do
        dependency_args+=(--search-dir "$(dirname "$library")")
    done
    # The Vulkan SDK can contain an older MinGW runtime. Let the dependency
    # resolver discover the compiler runtime before searching SDK directories.
    # This also keeps CPU packages self-contained when their DLLs use MinGW.
    if [[ "$BACKEND" == "cpu" || "$BACKEND" == "vulkan" ]]; then
        mingw_compiler_spec="${MINGW_CXX:-${CXX:-}}"
        if [[ -x "$mingw_compiler_spec" ]]; then
            mingw_compiler="$mingw_compiler_spec"
        elif [[ -n "$mingw_compiler_spec" ]]; then
            # CXX may include a wrapper or compiler arguments. Resolve only
            # the executable selected by the build instead of looking up the
            # entire command string.
            read -r mingw_compiler _ <<< "$mingw_compiler_spec"
        else
            mingw_compiler=g++
        fi
        if [[ "$mingw_compiler" == */* ]]; then
            if [[ -x "$mingw_compiler" ]]; then
                mingw_compiler_path="$mingw_compiler"
            else
                mingw_compiler_path=""
            fi
        else
            mingw_compiler_path="$(command -v "$mingw_compiler" || true)"
        fi
        if [[ -n "$mingw_compiler_path" ]]; then
            dependency_args+=(--search-dir "$(dirname "$mingw_compiler_path")")
        elif [[ -n "$mingw_compiler_spec" ]]; then
            echo "configured MinGW C++ compiler was not found: $mingw_compiler_spec" >&2
            exit 1
        fi
    fi
    for dependency_root in CUDA_PATH ROCM_PATH VULKAN_SDK; do
        dependency_dir="${!dependency_root:-}"
        if [[ -n "$dependency_dir" ]]; then
            dependency_dir="$dependency_dir/$(if [[ "$dependency_root" == VULKAN_SDK ]]; then printf Bin; else printf bin; fi)"
            if [[ -d "$dependency_dir" ]]; then
                dependency_args+=(--search-dir "$dependency_dir")
            fi
        fi
    done
    "$(python_bin)" "$SCRIPT_DIR/windows-native-runtime-deps.py" collect \
        --lib-dir "$stage_dir/lib" \
        --scan-dir "$stage_dir/tools" \
        "${dependency_args[@]}"

    library_paths=()
    while IFS= read -r library; do
        name="$(basename "$library")"
        if [[ "$name" != "$primary_name" ]]; then
            library_paths+=("lib/$name")
        fi
    done < <(find "$stage_dir/lib" -maxdepth 1 -type f -name '*.dll' | sort)
    library_paths+=("lib/$primary_name")
fi

primary_library="lib/$primary_name"
primary_sha="$(sha256_file "$stage_dir/$primary_library")"
mesh_version="$(workspace_version)"
abi_version="$(skippy_abi_version)"
cuda_major=""
case "$BACKEND" in
    cuda|cuda-blackwell)
        if ! cuda_major="$(cuda_toolkit_major)"; then
            exit 1
        fi
        ;;
esac

patched_sha=""
upstream_sha=""
patch_digest=""
if [[ -f "$LLAMA_WORKDIR/.mesh-llm-patched-sha" ]]; then
    patched_sha="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-patched-sha")"
fi
if [[ -f "$LLAMA_WORKDIR/.mesh-llm-upstream-sha" ]]; then
    upstream_sha="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-upstream-sha")"
fi
if [[ -f "$LLAMA_WORKDIR/.mesh-llm-patch-digest" ]]; then
    patch_digest="$(tr -d '[:space:]' < "$LLAMA_WORKDIR/.mesh-llm-patch-digest")"
fi

manifest_args=("$stage_dir/manifest.json" "$primary_library" "${library_paths[@]}" --)
if [[ "${#tool_paths[@]}" -gt 0 ]]; then
    manifest_args+=("${tool_paths[@]}")
fi

"$(python_bin)" - "${manifest_args[@]}" <<PY
import json
import hashlib
import os
import sys

manifest_path = sys.argv[1]
primary_library = sys.argv[2]
separator = sys.argv.index("--")
library_paths = sys.argv[3:separator]
tool_paths = sys.argv[separator + 1:]
backend = "$BACKEND"
kind = {"hip": "rocm", "cuda-blackwell": "cuda"}.get(backend, backend)

def split_arches(raw):
    values = []
    for comma_part in raw.split(","):
        values.extend(part.strip() for part in comma_part.split(";"))
    return [value for value in values if value]

cuda_arches = split_arches(
    os.environ.get("LLAMA_STAGE_CUDA_ARCHITECTURES")
    or os.environ.get("SKIPPY_CUDA_ARCHITECTURES")
    or ("sm_120" if backend == "cuda-blackwell" else "")
)
rocm_arches = split_arches(
    os.environ.get("LLAMA_STAGE_AMDGPU_TARGETS")
    or os.environ.get("SKIPPY_AMDGPU_TARGETS")
    or ""
)

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = {
    path: file_sha256(os.path.join(os.path.dirname(manifest_path), path))
    for path in library_paths
}
tools = {
    path: file_sha256(os.path.join(os.path.dirname(manifest_path), path))
    for path in tool_paths
}
backend_manifest = {"kind": kind}
if kind == "cuda":
    backend_manifest["cuda"] = {
        "toolkit_major": int("$cuda_major"),
        "gpu_arches": cuda_arches,
    }
    min_driver = os.environ.get("MESH_LLM_CUDA_MIN_DRIVER")
    if min_driver:
        backend_manifest["cuda"]["min_driver"] = min_driver
elif kind == "rocm":
    backend_manifest["rocm"] = {
        "gpu_arches": rocm_arches,
    }
    version = os.environ.get("MESH_LLM_ROCM_VERSION")
    if version:
        backend_manifest["rocm"]["version"] = version
elif kind == "vulkan":
    backend_manifest["vulkan"] = {}
    min_api = os.environ.get("MESH_LLM_VULKAN_MIN_API_VERSION")
    if min_api:
        backend_manifest["vulkan"]["min_api_version"] = min_api

manifest = {
    "runtime": {
        "id": "$artifact_id",
        "mesh_version": "$mesh_version",
        "skippy_abi": "$abi_version",
        "platform": {
            "os": "$runtime_os",
            "arch": "$runtime_arch",
            "target": "$TARGET_TRIPLE",
        },
        "backend": backend_manifest,
        "rank": int(os.environ.get("MESH_LLM_NATIVE_RUNTIME_RANK") or 0),
        "libraries": library_paths,
        "files": files,
        "tools": tools,
        "url": None,
        "sha256": None,
        "signature": None,
    },
    "build": {
        "platform": "$platform",
        "backend": "$BACKEND",
        "primary_library": primary_library,
        "library_sha256": "$primary_sha",
        "llama_upstream_sha": "$upstream_sha" or None,
        "llama_patched_sha": "$patched_sha" or None,
        "llama_patch_digest": "$patch_digest" or None,
    },
}
with open(manifest_path, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\\n")
PY

cat > "$stage_dir/README.md" <<EOF
# $artifact_id

This artifact contains MeshLLM native runtime shared libraries for:

- target: \`$TARGET_TRIPLE\`
- backend: \`$BACKEND\`
- flavor: \`$flavor\`
- MeshLLM version: \`$mesh_version\`
- Skippy ABI: \`$abi_version\`

\`mesh-llm runtime install\` reads \`manifest.json\`, verifies the archive
checksum from \`native-runtimes.json\`, installs the artifact into the
versioned native runtime cache, and loads these libraries before Skippy starts.
EOF

mkdir -p "$OUT_DIR"
archive="$OUT_DIR/$artifact_id.tar.gz"
COPYFILE_DISABLE=1 tar -C "$OUT_DIR" -czf "$archive" "$artifact_id"
archive_sha="$(sha256_file "$archive")"
printf '%s  %s\n' "$archive_sha" "$(basename "$archive")" > "$archive.sha256"

echo "packaged native runtime:"
echo "  artifact: $artifact_id"
echo "  primary:  $stage_dir/$primary_library"
echo "  archive:  $archive"
