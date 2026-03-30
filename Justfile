# Distributed LLM Inference — build & run tasks

llama_dir := "llama.cpp"
build_dir := llama_dir / "build"
mesh_dir := "mesh-llm"
ui_dir := mesh_dir / "ui"
models_dir := env("HOME") / ".models"
model := models_dir / "GLM-4.7-Flash-Q4_K_M.gguf"

# Build for the current platform (macOS→Metal, Linux→CUDA/ROCm/Vulkan auto-detected)
[macos]
build: build-mac

# Linux overrides:
#   just build backend=cpu
#   just build backend=cuda cuda_arch='120;86'
#   just build backend=rocm rocm_arch='gfx942;gfx90a'
#   just build backend=vulkan
[linux]
build backend="" cuda_arch="" rocm_arch="":
    @scripts/build-linux.sh --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

# Build on macOS Apple Silicon (Metal + RPC)
build-mac:
    @scripts/build-mac.sh

# Build on Linux with CUDA, ROCm, or Vulkan — delegates to scripts/build-linux.sh
build-linux backend="" cuda_arch="" rocm_arch="":
    @scripts/build-linux.sh --backend "{{ backend }}" --cuda-arch "{{ cuda_arch }}" --rocm-arch "{{ rocm_arch }}"

# Build release artifacts for the current platform.

# GitHub release builds use CPU backends on Linux and Metal on macOS.
release-build:
    @scripts/build-release.sh

# Build a Linux CUDA release artifact with an explicit architecture list.
release-build-cuda cuda_arch="75;80;86;89;90;120":
    @scripts/build-linux.sh --backend cuda --cuda-arch "{{ cuda_arch }}"

# Build a Linux AMD ROCm release artifact with an explicit architecture list.
release-build-amd amd_arch="gfx90a;gfx942;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201":
    @scripts/build-linux-amd.sh "{{ amd_arch }}"

# Build a Linux Vulkan release artifact.
release-build-vulkan:
    @scripts/build-linux.sh --backend vulkan

# Bump release version consistently across source and Cargo manifests.
release-version version:
    @scripts/release-version.sh "{{ version }}"

# Tag and push a release. Bumps version, updates Cargo.lock, commits, tags, pushes.
# CI builds and publishes the GitHub release automatically.
release version:
    #!/usr/bin/env bash
    set -euo pipefail
    current_branch="$(git branch --show-current)"
    if [[ "$current_branch" != "main" ]]; then
        echo "Error: release must be run from the 'main' branch (current: ${current_branch:-detached HEAD})" >&2
        exit 1
    fi
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "Error: working tree is not clean. Commit or stash changes before releasing." >&2
        exit 1
    fi
    tag="{{ version }}"
    if [[ "$tag" != v* ]]; then
        tag="v$tag"
    fi
    scripts/release-version.sh "$tag"
    git add -A
    git commit -m "$tag: release"
    git tag "$tag"
    git push origin main
    git push origin "$tag"

# Download the default model (GLM-4.7-Flash Q4_K_M, 17GB)
download-model:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{ models_dir }}"
    if [ -f "{{ model }}" ]; then
        echo "Model already exists: {{ model }}"
    else
        echo "Downloading GLM-4.7-Flash Q4_K_M (~17GB)..."
        curl -L -o "{{ model }}" \
            "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf"
    fi

# ── Raw TCP (no mesh) ──────────────────────────────────────────

# Start rpc-server (worker) with local GGUF loading
worker host="0.0.0.0" port="50052" device="" gguf=model:
    #!/usr/bin/env bash
    set -euo pipefail
    DEVICE="{{ device }}"
    if [ -z "$DEVICE" ]; then
        DEVICE="$(scripts/detect-llama-device.sh "{{ build_dir }}/bin/rpc-server")"
    fi
    exec {{ build_dir }}/bin/rpc-server --host {{ host }} --port {{ port }} -d "$DEVICE" --gguf {{ gguf }}

# Start llama-server (orchestrator) pointing at an RPC worker
serve rpc="127.0.0.1:50052" port="8080" gguf=model:
    {{ build_dir }}/bin/llama-server \
        --model {{ gguf }} \
        --rpc {{ rpc }} \
        -ngl 99 -fit off \
        --port {{ port }}

# Start both worker + server on localhost for testing
local: build download-model
    #!/usr/bin/env bash
    set -euo pipefail
    DEVICE="$(scripts/detect-llama-device.sh "{{ build_dir }}/bin/rpc-server")"
    echo "Starting rpc-server (worker)..."
    {{ build_dir }}/bin/rpc-server --host 127.0.0.1 --port 50052 -d "$DEVICE" --gguf {{ model }} &
    WORKER_PID=$!
    sleep 3
    echo "Starting llama-server (orchestrator)..."
    {{ build_dir }}/bin/llama-server \
        --model {{ model }} \
        --rpc 127.0.0.1:50052 \
        -ngl 99 -fit off \
        --port 8080 &
    SERVER_PID=$!
    echo "Waiting for server..."
    for i in $(seq 1 120); do
        curl -s http://localhost:8080/health 2>/dev/null | grep -q '"ok"' && break
        sleep 1
    done
    echo "Ready: http://localhost:8080"
    echo "Worker PID: $WORKER_PID  Server PID: $SERVER_PID"
    echo "Press Ctrl+C to stop"
    wait

# ── QUIC Mesh ──────────────────────────────────────────────────

mesh_bin := "target/release/mesh-llm"

# Start a mesh worker (no llama-server, just rpc-server + mesh)

# Prints an invite token for other nodes to join.
mesh-worker gguf=model:
    {{ mesh_bin }} --model {{ gguf }} --bin-dir {{ build_dir }}/bin

# Join an existing mesh. Auto-elects host, starts llama-server or contributes as worker.
mesh-join join="" port="9337" gguf=model split="":
    #!/usr/bin/env bash
    set -euo pipefail
    ARGS="--model {{ gguf }} --bin-dir {{ build_dir }}/bin --port {{ port }}"
    if [ -n "{{ join }}" ]; then
        ARGS="$ARGS --join {{ join }}"
    fi
    if [ -n "{{ split }}" ]; then
        ARGS="$ARGS --tensor-split {{ split }}"
    fi
    exec {{ mesh_bin }} $ARGS

# Create a portable tarball with all binaries for deployment to another machine
bundle output="/tmp/mesh-bundle.tar.gz":
    #!/usr/bin/env bash
    set -euo pipefail
    DIR=$(mktemp -d)
    BUNDLE="$DIR/mesh-bundle"
    mkdir -p "$BUNDLE"
    case "$(uname -s)" in
        Darwin) LLAMA_FLAVOR="metal" ;;
        Linux) LLAMA_FLAVOR="cpu" ;;
        *) LLAMA_FLAVOR="" ;;
    esac
    rpc_name="rpc-server"
    llama_name="llama-server"
    if [ -n "$LLAMA_FLAVOR" ]; then
        rpc_name="rpc-server-$LLAMA_FLAVOR"
        llama_name="llama-server-$LLAMA_FLAVOR"
    fi
    cp {{ mesh_bin }} "$BUNDLE/"
    cp {{ build_dir }}/bin/rpc-server "$BUNDLE/$rpc_name"
    cp {{ build_dir }}/bin/llama-server "$BUNDLE/$llama_name"
    cp {{ build_dir }}/bin/llama-moe-split "$BUNDLE/"
    for lib in {{ build_dir }}/bin/*.dylib; do
        cp "$lib" "$BUNDLE/" 2>/dev/null || true
    done
    # Fix rpaths for portability
    for bin in "$BUNDLE/mesh-llm" "$BUNDLE/$rpc_name" "$BUNDLE/$llama_name" "$BUNDLE/llama-moe-split"; do
        [ -f "$bin" ] || continue
        install_name_tool -add_rpath @executable_path/ "$bin" 2>/dev/null || true
    done
    # Include Apple Silicon benchmark binary if built
    BENCH="{{ mesh_dir }}/target/release/membench-fingerprint"
    if [ -f "$BENCH" ]; then
        cp "$BENCH" "$BUNDLE/"
        echo "Included: membench-fingerprint"
    else
        echo "Note: membench-fingerprint not found — run 'just benchmark-build-apple' to include it"
    fi
    tar czf {{ output }} -C "$DIR" mesh-bundle/
    rm -rf "$DIR"
    echo "Bundle: {{ output }} ($(du -sh {{ output }} | cut -f1))"

# Create release archive(s) for the current platform.

# `version` should be a tag like v0.30.0.
release-bundle version output="dist":
    @scripts/package-release.sh "{{ version }}" "{{ output }}"

# Create Linux CUDA release archive(s).
release-bundle-cuda version output="dist":
    MESH_RELEASE_FLAVOR=cuda scripts/package-release.sh "{{ version }}" "{{ output }}"

# Create Linux AMD ROCm release archive(s).
release-bundle-amd version output="dist":
    MESH_RELEASE_FLAVOR=rocm scripts/package-release.sh "{{ version }}" "{{ output }}"

# Create Linux Vulkan release archive(s).
release-bundle-vulkan version output="dist":
    MESH_RELEASE_FLAVOR=vulkan scripts/package-release.sh "{{ version }}" "{{ output }}"

# ── Benchmark Binaries ────────────────────────────────────────────────────────

# Build Apple Silicon memory bandwidth benchmark (macOS only)
[macos]
benchmark-build-apple:
    swiftc -O benchmarks/membench-fingerprint.swift -o {{mesh_dir}}/target/release/membench-fingerprint
    echo "Built: {{mesh_dir}}/target/release/membench-fingerprint"

# Build NVIDIA CUDA memory bandwidth benchmark (requires CUDA toolkit)
benchmark-build-cuda:
    nvcc -O3 -o {{mesh_dir}}/target/release/membench-fingerprint-cuda benchmarks/membench-fingerprint.cu
    echo "Built: {{mesh_dir}}/target/release/membench-fingerprint-cuda"

# Build AMD ROCm/HIP memory bandwidth benchmark (requires ROCm)
benchmark-build-hip:
    hipcc -O3 -std=c++17 -o {{mesh_dir}}/target/release/membench-fingerprint-hip benchmarks/membench-fingerprint.hip
    echo "Built: {{mesh_dir}}/target/release/membench-fingerprint-hip"

# Build Intel Arc SYCL memory bandwidth benchmark (requires Intel oneAPI) — UNVALIDATED
benchmark-build-intel:
    @echo "WARNING: Intel Arc benchmark is unvalidated — no Intel Arc hardware has been tested"
    icpx -O3 -fsycl -o {{mesh_dir}}/target/release/membench-fingerprint-intel benchmarks/membench-fingerprint-intel.cpp
    echo "Built: {{mesh_dir}}/target/release/membench-fingerprint-intel"

# Run the UI with Vite HMR and proxy /api to mesh-llm (default: http://127.0.0.1:3131)
ui-dev api="http://127.0.0.1:3131" port="5173":
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ ui_dir }}"
    MESH_UI_API_ORIGIN="{{ api }}" npm run dev -- --host 127.0.0.1 --port {{ port }}

# Start a lite client — no GPU, no model, just a local HTTP proxy to the mesh host.

# Only needs the mesh-llm binary (no llama.cpp binaries or model).
mesh-client join="" port="9337":
    {{ mesh_bin }} --client --port {{ port }} --join {{ join }}

# Build and auto-join a mesh (discover via Nostr)
auto: build
    {{ mesh_bin }} --auto --bin-dir {{ build_dir }}/bin

# ── Utilities ──────────────────────────────────────────────────

# Stop all running servers
stop:
    pkill -f "mesh-llm" 2>/dev/null || true
    pkill -f "rpc-server" 2>/dev/null || true
    pkill -f "llama-server" 2>/dev/null || true
    echo "Stopped"

# Quick test inference (works with any running server on 8080 or 8090)
test port="9337":
    curl -s http://localhost:{{ port }}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello! Write a haiku about distributed computing."}],"max_tokens":50}' \
        | python3 -c "import sys,json; d=json.load(sys.stdin); t=d['timings']; print(d['choices'][0]['message'].get('content','')[:200]); print(f\"  prompt: {t['prompt_per_second']:.1f} tok/s  gen: {t['predicted_per_second']:.1f} tok/s ({t['predicted_n']} tok)\")"

# Benchmark sticky-only vs prefix-only affinity on a 3-node local mesh.
bench-prefix-affinity:
    @scripts/benchmark-prefix-affinity.sh

# Show the diff from upstream llama.cpp
diff:
    cd {{ llama_dir }} && git log --oneline master..rebase-upstream-master
