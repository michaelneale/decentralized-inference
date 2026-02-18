# Distributed LLM Inference — build & run tasks

llama_dir := "llama.cpp"
build_dir := llama_dir / "build"
mesh_dir := "mesh-llm"
models_dir := env("HOME") / ".models"
model := models_dir / "GLM-4.7-Flash-Q4_K_M.gguf"

# Clone and build the patched llama.cpp fork + mesh-llm
build:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d "{{llama_dir}}" ]; then
        echo "Cloning michaelneale/llama.cpp (rpc-local-gguf branch)..."
        git clone -b rpc-local-gguf https://github.com/michaelneale/llama.cpp.git "{{llama_dir}}"
    fi
    cmake -B "{{build_dir}}" -S "{{llama_dir}}" -DGGML_METAL=ON -DGGML_RPC=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF
    cmake --build "{{build_dir}}" --config Release -j$(sysctl -n hw.ncpu)
    echo "Build complete: {{build_dir}}/bin/"
    if [ -d "{{mesh_dir}}" ]; then
        echo "Building mesh-llm..."
        cd "{{mesh_dir}}" && cargo build --release
        echo "Mesh binary: {{mesh_dir}}/target/release/mesh-llm"
    fi

# Download the default model (GLM-4.7-Flash Q4_K_M, 17GB)
download-model:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{models_dir}}"
    if [ -f "{{model}}" ]; then
        echo "Model already exists: {{model}}"
    else
        echo "Downloading GLM-4.7-Flash Q4_K_M (~17GB)..."
        curl -L -o "{{model}}" \
            "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf"
    fi

# ── Raw TCP (no mesh) ──────────────────────────────────────────

# Start rpc-server (worker) with local GGUF loading
worker host="0.0.0.0" port="50052" device="MTL0" gguf=model:
    {{build_dir}}/bin/rpc-server --host {{host}} --port {{port}} -d {{device}} --gguf {{gguf}}

# Start llama-server (orchestrator) pointing at an RPC worker
serve rpc="127.0.0.1:50052" port="8080" gguf=model:
    {{build_dir}}/bin/llama-server \
        --model {{gguf}} \
        --rpc {{rpc}} \
        -ngl 99 -fit off \
        --port {{port}}

# Start both worker + server on localhost for testing
local: build download-model
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Starting rpc-server (worker)..."
    {{build_dir}}/bin/rpc-server --host 127.0.0.1 --port 50052 -d MTL0 --gguf {{model}} &
    WORKER_PID=$!
    sleep 3
    echo "Starting llama-server (orchestrator)..."
    {{build_dir}}/bin/llama-server \
        --model {{model}} \
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

mesh_bin := mesh_dir / "target/release/mesh-llm"

# Start a mesh worker (no llama-server, just rpc-server + mesh)
# Prints an invite token for other nodes to join.
mesh-worker gguf=model:
    {{mesh_bin}} --model {{gguf}} --bin-dir {{build_dir}}/bin

# Join an existing mesh. Auto-elects host, starts llama-server or contributes as worker.
mesh-join join="" port="9337" gguf=model split="":
    #!/usr/bin/env bash
    set -euo pipefail
    ARGS="--model {{gguf}} --bin-dir {{build_dir}}/bin --port {{port}}"
    if [ -n "{{join}}" ]; then
        ARGS="$ARGS --join {{join}}"
    fi
    if [ -n "{{split}}" ]; then
        ARGS="$ARGS --tensor-split {{split}}"
    fi
    exec {{mesh_bin}} $ARGS

# Create a portable tarball with all binaries for deployment to another machine
bundle output="/tmp/mesh-bundle.tar.gz":
    #!/usr/bin/env bash
    set -euo pipefail
    DIR=$(mktemp -d)
    BUNDLE="$DIR/mesh-bundle"
    mkdir -p "$BUNDLE"
    cp {{mesh_bin}} "$BUNDLE/"
    cp {{build_dir}}/bin/rpc-server "$BUNDLE/"
    cp {{build_dir}}/bin/llama-server "$BUNDLE/"
    for lib in {{build_dir}}/bin/*.dylib; do
        cp "$lib" "$BUNDLE/" 2>/dev/null || true
    done
    # Fix rpaths for portability
    for bin in "$BUNDLE/mesh-llm" "$BUNDLE/rpc-server" "$BUNDLE/llama-server"; do
        [ -f "$bin" ] || continue
        install_name_tool -add_rpath @executable_path/ "$bin" 2>/dev/null || true
    done
    tar czf {{output}} -C "$DIR" mesh-bundle/
    rm -rf "$DIR"
    echo "Bundle: {{output}} ($(du -sh {{output}} | cut -f1))"

# Start a lite client — no GPU, no model, just a local HTTP proxy to the mesh host.
# Only needs the mesh-llm binary (no llama.cpp binaries or model).
mesh-client join="" port="9337":
    {{mesh_bin}} --client --port {{port}} --join {{join}}

# ── Utilities ──────────────────────────────────────────────────

# Stop all running servers
stop:
    pkill -f "mesh-llm" 2>/dev/null || true
    pkill -f "rpc-server" 2>/dev/null || true
    pkill -f "llama-server" 2>/dev/null || true
    echo "Stopped"

# Quick test inference (works with any running server on 8080 or 8090)
test port="9337":
    curl -s http://localhost:{{port}}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello! Write a haiku about distributed computing."}],"max_tokens":50}' \
        | python3 -c "import sys,json; d=json.load(sys.stdin); t=d['timings']; print(d['choices'][0]['message'].get('content','')[:200]); print(f\"  prompt: {t['prompt_per_second']:.1f} tok/s  gen: {t['predicted_per_second']:.1f} tok/s ({t['predicted_n']} tok)\")"

# Show the diff from upstream llama.cpp
diff:
    cd {{llama_dir}} && git log --oneline master..rpc-local-gguf
