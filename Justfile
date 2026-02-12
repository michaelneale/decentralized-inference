# Distributed LLM Inference — build & run tasks

llama_dir := "llama.cpp"
build_dir := llama_dir / "build"
models_dir := env("HOME") / ".models"
model := models_dir / "GLM-4.7-Flash-Q4_K_M.gguf"

# Clone and build the patched llama.cpp fork
build:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d "{{llama_dir}}" ]; then
        echo "Cloning michaelneale/llama.cpp (rpc-local-gguf branch)..."
        git clone -b rpc-local-gguf https://github.com/michaelneale/llama.cpp.git "{{llama_dir}}"
    fi
    cmake -B "{{build_dir}}" -S "{{llama_dir}}" -DGGML_METAL=ON -DGGML_RPC=ON
    cmake --build "{{build_dir}}" --config Release -j$(sysctl -n hw.ncpu)
    echo "Build complete: {{build_dir}}/bin/"

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

# Stop all running servers
stop:
    pkill -f "rpc-server" 2>/dev/null || true
    pkill -f "llama-server" 2>/dev/null || true
    echo "Stopped"

# Quick test inference
test:
    curl -s http://localhost:8080/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello! Say hi in one word."}],"max_tokens":10}' \
        | python3 -m json.tool

# Show the diff from upstream llama.cpp
diff:
    cd {{llama_dir}} && git log --oneline master..rpc-local-gguf
