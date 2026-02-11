#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# llama.cpp RPC Split Inference - Helper Script
#
# Usage:
#   ./run.sh              # default: GLM-4.7-Flash
#   ./run.sh glm          # GLM-4.7-Flash Q4_K_M (17GB, deepseek2 arch, thinking mode)
#   ./run.sh qwen3        # Qwen3-Coder-30B-A3B Q4_K_M (18GB, qwen3moe arch, from ollama)
#   ./run.sh /path/to.gguf  # any GGUF file
#   ./run.sh stop         # kill all rpc-server and llama-server processes
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/llama.cpp/build"
MODELS_DIR="$HOME/.models"

RPC_PORT_1=50052
RPC_PORT_2=50053
SERVER_PORT=8080

# --- Model definitions ---

GLM_GGUF="$MODELS_DIR/GLM-4.7-Flash-Q4_K_M.gguf"
GLM_URL="https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf"

QWEN3_GGUF="$HOME/.ollama/models/blobs/sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a"

# --- Functions ---

log() { echo "==> $*"; }
err() { echo "ERROR: $*" >&2; exit 1; }

stop_all() {
    log "Stopping all llama-server and rpc-server processes..."
    pkill -f "llama-server" 2>/dev/null && echo "  killed llama-server" || echo "  no llama-server running"
    pkill -f "rpc-server" 2>/dev/null && echo "  killed rpc-server(s)" || echo "  no rpc-server running"
}

build_llamacpp() {
    if [[ -x "$BUILD_DIR/bin/llama-server" && -x "$BUILD_DIR/bin/rpc-server" ]]; then
        log "llama.cpp already built"
        return
    fi

    log "Building llama.cpp with RPC support..."

    if [[ ! -d "$SCRIPT_DIR/llama.cpp" ]]; then
        git clone https://github.com/ggml-org/llama.cpp.git "$SCRIPT_DIR/llama.cpp"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DGGML_METAL=ON -DGGML_RPC=ON
    cmake --build . --config Release -j"$(sysctl -n hw.ncpu)"
    log "Build complete"
}

download_model() {
    local path="$1"
    local url="$2"
    local name="$3"

    if [[ -f "$path" ]]; then
        log "$name already downloaded: $path"
        return
    fi

    mkdir -p "$(dirname "$path")"
    log "Downloading $name (~17GB)..."
    log "  From: $url"
    log "  To:   $path"
    curl -L --progress-bar -o "$path" "$url"
    log "Download complete"
}

ensure_rpc_servers() {
    local need_start=0

    if ! lsof -i ":$RPC_PORT_1" -sTCP:LISTEN >/dev/null 2>&1; then
        need_start=1
    fi
    if ! lsof -i ":$RPC_PORT_2" -sTCP:LISTEN >/dev/null 2>&1; then
        need_start=1
    fi

    if [[ $need_start -eq 1 ]]; then
        # Kill any stale ones first
        pkill -f "rpc-server" 2>/dev/null || true
        sleep 1

        log "Starting RPC server 1 (Metal GPU) on port $RPC_PORT_1..."
        nohup "$BUILD_DIR/bin/rpc-server" -d MTL0 -p "$RPC_PORT_1" > /tmp/rpc-$RPC_PORT_1.log 2>&1 &

        log "Starting RPC server 2 (CPU) on port $RPC_PORT_2..."
        nohup "$BUILD_DIR/bin/rpc-server" -d CPU -p "$RPC_PORT_2" > /tmp/rpc-$RPC_PORT_2.log 2>&1 &

        # Wait for them to be ready
        log "Waiting for RPC servers..."
        for i in $(seq 1 10); do
            if lsof -i ":$RPC_PORT_1" -sTCP:LISTEN >/dev/null 2>&1 && \
               lsof -i ":$RPC_PORT_2" -sTCP:LISTEN >/dev/null 2>&1; then
                log "RPC servers ready"
                return
            fi
            sleep 1
        done
        err "RPC servers failed to start. Check /tmp/rpc-*.log"
    else
        log "RPC servers already running on ports $RPC_PORT_1 and $RPC_PORT_2"
    fi
}

start_server() {
    local model_path="$1"

    [[ -f "$model_path" ]] || err "Model not found: $model_path"

    # Kill existing llama-server
    pkill -f "llama-server" 2>/dev/null || true
    sleep 1

    log "Starting llama-server..."
    log "  Model: $model_path"
    log "  RPC:   127.0.0.1:$RPC_PORT_1, 127.0.0.1:$RPC_PORT_2"
    log "  Port:  $SERVER_PORT"

    nohup "$BUILD_DIR/bin/llama-server" \
        -m "$model_path" \
        --rpc "127.0.0.1:$RPC_PORT_1,127.0.0.1:$RPC_PORT_2" \
        -ngl 99 \
        --host 0.0.0.0 \
        --port "$SERVER_PORT" \
        > /tmp/llama-server.log 2>&1 &

    log "Waiting for llama-server to load model (this can take 30-60s)..."
    for i in $(seq 1 120); do
        if curl -sf http://localhost:$SERVER_PORT/health >/dev/null 2>&1; then
            log "llama-server ready!"
            echo ""
            log "Test with:"
            echo "  curl http://localhost:$SERVER_PORT/v1/chat/completions \\"
            echo "    -H 'Content-Type: application/json' \\"
            echo "    -d '{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":200}'"
            echo ""
            log "Logs: /tmp/llama-server.log, /tmp/rpc-$RPC_PORT_1.log, /tmp/rpc-$RPC_PORT_2.log"
            log "Stop with: ./run.sh stop"
            return
        fi
        sleep 1
    done
    err "llama-server failed to start within 120s. Check /tmp/llama-server.log"
}

# --- Main ---

MODEL_ARG="${1:-glm}"

case "$MODEL_ARG" in
    stop)
        stop_all
        exit 0
        ;;
    glm)
        build_llamacpp
        download_model "$GLM_GGUF" "$GLM_URL" "GLM-4.7-Flash Q4_K_M"
        ensure_rpc_servers
        start_server "$GLM_GGUF"
        ;;
    qwen3)
        if [[ ! -f "$QWEN3_GGUF" ]]; then
            err "Qwen3-Coder not found in ollama. Run: ollama pull qwen3-coder"
        fi
        build_llamacpp
        ensure_rpc_servers
        start_server "$QWEN3_GGUF"
        ;;
    *.gguf)
        build_llamacpp
        ensure_rpc_servers
        start_server "$MODEL_ARG"
        ;;
    *)
        echo "Usage: ./run.sh [glm|qwen3|/path/to/model.gguf|stop]"
        echo ""
        echo "  glm     GLM-4.7-Flash Q4_K_M (default, downloads if needed)"
        echo "  qwen3   Qwen3-Coder-30B-A3B from ollama"
        echo "  *.gguf  Any GGUF file path"
        echo "  stop    Kill all servers"
        exit 1
        ;;
esac
