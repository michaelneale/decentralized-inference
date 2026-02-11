#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# llama.cpp RPC Split Inference Demo
#
# Usage:
#   ./demo.sh              # default: GLM-4.7-Flash, no latency
#   ./demo.sh glm          # GLM-4.7-Flash Q4_K_M (17GB)
#   ./demo.sh qwen3        # Qwen3-Coder-30B-A3B Q4_K_M (18GB)
#   ./demo.sh /path/to.gguf  # any GGUF file
#   ./demo.sh stop         # kill all servers + proxies
#
# Environment variables:
#   LATENCY1=50  LATENCY2=100  ./demo.sh glm   # inject 50ms on node1, 100ms on node2
#   TENSOR_SPLIT="0.33,0.33,0.34" ./demo.sh glm  # force even 3-way split
#   VERBOSE=1 ./demo.sh glm                       # verbose logging (shows layer assignments)
#
# Models are downloaded to ~/.models/ if not already present.
# See notes.md for full details on how RPC split inference works.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/llama.cpp/build"
MODELS_DIR="$HOME/.models"

RPC_PORT_1=50052
RPC_PORT_2=50053
PROXY_PORT_1=60052
PROXY_PORT_2=60053
SERVER_PORT=8080

# Latency injection (milliseconds), 0 = no proxy
LATENCY1="${LATENCY1:-0}"
LATENCY2="${LATENCY2:-0}"

# Tensor split ratio (e.g. "0.33,0.33,0.34" for even 3-way)
TENSOR_SPLIT="${TENSOR_SPLIT:-}"

# Verbose logging
VERBOSE="${VERBOSE:-0}"

# --- Model definitions ---

GLM_GGUF="$MODELS_DIR/GLM-4.7-Flash-Q4_K_M.gguf"
GLM_URL="https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf"

QWEN3_GGUF="$MODELS_DIR/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
QWEN3_URL="https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"

# --- Functions ---

log() { echo "==> $*"; }
err() { echo "ERROR: $*" >&2; exit 1; }

stop_all() {
    log "Stopping all processes..."
    pkill -f "llama-server" 2>/dev/null && echo "  killed llama-server" || echo "  no llama-server running"
    pkill -f "latency-proxy.py" 2>/dev/null && echo "  killed latency proxies" || echo "  no proxies running"
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
    log "Downloading $name..."
    log "  From: $url"
    log "  To:   $path"
    curl -L --progress-bar -o "${path}.tmp" "$url"
    mv "${path}.tmp" "$path"
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

ensure_latency_proxies() {
    # Kill any existing proxies
    pkill -f "latency-proxy.py" 2>/dev/null || true
    sleep 0.5

    local use_proxy=0

    if [[ "$LATENCY1" != "0" ]]; then
        use_proxy=1
        log "Starting latency proxy: :$PROXY_PORT_1 → :$RPC_PORT_1 (${LATENCY1}ms)"
        nohup python3 "$SCRIPT_DIR/latency-proxy.py" \
            --listen-port "$PROXY_PORT_1" \
            --target-port "$RPC_PORT_1" \
            --latency-ms "$LATENCY1" \
            > /tmp/proxy-$PROXY_PORT_1.log 2>&1 &
    fi

    if [[ "$LATENCY2" != "0" ]]; then
        use_proxy=1
        log "Starting latency proxy: :$PROXY_PORT_2 → :$RPC_PORT_2 (${LATENCY2}ms)"
        nohup python3 "$SCRIPT_DIR/latency-proxy.py" \
            --listen-port "$PROXY_PORT_2" \
            --target-port "$RPC_PORT_2" \
            --latency-ms "$LATENCY2" \
            > /tmp/proxy-$PROXY_PORT_2.log 2>&1 &
    fi

    if [[ $use_proxy -eq 1 ]]; then
        # Wait for proxies to be ready
        log "Waiting for latency proxies..."
        for i in $(seq 1 5); do
            local ready=1
            if [[ "$LATENCY1" != "0" ]] && ! lsof -i ":$PROXY_PORT_1" -sTCP:LISTEN >/dev/null 2>&1; then
                ready=0
            fi
            if [[ "$LATENCY2" != "0" ]] && ! lsof -i ":$PROXY_PORT_2" -sTCP:LISTEN >/dev/null 2>&1; then
                ready=0
            fi
            if [[ $ready -eq 1 ]]; then
                log "Latency proxies ready"
                return
            fi
            sleep 0.5
        done
        err "Latency proxies failed to start. Check /tmp/proxy-*.log"
    fi
}

start_server() {
    local model_path="$1"

    [[ -f "$model_path" ]] || err "Model not found: $model_path"

    # Kill existing llama-server
    pkill -f "llama-server" 2>/dev/null || true
    sleep 1

    # Determine which ports llama-server connects to
    # If latency proxy is active for a node, connect to proxy port instead
    local rpc_addr_1="127.0.0.1:$RPC_PORT_1"
    local rpc_addr_2="127.0.0.1:$RPC_PORT_2"

    if [[ "$LATENCY1" != "0" ]]; then
        rpc_addr_1="127.0.0.1:$PROXY_PORT_1"
    fi
    if [[ "$LATENCY2" != "0" ]]; then
        rpc_addr_2="127.0.0.1:$PROXY_PORT_2"
    fi

    local rpc_endpoints="$rpc_addr_1,$rpc_addr_2"

    # Build extra args
    local extra_args=""
    if [[ -n "$TENSOR_SPLIT" ]]; then
        extra_args="--tensor-split $TENSOR_SPLIT"
    fi

    log "Starting llama-server..."
    log "  Model:  $model_path"
    log "  RPC:    $rpc_endpoints"
    log "  Port:   $SERVER_PORT"
    if [[ -n "$TENSOR_SPLIT" ]]; then
        log "  Split:  $TENSOR_SPLIT"
    fi
    if [[ "$LATENCY1" != "0" || "$LATENCY2" != "0" ]]; then
        log "  Latency: node1=${LATENCY1}ms, node2=${LATENCY2}ms"
    fi

    local env_prefix=""
    if [[ "$VERBOSE" == "1" ]]; then
        env_prefix="LLAMA_LOG_VERBOSITY=10"
    fi

    if [[ -n "$env_prefix" ]]; then
        nohup env $env_prefix "$BUILD_DIR/bin/llama-server" \
            -m "$model_path" \
            --rpc "$rpc_endpoints" \
            -ngl 99 \
            --host 0.0.0.0 \
            --port "$SERVER_PORT" \
            $extra_args \
            > /tmp/llama-server.log 2>&1 &
    else
        nohup "$BUILD_DIR/bin/llama-server" \
            -m "$model_path" \
            --rpc "$rpc_endpoints" \
            -ngl 99 \
            --host 0.0.0.0 \
            --port "$SERVER_PORT" \
            $extra_args \
            > /tmp/llama-server.log 2>&1 &
    fi

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
            log "Logs:"
            echo "  /tmp/llama-server.log"
            echo "  /tmp/rpc-$RPC_PORT_1.log"
            echo "  /tmp/rpc-$RPC_PORT_2.log"
            if [[ "$LATENCY1" != "0" ]]; then
                echo "  /tmp/proxy-$PROXY_PORT_1.log"
            fi
            if [[ "$LATENCY2" != "0" ]]; then
                echo "  /tmp/proxy-$PROXY_PORT_2.log"
            fi
            echo ""
            log "Stop with: ./demo.sh stop"
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
        download_model "$GLM_GGUF" "$GLM_URL" "GLM-4.7-Flash Q4_K_M (~17GB)"
        ensure_rpc_servers
        ensure_latency_proxies
        start_server "$GLM_GGUF"
        ;;
    qwen3)
        build_llamacpp
        download_model "$QWEN3_GGUF" "$QWEN3_URL" "Qwen3-Coder-30B-A3B Q4_K_M (~18GB)"
        ensure_rpc_servers
        ensure_latency_proxies
        start_server "$QWEN3_GGUF"
        ;;
    *.gguf)
        build_llamacpp
        ensure_rpc_servers
        ensure_latency_proxies
        start_server "$MODEL_ARG"
        ;;
    *)
        echo "Usage: ./demo.sh [glm|qwen3|/path/to/model.gguf|stop]"
        echo ""
        echo "  glm     GLM-4.7-Flash Q4_K_M - 17GB (default)"
        echo "  qwen3   Qwen3-Coder-30B-A3B Q4_K_M - 18GB"
        echo "  *.gguf  Any GGUF file path"
        echo "  stop    Kill all servers"
        echo ""
        echo "Environment variables:"
        echo "  LATENCY1=50  LATENCY2=100  ./demo.sh glm   # ms latency per compute op"
        echo "  TENSOR_SPLIT=\"0.33,0.33,0.34\" ./demo.sh glm  # force layer split ratio"
        echo "  VERBOSE=1 ./demo.sh glm                       # verbose layer assignment logging"
        echo ""
        echo "Models are downloaded to ~/.models/ if not already present."
        exit 1
        ;;
esac
