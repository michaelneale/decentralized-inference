#!/usr/bin/env bash
# ci-compat-smoke.sh — start 2 mesh nodes + 1 lite client, then run SDK compatibility smokes.
#
# Usage: scripts/ci-compat-smoke.sh <mesh-llm-binary> <bin-dir> <model-path>

set -euo pipefail

MESH_LLM="$1"
BIN_DIR="$2"
MODEL="$3"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NODE_BIN="${NODE_BIN:-node}"
NPM_BIN="${NPM_BIN:-npm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HOST_API_PORT=9337
HOST_CONSOLE_PORT=3131
HOST_BIND_PORT=7842

WORKER_API_PORT=9437
WORKER_CONSOLE_PORT=4131
WORKER_BIND_PORT=7843

CLIENT_API_PORT=9555
CLIENT_CONSOLE_PORT=5131
CLIENT_BIND_PORT=7844

MAX_WAIT=240
WORKDIR="$(mktemp -d)"
HOST_LOG="$WORKDIR/host.log"
WORKER_LOG="$WORKDIR/worker.log"
CLIENT_LOG="$WORKDIR/client.log"
NODE_SDK_DIR="$WORKDIR/openai-node"

echo "=== Compat Smoke Test ==="
echo "  mesh-llm:   $MESH_LLM"
echo "  bin-dir:    $BIN_DIR"
echo "  model:      $MODEL"
echo "  workdir:    $WORKDIR"

if [ ! -f "$MESH_LLM" ]; then
    echo "❌ Missing mesh-llm binary: $MESH_LLM"
    exit 1
fi

cleanup() {
    set +e
    for pid in "${CLIENT_PID:-}" "${WORKER_PID:-}" "${HOST_PID:-}"; do
        if [ -n "${pid:-}" ]; then
            kill "$pid" 2>/dev/null || true
            pkill -P "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${CLIENT_PID:-}" "${WORKER_PID:-}" "${HOST_PID:-}"; do
        if [ -n "${pid:-}" ]; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    pkill -9 -f "[/]rpc-server" 2>/dev/null || true
    pkill -9 -f "[/]llama-server" 2>/dev/null || true
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

fail_with_logs() {
    local message="$1"
    echo "❌ $message"
    echo "--- host log ---"
    tail -80 "$HOST_LOG" 2>/dev/null || true
    echo "--- worker log ---"
    tail -80 "$WORKER_LOG" 2>/dev/null || true
    echo "--- client log ---"
    tail -80 "$CLIENT_LOG" 2>/dev/null || true
    exit 1
}

assert_pid_alive() {
    local pid="$1"
    local name="$2"
    if ! kill -0 "$pid" 2>/dev/null; then
        fail_with_logs "$name exited unexpectedly"
    fi
}

json_field() {
    local url="$1"
    local field="$2"
    curl -sf "$url" | "$PYTHON_BIN" -c '
import json
import sys

field = sys.argv[1]
data = json.load(sys.stdin)
value = data
for part in field.split("."):
    if part.isdigit():
        value = value[int(part)]
    else:
        value = value.get(part)
print("" if value is None else value)
' "$field"
}

json_len() {
    local url="$1"
    local field="$2"
    curl -sf "$url" | "$PYTHON_BIN" -c '
import json
import sys

field = sys.argv[1]
data = json.load(sys.stdin)
value = data
for part in field.split("."):
    if part.isdigit():
        value = value[int(part)]
    else:
        value = value.get(part)
print(len(value) if value is not None else 0)
' "$field"
}

wait_for_status() {
    local port="$1"
    local pid="$2"
    local name="$3"
    for i in $(seq 1 "$MAX_WAIT"); do
        assert_pid_alive "$pid" "$name"
        if curl -sf "http://127.0.0.1:${port}/api/status" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    fail_with_logs "status endpoint on :$port for $name never came up"
}

wait_for_llama_ready() {
    local port="$1"
    local name="$2"
    local pid="$3"
    for i in $(seq 1 "$MAX_WAIT"); do
        assert_pid_alive "$pid" "$name"
        local ready
        ready="$(json_field "http://127.0.0.1:${port}/api/status" "llama_ready" 2>/dev/null || true)"
        if [ "$ready" = "True" ]; then
            echo "✅ $name ready after ${i}s"
            return 0
        fi
        if [ $((i % 20)) -eq 0 ]; then
            echo "  Waiting for $name model load... (${i}s)"
        fi
        sleep 1
    done
    fail_with_logs "$name did not become ready"
}

wait_for_client_mesh() {
    for i in $(seq 1 "$MAX_WAIT"); do
        assert_pid_alive "$CLIENT_PID" "client"
        assert_pid_alive "$HOST_PID" "host"
        assert_pid_alive "$WORKER_PID" "worker"
        local peers
        local models
        peers="$(json_len "http://127.0.0.1:${CLIENT_CONSOLE_PORT}/api/status" "peers" 2>/dev/null || echo 0)"
        models="$(curl -sf "http://127.0.0.1:${CLIENT_API_PORT}/v1/models" 2>/dev/null | "$PYTHON_BIN" -c 'import json,sys; print(len(json.load(sys.stdin).get("data", [])))' 2>/dev/null || echo 0)"
        if [ "$peers" -ge 2 ] && [ "$models" -ge 1 ]; then
            echo "✅ Client sees mesh: peers=$peers models=$models"
            return 0
        fi
        if [ $((i % 15)) -eq 0 ]; then
            echo "  Waiting for client mesh visibility... (${i}s, peers=$peers, models=$models)"
        fi
        sleep 1
    done
    fail_with_logs "client never saw both mesh nodes and models"
}

wait_for_split_mesh() {
    for i in $(seq 1 "$MAX_WAIT"); do
        assert_pid_alive "$HOST_PID" "host"
        assert_pid_alive "$WORKER_PID" "worker"
        local host_is_host
        local host_ready
        local host_peers
        local worker_is_host
        local worker_ready
        local worker_peers

        host_is_host="$(json_field "http://127.0.0.1:${HOST_CONSOLE_PORT}/api/status" "is_host" 2>/dev/null || true)"
        host_ready="$(json_field "http://127.0.0.1:${HOST_CONSOLE_PORT}/api/status" "llama_ready" 2>/dev/null || true)"
        host_peers="$(json_len "http://127.0.0.1:${HOST_CONSOLE_PORT}/api/status" "peers" 2>/dev/null || echo 0)"

        worker_is_host="$(json_field "http://127.0.0.1:${WORKER_CONSOLE_PORT}/api/status" "is_host" 2>/dev/null || true)"
        worker_ready="$(json_field "http://127.0.0.1:${WORKER_CONSOLE_PORT}/api/status" "llama_ready" 2>/dev/null || true)"
        worker_peers="$(json_len "http://127.0.0.1:${WORKER_CONSOLE_PORT}/api/status" "peers" 2>/dev/null || echo 0)"

        if [ "$host_is_host" = "True" ] && [ "$host_ready" = "True" ] && [ "$host_peers" -ge 1 ]; then
            echo "✅ Split mesh formed with host on :$HOST_CONSOLE_PORT after ${i}s"
            return 0
        fi
        if [ "$worker_is_host" = "True" ] && [ "$worker_ready" = "True" ] && [ "$worker_peers" -ge 1 ]; then
            echo "✅ Split mesh formed with host on :$WORKER_CONSOLE_PORT after ${i}s"
            return 0
        fi

        if [ $((i % 15)) -eq 0 ]; then
            echo "  Waiting for split mesh to elect a ready host... (${i}s)"
        fi
        sleep 1
    done
    fail_with_logs "split mesh never formed a ready host"
}

wait_for_routed_inference() {
    local model="$1"
    for i in $(seq 1 "$MAX_WAIT"); do
        assert_pid_alive "$CLIENT_PID" "client"
        assert_pid_alive "$HOST_PID" "host"
        assert_pid_alive "$WORKER_PID" "worker"

        local response
        local http_code
        local body
        response="$(curl -s --max-time 20 -w '\n%{http_code}' \
            "http://127.0.0.1:${CLIENT_API_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"${model}\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word ready.\"}],
                \"max_tokens\": 8,
                \"temperature\": 0
            }" 2>/dev/null || true)"
        http_code="$(printf '%s\n' "$response" | tail -n 1)"
        body="$(printf '%s\n' "$response" | sed '$d')"

        if [ "$http_code" = "200" ]; then
            echo "✅ Routed inference ready after ${i}s"
            return 0
        fi

        if [ "$http_code" != "503" ] && [ "$http_code" != "000" ] && [ -n "$http_code" ]; then
            echo "Unexpected readiness probe response ($http_code): $body"
            fail_with_logs "routed inference probe failed unexpectedly"
        fi

        if [ $((i % 10)) -eq 0 ]; then
            echo "  Waiting for routed inference readiness... (${i}s, last status=${http_code:-none})"
        fi
        sleep 1
    done
    fail_with_logs "routed inference never became ready"
}

ensure_openai_node_sdk() {
    if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
        fail_with_logs "node is not installed"
    fi
    if ! command -v "$NPM_BIN" >/dev/null 2>&1; then
        fail_with_logs "npm is not installed"
    fi
    mkdir -p "$NODE_SDK_DIR"
    "$NPM_BIN" install --silent --prefix "$NODE_SDK_DIR" openai >/dev/null
}

model_id() {
    curl -sf "http://127.0.0.1:${CLIENT_API_PORT}/v1/models" | "$PYTHON_BIN" -c '
import json
import sys

data = json.load(sys.stdin).get("data", [])
if not data:
    raise SystemExit("no models returned")
print(data[0]["id"])
'
}

echo "Starting host..."
MESH_LLM_EPHEMERAL_KEY=1 "$MESH_LLM" \
    --model "$MODEL" \
    --split \
    --no-draft \
    --bin-dir "$BIN_DIR" \
    --device CPU \
    --port "$HOST_API_PORT" \
    --console "$HOST_CONSOLE_PORT" \
    --bind-port "$HOST_BIND_PORT" \
    >"$HOST_LOG" 2>&1 &
HOST_PID=$!

wait_for_status "$HOST_CONSOLE_PORT" "$HOST_PID" "host"
TOKEN="$(json_field "http://127.0.0.1:${HOST_CONSOLE_PORT}/api/status" "token")"
if [ -z "$TOKEN" ]; then
    fail_with_logs "host did not expose an invite token"
fi
wait_for_llama_ready "$HOST_CONSOLE_PORT" "host" "$HOST_PID"

echo "Starting worker..."
MESH_LLM_EPHEMERAL_KEY=1 "$MESH_LLM" \
    --model "$MODEL" \
    --split \
    --no-draft \
    --bin-dir "$BIN_DIR" \
    --device CPU \
    --port "$WORKER_API_PORT" \
    --console "$WORKER_CONSOLE_PORT" \
    --bind-port "$WORKER_BIND_PORT" \
    --join "$TOKEN" \
    >"$WORKER_LOG" 2>&1 &
WORKER_PID=$!

wait_for_status "$WORKER_CONSOLE_PORT" "$WORKER_PID" "worker"
wait_for_split_mesh

echo "Starting lite client..."
MESH_LLM_EPHEMERAL_KEY=1 "$MESH_LLM" \
    --client \
    --no-draft \
    --port "$CLIENT_API_PORT" \
    --console "$CLIENT_CONSOLE_PORT" \
    --bind-port "$CLIENT_BIND_PORT" \
    --join "$TOKEN" \
    >"$CLIENT_LOG" 2>&1 &
CLIENT_PID=$!

wait_for_status "$CLIENT_CONSOLE_PORT" "$CLIENT_PID" "client"
wait_for_client_mesh
ROUTED_MODEL="$(model_id)"

echo "Using routed model: $ROUTED_MODEL"
wait_for_routed_inference "$ROUTED_MODEL"

echo "Running official openai-python smoke..."
"$PYTHON_BIN" "$REPO_ROOT/scripts/ci-openai-python-smoke.py" \
    --base-url "http://127.0.0.1:${CLIENT_API_PORT}/v1"

echo "Running official openai-node smoke..."
ensure_openai_node_sdk
NODE_PATH="$NODE_SDK_DIR/node_modules" "$NODE_BIN" \
    "$REPO_ROOT/scripts/ci-openai-node-smoke.cjs" \
    --base-url "http://127.0.0.1:${CLIENT_API_PORT}/v1"

echo "Running LiteLLM smoke..."
"$PYTHON_BIN" "$REPO_ROOT/scripts/ci-litellm-smoke.py" \
    --base-url "http://127.0.0.1:${CLIENT_API_PORT}/v1" \
    --model "$ROUTED_MODEL"

echo "Running langchain-openai smoke..."
"$PYTHON_BIN" "$REPO_ROOT/scripts/ci-langchain-openai-smoke.py" \
    --base-url "http://127.0.0.1:${CLIENT_API_PORT}/v1" \
    --model "$ROUTED_MODEL"

echo ""
echo "=== Compat smoke passed ==="
