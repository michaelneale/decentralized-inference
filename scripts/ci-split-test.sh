#!/usr/bin/env bash
# ci-split-test.sh — verify that split-mode routes chat requests to the Host, not the Worker.
#
# Starts three mesh-llm nodes on the same machine:
#   Node A + B — compute nodes (CPU) with --split (one becomes Host, one becomes Worker)
#   Node C    — client-only node (routes via target map built from peer gossip)
#
# The bug: Node C's target map could pick the Worker as an HTTP target.
# Workers only run rpc-server and return empty/broken responses to chat requests.
#
# Usage: scripts/ci-split-test.sh <mesh-llm-binary> <bin-dir> <model-path>
#
# Expects llama-server and rpc-server in <bin-dir>.
# Exits 0 on success, 1 on failure.

set -euo pipefail

MESH_LLM="$1"
BIN_DIR="$2"
MODEL="$3"

A_API_PORT=9347
A_CONSOLE_PORT=3141
B_API_PORT=9348
B_CONSOLE_PORT=3142
C_API_PORT=9349
C_CONSOLE_PORT=3143
MAX_WAIT=180
LOG_A=/tmp/mesh-llm-split-a.log
LOG_B=/tmp/mesh-llm-split-b.log
LOG_C=/tmp/mesh-llm-split-c.log

echo "=== CI Split-Mode Routing Test ==="
echo "  mesh-llm:  $MESH_LLM"
echo "  bin-dir:   $BIN_DIR"
echo "  model:     $MODEL"
echo "  os:        $(uname -s)"

if [ ! -f "$MESH_LLM" ]; then
    echo "❌ Missing mesh-llm binary: $MESH_LLM"
    exit 1
fi

A_PID=""
B_PID=""
C_PID=""

cleanup() {
    echo "Cleaning up..."
    for PID in $C_PID $B_PID $A_PID; do
        [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
        [ -n "$PID" ] && pkill -P "$PID" 2>/dev/null || true
    done
    sleep 2
    for PID in $C_PID $B_PID $A_PID; do
        [ -n "$PID" ] && kill -9 "$PID" 2>/dev/null || true
    done
    pkill -9 -f "rpc-server" 2>/dev/null || true
    pkill -9 -f "llama-server" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

# ── Start Node A ──
echo ""
echo "Starting Node A..."
MESH_LLM_EPHEMERAL_KEY=1 "$MESH_LLM" \
    --model "$MODEL" \
    --split \
    --no-draft \
    --bin-dir "$BIN_DIR" \
    --device CPU \
    --port "$A_API_PORT" \
    --console "$A_CONSOLE_PORT" \
    > "$LOG_A" 2>&1 &
A_PID=$!
echo "  PID: $A_PID"

# Wait for Node A's console API to get the invite token
echo "Waiting for Node A console API..."
TOKEN=""
for i in $(seq 1 60); do
    if ! kill -0 "$A_PID" 2>/dev/null; then
        echo "❌ Node A exited unexpectedly"
        tail -50 "$LOG_A" || true
        exit 1
    fi
    TOKEN=$(curl -sf "http://localhost:${A_CONSOLE_PORT}/api/status" 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" 2>/dev/null || echo "")
    if [ -n "$TOKEN" ]; then
        echo "  Got invite token in ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "❌ Node A console never came up"
        tail -30 "$LOG_A" || true
        exit 1
    fi
    sleep 1
done

# ── Start Node B ──
echo ""
echo "Starting Node B..."
MESH_LLM_EPHEMERAL_KEY=1 "$MESH_LLM" \
    --model "$MODEL" \
    --split \
    --no-draft \
    --bin-dir "$BIN_DIR" \
    --device CPU \
    --port "$B_API_PORT" \
    --console "$B_CONSOLE_PORT" \
    --join "$TOKEN" \
    > "$LOG_B" 2>&1 &
B_PID=$!
echo "  PID: $B_PID"

# ── Wait for one node to become Host with a peer ──
echo ""
echo "Waiting for split mesh to form (up to ${MAX_WAIT}s)..."
HOST_CONSOLE=""
HOST_API=""
for i in $(seq 1 "$MAX_WAIT"); do
    for PID in $A_PID $B_PID; do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "❌ Node (PID $PID) exited unexpectedly"
            tail -50 "$LOG_A" || true
            tail -50 "$LOG_B" || true
            exit 1
        fi
    done

    # Check both nodes — election picks the host, we don't know which one
    for CPORT in $A_CONSOLE_PORT $B_CONSOLE_PORT; do
        STATUS=$(curl -sf "http://localhost:${CPORT}/api/status" 2>/dev/null || echo "")
        if [ -n "$STATUS" ]; then
            READY=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('llama_ready',False))" 2>/dev/null || echo "False")
            PEERS=$(echo "$STATUS" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('peers',[])))" 2>/dev/null || echo "0")
            IS_HOST=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('is_host',False))" 2>/dev/null || echo "False")

            if [ "$READY" = "True" ] && [ "$PEERS" -ge 1 ] && [ "$IS_HOST" = "True" ]; then
                API_PORT=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('api_port',0))" 2>/dev/null || echo "0")
                HOST_CONSOLE="$CPORT"
                HOST_API="$API_PORT"
                echo "  ✅ Split mesh formed in ${i}s (host on console :${CPORT}, api :${API_PORT})"
                break 2
            fi
        fi
    done

    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "❌ Split mesh failed to form within ${MAX_WAIT}s"
        echo "--- Node A log tail ---"
        tail -40 "$LOG_A" || true
        echo "--- Node B log tail ---"
        tail -40 "$LOG_B" || true
        exit 1
    fi

    if [ $((i % 15)) -eq 0 ]; then
        echo "  Still waiting... (${i}s)"
    fi
    sleep 1
done

# ── Start Node C (client) ──
echo ""
echo "Starting Node C (client)..."
MESH_LLM_EPHEMERAL_KEY=1 "$MESH_LLM" \
    --client \
    --no-draft \
    --bin-dir "$BIN_DIR" \
    --port "$C_API_PORT" \
    --console "$C_CONSOLE_PORT" \
    --join "$TOKEN" \
    > "$LOG_C" 2>&1 &
C_PID=$!
echo "  PID: $C_PID"

# Wait for client to see peers
echo "Waiting for client to join mesh..."
for i in $(seq 1 60); do
    if ! kill -0 "$C_PID" 2>/dev/null; then
        echo "❌ Node C exited unexpectedly"
        tail -50 "$LOG_C" || true
        exit 1
    fi
    CLIENT_PEERS=$(curl -sf "http://localhost:${C_CONSOLE_PORT}/api/status" 2>/dev/null \
        | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('peers',[])))" 2>/dev/null || echo "0")
    if [ "$CLIENT_PEERS" -ge 2 ]; then
        echo "  ✅ Client sees $CLIENT_PEERS peers in ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "❌ Client never joined mesh"
        tail -30 "$LOG_C" || true
        exit 1
    fi
    sleep 1
done

# Give gossip a moment to propagate targets
sleep 3

# ── Verify topology from client's perspective ──
echo ""
echo "Topology from client:"
curl -sf "http://localhost:${C_CONSOLE_PORT}/api/status" | python3 -c "
import sys, json
s = json.load(sys.stdin)
for p in s.get('peers', []):
    serving = ', '.join(p.get('serving_models', [])) or p.get('serving', 'none')
    print(f'  {p[\"id\"][:8]}  role={p[\"role\"]:8s}  serving={serving}')
" 2>/dev/null || echo "  (failed to parse)"

# ── THE TEST: inference through the client ──
echo ""
MODEL_NAME=$(basename "$MODEL" .gguf)
echo "Testing /v1/chat/completions through Client (port $C_API_PORT)..."
echo "  Client has no local inference — must route to a peer."
echo "  If it picks the Worker (rpc-server only), this fails."

# Retry a few times — tunnel establishment can take a moment
CONTENT=""
for attempt in 1 2 3 4 5; do
    RESPONSE=$(curl -s --max-time 30 -w "\n%{http_code}" "http://localhost:${C_API_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL_NAME}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}],
            \"max_tokens\": 16,
            \"temperature\": 0
        }" 2>&1) || true

    HTTP_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" = "200" ]; then
        CONTENT=$(echo "$BODY" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "")
        if [ -n "$CONTENT" ]; then
            echo "  ✅ Response (attempt $attempt): $CONTENT"
            break
        fi
        echo "  ⚠️  Attempt $attempt: 200 but empty content — split routing bug?"
        echo "  Raw: $BODY"
    else
        echo "  ⚠️  Attempt $attempt: HTTP $HTTP_CODE (tunnel may not be ready)"
    fi

    if [ "$attempt" -lt 5 ]; then
        sleep 3
    fi
done

if [ -z "$CONTENT" ]; then
    echo "❌ Client inference failed after 5 attempts"
    echo "  Last HTTP code: $HTTP_CODE"
    echo "  Last body: $BODY"
    echo "--- Client log tail ---"
    tail -30 "$LOG_C" || true
    echo "--- Host log tail ---"
    tail -20 /tmp/mesh-llm-split-a.log /tmp/mesh-llm-split-b.log || true
    exit 1
fi

# ── Also verify Host directly ──
echo ""
echo "Testing Host directly (port $HOST_API)..."
if ! RESPONSE2=$(curl -s --max-time 60 "http://localhost:${HOST_API}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}],
        \"max_tokens\": 16,
        \"temperature\": 0
    }" 2>&1); then
    echo "❌ Host direct inference failed"
    exit 1
fi

CONTENT2=$(echo "$RESPONSE2" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "")
if [ -z "$CONTENT2" ]; then
    echo "❌ Empty response from host"
    exit 1
fi
echo "  ✅ Response: $CONTENT2"

echo ""
echo "=== Split-mode routing test passed ==="
