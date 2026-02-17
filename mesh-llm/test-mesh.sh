#!/bin/bash
# End-to-end mesh test against brad (remote Mac Studio M4 Max 128GB)
# Usage: ./test-mesh.sh [mode]
#   mode: client (default) — lite client proxy test
#         distributed      — both machines as workers, tensor split inference
set -euo pipefail

MODE="${1:-client}"
REMOTE="mic@home.dwyer.au"
REMOTE_PORT=23632
LOCAL_BIN="$(dirname "$0")/target/release/mesh-llm"
LOCAL_BIN_DIR="$(dirname "$0")/../llama.cpp/build/bin"
REMOTE_BIN="~/bin/mesh-llm"
MODEL_NAME="Qwen2.5-3B-Instruct-Q4_K_M.gguf"
REMOTE_MODEL="~/.models/$MODEL_NAME"
LOCAL_MODEL="$HOME/.models/$MODEL_NAME"
LOCAL_HTTP=8080
REMOTE_LOG="/tmp/mesh-test.log"
LOCAL_LOG="/tmp/mesh-test-local.log"

cleanup() {
    echo "Cleaning up..."
    [ -n "${LOCAL_PID:-}" ] && kill $LOCAL_PID 2>/dev/null || true
    ssh -p $REMOTE_PORT $REMOTE "pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server" 2>/dev/null || true
    # Also kill local llama processes if distributed mode
    pkill -f "rpc-server.*50052" 2>/dev/null || true
    pkill -f "llama-server.*$LOCAL_HTTP" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Mode: $MODE ==="
echo "=== Killing old processes ==="
ssh -p $REMOTE_PORT $REMOTE "pkill -f mesh-llm; pkill -f rpc-server; pkill -f llama-server" 2>/dev/null || true
pkill -f "mesh-llm" 2>/dev/null || true
pkill -f "rpc-server" 2>/dev/null || true
sleep 1

echo "=== Starting remote (brad) ==="
ssh -p $REMOTE_PORT $REMOTE "cd ~/bin && RUST_LOG=info nohup ./mesh-llm \
  --model $REMOTE_MODEL --bin-dir ~/bin --bind-port 7842 \
  > $REMOTE_LOG 2>&1 &"

echo "Waiting for remote llama-server..."
for i in $(seq 1 30); do
    if ssh -p $REMOTE_PORT $REMOTE "grep -q 'llama-server ready' $REMOTE_LOG 2>/dev/null" 2>/dev/null; then break; fi
    sleep 1
done

if ! ssh -p $REMOTE_PORT $REMOTE "grep -q 'llama-server ready' $REMOTE_LOG 2>/dev/null" 2>/dev/null; then
    echo "FAIL: Remote llama-server never started"
    ssh -p $REMOTE_PORT $REMOTE "tail -20 $REMOTE_LOG" 2>&1
    exit 1
fi

TOKEN=$(ssh -p $REMOTE_PORT $REMOTE "grep 'Invite token:' $REMOTE_LOG | head -1 | sed 's/Invite token: //'" 2>/dev/null)
if [ -z "$TOKEN" ]; then
    echo "FAIL: No invite token"
    exit 1
fi
echo "Got token: ${TOKEN:0:40}..."

if [ "$MODE" = "distributed" ]; then
    echo "=== Starting local (worker + auto-election) ==="
    RUST_LOG=info $LOCAL_BIN \
      --model "$LOCAL_MODEL" \
      --bin-dir "$LOCAL_BIN_DIR" \
      --join "$TOKEN" \
      --port $LOCAL_HTTP \
      > $LOCAL_LOG 2>&1 &
    LOCAL_PID=$!

    echo "Waiting for election + llama-server restart..."
    # Brad should restart llama-server with us as a worker
    for i in $(seq 1 45); do
        if grep -q "WORKER" $LOCAL_LOG 2>/dev/null || grep -q "HOST" $LOCAL_LOG 2>/dev/null; then break; fi
        if ! kill -0 $LOCAL_PID 2>/dev/null; then
            echo "FAIL: Local process died"
            tail -20 $LOCAL_LOG
            exit 1
        fi
        sleep 1
    done

    # Wait for llama-server to be ready on whichever node is host
    sleep 10

    # The host is brad (103GB > 51GB). Query brad's llama-server directly via SSH tunnel.
    LLAMA_PORT=$(ssh -p $REMOTE_PORT $REMOTE "grep 'llama-server ready' $REMOTE_LOG | tail -1 | grep -oE '[0-9]+$'" 2>/dev/null)
    echo "Remote llama-server on port ${LLAMA_PORT:-unknown}"

    echo ""
    echo "=== Local node role ==="
    grep -E "HOST|WORKER" $LOCAL_LOG | tail -1 || echo "(unknown)"

    echo ""
    echo "=== Remote log (election + restart) ==="
    ssh -p $REMOTE_PORT $REMOTE "grep -E 'Peer added|election|HOST|WORKER|llama-server|tensor.split|restart' $REMOTE_LOG" 2>/dev/null | tail -10

    echo ""
    echo "=== Test: inference through mesh ==="
    # Use lite client approach: the host is brad, we can query via the existing QUIC tunnel
    # Actually in distributed mode we're a worker, not a client. Need to hit brad's port.
    # Use SSH port forward for the test query.
    RESPONSE=$(ssh -p $REMOTE_PORT $REMOTE "curl -s --max-time 15 http://localhost:${LLAMA_PORT:-8090}/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Reply with just the number.\"}],\"max_tokens\":10}'" 2>/dev/null)
    ANSWER=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null)
    TIMINGS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); u=d.get('usage',{}); t=d.get('timings',{}); print(f\"prompt={t.get('prompt_ms',0):.0f}ms predicted={t.get('predicted_ms',0):.0f}ms\")" 2>/dev/null)
    if [ -n "$ANSWER" ]; then
        echo "Answer: $ANSWER ($TIMINGS)"
        echo "PASS"
    else
        echo "FAIL: $RESPONSE"
        exit 1
    fi

else
    # Client mode (default)
    echo "=== Starting local (lite client) ==="
    RUST_LOG=info $LOCAL_BIN --client --join "$TOKEN" --port $LOCAL_HTTP > $LOCAL_LOG 2>&1 &
    LOCAL_PID=$!

    echo "Waiting for connection..."
    for i in $(seq 1 20); do
        if grep -q "Lite client ready" $LOCAL_LOG 2>/dev/null; then break; fi
        if ! kill -0 $LOCAL_PID 2>/dev/null; then
            echo "FAIL: Local process died"
            tail -20 $LOCAL_LOG
            exit 1
        fi
        sleep 1
    done

    if ! grep -q "Lite client ready" $LOCAL_LOG 2>/dev/null; then
        echo "FAIL: Never connected"
        tail -20 $LOCAL_LOG
        exit 1
    fi
    echo "Connected!"

    echo ""
    echo "=== Test 1: /v1/models ==="
    MODELS=$(curl -s --max-time 5 http://localhost:$LOCAL_HTTP/v1/models 2>/dev/null)
    if echo "$MODELS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null; then
        echo "PASS"
    else
        echo "FAIL: $MODELS"
        exit 1
    fi

    echo ""
    echo "=== Test 2: /v1/chat/completions ==="
    RESPONSE=$(curl -s --max-time 15 http://localhost:$LOCAL_HTTP/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"max_tokens":10}' 2>/dev/null)
    ANSWER=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null)
    TIMINGS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('timings',{}); print(f\"prompt={t.get('prompt_ms',0):.0f}ms predicted={t.get('predicted_ms',0):.0f}ms\")" 2>/dev/null)
    if [ -n "$ANSWER" ]; then
        echo "Answer: $ANSWER ($TIMINGS)"
        echo "PASS"
    else
        echo "FAIL: $RESPONSE"
        exit 1
    fi
fi

echo ""
echo "=== Connection info ==="
grep -E "STUN|rpc-server on" $LOCAL_LOG 2>/dev/null || true
grep "STUN" <(ssh -p $REMOTE_PORT $REMOTE "cat $REMOTE_LOG" 2>/dev/null) 2>/dev/null || true
grep "Peer added" $LOCAL_LOG 2>/dev/null || true

echo ""
echo "All tests passed ✅"
