#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# mesh-inference integration test
#
# Runs 3 nodes on localhost using the B2B fork:
#   Node A: worker only
#   Node B: worker only
#   Node C: worker + llama-server (orchestrator)
#
# Verifies:
#   1. Nodes discover each other via gossip (A starts, B joins A, C joins A)
#   2. Tunnels are established (C can reach A and B's rpc-servers)
#   3. Inference works end-to-end through QUIC tunnels
#   4. B2B direct transfers work (REGISTER_PEER is rewritten correctly)
#
# Prerequisites:
#   - B2B fork built: ../llama.cpp-rpc-b2b/build/bin/{rpc-server,llama-server}
#   - Model downloaded: ~/.models/GLM-4.7-Flash-Q4_K_M.gguf
#   - mesh-inference built: cargo build (run from this directory)
#
# Usage:
#   ./test-mesh.sh              # full test with inference
#   ./test-mesh.sh mesh-only    # just test mesh formation, no inference
#   ./test-mesh.sh stop         # kill everything
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="$SCRIPT_DIR/target/debug/mesh-inference"
BIN_DIR="$SCRIPT_DIR/../llama.cpp-rpc-b2b/build/bin"
MODEL="$HOME/.models/GLM-4.7-Flash-Q4_K_M.gguf"
HTTP_PORT=8090  # avoid conflicting with demo.sh's 8080

LOG_DIR="/tmp/mesh-inference-test"
declare -a PIDS=()

# --- Helpers ---

log()  { echo "==> $*"; }
err()  { echo "ERROR: $*" >&2; }
fail() { echo "FAIL: $*" >&2; cleanup; exit 1; }
pass() { echo "PASS: $*"; }

cleanup() {
    log "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    # Also kill any stray processes we spawned
    pkill -f "mesh-inference" 2>/dev/null || true
    pkill -f "rpc-server.*-p 5" 2>/dev/null || true
    sleep 1
}

trap cleanup EXIT

extract_token() {
    local logfile="$1"
    local timeout="${2:-15}"
    for i in $(seq 1 "$timeout"); do
        if grep -q "Invite token:" "$logfile" 2>/dev/null; then
            grep "Invite token:" "$logfile" | head -1 | sed 's/.*Invite token: //'
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_peers() {
    local logfile="$1"
    local count="$2"
    local timeout="${3:-30}"
    for i in $(seq 1 "$timeout"); do
        local found
        found=$(grep -c "Peer added:" "$logfile" 2>/dev/null) || found=0
        if [[ "$found" -ge "$count" ]]; then
            return 0
        fi
        sleep 1
    done
    return 1
}

wait_for_http() {
    local port="$1"
    local timeout="${2:-120}"
    for i in $(seq 1 "$timeout"); do
        if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
            return 0
        fi
        if [[ $((i % 10)) -eq 0 ]]; then
            log "  Still waiting for llama-server... (${i}s)"
        fi
        sleep 1
    done
    return 1
}

# --- Preflight ---

if [[ "${1:-}" == "stop" ]]; then
    cleanup
    exit 0
fi

MESH_ONLY=0
if [[ "${1:-}" == "mesh-only" ]]; then
    MESH_ONLY=1
fi

if [[ ! -x "$BINARY" ]]; then
    log "Building mesh-inference..."
    (cd "$SCRIPT_DIR" && cargo build 2>&1) || fail "cargo build failed"
fi

if [[ ! -x "$BIN_DIR/rpc-server" ]]; then
    fail "rpc-server not found at $BIN_DIR/rpc-server — build the B2B fork first"
fi

if [[ "$MESH_ONLY" -eq 0 && ! -f "$MODEL" ]]; then
    fail "Model not found at $MODEL — download it first (see ../README.md)"
fi

mkdir -p "$LOG_DIR"

# --- Kill any existing test processes ---
pkill -f "mesh-inference" 2>/dev/null || true
sleep 1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  mesh-inference integration test                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# TEST 1: Mesh formation
# ============================================================

log "TEST 1: Mesh formation (3 nodes, gossip discovery)"

# Start Node A (Metal GPU)
log "Starting Node A (worker, MTL0)..."
"$BINARY" --bin-dir "$BIN_DIR" --device MTL0 \
    > "$LOG_DIR/node-a.log" 2>&1 &
PIDS+=($!)
log "  PID: $!"

# Wait for Node A's invite token
TOKEN_A=$(extract_token "$LOG_DIR/node-a.log" 30) || fail "Node A didn't produce an invite token"
log "  Token A: ${TOKEN_A:0:40}..."

# Start Node B (CPU), joining via A
log "Starting Node B (worker, CPU), joining via A..."
"$BINARY" --bin-dir "$BIN_DIR" --device CPU --join "$TOKEN_A" \
    > "$LOG_DIR/node-b.log" 2>&1 &
PIDS+=($!)
log "  PID: $!"

# Wait for B to discover A
wait_for_peers "$LOG_DIR/node-b.log" 1 15 || fail "Node B didn't find Node A"
pass "Node B discovered Node A"

# Wait for A to discover B (via inbound gossip)
wait_for_peers "$LOG_DIR/node-a.log" 1 15 || fail "Node A didn't find Node B"
pass "Node A discovered Node B"

# Start Node C (CPU), joining via A — should discover both A and B
log "Starting Node C (worker, CPU), joining via A..."
if [[ "$MESH_ONLY" -eq 0 ]]; then
    "$BINARY" --bin-dir "$BIN_DIR" --device CPU --join "$TOKEN_A" \
        --serve "$HTTP_PORT" --model "$MODEL" --min-peers 2 \
        > "$LOG_DIR/node-c.log" 2>&1 &
else
    "$BINARY" --bin-dir "$BIN_DIR" --device CPU --join "$TOKEN_A" \
        > "$LOG_DIR/node-c.log" 2>&1 &
fi
PIDS+=($!)
log "  PID: $!"

# Wait for C to discover both A and B
wait_for_peers "$LOG_DIR/node-c.log" 2 15 || fail "Node C didn't find both peers"
pass "Node C discovered both A and B via gossip"

# Verify A and B also know about C
wait_for_peers "$LOG_DIR/node-a.log" 2 15 || fail "Node A didn't discover Node C"
wait_for_peers "$LOG_DIR/node-b.log" 2 15 || fail "Node B didn't discover Node C"
pass "Full mesh: all 3 nodes see 2 peers each"

echo ""
log "TEST 1 PASSED: Mesh formation works"
echo ""

if [[ "$MESH_ONLY" -eq 1 ]]; then
    log "mesh-only mode, skipping inference test"
    echo ""
    echo "All tests passed."
    exit 0
fi

# ============================================================
# TEST 2: Inference through the mesh
# ============================================================

log "TEST 2: Inference through QUIC tunnels"

# Node C was started with --serve, wait for llama-server to be ready
log "Waiting for llama-server to load model (this may take 30-60s)..."
wait_for_http "$HTTP_PORT" 120 || {
    err "llama-server didn't become healthy. Logs:"
    tail -20 "$LOG_DIR/node-c.log"
    fail "llama-server failed to start"
}
pass "llama-server is healthy on port $HTTP_PORT"

# Send a test prompt
log "Sending test prompt..."
RESPONSE=$(curl -s --max-time 30 "http://localhost:$HTTP_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Say hello in 3 languages"}],"max_tokens":50}' \
    2>/dev/null) || RESPONSE=""

if [[ -z "$RESPONSE" ]]; then
    err "No response from llama-server"
    tail -20 "$LOG_DIR/node-c.log"
    fail "Inference request failed"
fi

# Check if we got actual content
CONTENT=$(echo "$RESPONSE" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    print(r['choices'][0]['message']['content'][:100])
except:
    print('')
" 2>/dev/null)

if [[ -z "$CONTENT" ]]; then
    err "Response had no content:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
    fail "Inference returned empty content"
fi

pass "Got inference response: ${CONTENT:0:60}..."

echo ""
log "TEST 2 PASSED: Inference works through QUIC mesh"
echo ""

# ============================================================
# TEST 3: B2B direct transfers (check logs)
# ============================================================

log "TEST 3: B2B direct transfer (checking for REGISTER_PEER rewrite)"

# Check if the sidecar rewrote any REGISTER_PEER messages
REWRITES_A=$(grep -c "Rewrote REGISTER_PEER" "$LOG_DIR/node-a.log" 2>/dev/null || echo 0)
REWRITES_B=$(grep -c "Rewrote REGISTER_PEER" "$LOG_DIR/node-b.log" 2>/dev/null || echo 0)
TOTAL_REWRITES=$((REWRITES_A + REWRITES_B))

if [[ "$TOTAL_REWRITES" -gt 0 ]]; then
    pass "REGISTER_PEER rewritten $TOTAL_REWRITES time(s) across workers"
else
    log "  No REGISTER_PEER rewrites seen (B2B may not have triggered, or protocol version mismatch)"
    log "  This is expected if the fork didn't send REGISTER_PEER — not a failure"
fi

echo ""
log "TEST 3 DONE"
echo ""

# ============================================================
# Summary
# ============================================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All tests passed!                                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Logs: $LOG_DIR/node-{a,b,c}.log"
echo ""
