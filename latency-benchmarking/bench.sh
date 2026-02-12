#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/llama.cpp/build"
MODEL="$HOME/.models/GLM-4.7-Flash-Q4_K_M.gguf"
SERVER_PORT=8080
RPC_BASE_PORT=50052
PROXY_BASE_PORT=60052

# 20 tokens keeps even the slow runs under ~10s
PROMPT='{"model":"test","messages":[{"role":"user","content":"Say hello in 5 languages"}],"max_tokens":20}'

nuke() {
    pkill -9 -f "llama-server" 2>/dev/null || true
    pkill -9 -f "latency-proxy" 2>/dev/null || true
    pkill -9 -f "rpc-server" 2>/dev/null || true
    sleep 1
}

start_rpc_workers() {
    local n=$1
    for i in $(seq 0 $((n - 1))); do
        local port=$((RPC_BASE_PORT + i))
        local dev="CPU"; [[ $i -eq 0 ]] && dev="MTL0"
        nohup "$BUILD_DIR/bin/rpc-server" -d "$dev" -p "$port" >"/tmp/rpc-${port}.log" 2>&1 &
    done
    for _ in $(seq 1 10); do
        local ok=1
        for i in $(seq 0 $((n - 1))); do
            lsof -i ":$((RPC_BASE_PORT + i))" -sTCP:LISTEN >/dev/null 2>&1 || ok=0
        done
        [[ $ok -eq 1 ]] && return
        sleep 1
    done
    echo "FAIL: rpc workers" >&2; exit 1
}

start_proxies() {
    local n=$1 ms=$2
    [[ "$ms" == "0" ]] && return
    for i in $(seq 0 $((n - 1))); do
        nohup python3 "$SCRIPT_DIR/latency-proxy.py" \
            --listen-port "$((PROXY_BASE_PORT + i))" \
            --target-port "$((RPC_BASE_PORT + i))" \
            --latency-ms "$ms" >"/tmp/proxy-$((PROXY_BASE_PORT + i)).log" 2>&1 &
    done
    for _ in $(seq 1 8); do
        local ok=1
        for i in $(seq 0 $((n - 1))); do
            lsof -i ":$((PROXY_BASE_PORT + i))" -sTCP:LISTEN >/dev/null 2>&1 || ok=0
        done
        [[ $ok -eq 1 ]] && return
        sleep 0.5
    done
    echo "FAIL: proxies" >&2; exit 1
}

start_server() {
    local nodes=$1
    local ms=$2
    local nrpc=$((nodes - 1))
    local rpc=""
    for i in $(seq 0 $((nrpc - 1))); do
        local p; [[ "$ms" != "0" ]] && p=$((PROXY_BASE_PORT + i)) || p=$((RPC_BASE_PORT + i))
        [[ -n "$rpc" ]] && rpc="$rpc,"
        rpc="${rpc}127.0.0.1:${p}"
    done
    local split
    split=$(python3 -c "n=$nodes;p=[round(1.0/n,4)]*n;p[-1]=round(1-sum(p[:-1]),4);print(','.join(str(x) for x in p))")

    nohup "$BUILD_DIR/bin/llama-server" \
        -m "$MODEL" --rpc "$rpc" -ngl 99 \
        --host 0.0.0.0 --port "$SERVER_PORT" \
        --tensor-split "$split" >"/tmp/llama-server.log" 2>&1 &

    for _ in $(seq 1 120); do
        curl -sf http://localhost:$SERVER_PORT/health >/dev/null 2>&1 && return
        sleep 1
    done
    echo "FAIL: server load timeout" >&2; tail -5 /tmp/llama-server.log; exit 1
}

run_single() {
    local nodes=$1
    local ms=$2
    local nrpc=$((nodes - 1))
    nuke
    start_rpc_workers "$nrpc"
    start_proxies "$nrpc" "$ms"
    start_server "$nodes" "$ms"

    # inference with timeout
    local r
    r=$(curl -s --max-time 30 http://localhost:$SERVER_PORT/v1/chat/completions \
        -H "Content-Type: application/json" -d "$PROMPT" 2>/dev/null) || r=""

    if [[ -z "$r" ]]; then
        printf "  %-8s %-12s %s\n" "${nodes}" "${ms}ms" "TIMEOUT"
        nuke; return
    fi

    local tps
    tps=$(grep "eval time" /tmp/llama-server.log | grep -v "prompt" | tail -1 | grep -oE '[0-9.]+ tokens per second' | awk '{print $1}') || tps=""
    if [[ -n "$tps" ]]; then
        printf "  %-8s %-12s %s tok/s\n" "$nodes" "${ms}ms" "$tps"
    else
        printf "  %-8s %-12s %s\n" "$nodes" "${ms}ms" "error"
    fi
    nuke
}

# --- Main ---
nuke

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Distributed Inference Benchmark                            ║"
echo "║  GLM-4.7-Flash Q4_K_M · Apple M4 Max 64GB · 20 tok/run    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
printf "  %-8s %-12s %s\n" "Nodes" "Latency" "Speed"
printf "  %-8s %-12s %s\n" "-----" "-------" "-----"

# 3 nodes: 0, 5, 10, 20ms
for ms in 0 5 10 20; do
    run_single 3 "$ms"
done

# 4 nodes: 0, 5, 10ms
for ms in 0 5 10; do
    run_single 4 "$ms"
done

# 5 nodes: 0, 5ms
for ms in 0 5; do
    run_single 5 "$ms"
done

echo ""
echo "Done."
