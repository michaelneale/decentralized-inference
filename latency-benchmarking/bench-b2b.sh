#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL="$HOME/.models/GLM-4.7-Flash-Q4_K_M.gguf"
SERVER_PORT=8080
RPC_BASE_PORT=50052
PROXY_BASE_PORT=60052

nuke() {
    pkill -9 -f "llama-server" 2>/dev/null || true
    pkill -9 -f "latency-proxy" 2>/dev/null || true
    pkill -9 -f "rpc-server" 2>/dev/null || true
    sleep 1
}

start_rpc_workers() {
    local build_dir=$1
    local n=$2
    for i in $(seq 0 $((n - 1))); do
        local port=$((RPC_BASE_PORT + i))
        local dev="CPU"; [[ $i -eq 0 ]] && dev="MTL0"
        nohup "$build_dir/bin/rpc-server" -d "$dev" -p "$port" >"/tmp/rpc-${port}.log" 2>&1 &
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
    local n=$1
    local ms=$2
    local mode=$3
    [[ "$ms" == "0" ]] && return
    for i in $(seq 0 $((n - 1))); do
        nohup python3 "$SCRIPT_DIR/latency-proxy.py" \
            --listen-port "$((PROXY_BASE_PORT + i))" \
            --target-port "$((RPC_BASE_PORT + i))" \
            --latency-ms "$ms" \
            --mode "$mode" \
            >"/tmp/proxy-$((PROXY_BASE_PORT + i)).log" 2>&1 &
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
    local build_dir=$1
    local nodes=$2
    local ms=$3
    local nrpc=$((nodes - 1))
    local rpc=""
    for i in $(seq 0 $((nrpc - 1))); do
        local p; [[ "$ms" != "0" ]] && p=$((PROXY_BASE_PORT + i)) || p=$((RPC_BASE_PORT + i))
        [[ -n "$rpc" ]] && rpc="$rpc,"
        rpc="${rpc}127.0.0.1:${p}"
    done
    local split
    split=$(python3 -c "n=$nodes;p=[round(1.0/n,4)]*n;p[-1]=round(1-sum(p[:-1]),4);print(','.join(str(x) for x in p))")

    nohup "$build_dir/bin/llama-server" \
        -m "$MODEL" --rpc "$rpc" -ngl 99 \
        --host 0.0.0.0 --port "$SERVER_PORT" \
        --tensor-split "$split" >"/tmp/llama-server.log" 2>&1 &

    for _ in $(seq 1 120); do
        curl -sf http://localhost:$SERVER_PORT/health >/dev/null 2>&1 && return
        sleep 1
    done
    echo "FAIL: server load timeout" >&2; tail -10 /tmp/llama-server.log; exit 1
}

run_single() {
    local build_dir=$1
    local label=$2
    local nodes=$3
    local ms=$4
    local mode=$5
    local nrpc=$((nodes - 1))
    nuke
    start_rpc_workers "$build_dir" "$nrpc"
    start_proxies "$nrpc" "$ms" "$mode"
    start_server "$build_dir" "$nodes" "$ms"

    # Measure using streaming client
    local result
    result=$(python3 "$SCRIPT_DIR/measure.py" \
        --url "http://localhost:$SERVER_PORT/v1/chat/completions" \
        --prompt "Say hello in 5 languages" \
        --max-tokens 20 2>/dev/null) || result='{"error":"timeout"}'

    # Parse JSON result
    local ttft total tps
    ttft=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); v=d.get('ttft_ms'); print(f'{v:.0f}ms' if v else 'err')" 2>/dev/null) || ttft="err"
    total=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); v=d.get('total_ms'); print(f'{v:.0f}ms' if v else 'err')" 2>/dev/null) || total="err"
    tps=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); v=d.get('tok_s'); print(f'{v:.1f}' if v else 'err')" 2>/dev/null) || tps="err"

    printf "  %-10s %-6s %-8s %-12s %-12s %s tok/s\n" "$label" "$nodes" "${ms}ms" "$ttft" "$total" "$tps"
    nuke
}

# --- Main ---
nuke

UPSTREAM="$PROJECT_DIR/llama.cpp/build"
B2B="$PROJECT_DIR/llama.cpp-rpc-b2b/build"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  RPC Backend-to-Backend Comms Benchmark                         ║"
echo "║  GLM-4.7-Flash Q4_K_M · Apple M4 Max 64GB · 20 tok/run        ║"
echo "║  Proxy mode: transfer (delays GET/SET_TENSOR + GRAPH_COMPUTE)  ║"
echo "║  TTFT = client-measured time to first token (streaming SSE)     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
printf "  %-10s %-6s %-8s %-12s %-12s %s\n" "Build" "Nodes" "Latency" "TTFT" "Total" "Speed"
printf "  %-10s %-6s %-8s %-12s %-12s %s\n" "-----" "-----" "-------" "----" "-----" "-----"

# 3 nodes: 5, 10, 20ms
for ms in 5 10 20; do
    run_single "$UPSTREAM" "upstream" 3 "$ms" "transfer"
    run_single "$B2B"      "b2b"      3 "$ms" "transfer"
done

echo ""

# 4 nodes: 5, 10ms
for ms in 5 10; do
    run_single "$UPSTREAM" "upstream" 4 "$ms" "transfer"
    run_single "$B2B"      "b2b"      4 "$ms" "transfer"
done

echo ""

# 5 nodes: 5ms
for ms in 5; do
    run_single "$UPSTREAM" "upstream" 5 "$ms" "transfer"
    run_single "$B2B"      "b2b"      5 "$ms" "transfer"
done

echo ""
echo "Done."
