#!/usr/bin/env bash
set -euo pipefail

# Runs a local 3-node benchmark:
# - node1: serving host
# - node2: serving host
# - client1: passive entrypoint that makes the routing decision
#
# Compares sticky-only routing against prefix-only routing for an agentic workload
# that keeps the scaffold constant while varying the first user turn.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESH_BIN="${MESH_BIN:-$ROOT/target/release/mesh-llm}"
BIN_DIR="${BIN_DIR:-$ROOT/llama.cpp/build/bin}"
MODEL_PATH="${MODEL_PATH:-$HOME/.models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH" .gguf)}"
NODE1_API_PORT="${NODE1_API_PORT:-9337}"
NODE2_API_PORT="${NODE2_API_PORT:-9338}"
NODE1_CONSOLE_PORT="${NODE1_CONSOLE_PORT:-3131}"
NODE2_CONSOLE_PORT="${NODE2_CONSOLE_PORT:-3132}"
CLIENT1_API_PORT="${CLIENT1_API_PORT:-9339}"
CLIENT1_CONSOLE_PORT="${CLIENT1_CONSOLE_PORT:-3133}"
BIND_PORT="${BIND_PORT:-7842}"
REQUESTS="${REQUESTS:-12}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
ENTRY_MODE="${ENTRY_MODE:-active}"
NO_DRAFT="${NO_DRAFT:-0}"

TMP_DIR="$(mktemp -d /tmp/mesh-prefix-affinity.XXXXXX)"
CONFIG_PATH="$TMP_DIR/bench-config.toml"
NODE1_LOG="$TMP_DIR/node1.log"
NODE2_LOG="$TMP_DIR/node2.log"
CLIENT1_LOG="$TMP_DIR/client1.log"

stop_nodes() {
  if [[ -n "${CLIENT1_PID:-}" ]] && kill -0 "$CLIENT1_PID" 2>/dev/null; then
    kill "$CLIENT1_PID" 2>/dev/null || true
    wait "$CLIENT1_PID" 2>/dev/null || true
  fi
  if [[ -n "${NODE2_PID:-}" ]] && kill -0 "$NODE2_PID" 2>/dev/null; then
    kill "$NODE2_PID" 2>/dev/null || true
    wait "$NODE2_PID" 2>/dev/null || true
  fi
  if [[ -n "${NODE1_PID:-}" ]] && kill -0 "$NODE1_PID" 2>/dev/null; then
    kill "$NODE1_PID" 2>/dev/null || true
    wait "$NODE1_PID" 2>/dev/null || true
  fi
  if [[ -n "${LLAMA_SERVER_PID:-}" ]] && kill -0 "$LLAMA_SERVER_PID" 2>/dev/null; then
    kill "$LLAMA_SERVER_PID" 2>/dev/null || true
    wait "$LLAMA_SERVER_PID" 2>/dev/null || true
  fi
  if [[ -n "${RPC_SERVER_PID:-}" ]] && kill -0 "$RPC_SERVER_PID" 2>/dev/null; then
    kill "$RPC_SERVER_PID" 2>/dev/null || true
    wait "$RPC_SERVER_PID" 2>/dev/null || true
  fi
  unset NODE1_PID NODE2_PID CLIENT1_PID
}

cleanup() {
  stop_nodes
  if [[ "${KEEP_TMP:-0}" != "1" ]]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing ${label}: $path" >&2
    exit 1
  fi
}

wait_for_token() {
  local log_file="$1"
  for _ in $(seq 1 120); do
    if grep -Eq "Invite token:|Invite:" "$log_file" 2>/dev/null; then
      grep -E "Invite token:|Invite:" "$log_file" | head -1 | sed -E 's/Invite token: |Invite: //'
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for invite token in $log_file" >&2
  exit 1
}

wait_for_http() {
  local url="$1"
  for _ in $(seq 1 120); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for $url" >&2
  exit 1
}

wait_for_multiple_hosts() {
  local status_url="$1"
  local expected_hosts="${2:-2}"
  python3 - "$status_url" "$MODEL_NAME" "$expected_hosts" <<'PY'
import json
import sys
import time
import urllib.request

status_url = sys.argv[1]
model_name = sys.argv[2]
expected_hosts = int(sys.argv[3])

for _ in range(120):
    try:
        with urllib.request.urlopen(status_url, timeout=5) as resp:
            payload = json.load(resp)
        serving_count = 0

        if payload.get("is_host") and (
            payload.get("model_name") == model_name
            or model_name in payload.get("serving_models", [])
        ):
            serving_count += 1

        for peer in payload.get("peers", []):
            if peer.get("role") == "Host" and (
                peer.get("serving") == model_name
                or model_name in peer.get("serving_models", [])
            ):
                serving_count += 1

        if serving_count >= expected_hosts:
            sys.exit(0)
    except Exception:
        pass
    time.sleep(1)

print(
    f"Timed out waiting for {expected_hosts} hosts serving {model_name}",
    file=sys.stderr,
)
sys.exit(1)
PY
}

start_phase() {
  local phase="$1"
  shift
  local -a extra_env=("$@")
  local -a launch_env=("RUST_LOG=info")
  local -a extra_args=()
  if ((${#extra_env[@]} > 0)); then
    launch_env=("${extra_env[@]}" "${launch_env[@]}")
  fi
  if [[ "$ENTRY_MODE" == "passive" ]]; then
    launch_env=("MESH_LLM_FORCE_DUPLICATE_HOSTS=1" "${launch_env[@]}")
  fi
  if [[ "$NO_DRAFT" == "1" ]]; then
    extra_args+=(--no-draft)
  fi

  : >"$NODE1_LOG"
  : >"$NODE2_LOG"
  : >"$CLIENT1_LOG"
  cat >"$CONFIG_PATH" <<'EOF'
[[plugin]]
name = "blackboard"
enabled = false
EOF

  env "${launch_env[@]}" \
    "$MESH_BIN" \
    --config "$CONFIG_PATH" \
    "${extra_args[@]}" \
    --model "$MODEL_PATH" \
    --bin-dir "$BIN_DIR" \
    --bind-port "$BIND_PORT" \
    --port "$NODE1_API_PORT" \
    --console "$NODE1_CONSOLE_PORT" \
    >"$NODE1_LOG" 2>&1 &
  NODE1_PID=$!

  local token
  token="$(wait_for_token "$NODE1_LOG")"

  env "${launch_env[@]}" \
    MESH_LLM_EPHEMERAL_KEY=1 \
    "$MESH_BIN" \
    --config "$CONFIG_PATH" \
    "${extra_args[@]}" \
    --model "$MODEL_PATH" \
    --bin-dir "$BIN_DIR" \
    --join "$token" \
    --port "$NODE2_API_PORT" \
    --console "$NODE2_CONSOLE_PORT" \
    >"$NODE2_LOG" 2>&1 &
  NODE2_PID=$!

  wait_for_http "http://127.0.0.1:${NODE1_CONSOLE_PORT}/api/status"
  wait_for_http "http://127.0.0.1:${NODE2_CONSOLE_PORT}/api/status"

  env "${launch_env[@]}" \
    "$MESH_BIN" \
    --config "$CONFIG_PATH" \
    "${extra_args[@]}" \
    --client \
    --join "$token" \
    --port "$CLIENT1_API_PORT" \
    --console "$CLIENT1_CONSOLE_PORT" \
    >"$CLIENT1_LOG" 2>&1 &
  CLIENT1_PID=$!

  wait_for_http "http://127.0.0.1:${CLIENT1_CONSOLE_PORT}/api/status"
  wait_for_multiple_hosts "http://127.0.0.1:${CLIENT1_CONSOLE_PORT}/api/status" 2

  echo "Phase ${phase} ready"
}

run_workload() {
  local label="$1"
  local output_path="$2"
  local api_url
  local status_url
  case "$ENTRY_MODE" in
    active)
      api_url="http://127.0.0.1:${NODE1_API_PORT}/v1/chat/completions"
      status_url="http://127.0.0.1:${NODE1_CONSOLE_PORT}/api/status"
      ;;
    passive)
      api_url="http://127.0.0.1:${CLIENT1_API_PORT}/v1/chat/completions"
      status_url="http://127.0.0.1:${CLIENT1_CONSOLE_PORT}/api/status"
      ;;
    *)
      echo "Unknown ENTRY_MODE: $ENTRY_MODE" >&2
      exit 1
      ;;
  esac
  python3 - "$label" "$output_path" "$api_url" "$status_url" "$MODEL_NAME" "$REQUESTS" "$WARMUP_REQUESTS" <<'PY'
import json
import statistics
import sys
import time
import urllib.request

label, output_path, api_url, status_url, model_name, request_count, warmup = sys.argv[1:]
request_count = int(request_count)
warmup = int(warmup)

system_prompt = (
    "You are a coding agent operating inside a local mesh. "
    "Always reason about correctness, preserve protocol compatibility, "
    "prefer exact diffs over broad rewrites, and return concise answers. "
    "You may call tools only when necessary. "
    * 80
)

tools = [
    {
        "type": "function",
        "function": {
            "name": f"tool_{i}",
            "description": f"Tool {i} for filesystem and repo inspection",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "mode": {"type": "string", "enum": ["read", "write", "exec"]},
                },
                "required": ["path", "mode"],
            },
        },
    }
    for i in range(10)
]

def hash_bytes(data: bytes) -> int:
    acc = 0xCBF29CE484222325
    for b in data:
        acc = ((acc ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return acc

def hash_combine(a: int, b: int) -> int:
    return ((a * 31) + b) & 0xFFFFFFFFFFFFFFFF

def hash_tagged_text(acc: int, tag: str, text: str) -> int:
    acc = hash_combine(acc, hash_bytes(tag.encode("utf-8")))
    return hash_combine(acc, hash_bytes(text.encode("utf-8")))

def hash_tagged_json(acc: int, tag: str, value) -> int:
    acc = hash_combine(acc, hash_bytes(tag.encode("utf-8")))
    payload = json.dumps(value, sort_keys=False).encode("utf-8")
    return hash_combine(acc, hash_bytes(payload))

def scaffold_prefix_hash() -> int:
    acc = 0
    acc = hash_tagged_json(acc, "tools", tools)
    acc = hash_tagged_text(acc, "system", system_prompt)
    return acc

def sticky_hash_for_task(prefix_hash: int, task_text: str) -> int:
    user_hash = hash_tagged_text(0, "user", task_text)
    return hash_combine(prefix_hash, user_hash)

def build_tasks() -> list[str]:
    prefix_hash = scaffold_prefix_hash()
    tasks = []
    for i in range(request_count):
        desired_bucket = i % 2
        base = f"Task {i}: inspect module {i}, summarize the bug, and propose a minimal patch."
        task = base
        salt = 0
        while sticky_hash_for_task(prefix_hash, task) % 2 != desired_bucket:
            salt += 1
            task = f"{base} Salt {salt}."
            if salt > 1000:
                raise RuntimeError(f"Could not generate task parity for index {i}")
        tasks.append(task)
    return tasks

tasks = build_tasks()

def fetch_json(url):
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.load(resp)

def affinity_stats():
    payload = fetch_json(status_url)
    return payload.get("routing_affinity", {})

def request_payload(task_text):
    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_text},
        ],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 48,
        "stream": False,
    }

before = affinity_stats()
results = []

for idx, task in enumerate(tasks):
    payload = request_payload(task)
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as resp:
        response_body = json.load(resp)
    elapsed_ms = (time.perf_counter() - start) * 1000
    timings = response_body.get("timings", {})
    results.append(
        {
            "index": idx,
            "task": task,
            "elapsed_ms": elapsed_ms,
            "prompt_ms": timings.get("prompt_ms", 0.0),
            "prompt_tps": timings.get("prompt_per_second", 0.0),
            "predicted_ms": timings.get("predicted_ms", 0.0),
            "predicted_tps": timings.get("predicted_per_second", 0.0),
        }
    )

after = affinity_stats()
measured = results[warmup:] if warmup < len(results) else results

summary = {
    "label": label,
    "request_count": len(results),
    "warmup_requests": warmup,
    "measured_count": len(measured),
    "prompt_ms_median": statistics.median(r["prompt_ms"] for r in measured),
    "prompt_ms_mean": statistics.fmean(r["prompt_ms"] for r in measured),
    "elapsed_ms_median": statistics.median(r["elapsed_ms"] for r in measured),
    "elapsed_ms_mean": statistics.fmean(r["elapsed_ms"] for r in measured),
    "prompt_tps_mean": statistics.fmean(r["prompt_tps"] for r in measured),
    "affinity_before": before,
    "affinity_after": after,
    "results": results,
}

with open(output_path, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

print(json.dumps(summary, indent=2))
PY
}

annotate_route_distribution() {
  local output_path="$1"
  local entry_log
  case "$ENTRY_MODE" in
    active)
      entry_log="$NODE1_LOG"
      ;;
    passive)
      entry_log="$CLIENT1_LOG"
      ;;
    *)
      echo "Unknown ENTRY_MODE: $ENTRY_MODE" >&2
      exit 1
      ;;
  esac
  python3 - "$output_path" "$entry_log" "$NODE1_LOG" "$NODE2_LOG" "$NODE1_API_PORT" "$NODE2_API_PORT" "$ENTRY_MODE" <<'PY'
import json
import re
import sys
from pathlib import Path

output_path, entry_log, node1_log, node2_log, node1_port, node2_port, entry_mode = sys.argv[1:]
needle = "Inbound HTTP tunnel stream → llama-server"
local_pattern = re.compile(r"API proxy: routing to target Local\(")
remote_pattern = re.compile(r"API proxy: routing to target Remote\(")

def count_hits(path: str) -> int:
    return Path(path).read_text(encoding="utf-8").count(needle)

def count_routes(path: str) -> tuple[int, int]:
    local = 0
    remote = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if local_pattern.search(line):
            local += 1
        elif remote_pattern.search(line):
            remote += 1
    return local, remote

with open(output_path, encoding="utf-8") as fh:
    summary = json.load(fh)

if entry_mode == "active":
    node1_routes, node2_routes = count_routes(entry_log)
    route_counter_source = "entry_proxy"
else:
    node1_routes = count_hits(node1_log)
    node2_routes = count_hits(node2_log)
    route_counter_source = "host_accepts"

summary["host_route_counts"] = {
    f"node_{node1_port}": node1_routes,
    f"node_{node2_port}": node2_routes,
}
summary["host_route_total"] = node1_routes + node2_routes
summary["dominant_host"] = (
    f"node_{node1_port}" if node1_routes >= node2_routes else f"node_{node2_port}"
)
summary["route_counter_source"] = route_counter_source
summary["remote_tunnel_accepts"] = {
    f"node_{node1_port}": count_hits(node1_log),
    f"node_{node2_port}": count_hits(node2_log),
}

with open(output_path, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
PY
}

report_comparison() {
  local baseline_path="$1"
  local prefix_path="$2"
  python3 - "$baseline_path" "$prefix_path" <<'PY'
import json
import sys

baseline_path, prefix_path = sys.argv[1:]
with open(baseline_path, encoding="utf-8") as fh:
    baseline = json.load(fh)
with open(prefix_path, encoding="utf-8") as fh:
    prefix = json.load(fh)

def delta(after, before, key):
    return after.get(key, 0) - before.get(key, 0)

baseline_prompt = baseline["prompt_ms_mean"]
prefix_prompt = prefix["prompt_ms_mean"]
gain = 0.0
if baseline_prompt:
    gain = ((baseline_prompt - prefix_prompt) / baseline_prompt) * 100.0

print("")
print("Comparison")
print(f"  sticky-only prompt mean:  {baseline_prompt:.1f} ms")
print(f"  prefix-only prompt mean:  {prefix_prompt:.1f} ms")
print(f"  prompt-time reduction:    {gain:.1f}%")
print("")
print("Affinity counters")
print(
    "  sticky-only: "
    f"hits={delta(baseline['affinity_after'], baseline['affinity_before'], 'prefix_hits')} "
    f"misses={delta(baseline['affinity_after'], baseline['affinity_before'], 'prefix_misses')} "
    f"learned={delta(baseline['affinity_after'], baseline['affinity_before'], 'learned')}"
)
print(
    "  prefix-only: "
    f"hits={delta(prefix['affinity_after'], prefix['affinity_before'], 'prefix_hits')} "
    f"misses={delta(prefix['affinity_after'], prefix['affinity_before'], 'prefix_misses')} "
    f"learned={delta(prefix['affinity_after'], prefix['affinity_before'], 'learned')}"
)
print("")
print("Host route counts")
print(
    "  sticky-only: "
    + ", ".join(
        f"{host}={count}"
        for host, count in baseline.get("host_route_counts", {}).items()
    )
)
print(
    "  prefix-only: "
    + ", ".join(
        f"{host}={count}"
        for host, count in prefix.get("host_route_counts", {}).items()
    )
)
PY
}

require_file "$MESH_BIN" "mesh-llm binary"
require_file "$BIN_DIR/llama-server" "llama.cpp llama-server"
require_file "$MODEL_PATH" "benchmark model"

echo "Logs: $TMP_DIR"

start_phase "sticky-only" MESH_LLM_DISABLE_PREFIX_AFFINITY=1
run_workload "sticky-only" "$TMP_DIR/sticky-only.json"
annotate_route_distribution "$TMP_DIR/sticky-only.json"
stop_nodes

start_phase "prefix-only" MESH_LLM_PREFIX_ONLY=1
run_workload "prefix-only" "$TMP_DIR/prefix-only.json"
annotate_route_distribution "$TMP_DIR/prefix-only.json"
report_comparison "$TMP_DIR/sticky-only.json" "$TMP_DIR/prefix-only.json"
