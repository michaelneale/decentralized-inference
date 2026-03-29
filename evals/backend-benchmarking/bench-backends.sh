#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MEASURE_PY="$REPO_DIR/evals/latency-benchmarking/measure.py"
DEFAULT_OUTPUT_DIR="$REPO_DIR/evals/results/backend-benchmarking/$(date +%Y%m%d-%H%M%S)"
DEFAULT_CASES_FILE="$REPO_DIR/evals/backend-benchmarking/default-cases.json"

MESH_BIN="$REPO_DIR/target/release/mesh-llm"
LLAMA_BIN_DIR="$REPO_DIR/llama.cpp/build/bin"
MLX_NATIVE_BIN="$REPO_DIR/target/release/mesh-llm-mlx"

PORT=9337
CONSOLE_PORT=3131
RUNS=3
WARMUP_RUNS=1
READY_TIMEOUT_S=240
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
CASES_FILE="$DEFAULT_CASES_FILE"

LLAMA_MODEL=""
MLX_MODEL=""
declare -a CUSTOM_CASES=()

usage() {
    cat <<EOF
Usage:
  $(basename "$0") --llama-model PATH --mlx-model PATH [options]

Options:
  --llama-model PATH        GGUF model path for the llama backend.
  --mlx-model PATH          MLX model directory for the mlx backend.
  --mesh-bin PATH           mesh-llm binary (default: $MESH_BIN)
  --llama-bin-dir PATH      llama.cpp bin dir with llama-server/rpc-server (default: $LLAMA_BIN_DIR)
  --mlx-native-bin PATH     mesh-llm-mlx binary (default: $MLX_NATIVE_BIN)
  --port N                  Inference port to benchmark (default: $PORT)
  --console-port N          Console/API port (default: $CONSOLE_PORT)
  --runs N                  Measured runs per case/backend (default: $RUNS)
  --warmup-runs N           Warmup runs per backend/case (default: $WARMUP_RUNS)
  --ready-timeout-s N       Seconds to wait for backend readiness (default: $READY_TIMEOUT_S)
  --output-dir PATH         Output directory (default: $DEFAULT_OUTPUT_DIR)
  --cases-file PATH         JSON file with benchmark cases (default: $DEFAULT_CASES_FILE)
  --case 'label|max|prompt' Add a simple user-message case.
  --case-json JSON          Add a full case object with label, max_tokens, and messages.
  --help                    Show this help.

Example:
  $(basename "$0") \\
    --llama-model ~/.models/Qwen3-0.6B-Q4_K_M.gguf \\
    --mlx-model ~/.models/mlx/Qwen3-0.6B-bf16 \\
    --runs 3
EOF
}

render_cases_tsv() {
    local cases_file="$1"
    shift || true

    python3 - "$cases_file" "$@" <<'PY'
import json
import sys

cases_path = sys.argv[1]
custom_cases = sys.argv[2:]

with open(cases_path, "r", encoding="utf-8") as fh:
    cases = json.load(fh)

for raw in custom_cases:
    if not raw:
        continue
    if raw.startswith("{"):
        case = json.loads(raw)
    else:
        label, max_tokens, prompt = raw.split("|", 2)
        case = {
            "label": label,
            "max_tokens": int(max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
    cases.append(case)

for case in cases:
    label = case["label"]
    max_tokens = int(case["max_tokens"])
    messages = case["messages"]
    preview_parts = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=True)
        preview_parts.append(f"{role}: {str(content).replace(chr(10), ' ')[:120]}")
    preview = " | ".join(preview_parts)
    print("\t".join([label, str(max_tokens), preview, json.dumps(messages, separators=(",", ":"))]))
PY
}

monotonic_ms() {
    python3 - <<'PY'
import time
print(int(time.monotonic() * 1000))
PY
}

json_get_first_model() {
    local port="$1"
    python3 - "$port" <<'PY'
import json
import sys
import urllib.request

port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/v1/models"
with urllib.request.urlopen(url, timeout=2) as resp:
    payload = json.load(resp)
models = payload.get("data") or []
if not models:
    raise SystemExit(1)
print(models[0]["id"])
PY
}

cleanup_processes() {
    pkill -x "mesh-llm-mlx" 2>/dev/null || true
    pkill -x "mesh-llm" 2>/dev/null || true
    pkill -x "llama-server" 2>/dev/null || true
    pkill -x "rpc-server" 2>/dev/null || true
    sleep 1
}

pid_is_alive() {
    local pid="$1"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -e "$path" ]]; then
        echo "Missing $label: $path" >&2
        exit 1
    fi
}

require_executable() {
    local path="$1"
    local label="$2"
    if [[ ! -x "$path" ]]; then
        echo "Missing executable $label: $path" >&2
        exit 1
    fi
}

wait_for_backend_ready() {
    local backend="$1"
    local log_path="$2"
    local pid="$3"
    local start_ms="$4"
    local deadline=$((start_ms + READY_TIMEOUT_S * 1000))

    while true; do
        local model_id=""
        if model_id="$(json_get_first_model "$PORT" 2>/dev/null)"; then
            local ready_ms
            ready_ms=$(monotonic_ms)
            local startup_ms=$((ready_ms - start_ms))
            printf '%s|%s\n' "$model_id" "$startup_ms"
            return 0
        fi

        if ! pid_is_alive "$pid"; then
            echo "Backend '$backend' exited before becoming ready" >&2
            echo "--- $backend log tail ---" >&2
            tail -n 80 "$log_path" >&2 || true
            return 1
        fi

        if [[ "$(monotonic_ms)" -ge "$deadline" ]]; then
            echo "Backend '$backend' failed to become ready within ${READY_TIMEOUT_S}s" >&2
            echo "--- $backend log tail ---" >&2
            tail -n 80 "$log_path" >&2 || true
            return 1
        fi
        sleep 0.5
    done
}

start_backend() {
    local backend="$1"
    local model_path="$2"
    local log_path="$3"
    local pid_path="$OUTPUT_DIR/${backend}.pid"

    local -a cmd=("$MESH_BIN" "--backend" "$backend" "--model" "$model_path" "--port" "$PORT" "--console" "$CONSOLE_PORT")
    if [[ "$backend" == "llama" ]]; then
        cmd+=("--bin-dir" "$LLAMA_BIN_DIR")
    else
        cmd+=("--mlx-native-bin" "$MLX_NATIVE_BIN")
    fi

    : >"$log_path"
    (
        cd "$REPO_DIR"
        nohup "${cmd[@]}" >>"$log_path" 2>&1 &
        echo $! >"$pid_path"
    )
}

read_backend_pid() {
    local backend="$1"
    local pid_path="$OUTPUT_DIR/${backend}.pid"
    [[ -f "$pid_path" ]] || return 1
    tr -d '[:space:]' <"$pid_path"
}

run_measure() {
    local model_id="$1"
    local messages_json="$2"
    local max_tokens="$3"
    python3 "$MEASURE_PY" \
        --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
        --model "$model_id" \
        --messages-json "$messages_json" \
        --max-tokens "$max_tokens"
}

append_result() {
    local results_file="$1"
    local backend="$2"
    local backend_model_path="$3"
    local served_model_id="$4"
    local case_label="$5"
    local input_preview="$6"
    local max_tokens="$7"
    local run_index="$8"
    local startup_ms="$9"
    local measured_json="${10}"
    local messages_json="${11}"

    python3 - "$results_file" "$backend" "$backend_model_path" "$served_model_id" "$case_label" "$input_preview" "$max_tokens" "$run_index" "$startup_ms" "$measured_json" "$messages_json" <<'PY'
import json
import pathlib
import sys

results_file = pathlib.Path(sys.argv[1])
record = json.loads(sys.argv[10])
record.update(
    {
        "backend": sys.argv[2],
        "backend_model_path": sys.argv[3],
        "served_model_id": sys.argv[4],
        "case": sys.argv[5],
        "input_preview": sys.argv[6],
        "max_tokens_requested": int(sys.argv[7]),
        "run": int(sys.argv[8]),
        "startup_ms": int(sys.argv[9]),
        "messages": json.loads(sys.argv[11]),
    }
)
with results_file.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(record, sort_keys=True) + "\n")
PY
}

write_summary() {
    local results_file="$1"
    local summary_file="$2"

    python3 - "$results_file" "$summary_file" <<'PY'
import json
import pathlib
import statistics
import sys
from collections import defaultdict

results_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]

def median(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(statistics.median(vals), 2)

summary = []
groups = defaultdict(list)
for row in rows:
    groups[(row["case"], row["backend"])].append(row)

for (case, backend), items in sorted(groups.items()):
    summary.append(
        {
            "case": case,
            "backend": backend,
            "runs": len(items),
            "served_model_id": items[0]["served_model_id"],
            "startup_ms_median": median([item["startup_ms"] for item in items]),
            "ttft_ms_median": median([item["ttft_ms"] for item in items]),
            "total_ms_median": median([item["total_ms"] for item in items]),
            "tok_s_median": median([item["tok_s"] for item in items]),
            "tokens_median": median([item["tokens"] for item in items]),
        }
    )

summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print("")
print("Summary")
print("-------")
print(f"{'Case':18} {'Backend':8} {'Startup':>10} {'TTFT':>10} {'Total':>10} {'tok/s':>10} {'Model':>18}")
for row in summary:
    startup = f"{row['startup_ms_median']} ms" if row["startup_ms_median"] is not None else "n/a"
    ttft = f"{row['ttft_ms_median']} ms" if row["ttft_ms_median"] is not None else "n/a"
    total = f"{row['total_ms_median']} ms" if row["total_ms_median"] is not None else "n/a"
    tps = f"{row['tok_s_median']}" if row["tok_s_median"] is not None else "n/a"
    print(f"{row['case'][:18]:18} {row['backend']:8} {startup:>10} {ttft:>10} {total:>10} {tps:>10} {row['served_model_id'][:18]:>18}")
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llama-model)
            LLAMA_MODEL="$2"
            shift 2
            ;;
        --mlx-model)
            MLX_MODEL="$2"
            shift 2
            ;;
        --mesh-bin)
            MESH_BIN="$2"
            shift 2
            ;;
        --llama-bin-dir)
            LLAMA_BIN_DIR="$2"
            shift 2
            ;;
        --mlx-native-bin)
            MLX_NATIVE_BIN="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --console-port)
            CONSOLE_PORT="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --warmup-runs)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --ready-timeout-s)
            READY_TIMEOUT_S="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cases-file)
            CASES_FILE="$2"
            shift 2
            ;;
        --case)
            CUSTOM_CASES+=("$2")
            shift 2
            ;;
        --case-json)
            CUSTOM_CASES+=("$2")
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -n "$LLAMA_MODEL" ]] && [[ -n "$MLX_MODEL" ]]; then
    :
else
    echo "--llama-model and --mlx-model are required." >&2
    usage
    exit 1
fi

require_executable "$MESH_BIN" "mesh-llm"
require_file "$MEASURE_PY" "measure.py"
require_file "$LLAMA_MODEL" "llama model"
require_file "$MLX_MODEL" "mlx model"
require_file "$CASES_FILE" "cases file"
require_executable "$MLX_NATIVE_BIN" "mesh-llm-mlx"
require_file "$LLAMA_BIN_DIR/llama-server" "llama-server"
require_file "$LLAMA_BIN_DIR/rpc-server" "rpc-server"

mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="$OUTPUT_DIR/results.jsonl"
SUMMARY_FILE="$OUTPUT_DIR/summary.json"

cleanup_processes
trap cleanup_processes EXIT

echo "Output dir: $OUTPUT_DIR"
echo ""

for backend in llama mlx; do
    if [[ "$backend" == "llama" ]]; then
        backend_model="$LLAMA_MODEL"
    else
        backend_model="$MLX_MODEL"
    fi

    echo "=== $backend ==="
    cleanup_processes

    log_path="$OUTPUT_DIR/${backend}.log"
    start_ms="$(monotonic_ms)"
    start_backend "$backend" "$backend_model" "$log_path"
    backend_pid="$(read_backend_pid "$backend")"
    ready_info="$(wait_for_backend_ready "$backend" "$log_path" "$backend_pid" "$start_ms")"
    model_id="${ready_info%%|*}"
    startup_ms="${ready_info##*|}"

    echo "ready: model=$model_id startup=${startup_ms}ms"

    while IFS=$'\t' read -r case_label max_tokens input_preview messages_json; do
        [[ -n "$case_label" ]] || continue
        echo "  case=$case_label warmup=$WARMUP_RUNS measured=$RUNS max_tokens=$max_tokens"
        echo "    context=${input_preview:0:140}"

        for _ in $(seq 1 "$WARMUP_RUNS"); do
            run_measure "$model_id" "$messages_json" "$max_tokens" >/dev/null
        done

        for run_index in $(seq 1 "$RUNS"); do
            measured_json="$(run_measure "$model_id" "$messages_json" "$max_tokens")"
            append_result "$RESULTS_FILE" "$backend" "$backend_model" "$model_id" "$case_label" "$input_preview" "$max_tokens" "$run_index" "$startup_ms" "$measured_json" "$messages_json"
            echo "    run=$run_index $measured_json"
        done
    done < <(render_cases_tsv "$CASES_FILE" "${CUSTOM_CASES[@]-}")
done

write_summary "$RESULTS_FILE" "$SUMMARY_FILE"
echo ""
echo "Raw results: $RESULTS_FILE"
echo "Summary JSON: $SUMMARY_FILE"
