#!/usr/bin/env bash
# ci-two-node-split-smoke.sh - verify real two-node split serving.
#
# Usage: scripts/ci-two-node-split-smoke.sh <mesh-llm-binary> <bin-dir> <model-path-or-ref>
#
# Unlike ci-two-node-client-serving-smoke.sh, both processes are serving nodes.
# The smoke requires the runtime to publish a topology with stages on at least
# two distinct nodes before it sends OpenAI requests through stage 0. For each
# model leg it sends every prompt length twice in a row (X, X, X+more, X+more,
# ...): the first sight of each length proves reuse keeps growing, and the
# identical re-send must restore more of the prompt from cache. Dense KV cache
# repeats must be near-full restores; recurrent KV cache repeats may restore
# from the latest checkpoint. With MESH_TWO_NODE_SPLIT_RECURRENT_MODEL set, the
# dense leg runs first and the recurrent leg repeats the whole flow against a
# second model in the same job.

set -euo pipefail

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}"
BIN_DIR="${2:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}"
MODEL="${MESH_TWO_NODE_SPLIT_MODEL:-${3:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path-or-ref>}}"

SEED_API_PORT="${MESH_TWO_NODE_SPLIT_SEED_API_PORT:-9367}"
SEED_CONSOLE_PORT="${MESH_TWO_NODE_SPLIT_SEED_CONSOLE_PORT:-3161}"
SEED_BIND_PORT="${MESH_TWO_NODE_SPLIT_SEED_BIND_PORT:-53647}"
WORKER_API_PORT="${MESH_TWO_NODE_SPLIT_WORKER_API_PORT:-9368}"
WORKER_CONSOLE_PORT="${MESH_TWO_NODE_SPLIT_WORKER_CONSOLE_PORT:-3162}"
WORKER_BIND_PORT="${MESH_TWO_NODE_SPLIT_WORKER_BIND_PORT:-53648}"
READINESS_TIMEOUT_SECONDS="${MESH_TWO_NODE_SPLIT_READINESS_TIMEOUT_SECONDS:-${MESH_TWO_NODE_SPLIT_MAX_WAIT:-300}}"
SNAPSHOT_REQUEST_TIMEOUT_SECONDS="${MESH_TWO_NODE_SPLIT_SNAPSHOT_REQUEST_TIMEOUT_SECONDS:-2}"
if [[ ! "$READINESS_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    [[ "$READINESS_TIMEOUT_SECONDS" -gt 300 ]]; then
    echo "MESH_TWO_NODE_SPLIT_READINESS_TIMEOUT_SECONDS must be an integer from 1 through 300" >&2
    exit 2
fi
if [[ ! "$SNAPSHOT_REQUEST_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] ||
    [[ "$SNAPSHOT_REQUEST_TIMEOUT_SECONDS" -gt 10 ]]; then
    echo "MESH_TWO_NODE_SPLIT_SNAPSHOT_REQUEST_TIMEOUT_SECONDS must be an integer from 1 through 10" >&2
    exit 2
fi
# Optional second model: when set, the smoke runs the whole split flow once
# against MODEL (dense) and once against this (recurrent), restarting both
# processes between legs so a single CI job covers both cache families.
RECURRENT_MODEL="${MESH_TWO_NODE_SPLIT_RECURRENT_MODEL:-}"
RECURRENT_MODEL_FILE="${MESH_TWO_NODE_SPLIT_RECURRENT_MODEL_FILE:-}"
RECURRENT_CTX_SIZE="${MESH_TWO_NODE_SPLIT_RECURRENT_CTX_SIZE:-4096}"
RECURRENT_EXPECTED_EXACT_PAYLOAD_KIND="${MESH_TWO_NODE_SPLIT_RECURRENT_EXPECTED_EXACT_PAYLOAD_KIND:-}"
if [[ -z "$RECURRENT_MODEL" && -n "$RECURRENT_MODEL_FILE" ]]; then
    RECURRENT_MODEL="${HOME}/.models/${RECURRENT_MODEL_FILE}"
fi
REQUEST_SETTLE_SECONDS="${MESH_TWO_NODE_SPLIT_REQUEST_SETTLE_SECONDS:-1}"
PREFIX_ATTEMPTS="${MESH_TWO_NODE_SPLIT_PREFIX_ATTEMPTS:-3}"
EXPECTED_EXACT_PAYLOAD_KIND="${MESH_TWO_NODE_SPLIT_EXPECTED_EXACT_PAYLOAD_KIND:-}"
CTX_SIZE="${MESH_TWO_NODE_SPLIT_CTX_SIZE:-${MESH_LLM_SMOKE_CONTEXT_SIZE:-}}"
MAX_VRAM="${MESH_TWO_NODE_SPLIT_MAX_VRAM:-1}"
DEVICE="${MESH_TWO_NODE_SPLIT_DEVICE:-CPU}"
WORK_DIR="${MESH_TWO_NODE_SPLIT_WORK_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/mesh-two-node-split.XXXXXX")}"
mkdir -p "$WORK_DIR"
# Keep this under /tmp with a short prefix because plugin Unix socket paths
# must fit platform SUN_LEN limits, especially on macOS where TMPDIR is long.
PROCESS_ROOT="${MESH_TWO_NODE_SPLIT_PROCESS_ROOT:-$(mktemp -d "/tmp/m2split.XXXXXX")}"
CLIENT_ROUTING="${MESH_TWO_NODE_SPLIT_CLIENT_ROUTING:-0}"
CLIENT_API_PORT="${MESH_TWO_NODE_SPLIT_CLIENT_API_PORT:-9369}"
CLIENT_CONSOLE_PORT="${MESH_TWO_NODE_SPLIT_CLIENT_CONSOLE_PORT:-3163}"
PRIMARY_MODEL_LABEL="${MESH_TWO_NODE_SPLIT_MODEL_LABEL:-}"
if [[ -z "$PRIMARY_MODEL_LABEL" ]]; then
    if [[ "$EXPECTED_EXACT_PAYLOAD_KIND" == "kv-recurrent" ]]; then
        PRIMARY_MODEL_LABEL="recurrent"
    else
        PRIMARY_MODEL_LABEL="dense"
    fi
fi
SEED_LOG="${WORK_DIR}/${PRIMARY_MODEL_LABEL}-seed.log"
WORKER_LOG="${WORK_DIR}/${PRIMARY_MODEL_LABEL}-worker.log"
CLIENT_LOG="${WORK_DIR}/${PRIMARY_MODEL_LABEL}-client.log"
MODEL_LABEL="$PRIMARY_MODEL_LABEL"
SPLIT_EVIDENCE_PATH=""
SPLIT_SNAPSHOT_DIR=""
SPLIT_RECONCILE_LOG=""

echo "=== CI Two-Node Split Smoke ==="
echo "  mesh-llm:       $MESH_LLM"
echo "  bin-dir:        $BIN_DIR (compatibility placeholder)"
echo "  model:          $MODEL"
echo "  seed api:       $SEED_API_PORT"
echo "  seed console:   $SEED_CONSOLE_PORT"
echo "  seed bind:      $SEED_BIND_PORT"
echo "  worker api:     $WORKER_API_PORT"
echo "  worker console: $WORKER_CONSOLE_PORT"
echo "  worker bind:    $WORKER_BIND_PORT"
echo "  readiness timeout: ${READINESS_TIMEOUT_SECONDS}s"
echo "  snapshot request timeout: ${SNAPSHOT_REQUEST_TIMEOUT_SECONDS}s"
echo "  request settle: ${REQUEST_SETTLE_SECONDS}s"
echo "  prefix attempts: ${PREFIX_ATTEMPTS}"
echo "  expected exact payload: ${EXPECTED_EXACT_PAYLOAD_KIND:-none}"
echo "  ctx size:       ${CTX_SIZE:-model default}"
echo "  max vram:       ${MAX_VRAM}GB"
echo "  device:         $DEVICE"
echo "  client routing: $CLIENT_ROUTING"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "Missing executable mesh-llm binary: $MESH_LLM" >&2
    exit 1
fi

RUNTIME_BUNDLE="${MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR:-$(cd "$(dirname "$MESH_LLM")" && pwd)/native-runtimes}"
if [[ ! -d "$RUNTIME_BUNDLE" ]]; then
    echo "Missing packaged native runtime beside mesh-llm: $RUNTIME_BUNDLE" >&2
    exit 1
fi
export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$RUNTIME_BUNDLE"

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    else
        shasum -a 256 "$path" | awk '{print $1}'
    fi
}

quant_selector_from_gguf_file() {
    local filename="$1"
    python3 - "$filename" <<'PY'
import re
import sys

stem = re.sub(r"-\d{5}-of-\d{5}$", "", sys.argv[1].removesuffix(".gguf"), flags=re.IGNORECASE)
matches = list(re.finditer(
    r"(?:^|[-_.])((?:IQ|Q)[1-8](?:_[0-9A-Z]+)+|F(?:16|32)|BF16)(?=$|[-_.])",
    stem,
    flags=re.IGNORECASE,
))
if not matches:
    raise SystemExit(f"cannot derive quant selector from GGUF filename: {sys.argv[1]}")
print(matches[-1].group(1))
PY
}

resolve_package_tool() {
    python3 - "$RUNTIME_BUNDLE" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

runtime_bundle = Path(sys.argv[1])
if not runtime_bundle.is_dir():
    raise SystemExit(f"native runtime bundle is not a directory: {runtime_bundle}")
bundle_root = runtime_bundle.resolve()

manifests = sorted(runtime_bundle.glob("*/manifest.json"))
if len(manifests) != 1:
    raise SystemExit(
        "expected exactly one native runtime manifest under "
        f"{runtime_bundle}; found {len(manifests)}"
    )

manifest_path = manifests[0]
runtime_dir = manifest_path.parent.resolve()
if runtime_dir.parent != bundle_root:
    raise SystemExit(f"native runtime manifest is outside its bundle: {manifest_path}")
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"cannot read native runtime manifest {manifest_path}: {error}")

runtime = manifest.get("runtime")
if not isinstance(runtime, dict):
    raise SystemExit(f"native runtime manifest has no runtime object: {manifest_path}")
tools = runtime.get("tools")
if not isinstance(tools, dict):
    raise SystemExit(f"native runtime manifest runtime.tools must be a checksum map: {manifest_path}")

def validate_safe_relative_path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in PurePosixPath(value).parts)
    ):
        raise SystemExit(f"declared runtime tool path is not safe: {value!r}")

for declared_path in tools:
    validate_safe_relative_path(declared_path)

# This is deliberately an exact manifest key. Looking up by basename could
# select an unrelated executable from a different declared path.
tool_rel = "tools/skippy-model-package"
if tool_rel not in tools:
    raise SystemExit(
        f"native runtime manifest does not declare {tool_rel} in runtime.tools: {manifest_path}"
    )
if not isinstance(tools[tool_rel], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", tools[tool_rel]):
    raise SystemExit(f"invalid SHA-256 for declared runtime tool {tool_rel}")

validate_safe_relative_path(tool_rel)

tool_path = (runtime_dir / tool_rel).resolve()
try:
    tool_path.relative_to(runtime_dir)
except ValueError:
    raise SystemExit(f"declared runtime tool escapes its bundle: {tool_rel}")
if not tool_path.is_file():
    raise SystemExit(f"declared runtime tool is not a file: {tool_rel}")
if not os.access(tool_path, os.X_OK):
    raise SystemExit(f"declared runtime tool is not executable: {tool_rel}")

digest = hashlib.sha256()
with tool_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != tools[tool_rel].lower():
    raise SystemExit(f"declared runtime tool checksum mismatch: {tool_rel}")

print(tool_path)
PY
}

prepare_split_package() {
    local label="$1"
    local source="$2"
    local package_tool="$3"

    if [[ -d "$source" && -s "$source/model-package.json" ]]; then
        printf '%s\n' "$source"
        return 0
    fi
    [[ -f "$source" ]] || {
        echo "Generation-8 split smoke input must be a package-v2 directory or local GGUF: $source" >&2
        return 1
    }

    local source_file selector digest package_dir model_id
    source_file="$(basename "$source")"
    selector="$(quant_selector_from_gguf_file "$source_file")"
    digest="$(sha256_file "$source")"
    package_dir="$WORK_DIR/prepared-packages/$label"
    model_id="ci/${label}-$(printf '%s' "$digest" | cut -c1-16):${selector}"
    rm -rf "$package_dir"
    mkdir -p "$(dirname "$package_dir")"

    local write_log="$WORK_DIR/${label}-write-package.log"
    local verify_log="$WORK_DIR/${label}-verify-package-v2.log"
    if ! "$package_tool" write-package "$source" \
        --model-id "$model_id" \
        --out-dir "$package_dir" \
        --source-repo "ci/two-node-split-smoke" \
        --source-revision "$digest" \
        --source-file "$source_file" >"$write_log" 2>&1; then
        echo "Package-v2 preparation failed for $source; see $write_log" >&2
        return 1
    fi
    if ! "$package_tool" verify-package-v2 "$package_dir" \
        --source "$source" \
        --source-file "$source_file" >"$verify_log" 2>&1; then
        echo "Package-v2 verification failed for $source; see $verify_log" >&2
        return 1
    fi
    [[ -s "$package_dir/model-package.json" ]] || {
        echo "Package-v2 preparation did not emit $package_dir/model-package.json" >&2
        return 1
    }
    printf '%s\n' "$package_dir"
}

prepare_split_inputs() {
    local package_tool
    if [[ -d "$MODEL" && -s "$MODEL/model-package.json" ]] &&
        { [[ -z "$RECURRENT_MODEL" ]] || [[ -d "$RECURRENT_MODEL" && -s "$RECURRENT_MODEL/model-package.json" ]]; }; then
        return 0
    fi
    package_tool="$(resolve_package_tool)"
    MODEL="$(prepare_split_package "$PRIMARY_MODEL_LABEL" "$MODEL" "$package_tool")"
    if [[ -n "$RECURRENT_MODEL" ]]; then
        RECURRENT_MODEL="$(prepare_split_package recurrent "$RECURRENT_MODEL" "$package_tool")"
    fi
}

descendant_pids() {
    local pid="$1"
    local children
    children="$(pgrep -P "$pid" 2>/dev/null || true)"
    for child in $children; do
        descendant_pids "$child"
        printf '%s\n' "$child"
    done
}

kill_tree() {
    local pid="${1:-}"
    [[ -n "$pid" ]] || return 0
    local children
    children="$(descendant_pids "$pid" | sort -u || true)"
    kill "$pid" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill 2>/dev/null || true
    fi
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    if [[ -n "$children" ]]; then
        printf '%s\n' "$children" | xargs kill -9 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}

SEED_PID=""
WORKER_PID=""
CLIENT_PID=""
cleanup() {
    kill_tree "$CLIENT_PID"
    kill_tree "$WORKER_PID"
    kill_tree "$SEED_PID"
    echo "--- seed log tail ---"
    tail -160 "$SEED_LOG" 2>/dev/null || true
    echo "--- worker log tail ---"
    tail -160 "$WORKER_LOG" 2>/dev/null || true
    echo "--- client log tail ---"
    tail -160 "$CLIENT_LOG" 2>/dev/null || true
    echo "--- end logs ---"
    if [[ -z "${MESH_TWO_NODE_SPLIT_WORK_DIR:-}" ]]; then
        rm -rf "$WORK_DIR"
    fi
    if [[ -z "${MESH_TWO_NODE_SPLIT_PROCESS_ROOT:-}" ]]; then
        rm -rf "$PROCESS_ROOT"
    fi
}
trap cleanup EXIT

prepare_split_inputs

status_json() {
    local console_port="$1"
    local request_timeout="${2:-$SNAPSHOT_REQUEST_TIMEOUT_SECONDS}"
    curl -fsS --connect-timeout "$request_timeout" \
        --max-time "$request_timeout" \
        "http://127.0.0.1:${console_port}/api/status" 2>/dev/null || true
}

query_token() {
    STATUS_JSON="$1" python3 - <<'PY'
import json
import os

try:
    status = json.loads(os.environ.get("STATUS_JSON", "") or "{}")
except Exception:
    status = {}
print(status.get("token") or "")
PY
}

wait_for_seed_token() {
    local context="$1"
    local started_at
    local deadline
    local now
    local remaining
    local request_timeout

    TOKEN=""
    started_at="$(date +%s)"
    deadline=$((started_at + READINESS_TIMEOUT_SECONDS))
    while :; do
        if ! kill -0 "$SEED_PID" 2>/dev/null; then
            echo "${context}seed exited unexpectedly" >&2
            tail -160 "$SEED_LOG" >&2 || true
            return 1
        fi
        now="$(date +%s)"
        remaining=$((deadline - now))
        if [[ "$remaining" -le 0 ]]; then
            echo "${context}timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for seed invite token" >&2
            tail -160 "$SEED_LOG" >&2 || true
            return 1
        fi
        request_timeout="$SNAPSHOT_REQUEST_TIMEOUT_SECONDS"
        if [[ "$remaining" -lt "$request_timeout" ]]; then
            request_timeout="$remaining"
        fi
        TOKEN="$(query_token "$(status_json "$SEED_CONSOLE_PORT" "$request_timeout")")"
        now="$(date +%s)"
        if [[ "$now" -ge "$deadline" ]]; then
            echo "${context}timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for seed invite token" >&2
            tail -160 "$SEED_LOG" >&2 || true
            return 1
        fi
        if [[ -n "$TOKEN" ]]; then
            echo "Seed produced invite token after $((now - started_at))s${context:+ (${context%: })}"
            return 0
        fi
        sleep 1
    done
}

configure_split_evidence_paths() {
    local label="$1"
    local prefix=""
    if [[ "$label" != "$PRIMARY_MODEL_LABEL" ]]; then
        prefix="${label}-"
    fi
    SPLIT_EVIDENCE_PATH="${WORK_DIR}/${prefix}split-evidence.json"
    SPLIT_SNAPSHOT_DIR="${WORK_DIR}/${prefix}split-evidence-snapshots"
    SPLIT_RECONCILE_LOG="${WORK_DIR}/${prefix}split-evidence-reconcile.log"
    mkdir -p "$SPLIT_SNAPSHOT_DIR"
}

capture_json_snapshot() {
    local kind="$1"
    local url="$2"
    local output="$3"
    local request_timeout="$4"
    local raw="${output}.curl.$$.tmp"

    if ! curl -fsS --connect-timeout "$request_timeout" \
        --max-time "$request_timeout" "$url" >"$raw" 2>/dev/null; then
        : >"$raw"
    fi
    python3 - "$kind" "$raw" "$output" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

kind, raw_path, output_path = sys.argv[1:]
raw = Path(raw_path).read_bytes()
try:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("response root is not an object")
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    payload = {
        "capture_error": str(error),
        "response_bytes": len(raw),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
    }
if kind == "status" and "capture_error" not in payload:
    peers = payload.get("peers")
    if not isinstance(peers, list):
        peers = []
    payload = {
        "mesh_id": payload.get("mesh_id"),
        "node_id": payload.get("node_id"),
        "peers": [
            {"id": peer.get("id")}
            for peer in peers
            if isinstance(peer, dict) and isinstance(peer.get("id"), str)
        ],
    }
temporary_path = f"{output_path}.tmp"
with open(temporary_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary_path, output_path)
PY
    rm -f "$raw"
}

capture_split_snapshots() {
    local request_timeout="$1"
    local -a capture_pids=()
    local capture_pid

    capture_json_snapshot status \
        "http://127.0.0.1:${SEED_CONSOLE_PORT}/api/status" \
        "$SPLIT_SNAPSHOT_DIR/seed-status.json" "$request_timeout" &
    capture_pids+=("$!")
    capture_json_snapshot stages \
        "http://127.0.0.1:${SEED_CONSOLE_PORT}/api/runtime/stages" \
        "$SPLIT_SNAPSHOT_DIR/seed-stages.json" "$request_timeout" &
    capture_pids+=("$!")
    capture_json_snapshot models \
        "http://127.0.0.1:${SEED_API_PORT}/v1/models" \
        "$SPLIT_SNAPSHOT_DIR/seed-models.json" "$request_timeout" &
    capture_pids+=("$!")
    capture_json_snapshot status \
        "http://127.0.0.1:${WORKER_CONSOLE_PORT}/api/status" \
        "$SPLIT_SNAPSHOT_DIR/worker-status.json" "$request_timeout" &
    capture_pids+=("$!")
    capture_json_snapshot stages \
        "http://127.0.0.1:${WORKER_CONSOLE_PORT}/api/runtime/stages" \
        "$SPLIT_SNAPSHOT_DIR/worker-stages.json" "$request_timeout" &
    capture_pids+=("$!")
    capture_json_snapshot models \
        "http://127.0.0.1:${WORKER_API_PORT}/v1/models" \
        "$SPLIT_SNAPSHOT_DIR/worker-models.json" "$request_timeout" &
    capture_pids+=("$!")

    for capture_pid in "${capture_pids[@]}"; do
        wait "$capture_pid"
    done
}

reconcile_split_snapshots() {
    python3 scripts/reconcile-two-node-split-evidence.py \
        --seed-status "$SPLIT_SNAPSHOT_DIR/seed-status.json" \
        --seed-stages "$SPLIT_SNAPSHOT_DIR/seed-stages.json" \
        --seed-models "$SPLIT_SNAPSHOT_DIR/seed-models.json" \
        --worker-status "$SPLIT_SNAPSHOT_DIR/worker-status.json" \
        --worker-stages "$SPLIT_SNAPSHOT_DIR/worker-stages.json" \
        --worker-models "$SPLIT_SNAPSHOT_DIR/worker-models.json" \
        --model-label "$MODEL_LABEL" \
        --output "$SPLIT_EVIDENCE_PATH"
}

report_split_readiness_failure() {
    local context="$1"
    local reason="$2"
    echo "${context}${reason}" >&2
    for snapshot in \
        "$SPLIT_SNAPSHOT_DIR/seed-status.json" \
        "$SPLIT_SNAPSHOT_DIR/seed-stages.json" \
        "$SPLIT_SNAPSHOT_DIR/seed-models.json" \
        "$SPLIT_SNAPSHOT_DIR/worker-status.json" \
        "$SPLIT_SNAPSHOT_DIR/worker-stages.json" \
        "$SPLIT_SNAPSHOT_DIR/worker-models.json" \
        "$SPLIT_EVIDENCE_PATH"; do
        echo "--- ${snapshot} at timeout ---" >&2
        cat "$snapshot" >&2 2>/dev/null || true
    done
    echo "--- split evidence reconciler error at timeout ---" >&2
    cat "$SPLIT_RECONCILE_LOG" >&2 2>/dev/null || true
    echo "--- seed log tail at timeout ---" >&2
    tail -160 "$SEED_LOG" >&2 || true
    echo "--- worker log tail at timeout ---" >&2
    tail -160 "$WORKER_LOG" >&2 || true
}

wait_for_split_topology() {
    local context="$1"
    local started_at
    local deadline
    local now
    local remaining
    local request_timeout
    local reconciliation_ready
    configure_split_evidence_paths "$MODEL_LABEL"
    started_at="$(date +%s)"
    deadline=$((started_at + READINESS_TIMEOUT_SECONDS))
    while :; do
        if ! kill -0 "$SEED_PID" 2>/dev/null; then
            capture_split_snapshots 1
            reconcile_split_snapshots 2>"$SPLIT_RECONCILE_LOG" || true
            report_split_readiness_failure "$context" \
                "seed exited unexpectedly before split readiness"
            return 1
        fi
        if ! kill -0 "$WORKER_PID" 2>/dev/null; then
            capture_split_snapshots 1
            reconcile_split_snapshots 2>"$SPLIT_RECONCILE_LOG" || true
            report_split_readiness_failure "$context" \
                "worker exited unexpectedly before split readiness"
            return 1
        fi

        now="$(date +%s)"
        remaining=$((deadline - now))
        if [[ "$remaining" -le 0 ]]; then
            report_split_readiness_failure "$context" \
                "timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for reconciled real split topology"
            return 1
        fi
        request_timeout="$SNAPSHOT_REQUEST_TIMEOUT_SECONDS"
        if [[ "$remaining" -lt "$request_timeout" ]]; then
            request_timeout="$remaining"
        fi
        capture_split_snapshots "$request_timeout"
        reconciliation_ready=0
        if READY_SUMMARY="$(reconcile_split_snapshots 2>"$SPLIT_RECONCILE_LOG")"; then
            reconciliation_ready=1
        fi
        now="$(date +%s)"
        if [[ "$now" -ge "$deadline" ]]; then
            report_split_readiness_failure "$context" \
                "timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for reconciled real split topology"
            return 1
        fi
        if [[ "$reconciliation_ready" -eq 1 ]]; then
            DRIVER_LABEL="$(python3 - "$SPLIT_EVIDENCE_PATH" <<'PY'
import json
import sys

evidence = json.load(open(sys.argv[1], encoding="utf-8"))
stage0 = next(
    (stage for stage in evidence["topology"]["stages"] if stage["stage_index"] == 0),
    None,
)
if stage0 is None:
    raise SystemExit("split evidence has no stage 0")

stage0_node = stage0["node_id"]
matches = [
    label
    for label, observer in evidence["observers"].items()
    if isinstance(observer, dict)
    and stage0_node.startswith(observer.get("node_id", "missing-observer-node"))
]
if len(matches) != 1:
    raise SystemExit(
        f"cannot map stage-0 node {stage0_node!r} to exactly one observer: {matches!r}"
    )
print(matches[0])
PY
)"
            case "$DRIVER_LABEL" in
                seed) DRIVER_API_PORT="$SEED_API_PORT" ;;
                worker) DRIVER_API_PORT="$WORKER_API_PORT" ;;
                *)
                    echo "split evidence selected unknown stage-0 driver: $DRIVER_LABEL" >&2
                    return 1
                    ;;
            esac
            echo "Split topology ready after $((now - started_at))s (${MODEL_LABEL}): ${READY_SUMMARY}"
            echo "Selected ${DRIVER_LABEL} as stage-0 OpenAI driver"
            return 0
        fi
        sleep 1
    done
}

start_node() {
    local label="$1"
    local join_token="$2"
    local api_port="$3"
    local console_port="$4"
    local bind_port="$5"
    local log_file="$6"
    local home="${PROCESS_ROOT}/${label}/h"
    local runtime="${PROCESS_ROOT}/${label}/r"
    mkdir -p "$home" "$runtime"

    local -a args=(
        --log-format json
        serve
        --model "$MODEL"
        --split
        --no-draft
        --device "$DEVICE"
        --max-vram "$MAX_VRAM"
        --port "$api_port"
        --console "$console_port"
        --bind-port "$bind_port"
        --headless
    )
    if [[ -n "$join_token" ]]; then
        args+=(--join "$join_token")
    fi
    if [[ -n "$CTX_SIZE" ]]; then
        args+=(--ctx-size "$CTX_SIZE")
    fi

    HOME="$home" \
        MESH_LLM_RUNTIME_ROOT="$runtime" \
        MESH_LLM_EPHEMERAL_KEY=1 \
        SKIPPY_TELEMETRY_STDERR=1 \
        "$MESH_LLM" "${args[@]}" >"$log_file" 2>&1 &
    printf '%s\n' "$!"
}

run_client_routing_probe() {
    [[ "$CLIENT_ROUTING" == "1" ]] || return 0
    echo "Starting passive client against dense split topology"
    local client_home="${PROCESS_ROOT}/client/h"
    local client_runtime="${PROCESS_ROOT}/client/r"
    mkdir -p "$client_home" "$client_runtime"
    HOME="$client_home" \
        MESH_LLM_RUNTIME_ROOT="$client_runtime" \
        MESH_LLM_EPHEMERAL_KEY=1 \
        "$MESH_LLM" --log-format json client --join "$TOKEN" \
        --port "$CLIENT_API_PORT" --console "$CLIENT_CONSOLE_PORT" --headless \
        >"$CLIENT_LOG" 2>&1 &
    CLIENT_PID=$!

    local started_at
    local deadline
    local now
    local remaining
    local request_timeout
    started_at="$(date +%s)"
    deadline=$((started_at + READINESS_TIMEOUT_SECONDS))
    while :; do
        if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
            echo "passive client exited unexpectedly" >&2
            tail -160 "$CLIENT_LOG" >&2 || true
            exit 1
        fi
        now="$(date +%s)"
        remaining=$((deadline - now))
        if [[ "$remaining" -le 0 ]]; then
            echo "timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for passive client model routing" >&2
            tail -160 "$CLIENT_LOG" >&2 || true
            exit 1
        fi
        request_timeout="$SNAPSHOT_REQUEST_TIMEOUT_SECONDS"
        if [[ "$remaining" -lt "$request_timeout" ]]; then
            request_timeout="$remaining"
        fi
        client_models="$(curl -fsS --connect-timeout "$request_timeout" \
            --max-time "$request_timeout" \
            "http://127.0.0.1:${CLIENT_API_PORT}/v1/models" 2>/dev/null || true)"
        now="$(date +%s)"
        if [[ "$now" -ge "$deadline" ]]; then
            echo "timed out after ${READINESS_TIMEOUT_SECONDS}s waiting for passive client model routing" >&2
            tail -160 "$CLIENT_LOG" >&2 || true
            exit 1
        fi
        if CLIENT_MODELS_JSON="$client_models" MODEL_ID="$MODEL_ID" python3 - <<'PY' 2>/dev/null; then
import json
import os

models = json.loads(os.environ.get("CLIENT_MODELS_JSON", "") or "{}").get("data", [])
raise SystemExit(0 if any(item.get("id") == os.environ["MODEL_ID"] for item in models) else 1)
PY
            break
        fi
        sleep 1
    done

    local probe_root="${WORK_DIR}/client-routing"
    mkdir -p "$probe_root"
    python3 - "$MODEL_ID" "$probe_root/request.json" "$probe_root/stream.json" <<'PY'
import json
import sys

model, request_path, stream_path = sys.argv[1:4]
for path, stream in ((request_path, False), (stream_path, True)):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({
            "model": model,
            "messages": [{"role": "user", "content": "Say ok."}],
            "stream": stream,
            "max_tokens": 8,
            "temperature": 0,
        }, handle)
PY
    curl -fsS --max-time 120 "http://127.0.0.1:${CLIENT_API_PORT}/v1/chat/completions" \
        -H 'content-type: application/json' -d @"$probe_root/request.json" \
        -o "$probe_root/response.json"
    python3 - "$probe_root/response.json" <<'PY'
import json
import sys

body = json.load(open(sys.argv[1], encoding="utf-8"))
if body.get("object") != "chat.completion" or not body.get("choices"):
    raise SystemExit(f"invalid passive-client response: {body!r}")
PY
    curl -fsS --max-time 120 -N "http://127.0.0.1:${CLIENT_API_PORT}/v1/chat/completions" \
        -H 'content-type: application/json' -d @"$probe_root/stream.json" \
        -o "$probe_root/stream.txt"
    grep -q 'data: \[DONE\]' "$probe_root/stream.txt"
    echo "Passive client routing and streaming validated against dense split topology"
}

SEED_PID="$(start_node seed "" "$SEED_API_PORT" "$SEED_CONSOLE_PORT" "$SEED_BIND_PORT" "$SEED_LOG")"

wait_for_seed_token ""

WORKER_PID="$(start_node worker "$TOKEN" "$WORKER_API_PORT" "$WORKER_CONSOLE_PORT" "$WORKER_BIND_PORT" "$WORKER_LOG")"

DRIVER_LABEL=""
DRIVER_API_PORT=""
wait_for_split_topology ""

if [[ -z "$DRIVER_API_PORT" ]]; then
    echo "no split driver API port was selected" >&2
    exit 1
fi
MODEL_ID="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("model_id", ""))' \
        "$SPLIT_EVIDENCE_PATH"
)"
if [[ -z "$MODEL_ID" ]]; then
    echo "${DRIVER_LABEL:-selected driver} split evidence did not return a model id" >&2
    exit 1
fi
export MODEL_ID MODEL_LABEL

# Stage readiness is published before the serving target finishes registering
# on every observer. Give routing the same bounded settle interval used between
# inference requests so the first probe cannot race that final handoff.
sleep "$REQUEST_SETTLE_SECONDS"

run_client_routing_probe

PREFIX_PAYLOAD_ROOT="${WORK_DIR}/prefix-payloads"
PREFIX_RESPONSE_ROOT="${WORK_DIR}/prefix-responses"

# Exit code the prefix validator uses for the one failure this smoke cannot
# distinguish from a regression on a single pass: every request restored
# nothing, which is what an unreleased stage lane looks like from the outside.
# Anything else is a real assertion failure and is never retried.
PREFIX_TRANSIENT_STATUS=75

write_prefix_payloads() {
    python3 - "$MODEL_ID" "$1" "$2" <<'PY'
import json
from pathlib import Path
import sys

model, output_dir, nonce = sys.argv[1:4]
output = Path(output_dir)
# The nonce leads the prompt so every attempt starts from a genuinely cold
# prefix. Without it, a retry would run against the cache the previous attempt
# already warmed and the cold-request assertion below would stop meaning
# anything.
shared = f"Split prefix cache smoke shared context {nonce}. " + (
    "Every request keeps these tokens in the same order. " * 48
)
extensions = [
    "First extension block remains reusable by later prompts. " * 16,
    "Second extension block makes the reusable prefix longer. " * 16,
    "Third extension block proves reuse keeps growing. " * 16,
]
# Every length is sent twice in a row: X, X, X+E1, X+E1, X+E1+E2, X+E1+E2.
# The first sight of each length proves reuse carries the established prefix
# forward, and the identical re-send proves a repeated prompt is served from
# cache instead of being recomputed.
index = 0
prompt = shared
for extension in extensions:
    prompt += extension
    for _ in range(2):
        index += 1
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "user": f"ci-split-prefix-growth-{nonce}",
            "stream": False,
            "max_tokens": 1,
            "temperature": 0,
        }
        with (output / f"prompt-{index}.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
PY
}

# Requests 1..6: odd indexes are first sights of each prompt length (growth
# arms), even indexes are identical re-sends of the previous prompt (repeat
# arms). Requests 1-2 share prompt X, 3-4 share X+E1, 5-6 share X+E1+E2.
PREFIX_REQUEST_COUNT=6

validate_prefix_responses() {
    python3 - "$1" "$PREFIX_REQUEST_COUNT" "$EXPECTED_EXACT_PAYLOAD_KIND" <<'PY'
import json
from pathlib import Path
import sys

TRANSIENT_STATUS = 75
# A repeated prompt is allowed to re-feed the final token for logits, so the
# restore can legitimately report prompt_tokens - 1 rather than the full
# prompt. Two tokens of slack keeps that off the failure path.
REPEAT_TOKEN_SLACK = 2

response_dir = Path(sys.argv[1])
request_count = int(sys.argv[2])
exact_payload_kind = sys.argv[3]
checkpointed_restore = exact_payload_kind == "kv-recurrent"
growth_indexes = list(range(1, request_count + 1, 2))
repeat_pairs = [(index, index + 1) for index in range(1, request_count + 1, 2)]
metrics = []
for index in range(1, request_count + 1):
    with (response_dir / f"response-{index}.json").open(encoding="utf-8") as fh:
        body = json.load(fh)
    if body.get("object") != "chat.completion":
        raise SystemExit(
            f"prefix request {index} returned unexpected object: {body.get('object')!r}"
        )
    if not body.get("choices"):
        raise SystemExit(f"prefix request {index} returned no choices")
    usage = body.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = details.get("cached_tokens", 0)
    if not isinstance(prompt_tokens, int) or not isinstance(cached_tokens, int):
        raise SystemExit(f"prefix request {index} omitted numeric cache usage: {usage!r}")
    metrics.append((prompt_tokens, cached_tokens))

prompt_counts = [prompt for prompt, _ in metrics]
cached_counts = [cached for _, cached in metrics]
growth_prompts = [prompt_counts[index - 1] for index in growth_indexes]
growth_cached = [cached_counts[index - 1] for index in growth_indexes]
repeat_cached = [cached_counts[repeat - 1] for _, repeat in repeat_pairs]
if prompt_counts[0] != prompt_counts[1] or any(
    prompt_counts[pair[0] - 1] != prompt_counts[pair[1] - 1] for pair in repeat_pairs
):
    raise SystemExit(f"repeated prompts diverged between sends: {prompt_counts}")
if not growth_prompts[0] < growth_prompts[1] < growth_prompts[2]:
    raise SystemExit(f"prompt token counts did not increase: {prompt_counts}")
if cached_counts[0] != 0:
    raise SystemExit(f"cold prefix request unexpectedly restored tokens: {cached_counts}")
# A completely cold attempt can arise while a stage lane is still releasing.
# A partial follow-up miss is the regression under test and must fail directly,
# not be hidden by a retry that starts from another cold prefix.
if all(cached == 0 for cached in cached_counts[1:]):
    print(
        f"split prefix reuse was empty on a follow-up request: {cached_counts}",
        file=sys.stderr,
    )
    raise SystemExit(TRANSIENT_STATUS)
# Dense KV cache can reuse an established prefix while processing a longer
# first-sight prompt, so its growth arms must increase. Exact recurrent state
# is checkpoint-aligned and cannot resume at an arbitrary nonzero token offset;
# a longer first-sight prompt may therefore be cold. Its identical repeat still
# has to restore a progressively later checkpoint for each longer prompt.
reuse_growth = repeat_cached if checkpointed_restore else growth_cached
if not reuse_growth[0] < reuse_growth[1] < reuse_growth[2]:
    raise SystemExit(f"split prefix reuse did not increase: {cached_counts}")
# Only growth arms need an uncached suffix; a repeat arm may legitimately
# restore everything except the final re-fed token.
for index in growth_indexes:
    if cached_counts[index - 1] >= prompt_counts[index - 1]:
        raise SystemExit(
            f"growing prompts must retain an uncached suffix: {metrics}"
        )
# Each identical re-send must extend beyond the first-sight cached region.
# Dense KV cache can additionally restore nearly the entire prompt. Recurrent
# KV cache restores at checkpoint boundaries, so a valid repeat may leave a
# larger suffix uncached even when exact-state cache reuse is working.
for first, repeat in repeat_pairs:
    if not checkpointed_restore and (
        cached_counts[repeat - 1]
        < prompt_counts[repeat - 1] - REPEAT_TOKEN_SLACK
    ):
        raise SystemExit(
            "identical re-send was not served from cache: "
            f"request {repeat} restored {cached_counts[repeat - 1]} of "
            f"{prompt_counts[repeat - 1]} prompt tokens"
        )
    if cached_counts[repeat - 1] <= cached_counts[first - 1]:
        raise SystemExit(
            f"re-send {repeat} must extend beyond the first-sight cache of "
            f"request {first}: {metrics}"
        )

print(
    "Split prefix cache reuse grew and repeated prompts restored from cache: "
    + ", ".join(
        f"request {index}: prompt_tokens={prompt}, cached_tokens={cached}"
        for index, (prompt, cached) in enumerate(metrics, start=1)
    )
)
PY
}

assert_expected_stage_payload() {
    [[ -n "$EXPECTED_EXACT_PAYLOAD_KIND" ]] || return 0
    python3 - "$EXPECTED_EXACT_PAYLOAD_KIND" "$SEED_LOG" "$WORKER_LOG" <<'PY'
import json
import sys

expected, *logs = sys.argv[1:]
for log_path in logs:
    with open(log_path, encoding="utf-8", errors="replace") as log:
        for line in log:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            attributes = event.get("attributes") or {}
            exact_kind = attributes.get("skippy.exact_cache.payload_kind")
            dense_resident = (
                expected == "kv-dense"
                and attributes.get("skippy.kv.payload") == "ResidentKv"
            )
            if exact_kind == expected or dense_resident:
                print(f"observed stage-state payload kind: {expected}")
                raise SystemExit(0)
raise SystemExit(
    f"did not observe stage-state payload kind {expected!r} in split-stage telemetry"
)
PY
}

prefix_validated=0
for attempt in $(seq 1 "$PREFIX_ATTEMPTS"); do
    payload_dir="${PREFIX_PAYLOAD_ROOT}/attempt-${attempt}"
    response_dir="${PREFIX_RESPONSE_ROOT}/attempt-${attempt}"
    mkdir -p "$payload_dir" "$response_dir"
    write_prefix_payloads "$payload_dir" "attempt-${attempt}"

    for index in $(seq 1 "$PREFIX_REQUEST_COUNT"); do
        response_path="${response_dir}/response-${index}.json"
        if ! curl --fail-with-body -sS --max-time 180 \
            "http://127.0.0.1:${DRIVER_API_PORT}/v1/chat/completions" \
            -H 'content-type: application/json' \
            -d @"${payload_dir}/prompt-${index}.json" \
            -o "$response_path"; then
            echo "split inference request ${index} failed through ${DRIVER_LABEL} stage-0 driver" >&2
            cat "$response_path" >&2 2>/dev/null || true
            exit 1
        fi
        # The host returns the OpenAI response before the stage connection has
        # released its single CI lane. Give graceful Stop enough time to finish
        # so the next request tests cache reuse rather than transient admission.
        sleep "$REQUEST_SETTLE_SECONDS"
    done

    set +e
    validate_prefix_responses "$response_dir"
    prefix_status=$?
    set -e
    if [[ "$prefix_status" -eq 0 ]]; then
        prefix_validated=1
        break
    fi
    if [[ "$prefix_status" -ne "$PREFIX_TRANSIENT_STATUS" ]]; then
        exit "$prefix_status"
    fi
    echo "prefix attempt ${attempt} of ${PREFIX_ATTEMPTS} saw no reuse; retrying from a cold prefix" >&2
done

if [[ "$prefix_validated" -ne 1 ]]; then
    echo "split prefix reuse never materialized across ${PREFIX_ATTEMPTS} attempts" >&2
    exit 1
fi

assert_expected_stage_payload

echo "Two-node split smoke passed for model leg: ${MODEL_LABEL:-default}"

# Optional recurrent leg: rerun the identical flow against a second model in
# the same process pair so one CI job proves both cache families. The phase
# function restores MODEL/CTX_SIZE, restarts both nodes from scratch (fresh
# token, fresh stage split), and reruns the prefix cache assertions.
run_recurrent_leg() {
    echo "=== Two-node split smoke: recurrent leg ==="
    kill_tree "$CLIENT_PID"
    CLIENT_PID=""
    kill_tree "$WORKER_PID"
    kill_tree "$SEED_PID"
    SEED_LOG="${WORK_DIR}/recurrent-seed.log"
    WORKER_LOG="${WORK_DIR}/recurrent-worker.log"

    MODEL="$RECURRENT_MODEL"
    CTX_SIZE="$RECURRENT_CTX_SIZE"
    MODEL_LABEL="recurrent"
    EXPECTED_EXACT_PAYLOAD_KIND="$RECURRENT_EXPECTED_EXACT_PAYLOAD_KIND"
    export MODEL CTX_SIZE

    SEED_PID="$(start_node seed "" "$SEED_API_PORT" "$SEED_CONSOLE_PORT" "$SEED_BIND_PORT" "$SEED_LOG")"

    wait_for_seed_token "recurrent leg: "

    WORKER_PID="$(start_node worker "$TOKEN" "$WORKER_API_PORT" "$WORKER_CONSOLE_PORT" "$WORKER_BIND_PORT" "$WORKER_LOG")"

    DRIVER_LABEL=""
    DRIVER_API_PORT=""
    wait_for_split_topology "recurrent leg: "

    if [[ -z "$DRIVER_API_PORT" ]]; then
        echo "recurrent leg: no split driver API port was selected" >&2
        exit 1
    fi
    MODEL_ID="$(
        python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("model_id", ""))' \
            "$SPLIT_EVIDENCE_PATH"
    )"
    if [[ -z "$MODEL_ID" ]]; then
        echo "${DRIVER_LABEL:-selected driver} split evidence did not return a model id (recurrent leg)" >&2
        exit 1
    fi
}

if [[ -n "$RECURRENT_MODEL" ]]; then
    run_recurrent_leg

    PREFIX_PAYLOAD_ROOT="${WORK_DIR}/prefix-payloads-recurrent"
    PREFIX_RESPONSE_ROOT="${WORK_DIR}/prefix-responses-recurrent"

    prefix_validated=0
    for attempt in $(seq 1 "$PREFIX_ATTEMPTS"); do
        payload_dir="${PREFIX_PAYLOAD_ROOT}/attempt-${attempt}"
        response_dir="${PREFIX_RESPONSE_ROOT}/attempt-${attempt}"
        mkdir -p "$payload_dir" "$response_dir"
        write_prefix_payloads "$payload_dir" "recurrent-attempt-${attempt}"

        for index in $(seq 1 "$PREFIX_REQUEST_COUNT"); do
            curl -fsS --max-time 180 \
                "http://127.0.0.1:${DRIVER_API_PORT}/v1/chat/completions" \
                -H 'content-type: application/json' \
                -d @"${payload_dir}/prompt-${index}.json" \
                -o "${response_dir}/response-${index}.json"
            sleep "$REQUEST_SETTLE_SECONDS"
        done

        set +e
        validate_prefix_responses "$response_dir"
        prefix_status=$?
        set -e
        if [[ "$prefix_status" -eq 0 ]]; then
            prefix_validated=1
            break
        fi
        if [[ "$prefix_status" -ne "$PREFIX_TRANSIENT_STATUS" ]]; then
            exit "$prefix_status"
        fi
        echo "recurrent leg: prefix attempt ${attempt} of ${PREFIX_ATTEMPTS} saw no reuse; retrying from a cold prefix" >&2
    done

    if [[ "$prefix_validated" -ne 1 ]]; then
        echo "recurrent leg: split prefix reuse never materialized across ${PREFIX_ATTEMPTS} attempts" >&2
        exit 1
    fi

    assert_expected_stage_payload

    echo "Two-node split smoke passed for model leg: recurrent"
fi
