#!/usr/bin/env bash
# ci-compat-smoke.sh - run OpenAI client compatibility probes against mesh-llm.
#
# Usage: scripts/ci-compat-smoke.sh <mesh-llm-binary> <bin-dir> <model-path>

set -euo pipefail

# Bounded hardware selection; CUDA can never silently opt out of offload checks.
SMOKE_DEVICE="${MESH_COMPAT_DEVICE:-${MESH_CI_DEVICE:-CPU}}"
case "$SMOKE_DEVICE" in
    CPU|CUDA0|Vulkan0|ROCm0|MTL0) ;;
    *) echo "Unsupported MESH_CI_DEVICE: $SMOKE_DEVICE (expected CPU, CUDA0, Vulkan0, ROCm0 or MTL0)" >&2; exit 1 ;;
esac

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
BIN_DIR="${2:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
MODEL="${3:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path>}"
API_PORT="${MESH_COMPAT_API_PORT:-9348}"
CONSOLE_PORT="${MESH_COMPAT_CONSOLE_PORT:-3142}"
MAX_WAIT="${MESH_COMPAT_MAX_WAIT:-180}"
LOG="${MESH_COMPAT_LOG:-/tmp/mesh-llm-compat-ci.log}"
ATTESTATION_PUBLIC_KEY_FILE="${MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE:-}"
ATTESTATION_EXPECTED_STATUS="${MESH_RELEASE_ATTESTATION_EXPECTED_STATUS:-valid}"
SMOKE_STATE_DIR="$(mktemp -d /tmp/mesh-llm-compat-state.XXXXXX)"
SMOKE_CONFIG_PATH="$SMOKE_STATE_DIR/config.toml"
SMOKE_RUNTIME_ROOT="$SMOKE_STATE_DIR/runtime"
trap 'rm -rf -- "$SMOKE_STATE_DIR"' EXIT

inspect_release_attestation() {
    if [[ -z "$ATTESTATION_PUBLIC_KEY_FILE" ]]; then
        echo "Release attestation inspect skipped (set MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE to verify stamped binaries)"
        return 0
    fi

    local inspect_json
    local inspect_status
    inspect_json="$(cargo run -q -p xtask -- release-attestation inspect --binary "$MESH_LLM" --public-key-file "$ATTESTATION_PUBLIC_KEY_FILE" --json)"
    echo "Release attestation inspect: $inspect_json"
    inspect_status="$(printf '%s' "$inspect_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
    if [[ "$inspect_status" != "$ATTESTATION_EXPECTED_STATUS" ]]; then
        echo "Unexpected embedded release-attestation status: expected $ATTESTATION_EXPECTED_STATUS, got $inspect_status" >&2
        exit 1
    fi
}

echo "=== CI OpenAI Compatibility Smoke ==="
echo "  mesh-llm:  $MESH_LLM"
echo "  bin-dir:   $BIN_DIR (compatibility placeholder)"
echo "  model:     $MODEL"
echo "  api port:  $API_PORT"
echo "  console:   $CONSOLE_PORT"
echo "  device:    $SMOKE_DEVICE"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "Missing executable mesh-llm binary: $MESH_LLM" >&2
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Missing model file: $MODEL" >&2
    exit 1
fi

inspect_release_attestation

RUNTIME_BUNDLE="${MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR:-$(cd "$(dirname "$MESH_LLM")" && pwd)/native-runtimes}"
if [[ ! -d "$RUNTIME_BUNDLE" ]]; then
    echo "Missing packaged native runtime beside mesh-llm: $RUNTIME_BUNDLE" >&2
    exit 1
fi
export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$RUNTIME_BUNDLE"

env MESH_LLM_CONFIG="$SMOKE_CONFIG_PATH" MESH_LLM_RUNTIME_ROOT="$SMOKE_RUNTIME_ROOT" \
    "$MESH_LLM" \
    --log-format json \
    serve \
    --model "$MODEL" \
    --no-draft \
    --device "$SMOKE_DEVICE" \
    --ctx-size "${MESH_COMPAT_CTX_SIZE:-256}" \
    --port "$API_PORT" \
    --console "$CONSOLE_PORT" \
    --headless \
    >"$LOG" 2>&1 &
MESH_PID=$!

cleanup() {
    kill "$MESH_PID" 2>/dev/null || true
    pkill -P "$MESH_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$MESH_PID" 2>/dev/null || true
    wait "$MESH_PID" 2>/dev/null || true
    echo "--- compat mesh-llm log tail ---"
    tail -100 "$LOG" 2>/dev/null || true
    echo "--- end log ---"
    rm -rf "$SMOKE_STATE_DIR"
}
trap cleanup EXIT

BASE_URL="http://127.0.0.1:${API_PORT}/v1"
MODEL_ID=""
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$MESH_PID" 2>/dev/null; then
        echo "mesh-llm exited unexpectedly" >&2
        tail -120 "$LOG" >&2 || true
        exit 1
    fi

    MODELS_JSON="$(curl -sf "${BASE_URL}/models" 2>/dev/null || true)"
    MODEL_ID="$(
        printf '%s' "$MODELS_JSON" |
            python3 -c 'import json,sys; data=json.load(sys.stdin).get("data", []); print(data[0].get("id", "") if data else "")' 2>/dev/null ||
            echo ""
    )"
    if [[ -n "$MODEL_ID" ]]; then
        echo "OpenAI endpoint ready with model: $MODEL_ID"
        break
    fi

    if [[ "$i" -eq "$MAX_WAIT" ]]; then
        echo "Timed out waiting for OpenAI endpoint" >&2
        tail -120 "$LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

RUNTIME_ATTESTATION_STATUS="$({
    curl -sf "http://127.0.0.1:${CONSOLE_PORT}/api/status" 2>/dev/null |
        python3 -c 'import json,sys; print(json.load(sys.stdin).get("release_attestation", {}).get("status", ""))' 2>/dev/null ||
        echo ""
})"
echo "Runtime release attestation status: ${RUNTIME_ATTESTATION_STATUS:-unknown}"
if [[ -n "$ATTESTATION_PUBLIC_KEY_FILE" && "$RUNTIME_ATTESTATION_STATUS" != "$ATTESTATION_EXPECTED_STATUS" ]]; then
    echo "Unexpected runtime release-attestation status: expected $ATTESTATION_EXPECTED_STATUS, got ${RUNTIME_ATTESTATION_STATUS:-<empty>}" >&2
    exit 1
fi

python3 scripts/ci-openai-python-smoke.py --base-url "$BASE_URL"
python3 scripts/ci-litellm-smoke.py --base-url "$BASE_URL" --model "$MODEL_ID"
python3 scripts/ci-langchain-openai-smoke.py --base-url "$BASE_URL" --model "$MODEL_ID"
NODE_PATH="${NODE_PATH:-$(npm root -g 2>/dev/null || true)}" node scripts/ci-openai-node-smoke.cjs --base-url "$BASE_URL"

python3 scripts/ci-verify-smoke-offload.py --device "$SMOKE_DEVICE" \
    --native-log "$SMOKE_RUNTIME_ROOT/$MESH_PID/logs/skippy-native.log"

echo "OpenAI compatibility smoke passed"
