#!/usr/bin/env bash
# ci-smoke-test.sh - start mesh-llm with a tiny GGUF, exercise OpenAI routes.
#
# Usage: scripts/ci-smoke-test.sh <mesh-llm-binary> <bin-dir> <model-path> [mmproj-path]
#
# <bin-dir> is retained for compatibility with older workflow call sites. The
# skippy runtime is embedded in mesh-llm and does not require external
# llama-server or rpc-server binaries.

set -euo pipefail

# Bounded hardware selection; CUDA can never silently opt out of offload checks.
SMOKE_DEVICE="${MESH_CI_DEVICE:-CPU}"
case "$SMOKE_DEVICE" in
    CPU|CUDA0|Vulkan0|ROCm0|MTL0) ;;
    *) echo "Unsupported MESH_CI_DEVICE: $SMOKE_DEVICE (expected CPU, CUDA0, Vulkan0, ROCm0 or MTL0)" >&2; exit 1 ;;
esac

MESH_LLM="${1:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path> [mmproj-path]}"
BIN_DIR="${2:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path> [mmproj-path]}"
MODEL="${3:?Usage: $0 <mesh-llm-binary> <bin-dir> <model-path> [mmproj-path]}"
MMPROJ="${4:-}"
API_PORT="${MESH_CI_API_PORT:-9337}"
CONSOLE_PORT="${MESH_CI_CONSOLE_PORT:-3131}"
MAX_WAIT="${MESH_CI_MAX_WAIT:-180}"
LOG="${MESH_CI_LOG:-/tmp/mesh-llm-ci.log}"
ATTESTATION_PUBLIC_KEY_FILE="${MESH_RELEASE_ATTESTATION_PUBLIC_KEY_FILE:-}"
ATTESTATION_EXPECTED_STATUS="${MESH_RELEASE_ATTESTATION_EXPECTED_STATUS:-valid}"
SMOKE_STATE_DIR="$(mktemp -d /tmp/mesh-llm-smoke-state.XXXXXX)"
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

runtime_release_attestation_status() {
    curl -sf "http://127.0.0.1:${CONSOLE_PORT}/api/status" 2>/dev/null |
        python3 -c 'import json,sys; print(json.load(sys.stdin).get("release_attestation", {}).get("status", ""))' 2>/dev/null ||
        echo ""
}

echo "=== CI Skippy Smoke Test ==="
echo "  mesh-llm:  $MESH_LLM"
echo "  bin-dir:   $BIN_DIR (compatibility placeholder)"
echo "  model:     $MODEL"
if [[ -n "$MMPROJ" ]]; then
    echo "  mmproj:    $MMPROJ"
fi
echo "  api port:  $API_PORT"
echo "  console:   $CONSOLE_PORT"
echo "  os:        $(uname -s)"
echo "  device:    $SMOKE_DEVICE"

if [[ ! -x "$MESH_LLM" ]]; then
    echo "Missing executable mesh-llm binary: $MESH_LLM" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "Missing model: $MODEL" >&2
    exit 1
fi

inspect_release_attestation

RUNTIME_BUNDLE="${MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR:-$(cd "$(dirname "$MESH_LLM")" && pwd)/native-runtimes}"
if [[ ! -d "$RUNTIME_BUNDLE" ]]; then
    echo "Missing packaged native runtime beside mesh-llm: $RUNTIME_BUNDLE" >&2
    exit 1
fi
export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$RUNTIME_BUNDLE"

# Exercise the SDK's CLI-JSON consumer with the actual composed host, even when
# semantic routing did not select a full SDK build. Mock JSON fixtures cannot
# catch a producer changing its output shape independently of this consumer.
MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK=0 scripts/ci-prepare-native-runtime.sh \
    "$SMOKE_STATE_DIR/sdk-runtime" cpu --reuse-from-binary "$MESH_LLM"

ARGS=(
    --log-format json
    serve
    --model "$MODEL"
    --no-draft
    --device "$SMOKE_DEVICE"
    --ctx-size "${MESH_CI_CTX_SIZE:-256}"
    --port "$API_PORT"
    --console "$CONSOLE_PORT"
)

if [[ -n "$MMPROJ" ]]; then
    ARGS+=(--mmproj "$MMPROJ")
fi

env MESH_LLM_CONFIG="$SMOKE_CONFIG_PATH" MESH_LLM_RUNTIME_ROOT="$SMOKE_RUNTIME_ROOT" \
    "$MESH_LLM" "${ARGS[@]}" >"$LOG" 2>&1 &
MESH_PID=$!

cleanup() {
    echo "Shutting down mesh-llm (PID $MESH_PID)..."
    kill "$MESH_PID" 2>/dev/null || true
    pkill -P "$MESH_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$MESH_PID" 2>/dev/null || true
    wait "$MESH_PID" 2>/dev/null || true
    echo "--- mesh-llm log tail ---"
    tail -100 "$LOG" 2>/dev/null || true
    echo "--- end log ---"
    rm -rf "$SMOKE_STATE_DIR"
}
trap cleanup EXIT

echo "Waiting for skippy runtime readiness (up to ${MAX_WAIT}s)..."
for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$MESH_PID" 2>/dev/null; then
        echo "mesh-llm exited unexpectedly" >&2
        tail -120 "$LOG" >&2 || true
        exit 1
    fi

    READY="$(
        curl -sf "http://127.0.0.1:${CONSOLE_PORT}/api/status" 2>/dev/null |
            python3 -c 'import json,sys; print(json.load(sys.stdin).get("llama_ready", False))' 2>/dev/null ||
            echo "False"
    )"
    if [[ "$READY" == "True" ]]; then
        echo "Runtime ready after ${i}s"
        break
    fi

    if [[ "$i" -eq "$MAX_WAIT" ]]; then
        echo "Timed out waiting for runtime readiness" >&2
        tail -120 "$LOG" >&2 || true
        exit 1
    fi

    if (( i % 15 == 0 )); then
        echo "  Still waiting... (${i}s)"
        tail -5 "$LOG" 2>/dev/null | sed 's/^/    /' || true
    fi
    sleep 1
done

echo "Waiting for /v1/models..."
SMOKE_MODEL_ID=""
for i in $(seq 1 60); do
    MODELS_JSON="$(curl -sf "http://127.0.0.1:${API_PORT}/v1/models" 2>/dev/null || true)"
    SMOKE_MODEL_ID="$(
        printf '%s' "$MODELS_JSON" |
            python3 -c 'import json,sys; data=json.load(sys.stdin).get("data", []); print(data[0].get("id", "") if data else "")' 2>/dev/null ||
            echo ""
    )"
    if [[ -n "$SMOKE_MODEL_ID" ]]; then
        echo "OpenAI API ready with model: $SMOKE_MODEL_ID"
        break
    fi

    if [[ "$i" -eq 60 ]]; then
        echo "OpenAI API did not publish a model" >&2
        tail -120 "$LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

RUNTIME_ATTESTATION_STATUS="$(runtime_release_attestation_status)"
echo "Runtime release attestation status: ${RUNTIME_ATTESTATION_STATUS:-unknown}"
if [[ -n "$ATTESTATION_PUBLIC_KEY_FILE" && "$RUNTIME_ATTESTATION_STATUS" != "$ATTESTATION_EXPECTED_STATUS" ]]; then
    echo "Unexpected runtime release-attestation status: expected $ATTESTATION_EXPECTED_STATUS, got ${RUNTIME_ATTESTATION_STATUS:-<empty>}" >&2
    exit 1
fi

BASE_URL="http://127.0.0.1:${API_PORT}/v1"

echo "Testing non-stream chat completion..."
CHAT_PAYLOAD="$(
    jq -cn --arg model "$SMOKE_MODEL_ID" '{
      model: $model,
      messages: [{role: "user", content: "Say hello in exactly 3 words."}],
      max_tokens: 4,
      temperature: 0
    }'
)"
RESPONSE="$(curl -fsS --max-time 60 "${BASE_URL}/chat/completions" -H 'content-type: application/json' -d "$CHAT_PAYLOAD")"
printf '%s' "$RESPONSE" | jq -e '.object == "chat.completion" and (.choices[0].message.content | length > 0)' >/dev/null

echo "Testing stream chat completion..."
STREAM_PAYLOAD="$(
    jq -cn --arg model "$SMOKE_MODEL_ID" '{
      model: $model,
      messages: [{role: "user", content: "Count from one to three."}],
      stream: true,
      stream_options: {include_usage: true},
      max_tokens: 4,
      temperature: 0
    }'
)"
STREAM_OUT="$(mktemp)"
if ! curl -fsS --max-time 60 -N "${BASE_URL}/chat/completions" -H 'content-type: application/json' -d "$STREAM_PAYLOAD" >"$STREAM_OUT"; then
    rm -f "$STREAM_OUT"
    exit 1
fi
if ! grep -q 'data: \[DONE\]' "$STREAM_OUT"; then
    rm -f "$STREAM_OUT"
    exit 1
fi
if ! grep -q '"role":"assistant"' "$STREAM_OUT"; then
    rm -f "$STREAM_OUT"
    exit 1
fi
rm -f "$STREAM_OUT"

echo "Testing model=auto routing..."
AUTO_PAYLOAD="$(
    jq -cn '{
      model: "auto",
      messages: [{role: "user", content: "Say hi."}],
      max_tokens: 4,
      temperature: 0
    }'
)"
AUTO_RESPONSE="$(curl -fsS --max-time 60 "${BASE_URL}/chat/completions" -H 'content-type: application/json' -d "$AUTO_PAYLOAD")"
printf '%s' "$AUTO_RESPONSE" | jq -e '(.choices[0].message.content | length > 0)' >/dev/null

python3 scripts/ci-verify-smoke-offload.py --device "$SMOKE_DEVICE" \
    --native-log "$SMOKE_RUNTIME_ROOT/$MESH_PID/logs/skippy-native.log"

echo "Testing headless mode subcase..."
HEADLESS_API_PORT="${MESH_CI_HEADLESS_API_PORT:-9338}"
HEADLESS_CONSOLE_PORT="${MESH_CI_HEADLESS_CONSOLE_PORT:-3132}"
HEADLESS_LOG="${MESH_CI_HEADLESS_LOG:-/tmp/mesh-llm-ci-headless.log}"
HEADLESS_ARGS=(
    --log-format json
    serve
    --model "$MODEL"
    --no-draft
    --device "$SMOKE_DEVICE"
    --ctx-size "${MESH_CI_CTX_SIZE:-256}"
    --port "$HEADLESS_API_PORT"
    --console "$HEADLESS_CONSOLE_PORT"
    --headless
)
if [[ -n "$MMPROJ" ]]; then
    HEADLESS_ARGS+=(--mmproj "$MMPROJ")
fi

env MESH_LLM_CONFIG="$SMOKE_CONFIG_PATH" MESH_LLM_RUNTIME_ROOT="$SMOKE_RUNTIME_ROOT" \
    "$MESH_LLM" "${HEADLESS_ARGS[@]}" >"$HEADLESS_LOG" 2>&1 &
HEADLESS_PID=$!

headless_cleanup() {
    kill "$HEADLESS_PID" 2>/dev/null || true
    pkill -P "$HEADLESS_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$HEADLESS_PID" 2>/dev/null || true
    wait "$HEADLESS_PID" 2>/dev/null || true
}
trap 'headless_cleanup; cleanup' EXIT

for i in $(seq 1 "$MAX_WAIT"); do
    if ! kill -0 "$HEADLESS_PID" 2>/dev/null; then
        echo "headless mesh-llm exited unexpectedly" >&2
        tail -100 "$HEADLESS_LOG" >&2 || true
        exit 1
    fi

    if curl -sf "http://127.0.0.1:${HEADLESS_API_PORT}/v1/models" | jq -e '.data | length > 0' >/dev/null 2>&1 &&
       curl -sf "http://127.0.0.1:${HEADLESS_CONSOLE_PORT}/api/status" | jq -e '.llama_ready == true' >/dev/null 2>&1; then
        echo "Headless mode ready after ${i}s"
        break
    fi

    if [[ "$i" -eq "$MAX_WAIT" ]]; then
        echo "Timed out waiting for headless mode" >&2
        tail -100 "$HEADLESS_LOG" >&2 || true
        exit 1
    fi
    sleep 1
done

HEADLESS_ATTESTATION_STATUS="$({
    curl -sf "http://127.0.0.1:${HEADLESS_CONSOLE_PORT}/api/status" 2>/dev/null |
        python3 -c 'import json,sys; print(json.load(sys.stdin).get("release_attestation", {}).get("status", ""))' 2>/dev/null ||
        echo ""
})"
echo "Headless runtime release attestation status: ${HEADLESS_ATTESTATION_STATUS:-unknown}"
if [[ -n "$ATTESTATION_PUBLIC_KEY_FILE" && "$HEADLESS_ATTESTATION_STATUS" != "$ATTESTATION_EXPECTED_STATUS" ]]; then
    echo "Unexpected headless runtime release-attestation status: expected $ATTESTATION_EXPECTED_STATUS, got ${HEADLESS_ATTESTATION_STATUS:-<empty>}" >&2
    exit 1
fi

# HTTP readiness alone can precede model loading. Exercise this process too,
# rather than borrowing the first process's successful completion/offload log.
echo "Testing headless chat completion..."
HEADLESS_RESPONSE="$(curl -fsS --max-time 60 \
    "http://127.0.0.1:${HEADLESS_API_PORT}/v1/chat/completions" \
    -H 'content-type: application/json' -d "$AUTO_PAYLOAD")"
printf '%s' "$HEADLESS_RESPONSE" | jq -e '(.choices[0].message.content | length > 0)' >/dev/null
python3 scripts/ci-verify-smoke-offload.py --device "$SMOKE_DEVICE" \
    --native-log "$SMOKE_RUNTIME_ROOT/$HEADLESS_PID/logs/skippy-native.log"

echo "Skippy smoke passed"
