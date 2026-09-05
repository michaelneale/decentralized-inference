#!/usr/bin/env bash
# ci-product-integration-smoke.sh - registry-backed composed-product acceptance.
#
# Usage: scripts/ci-product-integration-smoke.sh <mesh-llm> <artifact-dir>
#        <dense-model> <recurrent-model> <platform> <backend>

set -euo pipefail

MESH_LLM="${1:?Usage: $0 <mesh-llm> <artifact-dir> <dense-model> <recurrent-model> <platform> <backend>}"
ARTIFACT_DIR="${2:?Usage: $0 <mesh-llm> <artifact-dir> <dense-model> <recurrent-model> <platform> <backend>}"
DENSE_MODEL="${3:?Usage: $0 <mesh-llm> <artifact-dir> <dense-model> <recurrent-model> <platform> <backend>}"
RECURRENT_MODEL="${4:?Usage: $0 <mesh-llm> <artifact-dir> <dense-model> <recurrent-model> <platform> <backend>}"
PLATFORM="${5:?Usage: $0 <mesh-llm> <artifact-dir> <dense-model> <recurrent-model> <platform> <backend>}"
BACKEND="${6:?Usage: $0 <mesh-llm> <artifact-dir> <dense-model> <recurrent-model> <platform> <backend>}"
PHASE_ROOT="${MESH_PRODUCT_INTEGRATION_PHASE_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/mesh-product-integration.XXXXXX")}"

case "${PLATFORM}/${BACKEND}" in
    linux/cpu) DEVICE=CPU ;;
    linux/cuda) DEVICE=CUDA0 ;;
    macos/metal) DEVICE=MTL0 ;;
    *)
        echo "unsupported typed product suite combination: ${PLATFORM}/${BACKEND}" >&2
        exit 2
        ;;
esac

for required in "$MESH_LLM" "$DENSE_MODEL" "$RECURRENT_MODEL" "$ARTIFACT_DIR/product-manifest.json"; do
    [[ -f "$required" || -x "$required" ]] || {
        echo "missing product-integration input: $required" >&2
        exit 1
    }
done

bundle_backend="$(python3 - "$ARTIFACT_DIR/product-manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("backend", ""))
PY
)"
if [[ "$bundle_backend" != "$BACKEND" ]]; then
    echo "composed product backend mismatch: expected $BACKEND, got ${bundle_backend:-empty}" >&2
    exit 1
fi

mkdir -p "$PHASE_ROOT"
printf 'platform=%s\nbackend=%s\ndevice=%s\nproduct_backend=%s\n' \
    "$PLATFORM" "$BACKEND" "$DEVICE" "$bundle_backend" >"$PHASE_ROOT/provenance.txt"

echo "=== Product integration suite ==="
echo "  platform/backend: ${PLATFORM}/${BACKEND}"
echo "  requested device:  $DEVICE"
echo "  phase root:        $PHASE_ROOT"

run_phase() {
    local phase="$1"
    shift
    local phase_dir="${PHASE_ROOT}/${phase}"
    mkdir -p "$phase_dir"
    echo "=== phase: ${phase} ==="
    "$@"
}

run_phase dense-standalone env \
    MESH_CI_DEVICE="$DEVICE" \
    MESH_CI_API_PORT=9337 MESH_CI_CONSOLE_PORT=3131 \
    MESH_CI_HEADLESS_API_PORT=9338 MESH_CI_HEADLESS_CONSOLE_PORT=3132 \
    MESH_CI_LOG="$PHASE_ROOT/dense-standalone/server.log" \
    MESH_CI_HEADLESS_LOG="$PHASE_ROOT/dense-standalone/headless.log" \
    scripts/ci-smoke-test.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

run_phase dense-openai-sdk env \
    MESH_COMPAT_DEVICE="$DEVICE" \
    MESH_COMPAT_API_PORT=9348 MESH_COMPAT_CONSOLE_PORT=3142 \
    MESH_COMPAT_LOG="$PHASE_ROOT/dense-openai-sdk/server.log" \
    scripts/ci-compat-smoke.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

run_phase dense-constrained-tokio-restart env \
    MESH_CI_DEVICE="$DEVICE" MESH_TOKIO_STACK_SIZE=2097152 \
    MESH_CI_API_PORT=9347 MESH_CI_CONSOLE_PORT=3141 \
    MESH_CI_HEADLESS_API_PORT=9349 MESH_CI_HEADLESS_CONSOLE_PORT=3143 \
    MESH_CI_LOG="$PHASE_ROOT/dense-constrained-tokio-restart/server.log" \
    MESH_CI_HEADLESS_LOG="$PHASE_ROOT/dense-constrained-tokio-restart/headless.log" \
    scripts/ci-smoke-test.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

run_phase dense-and-recurrent-split-kv env \
    MESH_TWO_NODE_SPLIT_DEVICE="$DEVICE" \
    MESH_TWO_NODE_SPLIT_MODEL="$DENSE_MODEL" \
    MESH_TWO_NODE_SPLIT_RECURRENT_MODEL="$RECURRENT_MODEL" \
    MESH_TWO_NODE_SPLIT_RECURRENT_CTX_SIZE=4096 \
    MESH_TWO_NODE_SPLIT_RECURRENT_EXPECTED_EXACT_PAYLOAD_KIND=kv-recurrent \
    MESH_TWO_NODE_SPLIT_CLIENT_ROUTING=1 \
    MESH_TWO_NODE_SPLIT_WORK_DIR="$PHASE_ROOT/dense-and-recurrent-split-kv" \
    scripts/ci-two-node-split-smoke.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

echo "Product integration suite passed; phase logs retained at $PHASE_ROOT"
