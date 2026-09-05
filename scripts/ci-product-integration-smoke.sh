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
PHASE_MANIFEST="$PHASE_ROOT/phase-results.json"
PHASE_RECORDS="$PHASE_ROOT/.phase-results.jsonl"
SUITE_STARTED_AT_UNIX_NS=""
SUITE_FINALIZED=0

readonly -a REQUIRED_PHASES=(
    dense-standalone
    dense-openai-sdk
    dense-constrained-tokio-restart
    dense-split-kv
    recurrent-split-kv
)

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

phase_now_unix_ns() {
    python3 -c 'import time; print(time.time_ns())'
}

append_phase_record() {
    local phase="$1"
    local status="$2"
    local model_label="$3"
    local model_identity="$4"
    local workdir="$5"
    local log_paths_json="$6"
    local started_at_unix_ns="$7"
    local ended_at_unix_ns="$8"
    local exit_code="$9"

    python3 - "$PHASE_RECORDS" "$phase" "$status" "$model_label" \
        "$model_identity" "$workdir" "$log_paths_json" "$started_at_unix_ns" \
        "$ended_at_unix_ns" "$exit_code" <<'PY'
import json
import sys

(
    records_path,
    phase,
    status,
    model_label,
    model_identity,
    workdir,
    log_paths_json,
    started_at_unix_ns,
    ended_at_unix_ns,
    exit_code,
) = sys.argv[1:]

record = {
    "phase": phase,
    "status": status,
    "model": {"label": model_label, "identity": model_identity},
    "workdir": workdir,
    "log_paths": json.loads(log_paths_json),
    "started_at_unix_ns": int(started_at_unix_ns),
    "ended_at_unix_ns": int(ended_at_unix_ns),
    "exit_code": int(exit_code),
}
with open(records_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
PY
}

write_phase_manifest() {
    local suite_status="$1"
    local failure_phase="$2"
    local finalize="$3"

    python3 - "$PHASE_RECORDS" "$PHASE_MANIFEST" "$PLATFORM" "$BACKEND" \
        "$DEVICE" "$bundle_backend" "$SUITE_STARTED_AT_UNIX_NS" "$suite_status" \
        "$failure_phase" "$finalize" "${REQUIRED_PHASES[@]}" <<'PY'
import json
import os
import sys

(
    records_path,
    manifest_path,
    platform,
    backend,
    device,
    product_backend,
    suite_started_at_unix_ns,
    suite_status,
    failure_phase,
    finalize,
    *required_phases,
) = sys.argv[1:]
finalize = bool(int(finalize))

records = []
if os.path.exists(records_path):
    with open(records_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"invalid phase record {line_number}: {error}")

errors = []
seen = set()
required = set(required_phases)
for record in records:
    phase = record.get("phase")
    if phase not in required:
        errors.append(f"unplanned phase record: {phase!r}")
        continue
    if phase in seen:
        errors.append(f"duplicate phase record: {phase}")
    seen.add(phase)
    model = record.get("model")
    if (
        record.get("status") not in {"passed", "failed"}
        or not isinstance(model, dict)
        or not isinstance(model.get("label"), str)
        or not model["label"]
        or not isinstance(model.get("identity"), str)
        or not model["identity"]
        or not isinstance(record.get("workdir"), str)
        or not record["workdir"]
        or not isinstance(record.get("log_paths"), list)
        or not record["log_paths"]
        or any(not isinstance(path, str) or not path for path in record["log_paths"])
        or not isinstance(record.get("started_at_unix_ns"), int)
        or not isinstance(record.get("ended_at_unix_ns"), int)
        or record["ended_at_unix_ns"] < record["started_at_unix_ns"]
        or not isinstance(record.get("exit_code"), int)
    ):
        errors.append(f"incomplete phase record: {phase!r}")

missing = [phase for phase in required_phases if phase not in seen]
if finalize:
    errors.extend(f"missing required phase: {phase}" for phase in missing)
    errors.extend(
        f"phase did not pass: {record['phase']}"
        for record in records
        if record.get("status") != "passed"
    )
    if suite_status != "passed":
        errors.append(f"suite ended with status: {suite_status}")
    if failure_phase and failure_phase not in required:
        errors.append(f"unplanned failing phase: {failure_phase}")

reconciliation_status = (
    "passed" if finalize and not errors else "failed" if finalize else "in-progress"
)
manifest = {
    "schema_version": 1,
    "suite": "product-integration",
    "suite_status": suite_status,
    "provenance": {
        "platform": platform,
        "backend": backend,
        "device": device,
        "product_backend": product_backend,
    },
    "started_at_unix_ns": int(suite_started_at_unix_ns),
    "required_phases": required_phases,
    "phases": records,
    "reconciliation": {
        "status": reconciliation_status,
        "finalized": finalize,
        "failure_phase": failure_phase or None,
        "missing_phases": missing,
        "errors": errors,
    },
}
temporary_path = f"{manifest_path}.tmp"
with open(temporary_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary_path, manifest_path)

if finalize and errors:
    raise SystemExit(1)
PY
}

ensure_phase_is_planned_once() {
    local phase="$1"

    python3 - "$PHASE_RECORDS" "$phase" "${REQUIRED_PHASES[@]}" <<'PY'
import json
import os
import sys

records_path, phase, *required_phases = sys.argv[1:]
if phase not in required_phases:
    raise SystemExit(f"unplanned product integration phase: {phase}")
if not os.path.exists(records_path):
    raise SystemExit(0)
with open(records_path, encoding="utf-8") as handle:
    for line in handle:
        if line.strip() and json.loads(line).get("phase") == phase:
            raise SystemExit(f"duplicate product integration phase: {phase}")
PY
}

finalize_interrupted_suite() {
    local exit_code="$?"
    trap - EXIT
    if [[ "$SUITE_FINALIZED" -eq 0 && -n "$SUITE_STARTED_AT_UNIX_NS" ]]; then
        set +e
        write_phase_manifest failed "" 1
    fi
    exit "$exit_code"
}

SUITE_STARTED_AT_UNIX_NS="$(phase_now_unix_ns)"
: >"$PHASE_RECORDS"
write_phase_manifest in-progress "" 0
trap finalize_interrupted_suite EXIT

echo "=== Product integration suite ==="
echo "  platform/backend: ${PLATFORM}/${BACKEND}"
echo "  requested device:  $DEVICE"
echo "  phase root:        $PHASE_ROOT"

run_phase() {
    local phase="$1"
    local model_label="$2"
    local model_identity="$3"
    local log_paths_json="$4"
    shift 4
    local phase_dir="${PHASE_ROOT}/${phase}"
    local started_at_unix_ns
    local ended_at_unix_ns
    local phase_exit_code
    local phase_status

    if ! ensure_phase_is_planned_once "$phase"; then
        write_phase_manifest failed "$phase" 1 || true
        SUITE_FINALIZED=1
        return 70
    fi
    mkdir -p "$phase_dir"
    echo "=== phase: ${phase} ==="
    started_at_unix_ns="$(phase_now_unix_ns)"
    if MESH_PRODUCT_INTEGRATION_PHASE="$phase" "$@"; then
        phase_exit_code=0
        phase_status=passed
    else
        phase_exit_code=$?
        phase_status=failed
    fi
    ended_at_unix_ns="$(phase_now_unix_ns)"
    append_phase_record "$phase" "$phase_status" "$model_label" "$model_identity" \
        "$phase_dir" "$log_paths_json" "$started_at_unix_ns" "$ended_at_unix_ns" \
        "$phase_exit_code"
    if [[ "$phase_exit_code" -ne 0 ]]; then
        write_phase_manifest failed "$phase" 1 || true
        SUITE_FINALIZED=1
        return "$phase_exit_code"
    fi
    write_phase_manifest in-progress "" 0
}

run_phase dense-standalone dense "$DENSE_MODEL" \
    "[\"$PHASE_ROOT/dense-standalone/server.log\",\"$PHASE_ROOT/dense-standalone/headless.log\"]" env \
    MESH_CI_DEVICE="$DEVICE" \
    MESH_CI_API_PORT=9337 MESH_CI_CONSOLE_PORT=3131 \
    MESH_CI_HEADLESS_API_PORT=9338 MESH_CI_HEADLESS_CONSOLE_PORT=3132 \
    MESH_CI_LOG="$PHASE_ROOT/dense-standalone/server.log" \
    MESH_CI_HEADLESS_LOG="$PHASE_ROOT/dense-standalone/headless.log" \
    scripts/ci-smoke-test.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

run_phase dense-openai-sdk dense "$DENSE_MODEL" \
    "[\"$PHASE_ROOT/dense-openai-sdk/server.log\"]" env \
    MESH_COMPAT_DEVICE="$DEVICE" \
    MESH_COMPAT_API_PORT=9348 MESH_COMPAT_CONSOLE_PORT=3142 \
    MESH_COMPAT_LOG="$PHASE_ROOT/dense-openai-sdk/server.log" \
    scripts/ci-compat-smoke.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

run_phase dense-constrained-tokio-restart dense "$DENSE_MODEL" \
    "[\"$PHASE_ROOT/dense-constrained-tokio-restart/server.log\",\"$PHASE_ROOT/dense-constrained-tokio-restart/headless.log\"]" env \
    MESH_CI_DEVICE="$DEVICE" MESH_TOKIO_STACK_SIZE=2097152 \
    MESH_CI_API_PORT=9347 MESH_CI_CONSOLE_PORT=3141 \
    MESH_CI_HEADLESS_API_PORT=9349 MESH_CI_HEADLESS_CONSOLE_PORT=3143 \
    MESH_CI_LOG="$PHASE_ROOT/dense-constrained-tokio-restart/server.log" \
    MESH_CI_HEADLESS_LOG="$PHASE_ROOT/dense-constrained-tokio-restart/headless.log" \
    scripts/ci-smoke-test.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

run_phase dense-split-kv dense "$DENSE_MODEL" \
    "[\"$PHASE_ROOT/dense-split-kv/dense-seed.log\",\"$PHASE_ROOT/dense-split-kv/dense-worker.log\",\"$PHASE_ROOT/dense-split-kv/dense-client.log\"]" env \
    MESH_TWO_NODE_SPLIT_DEVICE="$DEVICE" \
    MESH_TWO_NODE_SPLIT_MODEL="$DENSE_MODEL" \
    MESH_TWO_NODE_SPLIT_MODEL_LABEL=dense \
    MESH_TWO_NODE_SPLIT_CLIENT_ROUTING=1 \
    MESH_TWO_NODE_SPLIT_WORK_DIR="$PHASE_ROOT/dense-split-kv" \
    scripts/ci-two-node-split-smoke.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

run_phase recurrent-split-kv recurrent "$RECURRENT_MODEL" \
    "[\"$PHASE_ROOT/recurrent-split-kv/recurrent-seed.log\",\"$PHASE_ROOT/recurrent-split-kv/recurrent-worker.log\",\"$PHASE_ROOT/recurrent-split-kv/recurrent-client.log\"]" env \
    MESH_TWO_NODE_SPLIT_DEVICE="$DEVICE" \
    MESH_TWO_NODE_SPLIT_MODEL="$RECURRENT_MODEL" \
    MESH_TWO_NODE_SPLIT_MODEL_LABEL=recurrent \
    MESH_TWO_NODE_SPLIT_CTX_SIZE=4096 \
    MESH_TWO_NODE_SPLIT_EXPECTED_EXACT_PAYLOAD_KIND=kv-recurrent \
    MESH_TWO_NODE_SPLIT_WORK_DIR="$PHASE_ROOT/recurrent-split-kv" \
    scripts/ci-two-node-split-smoke.sh "$MESH_LLM" "$ARTIFACT_DIR" "$RECURRENT_MODEL"

write_phase_manifest passed "" 1
SUITE_FINALIZED=1
trap - EXIT

echo "Product integration suite passed; phase manifest and logs retained at $PHASE_ROOT"
