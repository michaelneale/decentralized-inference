#!/usr/bin/env bash
# ci-product-integration-smoke.sh - registry-backed composed-product acceptance.
#
# Usage: scripts/ci-product-integration-smoke.sh <mesh-llm> <artifact-dir>
#        <dense-model> <recurrent-model> <platform> <backend>
#        <dense-artifact-id> <dense-sha256> <recurrent-artifact-id> <recurrent-sha256>

set -euo pipefail

readonly USAGE="$0 <mesh-llm> <artifact-dir> <dense-model> <recurrent-model> <platform> <backend> <dense-artifact-id> <dense-sha256> <recurrent-artifact-id> <recurrent-sha256>"
MESH_LLM="${1:?Usage: $USAGE}"
ARTIFACT_DIR="${2:?Usage: $USAGE}"
DENSE_MODEL="${3:?Usage: $USAGE}"
RECURRENT_MODEL="${4:?Usage: $USAGE}"
PLATFORM="${5:?Usage: $USAGE}"
BACKEND="${6:?Usage: $USAGE}"
DENSE_ARTIFACT_ID="${7:?Usage: $USAGE}"
DENSE_SHA256="${8:?Usage: $USAGE}"
RECURRENT_ARTIFACT_ID="${9:?Usage: $USAGE}"
RECURRENT_SHA256="${10:?Usage: $USAGE}"
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

fixture_sha256() {
    python3 - "$1" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

verify_fixture_identity() {
    local label="$1"
    local expected_artifact_id="$2"
    local artifact_id="$3"
    local expected_sha256="$4"
    local model_path="$5"
    local actual_sha256

    if [[ "$artifact_id" != "$expected_artifact_id" ]]; then
        echo "unexpected ${label} fixture artifact id: expected ${expected_artifact_id}, got ${artifact_id}" >&2
        return 1
    fi
    if [[ ! "$expected_sha256" =~ ^[0-9a-f]{64}$ ]]; then
        echo "invalid ${label} fixture SHA-256" >&2
        return 1
    fi
    actual_sha256="$(fixture_sha256 "$model_path")"
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
        echo "${label} fixture digest mismatch: expected ${expected_sha256}, got ${actual_sha256}" >&2
        return 1
    fi
}

fixture_manifest_model() {
    local label="$1"
    local artifact_id
    local sha256
    local path

    case "$label" in
        dense)
            artifact_id="$DENSE_ARTIFACT_ID"
            sha256="$DENSE_SHA256"
            path="$DENSE_MODEL"
            ;;
        recurrent)
            artifact_id="$RECURRENT_ARTIFACT_ID"
            sha256="$RECURRENT_SHA256"
            path="$RECURRENT_MODEL"
            ;;
        *)
            echo "unplanned fixture label: ${label}" >&2
            return 1
            ;;
    esac

    python3 - "$label" "$artifact_id" "$sha256" "$path" <<'PY'
import json
import sys

label, artifact_id, sha256, path = sys.argv[1:]
print(json.dumps({
    "label": label,
    "artifact_id": artifact_id,
    "sha256": sha256,
    "path": path,
}, sort_keys=True, separators=(",", ":")))
PY
}

verify_fixture_identity dense smollm2-q8-inference "$DENSE_ARTIFACT_ID" "$DENSE_SHA256" "$DENSE_MODEL"
verify_fixture_identity recurrent family-granite-hybrid "$RECURRENT_ARTIFACT_ID" "$RECURRENT_SHA256" "$RECURRENT_MODEL"

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
    local model_json="$3"
    local workdir="$4"
    local log_paths_json="$5"
    local split_evidence_json="$6"
    local started_at_unix_ns="$7"
    local ended_at_unix_ns="$8"
    local exit_code="$9"

    python3 - "$PHASE_RECORDS" "$phase" "$status" "$model_json" \
        "$workdir" "$log_paths_json" "$split_evidence_json" "$started_at_unix_ns" \
        "$ended_at_unix_ns" "$exit_code" <<'PY'
import json
import sys

(
    records_path,
    phase,
    status,
    model_json,
    workdir,
    log_paths_json,
    split_evidence_json,
    started_at_unix_ns,
    ended_at_unix_ns,
    exit_code,
) = sys.argv[1:]

record = {
    "phase": phase,
    "status": status,
    "model": json.loads(model_json),
    "workdir": workdir,
    "log_paths": json.loads(log_paths_json),
    "split_evidence": json.loads(split_evidence_json),
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
import hashlib
import json
import os
import re
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
split_phases = {
    "dense-split-kv": "dense",
    "recurrent-split-kv": "recurrent",
}
for record in records:
    phase = record.get("phase")
    if phase not in required:
        errors.append(f"unplanned phase record: {phase!r}")
        continue
    if phase in seen:
        errors.append(f"duplicate phase record: {phase}")
    seen.add(phase)
    model = record.get("model")
    split_evidence = record.get("split_evidence")
    if (
        record.get("status") not in {"passed", "failed"}
        or not isinstance(model, dict)
        or not isinstance(model.get("label"), str)
        or not model["label"]
        or not isinstance(model.get("artifact_id"), str)
        or not model["artifact_id"]
        or not isinstance(model.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", model["sha256"])
        or not isinstance(model.get("path"), str)
        or not model["path"]
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
    expected_model_label = split_phases.get(phase)
    if expected_model_label is None:
        if split_evidence is not None:
            errors.append(f"unexpected split evidence for phase: {phase!r}")
    elif record.get("status") == "passed":
        if (
            not isinstance(split_evidence, dict)
            or not isinstance(split_evidence.get("path"), str)
            or not split_evidence["path"]
            or not isinstance(split_evidence.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", split_evidence["sha256"])
        ):
            errors.append(f"missing split evidence for passed phase: {phase!r}")
        else:
            evidence_path = split_evidence["path"]
            try:
                with open(evidence_path, "rb") as evidence_file:
                    raw_evidence = evidence_file.read()
                actual_digest = hashlib.sha256(raw_evidence).hexdigest()
                evidence_payload = json.loads(raw_evidence)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                errors.append(f"unreadable split evidence for phase {phase!r}: {error}")
            else:
                if actual_digest != split_evidence["sha256"]:
                    errors.append(f"split evidence digest mismatch for phase: {phase!r}")
                if (
                    not isinstance(evidence_payload, dict)
                    or evidence_payload.get("kind")
                    != "mesh-llm-two-node-split-readiness"
                    or evidence_payload.get("status") != "ready"
                    or evidence_payload.get("model_label") != expected_model_label
                ):
                    errors.append(f"invalid split evidence payload for phase: {phase!r}")

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

verify_split_phase_evidence() {
    local phase="$1"
    local model_label="$2"
    local phase_dir="$3"
    local evidence_path="${phase_dir}/split-evidence.json"
    local snapshot_dir="${phase_dir}/split-evidence-snapshots"
    local evidence_sha256

    case "$phase" in
        dense-split-kv|recurrent-split-kv) ;;
        *)
            printf 'null\n'
            return 0
            ;;
    esac

    if ! python3 scripts/reconcile-two-node-split-evidence.py \
        --seed-status "$snapshot_dir/seed-status.json" \
        --seed-stages "$snapshot_dir/seed-stages.json" \
        --seed-models "$snapshot_dir/seed-models.json" \
        --worker-status "$snapshot_dir/worker-status.json" \
        --worker-stages "$snapshot_dir/worker-stages.json" \
        --worker-models "$snapshot_dir/worker-models.json" \
        --model-label "$model_label" \
        --verify "$evidence_path" >&2; then
        echo "${phase} did not produce independently verifiable split evidence" >&2
        return 1
    fi
    evidence_sha256="$(fixture_sha256 "$evidence_path")"
    python3 - "$evidence_path" "$evidence_sha256" <<'PY'
import json
import sys

path, sha256 = sys.argv[1:]
print(json.dumps({"path": path, "sha256": sha256}, sort_keys=True, separators=(",", ":")))
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
    local log_paths_json="$3"
    local model_json
    shift 3
    local phase_dir="${PHASE_ROOT}/${phase}"
    local started_at_unix_ns
    local ended_at_unix_ns
    local phase_exit_code
    local phase_status
    local split_evidence_json=null

    if ! ensure_phase_is_planned_once "$phase"; then
        write_phase_manifest failed "$phase" 1 || true
        SUITE_FINALIZED=1
        return 70
    fi
    mkdir -p "$phase_dir"
    echo "=== phase: ${phase} ==="
    model_json="$(fixture_manifest_model "$model_label")"
    started_at_unix_ns="$(phase_now_unix_ns)"
    if MESH_PRODUCT_INTEGRATION_PHASE="$phase" "$@"; then
        phase_exit_code=0
        phase_status=passed
    else
        phase_exit_code=$?
        phase_status=failed
    fi
    if [[ "$phase_exit_code" -eq 0 ]]; then
        if ! split_evidence_json="$(verify_split_phase_evidence "$phase" "$model_label" "$phase_dir")"; then
            phase_exit_code=71
            phase_status=failed
            split_evidence_json=null
        fi
    fi
    ended_at_unix_ns="$(phase_now_unix_ns)"
    append_phase_record "$phase" "$phase_status" "$model_json" "$phase_dir" \
        "$log_paths_json" "$split_evidence_json" "$started_at_unix_ns" "$ended_at_unix_ns" \
        "$phase_exit_code"
    if [[ "$phase_exit_code" -ne 0 ]]; then
        write_phase_manifest failed "$phase" 1 || true
        SUITE_FINALIZED=1
        return "$phase_exit_code"
    fi
    write_phase_manifest in-progress "" 0
}

run_phase dense-standalone dense \
    "[\"$PHASE_ROOT/dense-standalone/server.log\",\"$PHASE_ROOT/dense-standalone/headless.log\"]" env \
    MESH_CI_DEVICE="$DEVICE" \
    MESH_CI_API_PORT=9337 MESH_CI_CONSOLE_PORT=3131 \
    MESH_CI_HEADLESS_API_PORT=9338 MESH_CI_HEADLESS_CONSOLE_PORT=3132 \
    MESH_CI_LOG="$PHASE_ROOT/dense-standalone/server.log" \
    MESH_CI_HEADLESS_LOG="$PHASE_ROOT/dense-standalone/headless.log" \
    scripts/ci-smoke-test.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

run_phase dense-openai-sdk dense \
    "[\"$PHASE_ROOT/dense-openai-sdk/server.log\"]" env \
    MESH_COMPAT_DEVICE="$DEVICE" \
    MESH_COMPAT_API_PORT=9348 MESH_COMPAT_CONSOLE_PORT=3142 \
    MESH_COMPAT_LOG="$PHASE_ROOT/dense-openai-sdk/server.log" \
    scripts/ci-compat-smoke.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

run_phase dense-constrained-tokio-restart dense \
    "[\"$PHASE_ROOT/dense-constrained-tokio-restart/server.log\",\"$PHASE_ROOT/dense-constrained-tokio-restart/headless.log\"]" env \
    MESH_CI_DEVICE="$DEVICE" MESH_TOKIO_STACK_SIZE=2097152 \
    MESH_CI_API_PORT=9347 MESH_CI_CONSOLE_PORT=3141 \
    MESH_CI_HEADLESS_API_PORT=9349 MESH_CI_HEADLESS_CONSOLE_PORT=3143 \
    MESH_CI_LOG="$PHASE_ROOT/dense-constrained-tokio-restart/server.log" \
    MESH_CI_HEADLESS_LOG="$PHASE_ROOT/dense-constrained-tokio-restart/headless.log" \
    scripts/ci-smoke-test.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

run_phase dense-split-kv dense \
    "[\"$PHASE_ROOT/dense-split-kv/dense-seed.log\",\"$PHASE_ROOT/dense-split-kv/dense-worker.log\",\"$PHASE_ROOT/dense-split-kv/dense-client.log\"]" env \
    MESH_TWO_NODE_SPLIT_DEVICE="$DEVICE" \
    MESH_TWO_NODE_SPLIT_MODEL="$DENSE_MODEL" \
    MESH_TWO_NODE_SPLIT_MODEL_LABEL=dense \
    MESH_TWO_NODE_SPLIT_CLIENT_ROUTING=1 \
    MESH_TWO_NODE_SPLIT_WORK_DIR="$PHASE_ROOT/dense-split-kv" \
    scripts/ci-two-node-split-smoke.sh "$MESH_LLM" "$ARTIFACT_DIR" "$DENSE_MODEL"

run_phase recurrent-split-kv recurrent \
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
