#!/usr/bin/env bash
set -euo pipefail

# Supported-families certification battery (issue #1434; tiers dropped 2026-08-25).
#
# Every row of the single manifest gets core certification: single-step,
# chain, and state-handoff lanes. Models with MTP/NextN tensors require the
# native draft sideband and verify it against the target in the correctness
# lanes. Dense rows run them at the first, midpoint, and last interior cuts.
# Hybrid/recurrent rows (sweep_period > 0) run a boundary sweep — one
# representative split layer for every cut offset modulo the family's
# interleaving period.
#
# Models are NEVER cached through GitHub Actions cache. The family-certify
# runner ships a large pre-warmed, read-only HF cache. When HF_CACHE is set,
# model resolution is forced offline so a cache miss fails without attempting
# to mutate the shared NFS cache. Local runs without HF_CACHE may download into
# their normal user cache.
#
# Policy is owned by the versioned JSON manifest and resolved by
# scripts/plan-family-battery.py. The planner enforces the three-lane minimum,
# exact artifact revisions, and deterministic shards before this script loads
# any model.
#
# Usage:
#   scripts/skippy-family-battery.sh [--manifest PATH] [--plan PATH]
#     [--families CSV] [--shard-index N]
#     [--preflight-only] [--skip-build] [--dry-run]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${FAMILY_BATTERY_BIN_DIR:-$ROOT/target/debug}"

MANIFEST="$ROOT/ci/llama-canary/family-certified.json"
POLICY_PLAN=""
SHARD_INDEX=""
SKIP_BUILD=0
DRY_RUN=0
PREFLIGHT_ONLY=0
FAMILY_FILTER=""
SWEEP_MAX_CUTS="${FAMILY_BATTERY_SWEEP_MAX_CUTS:-3}"
STARTUP_TIMEOUT_MIN_SECS="${FAMILY_BATTERY_STARTUP_TIMEOUT_MIN_SECS:-180}"
STARTUP_TIMEOUT_PER_GIB_SECS="${FAMILY_BATTERY_STARTUP_TIMEOUT_PER_GIB_SECS:-10}"
STARTUP_TIMEOUT_MAX_SECS="${FAMILY_BATTERY_STARTUP_TIMEOUT_MAX_SECS:-900}"
CERT_TIMEOUT_MIN_SECS="${FAMILY_BATTERY_CERT_TIMEOUT_MIN_SECS:-1200}"
CERT_TIMEOUT_STARTUP_MULTIPLIER="${FAMILY_BATTERY_CERT_TIMEOUT_STARTUP_MULTIPLIER:-3}"
CERT_TIMEOUT_MAX_SECS="${FAMILY_BATTERY_CERT_TIMEOUT_MAX_SECS:-3600}"
MIN_FREE_GIB="${FAMILY_BATTERY_MIN_FREE_GIB:-5}"
BATTERY_RUN_ID="${FAMILY_BATTERY_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
ARTIFACT_ROOT="${FAMILY_BATTERY_ARTIFACT_ROOT:-$ROOT/target/family-battery}"
ARTIFACT_DIR="$ARTIFACT_ROOT/$BATTERY_RUN_ID"
MODEL_SCAN_DIR="$ARTIFACT_DIR/model-scans"
PREFLIGHT_DIR="$ARTIFACT_DIR/preflight"
CERT_DIR="$ARTIFACT_DIR/certifications"
RESULTS_JSONL="$ARTIFACT_DIR/results.jsonl"
NATIVE_MTP_MODELS_TSV="$ARTIFACT_DIR/native-mtp-models.tsv"
RESOLVED_MANIFEST="$ARTIFACT_DIR/resolved-models.tsv"
SUMMARY_TSV="$ARTIFACT_DIR/summary.tsv"
SUMMARY_MD="$ARTIFACT_DIR/summary.md"
PLANNER="$ROOT/scripts/plan-family-battery.py"
POLICY_PLAN_COPY="$ARTIFACT_DIR/policy-plan.json"

usage() {
  cat >&2 <<'EOF'
usage: scripts/skippy-family-battery.sh [options]

options:
  --manifest PATH           certification manifest;
                            default: ci/llama-canary/family-certified.json
  --plan PATH               consume a previously validated planner result
  --families CSV            run only these exact family labels
  --shard-index N           run only one deterministic shard from the plan
  --preflight-only          resolve, pin, scan, and probe models without certification
  --skip-build              skip the one-time certification binary build;
                            required binaries must already exist
  --dry-run                 print the certification commands only
  -h, --help                show this help
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --manifest) MANIFEST="$2"; shift ;;
    --plan) POLICY_PLAN="$2"; shift ;;
    --families) FAMILY_FILTER="$2"; shift ;;
    --shard-index) SHARD_INDEX="$2"; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

if [[ ! -f "$MANIFEST" ]]; then
  echo "error: manifest not found: $MANIFEST" >&2
  exit 1
fi
if [[ -n "$POLICY_PLAN" && -n "$FAMILY_FILTER" ]]; then
  echo "--families cannot be combined with an existing --plan" >&2
  exit 1
fi
if [[ -n "$SHARD_INDEX" && ! "$SHARD_INDEX" =~ ^[0-9]+$ ]]; then
  echo "--shard-index must be a non-negative integer" >&2
  exit 1
fi

# The family-certify self-hosted runner exports HF_CACHE pointing at its
# pre-warmed Hugging Face cache. Normalize it into the variables `hf` and
# the parity tooling already honor, so downloads only happen on misses.
if [[ -n "${HF_CACHE:-}" ]]; then
  export HF_HOME="$HF_CACHE"
  export HF_HUB_CACHE="$HF_CACHE/hub"
  export HF_HUB_OFFLINE=1
fi

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "required command not found: $1" >&2
    exit 1
  }
}

require_cmd hf
require_cmd jq
require_cmd python3
if [[ ! -x "$PLANNER" ]]; then
  echo "family battery planner is not executable: $PLANNER" >&2
  exit 1
fi

for value in \
  "$SWEEP_MAX_CUTS" \
  "$STARTUP_TIMEOUT_MIN_SECS" \
  "$STARTUP_TIMEOUT_PER_GIB_SECS" \
  "$STARTUP_TIMEOUT_MAX_SECS" \
  "$CERT_TIMEOUT_MIN_SECS" \
  "$CERT_TIMEOUT_STARTUP_MULTIPLIER" \
  "$CERT_TIMEOUT_MAX_SECS" \
  "$MIN_FREE_GIB"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "family battery numeric settings must be non-negative integers" >&2
    exit 1
  fi
done
if (( STARTUP_TIMEOUT_MIN_SECS == 0 || STARTUP_TIMEOUT_MAX_SECS < STARTUP_TIMEOUT_MIN_SECS )); then
  echo "startup timeout bounds are invalid" >&2
  exit 1
fi
if (( CERT_TIMEOUT_MIN_SECS == 0 || CERT_TIMEOUT_STARTUP_MULTIPLIER == 0 || CERT_TIMEOUT_MAX_SECS < CERT_TIMEOUT_MIN_SECS )); then
  echo "certification timeout bounds are invalid" >&2
  exit 1
fi
if [[ -n "$FAMILY_FILTER" && ! "$FAMILY_FILTER" =~ ^[a-zA-Z0-9._-]+(,[a-zA-Z0-9._-]+)*$ ]]; then
  echo "--families must be a comma-separated list of exact family labels" >&2
  exit 1
fi

mkdir -p "$MODEL_SCAN_DIR" "$PREFLIGHT_DIR" "$CERT_DIR"
: > "$RESULTS_JSONL"
printf 'family\tmodel_id\tsource_revision\tmodel_path\tmtp_layers\n' > "$NATIVE_MTP_MODELS_TSV"
printf 'family|repo|source_revision|file|selector|sweep_period|layer_end|notes|target_path|draft_repo|draft_revision|draft_file|draft_path|native_mtp|model_size_bytes|mtp_layers|activation_width|startup_timeout_secs|mmproj_repo|mmproj_revision|mmproj_file|mmproj_path\n' > "$RESOLVED_MANIFEST"

prepare_policy_plan() {
  local plan_args=(
    "$PLANNER"
    --manifest "$MANIFEST"
    --output "$POLICY_PLAN_COPY"
  )
  if [[ -n "$POLICY_PLAN" ]]; then
    if [[ ! -f "$POLICY_PLAN" ]]; then
      echo "policy plan not found: $POLICY_PLAN" >&2
      return 1
    fi
    if [[ "$POLICY_PLAN" != "$POLICY_PLAN_COPY" ]]; then
      cp "$POLICY_PLAN" "$POLICY_PLAN_COPY"
    fi
  else
    if [[ -n "$FAMILY_FILTER" ]]; then
      plan_args+=(--families "$FAMILY_FILTER")
    fi
    # The persistent family runner is offline and read-only. Validate every
    # exact snapshot/file before native compilation when the battery is run
    # directly; the workflow performs the same gate and passes its plan in.
    if [[ -n "${HF_CACHE:-}" ]]; then
      plan_args+=(--check-cache --cache-root "$HF_CACHE")
    fi
    "${plan_args[@]}"
  fi

  python3 - "$MANIFEST" "$POLICY_PLAN_COPY" "$SHARD_INDEX" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, plan_path, shard_index = sys.argv[1:]
manifest_sha = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
core = ["single-step", "chain", "state-handoff"]
if plan.get("schema_version") != 1:
    raise SystemExit("policy plan has an unsupported schema_version")
if plan.get("manifest_sha256") != manifest_sha:
    raise SystemExit("policy plan does not match the checked-in manifest bytes")
if plan.get("required_certification_lanes") != core:
    raise SystemExit("policy plan does not preserve the three-lane certification contract")
if not plan.get("selected_models"):
    raise SystemExit("policy plan selected no models")
if shard_index:
    requested = int(shard_index)
    if not any(shard.get("shard_index") == requested for shard in plan.get("shards", [])):
        raise SystemExit(f"policy plan has no shard index {requested}")
PY
}

prepare_policy_plan

FAILURES=()
TOTAL=0
EXPECTED_TOTAL=0
MM_SMOKE_TOTAL=0
EXPECTED_MM_SMOKE_TOTAL=0
MM_SMOKE_FAILURE_COUNT=0
EXPECTED_FAMILY_COUNT=0
CERT_FAILURE_COUNT=0
PREFLIGHT_FAILURE_COUNT=0
PREFLIGHT_FIRST_TARGET=""

snapshot_revision_from_path() {
  python3 - "$1" <<'PY'
import re
import sys
from pathlib import Path

parts = Path(sys.argv[1]).parts
for index, part in enumerate(parts[:-1]):
    if part == "snapshots" and index + 1 < len(parts):
        revision = parts[index + 1]
        if re.fullmatch(r"[0-9a-f]{40,64}", revision):
            print(revision)
            raise SystemExit(0)
raise SystemExit(1)
PY
}

record_preflight_outcome() {
  local name="$1" family="$2" model_id="$3" status="$4" outcome="$5" note="$6"
  jq -n \
    --arg family "$family" \
    --arg model_id "$model_id" \
    --arg status "$status" \
    --arg outcome "$outcome" \
    --arg note "$note" \
    --arg name "$name" \
    '{family:$family,model_id:$model_id,exit_code:(if $status == "pass" then 0 else 1 end),outcomes:[{name:$name,status:$status,outcome:$outcome,exit_code:(if $status == "pass" then 0 else 1 end),note:$note}]}' \
    >> "$RESULTS_JSONL"
}

# resolve_model REPO FILE [REVISION] -> prints a local path. On the family-certify runner,
# HF_HUB_OFFLINE=1 makes a missing pre-warmed artifact a hard, read-only miss.
# Local runs without HF_CACHE retain the normal hf download behavior.
resolve_model() {
  local repo="$1" file="$2" revision="${3:-}"
  local out raw
  local command=(hf download "$repo" "$file")
  if [[ -n "$revision" ]]; then
    command+=(--revision "$revision")
  fi
  if ! raw="$("${command[@]}" 2>/dev/null)"; then
    return 0
  fi
  out="$(
    printf '%s\n' "$raw" \
      | sed -n \
        -e 's/^path=//p' \
        -e 's/^[[:space:]]*path:[[:space:]]*//p' \
      | tail -n 1
  )"
  if [[ -z "$out" ]]; then
    # newer hub-cli versions print the bare path
    out="$(printf '%s\n' "$raw" | tail -n 1)"
  fi
  printf '%s\n' "$out"
}

# Resolve only the immutable revision checked into the policy manifest. The
# cheap workflow planner has already verified that this exact cache path exists;
# `hf download --revision` remains the canonical cache lookup interface.
resolve_pinned_model() {
  local repo="$1" file="$2" revision="$3"
  local pinned resolved_revision
  pinned="$(resolve_model "$repo" "$file" "$revision")"
  if [[ -z "$pinned" || ! -f "$pinned" ]]; then
    return 1
  fi
  resolved_revision="$(snapshot_revision_from_path "$pinned")" || return 1
  if [[ "$resolved_revision" != "$revision" ]]; then
    return 1
  fi
  printf '%s|%s\n' "$pinned" "$revision"
}

build_certification_binaries() {
  local bins=(skippy-correctness skippy-server skippy-model-package)
  local bin

  if (( DRY_RUN == 1 )); then
    if (( SKIP_BUILD == 0 )); then
      echo "env LLAMA_STAGE_BUILD_DIR='<repo>/.deps/llama-build/build-stage-abi-static' cargo build -p skippy-correctness -p skippy-server -p skippy-model-package"
    fi
    return 0
  fi

  if (( SKIP_BUILD == 0 )); then
    env LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-$ROOT/.deps/llama-build/build-stage-abi-static}" \
      cargo build -p skippy-correctness -p skippy-server -p skippy-model-package
    return 0
  fi

  for bin in "${bins[@]}"; do
    if [[ ! -x "$BIN_DIR/$bin" ]]; then
      echo "--skip-build requires existing binary: $BIN_DIR/$bin" >&2
      return 1
    fi
  done
}

slugify() {
  printf '%s' "$1" | tr '/[:upper:]' '-[:lower:]' | tr -cs 'a-z0-9._-' '-'
}

startup_timeout_for_bytes() {
  local bytes="$1"
  local gib=$(( (bytes + 1073741823) / 1073741824 ))
  local timeout=$(( STARTUP_TIMEOUT_MIN_SECS + gib * STARTUP_TIMEOUT_PER_GIB_SECS ))
  if (( timeout > STARTUP_TIMEOUT_MAX_SECS )); then
    timeout="$STARTUP_TIMEOUT_MAX_SECS"
  fi
  printf '%s\n' "$timeout"
}

cert_timeout_for_startup() {
  local startup_timeout="$1"
  local timeout=$(( CERT_TIMEOUT_MIN_SECS + startup_timeout * CERT_TIMEOUT_STARTUP_MULTIPLIER ))
  if (( timeout > CERT_TIMEOUT_MAX_SECS )); then
    timeout="$CERT_TIMEOUT_MAX_SECS"
  fi
  printf '%s\n' "$timeout"
}

scan_model() {
  local family="$1" target="$2" model_id="$3" source_revision="$4"
  local family_slug scan_json scan_log
  family_slug="$(slugify "$family")"
  scan_json="$MODEL_SCAN_DIR/$family_slug.json"
  scan_log="$MODEL_SCAN_DIR/$family_slug.log"
  MODEL_HAS_MTP=0
  MODEL_SIZE_BYTES=0
  MODEL_MTP_LAYERS=""
  MODEL_LAYER_COUNT=0

  if (( DRY_RUN == 1 )); then
    echo "$BIN_DIR/skippy-model-package inspect '$target' > '$scan_json'"
    return 0
  fi
  if ! "$BIN_DIR/skippy-model-package" inspect "$target" >"$scan_json" 2>"$scan_log"; then
    jq -n \
      --arg family "$family" \
      --arg model_id "$model_id" \
      --arg target "$target" \
      --arg scan_log "$scan_log" \
      '{family:$family,model_id:$model_id,target_model:$target,exit_code:1,outcomes:[{name:"model-scan",status:"fail",outcome:"harness",log:$scan_log}]}' \
      >> "$RESULTS_JSONL"
    FAILURES+=("$family(mtp-scan)")
    return 1
  fi

  MODEL_SIZE_BYTES="$(jq '[.tensors[].byte_size] | add // 0' "$scan_json")"
  MODEL_LAYER_COUNT="$(jq '[.tensors[] | select(.layer_index != null) | .layer_index] | unique | length' "$scan_json")"
  MODEL_MTP_LAYERS="$(jq -r '
    [.tensors[]
      | select(.layer_index != null)
      | {layer: .layer_index, name: (.name | ascii_downcase)}]
    | group_by(.layer)
    | map(select(
        any(.[]; .name | contains(".nextn.eh_proj"))
        and any(.[]; .name | contains(".nextn.enorm"))
        and any(.[]; .name | contains(".nextn.hnorm"))
      ))
    | map(.[0].layer)
    | sort
    | map(tostring)
    | join(",")
  ' "$scan_json")"
  if [[ -n "$MODEL_MTP_LAYERS" ]]; then
    MODEL_HAS_MTP=1
    printf '%s\t%s\t%s\t%s\t%s\n' "$family" "$model_id" "$source_revision" "$target" "$MODEL_MTP_LAYERS" >> "$NATIVE_MTP_MODELS_TSV"
  fi
}

preflight_environment() {
  local model_root="${HF_HOME:-$(dirname "$PREFLIGHT_FIRST_TARGET")}"
  python3 - "$ARTIFACT_DIR" "$model_root" "$MIN_FREE_GIB" "$PREFLIGHT_DIR/environment.json" <<'PY'
import json
import shutil
import socket
import sys
from pathlib import Path

artifact_root, model_root, minimum_gib, output = sys.argv[1:]
minimum_bytes = int(minimum_gib) * 1024**3
filesystems = []
for label, path_text in (("artifacts", artifact_root), ("models", model_root)):
    path = Path(path_text)
    while not path.exists() and path != path.parent:
        path = path.parent
    usage = shutil.disk_usage(path)
    filesystems.append(
        {
            "label": label,
            "path": str(path),
            "free_bytes": usage.free,
            "minimum_free_bytes": minimum_bytes,
            "sufficient": usage.free >= minimum_bytes,
        }
    )

busy_ports = []
for port in range(19000, 20032):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.01)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            busy_ports.append(port)
    except OSError:
        busy_ports.append(port)
    finally:
        sock.close()

report = {
    "filesystems": filesystems,
    "port_range": {"start": 19000, "end": 20031, "busy": busy_ports},
}
Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if any(not item["sufficient"] for item in filesystems) or busy_ports:
    raise SystemExit(1)
PY
}

run_certify() {
  local family="$1" target="$2" model_id="$3" source_revision="$4" split_layer="$5" layer_end="$6" native_mtp="$7"
  local startup_timeout="$8" model_size_bytes="$9" activation_width="${10}"
  # Two distinct interior cut points so the chain lane (exactly two split
  # indexes) always has valid inputs; distinct from each other and from 0.
  local chain_a=$(( layer_end / 3 ))
  local chain_b=$(( ( layer_end * 2 ) / 3 ))
  if (( chain_b == chain_a || chain_a < 1 )); then
    chain_a=1
    chain_b=2
  fi
  TOTAL=$((TOTAL + 1))
  local cert_run_id cert_run_dir exit_code manifest_path cert_timeout
  cert_run_id="$(printf '%03d-%s-split-%s' "$TOTAL" "$(slugify "$family")" "$split_layer")"
  cert_run_dir="$CERT_DIR/$cert_run_id"
  cert_timeout="$(cert_timeout_for_startup "$startup_timeout")"
  echo "==> family-certify: family=$family split=$split_layer mtp=$native_mtp startup_timeout=${startup_timeout}s cert_timeout=${cert_timeout}s model=$(basename "$target")"
  local command=(
    "$ROOT/scripts/family-certify.sh"
    --family "$family"
    --target-model "$target"
    --model-id "$model_id"
    --split-layer "$split_layer"
    --layer-end "$layer_end"
    --splits "$chain_a,$chain_b"
    --activation-width "$activation_width"
    --startup-timeout-secs "$startup_timeout"
    --cert-root "$cert_run_dir"
    --run-id certification
    --require-lanes
    --skip-build
  )
  if (( native_mtp == 1 )); then
    command+=(--require-native-mtp-draft)
  fi
  command+=(--skip-speculative)
  if (( DRY_RUN == 1 )); then
    printf '%q ' "${command[@]}"
    printf '\n'
    return 0
  fi
  exit_code=0
  "$ROOT/scripts/run-command-with-timeout.py" \
    --seconds "$cert_timeout" \
    --label "family certification $family split $split_layer" \
    -- "${command[@]}" || exit_code=$?
  manifest_path=""
  if [[ -d "$cert_run_dir" ]]; then
    manifest_path="$(find "$cert_run_dir" -name manifest.json -type f -print -quit)"
  fi
  if [[ -n "$manifest_path" ]]; then
    jq -c \
      --arg family "$family" \
      --arg model_id "$model_id" \
      --arg source_revision "$source_revision" \
      --argjson split_layer "$split_layer" \
      --argjson model_size_bytes "$model_size_bytes" \
      --argjson activation_width "$activation_width" \
      --argjson startup_timeout_secs "$startup_timeout" \
      --argjson certification_timeout_secs "$cert_timeout" \
      --argjson native_mtp "$native_mtp" \
      --argjson exit_code "$exit_code" \
      '{family:$family,model_id:$model_id,source_revision:$source_revision,split_layer:$split_layer,model_size_bytes:$model_size_bytes,activation_width:$activation_width,startup_timeout_secs:$startup_timeout_secs,certification_timeout_secs:$certification_timeout_secs,native_mtp:($native_mtp == 1),exit_code:$exit_code,manifest:input_filename,outcomes:.commands}' \
      "$manifest_path" >> "$RESULTS_JSONL"
  else
    jq -n \
      --arg family "$family" \
      --arg model_id "$model_id" \
      --arg source_revision "$source_revision" \
      --argjson split_layer "$split_layer" \
      --argjson model_size_bytes "$model_size_bytes" \
      --argjson activation_width "$activation_width" \
      --argjson startup_timeout_secs "$startup_timeout" \
      --argjson certification_timeout_secs "$cert_timeout" \
      --argjson native_mtp "$native_mtp" \
      --argjson exit_code "$exit_code" \
      --arg outcome "$(if (( exit_code == 124 )); then printf timeout; else printf harness; fi)" \
      '{family:$family,model_id:$model_id,source_revision:$source_revision,split_layer:$split_layer,model_size_bytes:$model_size_bytes,activation_width:$activation_width,startup_timeout_secs:$startup_timeout_secs,certification_timeout_secs:$certification_timeout_secs,native_mtp:($native_mtp == 1),exit_code:$exit_code,outcomes:[{name:"certification-manifest",status:"fail",outcome:$outcome,note:(if $outcome == "timeout" then "family-certify exceeded its wall-clock budget before writing a manifest" else "family-certify produced no manifest" end)}]}' \
      >> "$RESULTS_JSONL"
  fi
  if (( exit_code != 0 )); then
    FAILURES+=("$family@split=$split_layer")
    CERT_FAILURE_COUNT=$((CERT_FAILURE_COUNT + 1))
  fi
}

preflight_manifest() {
  local plan="$1"
  [[ -f "$plan" ]] || { echo "missing policy plan: $plan" >&2; exit 1; }

  if [[ -n "$SHARD_INDEX" ]]; then
    EXPECTED_FAMILY_COUNT="$(jq -r --argjson shard_index "$SHARD_INDEX" \
      '.shards[] | select(.shard_index == $shard_index) | (.families | length)' "$plan")"
  else
    EXPECTED_FAMILY_COUNT="$(jq -r '.selected_family_count' "$plan")"
  fi
  if [[ ! "$EXPECTED_FAMILY_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "policy plan has no positive selected family count" >&2
    PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
    return 1
  fi

  while IFS='|' read -r family profile repo source_revision file selector sweep_period layer_end activation_width notes draft_repo draft_revision draft_file expected_model_bytes startup_timeout_override expected_mtp_layers lane_csv _speculative_policy mmproj_repo mmproj_revision mmproj_file; do
    if [[ "$profile" != "full" ]]; then
      echo "the local monolithic battery cannot execute profile $profile for $family" >&2
      exit 1
    fi
    if [[ "$lane_csv" != "single-step,chain,state-handoff" ]]; then
      echo "certified family $family does not preserve the three-lane contract" >&2
      exit 1
    fi

    local target resolved model_id
    model_id="$repo:$selector"
    if (( DRY_RUN == 0 )); then
      if ! resolved="$(resolve_pinned_model "$repo" "$file" "$source_revision")"; then
        echo "failed to resolve and pin $repo/$file from the HF cache or hub" >&2
        FAILURES+=("$family(model-pin)")
        PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
        record_preflight_outcome "model-preflight" "$family" "$model_id" "fail" "harness" "failed to resolve checked-in immutable snapshot $source_revision for $repo/$file"
        continue
      fi
      IFS='|' read -r target _ <<< "$resolved"
      if [[ -z "$PREFLIGHT_FIRST_TARGET" ]]; then
        PREFLIGHT_FIRST_TARGET="$target"
      fi
    else
      target="<hf-cache>/$repo/$file"
      source_revision="dry-run"
    fi
    if ! scan_model "$family" "$target" "$model_id" "$source_revision"; then
      PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
      continue
    fi
    if (( DRY_RUN == 0 )); then
      local actual_mtp_layers=0
      if [[ -n "$MODEL_MTP_LAYERS" ]]; then
        IFS=',' read -r -a mtp_layer_parts <<< "$MODEL_MTP_LAYERS"
        actual_mtp_layers="${#mtp_layer_parts[@]}"
      fi
      if (( MODEL_LAYER_COUNT != layer_end )); then
        echo "policy/runtime layer mismatch for $family: planned $layer_end, scanned $MODEL_LAYER_COUNT" >&2
        FAILURES+=("$family(layer-range)")
        PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
        record_preflight_outcome "model-preflight" "$family" "$model_id" "fail" "model-invalid" "planned runtime range $layer_end does not match scanned layer count $MODEL_LAYER_COUNT"
        continue
      fi
      if (( actual_mtp_layers != expected_mtp_layers )); then
        echo "policy/MTP mismatch for $family: planned $expected_mtp_layers, scanned $actual_mtp_layers" >&2
        FAILURES+=("$family(mtp-policy)")
        PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
        record_preflight_outcome "model-preflight" "$family" "$model_id" "fail" "model-invalid" "planned MTP layer count $expected_mtp_layers does not match scanned complete-head count $actual_mtp_layers"
        continue
      fi
      if (( MODEL_SIZE_BYTES != expected_model_bytes )); then
        echo "policy/model-size mismatch for $family: planned $expected_model_bytes, scanned $MODEL_SIZE_BYTES" >&2
        FAILURES+=("$family(model-size)")
        PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
        record_preflight_outcome "model-preflight" "$family" "$model_id" "fail" "model-invalid" "planned tensor bytes $expected_model_bytes do not match immutable artifact scan $MODEL_SIZE_BYTES"
        continue
      fi
    fi

    # Native MTP is part of the target model. It is exercised through the
    # correctness lanes' draft sideband and target verification, never by
    # reopening the same GGUF as a separate full draft model.
    local draft=""
    draft_repo=""
    draft_revision=""
    draft_file=""

    local startup_timeout
    startup_timeout="${startup_timeout_override:-$(startup_timeout_for_bytes "$MODEL_SIZE_BYTES")}"
    local mmproj_path=""
    if [[ -n "$mmproj_repo" ]]; then
      if (( DRY_RUN == 0 )); then
        if ! resolved="$(resolve_pinned_model "$mmproj_repo" "$mmproj_file" "$mmproj_revision")"; then
          echo "failed to resolve and pin mmproj $mmproj_repo/$mmproj_file" >&2
          FAILURES+=("$family(mmproj-pin)")
          PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
          record_preflight_outcome "model-preflight" "$family" "$model_id" "fail" "harness" "failed to resolve immutable mmproj snapshot for $mmproj_repo/$mmproj_file"
          continue
        fi
        IFS='|' read -r mmproj_path _ <<< "$resolved"
      else
        mmproj_path="<hf-cache>/$mmproj_repo/$mmproj_file"
      fi
    fi
    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
      "$family" "$repo" "$source_revision" "$file" "$selector" "$sweep_period" "$layer_end" "$notes" "$target" \
      "$draft_repo" "$draft_revision" "$draft_file" "$draft" "$MODEL_HAS_MTP" "$MODEL_SIZE_BYTES" "$MODEL_MTP_LAYERS" "$activation_width" "$startup_timeout" \
      "$mmproj_repo" "$mmproj_revision" "$mmproj_file" "$mmproj_path" \
      >> "$RESOLVED_MANIFEST"
    if (( DRY_RUN == 0 )); then
      record_preflight_outcome "model-preflight" "$family" "$model_id" "pass" "pass" "resolved immutable snapshot $source_revision; tensor scan complete"
    fi
  done < <(
    jq -r --arg shard_index "$SHARD_INDEX" '
      (if $shard_index == "" then
         [.selected_models[].family]
       else
         [.shards[]
          | select(.shard_index == ($shard_index | tonumber))
          | .families[]]
       end) as $selected
      | .selected_models[]
      | select(.family as $family | $selected | index($family))
      | [
          .family,
          .profile,
          .artifact.repo,
          .artifact.revision,
          .artifact.files[0],
          .artifact.selector,
          .execution.boundary_sweep_period,
          .execution.layer_end,
          .execution.activation_width,
          .notes,
          (.draft_artifact.repo // ""),
          (.draft_artifact.revision // ""),
          (.draft_artifact.files[0] // ""),
          .resources.estimated_model_bytes,
          (.resources.startup_timeout_secs // ""),
          .execution.mtp_layers,
          (.certification_lanes | join(",")),
          .execution.speculative_policy,
          (.mmproj_artifact.repo // ""),
          (.mmproj_artifact.revision // ""),
          (.mmproj_artifact.files[0] // "")
        ]
      | join("|")
    ' "$plan"
  )

  local resolved_count=$(( $(wc -l < "$RESOLVED_MANIFEST") - 1 ))
  if (( resolved_count != EXPECTED_FAMILY_COUNT )); then
    echo "preflight resolved $resolved_count model rows but policy planned $EXPECTED_FAMILY_COUNT" >&2
    PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
  fi
  local planned_families resolved_families
  planned_families="$(jq -r --arg shard_index "$SHARD_INDEX" '
    if $shard_index == "" then
      [.selected_models[].family]
    else
      [.shards[]
       | select(.shard_index == ($shard_index | tonumber))
       | .families[]]
    end
    | join(",")
  ' "$plan")"
  resolved_families="$(tail -n +2 "$RESOLVED_MANIFEST" | cut -d '|' -f 1 | paste -sd, -)"
  if [[ "$resolved_families" != "$planned_families" ]]; then
    echo "resolved family order does not exactly match the validated policy plan" >&2
    PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
  fi
  if (( DRY_RUN == 1 )); then
    return 0
  fi
  if (( PREFLIGHT_FAILURE_COUNT > 0 )); then
    return 1
  fi
  if ! preflight_environment; then
    record_preflight_outcome "environment-preflight" "battery" "environment" "fail" "harness" "insufficient disk headroom or occupied certification ports; see preflight/environment.json"
    PREFLIGHT_FAILURE_COUNT=$((PREFLIGHT_FAILURE_COUNT + 1))
    return 1
  fi
  record_preflight_outcome "environment-preflight" "battery" "environment" "pass" "pass" "disk headroom and certification port range validated"

}

planned_certification_count() {
  local resolved_manifest="$1"
  local planned=0
  while IFS='|' read -r family _repo _source_revision _file _selector sweep_period layer_end _rest; do
    [[ "$family" == "family" ]] && continue
    local base_split=$(( layer_end / 2 ))
    local first_split=1
    local last_split=$(( layer_end - 1 ))
    planned=$((planned + 1))
    if [[ "$sweep_period" == "0" ]]; then
      if (( first_split != base_split )); then
        planned=$((planned + 1))
      fi
      if (( last_split != base_split && last_split != first_split )); then
        planned=$((planned + 1))
      fi
    else
      local offset cut cuts
      for (( offset = 1; offset <= sweep_period; offset++ )); do
        cuts=0
        for (( cut = offset; cut < layer_end && cuts < SWEEP_MAX_CUTS; cut += sweep_period )); do
          (( cut == base_split )) && continue
          planned=$((planned + 1))
          cuts=$((cuts + 1))
        done
      done
    fi
  done < "$resolved_manifest"
  printf '%s\n' "$planned"
}

# Multimodal smoke lane: exercise the real projector + image prefill path
# through the skippy-server frontend (local monolithic and split stages) using
# the env-gated real-model harness in
# crates/skippy-server/src/frontend/tests/multimodal.rs. Runs once per family
# that pins an mmproj artifact, after its core certification lanes.
run_mmproj_smoke() {
  local family="$1" target="$2" mmproj="$3" model_id="$4" startup_timeout="$5" activation_width="$6" layer_end="$7"
  local smoke_run_id smoke_run_dir exit_code log_path smoke_timeout
  smoke_timeout="$(cert_timeout_for_startup "$startup_timeout")"
  smoke_run_id="$(printf '%03d-%s-mmproj' "$TOTAL" "$(slugify "$family")")"
  smoke_run_dir="$CERT_DIR/$smoke_run_id"
  mkdir -p "$smoke_run_dir"
  log_path="$smoke_run_dir/mmproj-smoke.log"
  echo "==> family-certify mmproj smoke: family=$family model=$(basename "$target") mmproj=$(basename "$mmproj")"
  MM_SMOKE_TOTAL=$((MM_SMOKE_TOTAL + 1))
  if (( DRY_RUN == 1 )); then
    echo "env SKIPPY_MM_MODEL='$target' SKIPPY_MM_PROJECTOR='$mmproj' SKIPPY_MM_IMAGE='$ROOT/ci/llama-canary/fixtures/multimodal-smoke.png' SKIPPY_MM_ACTIVATION_WIDTH='$activation_width' SKIPPY_MM_SPLIT_LAYER='$(( layer_end / 2 ))' cargo test --manifest-path '$ROOT/Cargo.toml' -p skippy-server --lib frontend::tests::multimodal -- --nocapture --test-threads=1"
    return 0
  fi
  exit_code=0
  "$ROOT/scripts/run-command-with-timeout.py" \
    --seconds "$smoke_timeout" \
    --label "mmproj smoke $family" \
    -- env \
      SKIPPY_MM_MODEL="$target" \
      SKIPPY_MM_PROJECTOR="$mmproj" \
      SKIPPY_MM_IMAGE="$ROOT/ci/llama-canary/fixtures/multimodal-smoke.png" \
      SKIPPY_MM_ACTIVATION_WIDTH="$activation_width" \
      SKIPPY_MM_CTX_SIZE=2048 \
      SKIPPY_MM_MAX_TOKENS=16 \
      SKIPPY_MM_N_GPU_LAYERS=999 \
      SKIPPY_MM_SPLIT_LAYER="$(( layer_end / 2 ))" \
      LLAMA_STAGE_BACKEND=metal \
      cargo test --manifest-path "$ROOT/Cargo.toml" -p skippy-server --lib frontend::tests::multimodal -- --nocapture --test-threads=1 \
      >"$log_path" 2>&1 || exit_code=$?
  if (( exit_code != 0 )); then
    echo "mmproj smoke failed for $family; log: $log_path" >&2
    FAILURES+=("$family@mmproj")
    MM_SMOKE_FAILURE_COUNT=$((MM_SMOKE_FAILURE_COUNT + 1))
  fi
  jq -n \
    --arg family "$family" \
    --arg model_id "$model_id" \
    --argjson exit_code "$exit_code" \
    '{family:$family,model_id:$model_id,mmproj_smoke:true,exit_code:$exit_code,outcomes:[{name:"mmproj-smoke",status:(if $exit_code == 0 then "pass" else "fail" end),outcome:(if $exit_code == 0 then "pass" else "fail" end),exit_code:$exit_code,note:"real projector + image prefill through skippy-server frontend (local + split)"}]}' \
    >> "$RESULTS_JSONL"
}

run_resolved_manifest() {
  local resolved_manifest="$1"
  while IFS='|' read -r family repo source_revision file selector sweep_period layer_end _notes target _draft_repo _draft_revision _draft_file _draft native_mtp model_size_bytes _mtp_layers activation_width startup_timeout mmproj_repo mmproj_revision mmproj_file mmproj_path; do
    [[ "$family" == "family" ]] && continue
    local model_id="$repo:$selector"

    # Dense families exercise both endpoint ownership cases plus an ordinary
    # interior handoff. Collapse duplicates for tiny models.
    local base_split=$(( layer_end / 2 ))
    local first_split=1
    local last_split=$(( layer_end - 1 ))
    run_certify "$family" "$target" "$model_id" "$source_revision" "$base_split" "$layer_end" "$native_mtp" "$startup_timeout" "$model_size_bytes" "$activation_width"
    if [[ "$sweep_period" == "0" ]]; then
      if (( first_split != base_split )); then
        run_certify "$family" "$target" "$model_id" "$source_revision" "$first_split" "$layer_end" "$native_mtp" "$startup_timeout" "$model_size_bytes" "$activation_width"
      fi
      if (( last_split != base_split && last_split != first_split )); then
        run_certify "$family" "$target" "$model_id" "$source_revision" "$last_split" "$layer_end" "$native_mtp" "$startup_timeout" "$model_size_bytes" "$activation_width"
      fi
    else
      # Boundary sweep: every cut offset mod the interleaving period, one
      # representative cut each (then every period up to SWEEP_MAX_CUTS cuts),
      # so planner-cut dependence (the B1 bug class) cannot hide.
      local offset cut cuts
      for (( offset = 1; offset <= sweep_period; offset++ )); do
        cuts=0
        for (( cut = offset; cut < layer_end && cuts < SWEEP_MAX_CUTS; cut += sweep_period )); do
          (( cut == base_split )) && continue
          run_certify "$family" "$target" "$model_id" "$source_revision" "$cut" "$layer_end" "$native_mtp" "$startup_timeout" "$model_size_bytes" "$activation_width"
          cuts=$((cuts + 1))
        done
      done
    fi

    if [[ -n "$mmproj_repo" ]]; then
      run_mmproj_smoke "$family" "$target" "$mmproj_path" "$model_id" "$startup_timeout" "$activation_width" "$layer_end"
    fi
  done < "$resolved_manifest"
}

build_certification_binaries
if ! preflight_manifest "$POLICY_PLAN_COPY"; then
  echo "family battery preflight failed; no certification lane was started" >&2
elif (( PREFLIGHT_ONLY == 0 )); then
  EXPECTED_TOTAL="$(planned_certification_count "$RESOLVED_MANIFEST")"
  EXPECTED_MM_SMOKE_TOTAL="$(tail -n +2 "$RESOLVED_MANIFEST" | awk -F'|' '$19 != "" { count += 1 } END { print count + 0 }')"
  run_resolved_manifest "$RESOLVED_MANIFEST"
  if (( TOTAL != EXPECTED_TOTAL )); then
    echo "executed $TOTAL certifications but validated plan requires $EXPECTED_TOTAL" >&2
    FAILURES+=("battery(planned-vs-executed)")
    CERT_FAILURE_COUNT=$((CERT_FAILURE_COUNT + 1))
  fi
  if (( MM_SMOKE_TOTAL != EXPECTED_MM_SMOKE_TOTAL )); then
    echo "executed $MM_SMOKE_TOTAL mmproj smokes but validated plan requires $EXPECTED_MM_SMOKE_TOTAL" >&2
    FAILURES+=("battery(mmproj-planned-vs-executed)")
    CERT_FAILURE_COUNT=$((CERT_FAILURE_COUNT + 1))
  fi
  if (( DRY_RUN == 0 )); then
    actual_result_count="$(jq -s '[.[] | select(.split_layer != null)] | length' "$RESULTS_JSONL")"
    if (( actual_result_count != EXPECTED_TOTAL )); then
      echo "recorded $actual_result_count certification results but validated plan requires $EXPECTED_TOTAL" >&2
      FAILURES+=("battery(result-reconciliation)")
      CERT_FAILURE_COUNT=$((CERT_FAILURE_COUNT + 1))
    fi
  fi
fi

echo
if (( DRY_RUN == 0 )); then
  jq -sr '
    ["family","split_layer","lane","status","outcome","exit_code"],
    (.[] as $row | $row.outcomes[] | [$row.family,($row.split_layer // ""),.name,.status,.outcome,.exit_code])
    | @tsv
  ' "$RESULTS_JSONL" > "$SUMMARY_TSV"
  {
    echo "# Supported-families battery"
    echo
    echo "- Run ID: \`$BATTERY_RUN_ID\`"
    echo "- Policy plan: \`$POLICY_PLAN_COPY\`"
    if [[ -n "$SHARD_INDEX" ]]; then
      echo "- Policy shard: \`$SHARD_INDEX\`"
    fi
    echo "- Certifications: $TOTAL"
    echo "- Planned certifications: $EXPECTED_TOTAL"
    echo "- Multimodal smokes: $MM_SMOKE_TOTAL (planned: $EXPECTED_MM_SMOKE_TOTAL; failures: $MM_SMOKE_FAILURE_COUNT)"
    echo "- Native MTP models: $(( $(wc -l < "$NATIVE_MTP_MODELS_TSV") - 1 ))"
    echo "- Preflight failures: $PREFLIGHT_FAILURE_COUNT"
    echo "- Startup timeout policy: min ${STARTUP_TIMEOUT_MIN_SECS}s + ${STARTUP_TIMEOUT_PER_GIB_SECS}s/GiB, capped at ${STARTUP_TIMEOUT_MAX_SECS}s"
    echo "- Certification wall-clock policy: min ${CERT_TIMEOUT_MIN_SECS}s + ${CERT_TIMEOUT_STARTUP_MULTIPLIER}x startup timeout, capped at ${CERT_TIMEOUT_MAX_SECS}s"
    echo "- Minimum free space: ${MIN_FREE_GIB} GiB"
    echo
    echo "## Typed outcomes"
    echo
    echo "| Outcome | Count |"
    echo "| --- | ---: |"
    jq -sr '[.[].outcomes[]] | group_by(.outcome)[] | "| \(.[0].outcome) | \(length) |"' "$RESULTS_JSONL"
    echo
    echo "- Results: \`$RESULTS_JSONL\`"
    echo "- Lane summary: \`$SUMMARY_TSV\`"
    echo "- Native MTP model inventory: \`$NATIVE_MTP_MODELS_TSV\`"
    echo "- Immutable resolved model manifest: \`$RESOLVED_MANIFEST\`"
    echo "- Validated policy plan: \`$POLICY_PLAN_COPY\`"
    echo "- Environment preflight: \`$PREFLIGHT_DIR/environment.json\`"
    echo "- Certifications and logs: \`$CERT_DIR\`"
  } > "$SUMMARY_MD"
fi

if (( PREFLIGHT_ONLY == 1 )); then
  echo "family battery preflight complete: $PREFLIGHT_FAILURE_COUNT failures"
else
  echo "family battery complete: $((TOTAL - CERT_FAILURE_COUNT))/$TOTAL certifications passed"
fi
echo "artifacts: $ARTIFACT_DIR"
if (( ${#FAILURES[@]} > 0 )); then
  printf 'failed: %s\n' "${FAILURES[@]}"
fi
if (( ${#FAILURES[@]} > 0 || PREFLIGHT_FAILURE_COUNT > 0 )); then
  exit 1
fi
