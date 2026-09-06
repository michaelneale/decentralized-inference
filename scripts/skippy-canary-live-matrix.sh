#!/usr/bin/env bash
# skippy-canary-live-matrix.sh - executable live proof for parity model_pins.
#
# For every runnable parity row carrying a model_pin (the minimum live
# package-v2 two-node matrix), this script:
#   1. resolves the pinned GGUF from the local HF cache (hf download with
#      an exact --revision is only a miss backstop),
#   2. verifies size and sha256 against the immutable pin,
#   3. writes a source-complete package-v2 with skippy-model-package
#      write-package (provenance flags from the pin),
#   4. independently verifies the package with verify-package-v2 against
#      the original GGUF,
#   5. runs scripts/ci-two-node-split-smoke.sh against the package dir
#      (package-v2 serving path) with the row's expected payload kind.
#
# A failure in any row fails the matrix (exit 1) and the run's evidence
# root (SKIPPY_CANARY_LIVE_MATRIX_ROOT, default
# target/family-battery/$FAMILY_BATTERY_RUN_ID) receives per-row logs so
# the canary battery-mode repair loop can reuse them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${SKIPPY_PARITY_MANIFEST:-$ROOT/docs/skippy/llama-parity-candidates.json}"
EVIDENCE_ROOT="${SKIPPY_CANARY_LIVE_MATRIX_ROOT:-$ROOT/target/family-battery/${FAMILY_BATTERY_RUN_ID:-manual}}"
WORK_ROOT="${SKIPPY_CANARY_LIVE_MATRIX_WORK_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/skippy-live-matrix.XXXXXX")}"
CTX_SIZE="${SKIPPY_CANARY_LIVE_MATRIX_CTX_SIZE:-2048}"
READINESS_TIMEOUT_SECONDS="${SKIPPY_CANARY_LIVE_MATRIX_READINESS_TIMEOUT_SECONDS:-300}"
STAGE_SERVER_BIN="${STAGE_SERVER_BIN:-}"
MESH_LLM_BIN="${MESH_LLM_BIN:-}"
PKG_TOOL_BIN="${SKIPPY_CANARY_LIVE_MATRIX_PKG_TOOL:-}"
SPLIT_SMOKE_SCRIPT="${SKIPPY_CANARY_LIVE_MATRIX_SPLIT_SMOKE:-$ROOT/scripts/ci-two-node-split-smoke.sh}"
HF_DOWNLOAD_BIN="${SKIPPY_CANARY_LIVE_MATRIX_HF_DOWNLOAD:-hf}"
LIMIT="${SKIPPY_CANARY_LIVE_MATRIX_LIMIT:-}"

usage() {
  cat >&2 <<'EOF'
usage: scripts/skippy-canary-live-matrix.sh [--dry-run] [--prepare] [--model NAME]

Runs the minimum live package-v2 two-node matrix for parity model_pin rows.
--model executes exactly the named row (llama_model); the battery-mode
repair loop uses it to prove the coverage-expansion target's new pin before
the repair can report success. A filter matching no runnable row fails.

--prepare builds this run's exact producers before any live row: the mesh-llm
host binary (cargo build -p mesh-llm) and the patched native runtime bundle
(scripts/package-native-runtime.sh --build --backend, packaged straight into
the host binary's directory so the two-node smoke loads this run's llama
tree, never a cached one), then writes producer-provenance.json into the
matrix evidence root. The backend is explicit: metal on the arm64
family-certify runner (SKIPPY_CANARY_LIVE_MATRIX_BACKEND).

Environment:
  SKIPPY_PARITY_MANIFEST     parity candidates manifest
  STAGE_SERVER_BIN           skippy-server binary (skippy-package + skippy-model-package
                             are built from this repo when unset)
  MESH_LLM_BIN               mesh-llm binary; required (two-node split smoke)
  HF_HOME                    Hugging Face cache root for pinned GGUFs
  SKIPPY_CANARY_LIVE_MATRIX_LIMIT   run at most N rows (bounded canary use)
  SKIPPY_CANARY_LIVE_MATRIX_BACKEND native runtime backend for --prepare (default metal)
  SKIPPY_CANARY_LIVE_MATRIX_PKG_TOOL    package tool override (tests)
  SKIPPY_CANARY_LIVE_MATRIX_SPLIT_SMOKE split smoke script override (tests)
  SKIPPY_CANARY_LIVE_MATRIX_HF_DOWNLOAD hf download command override (tests)
EOF
}

DRY_RUN=0
PREPARE=0
MODEL_FILTER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --prepare) PREPARE=1; shift ;;
    --model) [[ $# -ge 2 ]] || { echo "--model requires a value" >&2; exit 2; }
             MODEL_FILTER="$2"; shift 2 ;;
    --model=*) MODEL_FILTER="${1#--model=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$EVIDENCE_ROOT/live-matrix" "$WORK_ROOT"

write_producer_provenance() {
  # Record exactly which producers this matrix ran against: Rust HEAD,
  # upstream + patched llama SHAs, host binary sha256, and the native
  # runtime manifest identity (runtime id, skippy_abi, backend). The
  # manifest is read from THIS run's unique bundle directory by the
  # explicit backend's runtime id — never discovered by find, so a stale
  # decoy manifest cannot be attested.
  local provenance="$EVIDENCE_ROOT/live-matrix/producer-provenance.json"
  local bundle_root="${MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR:?MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR must be set (--prepare)}"
  # Runtime id follows package-native-runtime.sh's target_platform naming:
  # darwin-aarch64-metal, linux-x86_64-cuda, ...
  local os_id arch_id
  os_id="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch_id="$(uname -m)"
  [[ "$arch_id" == "arm64" ]] && arch_id="aarch64"
  local runtime_id="meshllm-native-runtime-${os_id}-${arch_id}-${SKIPPY_CANARY_LIVE_MATRIX_BACKEND:-metal}"
  local runtime_manifest="$bundle_root/$runtime_id/manifest.json"
  if [[ ! -f "$runtime_manifest" ]]; then
    echo "cannot record producer provenance: no manifest for $runtime_id under $bundle_root" >&2
    return 1
  fi
  RUST_HEAD="$(git -C "$ROOT" rev-parse HEAD)" \
  LLAMA_UPSTREAM_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["build"]["llama_upstream_sha"])' "$runtime_manifest")" \
  LLAMA_PATCHED_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["build"]["llama_patched_sha"])' "$runtime_manifest")" \
  RUNTIME_MANIFEST="$runtime_manifest" \
  MESH_LLM_SHA="$(shasum -a 256 "$MESH_LLM_BIN" | awk '{print $1}')" \
  python3 - <<'PY' >"$provenance"
import json, os
runtime = json.load(open(os.environ["RUNTIME_MANIFEST"]))
print(json.dumps({
    "rust_head": os.environ["RUST_HEAD"],
    "mesh_llm_bin": os.environ.get("MESH_LLM_BIN", ""),
    "mesh_llm_sha256": os.environ["MESH_LLM_SHA"],
    "llama_upstream_sha": os.environ["LLAMA_UPSTREAM_SHA"],
    "llama_patched_sha": os.environ["LLAMA_PATCHED_SHA"],
    "native_runtime": {
        "manifest": os.environ["RUNTIME_MANIFEST"],
        "id": runtime["runtime"]["id"],
        "skippy_abi": runtime["runtime"]["skippy_abi"],
        "backend": runtime["build"]["backend"],
        "platform": runtime["build"]["platform"],
        "llama_patch_digest": runtime["build"]["llama_patch_digest"],
    },
}, indent=2))
PY
  echo "producer provenance written: $provenance"
}

if [[ "$PREPARE" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "--prepare is incompatible with --dry-run" >&2
    exit 2
  fi
  BACKEND="${SKIPPY_CANARY_LIVE_MATRIX_BACKEND:-metal}"
  MESH_LLM_BIN="${MESH_LLM_BIN:-$ROOT/target/debug/mesh-llm}"
  echo "=== preparing exact producers (backend: $BACKEND) ==="
  if [[ "${SKIPPY_CANARY_LIVE_MATRIX_PREPARE_SKIP_BUILD:-}" != "1" ]]; then
    (cd "$ROOT" && cargo build -p mesh-llm -p skippy-model-package -p skippy-server) >&2
  fi
  MESH_LLM_BIN="$(cd "$(dirname "$MESH_LLM_BIN")" && pwd)/$(basename "$MESH_LLM_BIN")"
  # Package into a unique clean directory under this matrix work root (a
  # shared native-runtimes dir can hold stale runtime directories, and the
  # smoke's auto-resolution would attest the wrong manifest). The exact
  # bundle dir is exported so ci-two-node-split-smoke.sh loads only this
  # run's runtime.
  RUNTIME_BUNDLE_ROOT="$WORK_ROOT/native-runtimes"
  rm -rf "$RUNTIME_BUNDLE_ROOT"
  mkdir -p "$RUNTIME_BUNDLE_ROOT"
  if [[ "${SKIPPY_CANARY_LIVE_MATRIX_PREPARE_SKIP_BUILD:-}" != "1" ]]; then
    "$ROOT/scripts/package-native-runtime.sh" --build --backend "$BACKEND" \
      --out "$RUNTIME_BUNDLE_ROOT" >&2
  fi
  export MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$RUNTIME_BUNDLE_ROOT"
  export MESH_LLM_BIN
  if ! write_producer_provenance; then
    exit 1
  fi
fi

echo "=== Skippy canary live matrix (package-v2 two-node rows) ==="
echo "  manifest:  $MANIFEST"
echo "  evidence:  $EVIDENCE_ROOT/live-matrix"
echo "  work root: $WORK_ROOT"

if [[ "$DRY_RUN" -eq 0 ]]; then
  command -v "$HF_DOWNLOAD_BIN" >/dev/null 2>&1 || { echo "hf CLI is required ($HF_DOWNLOAD_BIN)" >&2; exit 1; }
  [[ -n "$MESH_LLM_BIN" && -x "$MESH_LLM_BIN" ]] || {
    echo "MESH_LLM_BIN must point at an executable mesh-llm binary" >&2; exit 1; }
fi

build_tool() {
  local bin="$1"
  if [[ -x "$bin" ]]; then printf '%s\n' "$bin"; return 0; fi
  cargo build -p "$2" >&2 || return 1
  printf '%s\n' "$ROOT/target/debug/$bin"
}

ROWS_JSON="$(python3 - "$MANIFEST" "$MODEL_FILTER" <<'PY'
import json, sys
rows = []
for row in json.loads(open(sys.argv[1]).read()).get("candidates", []):
    pin = row.get("model_pin")
    if not pin:
        continue
    if row.get("status") not in ("certified", "candidate", "candidate_stateful"):
        continue
    if sys.argv[2] and row["llama_model"] != sys.argv[2]:
        continue
    rows.append({
        "llama_model": row["llama_model"],
        "expected_payload_kind": ("kv-recurrent" if row.get("recurrent") == "all" else "kv-dense"),
        **pin,
    })
rows.sort(key=lambda r: r["llama_model"])
print(json.dumps(rows))
PY
)"

ROW_COUNT="$(python3 -c 'import json,sys; print(len(json.loads(sys.stdin.read())))' <<<"$ROWS_JSON")"
echo "  rows:      $ROW_COUNT"
if [[ -n "$MODEL_FILTER" ]]; then
  echo "  model:     $MODEL_FILTER"
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  python3 -c 'import json,sys
for r in json.loads(sys.stdin.read()):
    print("  - {}: {}@{} {}".format(r["llama_model"], r["repo"], r["revision"][:12], r["file"]))' <<<"$ROWS_JSON"
  echo "dry run: no rows executed"
  exit 0
fi
if [[ "$ROW_COUNT" -eq 0 ]]; then
  echo "model filter matched no runnable model_pin row: ${MODEL_FILTER:-<none>}" >&2
  exit 1
fi

if [[ -n "$LIMIT" ]]; then
  ROWS_JSON="$(python3 -c 'import json,sys; print(json.dumps(json.loads(sys.stdin.read())[:int(sys.argv[1])]))' <<<"$ROWS_JSON" "$LIMIT")"
  ROW_COUNT="$(python3 -c 'import json,sys; print(len(json.loads(sys.stdin.read())))' <<<"$ROWS_JSON")"
  echo "  limited to: $ROW_COUNT rows"
fi

if [[ -z "$STAGE_SERVER_BIN" ]]; then
  STAGE_SERVER_BIN="$(cargo build -p skippy-model-package -p skippy-server >&2 && printf '%s\n' "$ROOT/target/debug/skippy-server")"
fi
if [[ -z "$PKG_TOOL_BIN" ]]; then
  PKG_TOOL_BIN="$ROOT/target/debug/skippy-model-package"
  if [[ ! -x "$PKG_TOOL_BIN" ]]; then
    cargo build -p skippy-model-package >&2
  fi
fi
export SKIPPY_SERVER_BIN="$STAGE_SERVER_BIN"

declare -a FAILED_ROWS=()
ROW_INDEX=0
while IFS= read -r ROW; do
  ROW_INDEX=$((ROW_INDEX + 1))
  read -r MODEL_NAME REPO REVISION FILE SIZE SHA PAYLOAD_KIND <<<"$(python3 -c 'import json,sys
r = json.loads(sys.stdin.read())
print(r["llama_model"], r["repo"], r["revision"], r["file"], r["size_bytes"], r["blob_sha256"], r["expected_payload_kind"])' <<<"$ROW")"
  # Evidence/work dirs exist before ANY per-row validation can fail: a
  # classified failure must always be recorded (set -u would otherwise
  # abort on the unbound ROW_DIR).
  ROW_DIR="$EVIDENCE_ROOT/live-matrix/$MODEL_NAME"
  PKG_DIR="$WORK_ROOT/$MODEL_NAME"
  mkdir -p "$ROW_DIR"
  # The selector is carried by the model_pin and joined against
  # family-certified.json by validate_pin_manifest_join — never re-derived
  # from the filename here.
  SELECTOR="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["selector"])' <<<"$ROW")"
  if [[ -z "$SELECTOR" ]]; then
    echo "row $MODEL_NAME: model_pin carries no selector (join failure — run validate)" | tee -a "$ROW_DIR/row.log"
    FAILED_ROWS+=("$MODEL_NAME:selector")
    continue
  fi
  MODEL_ID="$REPO:$SELECTOR"
  echo
  echo "--- [$ROW_INDEX/$ROW_COUNT] $MODEL_NAME ($PAYLOAD_KIND) ---"

  if ! GGUF_PATH="$("$HF_DOWNLOAD_BIN" download "$REPO" "$FILE" --revision "$REVISION" 2>>"$ROW_DIR/download.log" | tail -n 1)" \
      || [[ -z "$GGUF_PATH" || ! -f "$GGUF_PATH" ]]; then
    echo "row $MODEL_NAME: download failed (see $ROW_DIR/download.log)" | tee -a "$ROW_DIR/row.log"
    FAILED_ROWS+=("$MODEL_NAME:download")
    continue
  fi
  ACTUAL_SIZE="$(stat -f%z "$GGUF_PATH" 2>/dev/null || stat -c%s "$GGUF_PATH")"
  if [[ "$ACTUAL_SIZE" != "$SIZE" ]]; then
    echo "row $MODEL_NAME: pinned size mismatch ($ACTUAL_SIZE != $SIZE)" | tee -a "$ROW_DIR/row.log"
    FAILED_ROWS+=("$MODEL_NAME:size")
    continue
  fi
  ACTUAL_SHA="$(shasum -a 256 "$GGUF_PATH" | awk '{print $1}')"
  if [[ "$ACTUAL_SHA" != "$SHA" ]]; then
    echo "row $MODEL_NAME: pinned sha256 mismatch" | tee -a "$ROW_DIR/row.log"
    FAILED_ROWS+=("$MODEL_NAME:sha256")
    continue
  fi
  echo "verified pinned source: $FILE ($SIZE bytes, sha256 ${SHA:0:16}…)"

  rm -rf "$PKG_DIR"
  if ! "$PKG_TOOL_BIN" write-package "$GGUF_PATH" \
      --model-id "$MODEL_ID" \
      --out-dir "$PKG_DIR" \
      --source-repo "$REPO" --source-revision "$REVISION" --source-file "$FILE" \
      >"$ROW_DIR/write-package.log" 2>&1; then
    echo "row $MODEL_NAME: write-package failed (see $ROW_DIR/write-package.log)" | tee -a "$ROW_DIR/row.log"
    FAILED_ROWS+=("$MODEL_NAME:write-package")
    continue
  fi
  if ! "$PKG_TOOL_BIN" verify-package-v2 "$PKG_DIR" --source "$GGUF_PATH" --source-file "$FILE" \
      >"$ROW_DIR/verify-package-v2.log" 2>&1; then
    echo "row $MODEL_NAME: verify-package-v2 failed (see $ROW_DIR/verify-package-v2.log)" | tee -a "$ROW_DIR/row.log"
    FAILED_ROWS+=("$MODEL_NAME:verify-package-v2")
    continue
  fi
  echo "package-v2 written and independently verified"

  if ! MESH_TWO_NODE_SPLIT_MODEL="$PKG_DIR" \
      MESH_TWO_NODE_SPLIT_MODEL_LABEL="$MODEL_NAME" \
      MESH_TWO_NODE_SPLIT_CTX_SIZE="$CTX_SIZE" \
      MESH_TWO_NODE_SPLIT_EXPECTED_EXACT_PAYLOAD_KIND="$PAYLOAD_KIND" \
      MESH_TWO_NODE_SPLIT_READINESS_TIMEOUT_SECONDS="$READINESS_TIMEOUT_SECONDS" \
      MESH_TWO_NODE_SPLIT_WORK_DIR="$ROW_DIR/split" \
      "$SPLIT_SMOKE_SCRIPT" "$MESH_LLM_BIN" "$(dirname "$MESH_LLM_BIN")" "$PKG_DIR" \
      >"$ROW_DIR/two-node-split.log" 2>&1; then
    echo "row $MODEL_NAME: two-node split smoke failed (see $ROW_DIR/two-node-split.log)" | tee -a "$ROW_DIR/row.log"
    FAILED_ROWS+=("$MODEL_NAME:two-node")
    continue
  fi
  echo "row $MODEL_NAME: PASS" | tee -a "$ROW_DIR/row.log"
done < <(python3 -c 'import json,sys
for r in json.loads(sys.stdin.read()): print(json.dumps(r))' <<<"$ROWS_JSON")

echo
if [[ "${#FAILED_ROWS[@]}" -gt 0 ]]; then
  echo "live matrix FAILED rows:"
  for row in "${FAILED_ROWS[@]}"; do echo "  - $row"; done
  exit 1
fi
echo "live matrix passed: $ROW_COUNT/$ROW_COUNT rows"
