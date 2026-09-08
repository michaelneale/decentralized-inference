#!/bin/bash
set -euo pipefail

# This script runs inside an HF Job container.
# It clones mesh-llm, builds the splitter, splits the model, validates, and publishes.
#
# Environment variables (set by mesh-llm model-package job spec):
#   SOURCE_REPO, SOURCE_FILE, SOURCE_QUANT, TARGET_REPO, MODEL_ID, SOURCE_REVISION
#   SOURCE_PROJECTOR_FILES — optional newline-delimited repo-relative mmproj GGUFs
#   SOURCE_PIPELINE_TAG — source model pipeline tag for the published model card
#   MESH_LLM_REF — git ref to build from (default: main)
#   CATALOG_CREATE_PR — "true" to open a PR for catalog updates (non-org members)
#   PACKAGE_EXPERIMENTAL — "true" to label the public package as not runtime-certified
#   HF_TOKEN — injected as a secret by HF Jobs
#
# Volumes:
#   /bucket  — writable storage bucket for script and fallback source cache

MESH_LLM_REF="${MESH_LLM_REF:-main}"
SOURCE_REVISION="${SOURCE_REVISION:-main}"
SOURCE_QUANT="${SOURCE_QUANT:-}"
: "${SOURCE_REPO:?SOURCE_REPO is required}"
if [ -z "$SOURCE_QUANT" ] && [[ "${MODEL_ID:-}" == *:* ]]; then
    SOURCE_QUANT="${MODEL_ID##*:}"
fi
if [ -z "$SOURCE_QUANT" ]; then
    echo "ERROR: SOURCE_QUANT is required to resolve the source GGUF without a model volume" >&2
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Layer Package Split Job                                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Source: ${SOURCE_REPO}/${SOURCE_FILE}"
echo "║  Quant:  ${SOURCE_QUANT}"
echo "║  Target: ${TARGET_REPO}"
echo "║  Model:  ${MODEL_ID}"
echo "║  Build:  mesh-llm @ ${MESH_LLM_REF}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Keep executable toolchains/build products on local ephemeral storage:
# HF bucket mounts can be unsuitable for dynamic loader/toolchain execution.
# Package artifacts are also written locally, uploaded one at a time, and
# removed immediately so the job never accumulates a full 400GB+ package.
JOB_WORK_ROOT="${JOB_WORK_ROOT:-/bucket/job-work}"
SAFE_TARGET_REPO="$(printf '%s' "$TARGET_REPO" | tr -c '[:alnum:]._-' '_')"
LOCAL_WORK_DIR="${LOCAL_WORK_DIR:-/tmp/meshllm-layer-job-${SAFE_TARGET_REPO}-$$}"
if [ -z "${JOB_WORK_DIR:-}" ]; then
    JOB_WORK_DIR="${JOB_WORK_ROOT}/${SAFE_TARGET_REPO}-$(date +%Y%m%d%H%M%S)-$$"
    CLEANUP_JOB_WORK_DIR="${CLEANUP_JOB_WORK_DIR:-true}"
else
    CLEANUP_JOB_WORK_DIR="${CLEANUP_JOB_WORK_DIR:-false}"
fi
PACKAGE_DIR="${PACKAGE_DIR:-${LOCAL_WORK_DIR}/package}"
HF_HOME="${HF_HOME:-${JOB_WORK_DIR}/hf-home}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
JOB_TMP_DIR="${JOB_TMP_DIR:-${LOCAL_WORK_DIR}/tmp}"
BUILD_DIR="${BUILD_DIR:-${LOCAL_WORK_DIR}/build}"
TOOL_DIR="${TOOL_DIR:-${LOCAL_WORK_DIR}/tools}"
VENV_DIR="${VENV_DIR:-${LOCAL_WORK_DIR}/venv}"
ARTIFACT_UPLOAD_SCRIPT="${ARTIFACT_UPLOAD_SCRIPT:-${LOCAL_WORK_DIR}/upload-package-artifact.py}"
ARTIFACT_UPLOAD_HOOK="${ARTIFACT_UPLOAD_HOOK:-${LOCAL_WORK_DIR}/upload-package-artifact.sh}"
CARGO_HOME="${CARGO_HOME:-${LOCAL_WORK_DIR}/cargo-home}"
RUSTUP_HOME="${RUSTUP_HOME:-${LOCAL_WORK_DIR}/rustup-home}"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${LOCAL_WORK_DIR}/cargo-target}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LOCAL_WORK_DIR}/xdg-cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${LOCAL_WORK_DIR}/pip-cache}"
BUILD_TMP_DIR="${BUILD_TMP_DIR:-${LOCAL_WORK_DIR}/tmp}"
TMPDIR="$BUILD_TMP_DIR"
TEMP="$BUILD_TMP_DIR"
TMP="$BUILD_TMP_DIR"
export JOB_WORK_DIR PACKAGE_DIR HF_HOME HF_HUB_CACHE HF_XET_CACHE VENV_DIR ARTIFACT_UPLOAD_SCRIPT
export TMPDIR TEMP TMP CARGO_HOME RUSTUP_HOME CARGO_TARGET_DIR XDG_CACHE_HOME PIP_CACHE_DIR

cleanup_job_work_dir() {
    if [ -n "${LOCAL_WORK_DIR:-}" ]; then
        echo "Cleaning local work dir: ${LOCAL_WORK_DIR}"
        rm -rf "$LOCAL_WORK_DIR" || true
    fi
    if [ "${CLEANUP_JOB_WORK_DIR}" = "true" ] && [ -n "${JOB_WORK_DIR:-}" ]; then
        echo "Cleaning job work dir: ${JOB_WORK_DIR}"
        rm -rf "$JOB_WORK_DIR" || true
    fi
}
trap cleanup_job_work_dir EXIT

log_storage_snapshot() {
    local label="$1"
    echo "  Storage snapshot (${label}):"
    df -h / /bucket "$PACKAGE_DIR" "$TMPDIR" 2>/dev/null || true
    echo "  Mounts (${label}):"
    mount | grep -E ' on / | on /bucket ' || true
}

on_error() {
    local status=$?
    local line=${BASH_LINENO[0]:-unknown}
    local command=${BASH_COMMAND:-unknown}
    echo "ERROR: split job command failed at line ${line} with status ${status}: ${command}" >&2
    log_storage_snapshot "error" >&2 || true
    exit "$status"
}
trap on_error ERR

start_heartbeat() {
    local label="$1"
    (
        while true; do
            sleep "${JOB_HEARTBEAT_SECONDS:-60}"
            echo "  Heartbeat (${label}) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
            df -h / /bucket "$PACKAGE_DIR" "$TMPDIR" 2>/dev/null || true
            if [ -d "$PACKAGE_DIR" ]; then
                du -sh "$PACKAGE_DIR" 2>/dev/null || true
            fi
            if [ -d "$HF_HUB_CACHE" ]; then
                du -sh "$HF_HUB_CACHE" 2>/dev/null || true
            fi
        done
    ) &
    HEARTBEAT_PID=$!
}

stop_heartbeat() {
    if [ -n "${HEARTBEAT_PID:-}" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
        HEARTBEAT_PID=""
    fi
}

mkdir -p "$PACKAGE_DIR" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$JOB_TMP_DIR" "$TOOL_DIR" \
    "$CARGO_HOME" "$RUSTUP_HOME" "$CARGO_TARGET_DIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" \
    "$BUILD_TMP_DIR"

format_bytes() {
    python3 - "$1" <<'PYTHON'
import sys
value = float(int(sys.argv[1]))
for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
    if value < 1024 or unit == "PiB":
        if unit == "B":
            print(f"{int(value)} {unit}")
        else:
            print(f"{value:.1f} {unit}")
        break
    value /= 1024
PYTHON
}

estimate_bucket_workspace_bytes() {
    python3 - "$1" <<'PYTHON'
import sys
source = int(sys.argv[1])
# Source and package artifacts are not meant to accumulate in the bucket. This
# estimate is retained only as a fallback-source-cache warning when /source is
# unavailable.
headroom = 32 * 1024 ** 3
print(source + headroom)
PYTHON
}

# ─── Build tools ──────────────────────────────────────────────────────────
echo "=== [1/9] Installing build dependencies ==="
apt-get update -qq && apt-get install -y -qq \
    cmake git curl build-essential pkg-config libssl-dev \
    python3-pip python3-venv > /dev/null 2>&1
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "=== [2/9] Installing Rust ==="
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null 2>&1
# shellcheck source=/dev/null
source "${CARGO_HOME}/env"

echo "=== [3/9] Cloning mesh-llm and building skippy-model-package ==="
git clone --filter=blob:none https://github.com/Mesh-LLM/mesh-llm.git "$BUILD_DIR"
cd "$BUILD_DIR"
if git ls-remote --exit-code --heads origin "$MESH_LLM_REF" >/dev/null 2>&1 || \
   git ls-remote --exit-code --tags origin "$MESH_LLM_REF" >/dev/null 2>&1; then
    git fetch --depth 1 origin "$MESH_LLM_REF"
    git checkout --detach FETCH_HEAD
elif git cat-file -e "$MESH_LLM_REF^{commit}" 2>/dev/null; then
    git checkout --detach "$MESH_LLM_REF"
else
    git fetch --depth 1 origin "$MESH_LLM_REF"
    git checkout --detach FETCH_HEAD
fi

# Full clone needed for git-am patches in prepare-llama
sed -i 's/--filter=blob:none //' scripts/prepare-llama.sh
echo "  Running prepare-llama.sh..."
scripts/prepare-llama.sh pinned 2>&1 | tail -5
echo "  Running build-llama.sh..."
scripts/build-llama.sh 2>&1 | tail -5

# Locate the llama.cpp build directory (build-llama.sh puts it here)
LLAMA_BUILD_DIR=".deps/llama-build/build-stage-abi-cpu"
echo "  Verifying llama.cpp build at $LLAMA_BUILD_DIR..."
find "$LLAMA_BUILD_DIR" -name "*.a" 2>/dev/null | head -10 || echo "  WARNING: no .a files found"

# Build the splitter binary
echo "  Building skippy-model-package..."
SKIPPY_LLAMA_BUILD_DIR="$LLAMA_BUILD_DIR" \
    cargo build --release -p skippy-model-package 2>&1 | tail -20
SLICER="${CARGO_TARGET_DIR}/release/skippy-model-package"
if [ ! -f "$SLICER" ]; then
    echo "ERROR: Build failed — binary not found at $SLICER"
    echo "Retrying with full output..."
    SKIPPY_LLAMA_BUILD_DIR=.deps/llama.cpp/build-stage-abi-static \
        cargo build --release -p skippy-model-package 2>&1
    exit 1
fi
cp "$SLICER" "${TOOL_DIR}/skippy-model-package"
SLICER="${TOOL_DIR}/skippy-model-package"
chmod +x "$SLICER"
cd /
rm -rf "$BUILD_DIR" "$CARGO_TARGET_DIR" "$CARGO_HOME" "$RUSTUP_HOME"
TMPDIR="$JOB_TMP_DIR"
TEMP="$JOB_TMP_DIR"
TMP="$JOB_TMP_DIR"
export TMPDIR TEMP TMP
echo "  ✓ Built: $SLICER"
echo "  Root filesystem after build cleanup:"
df -h / || true

echo "  Preparing Hugging Face uploader..."
python3 -m venv "$VENV_DIR" > /dev/null
"$VENV_DIR/bin/pip" install -q huggingface_hub
"$VENV_DIR/bin/python3" << 'PYTHON'
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(os.environ["TARGET_REPO"], exist_ok=True)
PYTHON
cat > "$ARTIFACT_UPLOAD_SCRIPT" <<'PYTHON'
from huggingface_hub import HfApi
from pathlib import Path
import os
import time

path = Path(os.environ["SKIPPY_PACKAGE_ARTIFACT_PATH"])
relative = os.environ["SKIPPY_PACKAGE_ARTIFACT_RELATIVE_PATH"]
target_repo = os.environ["TARGET_REPO"]
max_attempts = int(os.environ.get("ARTIFACT_UPLOAD_ATTEMPTS", "4"))

api = HfApi(token=os.environ["HF_TOKEN"])
last_error = None
for attempt in range(1, max_attempts + 1):
    try:
        api.upload_file(
            repo_id=target_repo,
            path_or_fileobj=str(path),
            path_in_repo=relative,
            repo_type="model",
            commit_message=f"Add package artifact {relative}",
        )
        last_error = None
        break
    except Exception as err:
        last_error = err
        if attempt == max_attempts:
            break
        delay = min(60, 5 * attempt)
        print(
            f"  Upload failed for {relative} on attempt {attempt}/{max_attempts}: {err}. "
            f"Retrying in {delay}s...",
            flush=True,
        )
        time.sleep(delay)
if last_error is not None:
    raise last_error
size = path.stat().st_size
path.unlink()
print(f"  Uploaded and removed {relative} ({size} bytes)")
PYTHON
cat > "$ARTIFACT_UPLOAD_HOOK" <<'BASH'
#!/bin/bash
set -euo pipefail
"${VENV_DIR}/bin/python3" "${ARTIFACT_UPLOAD_SCRIPT}"
BASH
chmod +x "$ARTIFACT_UPLOAD_HOOK"

# ─── Split ────────────────────────────────────────────────────────────────
echo ""
echo "=== [4/9] Splitting model ==="
if [ "$SOURCE_REVISION" = "main" ]; then
    SOURCE_REF="${SOURCE_REPO}:${SOURCE_QUANT}"
else
    SOURCE_REF="${SOURCE_REPO}@${SOURCE_REVISION}:${SOURCE_QUANT}"
fi
echo "  Source ref: $SOURCE_REF"
if [ -n "${SOURCE_TOTAL_BYTES:-}" ]; then
    echo "  Source bytes: $SOURCE_TOTAL_BYTES"
    ESTIMATED_BUCKET_BYTES="$(estimate_bucket_workspace_bytes "$SOURCE_TOTAL_BYTES")"
    echo "  Estimated fallback /bucket cache needed: $(format_bytes "$ESTIMATED_BUCKET_BYTES")"
fi
MOUNTED_SOURCE_PATH="/source/${SOURCE_FILE}"
if [ -f "$MOUNTED_SOURCE_PATH" ]; then
    WRITE_PACKAGE_INPUT="$MOUNTED_SOURCE_PATH"
    WRITE_PACKAGE_IDENTITY_ARGS=(
        --model-id "$MODEL_ID"
        --source-repo "$SOURCE_REPO"
        --source-revision "$SOURCE_REVISION"
        --source-file "$SOURCE_FILE"
    )
    echo "  Source mount: $MOUNTED_SOURCE_PATH"
else
    WRITE_PACKAGE_INPUT="$SOURCE_REF"
    WRITE_PACKAGE_IDENTITY_ARGS=()
    echo "  Source mount: not available; falling back to Hugging Face cache download"
fi
WRITE_PACKAGE_PROJECTOR_ARGS=()
while IFS= read -r PROJECTOR_FILE; do
    if [ -z "$PROJECTOR_FILE" ]; then
        continue
    fi
    MOUNTED_PROJECTOR_PATH="/source/${PROJECTOR_FILE}"
    if [ -f "$MOUNTED_PROJECTOR_PATH" ]; then
        PROJECTOR_PATH="$MOUNTED_PROJECTOR_PATH"
    else
        echo "  Projector mount missing; downloading ${PROJECTOR_FILE} at ${SOURCE_REVISION}"
        PROJECTOR_PATH="$("$VENV_DIR/bin/python3" - "$PROJECTOR_FILE" <<'PYTHON'
from huggingface_hub import hf_hub_download
import os
import sys

print(hf_hub_download(
    repo_id=os.environ["SOURCE_REPO"],
    filename=sys.argv[1],
    revision=os.environ["SOURCE_REVISION"],
    cache_dir=os.environ["HF_HUB_CACHE"],
    token=os.environ.get("HF_TOKEN"),
))
PYTHON
)"
    fi
    echo "  Projector: $PROJECTOR_PATH"
    WRITE_PACKAGE_PROJECTOR_ARGS+=(--projector "$PROJECTOR_PATH")
done <<< "${SOURCE_PROJECTOR_FILES:-}"
echo "  Hugging Face cache: $HF_HUB_CACHE"
echo "  Package workspace: $PACKAGE_DIR"
echo "  Temporary workspace: $TMPDIR"
log_storage_snapshot "before write-package"
ROOT_FS="$(df -P / | awk 'NR==2 {print $1}')"
PACKAGE_FS="$(df -P "$PACKAGE_DIR" | awk 'NR==2 {print $1}')"
if [ -n "$ROOT_FS" ] && [ "$ROOT_FS" = "$PACKAGE_FS" ]; then
    echo "WARNING: package workspace is on the container root filesystem; very large splits may hit the HF Jobs 50G ephemeral storage limit." >&2
fi
if [ -n "${ESTIMATED_BUCKET_BYTES:-}" ]; then
    PACKAGE_AVAILABLE_BYTES="$(df -Pk "$PACKAGE_DIR" | awk 'NR==2 {printf "%.0f", $4 * 1024}')"
    if [ -n "$PACKAGE_AVAILABLE_BYTES" ] && [ "$PACKAGE_AVAILABLE_BYTES" -gt 0 ] && \
        [ "$PACKAGE_AVAILABLE_BYTES" -lt "$ESTIMATED_BUCKET_BYTES" ]; then
        echo "WARNING: package workspace has $(format_bytes "$PACKAGE_AVAILABLE_BYTES") available, below estimated need $(format_bytes "$ESTIMATED_BUCKET_BYTES")." >&2
    fi
fi
echo "  Starting write-package at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
start_heartbeat "write-package"
set +e
time "$SLICER" write-package "$WRITE_PACKAGE_INPUT" \
    --out-dir "$PACKAGE_DIR" \
    --after-artifact-command "$ARTIFACT_UPLOAD_HOOK" \
    "${WRITE_PACKAGE_PROJECTOR_ARGS[@]}" \
    "${WRITE_PACKAGE_IDENTITY_ARGS[@]}"
WRITE_PACKAGE_STATUS=$?
set -e
stop_heartbeat
if [ "$WRITE_PACKAGE_STATUS" -ne 0 ]; then
    echo "ERROR: write-package failed with status $WRITE_PACKAGE_STATUS" >&2
    log_storage_snapshot "write-package failed" >&2 || true
    exit "$WRITE_PACKAGE_STATUS"
fi
echo "  Finished write-package at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_storage_snapshot "after write-package"

SOURCE_IDENTITY="$(python3 -c "import json, os; m=json.load(open(os.path.join(os.environ['PACKAGE_DIR'], 'model-package.json'))); s=m['source_model']; print(s.get('canonical_ref') or s.get('primary_file') or s['sha256'])")"
echo "  Source identity: $SOURCE_IDENTITY"

LAYER_COUNT="$(python3 -c "import json, os; m=json.load(open(os.path.join(os.environ['PACKAGE_DIR'], 'model-package.json'))); print(m['layer_count'])")"
TOTAL_SIZE="$(python3 -c "import json, os; m=json.load(open(os.path.join(os.environ['PACKAGE_DIR'], 'model-package.json'))); print(sum(int(a['byte_size']) for a in m['artifact_catalog']['entries']))")"
TOTAL_SIZE_LABEL="$(format_bytes "$TOTAL_SIZE")"
echo "  ✓ Split into $LAYER_COUNT layers; artifacts uploaded incrementally (${TOTAL_SIZE_LABEL} total)"

# ─── Verify manifest ──────────────────────────────────────────────────────
echo ""
echo "=== [5/9] Verifying package manifest ==="
"$VENV_DIR/bin/python3" << 'PYTHON'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["PACKAGE_DIR"]) / "model-package.json"
manifest = json.loads(manifest_path.read_text())
required = manifest["artifact_catalog"]["entries"]
missing = [
    artifact
    for artifact in required
    if not artifact.get("id")
    or not artifact.get("path")
    or not artifact.get("sha256")
    or not isinstance(artifact.get("byte_size"), int)
    or artifact["byte_size"] <= 0
]
if missing:
    raise SystemExit(f"manifest contains {len(missing)} incomplete artifact catalog entries")
print(f"  ✓ Manifest records {len(required)} uploaded artifacts")
PYTHON

# ─── Publish ──────────────────────────────────────────────────────────────
echo ""
echo "=== [6/9] Publishing to HuggingFace ==="
"$VENV_DIR/bin/python3" << PYTHON
from huggingface_hub import HfApi
import os, json
from pathlib import Path

api = HfApi(token=os.environ['HF_TOKEN'])
target_repo = os.environ['TARGET_REPO']
source_repo = os.environ['SOURCE_REPO']
model_id = os.environ.get('MODEL_ID', '')
manifest_path = Path(os.environ['PACKAGE_DIR']) / 'model-package.json'

api.upload_file(
    repo_id=target_repo,
    path_or_fileobj=str(manifest_path),
    path_in_repo='model-package.json',
    repo_type='model',
    commit_message=f'Add layer package manifest from {source_repo} ({model_id})',
)

# Print summary
manifest = json.load(open(manifest_path))
print(f'  ✓ Published: https://huggingface.co/{target_repo}')
print(f'    Model:  {manifest["model_id"]}')
print(f'    Layers: {manifest["layer_count"]}')
print(f'    Schema: {manifest["schema_version"]}')
PYTHON

# ─── Update catalog ───────────────────────────────────────────────────────
echo ""
echo "=== [7/9] Updating meshllm/catalog ==="
"$VENV_DIR/bin/python3" << 'PYTHON'
from huggingface_hub import HfApi
import os, json, tempfile

api = HfApi(token=os.environ['HF_TOKEN'])
source_repo = os.environ['SOURCE_REPO']
target_repo = os.environ['TARGET_REPO']
source_file = os.environ['SOURCE_FILE']
source_revision = os.environ.get('SOURCE_REVISION', 'main')
model_id = os.environ.get('MODEL_ID', '')
package_dir = os.environ['PACKAGE_DIR']

# Read manifest for metadata
manifest = json.load(open(os.path.join(package_dir, 'model-package.json')))
layer_count = manifest['layer_count']

# Determine catalog entry path: entries/<owner>/<repo-name>.json
owner, repo_name = source_repo.split('/', 1)
entry_path = f"entries/{owner}/{repo_name}.json"

# Try to fetch existing entry
catalog_repo = "meshllm/catalog"
try:
    existing_path = api.hf_hub_download(
        repo_id=catalog_repo,
        filename=entry_path,
        repo_type="dataset",
    )
    entry = json.load(open(existing_path))
except Exception:
    # Create new entry
    entry = {"schema_version": 1, "source_repo": source_repo, "variants": {}}

# Build variant name from source file stem (not MODEL_ID).
# For "UD-Q4_K_XL/Qwen3-32B-UD-Q4_K_XL-00001-of-00002.gguf" → "Qwen3-32B-UD-Q4_K_XL"
import re
file_stem = source_file.split('/')[-1].replace('.gguf', '')
# Strip shard suffix like "-00001-of-00002"
variant_name = re.sub(r'-\d{5}-of-\d{5}$', '', file_stem)

package_entry = {
    "type": "layer-package",
    "repo": target_repo,
    "layer_count": layer_count,
    "source_revision": source_revision,
}

source_entry = {
    "repo": source_repo,
    "file": source_file,
    "revision": source_revision,
}

# Handle both dict-style and list-style variants
variants = entry.get("variants", {})
if isinstance(variants, dict):
    # Dict-keyed by variant name (existing catalog format)
    if variant_name in variants:
        packages = variants[variant_name].get("packages", [])
        packages = [p for p in packages if p.get("repo") != target_repo]
        packages.append(package_entry)
        variants[variant_name]["source"] = source_entry
        variants[variant_name]["packages"] = packages
    else:
        variants[variant_name] = {
            "source": source_entry,
            "curated": {
                "name": variant_name,
                "size": f"{layer_count} layers",
                "description": f"Layer package for {model_id}",
            },
            "packages": [package_entry],
        }
    entry["variants"] = variants
else:
    # List-style (fallback)
    existing_variant = None
    for v in variants:
        if v.get("curated", {}).get("name") == variant_name:
            existing_variant = v
            break
    if existing_variant:
        packages = existing_variant.get("packages", [])
        packages = [p for p in packages if p.get("repo") != target_repo]
        packages.append(package_entry)
        existing_variant["source"] = source_entry
        existing_variant["packages"] = packages
    else:
        variants.append({
            "source": source_entry,
            "curated": {
                "name": variant_name,
                "size": f"{layer_count} layers",
                "description": f"Layer package for {model_id}",
            },
            "packages": [package_entry],
        })

# Write and upload
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(entry, f, indent=2)
    tmp_path = f.name

create_pr = os.environ.get('CATALOG_CREATE_PR', 'false').lower() == 'true'

api.upload_file(
    repo_id=catalog_repo,
    path_or_fileobj=tmp_path,
    path_in_repo=entry_path,
    repo_type="dataset",
    commit_message=f"Add layer package for {model_id} ({target_repo})",
    create_pr=create_pr,
)
print(f"  ✓ Catalog updated: {catalog_repo}/{entry_path}")
print(f"    Variant: {variant_name}")
print(f"    Package: {target_repo} ({layer_count} layers)")
PYTHON

# ─── Model Card ────────────────────────────────────────────────────────────
echo ""
echo "=== [8/9] Uploading model card ==="
"$VENV_DIR/bin/python3" << 'PYTHON'
from huggingface_hub import HfApi
from pathlib import Path
import hashlib
import json
import os

package_dir = Path(os.environ["PACKAGE_DIR"])
manifest_path = package_dir / "model-package.json"
manifest = json.loads(manifest_path.read_text())

source_repo = os.environ["SOURCE_REPO"]
source_file = os.environ["SOURCE_FILE"]
source_revision = os.environ.get("SOURCE_REVISION", "main")
target_repo = os.environ["TARGET_REPO"]
model_id = os.environ.get("MODEL_ID", manifest.get("model_id", target_repo))
mesh_llm_ref = os.environ.get("MESH_LLM_REF", "main")
source_pipeline_tag = os.environ.get("SOURCE_PIPELINE_TAG", "text-generation").strip()
if not source_pipeline_tag:
    source_pipeline_tag = "text-generation"
experimental = os.environ.get("PACKAGE_EXPERIMENTAL", "false").lower() == "true"
experimental_tag = "- experimental\n" if experimental else ""
experimental_warning = (
    "> [!WARNING]\n"
    "> **Experimental package:** artifact integrity may be validated, but runtime, "
    "split-correctness, and multimodal certification are still pending. This package "
    "is not discoverable through `meshllm/catalog@main` until its Hugging Face catalog "
    "PR is reviewed and merged.\n\n"
    if experimental
    else ""
)
api = HfApi(token=os.environ["HF_TOKEN"])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def fmt_bytes(size: int) -> str:
    value = float(size)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024

def artifact_bytes(artifact: dict) -> int:
    return int(artifact.get("artifact_bytes") or 0)

def md_cell(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")

def link(label: str, url: str) -> str:
    return f"[{md_cell(label)}]({url})"

def code(value) -> str:
    return f"`{md_cell(value)}`"

def yaml_quote(value: str) -> str:
    return json.dumps(value)

def card_value(info, key: str):
    card_data = getattr(info, "card_data", None)
    if card_data is None:
        return None
    if isinstance(card_data, dict):
        return card_data.get(key)
    return getattr(card_data, key, None)

def first_model_id(value):
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        return value.get("id") or value.get("modelId")
    return value if isinstance(value, str) and value else None

def resolve_upstream_license():
    try:
        source_info = api.model_info(source_repo, revision=source_revision)
        source_license = card_value(source_info, "license")
        if source_license:
            return str(source_license), source_repo

        base_repo = first_model_id(card_value(source_info, "base_model"))
        if base_repo:
            base_info = api.model_info(base_repo)
            base_license = card_value(base_info, "license")
            if base_license:
                return str(base_license), base_repo
    except Exception as error:
        print(f"  WARNING: could not resolve upstream license metadata: {error}")
    return None, None

def infer_model_family(name: str) -> str:
    lowered = name.lower()
    for family in ["Qwen3", "Qwen2.5", "DeepSeek", "Kimi", "Gemma", "GLM", "Llama"]:
        if family.lower() in lowered:
            return family
    return name.split("-")[0] if name else "Unknown"

def infer_parameter_scale(name: str) -> str:
    import re
    match = re.search(r"(?i)(\d+(?:\.\d+)?[BM](?:-A\d+(?:\.\d+)?B)?)", name)
    return match.group(1) if match else "not recorded"

def infer_quantization(name: str, source_path: str) -> str:
    import re
    combined = f"{name}/{source_path}"
    patterns = [
        r"UD-Q\d+_[A-Z]+(?:_[A-Z]+)?",
        r"Q\d+_[A-Z]+(?:_[A-Z]+)?",
        r"IQ\d+_[A-Z]+(?:_[A-Z]+)?",
        r"BF16",
        r"F16",
    ]
    for pattern in patterns:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            return match.group(0)
    return "not recorded"

shared = manifest.get("shared", {})
layers = manifest.get("layers", [])
projectors = manifest.get("projectors", [])
manifest_hash = sha256(manifest_path)
total_bytes = sum(artifact_bytes(artifact) for artifact in shared.values())
total_bytes += sum(artifact_bytes(layer) for layer in layers)
total_bytes += sum(artifact_bytes(projector) for projector in projectors)

source_model = manifest.get("source_model", {})
display_name = source_model.get("distribution_id") or model_id
model_family = infer_model_family(display_name)
parameter_scale = infer_parameter_scale(display_name)
quantization = infer_quantization(display_name, source_file)
source_path = source_model.get("path") or f"/hf-cache/{source_file}"
activation_width = manifest.get("activation_width") or "not recorded"
skippy_abi = manifest.get("skippy_abi_version") or "not recorded"
source_sha = source_model.get("sha256") or "not recorded"
canonical_ref = source_model.get("canonical_ref") or f"{source_repo}@{source_revision}/{source_file}"
upstream_license, license_source_repo = resolve_upstream_license()
license_frontmatter = (
    f"license: {yaml_quote(upstream_license)}\n" if upstream_license else ""
)

file_rows = [
    ("Manifest", "model-package.json", "Package schema, source identity, checksums", manifest_hash),
]
for label, key in [
    ("Metadata", "metadata"),
    ("Embeddings", "embeddings"),
    ("Output head", "output"),
]:
    artifact = shared.get(key)
    if artifact:
        file_rows.append((
            label,
            artifact.get("path", f"shared/{key}.gguf"),
            f"{artifact.get('tensor_count', 'unknown')} tensors, {fmt_bytes(artifact_bytes(artifact))}",
            artifact.get("sha256", "not recorded"),
        ))
if layers:
    layer_bytes = sum(artifact_bytes(layer) for layer in layers)
    layer_tensors = sum(int(layer.get("tensor_count") or 0) for layer in layers)
    file_rows.append((
        "Transformer layers",
        "layers/layer-*.gguf",
        f"{len(layers)} layer artifacts, {layer_tensors} tensors, {fmt_bytes(layer_bytes)}",
        "see model-package.json",
    ))
for projector in projectors:
    file_rows.append((
        "Projector",
        projector.get("path", "projectors/projector.gguf"),
        f"{projector.get('kind', 'multimodal')} projector, {fmt_bytes(artifact_bytes(projector))}",
        projector.get("sha256", "not recorded"),
    ))

rows = [
    ("Source model", link(source_repo, f"https://huggingface.co/{source_repo}")),
    ("Model id", code(model_id)),
    ("Family", model_family),
    ("Parameter scale", parameter_scale),
    ("Quantization", code(quantization)),
    ("Layer count", manifest.get("layer_count", len(layers))),
    ("Activation width", activation_width),
    ("Package size", fmt_bytes(total_bytes)),
    ("Source file", code(source_file)),
    ("Package repo", link(target_repo, f"https://huggingface.co/{target_repo}")),
]
if upstream_license and license_source_repo:
    rows.append((
        "License",
        f"{code(upstream_license)} from "
        f"{link(license_source_repo, f'https://huggingface.co/{license_source_repo}')}",
    ))

readme = f"""---
library_name: mesh-llm
{license_frontmatter}base_model:
- {yaml_quote(source_repo)}
pipeline_tag: {yaml_quote(source_pipeline_tag)}
tags:
- gguf
- mesh-llm
- layer-package
- skippy
- distributed-inference
- local-inference
- openai-compatible
{experimental_tag}---

<div align="center">
  <a href="https://www.meshllm.cloud">
    <img src="https://meshllm.cloud/assets/images/jelly-logo-wordmark.png" alt="Mesh LLM" width="220">
  </a>

  <h1>{display_name}</h1>

  <p>
    <strong>Distributed GGUF inference package for Mesh LLM</strong>
  </p>

  <p>
    <a href="https://www.meshllm.cloud"><img alt="Website" src="https://img.shields.io/badge/Website-meshllm.cloud-111111?style=for-the-badge"></a>
    <a href="https://github.com/Mesh-LLM/mesh-llm"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Mesh--LLM-24292f?style=for-the-badge&logo=github"></a>
    <a href="https://discord.gg/rs6fmc63eN"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white"></a>
  </p>
</div>

{experimental_warning}GGUF layer package for running **{display_name}** across a local Mesh LLM cluster.

This package is derived from [{source_repo}](https://huggingface.co/{source_repo}) and keeps the original GGUF distribution split into per-layer artifacts for distributed inference.

## Highlights

| Run locally | Pool multiple machines | OpenAI-compatible | Package variant |
|---|---|---|---|
| Private inference on your hardware | Split layers across peers | Serve `/v1/chat/completions` locally | `{quantization}` layer package |

## Model Overview

| Property | Value |
|---|---|
"""

for key, value in rows:
    readme += f"| **{md_cell(key)}** | {md_cell(value)} |\n"

readme += f"""
## Recommended Use

- Local and private inference with Mesh LLM.
- Multi-machine serving when the full GGUF is too large for one host.
- OpenAI-compatible chat/completions workflows through Mesh LLM's local API.

For upstream architecture details, chat template guidance, sampling recommendations, license terms, and benchmark notes, see the source model card: [{source_repo}](https://huggingface.co/{source_repo}).

## Quickstart

```bash
# Run this on each machine that should contribute memory/compute.
mesh-llm serve --model "{target_repo}" --split
```

```bash
# Check the mesh and discover the OpenAI-compatible model name.
curl -s http://localhost:3131/api/status
curl -s http://localhost:3131/v1/models
```

```bash
# Send an OpenAI-compatible chat request.
curl -s http://localhost:3131/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{model_id}",
    "messages": [{{"role": "user", "content": "Write a tiny hello-world function in Rust."}}],
    "max_tokens": 128
  }}'
```

## Package Variant

| Property | Value |
|---|---|
"""

for key, value in [
    ("Format", code(manifest.get("format", "layer-package"))),
    ("Canonical source ref", code(canonical_ref)),
    ("Source revision", code(source_revision)),
    ("Source SHA-256", code(source_sha)),
    ("Skippy ABI", code(skippy_abi)),
    ("Package manifest SHA-256", code(manifest_hash)),
]:
    readme += f"| **{md_cell(key)}** | {md_cell(value)} |\n"

readme += f"""
## What Is Included

| Artifact | Path | Contents | SHA-256 |
|---|---|---|---|
"""

for label, path, contents, checksum in file_rows:
    readme += f"| {md_cell(label)} | {code(path)} | {md_cell(contents)} | {code(checksum)} |\n"

readme += f"""
## Validation

Generated by the Mesh LLM HF Jobs splitter from `mesh-llm` ref `{mesh_llm_ref}`.
Each artifact is checksummed as it is written, uploaded to this repository, and removed from the job workspace before the next artifact is produced.

```bash
skippy-model-package write-package "{source_path}" --out-dir "{package_dir}"
```

## Links

- Source model: [{source_repo}](https://huggingface.co/{source_repo})
- Mesh LLM website: [meshllm.cloud](https://www.meshllm.cloud)
- Mesh LLM: [github.com/Mesh-LLM/mesh-llm](https://github.com/Mesh-LLM/mesh-llm)
- Discord: [discord.gg/rs6fmc63eN](https://discord.gg/rs6fmc63eN)
- Package catalog: [meshllm/catalog](https://huggingface.co/datasets/meshllm/catalog)
- Package format: [layer-package-repos.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/specs/layer-package-repos.md)
"""

Path("/tmp/README.md").write_text(readme)

api.upload_file(
    path_or_fileobj="/tmp/README.md",
    path_in_repo="README.md",
    repo_id=target_repo,
    repo_type="model",
)
print("  ✓ Model card uploaded")
PYTHON

# ─── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=== [9/9] Done ==="
echo ""
echo "  Published:  https://huggingface.co/${TARGET_REPO}"
echo "  Layers:     ${LAYER_COUNT}"
echo "  Total size: ${TOTAL_SIZE_LABEL}"
echo ""
echo "  Use with mesh-llm:"
echo "    mesh-llm serve --model ${TARGET_REPO} --split"
