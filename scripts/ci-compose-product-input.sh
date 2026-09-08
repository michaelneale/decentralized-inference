#!/usr/bin/env bash
set -euo pipefail

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
: "${INPUT_HOST_INPUT_DIR:?INPUT_HOST_INPUT_DIR is required}"
: "${INPUT_RUNTIME_INPUT_DIR:?INPUT_RUNTIME_INPUT_DIR is required}"
: "${INPUT_OUTPUT_DIR:?INPUT_OUTPUT_DIR is required}"
: "${INPUT_BACKEND:?INPUT_BACKEND is required}"
: "${INPUT_BINARY_NAME:?INPUT_BINARY_NAME is required}"
: "${INPUT_READINESS_SMOKE:?INPUT_READINESS_SMOKE is required}"
INPUT_ATTESTATION_PUBLIC_KEY_FILE="${INPUT_ATTESTATION_PUBLIC_KEY_FILE:-}"
INPUT_ATTESTATION_VERIFIER="${INPUT_ATTESTATION_VERIFIER:-}"

if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
elif command -v python >/dev/null 2>&1; then
    python_bin="python"
else
    echo "python3 or python is required to compose a product input" >&2
    exit 1
fi

to_shell_path() {
    local path="${1%$'\r'}"
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -u "$path"
    else
        printf '%s\n' "$path"
    fi
}

to_workflow_path() {
    local path="$1"
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -m "$path"
    else
        printf '%s\n' "$path"
    fi
}

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "$path" ]]; then
        echo "$label is missing: $path" >&2
        exit 1
    fi
}

require_nonempty_file() {
    local label="$1"
    local path="$2"
    if [[ ! -s "$path" ]]; then
        echo "$label is missing or empty: $path" >&2
        exit 1
    fi
}

canonical_paths=()
while IFS= read -r path; do
    canonical_paths+=("$(to_shell_path "$path")")
done < <(
    "$python_bin" - \
        "$GITHUB_WORKSPACE" \
        "$INPUT_HOST_INPUT_DIR" \
        "$INPUT_RUNTIME_INPUT_DIR" \
        "$INPUT_OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path


def resolve_in_workspace(workspace: Path, raw: str, *, require_dir: bool) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = candidate.resolve(strict=False)
    try:
        candidate.relative_to(workspace)
    except ValueError as error:
        raise SystemExit(
            f"CI artifact path escapes GITHUB_WORKSPACE: {raw} -> {candidate}"
        ) from error
    if require_dir and not candidate.is_dir():
        raise SystemExit(f"CI producer input is not a directory: {candidate}")
    return candidate


def overlaps(left: Path, right: Path) -> bool:
    return (
        left == right
        or left in right.parents
        or right in left.parents
    )


workspace = Path(sys.argv[1]).resolve(strict=True)
host_input = resolve_in_workspace(workspace, sys.argv[2], require_dir=True)
runtime_input = resolve_in_workspace(workspace, sys.argv[3], require_dir=True)
output = resolve_in_workspace(workspace, sys.argv[4], require_dir=False)

if output == workspace:
    raise SystemExit(f"product output cannot be GITHUB_WORKSPACE: {output}")
for label, producer_input in (
    ("host", host_input),
    ("runtime", runtime_input),
):
    if overlaps(output, producer_input):
        raise SystemExit(
            f"product output overlaps {label} producer input: "
            f"{output} and {producer_input}"
        )

print(host_input)
print(runtime_input)
print(output)
PY
)

if [[ "${#canonical_paths[@]}" -ne 3 ]]; then
    echo "failed to canonicalize CI artifact paths" >&2
    exit 1
fi

host_input_dir="${canonical_paths[0]}"
runtime_input_dir="${canonical_paths[1]}"
output_dir="${canonical_paths[2]}"
host="$host_input_dir/$INPUT_BINARY_NAME"
host_imports="$host_input_dir/host-imports.json"
host_checksum="$host_input_dir/$INPUT_BINARY_NAME.sha256"

GITHUB_OUTPUT="$(to_shell_path "$GITHUB_OUTPUT")"
if [[ -n "$INPUT_ATTESTATION_PUBLIC_KEY_FILE" ]]; then
    INPUT_ATTESTATION_PUBLIC_KEY_FILE="$(
        to_shell_path "$INPUT_ATTESTATION_PUBLIC_KEY_FILE"
    )"
fi
if [[ -n "$INPUT_ATTESTATION_VERIFIER" ]]; then
    INPUT_ATTESTATION_VERIFIER="$(to_shell_path "$INPUT_ATTESTATION_VERIFIER")"
fi

require_file "immutable host" "$host"
chmod +x "$host"
require_nonempty_file "host import report" "$host_imports"
require_nonempty_file "host checksum" "$host_checksum"
"$python_bin" scripts/verify-checksum-sidecar.py "$host"

if [[ -n "$INPUT_ATTESTATION_PUBLIC_KEY_FILE" ]]; then
    attestation_verifier="${INPUT_ATTESTATION_VERIFIER:-$host_input_dir/release-attestation-verifier}"
    verifier_checksum="$attestation_verifier.sha256"
    require_nonempty_file \
        "release attestation public key" \
        "$INPUT_ATTESTATION_PUBLIC_KEY_FILE"
    require_file "release attestation verifier" "$attestation_verifier"
    require_nonempty_file "release attestation verifier checksum" "$verifier_checksum"
    "$python_bin" scripts/verify-checksum-sidecar.py \
        "$attestation_verifier"
    chmod +x "$attestation_verifier"
    "$attestation_verifier" release-attestation inspect \
        --binary "$host" \
        --public-key-file "$INPUT_ATTESTATION_PUBLIC_KEY_FILE" \
        --json
elif [[ -n "$INPUT_ATTESTATION_VERIFIER" ]]; then
    echo "INPUT_ATTESTATION_VERIFIER requires INPUT_ATTESTATION_PUBLIC_KEY_FILE" >&2
    exit 1
fi

rm -rf -- "$output_dir"
mkdir -p "$output_dir/native-runtimes"
cp "$host" "$output_dir/$INPUT_BINARY_NAME"
chmod +x "$output_dir/$INPUT_BINARY_NAME"
cp "$host_imports" "$output_dir/host-imports.json"

runtime_archives=()
while IFS= read -r archive; do
    runtime_archives+=("$archive")
done < <(find "$runtime_input_dir" -type f -name '*.tar.gz' -print)
runtime_sidecars=()
while IFS= read -r sidecar; do
    runtime_sidecars+=("$sidecar")
done < <(find "$runtime_input_dir" -type f -name '*.tar.gz.sha256' -print)
if [[ "${#runtime_archives[@]}" -gt 1 ]]; then
    echo "expected at most one runtime archive; found ${#runtime_archives[@]}" >&2
    exit 1
elif [[ "${#runtime_archives[@]}" -eq 1 ]]; then
    expected_sidecar="${runtime_archives[0]}.sha256"
    if [[ "${#runtime_sidecars[@]}" -ne 1 || "${runtime_sidecars[0]}" != "$expected_sidecar" ]]; then
        echo "expected exactly one checksum sidecar for ${runtime_archives[0]}; found ${#runtime_sidecars[@]}" >&2
        exit 1
    fi
    scripts/verify-native-runtime-package.sh "${runtime_archives[0]}"
    "$python_bin" scripts/safe-extract-tar.py \
        "${runtime_archives[0]}" \
        "$output_dir/native-runtimes"
else
    if [[ "${#runtime_sidecars[@]}" -ne 0 ]]; then
        echo "runtime checksum sidecar exists without a runtime archive" >&2
        exit 1
    fi
    runtime_dirs=()
    while IFS= read -r manifest; do
        runtime_dirs+=("$(dirname "$manifest")")
    done < <(
        find "$runtime_input_dir" \
            -mindepth 2 \
            -maxdepth 2 \
            -type f \
            -name manifest.json \
            -print
    )
    if [[ "${#runtime_dirs[@]}" -ne 1 ]]; then
        echo "expected exactly one extracted runtime; found ${#runtime_dirs[@]}" >&2
        exit 1
    fi
    cp -a "${runtime_dirs[0]}" "$output_dir/native-runtimes/"
fi

composed_runtime_dirs=()
while IFS= read -r manifest; do
    composed_runtime_dirs+=("$(dirname "$manifest")")
done < <(
    find "$output_dir/native-runtimes" \
        -mindepth 2 \
        -maxdepth 2 \
        -type f \
        -name manifest.json \
        -print
)
if [[ "${#composed_runtime_dirs[@]}" -ne 1 ]]; then
    echo "expected exactly one composed runtime; found ${#composed_runtime_dirs[@]}" >&2
    exit 1
fi
runtime_dir="${composed_runtime_dirs[0]}"
scripts/verify-native-runtime-package.sh "$runtime_dir"

version="${INPUT_VERSION:-}"
if [[ -z "$version" ]]; then
    version="$(
        "$python_bin" - "$runtime_dir/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["runtime"]["mesh_version"])
PY
    )"
fi
version="${version#v}"
host_version_output="$("$output_dir/$INPUT_BINARY_NAME" --version)"
host_version="$(awk '{print $NF}' <<<"$host_version_output")"
# Non-release hosts carry semver build metadata (`+g<sha>[.dirty]`) so a dev
# binary cannot be mistaken for a release one. Build metadata is not part of
# version identity, so compare the release version and ignore any suffix.
host_release_version="${host_version%%+*}"
if [[ "$host_release_version" != "$version" ]]; then
    echo "composed host version mismatch: expected $version, got ${host_version:-<empty>}" >&2
    echo "Output: $host_version_output" >&2
    exit 1
fi
"$python_bin" scripts/compose-product-bundle.py \
    --bundle "$output_dir" \
    --host "$output_dir/$INPUT_BINARY_NAME" \
    --runtime "$runtime_dir" \
    --version "$version" \
    --backend "$INPUT_BACKEND"
require_nonempty_file "composed product manifest" "$output_dir/product-manifest.json"

if [[ "$INPUT_READINESS_SMOKE" == "true" ]]; then
    MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$output_dir/native-runtimes" \
        "$output_dir/$INPUT_BINARY_NAME" --log-format json --version
    MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR="$output_dir/native-runtimes" \
        "$output_dir/$INPUT_BINARY_NAME" --log-format json runtime list
    # Exercise the shared SDK reader even when full SDK suites are unselected.
    # CPU needs no accelerator/driver; reuse only the just-composed producer bytes.
    if [[ "$INPUT_BACKEND" == "cpu" && "$(uname -s)" == "Linux" ]]; then
        echo "Checking CLI runtime JSON with the shared SDK runtime reader"
        MESH_SDK_NATIVE_RUNTIME_BUILD_FALLBACK=0 \
            scripts/ci-prepare-native-runtime.sh \
                "$output_dir/sdk-runtime-fallback" cpu \
                --reuse-from-binary "$output_dir/$INPUT_BINARY_NAME"
    fi
    scripts/ci-client-readiness-smoke.sh \
        "$output_dir/$INPUT_BINARY_NAME" \
        "$output_dir/native-runtimes"
elif [[ "$INPUT_READINESS_SMOKE" != "false" ]]; then
    echo "INPUT_READINESS_SMOKE must be true or false" >&2
    exit 1
fi

product_dir="$(cd "$output_dir" && pwd -P)"
runtime_name="$(basename "$runtime_dir")"
archive_path="$product_dir.tar.gz"
rm -f -- "$archive_path"
tar -C "$product_dir" -czf "$archive_path" .
require_nonempty_file "composed product archive" "$archive_path"
{
    printf 'product_dir=%s\n' "$(to_workflow_path "$product_dir")"
    printf 'binary_path=%s\n' \
        "$(to_workflow_path "$product_dir/$INPUT_BINARY_NAME")"
    printf 'runtime_root=%s\n' \
        "$(to_workflow_path "$product_dir/native-runtimes")"
    printf 'runtime_dir=%s\n' \
        "$(to_workflow_path "$product_dir/native-runtimes/$runtime_name")"
    printf 'archive_path=%s\n' "$(to_workflow_path "$archive_path")"
} >> "$GITHUB_OUTPUT"
