#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${SKIPPY_REWRITER_SOURCE_ROOT:-$ROOT/.deps/llama.cpp}"
LLAMA_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-${LLAMA_BUILD_DIR:-$ROOT/.deps/llama-build/build-stage-abi-static-metal}}"
CHECKED_PATCH_DIR="${SKIPPY_REWRITER_PATCH_DIR:-$ROOT/third_party/llama.cpp/patches/generated}"
CHECKED_PATCH_SERIES="$CHECKED_PATCH_DIR/series"
FAMILY_SOURCE_MAP="${SKIPPY_REWRITER_FAMILY_MAP:-$ROOT/ci/llama-canary/generated-family-map.json}"
FAMILY_MANIFEST="${SKIPPY_REWRITER_FAMILY_MANIFEST:-$ROOT/ci/llama-canary/family-certified.json}"
ARTIFACT_ROOT="${SKIPPY_REWRITER_ARTIFACT_ROOT:-$ROOT/target/skippy-stage-rewriter-check}"
TOOL_BUILD="$ARTIFACT_ROOT/tool-build"
GENERATED_PATCH="$ARTIFACT_ROOT/generated-family.patch"
GENERATED_PATCH_DIR="$ARTIFACT_ROOT/generated-family-shards"
FIRST_REPORT="$ARTIFACT_ROOT/report.json"
SECOND_REPORT="$ARTIFACT_ROOT/report-second.json"

if [[ ! -d "$SOURCE_ROOT/.git" && ! -f "$SOURCE_ROOT/.git" ]]; then
  echo "prepared llama.cpp checkout not found: $SOURCE_ROOT" >&2
  exit 1
fi
if [[ ! -f "$LLAMA_BUILD_DIR/compile_commands.json" ]]; then
  echo "llama.cpp compilation database not found: $LLAMA_BUILD_DIR/compile_commands.json" >&2
  exit 1
fi
if [[ ! -x "$LLAMA_BUILD_DIR/bin/skippy-noalloc-graph-planning" ]]; then
  echo "no-allocation graph verifier not found: $LLAMA_BUILD_DIR/bin/skippy-noalloc-graph-planning" >&2
  echo "build with LLAMA_STAGE_BUILD_TESTS=ON before checking the generated patch" >&2
  exit 1
fi
if [[ ! -f "$CHECKED_PATCH_SERIES" || ! -f "$CHECKED_PATCH_DIR/series.json" ]]; then
  echo "checked-in generated family patch series not found: $CHECKED_PATCH_DIR" >&2
  exit 1
fi
if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "prepared llama.cpp checkout must be clean before the generated-patch check" >&2
  exit 1
fi

# Normal preparation applies the checked-in generated family shards after the
# core Skippy queue. Generation must never consume those model edits as input:
# otherwise every builder is classified as already_transformed and inherited
# mistakes become self-validating. Rewind exactly the generated tail while the
# checker runs, then restore the prepared checkout for subsequent CI steps.
ORIGINAL_SOURCE_HEAD="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
GENERATED_PATCH_COUNT="$(sed '/^[[:space:]]*$/d' "$CHECKED_PATCH_SERIES" | wc -l | tr -d '[:space:]')"
if [[ ! "$GENERATED_PATCH_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "generated family patch series must contain at least one patch" >&2
  exit 1
fi
CORE_SOURCE_HEAD="$(git -C "$SOURCE_ROOT" rev-parse "$ORIGINAL_SOURCE_HEAD~$GENERATED_PATCH_COUNT")"

cleanup() {
  git -C "$SOURCE_ROOT" reset --hard >/dev/null
  git -C "$SOURCE_ROOT" checkout --force --detach "$ORIGINAL_SOURCE_HEAD" >/dev/null
}
trap cleanup EXIT

git -C "$SOURCE_ROOT" checkout --force --detach "$CORE_SOURCE_HEAD" >/dev/null
if marker_matches="$(rg -n '\bstage_filter\b|\bbegin_block[[:space:]]*\(|\bend_block[[:space:]]*\(' "$SOURCE_ROOT/src/models" || true)" &&
   [[ -n "$marker_matches" ]]; then
  echo "core-only generator input already contains model-stage transformations:" >&2
  printf '%s\n' "$marker_matches" >&2
  exit 1
fi

LLVM_PREFIX="${SKIPPY_REWRITER_LLVM_PREFIX:-}"
if [[ -z "$LLVM_PREFIX" ]] && command -v brew >/dev/null 2>&1; then
  LLVM_PREFIX="$(brew --prefix llvm@22 2>/dev/null || brew --prefix llvm 2>/dev/null || true)"
fi
if [[ -z "$LLVM_PREFIX" || ! -x "$LLVM_PREFIX/bin/clang" ]]; then
  echo "set SKIPPY_REWRITER_LLVM_PREFIX to the pinned LLVM installation" >&2
  exit 1
fi

mkdir -p "$ARTIFACT_ROOT"
cmake -S "$ROOT/tools/skippy-stage-rewriter" -B "$TOOL_BUILD" -G Ninja \
  -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
  -DClang_DIR="$LLVM_PREFIX/lib/cmake/clang"
cmake --build "$TOOL_BUILD"
ctest --test-dir "$TOOL_BUILD" --output-on-failure

EXTRA_ARGS=(--extra-arg=-resource-dir --extra-arg="$("$LLVM_PREFIX"/bin/clang -print-resource-dir)")
if command -v xcrun >/dev/null 2>&1; then
  EXTRA_ARGS+=(--extra-arg=-isysroot --extra-arg="$(xcrun --show-sdk-path)")
fi

python3 "$ROOT/scripts/generate-skippy-family-patch.py" \
  --source-root "$SOURCE_ROOT" \
  --build-dir "$LLAMA_BUILD_DIR" \
  --rewriter "$TOOL_BUILD/skippy-stage-rewriter" \
  --report "$FIRST_REPORT" \
  --diff-base "$(tr -d '[:space:]' < "$SOURCE_ROOT/.mesh-llm-upstream-sha")" \
  --output "$GENERATED_PATCH" \
  --shard-output-dir "$GENERATED_PATCH_DIR" \
  --family-source-map "$FAMILY_SOURCE_MAP" \
  --family-manifest "$FAMILY_MANIFEST" \
  "${EXTRA_ARGS[@]}"

TRANSFORMED_TREE_TARGETS=(
  llama
  skippy-graph-build-inputs
  skippy-hardware-application-probe
  skippy-model-fixture-generator
  skippy-model-loader-accounting
  skippy-noalloc-graph-planning
  skippy-renamed-multishard-planning
  skippy-stage-plan-header-c
  skippy-stage-plan-header-cpp
  skippy-stage-slice-plan
)
if cmake --build "$LLAMA_BUILD_DIR" --target "${TRANSFORMED_TREE_TARGETS[@]}" 2>&1 \
  | tee "$ARTIFACT_ROOT/transformed-tree-compile.log"; then
  compile_result=pass
else
  compile_result=fail
fi

if [[ "$compile_result" == pass ]] && \
  ctest --test-dir "$LLAMA_BUILD_DIR" --output-on-failure -R '^skippy_' 2>&1 \
    | tee "$ARTIFACT_ROOT/noalloc-graph-verifier.log"; then
  graph_verify_result=pass
else
  graph_verify_result=fail
fi

if diff -qr "$CHECKED_PATCH_DIR" "$GENERATED_PATCH_DIR" >/dev/null; then
  patch_result=pass
else
  patch_result=fail
  diff -ru "$CHECKED_PATCH_DIR" "$GENERATED_PATCH_DIR" > "$ARTIFACT_ROOT/patch-drift.diff" || true
  python3 "$ROOT/scripts/select-skippy-family-shards.py" \
    --base "$CHECKED_PATCH_DIR/series.json" \
    --current "$GENERATED_PATCH_DIR/series.json" \
    --output "$ARTIFACT_ROOT/changed-family-selection.json"
fi

python3 "$ROOT/scripts/skippy-rewriter-harness.py" \
  --report "$FIRST_REPORT" \
  --mode validate \
  --patch-check "$patch_result" \
  --patch-drift-gate fail \
  --compile-result "$compile_result" \
  --graph-verify-result "$graph_verify_result"
python3 "$ROOT/scripts/skippy-rewriter-harness.py" \
  --report "$SECOND_REPORT" \
  --mode idempotence

echo "generated family patch series is deterministic and current: $CHECKED_PATCH_DIR"
