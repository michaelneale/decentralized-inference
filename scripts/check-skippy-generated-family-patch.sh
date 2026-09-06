#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${SKIPPY_REWRITER_SOURCE_ROOT:-$ROOT/.deps/llama.cpp}"
LLAMA_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-${LLAMA_BUILD_DIR:-$ROOT/.deps/llama-build/build-stage-abi-static-metal}}"
CHECKED_PATCH="${SKIPPY_REWRITER_PATCH:-$ROOT/third_party/llama.cpp/patches/0076-skippy-generate-model-family-stage-controls.patch}"
ARTIFACT_ROOT="${SKIPPY_REWRITER_ARTIFACT_ROOT:-$ROOT/target/skippy-stage-rewriter-check}"
TOOL_BUILD="$ARTIFACT_ROOT/tool-build"
GENERATED_PATCH="$ARTIFACT_ROOT/generated-family.patch"
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
if [[ ! -f "$CHECKED_PATCH" ]]; then
  echo "checked-in generated family patch not found: $CHECKED_PATCH" >&2
  exit 1
fi
if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "prepared llama.cpp checkout must be clean before the generated-patch check" >&2
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
ctest --test-dir "$LLAMA_BUILD_DIR" --output-on-failure -R '^skippy_' \
  | tee "$ARTIFACT_ROOT/noalloc-graph-verifier.log"

EXTRA_ARGS=(--extra-arg=-resource-dir --extra-arg="$("$LLVM_PREFIX"/bin/clang -print-resource-dir)")
if command -v xcrun >/dev/null 2>&1; then
  EXTRA_ARGS+=(--extra-arg=-isysroot --extra-arg="$(xcrun --show-sdk-path)")
fi

cleanup() {
  git -C "$SOURCE_ROOT" reset --hard HEAD >/dev/null
}
trap cleanup EXIT

python3 "$ROOT/scripts/generate-skippy-family-patch.py" \
  --source-root "$SOURCE_ROOT" \
  --build-dir "$LLAMA_BUILD_DIR" \
  --rewriter "$TOOL_BUILD/skippy-stage-rewriter" \
  --report "$FIRST_REPORT" \
  --diff-base "$(tr -d '[:space:]' < "$SOURCE_ROOT/.mesh-llm-upstream-sha")" \
  --output "$GENERATED_PATCH" \
  "${EXTRA_ARGS[@]}"

if cmp -s "$GENERATED_PATCH" "$CHECKED_PATCH"; then
  patch_result=pass
else
  patch_result=fail
  diff -u "$CHECKED_PATCH" "$GENERATED_PATCH" > "$ARTIFACT_ROOT/patch-drift.diff" || true
fi

python3 "$ROOT/scripts/skippy-rewriter-harness.py" \
  --report "$FIRST_REPORT" \
  --mode validate \
  --patch-check "$patch_result" \
  --patch-drift-gate fail \
  --compile-result pass \
  --graph-verify-result pass
python3 "$ROOT/scripts/skippy-rewriter-harness.py" \
  --report "$SECOND_REPORT" \
  --mode idempotence

echo "generated family patch is deterministic and current: $CHECKED_PATCH"
