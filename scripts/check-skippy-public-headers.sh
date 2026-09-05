#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_WORKDIR="${LLAMA_WORKDIR:-$ROOT/.deps/llama.cpp}"
HEADER_DIR="$LLAMA_WORKDIR/include/skippy"
UMBRELLA_HEADER="$LLAMA_WORKDIR/include/skippy.h"

if [[ ! -d "$HEADER_DIR" ]]; then
  echo "missing prepared skippy header directory: $HEADER_DIR" >&2
  echo "run scripts/prepare-llama.sh first" >&2
  exit 1
fi

if [[ ! -f "$UMBRELLA_HEADER" ]]; then
  echo "missing installed umbrella header: $UMBRELLA_HEADER" >&2
  echo "run scripts/prepare-llama.sh first" >&2
  exit 1
fi

CC_BIN="${CC:-cc}"
CXX_BIN="${CXX:-c++}"
FAILED=0

# The umbrella (include/skippy.h) aggregates every capability header, so it
# is the header most likely to break from a per-capability change; it must
# be checked alongside include/skippy/*.h, not skipped.
for header in "$UMBRELLA_HEADER" "$HEADER_DIR"/*.h; do
  rel="${header#"$LLAMA_WORKDIR"/include/}"

  if ! "$CC_BIN" -std=c11 -Wall -Wextra -Werror \
      -I "$LLAMA_WORKDIR/include" -I "$LLAMA_WORKDIR/ggml/include" \
      -x c -fsyntax-only "$header"; then
    echo "C11 compile failed: include/$rel" >&2
    FAILED=1
  fi

  if ! "$CXX_BIN" -std=c++17 -Wall -Wextra -Werror \
      -I "$LLAMA_WORKDIR/include" -I "$LLAMA_WORKDIR/ggml/include" \
      -x c++ -fsyntax-only "$header"; then
    echo "C++17 compile failed: include/$rel" >&2
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "one or more installed skippy public headers failed independent compilation" >&2
  exit 1
fi

echo "all installed skippy public headers compile independently as C11 and C++17"
