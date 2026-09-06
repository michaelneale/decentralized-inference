# Skippy stage rewriter

This standalone Clang LibTooling program derives the repeated Skippy stage
filter edits from llama.cpp model-builder syntax. It accepts a prepared
llama.cpp source tree and its compilation database, classifies every graph
constructor, and emits an atomic edit set only when it can prove the complete
conventional builder shape.

The tool deliberately refuses ambiguous loops, activation chains, ownership
sites, and non-local exits. Compilation and the no-allocation graph verifier
remain the semantic gates for generated changes.

Build with a pinned LLVM/Clang installation:

```sh
cmake -S tools/skippy-stage-rewriter \
  -B .scratch/skippy-stage-rewriter-build \
  -G Ninja \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DClang_DIR=/path/to/llvm/lib/cmake/clang
cmake --build .scratch/skippy-stage-rewriter-build
```

Classify without editing:

```sh
.scratch/skippy-stage-rewriter-build/skippy-stage-rewriter \
  --source-root .deps/llama.cpp \
  --llama-commit "$(git -C .deps/llama.cpp rev-parse HEAD)" \
  --report .scratch/skippy-stage-rewriter-report.json \
  -p .deps/llama-build/build \
  .deps/llama.cpp/src/models/*.cpp
```

Add `--apply` to apply only the `transformable` edit sets. Run the tool again
over the result to prove idempotence: every edited builder must report
`already_transformed` with no edits.
