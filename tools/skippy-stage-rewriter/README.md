# Skippy stage rewriter

This standalone Clang LibTooling program derives the repeated Skippy stage
filter edits from llama.cpp model-builder syntax. It accepts a prepared
llama.cpp source tree and its compilation database, classifies every graph
constructor, and emits an atomic edit set only when it can prove the complete
partitioned-decoder builder shape.

The report distinguishes partitioned decoders from final-stage auxiliary
contexts such as MTP/draft heads and from whole-model graphs with multiple
sequential layer domains. Auxiliary and multi-domain graphs receive no generic
decoder-range edits. The tool still refuses ambiguous loops, activation
chains, ownership sites, and non-local exits that do not have one of those
proven execution scopes. Compilation and the no-allocation graph verifier
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
`already_transformed` with no edits. Builders reported as
`supported_auxiliary` or `supported_whole_model` retain an empty edit set and
carry the structural evidence for their execution scope.

The checked-in family shards are produced from a clean llama.cpp tree containing
the existing PR family behavior. The rewriter adds any newly proven edits, and
the generator diffs the result against the pinned upstream revision so the
complete model-family delta is grouped by the certified-family source map:

```sh
python3 scripts/generate-skippy-family-patch.py \
  --source-root .scratch/llama-central \
  --build-dir .scratch/llama-central-build \
  --rewriter .scratch/skippy-stage-rewriter-build/skippy-stage-rewriter \
  --report .scratch/skippy-stage-rewriter-report.json \
  --diff-base "$(cat .scratch/llama-full/.mesh-llm-upstream-sha)" \
  --output target/generated-family-combined.patch \
  --shard-output-dir third_party/llama.cpp/patches/generated \
  --family-source-map ci/llama-canary/generated-family-map.json \
  --family-manifest ci/llama-canary/family-certified.json
```

The wrapper refuses a dirty input tree, applies the Clang-proven edits, checks
a second pass for zero edits, and writes fixed-header mail-patch shards after
the core queue. The generated `series` file fixes application order;
`series.json` records each shard's digest, sources, and affected certified
families. A tree in which every supported builder is already transformed is
also valid: this lets CI canonicalize the complete prepared model-tree diff and
compare the generated directory byte-for-byte with the checked-in shards.

After `scripts/build-llama.sh` has produced the compilation database and run the
native Skippy verifier, CI can run the complete deterministic check with:

```sh
SKIPPY_REWRITER_LLVM_PREFIX=/path/to/pinned/llvm \
  scripts/check-skippy-generated-family-patch.sh
```
