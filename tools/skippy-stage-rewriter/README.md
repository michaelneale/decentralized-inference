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

The checked-in family shards are produced from a clean llama.cpp tree after the
shared core patches and static model-semantics patches. Those static patches must
contain no stage controls. The first rewriter pass must see zero
`already_transformed` builders and generates every stage edit. The generator then
diffs the result against that stage-free semantic baseline so static support can
never leak into or poison the generated shards:

```sh
python3 scripts/generate-skippy-family-patch.py \
  --source-root .scratch/llama-central \
  --build-dir .scratch/llama-central-build \
  --rewriter .scratch/skippy-stage-rewriter-build/skippy-stage-rewriter \
  --report .scratch/skippy-stage-rewriter-report.json \
  --diff-base "$(git -C .scratch/llama-central rev-parse HEAD)" \
  --output target/generated-family-combined.patch \
  --shard-output-dir third_party/llama.cpp/patches/generated \
  --family-source-map ci/llama-canary/generated-family-map.json \
  --family-manifest ci/llama-canary/family-certified.json
```

The wrapper refuses a dirty or pre-transformed input tree, applies the
Clang-proven edits, checks a second pass for zero edits, and writes fixed-header
mail-patch shards after the core queue. `already_transformed` is valid only on
that second idempotence pass. The generated `series` file fixes application order;
`series.json` records each shard's digest, sources, and affected certified
families. A pre-transformed first-pass tree is always rejected, even when every
builder would otherwise be classified as supported.

After `scripts/build-llama.sh` has produced the compilation database and run the
native Skippy verifier, CI can run the complete deterministic check with:

```sh
SKIPPY_REWRITER_LLVM_PREFIX=/path/to/pinned/llvm \
  scripts/check-skippy-generated-family-patch.sh
```
