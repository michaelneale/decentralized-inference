# llama.cpp canary patch-queue repair runbook (agent instructions)

You are running on the `family-certify` self-hosted runner inside a mesh-llm
checkout. The nightly llama-upstream canary either failed to apply our patch
queue in `third_party/llama.cpp/patches/` onto the new upstream pin
(patch-queue mode) or applied the queue but failed a certification lane
(battery mode). Your job:

**Before touching the queue, read the repo skills and follow them:**
`.agents/skills/llama-patch-changes/SKILL.md` (queue edits, upstream pin,
prepare/build flow, patch ownership boundaries) and, when a patch changes the
stage ABI, `.agents/skills/llama-stage-patch-changes/SKILL.md`. The boundaries
in those skills are hard requirements for this repair, not suggestions.

1. **Reproduce.** Run `scripts/prepare-llama.sh "$(cat .deps/llama-canary-target-sha)"`
   and capture which patch fails to apply (`git -C .deps/llama.cpp am --3way ...`).
   A `.git/rebase-apply` state may be left behind; use `git am --show-current-patch`
   and `git am --3way --continue`/`--abort` to inspect the conflict.

2. **Fix the queue — follow `llama-patch-changes`, do not loop on `git am`.**
   If a patch fails to apply, `git am --3way` retry alone is not an acceptable
   resolution: a conflict means upstream refactored code a patch owns, and the
   skill's deliberate queue rewrite is the required path. Resolve the conflict
   on a llama.cpp branch (base on upstream `ggml-org/llama.cpp` `master` at the
   canary target SHA), reconstruct capability-owned commits, verify the
   reconstructed head is tree-identical to the intended final checkout, then
   regenerate the series with `git format-patch` per the skill. Keep the series
   ordered, keep every patch that still applies unchanged, and make the minimal
   semantic fix in the broken ones. Regenerate the series so
   `scripts/prepare-llama.sh` runs clean end to end.

3. **Build.** `scripts/build-llama.sh` then
   `cargo check -p skippy-ffi -p skippy-runtime -p skippy-server`.

4. **Certify.** `scripts/skippy-family-battery.sh --skip-build`.
   All lanes must pass. Do not weaken a failing lane; if a model is genuinely
   broken by upstream, revert to fixing our patches or flag it in the PR body.
   The wrapper re-runs the battery itself after your turn; if lanes fail you
   will get the failure output in a follow-up repair turn — the loop only
   ends when the wrapper's own battery run passes.

5. **Commit locally; the wrapper owns the PR.** Work on branch
   `llama-canary/patch-queue-fix`. Commit the patch-queue changes with a
   `fix(llama): rebase patch queue onto upstream <short-sha>` message. You
   have no GitHub credentials: the deterministic wrapper that drives you
   commits any remaining work, pushes the branch, and creates/updates the
   repair PR itself. The wrapper separately asks you to write the full PR
   description (key upstream changes, how the patch queue evolved, risks for
   reviewers) — when that turn arrives, write the finished Markdown to the
   file it names and touch nothing else. After the wrapper's own battery run
   passes, a separate review agent — not you — gets one fresh-context turn
   to review the certified repair and fix any dropped intent or rebase
   leftovers it finds; its changes land as their own `review(llama):`
   commit, and the next canary re-certifies everything after the merge.

Notes:
- Models come from the runner's pre-warmed HF cache (`HF_CACHE`); `hf download`
  is only a miss backstop. Never add GitHub Actions model caching.
- The deterministic wrapper owns the sole upstream selector,
  `third_party/llama.cpp/upstream.txt`. It writes it to the repair target and
  validates the queue through `scripts/prepare-llama.sh pinned`; do not edit
  the pin file yourself.
- Do not modify files outside `third_party/llama.cpp/patches/` unless the
  Rust ABI mirrors in `crates/` genuinely need to track a patch ABI change
  (bump `PREPARE_SCHEMA`/ABI version together in that case).

## New upstream model families (boundary registration)

`scripts/skippy-llama-parity.py validate` fails when a runnable parity row
(status `certified`, `candidate`, or `candidate_stateful`) refers to a model
file in `.deps/llama.cpp/src/models/` that does not register per-layer
`begin_block`/`end_block` stage boundaries. Use
`scripts/skippy-llama-parity.py classify-boundaries` to list exactly those
models. For each one:

1. **Either register the family**: add `begin_block(inpL, il)` /
   `end_block(cur, il)` hooks in the model's build loop following the
   existing registrations (see `src/models/qwen3.cpp`, patch 0072, and the
   llama-patch-changes skill), AND pin the smallest practical GGUF for the
   family as an immutable `model_pin` row in
   `docs/skippy/llama-parity-candidates.json` (repo, 40-hex revision, file,
   size_bytes, 64-hex blob_sha256 — the `file_integrity` schema of
   `ci/llama-canary/family-certified.json`), AND mirror the same model in
   `ci/llama-canary/family-certified.json`. The repair PR must update both
   manifests in the same commit and rerun the row via
   `scripts/skippy-family-battery.sh`; a family cannot be marked supported
   without executable evidence. Floating refs (repo without revision, or a
   moving tag) are rejected by `validate`.
2. **Or classify it explicitly**: set the row's status to
   `needs_boundary_registration` (support work queued, not yet runnable) and
   leave the certification untouched, or add an `unsupported_reason` string
   for families we deliberately do not serve.

A new upstream model file with no manifest row at all still fails validation
as `missing_candidate` — the repair PR must classify every new family before
the battery can pass.
