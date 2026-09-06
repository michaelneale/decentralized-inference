#!/usr/bin/env bash
set -euo pipefail

# llama-upstream canary agent repair (issue #1434; wired into
# llama-upstream-canary.yml for both patch-queue apply failures and family
# battery failures).
#
# Usage: llama-canary-agent-repair.sh <mode> [upstream-sha]
#   mode: patch-queue  - prepare-llama.sh failed to apply the patch queue
#                        onto the new upstream; the agent rebases the queue.
#   mode: battery      - the patch queue applied but the family battery
#                        failed; the agent fixes the root cause.
#   upstream-sha: 40-hex llama.cpp commit, preferentially passed via the
#                  UPSTREAM_SHA_INPUT environment variable (callers must
#                  never interpolate untrusted dispatch values into shell).
#                  When omitted, resolves master via git ls-remote. "latest"
#                  is also accepted. Battery-mode evidence: when the workflow
#                  tees its battery log to $BATTERY_LOG, it is reused instead
#                  of re-running the battery before the first repair turn.
#
# Drives a non-interactive `opencode` agent (model:
# zai-coding-plan/glm-5.3-flash by default) to repair, then the wrapper
# itself re-runs the certification battery. If it fails, each failure gets
# its own opencode repair turn (with the battery failure summary in the
# prompt) followed by a recertify, up to CANARY_REPAIR_MAX_TURNS (default 2)
# repair turns. The script only succeeds when the battery actually passes
# on this runner.
#
# PR guarantee: whatever the outcome, the wrapper (not the agent) ensures a
# repair PR exists on $BRANCH and posts a status comment describing the work
# done and, on failure, what the agent is stuck on and needs human help
# with. The PR description is written by an agent turn that runs BEFORE
# certification (upstream changes, patch-queue evolution, risks) with a
# deterministic fallback body. After a GREEN battery exactly one more agent
# turn runs: a fresh-context review of the certified repair that may modify
# the tree (dropped patch intent, rebase leftovers, ABI mirror drift); its
# changes ride as a separate review(llama): commit and are flagged in the
# success comment. That review is fail-open and never blocks a green repair.
#
# Credential split: the agent never sees a GitHub token — CANARY_REPAIR_TOKEN
# is stripped from its environment, and only the deterministic wrapper
# performs git pushes, PR creation, PR edits, and comments with the token
# scoped to individual commands. The wrapper — never the agent — commits the
# certified tree, pushes it, and verifies the repair PR head equals the
# certified commit before reporting success.
#
# Credentials: pushes/PRs use $CANARY_REPAIR_TOKEN (fine-grained PAT with
# Contents+PR write; the canary job itself stays contents: read). The agent
# needs OPENCODE_API_KEY/NEMOTRON_API_KEY or an `opencode auth login`
# profile on the runner.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${1:?usage: llama-canary-agent-repair.sh <patch-queue|battery> [upstream-sha]}"
case "$MODE" in
  patch-queue | battery) ;;
  *)
    echo "unknown repair mode: $MODE (expected patch-queue or battery)" >&2
    exit 1
    ;;
esac

UPSTREAM_SHA="${2:-${UPSTREAM_SHA_INPUT:-latest}}"
if [[ "$UPSTREAM_SHA" == "latest" || -z "$UPSTREAM_SHA" ]]; then
  UPSTREAM_SHA="$(git ls-remote https://github.com/ggml-org/llama.cpp.git master | awk '{print $1}')"
fi
if [[ ! "$UPSTREAM_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "refusing to repair against a non-40-hex upstream SHA: $UPSTREAM_SHA" >&2
  exit 1
fi

OLD_SHA="$(tr -d '[:space:]' < "$ROOT/third_party/llama.cpp/upstream.txt")"
AGENT_MODEL="${CANARY_AGENT_MODEL:-zai-coding-plan/glm-5.3-flash}"
MAX_REPAIR_TURNS="${CANARY_REPAIR_MAX_TURNS:-2}"
BRANCH="llama-canary/patch-queue-fix"
BATTERY_LOG="$ROOT/.deps/llama-canary-repair-battery.log"
PIN_FILE="$ROOT/third_party/llama.cpp/upstream.txt"

mkdir -p "$ROOT/.deps"
echo "$UPSTREAM_SHA" > "$ROOT/.deps/llama-canary-target-sha"

if ! command -v opencode >/dev/null 2>&1; then
  echo "opencode CLI not found on runner; install opencode-ai on the family-certify image" >&2
  exit 1
fi
# Agent credentials: either an explicit API key env var, or an opencode CLI
# that has been logged in on the runner (`opencode auth login`), which
# `opencode run` picks up from its own auth store.
if [[ -z "${OPENCODE_API_KEY:-}" && -z "${NEMOTRON_API_KEY:-}" ]]; then
  if [[ ! -s "${HOME}/.local/share/opencode/auth.json" ]] && ! opencode auth list 2>/dev/null | grep -Eq '[1-9][0-9]* credentials'; then
    echo "no agent credentials: set OPENCODE_API_KEY/NEMOTRON_API_KEY or run 'opencode auth login' on the runner" >&2
    exit 1
  fi
fi
# The canary job itself is read-only; the repair branch push and PR need the
# dedicated fine-grained token. The token is never exported into the process
# environment — every GitHub mutation is routed through gh_repair(), which
# scopes it to that single command — so agent turns (and anything they spawn)
# never inherit repository-write credentials.
if [[ -z "${CANARY_REPAIR_TOKEN:-}" ]]; then
  echo "CANARY_REPAIR_TOKEN is not set; cannot push the repair branch or open the repair PR" >&2
  exit 1
fi

gh_repair() {
  # Deterministic GitHub mutations only. The write PAT is scoped to this one
  # command; it is deliberately absent from the ambient environment.
  GH_TOKEN="$CANARY_REPAIR_TOKEN" "$@"
}

check_repair_token_permissions() {
  # Preflight (issue #1434 follow-up): fail in seconds when the identity behind
  # CANARY_REPAIR_TOKEN cannot actually write to this repository, instead of
  # discovering it via a git 403 after hours of repair work (seen live: a
  # fine-grained PAT whose account HAD repo write, but whose token lacked the
  # Contents: write permission). The REST permissions object reflects the
  # ACCOUNT's access, not the token's fine-grained scope, so reading
  # .permissions is not sufficient — the check probes the actual capability by
  # creating and deleting a temporary ref, which requires exactly the
  # Contents: write permission a push needs. The token is used only in scoped
  # single commands, never exported, and never echoed.
  local login default_branch head_sha probe_ref
  if ! login="$(gh_repair gh api user --jq .login 2>/dev/null)"; then
    echo "preflight: CANARY_REPAIR_TOKEN does not authenticate (gh api user failed); check the secret value" >&2
    return 1
  fi
  default_branch="$(gh_repair gh api "repos/${GITHUB_REPOSITORY:?}" --jq .default_branch 2>/dev/null)"
  if [[ -z "$default_branch" ]]; then
    echo "preflight: could not read ${GITHUB_REPOSITORY} with the repair token; grant the PAT access to this repository" >&2
    return 1
  fi
  head_sha="$(gh_repair gh api "repos/${GITHUB_REPOSITORY:?}/branches/${default_branch}" --jq .commit.sha 2>/dev/null)"
  if [[ -z "$head_sha" ]]; then
    echo "preflight: could not resolve ${default_branch} on ${GITHUB_REPOSITORY} with the repair token" >&2
    return 1
  fi
  probe_ref="refs/heads/canary-repair-token-preflight"
  if ! gh_repair gh api --method POST "repos/${GITHUB_REPOSITORY:?}/git/refs" \
      -f ref="$probe_ref" -f sha="$head_sha" >/dev/null 2>&1; then
    echo "preflight: identity '${login}' cannot write refs on ${GITHUB_REPOSITORY}: the fine-grained PAT must include this repository with Contents: Read and write (account-level write is not enough). Edit the PAT's permissions or mint one with Contents: RW, then re-save CANARY_REPAIR_TOKEN." >&2
    return 1
  fi
  # The delete URL uses the branch name only (slashes percent-encoded); the
  # refs/ prefix is part of the path, not the ref identifier.
  if ! gh_repair gh api --method DELETE \
      "repos/${GITHUB_REPOSITORY:?}/git/refs/heads%2Fcanary-repair-token-preflight" >/dev/null 2>&1; then
    echo "preflight: WARNING: write probe succeeded but the temporary ref ${probe_ref} could not be deleted; delete it manually" >&2
  fi
  echo "preflight: repair token identity '${login}' verified read+write on ${GITHUB_REPOSITORY}"
}

cd "$ROOT"

# Preflight the repair token before any repair work: a permission gap here
# fails the run in seconds with the exact fix, instead of surfacing as a git
# 403 after a potentially hours-long certified repair (seen live, run 33151501701).
check_repair_token_permissions

# Run-scope the persistent-runner artifacts before anything else can fail and
# leave a previous run's state behind. The PR-body file is always cleared. The
# battery evidence log is cleared only in patch-queue mode: in battery mode it
# holds THIS run's workflow battery output (teed by the workflow immediately
# before invoking this script) and must survive to seed the first repair turn.
if [[ "$MODE" == "patch-queue" ]]; then
  rm -f "$BATTERY_LOG"
fi
rm -f "$ROOT/.deps/llama-canary-pr-body.md"
rm -f "$ROOT/.deps/llama-canary-review-report.md"

# Repair turns routinely build scratch worktrees under /tmp (e.g.
# /tmp/llama-old-pin, /tmp/llama-repair). On this persistent runner those
# directories and their registrations in .deps/llama.cpp/.git/worktrees
# survive across runs, and a later `git worktree add` for the same path
# fails ("missing but already registered" / stale admin files) — the agent
# treats that as fatal and ends its turn early (live: run 33158798988
# aborted at `rm -rf /tmp/llama-old-pin && git worktree add ...`). Prune
# stale registrations and clear the known scratch paths up front.
git -C "$ROOT/.deps/llama.cpp" worktree prune >/dev/null 2>&1 || true
rm -rf /tmp/llama-old-pin /tmp/llama-repair /tmp/llama-repair-* 2>/dev/null || true

# The agent reuses the open repair PR on $BRANCH if one exists, so repeated
# canary failures amend a single PR instead of stacking duplicates. Surface
# the current PR number (if any) in the prompt so it does not have to guess.
# Every GitHub call — read or write — carries the repair token: the workflow
# job never exports GH_TOKEN/GITHUB_TOKEN and checks out with
# persist-credentials disabled, so an unauthenticated gh would silently fail.
EXISTING_PR=""
if command -v gh >/dev/null 2>&1; then
  EXISTING_PR="$(gh_repair gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number' || true)"
fi

agent_turn() {
  # Non-fatal: a crashed agent turn must not skip PR reporting. The model
  # runs without any GitHub token: push/PR/comment mutations are wrapper-only.
  # A heartbeat monitor prints elapsed time every 10 minutes so multi-hour
  # repair turns show progress in the Actions log instead of looking stuck.
  # It also reports the newest file modification under the llama.cpp worktree,
  # so watchers can tell "agent is editing" from "turn is hung" without
  # runner access.
  local prompt="$1" started heartbeat_pid
  started="$(date +%s)"
  # Job control makes the heartbeat its own process group (portable on both
  # the macOS family-certify runner and Linux CI); env -i keeps the monitor
  # credential-free. GNU find's printf action and util-linux's session
  # utility are not portable to macOS, hence the plain find and the
  # set -m process group.
  set -m
  # shellcheck disable=SC2016  # heartbeat script takes its inputs as $1/$2
  env -i PATH="$PATH" bash -c '
    ROOT="$1"
    started="$2"
    while sleep 600; do
      newest="$(find "$ROOT/.deps/llama.cpp" -type f -newer "$ROOT/.deps/llama-canary-target-sha" -print -quit 2>/dev/null || true)"
      printf "heartbeat: agent turn running for %dm; recent worktree activity: %s\n" \
        "$(( ($(date +%s) - started) / 60 ))" "${newest:-none observed yet}"
    done
  ' heartbeat "$ROOT" "$started" &
  heartbeat_pid=$!
  set +m
  # --auto: auto-approve opencode's permission prompts. The sandbox auto-
  # REJECTS out-of-workspace writes (e.g. scratch dirs under /tmp) in non-
  # interactive mode, and a rejection terminates the whole turn — every
  # short-turn failure so far traces to this (live: run 33160131810 aborted
  # on "external_directory (/tmp/opencode/*); auto-rejecting"). The agent
  # runs with all GitHub tokens stripped and only the wrapper holds write
  # credentials, so auto-approval inside this sandbox is safe.
  env -u GH_TOKEN -u GITHUB_TOKEN -u CANARY_REPAIR_TOKEN \
    opencode run --auto --model "$AGENT_MODEL" "$prompt" \
    || echo "warning: opencode turn exited non-zero" >&2
  kill -- "-$heartbeat_pid" 2>/dev/null || kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
}

battery_summary() {
  # Last 80 lines of the most recent battery evidence — either this run's
  # workflow-teed log (battery mode) or the wrapper's own certification run —
  # enough to name the failing family/split lanes without flooding the agent
  # prompt.
  local log="$1"
  if [[ ! -s "$log" ]]; then
    echo "(no battery output captured; see the canary run log)"
    return 0
  fi
  tail -n 80 "$log"
}

current_pr() {
  gh_repair gh pr list --head "$BRANCH" --state open --json number --jq '.[0].number' 2>/dev/null || true
}

# Repair remote: the write PAT is embedded in the push URL (never echoed) and
# exists only for the lifetime of the single git push command.
repair_remote() {
  echo "https://x-access-token:${CANARY_REPAIR_TOKEN}@github.com/${GITHUB_REPOSITORY:?GITHUB_REPOSITORY not set}.git"
}

redact_token() {
  # Strip the repair token from anything that might reach the log.
  sed "s/${CANARY_REPAIR_TOKEN}/***redacted***/g"
}

write_repair_pin() {
  # Pin ownership is deterministic wrapper work, never agent work. The repair
  # branch must select the same upstream that its queue and battery certify.
  scripts/update-llama-pin.sh "$UPSTREAM_SHA"
}

verify_repair_pin() {
  local pin
  pin="$(tr -d '[:space:]' < "$PIN_FILE")"
  if [[ "$pin" != "$UPSTREAM_SHA" ]]; then
    echo "repair pin does not select certified upstream $UPSTREAM_SHA (upstream.txt=$pin)" >&2
    return 1
  fi
}

prepare_repair_target() {
  local prepared_upstream
  write_repair_pin || return 1
  verify_repair_pin || return 1
  scripts/prepare-llama.sh pinned || return 1
  prepared_upstream="$(tr -d '[:space:]' < "$ROOT/.deps/llama.cpp/.mesh-llm-upstream-sha")"
  if [[ "$prepared_upstream" != "$UPSTREAM_SHA" ]]; then
    echo "pinned repair prepared upstream $prepared_upstream, expected $UPSTREAM_SHA" >&2
    return 1
  fi
}

publish_repair_branch() {
  # The wrapper — never the agent — owns commit and push. Puts the current
  # working tree (agent's repaired queue) on $BRANCH and records the exact
  # certified SHA. Force-push is intentional: the agent rebases the patch
  # queue, so non-fast-forward updates are the normal case on this
  # wrapper-owned branch.
  git checkout -B "$BRANCH"
  if [[ -n "$(git status --porcelain)" ]]; then
    git add -A
    git commit -m "fix(llama): canary repair at upstream ${UPSTREAM_SHA:0:10}" \
      -m "Automated llama.cpp canary repair (mode: ${MODE}). Certified by the family battery on the family-certify runner."
  fi
  CERTIFIED_SHA="$(git rev-parse HEAD)"
  # Push failures are almost always a token-identity permission gap (seen
  # live: 403 "denied to i386" because the PAT account lacked repo write).
  # Surface a precise, actionable message instead of a bare git error, and
  # never leak the token in the hint.
  if ! git push "$(repair_remote)" "+HEAD:refs/heads/${BRANCH}" 2> >(redact_token >&2); then
    echo "ERROR: could not push ${BRANCH}. If git reported 403/denied above, the identity behind CANARY_REPAIR_TOKEN lacks write access to ${GITHUB_REPOSITORY:?}: grant that account Contents+Pull requests write (or mint the PAT from an account that has it) and update the secret." >&2
    return 1
  fi
}

ensure_pr() {
  # Wrapper-owned PR guarantee: if no open PR exists on $BRANCH (the branch
  # was just pushed by publish_repair_branch), create one. If the branch has
  # no diff against the base (agent produced nothing), fall back to an issue
  # so the outcome is still visible to humans.
  local pr title body
  pr="$(current_pr)"
  if [[ -n "$pr" ]]; then
    printf '%s\n' "$pr"
    return 0
  fi
  title="fix(llama): rebase patch queue onto upstream ${UPSTREAM_SHA:0:10}"
  body="Automated canary repair PR for the llama.cpp patch queue at upstream ${UPSTREAM_SHA}."
  if ! git diff --quiet origin/main..."$BRANCH" 2>/dev/null; then
    if pr="$(gh_repair gh pr create --head "$BRANCH" --title "$title" --body "$body" 2>/dev/null \
             | grep -oE '[0-9]+$')"; then
      printf '%s\n' "$pr"
      return 0
    fi
  fi
  gh_repair gh issue create --title "llama canary repair needs human assistance (upstream ${UPSTREAM_SHA:0:10})" \
    --body "The canary repair loop could not open a PR on \`$BRANCH\` (branch missing or no diff). See the canary run log for the repair-loop outcome." \
    | grep -oE '[0-9]+$' || true
  return 0
}

verify_pr_head_is_certified() {
  # The PR must carry exactly the bytes the wrapper published: the remote PR
  # head must equal the last commit the wrapper pushed — the certified
  # commit, or the post-green review head when the review agent modified the
  # tree. The success comment names both, so reviewers can tell certified
  # bytes from review bytes; review bytes re-certify on the next canary run
  # after the PR merges.
  local pr remote_head attempt expected
  pr="$(current_pr)"
  if [[ -z "$pr" ]]; then
    echo "no repair PR to verify" >&2
    return 1
  fi
  expected="${REVIEW_HEAD:-${CERTIFIED_SHA:?}}"
  remote_head="$(gh_repair gh pr view "$pr" --json headRefOid --jq .headRefOid 2>/dev/null || true)"
  for attempt in 1 2 3; do
    if [[ "$remote_head" == "$expected" ]]; then
      return 0
    fi
    sleep "$attempt"
    remote_head="$(gh_repair gh pr view "$pr" --json headRefOid --jq .headRefOid 2>/dev/null || true)"
  done
  echo "repair PR #${pr} head (${remote_head:-none}) does not match the wrapper-published commit ${expected}" >&2
  return 1
}

pr_comment() {
  # Post a status comment on the repair PR; never fails the loop.
  local body="$1" resource
  resource="$(current_pr)"
  [[ -n "$resource" ]] || resource="$(ensure_pr)"
  [[ -n "$resource" ]] || return 0
  # ensure_pr returns an issue number when no PR exists; use the right command.
  if gh_repair gh pr view "$resource" >/dev/null 2>&1; then
    gh_repair gh pr comment "$resource" --body "$body" >/dev/null 2>&1 || true
  else
    gh_repair gh issue comment "$resource" --body "$body" >/dev/null 2>&1 || true
  fi
}

report_success() {
  # Green-battery closeout. The wrapper publishes the certified tree, ensures
  # the PR and body, then — the change review asked for — one fresh-context
  # review agent re-reads the certified repair and may modify it (its commit
  # rides as a separate review(llama): commit on the same branch; see
  # post_green_review_turn). Only after that does the wrapper verify the
  # PR-head binding, so the verification and the success comment reflect the
  # final PR head, not a stale certified one.
  publish_repair_branch
  # Ensure the PR exists BEFORE applying the body: on a first run no PR
  # exists yet, ensure_pr creates it (with the generic body), and the agent's
  # pre-certification draft is then applied on top (live: run 33163990453
  # pushed a certified branch and opened the PR via pr_comment's ensure_pr
  # AFTER apply_pr_body had already no-op'd, so the PR kept the generic body
  # and the agent's 103-line analysis was never shown).
  ensure_pr >/dev/null
  apply_pr_body
  post_green_review_turn
  # The literal backticks around the certified SHA are Markdown, not command
  # substitution.
  # shellcheck disable=SC2016
  pr_comment "$(printf '**Family battery passed** after the agent repair at upstream %s.\nAll certification lanes green on the family-certify runner; certified commit: `%s`.%s%s' \
    "$UPSTREAM_SHA" "${CERTIFIED_SHA:?}" "${REVIEW_STATUS:-}" "${REVIEW_REPORT_TAIL:-}")"
  verify_pr_head_is_certified
}

draft_pr_body() {
  # One agent turn drafts the repair PR description BEFORE certification: key
  # upstream changes between the old pin and the repair target, how the patch
  # queue evolved, risks, and validation. Runs strictly before any battery
  # attempt it describes; a failed or empty turn falls back to the
  # deterministic body in apply_pr_body.
  local pr body_file
  pr="$(current_pr)"
  [[ -n "$pr" ]] || return 0
  body_file="$ROOT/.deps/llama-canary-pr-body.md"
  agent_turn "$(printf 'Write the description for repair PR #%s.\nAnalyze the llama.cpp changes between %s (old pinned upstream) and %s\n(repair target), summarize the key upstream changes, explain how the patch\nqueue in third_party/llama.cpp/patches/ evolved in this repair (per patch:\nwhat conflicted and how it was resolved), and identify risks for reviewers\n(including any ABI impact and any lane that is newly failing or excluded).\nWrite the finished Markdown description to %s using your file tools. Do not\nedit any other file. Note: you have no GitHub credentials; the wrapper owns\nall pushes and PR updates.' \
    "$pr" "${OLD_SHA:0:10}" "${UPSTREAM_SHA:0:10}" "$body_file")"
}

apply_pr_body() {
  # Publish the PR description from the pre-certification agent draft (or the
  # deterministic fallback). No agent involvement: this may run after a green
  # battery, so it must be token-only and deterministic.
  local pr body_file
  pr="$(current_pr)"
  [[ -n "$pr" ]] || return 0
  body_file="$ROOT/.deps/llama-canary-pr-body.md"
  if [[ ! -s "$body_file" ]]; then
    {
      echo "Automated canary repair at upstream ${UPSTREAM_SHA}."
      echo
      echo "- Old pinned upstream: ${OLD_SHA}"
      echo "- Repair target upstream: ${UPSTREAM_SHA}"
      echo "- Mode: ${MODE}"
      echo
      echo "The agent-written upstream/queue analysis was unavailable; reviewers"
      echo "should diff the patch queue against main directly."
    } > "$body_file"
  fi
  gh_repair gh pr edit "$pr" --body-file "$body_file" >/dev/null 2>&1 || true
}

run_battery() {
  # Runs the certification battery; prints the summary line and returns the
  # battery exit code. The build runs under `arch -arm64` mirroring the
  # workflow's own build step: the family-certify job runs under Rosetta, and
  # a plain build-llama.sh rebuild would reconfigure for x86_64, leaving
  # arm64 Rust objects unable to link against x86_64 native archives (seen
  # live in run 33140672269). An arm64 sanity check follows the build so a
  # misconfigured toolchain fails loudly instead of as symbol errors.
  prepare_repair_target || return 1
  # Battery-mode repairs still prove the complete pinned live matrix. New
  # conventional model builders are handled by the source rewriter and the
  # single generated family patch; the repair loop no longer selects one
  # missing boundary-registration family or grows the model manifest.
  if [[ "$MODE" == "battery" ]]; then
    if ! python3 scripts/skippy-llama-parity.py --llama-src .deps/llama.cpp \
        validate >>"$BATTERY_LOG" 2>&1; then
      tail -n 2 "$BATTERY_LOG"
      echo "parity manifest validation failed; repair cannot certify" >&2
      return 1
    fi
    export SKIPPY_CANARY_LIVE_MATRIX_BACKEND="${SKIPPY_CANARY_LIVE_MATRIX_BACKEND:-metal}"
    if ! arch -arm64 scripts/skippy-canary-live-matrix.sh --prepare \
        >>"$BATTERY_LOG" 2>&1; then
      tail -n 2 "$BATTERY_LOG"
      echo "live package-v2 matrix failed; repair cannot certify" >&2
      return 1
    fi
  fi
  LLAMA_STAGE_BUILD_TESTS=ON arch -arm64 scripts/build-llama.sh \
    -DCMAKE_OSX_ARCHITECTURES=arm64 || return 1
  local archive
  archive="${LLAMA_STAGE_BUILD_DIR:-}/src/libllama.a"
  if [[ -n "${LLAMA_STAGE_BUILD_DIR:-}" && -f "$archive" ]] \
     && [[ "$(lipo -archs "$archive" 2>/dev/null)" != "arm64" ]]; then
    echo "refusing to certify: native archive is not arm64: $(lipo -archs "$archive" 2>/dev/null)" >&2
    return 1
  fi
  if ! scripts/check-skippy-generated-family-patch.sh >>"$BATTERY_LOG" 2>&1; then
    tail -n 5 "$BATTERY_LOG"
    echo "generated model-family patch is stale or invalid; repair cannot certify" >&2
    return 1
  fi
  if scripts/skippy-family-battery.sh >"$BATTERY_LOG" 2>&1; then
    tail -n 2 "$BATTERY_LOG"
    return 0
  fi
  tail -n 2 "$BATTERY_LOG"
  return 1
}

post_green_review_turn() {
  # Fresh-context review after a green battery (the change review asked for:
  # even a certified repair PR gets an agent review that may modify it).
  # Parity certification cannot see a rebase that silently drops a patch's
  # intent (e.g. a conflict resolution that yields parity but deletes an
  # upstream feature we had deliberately stopped deleting), so one review
  # turn runs on the published, certified tree. It is explicitly told it did
  # NOT author the repair. It may modify the tree — fix dropped intent,
  # rebase leftovers, stale patch metadata, Rust ABI mirror drift — and its
  # changes become a separate `review(llama):` commit pushed to the same
  # branch. The human merging the PR can tell the two commits apart; the
  # next canary re-certifies merged main before the pin advances, so review
  # modifications can never bypass certification. Fail-open by design: a
  # crashed or disabled review turn never fails a green repair, and a review
  # with no findings makes no commit. Runs before verify_pr_head_is_certified
  # so the success comment reports the final PR head, not a stale one.
  if [[ "${CANARY_AGENT_REVIEW:-true}" != "true" ]]; then
    echo "post-green agent review disabled (CANARY_AGENT_REVIEW != true); skipping"
    REVIEW_STATUS=" Post-green agent review skipped (disabled via CANARY_AGENT_REVIEW)."
    return 0
  fi
  echo "post-green agent review turn (fresh context)..."
  agent_turn "$(printf 'You are a DIFFERENT agent reviewing a completed llama.cpp canary repair — you did NOT write it. Everything below is already certified green by the family battery, so do not re-run it.

Read the repair PR branch (HEAD of this checkout, branch %s): the patch queue in third_party/llama.cpp/patches/ and the commits since main, plus ci/llama-canary/agent-repair-prompt.md and the repo skills it names for patch-ownership boundaries. The repair rebased the queue onto upstream %s.

Review the repair, not the upstream code. Certification parity cannot see semantic losses, so check:
1. Dropped intent: does any conflict resolution or regenerated patch silently stop doing what the old patch did (a deliberately-kept upstream feature accidentally deleted, a guard or accounting change quietly dropped)?
2. Rebase leftovers: conflict markers, duplicate patch fragments, hunks that now apply as no-ops, stale patch descriptions.
3. Patch hygiene: series ordering, patch subjects/bodies still matching content, no accidental upstream-code deletion (per the skills, patches must not delete upstream behavior we are not chartered to delete).
4. ABI mirrors: if any patch changed the stage ABI, did the Rust mirrors in crates/ track it (version + PREPARE_SCHEMA bumped together)?
5. Weakened lanes: any manifest, policy, or battery change that certifies less than before?

If (and only if) you find a real defect, fix it minimally in the patch queue (or its Rust ABI mirror) and commit locally with message "review(llama): <what and why>". Never weaken a certification lane; if a defect cannot be safely fixed without recertification, leave the tree unchanged and report it.

Write your review findings (verification steps, defects found, fixes made or recommended) to %s using your file tools. Then stop. You have no GitHub credentials — the wrapper owns all pushes and PR updates.' \
    "$BRANCH" "$UPSTREAM_SHA" "$ROOT/.deps/llama-canary-review-report.md")" \
    || echo "warning: post-green review turn exited non-zero; continuing with the certified tree" >&2
  if [[ ! -s "$ROOT/.deps/llama-canary-review-report.md" ]]; then
    echo "post-green review produced no report; continuing with the certified tree" >&2
    REVIEW_STATUS=" Post-green agent review ran but produced no report; the certified tree is unchanged."
    return 0
  fi
  # A review turn may edit the worktree, but it never owns the selected
  # upstream. Restore and verify the pin before publishing its result.
  write_repair_pin || return 1
  verify_repair_pin || return 1
  # The agent may have committed locally (per its prompt) and/or left work in
  # the working tree; the wrapper owns every push. Publish whatever HEAD is
  # now only when it differs from the certified commit.
  if [[ "$(git rev-parse HEAD)" != "${CERTIFIED_SHA:?}" || -n "$(git status --porcelain)" ]]; then
    if [[ -n "$(git status --porcelain)" ]]; then
      git add -A
      if ! git diff --cached --quiet; then
        git commit -m "review(llama): agent review fixes at upstream ${UPSTREAM_SHA:0:10}" \
          -m "Modifications from the post-certification agent review of the canary repair (see the repair PR comment for the review report)."
      fi
    fi
    echo "post-green review made changes; publishing review head $(git rev-parse --short HEAD)"
    git push "$(repair_remote)" "+HEAD:refs/heads/${BRANCH}" 2> >(redact_token >&2) \
      || echo "warning: could not push the review commit; the certified tree remains the PR head" >&2
    # shellcheck disable=SC2016  # backticks are intentional Markdown
    REVIEW_STATUS="$(printf ' Post-green agent review made modifications: PR head is now the review commit `%s` (certified commit `%s` above; review changes re-certify when this PR merges and the next canary runs).' \
      "$(git rev-parse --short HEAD)" "${CERTIFIED_SHA:0:10}")"
  else
    echo "post-green review found nothing to change; certified tree unchanged"
    REVIEW_STATUS=" Post-green agent review found nothing to change; the certified tree stands."
  fi
  # shellcheck disable=SC2016  # fenced backticks are intentional Markdown
  REVIEW_REPORT_TAIL="$(printf '\n\n<details><summary>Post-green agent review report (tail)</summary>\n\n```\n%s\n```\n\n</details>' \
    "$(tail -n 20 "$ROOT/.deps/llama-canary-review-report.md")")"
  REVIEW_HEAD="$(git rev-parse HEAD)"
}

repair_followup_prompt() {
  # Shared prompt for every repair turn. In battery mode turn 1 this is
  # seeded directly from the workflow's teed failure evidence (no battery
  # re-run first); later turns carry the wrapper's own certification output.
  # The agent has no GitHub credentials; the wrapper commits, pushes, and
  # updates the PR.
  printf 'The family certification battery failed after the patch-queue repair
at upstream %s (attempt %s of %s). You are working in this repository checkout.

Read ci/llama-canary/agent-repair-prompt.md and the repo skills it names, then
fix the root cause — do not weaken a failing lane. If a model is genuinely
broken by upstream, fix our patches or flag it in the PR body. The failing
battery output (tail):

%s

Re-run scripts/skippy-family-battery.sh --skip-build yourself to confirm your
fix, and leave your work in the working tree or on local commits — the wrapper
will commit, push, and update the repair PR.' \
    "$UPSTREAM_SHA" "$1" "$MAX_REPAIR_TURNS" "$(battery_summary "$BATTERY_LOG")"
}

publish_work_in_progress() {
  # Put the agent's current work on the repair PR early, so even a stuck run
  # leaves reviewable bytes behind. Best-effort: failures here do not stop
  # the repair loop.
  publish_repair_branch || echo "warning: could not publish repair branch" >&2
  ensure_pr >/dev/null
  apply_pr_body
}

if [[ "$MODE" == "patch-queue" ]]; then
  agent_turn "$(printf 'The canary failed to apply the llama.cpp patch queue at upstream %s.
Read ci/llama-canary/agent-repair-prompt.md in this repo and follow it exactly.
Commit your work locally when done. You have no GitHub credentials — the
wrapper that invoked you owns all pushes and PR updates. Reuse open PR %s on
branch %s if listed.' \
    "$UPSTREAM_SHA" "${EXISTING_PR:-none}" "$BRANCH")"

  echo "agent repair turn finished; verifying queue applies..."
  if ! prepare_repair_target; then
    publish_work_in_progress
    pr_comment "$(printf '**Repair stuck — needs human assistance.** The patch queue still does not apply through the checked-in pin at upstream %s after the agent repair turn (see the canary run log for the failing patch). The agent work so far is on this branch.' \
      "$UPSTREAM_SHA")"
    exit 1
  fi
else
  # battery mode: the queue already applies and the workflow's own battery
  # step just failed on this runner. Its evidence log (teed to
  # $BATTERY_LOG by the workflow) seeds the first repair turn, so no build
  # or battery run is repeated before the agent gets the failure output.
  if ! prepare_repair_target; then
    publish_work_in_progress
    pr_comment "$(printf '**Repair stuck — needs human assistance.** The battery-mode repair no longer applies through the checked-in pin at upstream %s. The agent work so far is on this branch.' \
      "$UPSTREAM_SHA")"
    exit 1
  fi
  if [[ ! -s "$BATTERY_LOG" ]]; then
    echo "battery mode: no workflow battery evidence at $BATTERY_LOG; running one diagnostic battery attempt..." >&2
    run_battery || true
  else
    echo "battery mode: reusing workflow battery evidence from $BATTERY_LOG"
  fi
fi

# Publish the agent's repair work and its PR before certification, so the
# PR-description agent turn (draft_pr_body) also runs strictly before any
# certification attempt — no agent turn ever runs after a green battery.
publish_work_in_progress
draft_pr_body
apply_pr_body

# Certify → repair → recertify loop. The wrapper — not the agent — decides
# when certification passes, so a lane failure can never be talked past.
# In battery mode the workflow's own battery step already failed on this
# runner (or the diagnostic attempt above did): iteration 1 is the repair
# turn seeded from that evidence, never another full build+battery run
# before the agent gets a chance to fix anything.
attempt=0
while (( attempt < MAX_REPAIR_TURNS )); do
  attempt=$((attempt + 1))
  if [[ "$MODE" == "battery" && "$attempt" -eq 1 ]]; then
    echo "battery mode: repair turn 1 seeded from the workflow battery failure evidence"
  else
    echo "certification attempt $attempt..."
    if run_battery; then
      echo "family battery passed; repair complete"
      report_success
      exit 0
    fi
  fi
  echo "family battery failed on repair turn $attempt; handing failures to the agent"
  agent_turn "$(repair_followup_prompt "$attempt")"
  echo "agent repair turn $attempt finished; verifying queue applies..."
  if ! prepare_repair_target; then
    publish_work_in_progress
    pr_comment "$(printf '**Repair stuck — needs human assistance.** The patch queue regressed or still does not apply through the checked-in pin at upstream %s after repair turn %s/%s. The agent work is on this branch; see the canary run log for the failing patch.' \
      "$UPSTREAM_SHA" "$attempt" "$MAX_REPAIR_TURNS")"
    exit 1
  fi
done

echo "final certification attempt..."
if run_battery; then
  echo "family battery passed; repair complete"
  report_success
  exit 0
fi

publish_work_in_progress
# The final status comment embeds a fenced battery tail; the literal
# backticks are intentional Markdown, not command substitution.
# shellcheck disable=SC2016
pr_comment "$(printf '**Repair stuck — needs human assistance.** The family battery is still failing after %s agent repair turns at upstream %s. The agent work is on this branch; the failing battery output (tail):\n\n```\n%s\n```' \
  "$MAX_REPAIR_TURNS" "$UPSTREAM_SHA" "$(battery_summary "$BATTERY_LOG")")"
echo "family battery still failing after $MAX_REPAIR_TURNS agent repair turns" >&2
exit 1
