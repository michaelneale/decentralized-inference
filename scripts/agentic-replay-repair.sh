#!/usr/bin/env bash
# Regression repair loop for the agentic-replay nightly.
# Mirrors scripts/llama-canary-agent-repair.sh: on a gated regression, give
# opencode the run evidence and let it analyze + attempt a fix, re-run the
# benchmark, then always open a PR — labelled either as a verified fix or as
# an unresolved regression that needs human attention.
set -euo pipefail

OUTPUT_DIR="${1:?usage: agentic-replay-repair.sh <output-dir>}"
BRANCH="agentic-replay-nightly/repair-${GITHUB_RUN_ID:-local}"
RESOLVED=0

git config user.name "mesh-replay-bot"
git config user.email "replay-bot@meshllm.invalid"
git checkout -b "$BRANCH"

# Repair evidence and downloaded history are inputs, never source changes.
REPO_ROOT=$(git rev-parse --show-toplevel)
EXCLUDE_FILE=$(git rev-parse --git-path info/exclude)
for artifact_root in "$OUTPUT_DIR" "${HISTORY_LOCAL:-$REPO_ROOT/.replay-history-cache}"; do
  if [[ "$artifact_root" == "$REPO_ROOT/"* ]]; then
    relative_root=${artifact_root#"$REPO_ROOT/"}
    printf '/%s/\n' "${relative_root%/}" >> "$EXCLUDE_FILE"
  fi
done

# 1. opencode analyzes the regression evidence and attempts a fix. It must not
# see GitHub credentials.
env -u GH_TOKEN -u GITHUB_TOKEN opencode run --mode agent \
  "The nightly agentic replay benchmark on micstudio regressed. Evidence: $OUTPUT_DIR/summary/history.jsonl and per-model artifacts in $OUTPUT_DIR. Analyze the regression, identify the offending change (git log origin/main is available), and attempt a minimal fix. Do not touch ci/agentic-replay-nightly baselines or thresholds." || true

if [[ -z "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "opencode produced no changes — needs-attention" >&2
  git commit --allow-empty -m "chore: agentic replay nightly regression needs attention (run ${GITHUB_RUN_ID:-local})

Automated repair produced no changes; PR opened for human triage with the
run evidence attached."
else
  git add -A
  git commit -m "fix: agentic replay nightly regression (run ${GITHUB_RUN_ID:-local})

Attempted automated repair by opencode from nightly run evidence.

Co-authored-by: opencode <opencode@meshllm.invalid>"

  # 2. Re-run the benchmark on the repaired tree with the nightly benchmark
  # shape, then re-normalize and gate the repaired summaries.
  LEVELS=$(python3 -c "import json;print(' '.join(map(str, json.load(open('ci/agentic-replay-nightly/matrix.json'))['replay']['concurrency'])))")
  LEVEL_ARGS=()
  for level in $LEVELS; do LEVEL_ARGS+=(--concurrency "$level"); done
  PASSES=$(python3 -c "import json;print(json.load(open('ci/agentic-replay-nightly/matrix.json'))['replay']['passes'])")
  REPLAY_DATASET_FILE="${DATASET_FILE:-${MESH_AGENTIC_REPLAY_DATASET_FILE:-}}"
  RERUN_FAILED=0
  if [[ -z "$REPLAY_DATASET_FILE" ]]; then
    echo "replay dataset file is unavailable — needs-attention" >&2
    RERUN_FAILED=1
  fi
  for family in $(python3 -c "import json;print(' '.join(m['family'] for m in json.load(open('ci/agentic-replay-nightly/matrix.json'))['models']))"); do
    if [[ "$RERUN_FAILED" == "1" && -z "$REPLAY_DATASET_FILE" ]]; then break; fi
    model_uri=$(python3 -c "import json;m=[m for m in json.load(open('ci/agentic-replay-nightly/matrix.json'))['models'] if m['family']=='$family'][0];print(m['repo']+'@'+m['revision']+'/'+m['file'])")
    python3 evals/agentic-replay.py run \
      --ref fixed=HEAD \
      --ref base=origin/main \
      --model "$model_uri" \
      --backend metal \
      --trajectories-per-framework 8 \
      "${LEVEL_ARGS[@]}" \
      --passes "$PASSES" \
      --warmup-turns 4 \
      --dataset-file "$REPLAY_DATASET_FILE" \
      --output "$OUTPUT_DIR/repair/$family" || RERUN_FAILED=1
  done
  HISTORY_ARGS=(
    --matrix ci/agentic-replay-nightly/matrix.json
    --replay-dir "$OUTPUT_DIR/repair"
    --label fixed
    --hardware "$OUTPUT_DIR/hardware.json"
    --source-sha "$(git rev-parse HEAD)"
    --replay <(python3 -c "import json;print(json.dumps(json.load(open('ci/agentic-replay-nightly/matrix.json'))['replay']))")
    --output "$OUTPUT_DIR/summary/history-repair.jsonl"
    --gate
  )
  HISTORY_RUNS="${HISTORY_LOCAL:-$REPO_ROOT/.replay-history-cache}/data/runs"
  if [[ -d "$HISTORY_RUNS" ]]; then
    HISTORY_ARGS+=(--baseline "$HISTORY_RUNS")
  fi
  if [[ "$RERUN_FAILED" == "0" ]] && python3 scripts/agentic-replay-history.py "${HISTORY_ARGS[@]}"; then
    RESOLVED=1
  fi
fi

# 3. Always open a PR with the evidence; flag resolution status.
LABELS="agentic-replay,nightly"
TITLE="Agentic replay nightly regression — run ${GITHUB_RUN_ID:-local}"
TEMPLATE_FILE="$(git rev-parse --show-toplevel)/.github/AGENTIC_REPLAY_REPAIR_PR_TEMPLATE.md"
if [[ "$RESOLVED" == "1" ]]; then
  TITLE="$TITLE (fix verified)"
  BODY=$'The nightly agentic replay regressed; opencode analyzed the evidence and this fix passes the re-run benchmark.\n\nResults (HF card format) are linked in the run report artifact and the dataset shard.'
else
  LABELS="$LABELS,needs-attention"
  TITLE="$TITLE — NEEDS ATTENTION"
  BODY=$'The nightly agentic replay regressed and the automated repair did not clear it. Review the evidence: history.jsonl, per-model artifacts, and the HF dataset shard for this run.'
fi

# Push with the token explicitly (no gh auth setup-git on this runner).
PUSH_REMOTE="https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY:-Mesh-LLM/mesh-llm}.git"
git push "$PUSH_REMOTE" "$BRANCH"

# Fill the PR template from the run evidence where available; fall back to
# the short body if the template or fill data is missing.
BODY_FILE=$(mktemp)
if [[ -f "$TEMPLATE_FILE" ]]; then
  sed -e "s|{{RESOLUTION_STATUS}}|$( [[ $RESOLVED == 1 ]] && echo 'fix verified' || echo 'NEEDS ATTENTION' )|" \
      -e "s|{{RUN_URL}}|${GITHUB_SERVER_URL:-}/Mesh-LLM/mesh-llm/actions/runs/${GITHUB_RUN_ID:-local}|g" \
      -e "s|{{RUN_DATE}}|$(date -u +%F)|g" \
      -e "s|{{RUN_ID}}|${GITHUB_RUN_ID:-local}|g" \
      -e "s|{{SOURCE_SHA}}|$(git rev-parse HEAD)|g" \
      -e "s|{{DATASET_REPO}}|${DATASET_REPO:-meshllm/agentic-replay-nightly}|g" \
      -e "s|{{REGRESSING_COHORTS}}|${REPAIR_REGRESSING_COHORTS:-unavailable}|g" \
      -e "s|{{GATE_OUTPUT}}|see run artifacts|g" \
      -e "s|{{DIAGNOSIS}}|see opencode session log|g" \
      -e "s|{{FIX_SUMMARY}}|$( git log -1 --format=%s HEAD )|g" \
      -e "s|{{FILES_CHANGED}}|$( git diff --name-only HEAD~1..HEAD | tr '\n' ' ' )|g" \
      -e "s|{{RATIONALE}}|automated repair attempt|g" \
      -e "s|{{RESULT_ROWS}}|see history-repair.jsonl artifact|g" \
      -e "s|{{RERUN_GATE_RESULT}}|$( [[ $RESOLVED == 1 ]] && echo 'pass' || echo 'fail' )|g" \
      -e "s|{{BOOTSTRAP_STATE}}|${REPAIR_BOOTSTRAP_STATE:-unavailable}|g" \
      "$TEMPLATE_FILE" >> "$BODY_FILE"
  printf '\n---\n%s\n' "$BODY" >> "$BODY_FILE"
else
  printf '%s\n' "$BODY" > "$BODY_FILE"
fi
gh pr create --title "$TITLE" --body-file "$BODY_FILE" --label "$LABELS" --base main --head "$BRANCH" || \
  echo "PR creation failed — evidence retained in run artifacts" >&2
rm -f "$BODY_FILE"
exit 0 # the nightly is red; the PR is the actionable output
