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

# 1. opencode analyzes the regression evidence and attempts a fix.
opencode run --mode agent \
  "The nightly agentic replay benchmark on micstudio regressed. Evidence: $OUTPUT_DIR/summary/history.jsonl and per-model artifacts in $OUTPUT_DIR. Analyze the regression, identify the offending change (git log origin/main is available), and attempt a minimal fix. Do not touch ci/agentic-replay-nightly baselines or thresholds." || true

if git diff --quiet; then
  echo "opencode produced no changes" >&2
else
  git add -A
  git commit -m "fix: agentic replay nightly regression (run ${GITHUB_RUN_ID:-local})

Attempted automated repair by opencode from nightly run evidence.

Co-authored-by: opencode <opencode@meshllm.invalid>"
  git push origin "$BRANCH"

  # 2. Re-run the benchmark on the repaired tree (single pass, gate armed).
  if python3 scripts/agentic-replay-history.py \
      --matrix ci/agentic-replay-nightly/matrix.json \
      --summary-dir "$OUTPUT_DIR/summary" \
      --hardware "$OUTPUT_DIR/hardware.json" \
      --source-sha "$(git rev-parse HEAD)" \
      --output "$OUTPUT_DIR/summary/history-repair.jsonl" \
      $( [[ -d .replay-history-cache/data/runs ]] && echo --baseline .replay-history-cache/data/runs ) \
      --gate; then
    RESOLVED=1
  fi
fi

# 3. Always open a PR with the evidence; flag resolution status.
LABELS="agentic-replay,nightly"
TITLE="Agentic replay nightly regression — run ${GITHUB_RUN_ID:-local}"
if [[ "$RESOLVED" == "1" ]]; then
  TITLE="$TITLE (fix verified)"
  BODY=$'The nightly agentic replay regressed; opencode analyzed the evidence and this fix passes the re-run benchmark.\n\nResults (HF card format) are linked in the run report artifact and the dataset shard.'
else
  LABELS="$LABELS,needs-attention"
  TITLE="$TITLE — NEEDS ATTENTION"
  BODY=$'The nightly agentic replay regressed and the automated repair did not clear it. Review the evidence: history.jsonl, per-model artifacts, and the HF dataset shard for this run.'
fi

gh pr create --title "$TITLE" --body "$BODY" --label "$LABELS" --base main || \
  echo "PR creation failed — evidence retained in run artifacts" >&2
exit 0 # the nightly is red; the PR is the actionable output
