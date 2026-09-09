<!-- Template for PRs raised by the agentic-replay nightly repair loop.
     scripts/agentic-replay-repair.sh fills every {{PLACEHOLDER}} from the run
     artifact before creating the PR. Two variants: resolved / needs-attention. -->

## Agentic replay nightly — {{RESOLUTION_STATUS}}

<!-- RESOLUTION_STATUS is one of:
     "fix verified" (re-run passed) or "NEEDS ATTENTION" (repair did not clear the regression) -->

**Nightly run:** [{{RUN_URL}}] · {{RUN_DATE}} · source `{{SOURCE_SHA}}`
**Regressing cohorts:** {{REGRESSING_COHORTS}} (model × concurrency × metric with observed vs baseline)

## Regression evidence

<!-- The exact failing comparator output, verbatim from history.jsonl -->

```
{{GATE_OUTPUT}}
```

## Diagnosis

<!-- opencode's analysis of the offending change -->

{{DIAGNOSIS}}

<!-- Which commit introduced the regression and why it affects the measured path -->

## Fix

{{FIX_SUMMARY}}

- Changed: {{FILES_CHANGED}}
- Rationale: {{RATIONALE}}

## Re-run benchmark results (HF card format)

<!-- Same format as the nightly run report / dataset card; filled from the
     post-fix re-run summary. Omitted for needs-attention PRs where the fix
     failed or produced no changes. -->

| Model | Class | Conc. | Decode tok/s | Δ vs baseline | TTFT mean | Cache hit % | Finish=length % |
|---|---|---:|---:|---:|---:|---:|---:|
{{RESULT_ROWS}}

**Gate:** {{RERUN_GATE_RESULT}} (bootstrap state: {{BOOTSTRAP_STATE}})

## Provenance

| Field | Value |
|---|---|
| Runner | micstudio (M3 Ultra, 256 GB) — fingerprint verified |
| Nightly run | `{{RUN_ID}}` |
| History shard | `data/runs/{{RUN_DATE}}/{{RUN_ID}}.jsonl` in {{DATASET_REPO}} |
| Raw evidence | GitHub Actions artifact `replay-artifacts` (30-day retention) |
| Repair agent | opencode (agent mode), session log in run artifact |

## Reviewer checklist

- [ ] Diagnosis names a specific commit and mechanism (not just "made it faster")
- [ ] Fix does not touch `ci/agentic-replay-nightly/` baselines, thresholds, or the harness itself
- [ ] Re-run results table present (resolved PRs) or absence explained (needs-attention)
- [ ] `finish_reason_length_pct` did not shift — a budget change is not a throughput fix
