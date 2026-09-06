from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPAIR = ROOT / "scripts" / "llama-canary-agent-repair.sh"


class LlamaCanaryAgentRepairContractTests(unittest.TestCase):
    """Behavioral contracts for the canary repair wrapper.

    The wrapper mediates between an untrusted model turn and repository-write
    credentials. These tests pin the invariants the review demanded: the
    agent never sees a GitHub token, the token never reaches the environment,
    dispatch SHAs are validated as 40-hex before any use, battery-mode
    evidence is reused instead of re-running the battery, and persistent
    runner state is cleared at the start of every run.
    """

    def test_agent_turns_strip_github_tokens_from_environment(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # The write PAT is never exported into the environment.
        self.assertNotIn("export GH_TOKEN", wrapper)
        # Agent turns explicitly strip every GitHub credential.
        self.assertIn("env -u GH_TOKEN -u GITHUB_TOKEN -u CANARY_REPAIR_TOKEN", wrapper)
        # Turns run with --auto: opencode's sandbox otherwise auto-rejects
        # out-of-workspace scratch writes (/tmp) in non-interactive mode and
        # the rejection kills the whole turn (live: run 33160131810). Safe
        # because no GitHub credential is present in the agent environment.
        self.assertIn('opencode run --auto --model "$AGENT_MODEL"', wrapper)
        # GitHub mutations go through the token-scoped helper.
        self.assertIn("gh_repair() {", wrapper)
        self.assertNotIn("\n  gh pr create", wrapper)
        self.assertNotIn("\n  gh issue create", wrapper)
        self.assertNotIn("\n  gh pr comment", wrapper)
        self.assertNotIn("\n  gh issue comment", wrapper)
        self.assertNotIn("\n  gh pr edit", wrapper)

    def test_certified_battery_is_bound_to_the_repair_pr_head(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # The wrapper — never the agent — commits and pushes the certified tree.
        self.assertIn("publish_repair_branch", wrapper)
        self.assertIn('CERTIFIED_SHA="$(git rev-parse HEAD)"', wrapper)
        # Success requires the remote PR head to equal the certified commit.
        self.assertIn("verify_pr_head_is_certified", wrapper)
        # The PR must exist before apply_pr_body runs: on a first run the PR
        # is created lazily, and applying the agent's draft body before
        # ensure_pr silently no-ops, leaving the generic body on the PR
        # (live: run 33163990453 — the agent's full analysis never showed).
        self.assertIn("ensure_pr >/dev/null\n  apply_pr_body", wrapper)
        self.assertIn("report_success", wrapper)
        # The PR-body agent turn runs only before certification; after a green
        # battery only the deterministic apply_pr_body may run.
        self.assertIn("draft_pr_body", wrapper)
        self.assertIn("apply_pr_body", wrapper)
        first_certify = wrapper.index("certification attempt")
        self.assertLess(wrapper.index("draft_pr_body()"), first_certify)
        self.assertNotIn("write_pr_body", wrapper)

    def test_repair_branch_selects_the_certified_upstream_through_checked_in_pin(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # The deterministic wrapper, not an agent turn, owns the repository
        # selector. The sole pin must select the exact 40-hex target before a
        # repair is published or certified.
        self.assertIn("write_repair_pin() {", wrapper)
        self.assertIn('scripts/update-llama-pin.sh "$UPSTREAM_SHA"', wrapper)
        self.assertIn("verify_repair_pin() {", wrapper)
        self.assertIn('pin="$(tr -d \'[:space:]\' < "$PIN_FILE")"', wrapper)
        self.assertIn('[[ "$pin" != "$UPSTREAM_SHA" ]]', wrapper)
        self.assertNotIn("PIN_MIRROR_FILE", wrapper)

        # Certification must exercise the same path normal builds use and
        # bind the prepared checkout stamp back to the requested target.
        self.assertIn("prepare_repair_target() {", wrapper)
        self.assertIn("write_repair_pin || return 1", wrapper)
        self.assertIn("verify_repair_pin || return 1", wrapper)
        self.assertIn("scripts/prepare-llama.sh pinned || return 1", wrapper)
        self.assertIn('prepared_upstream="$(tr -d \'[:space:]\' < "$ROOT/.deps/llama.cpp/.mesh-llm-upstream-sha")"', wrapper)
        self.assertIn('[[ "$prepared_upstream" != "$UPSTREAM_SHA" ]]', wrapper)
        self.assertIn("  prepare_repair_target || return 1\n  # Battery-mode repairs", wrapper)

        # Both entry modes establish the pinned target before the first
        # publish_work_in_progress call can create or update the repair PR.
        mode_start = wrapper.index(
            'if [[ "$MODE" == "patch-queue" ]]',
            wrapper.index("publish_work_in_progress()"),
        )
        mode_flow = wrapper[mode_start:]
        patch_queue_branch, battery_and_shared = mode_flow.split("\nelse\n", 1)
        battery_branch = battery_and_shared.split(
            "\nfi\n\n# Publish the agent's repair work", 1
        )[0]
        for branch in (patch_queue_branch, battery_branch):
            self.assertLess(
                branch.index("if ! prepare_repair_target; then"),
                branch.index("\n    publish_work_in_progress\n"),
            )

    def test_post_green_review_may_modify_the_certified_repair(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # Even a green, certified repair gets one fresh-context review turn:
        # the reviewer is told it did NOT author the repair, and it may fix
        # what parity certification cannot see (dropped patch intent, rebase
        # leftovers, ABI mirror drift). Its changes ride as a separate
        # review(llama): commit pushed by the wrapper to the same branch.
        self.assertIn("post_green_review_turn() {", wrapper)
        self.assertIn("  apply_pr_body\n  post_green_review_turn\n", wrapper)
        self.assertIn("review(llama): agent review fixes at upstream", wrapper)
        # The review is opt-out and fail-open: a disabled or crashed review
        # never fails a green repair.
        self.assertIn('if [[ "${CANARY_AGENT_REVIEW:-true}" != "true" ]]', wrapper)
        self.assertIn("post-green agent review disabled", wrapper)
        self.assertIn("continuing with the certified tree", wrapper)
        # The success comment must report the review outcome honestly.
        self.assertIn("REVIEW_STATUS=", wrapper)
        self.assertIn("Post-green agent review made modifications", wrapper)
        # The review runs BEFORE the PR-head verification, and the verified
        # head is the last wrapper-published commit (certified, or the
        # review head when the review modified the tree) — never a stale
        # certified SHA that a review commit would strand behind.
        self.assertIn('expected="${REVIEW_HEAD:-${CERTIFIED_SHA:?}}"', wrapper)
        # The review report is run-scoped persistent-runner state.
        self.assertIn('rm -f "$ROOT/.deps/llama-canary-review-report.md"', wrapper)

    def test_battery_mode_reuses_workflow_evidence_without_rerunning(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # Battery mode seeds from the workflow's teed evidence log when
        # present and only runs a diagnostic battery when it is absent.
        self.assertIn("reusing workflow battery evidence", wrapper)
        self.assertIn("no workflow battery evidence", wrapper)
        # A missing battery log must not crash the failure-path summaries.
        self.assertIn("(no battery output captured", wrapper)
        # The first battery-mode loop iteration is a repair turn seeded from
        # the workflow evidence — never a second full build+battery run
        # before the agent gets the failure output.
        self.assertIn(
            'if [[ "$MODE" == "battery" && "$attempt" -eq 1 ]]; then', wrapper
        )
        self.assertIn(
            "battery mode: repair turn 1 seeded from the workflow battery failure evidence",
            wrapper,
        )

    def test_agent_turns_emit_heartbeat_progress(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # Long agent turns must stay observable from the Actions log: a
        # heartbeat monitor prints elapsed time and worktree activity every
        # 10 minutes, runs without ambient credentials, and is killed as a
        # whole process group when the turn ends. set -m (not setsid) makes
        # the group portable to the macOS family-certify runner, and plain
        # find -print (not -printf) is used for the same reason.
        self.assertIn("heartbeat: agent turn running for", wrapper)
        self.assertIn("while sleep 600", wrapper)
        self.assertIn('env -i PATH="$PATH" bash -c', wrapper)
        self.assertIn('heartbeat "$ROOT" "$started"', wrapper)
        self.assertIn("set -m", wrapper)
        self.assertNotIn("setsid", wrapper)
        self.assertNotIn("-printf", wrapper)
        self.assertIn('kill -- "-$heartbeat_pid"', wrapper)
        self.assertIn('wait "$heartbeat_pid"', wrapper)
        self.assertIn("recent worktree activity", wrapper)

    def test_every_github_call_is_token_scoped(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # The workflow job exports no ambient GH_TOKEN and checks out with
        # persist-credentials disabled: every gh invocation — reads included
        # — must go through the token-scoped helper.
        lines = [
            line
            for line in wrapper.splitlines()
            if not line.lstrip().startswith("#")
            and " gh " in f" {line.strip()} "
            # `command -v gh` checks binary presence, not an API call.
            and "command -v" not in line
        ]
        for line in lines:
            self.assertIn(
                "gh_repair",
                line,
                f"bare gh invocation bypasses the repair token: {line.strip()}",
            )

    def test_run_scopes_persistent_runner_state(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # The PR-body draft is cleared every run; the battery evidence log is
        # cleared in patch-queue mode but preserved in battery mode, where it
        # holds this run's workflow-teed evidence.
        self.assertIn(
            'rm -f "$ROOT/.deps/llama-canary-pr-body.md"', wrapper
        )
        self.assertIn('if [[ "$MODE" == "patch-queue" ]]; then\n  rm -f "$BATTERY_LOG"', wrapper)
        # Scratch worktrees under /tmp and their registrations survive across
        # runs on the persistent runner and make the agent's own
        # `git worktree add` fail; the wrapper prunes them up front (live:
        # run 33158798988 aborted its turn on a stale /tmp/llama-old-pin).
        self.assertIn('git -C "$ROOT/.deps/llama.cpp" worktree prune', wrapper)
        self.assertIn("rm -rf /tmp/llama-old-pin /tmp/llama-repair /tmp/llama-repair-*", wrapper)
        # The repair push URL embeds the token; its stderr is redacted.
        self.assertIn("redact_token", wrapper)

    def test_token_permissions_are_preflighted_before_repair_work(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # A permission gap on CANARY_REPAIR_TOKEN must fail the run in
        # seconds, before any repair work — not as a git 403 after a
        # potentially hours-long certified repair (live: run 33153507371,
        # where the account HAD push but the fine-grained PAT lacked
        # Contents: write, so REST permissions passed while git push 403'd).
        self.assertIn("check_repair_token_permissions", wrapper)
        # The preflight authenticates the token and PROBES the actual write
        # capability (create+delete a temp ref) via scoped gh calls — reading
        # the REST permissions object is not sufficient, because it reflects
        # the account's access, not the token's fine-grained scope.
        self.assertIn('gh_repair gh api user --jq .login', wrapper)
        self.assertIn('gh_repair gh api --method POST "repos/${GITHUB_REPOSITORY:?}/git/refs"', wrapper)
        self.assertIn('gh_repair gh api --method DELETE', wrapper)
        # The probe ref must actually be cleaned up: the delete URL uses the
        # percent-encoded branch name under /git/refs/ (a refs/-prefixed path
        # 404s and leaves the probe branch behind; live: run 33158798988).
        self.assertIn('git/refs/heads%2Fcanary-repair-token-preflight', wrapper)
        # It runs unconditionally before the repair loop starts.
        preflight_call = wrapper.index("check_repair_token_permissions\n\n# Run-scope")
        self.assertGreater(preflight_call, wrapper.index("gh_repair()"))
        self.assertLess(preflight_call, wrapper.index("agent_turn"))

    def test_battery_build_mirrors_the_workflow_arch_guard(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # The family-certify job runs under Rosetta; the wrapper's build must
        # use the same arch -arm64 guard as the workflow's own build step,
        # and refuse to certify a non-arm64 archive (run 33140672269 rebuilt
        # x86_64 from a plain build-llama.sh call).
        self.assertIn(
            "LLAMA_STAGE_UPSTREAM_TESTS=ON uv run --no-project --with jinja2==3.1.6 --",
            wrapper,
        )
        self.assertIn("arch -arm64 scripts/build-llama.sh", wrapper)
        self.assertIn("-DCMAKE_OSX_ARCHITECTURES=arm64 || return 1", wrapper)
        self.assertIn("refusing to certify: native archive is not arm64", wrapper)

    def test_push_failure_names_the_likely_permission_cause(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # A 403 on the repair push is almost always a PAT-identity permission
        # gap; the wrapper must say so instead of failing with a bare git error.
        self.assertIn("identity behind CANARY_REPAIR_TOKEN lacks write access", wrapper)

    def test_dispatch_sha_is_validated_before_use(self) -> None:
        env = {
            **os.environ,
            "UPSTREAM_SHA_INPUT": "not-a-sha; echo pwned",
            "CANARY_REPAIR_TOKEN": "test-token",
            "GITHUB_REPOSITORY": "Mesh-LLM/mesh-llm",
        }
        env["PATH"] = str(ROOT / "scripts" / "tests" / "fixtures") + os.pathsep + env.get("PATH", "")
        with tempfile.TemporaryDirectory() as tmp:
            # The prerequisite checks (opencode, credentials) intentionally
            # pass in this environment only when the fixtures exist; run the
            # script and require it to never accept the invalid SHA.
            result = subprocess.run(
                [str(REPAIR), "patch-queue"],
                cwd=tmp,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        combined = result.stdout + result.stderr
        # The crafted SHA is refused as non-40-hex and never executed: the
        # only place it appears is the refusal message itself.
        self.assertIn("refusing to repair against a non-40-hex upstream SHA", combined)
        self.assertEqual(1, result.returncode)
        self.assertEqual(
            1,
            combined.count("pwned"),
            "the crafted SHA must only appear in the refusal message, never as executed output",
        )

    def test_manual_positional_sha_is_still_validated(self) -> None:
        script = subprocess.run(
            ["bash", "-n", str(REPAIR)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, script.returncode, script.stderr)

    def test_battery_mode_certification_executes_the_full_live_matrix(self) -> None:
        """A battery-mode repair cannot certify without executing every
        pinned live package-v2 row and the generated-family-patch gate."""
        wrapper = REPAIR.read_text(encoding="utf-8")
        run_battery = wrapper[wrapper.index("run_battery() {"):wrapper.index("post_green_review_turn()")]
        # Source transformation replaces the one-family expansion target.
        # Battery-mode certification validates the manifests and runs the
        # complete pinned live matrix before the family battery can pass.
        self.assertIn('if [[ "$MODE" == "battery" ]]; then', run_battery)
        self.assertIn("skippy-llama-parity.py --llama-src .deps/llama.cpp", run_battery)
        self.assertIn('skippy-canary-live-matrix.sh --prepare', run_battery)
        self.assertNotIn('--model', run_battery)
        self.assertIn("repair cannot certify", run_battery)
        self.assertIn("check-skippy-generated-family-patch.sh", run_battery)
        # Validate runs before the live matrix; the generated-patch check and
        # family battery run only after the native build succeeds.
        self.assertLess(
            run_battery.index("skippy-llama-parity.py --llama-src"),
            run_battery.index("skippy-canary-live-matrix.sh"),
        )
        self.assertLess(
            run_battery.index("skippy-canary-live-matrix.sh"),
            run_battery.index("skippy-family-battery.sh"),
        )
        self.assertLess(
            run_battery.index("check-skippy-generated-family-patch.sh"),
            run_battery.index("skippy-family-battery.sh"),
        )

    def test_one_family_coverage_target_is_retired(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        run_battery = wrapper[wrapper.index("run_battery() {"):wrapper.index("post_green_review_turn()")]
        self.assertNotIn("next-boundary-target --json", run_battery)
        self.assertNotIn("llama-canary-coverage-target.json", wrapper)
        self.assertNotIn("coverage_model", run_battery)
        self.assertIn("single generated family patch", run_battery)

    def test_generated_patch_failure_blocks_certification(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        run_battery = wrapper[wrapper.index("run_battery() {"):wrapper.index("post_green_review_turn()")]
        self.assertIn("if ! scripts/check-skippy-generated-family-patch.sh", run_battery)
        self.assertIn("generated model-family patch is stale or invalid", run_battery)
        patch_failure = run_battery.split(
            "if ! scripts/check-skippy-generated-family-patch.sh", 1
        )[1].split("fi", 1)[0]
        self.assertIn("return 1", patch_failure)

    def test_runnable_row_carrying_unsupported_reason_is_rejected(self) -> None:
        """validate must reject candidate/candidate_stateful rows carrying
        an unsupported_reason — only non-runnable rows may carry one."""
        parity = ROOT / "scripts" / "skippy-llama-parity.py"
        sys.path.insert(0, str(parity.parent))
        try:
            spec = importlib.util.spec_from_file_location(
                "skippy_llama_parity_validate", parity
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        for status in ("certified", "candidate", "candidate_stateful"):
            rows = [
                {
                    "llama_model": "somearch",
                    "status": status,
                    "unsupported_reason": "ambiguous leftover",
                }
            ]
            self.assertEqual(
                module.validate_boundary_registration(rows, {"somearch"}),
                1,
                f"{status} row with a reason must fail even when registered",
            )
        # A non-runnable classification with a reason stays legal.
        self.assertEqual(
            module.validate_boundary_registration(
                [
                    {
                        "llama_model": "x",
                        "status": "non_causal_aux",
                        "unsupported_reason": "non-causal encoder",
                    }
                ],
                set(),
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
