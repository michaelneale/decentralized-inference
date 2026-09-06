from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "llama-upstream-canary.yml"
PARITY = ROOT / "scripts" / "skippy-llama-parity.py"
UPDATE_PIN = ROOT / "scripts" / "update-llama-pin.sh"
BATTERY = ROOT / "scripts" / "skippy-family-battery.sh"
BATTERY_PLANNER = ROOT / "scripts" / "plan-family-battery.py"
FAMILY_CERTIFY = ROOT / "scripts" / "family-certify.sh"
FAMILY_OUTCOME = ROOT / "scripts" / "lib" / "family-outcome.sh"
TIMEOUT_RUNNER = ROOT / "scripts" / "run-command-with-timeout.py"


def _step_block(workflow: str, name: str) -> str:
    marker = f"      - name: {name}\n"
    start = workflow.index(marker)
    end = workflow.find("\n      - name: ", start + len(marker))
    return workflow[start:] if end == -1 else workflow[start:end]


class ParityCliInvocationTests(unittest.TestCase):
    """Executable contract: the exact parity invocations the workflow and
    repair wrapper use must parse (argparse rejects a global option placed
    after the subcommand — seen live as exit 2)."""

    def _temp_llama_src(self) -> tempfile.TemporaryDirectory:
        # Hermetic: CI's quality job has no .deps/llama.cpp checkout. A minimal
        # source tree with real boundary hooks is enough — argparse errors
        # precede any source validation, so the global-first property under
        # test is unchanged.
        tmp = tempfile.TemporaryDirectory(prefix="parity-cli-src-")
        models = Path(tmp.name) / "src" / "models"
        models.mkdir(parents=True)
        (models / "llama.cpp").write_text(
            "void f() {\n"
            "    begin_block(x, 0);\n"
            "    end_block(y, 0);\n"
            "}\n",
            encoding="utf-8",
        )
        self.addCleanup(tmp.cleanup)
        return tmp

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(PARITY), *args],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )

    def test_validate_global_llama_src_before_subcommand(self) -> None:
        tmp = self._temp_llama_src()
        result = self._run("--llama-src", tmp.name, "validate")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_next_boundary_target_global_llama_src_before_subcommand(self) -> None:
        tmp = self._temp_llama_src()
        result = self._run(
            "--llama-src", tmp.name, "next-boundary-target", "--json"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # Valid single-target JSON (or null), not an argparse usage error.
        payload = json.loads(result.stdout.strip() or "null")
        self.assertTrue(payload is None or "llama_model" in payload)

    def test_workflow_and_wrapper_use_valid_invocations(self) -> None:
        # The exact command strings embedded in the workflow and wrapper
        # must be the valid global-first form, never the rejected
        # subcommand-first form.
        workflow = WORKFLOW.read_text(encoding="utf-8")
        wrapper = (ROOT / "scripts" / "llama-canary-agent-repair.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "skippy-llama-parity.py --llama-src .deps/llama.cpp validate", workflow
        )
        self.assertIn(
            "skippy-llama-parity.py --llama-src .deps/llama.cpp next-boundary-target",
            workflow,
        )
        self.assertIn(
            "skippy-llama-parity.py --llama-src .deps/llama.cpp \\", wrapper
        )
        for text in (workflow, wrapper):
            self.assertNotIn("validate --llama-src", text)
            self.assertNotIn("--json --llama-src", text)


class LlamaUpstreamCanaryWorkflowTests(unittest.TestCase):
    def test_workflow_runs_only_daily_or_by_manual_dispatch(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('    - cron: "47 3 * * *"', workflow)
        self.assertIn("  workflow_dispatch:", workflow)
        self.assertNotIn("\n  push:", workflow)

    def test_workflow_builds_binaries_before_skipping_per_lane_builds(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("force_certify:", workflow)
        self.assertIn("FORCE_CERTIFY:", workflow)
        self.assertIn(
            "LLAMA_BUILD_DIR: ${{ github.workspace }}/.deps/llama-canary-${{ github.run_id }}-${{ github.run_attempt }}",
            workflow,
        )
        self.assertIn(
            "LLAMA_STAGE_BUILD_DIR: ${{ github.workspace }}/.deps/llama-canary-${{ github.run_id }}-${{ github.run_attempt }}",
            workflow,
        )
        self.assertIn("LLAMA_STAGE_BACKEND: metal", workflow)
        self.assertNotIn("mozilla-actions/sccache-action", workflow)
        self.assertNotIn("SCCACHE_GHA_ENABLED", workflow)
        self.assertNotIn("SCCACHE_C_CUSTOM_CACHE_BUSTER", workflow)
        self.assertIn('RUSTC_WRAPPER: ""', workflow)
        self.assertIn('LLAMA_STAGE_USE_SCCACHE: "0"', workflow)
        self.assertNotIn("Show sccache stats", workflow)

        native_build = _step_block(workflow, "Build patched llama.cpp ABI")
        self.assertIn(
            "arch -arm64 bash scripts/build-llama.sh -DCMAKE_OSX_ARCHITECTURES=arm64",
            native_build,
        )

        family_plan = _step_block(workflow, "Plan and verify family certification cache")
        self.assertIn("python3 scripts/plan-family-battery.py", family_plan)
        self.assertIn('--cadence "${{ steps.sha.outputs.cadence }}"', family_plan)
        self.assertIn("--check-cache", family_plan)
        self.assertIn('--cache-root "$HF_CACHE"', family_plan)
        self.assertIn('--github-output "$GITHUB_OUTPUT"', family_plan)
        self.assertLess(workflow.index(family_plan), workflow.index(native_build))

        build = _step_block(workflow, "Build stage runtime crates")
        self.assertIn("cargo build", build)
        self.assertIn("steps.sha.outputs.certify == 'true'", build)
        self.assertNotIn("-p skippy-ffi", build)
        for package in (
            "skippy-correctness",
            "skippy-server",
            "skippy-model-package",
            "llama-spec-bench",
        ):
            self.assertIn(f"-p {package}", build)

        architecture = _step_block(workflow, "Verify native archive architecture")
        self.assertIn('lipo -archs "$archive"', architecture)
        self.assertIn('[[ "$arches" != "arm64" ]]', architecture)

        battery = _step_block(
            workflow, "Supported-families certification battery (parity gate)"
        )
        self.assertIn("scripts/skippy-family-battery.sh --skip-build --plan", battery)
        self.assertIn("steps.sha.outputs.certify == 'true'", battery)
        self.assertIn("FAMILY_BATTERY_RUN_ID:", battery)
        self.assertIn("timeout-minutes: 720", battery)

        upload = _step_block(workflow, "Upload supported-families battery evidence")
        self.assertIn("if: ${{ !cancelled()", upload)
        self.assertIn("actions/upload-artifact@", upload)
        self.assertIn("target/family-battery/", upload)
        self.assertIn("retention-days: 14", upload)

        capture = _step_block(workflow, "Capture upstream SHAs")
        self.assertIn('"$FORCE_CERTIFY" == "true"', capture)
        self.assertIn('echo "certify=true"', capture)
        self.assertIn('echo "cadence=manual-full"', capture)
        self.assertIn('echo "cadence=llama-bump"', capture)
        self.assertIn('"$GITHUB_EVENT_NAME" == "schedule"', capture)
        self.assertIn('echo "cadence=nightly"', capture)

        forced_report = _step_block(workflow, "Report forced certification result")
        self.assertIn("steps.sha.outputs.cadence == 'manual-full'", forced_report)

        nightly_report = _step_block(workflow, "Report nightly family result")
        self.assertIn("steps.sha.outputs.cadence == 'nightly'", nightly_report)
        self.assertIn("steps.family_plan.outputs.family_count", nightly_report)

    def test_persistent_runner_executes_only_trusted_main_with_read_access(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        latest_job = workflow[workflow.index("  latest-upstream:") : workflow.index("  update-pin:")]
        update_job = workflow[workflow.index("  update-pin:") :]
        self.assertIn("runs-on: [self-hosted, family-certify]", latest_job)
        self.assertIn("permissions:\n      contents: read", latest_job)
        self.assertIn("ref: main", latest_job)
        self.assertNotIn("queue_ref", workflow)
        self.assertNotIn("github.token", latest_job)
        self.assertIn("runs-on: ubuntu-latest", update_job)
        self.assertIn("permissions:\n      contents: write", update_job)
        self.assertIn("trusted_queue_sha", update_job)

    def test_update_job_writes_the_single_upstream_pin(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        update_step = _step_block(workflow, "Commit validated upstream pin to main")
        self.assertIn('scripts/update-llama-pin.sh "$VALIDATED_SHA"', update_step)
        self.assertIn(
            "git add third_party/llama.cpp/upstream.txt",
            update_step,
        )
        self.assertNotIn("LLAMA_CPP_SHA", update_step)

    def test_update_pin_script_writes_pin_and_rejects_invalid_sha(self) -> None:
        updater = UPDATE_PIN.read_text(encoding="utf-8")
        self.assertNotIn("LLAMA_CPP_SHA", updater)
        self.assertNotIn("LLAMA_PIN_MIRROR_FILE", updater)
        target = "a" * 40
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            pin = temp / "upstream.txt"
            env = {
                **os.environ,
                "LLAMA_PIN_FILE": str(pin),
            }
            result = subprocess.run(
                [str(UPDATE_PIN), target],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual(target + "\n", pin.read_text(encoding="utf-8"))

            invalid = subprocess.run(
                [str(UPDATE_PIN), "not-a-sha"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(1, invalid.returncode)
            self.assertIn("refusing to write a non-40-hex", invalid.stderr)
            self.assertEqual(target + "\n", pin.read_text(encoding="utf-8"))

            prepared_target = "b" * 40
            workdir = temp / "llama.cpp"
            workdir.mkdir()
            (workdir / ".mesh-llm-upstream-sha").write_text(
                prepared_target + "\n", encoding="utf-8"
            )
            prepared_env = {**env, "LLAMA_WORKDIR": str(workdir)}
            prepared = subprocess.run(
                [str(UPDATE_PIN)],
                cwd=ROOT,
                env=prepared_env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, prepared.returncode, prepared.stderr)
            self.assertEqual(prepared_target + "\n", pin.read_text(encoding="utf-8"))

    def test_repair_loop_is_wired_for_both_failure_modes(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        queue_repair = _step_block(workflow, "Agent repair loop (patch-queue failure)")
        self.assertIn("steps.prepare.outcome == 'failure'", queue_repair)
        self.assertIn("scripts/llama-canary-agent-repair.sh patch-queue", queue_repair)
        # The repair loop must not silently turn the canary green.
        self.assertIn("continue-on-error: true", queue_repair)

        battery_repair = _step_block(workflow, "Agent repair loop (battery failure)")
        self.assertIn("steps.prepare.outcome == 'success'", battery_repair)
        self.assertIn("steps.battery.outcome == 'failure'", battery_repair)
        self.assertIn("steps.sha.outputs.cadence != 'nightly'", battery_repair)
        self.assertIn("scripts/llama-canary-agent-repair.sh battery", battery_repair)
        self.assertIn("continue-on-error: true", battery_repair)

        # Any repair outcome keeps the run red: the certified fix must merge
        # through the repair PR before trusted main can certify.
        fail_step = _step_block(workflow, "Fail when the canary needs human attention")
        self.assertIn("steps.repair_queue.outcome", fail_step)
        self.assertIn("steps.repair_battery.outcome", fail_step)
        self.assertIn("CANARY_CADENCE:", fail_step)
        self.assertIn('[[ "$CANARY_CADENCE" == "nightly" ]]', fail_step)
        self.assertIn("repair agent was intentionally not invoked", fail_step)
        self.assertIn("exit 1", fail_step)

        # The battery lane itself no longer hard-fails the job before the
        # repair loop can run.
        battery = _step_block(workflow, "Supported-families certification battery (parity gate)")
        self.assertIn("continue-on-error: true", battery)

        # Both repair paths use the dedicated token, never the job token.
        self.assertNotIn("github.token", workflow[workflow.index("  latest-upstream:") : workflow.index("  update-pin:")])

        # Untrusted dispatch SHAs reach Bash only as environment variables.
        for repair_step in (queue_repair, battery_repair):
            self.assertIn("UPSTREAM_SHA_INPUT:", repair_step)
            self.assertIn("CANARY_REPAIR_TOKEN:", repair_step)
            # Extract the run: command body (everything after "run: |" or "run:")
            # to ensure inline interpolation checks only inspect shell commands,
            # not the env: mapping where ${{ }} is safe and intended.
            run_marker = repair_step.find("\n        run:")
            self.assertNotEqual(-1, run_marker, "repair step must have a run: key")
            run_body = repair_step[run_marker + len("\n        run:"):]
            self.assertNotIn("github.event.inputs", run_body)
            self.assertNotIn("steps.sha.outputs", run_body)

        # The workflow's battery step appends its evidence log (the parity
        # validation gate writes the head of the same file) so a
        # battery-mode repair turn reuses it instead of re-running.
        self.assertIn("tee -a .deps/llama-canary-repair-battery.log", battery)

        # The parity manifest validation gate runs before the family plan,
        # fail-closed, and its failure feeds the same battery repair loop.
        parity_gate = _step_block(
            workflow, "Parity manifest validation (boundary registration gate)"
        )
        self.assertIn("skippy-llama-parity.py --llama-src .deps/llama.cpp validate", parity_gate)
        self.assertIn("continue-on-error: true", parity_gate)
        gate_idx = workflow.index("Parity manifest validation (boundary registration gate)")
        plan_idx = workflow.index("Plan and verify family certification cache")
        self.assertLess(gate_idx, plan_idx, "parity gate must run before the family plan")
        self.assertIn("steps.parity_validate.outcome == 'failure'", battery_repair)
        self.assertIn("steps.parity_validate.outcome == 'failure'", fail_step)

        # The live package-v2 two-node matrix makes model_pin rows executable
        # evidence and routes failures to the same repair loop.
        live_matrix = _step_block(
            workflow, "Live package-v2 two-node matrix (model_pin proof)"
        )
        self.assertIn("scripts/skippy-canary-live-matrix.sh --prepare", live_matrix)
        self.assertIn("continue-on-error: true", live_matrix)
        self.assertIn("steps.live_matrix.outcome == 'failure'", battery_repair)
        self.assertIn("steps.live_matrix.outcome == 'failure'", fail_step)
        # The live step must build this run's exact producers (host binary +
        # patched native runtime) with an explicit backend — no cached
        # binary/bundle may supply the matrix.
        self.assertIn("SKIPPY_CANARY_LIVE_MATRIX_BACKEND", live_matrix)
        self.assertIn("metal", live_matrix)

        # A pending coverage-expansion target must actually trigger the
        # battery-mode repair loop: the step exits nonzero under
        # continue-on-error and its failure outcome feeds both the repair
        # condition and the final fail condition. The no-gap path must
        # remain successful (no exit 1 on that branch).
        coverage = _step_block(workflow, "Append coverage-expansion target (one family per run)")
        self.assertIn("id: coverage_target", coverage)
        self.assertIn("continue-on-error: true", coverage)
        self.assertIn("next-boundary-target --json", coverage)
        self.assertIn("tee -a .deps/llama-canary-repair-battery.log", coverage)
        self.assertIn("exit 1", coverage)
        # The ORIGINAL workflow-selected target is persisted before any
        # agent turn so the repair wrapper verifies exactly this row —
        # re-selecting after a graduated repair would advance to the next
        # queue entry and test the wrong model.
        self.assertIn(".deps/llama-canary-coverage-target.json", coverage)
        target_branch = coverage.split("if [[ \"$TARGET\"")[1].split("else")[0]
        no_gap_branch = coverage.split("else", 1)[1]
        self.assertIn("exit 1", target_branch)
        self.assertNotIn("exit 1", no_gap_branch)
        self.assertIn("no needs_boundary_registration gaps remain", no_gap_branch)
        # Only full certification cadences fail on a pending target;
        # nightly runs record + defer it.
        self.assertIn('!= "nightly" ]]', target_branch)
        self.assertIn("steps.coverage_target.outcome == 'failure'", battery_repair)
        self.assertIn("steps.coverage_target.outcome == 'failure'", fail_step)
        # Truthful success reporting: the pin-eligible and forced-cert
        # reports require parity + live + coverage all green alongside the
        # battery; nightly requires parity + live + battery (coverage is
        # recorded-only there).
        pin_report = _step_block(workflow, "Report upstream pin update")
        forced_report = _step_block(workflow, "Report forced certification result")
        nightly_report = _step_block(workflow, "Report nightly family result")
        for report in (pin_report, forced_report):
            self.assertIn("steps.battery.outcome == 'success'", report)
            self.assertIn("steps.parity_validate.outcome == 'success'", report)
            self.assertIn("steps.live_matrix.outcome == 'success'", report)
            self.assertIn("steps.coverage_target.outcome == 'success'", report)
        self.assertIn("steps.battery.outcome == 'success'", nightly_report)
        self.assertIn("steps.parity_validate.outcome == 'success'", nightly_report)
        self.assertIn("steps.live_matrix.outcome == 'success'", nightly_report)
        self.assertNotIn("steps.coverage_target.outcome == 'success'", nightly_report)
        # Live-matrix evidence lands under the uploaded battery evidence root.
        upload = _step_block(workflow, "Upload supported-families battery evidence")
        self.assertIn("target/family-battery/", upload)

    def test_post_green_agent_review_is_wired_and_opt_out(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        # After a certified repair, the wrapper runs one fresh-context review
        # turn that may modify the repair (a separate review(llama): commit).
        # Both repair steps pass the opt-out var with the same vars-pattern
        # as CANARY_AGENT_MODEL, defaulting to enabled.
        for repair_step in (
            _step_block(workflow, "Agent repair loop (patch-queue failure)"),
            _step_block(workflow, "Agent repair loop (battery failure)"),
        ):
            self.assertIn("CANARY_AGENT_REVIEW:", repair_step)
            self.assertIn(
                "CANARY_AGENT_REVIEW: ${{ vars.LLAMA_CANARY_AGENT_REVIEW || 'true' }}",
                repair_step,
            )

    def test_family_results_have_typed_failure_outcomes(self) -> None:
        certify = FAMILY_CERTIFY.read_text(encoding="utf-8")
        classifier = FAMILY_OUTCOME.read_text(encoding="utf-8")
        for outcome in (
            "timeout",
            "unsupported",
            "model-invalid",
            "harness",
            "mismatch",
            "runtime-error",
        ):
            self.assertIn(f"printf '{outcome}\\n'", classifier)
        self.assertIn("outcome:$outcome", certify)
        self.assertNotIn("timed out|timeout|", classifier)

    def test_outcome_classifier_uses_terminal_evidence_not_option_names(self) -> None:
        fixtures = [
            ("runtime-error", "+ tool --startup-timeout-secs 900\nlistener disconnected\n"),
            ("runtime-error", "+ tool --allow-mismatch\nlistener disconnected\n"),
            ("timeout", "stage 1 binary server did not become ready\n"),
            (
                "unsupported",
                "Unsupported: stage graph did not expose a stable output activation boundary\n",
            ),
            ("model-invalid", "missing tensor blk.5.ssm_in.weight\n"),
            ("mismatch", "authoritative token mismatch\n"),
            ("harness", "corpus file does not exist\n"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            log = Path(temp_dir) / "lane.log"
            for expected, evidence in fixtures:
                with self.subTest(expected=expected, evidence=evidence):
                    log.write_text(evidence, encoding="utf-8")
                    result = subprocess.run(
                        [
                            "bash",
                            "-c",
                            'source "$1"; classify_family_outcome fail "$2" ""',
                            "classifier-test",
                            str(FAMILY_OUTCOME),
                            str(log),
                        ],
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertEqual(expected, result.stdout.strip())

    def test_portable_timeout_runner_bounds_the_process_group(self) -> None:
        result = subprocess.run(
            [
                str(TIMEOUT_RUNNER),
                "--seconds",
                "1",
                "--label",
                "fixture",
                "--",
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(124, result.returncode)
        self.assertIn("fixture timed out after 1s", result.stderr)

    @unittest.skipIf(os.name == "nt", "POSIX process-group signal semantics")
    def test_timeout_runner_cleans_process_group_when_signalled(self) -> None:
        for received_signal in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(received_signal=received_signal):
                wrapper = subprocess.Popen(
                    [
                        str(TIMEOUT_RUNNER),
                        "--seconds",
                        "30",
                        "--label",
                        "signal-fixture",
                        "--",
                        sys.executable,
                        "-c",
                        "import os,time; print(os.getpid(), flush=True); time.sleep(30)",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                assert wrapper.stdout is not None
                child_pid = int(wrapper.stdout.readline())
                wrapper.send_signal(received_signal)
                _, stderr = wrapper.communicate(timeout=15)

                self.assertEqual(128 + received_signal, wrapper.returncode)
                self.assertIn(
                    f"signal-fixture received signal {received_signal}", stderr
                )
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)

    def test_timeout_runner_closes_manifest_stdin_for_children(self) -> None:
        result = subprocess.run(
            [
                str(TIMEOUT_RUNNER),
                "--seconds",
                "5",
                "--label",
                "stdin-fixture",
                "--",
                sys.executable,
                "-c",
                "import sys; raise SystemExit(0 if sys.stdin.read() == '' else 9)",
            ],
            input="a later manifest row\n",
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(0, result.returncode, result.stderr)


class SkippyFamilyBatteryTests(unittest.TestCase):
    @staticmethod
    def _manifest(model: dict[str, object]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "profiles": {
                    "full": {
                        "status": "certified",
                        "oracle": "local-monolithic",
                        "required_lanes": [
                            "single-step",
                            "chain",
                            "state-handoff",
                        ],
                    },
                    "package-oracle": {
                        "status": "certified",
                        "oracle": "independent-trace",
                        "required_lanes": [
                            "single-step",
                            "chain",
                            "state-handoff",
                        ],
                    },
                    "graph-only": {
                        "status": "provisional",
                        "oracle": "none",
                        "required_lanes": [
                            "graph-parse",
                            "tensor-ownership",
                            "stage-load",
                        ],
                    },
                },
                "cadences": ["llama-bump", "manual-full", "nightly", "rotating"],
            },
            "models": [model],
        }

    @staticmethod
    def _model(revision: str = "a" * 40) -> dict[str, object]:
        return {
            "family": "test-family",
            "profile": "full",
            "cadences": ["llama-bump", "manual-full"],
            "artifact": {
                "repo": "org/model",
                "revision": revision,
                "files": ["model.gguf"],
                "file_integrity": {
                    "model.gguf": {"size_bytes": 1, "blob_id": "b" * 64}
                },
                "selector": "Q4_K_M",
            },
            "execution": {
                "trunk_layers": 6,
                "mtp_layers": 0,
                "activation_width": 1024,
                "boundary_sweep_period": 0,
                "speculative_policy": "mtp-if-present",
            },
            "resources": {
                "runner_role": "family-certify",
                "cache_policy": "immutable-local",
                "estimated_model_bytes": 1024,
            },
            "notes": "fixture",
        }

    def _dry_run(
        self, *args: str, models: list[dict[str, object]] | None = None
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            bin_dir = temp / "bin"
            bin_dir.mkdir()
            for command in ("hf",):
                executable = bin_dir / command
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

            manifest = temp / "manifest.json"
            selected_models = models or [self._model()]
            policy = self._manifest(selected_models[0])
            policy["models"] = selected_models
            manifest.write_text(
                json.dumps(policy) + "\n", encoding="utf-8"
            )
            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
            return subprocess.run(
                [
                    str(BATTERY),
                    "--manifest",
                    str(manifest),
                    "--dry-run",
                    *args,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_battery_builds_once_then_skips_build_in_each_lane(self) -> None:
        result = self._dry_run()
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(1, result.stdout.count("cargo build -p skippy-correctness"))
        commands = [
            line
            for line in result.stdout.splitlines()
            if line.startswith(str(FAMILY_CERTIFY) + " ")
        ]
        self.assertEqual(1, len(commands))
        self.assertTrue(
            commands[0]
            .strip()
            .endswith(
                "--require-lanes --skip-build --skip-speculative"
            )
        )

    def test_family_battery_has_no_activation_wire_dtype_switches(self) -> None:
        script = BATTERY.read_text(encoding="utf-8")

        self.assertNotIn("--wire-dtype", script)
        self.assertNotIn("--wire-dtypes", script)
        self.assertNotIn("--strict-dtype", script)

    def test_dry_run_reconciles_every_planned_family(self) -> None:
        first = self._model()
        second = self._model()
        second["family"] = "second-family"
        result = self._dry_run(models=[first, second])
        self.assertEqual(0, result.returncode, result.stderr)
        commands = [
            line
            for line in result.stdout.splitlines()
            if line.startswith(str(FAMILY_CERTIFY) + " ")
        ]
        self.assertEqual(2, len(commands))
        self.assertIn("--family test-family", commands[0])
        self.assertIn("--family second-family", commands[1])

    def test_family_filter_limits_the_resolved_dry_run(self) -> None:
        selected = self._dry_run("--families", "test-family")
        self.assertEqual(0, selected.returncode, selected.stderr)
        self.assertIn("--family test-family", selected.stdout)

        omitted = self._dry_run("--families", "another-family")
        self.assertEqual(2, omitted.returncode)
        self.assertIn("unknown selected families: another-family", omitted.stderr)
        self.assertNotIn("--family test-family", omitted.stdout)

    def test_skip_build_omits_the_one_time_build(self) -> None:
        result = self._dry_run("--skip-build")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn("cargo build -p skippy-correctness", result.stdout)

    def test_mmproj_smoke_lane_runs_only_for_families_with_a_projector(self) -> None:
        result = self._dry_run("--skip-build")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn("mmproj", result.stdout)

        model = self._model()
        model["mmproj_artifact"] = {
            "repo": "org/model",
            "revision": "a" * 40,
            "files": ["mmproj-model-f16.gguf"],
            "file_integrity": {
                "mmproj-model-f16.gguf": {"size_bytes": 1, "blob_id": "b" * 64}
            },
            "selector": "f16",
        }
        with_mmproj = self._dry_run("--skip-build", models=[model])
        self.assertEqual(0, with_mmproj.returncode, with_mmproj.stderr)
        smokes = [
            line
            for line in with_mmproj.stdout.splitlines()
            if line.startswith("env SKIPPY_MM_MODEL=")
        ]
        self.assertEqual(1, len(smokes))
        self.assertIn("SKIPPY_MM_PROJECTOR=", smokes[0])
        self.assertIn("frontend::tests::multimodal", smokes[0])
        self.assertIn("--test-threads=1", smokes[0])
        self.assertIn("family battery complete: 1/1", with_mmproj.stdout)

    def test_mmproj_failure_is_accounted_separately_from_core_certification(self) -> None:
        script = BATTERY.read_text(encoding="utf-8")
        smoke_body = script.split("run_mmproj_smoke() {", 1)[1].split(
            "\n}\n\nrun_resolved_manifest()", 1
        )[0]

        self.assertIn("MM_SMOKE_FAILURE_COUNT=0", script)
        self.assertIn(
            "MM_SMOKE_FAILURE_COUNT=$((MM_SMOKE_FAILURE_COUNT + 1))",
            smoke_body,
        )
        self.assertNotIn("CERT_FAILURE_COUNT=$((CERT_FAILURE_COUNT + 1))", smoke_body)

    def test_mmproj_smoke_image_fixture_is_deterministic(self) -> None:
        fixture = (
            ROOT / "ci" / "llama-canary" / "fixtures" / "multimodal-smoke.png"
        )
        self.assertTrue(fixture.is_file())
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        self.assertEqual(
            "308ff69210df5efdcc7c79abd65f68f7ed8545f469222e0a3c7f774d074a5034",
            digest,
        )

    def test_preflight_pins_snapshot_and_limits_speculative_corpus_to_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            revision = "a" * 40
            model = (
                temp
                / "hf"
                / "hub"
                / "models--org--mtp-model"
                / "snapshots"
                / revision
                / "model.gguf"
            )
            model.parent.mkdir(parents=True)
            model.write_bytes(b"gguf-fixture")

            bin_dir = temp / "bin"
            bin_dir.mkdir()
            hf = bin_dir / "hf"
            hf.write_text('#!/bin/sh\nprintf "path=%s\\n" "$FAKE_MODEL_PATH"\n', encoding="utf-8")
            inspect = bin_dir / "skippy-model-package"
            complete_tensors = [
                {
                    "name": f"blk.{layer}.weight",
                    "layer_index": layer,
                    "role": "layer",
                    "ggml_type": 1,
                    "byte_size": 0,
                }
                for layer in range(5)
            ] + [
                {
                    "name": "blk.5.nextn.eh_proj.weight",
                    "layer_index": 5,
                    "role": "layer",
                    "ggml_type": 1,
                    "byte_size": 1024,
                },
                {
                    "name": "blk.5.nextn.enorm.weight",
                    "layer_index": 5,
                    "role": "layer",
                    "ggml_type": 1,
                    "byte_size": 0,
                },
                {
                    "name": "blk.5.nextn.hnorm.weight",
                    "layer_index": 5,
                    "role": "layer",
                    "ggml_type": 1,
                    "byte_size": 0,
                },
            ]
            complete_scan = json.dumps(
                {"tensor_count": len(complete_tensors), "tensors": complete_tensors},
                separators=(",", ":"),
            )
            inspect.write_text(
                f"#!/bin/sh\nprintf '%s\\n' '{complete_scan}'\n",
                encoding="utf-8",
            )
            spec = bin_dir / "llama-spec-bench"
            spec.write_text(
                "#!/bin/sh\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  if [ \"$1\" = --json-out ]; then printf '{}\\n' > \"$2\"; exit 0; fi\n"
                "  shift\n"
                "done\n"
                "exit 1\n",
                encoding="utf-8",
            )
            for name in ("hf", "skippy-model-package", "llama-spec-bench"):
                path = bin_dir / name
                path.chmod(path.stat().st_mode | stat.S_IXUSR)
            for name in ("skippy-correctness", "skippy-server"):
                path = bin_dir / name
                path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                path.chmod(path.stat().st_mode | stat.S_IXUSR)

            model_policy = self._model(revision)
            model_policy["family"] = "mtp-family"
            model_policy["artifact"] = {
                "repo": "org/mtp-model",
                "revision": revision,
                "files": ["model.gguf"],
                "file_integrity": {
                    "model.gguf": {"size_bytes": 1, "blob_id": "b" * 64}
                },
                "selector": "Q4_K_M",
            }
            model_policy["execution"] = {
                "trunk_layers": 5,
                "mtp_layers": 1,
                "activation_width": 1024,
                "boundary_sweep_period": 0,
                "speculative_policy": "mtp-if-present",
            }
            manifest = temp / "manifest.json"
            manifest.write_text(
                json.dumps(self._manifest(model_policy)) + "\n", encoding="utf-8"
            )
            artifacts = temp / "artifacts"
            env = os.environ.copy()
            env.pop("HF_CACHE", None)
            env.pop("HF_HUB_OFFLINE", None)
            env.update(
                {
                    "FAKE_MODEL_PATH": str(model),
                    "FAMILY_BATTERY_BIN_DIR": str(bin_dir),
                    "FAMILY_BATTERY_ARTIFACT_ROOT": str(artifacts),
                    "FAMILY_BATTERY_MIN_FREE_GIB": "0",
                    "HF_HOME": str(temp / "hf"),
                    "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
                }
            )
            result = subprocess.run(
                [
                    str(BATTERY),
                    "--manifest",
                    str(manifest),
                    "--preflight-only",
                    "--skip-build",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            run_dir = next(artifacts.iterdir())
            resolved = (run_dir / "resolved-models.tsv").read_text(encoding="utf-8")
            self.assertIn(revision, resolved)
            self.assertIn("|1|1024|5|", resolved)
            mtp_corpus = (run_dir / "mtp-corpus.tsv").read_text(encoding="utf-8")
            self.assertIn("mtp-family", mtp_corpus)
            self.assertTrue((run_dir / "preflight" / "speculative-smoke.json").is_file())

            incomplete_tensors = complete_tensors[:5] + [complete_tensors[5]]
            incomplete_scan = json.dumps(
                {"tensor_count": len(incomplete_tensors), "tensors": incomplete_tensors},
                separators=(",", ":"),
            )
            inspect.write_text(
                f"#!/bin/sh\nprintf '%s\\n' '{incomplete_scan}'\n", encoding="utf-8"
            )
            incomplete_artifacts = temp / "incomplete-artifacts"
            env["FAMILY_BATTERY_ARTIFACT_ROOT"] = str(incomplete_artifacts)
            incomplete = subprocess.run(
                [
                    str(BATTERY),
                    "--manifest",
                    str(manifest),
                    "--preflight-only",
                    "--skip-build",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(1, incomplete.returncode)
            incomplete_run = next(incomplete_artifacts.iterdir())
            incomplete_corpus = (incomplete_run / "mtp-corpus.tsv").read_text(
                encoding="utf-8"
            )
            self.assertEqual(
                ["family\tmodel_id\tsource_revision\tmodel_path\tmtp_layers"],
                incomplete_corpus.splitlines(),
            )
            self.assertFalse(
                (incomplete_run / "preflight" / "speculative-smoke.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
