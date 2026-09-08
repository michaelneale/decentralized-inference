from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = ROOT / ".github" / "actions" / "capture-sccache-stats"
CAPTURE = ACTION_DIR / "capture.py"
SUMMARY = ROOT / "scripts" / "summarize-sccache-stats.py"
CONFIGURE_ACTION = (
    ROOT / ".github" / "actions" / "configure-sccache-gha" / "action.yml"
)
WORKFLOWS = {
    "quality": ROOT / ".github" / "workflows" / "ci-quality-slice.yml",
    "rust-tests": ROOT / ".github" / "workflows" / "ci-rust-tests-slice.yml",
    "host": ROOT / ".github" / "workflows" / "ci-linux-host-slice.yml",
    "runtime": ROOT / ".github" / "workflows" / "ci-linux-runtime-slice.yml",
    "static-abi": ROOT / ".github" / "workflows" / "static-abi-artifact.yml",
    "main": ROOT / ".github" / "workflows" / "ci.yml",
    "swift-sdk": ROOT / ".github" / "workflows" / "swift-sdk-artifact.yml",
}
INSTRUMENTED = WORKFLOWS.keys() - {"main"}
HF_WORKFLOW = ROOT / ".github" / "workflows" / "hf-download-smoke.yml"
NATIVE_SDK_WORKFLOW = (
    ROOT / ".github" / "workflows" / "native-sdk-artifact.yml"
)
SDK_SMOKE_WORKFLOW = ROOT / ".github" / "workflows" / "sdk-smoke.yml"
SEED_WARMER = ROOT / ".github" / "workflows" / "cache-warm-sccache.yml"
SEED_KEY_PATTERN = re.compile(
    r"mesh-llm-sccache-seed-[^\n]+-\$\{\{ hashFiles\('"
    r"Cargo\.lock', '\.github/cache-version\.txt', 'Justfile', 'just/\*\*'\) \}\}"
)
SEED_IMAGE = (
    "ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:"
    "8d93de6ba30173e825a16fdecf011f9c632edc6e1259df7289e491b0a05f829d"
)
SEED_EPOCH = (
    "mesh-llm-cuda-runner-sha256-"
    "8d93de6ba30173e825a16fdecf011f9c632edc6e1259df7289e491b0a05f829d"
)


def valid_payload(*, compile_requests: int = 12) -> dict[str, object]:
    return {
        "stats": {
            "compile_requests": compile_requests,
            "requests_executed": 10,
            "compilations": 4,
            "cache_writes": 3,
            "cache_read_errors": 0,
            "cache_write_errors": 0,
            "cache_hits": {
                "counts": {"Rust": 6},
                "adv_counts": {},
            },
            "cache_misses": {
                "counts": {"Rust": 4},
                "adv_counts": {},
            },
            "cache_errors": {
                "counts": {},
                "adv_counts": {},
            },
        },
        "version": "test",
    }


class SccacheEvidenceTests(unittest.TestCase):
    def test_configure_callers_declare_native_cache_policy(self) -> None:
        policy = "${{ needs.runner_policy.outputs.allow_native_github_cache }}"
        effective_release = (
            "${{ ((matrix.target == 'x86_64-unknown-linux-gnu' && "
            "startsWith(needs.metadata.outputs.runner_8, 'depot-')) || "
            "(matrix.target == 'aarch64-unknown-linux-gnu' && "
            "startsWith(needs.metadata.outputs.runner_arm_8, 'depot-'))) "
            "&& 'false' || 'true' }}"
        )
        effective_release_runner_16 = (
            "${{ startsWith(needs.metadata.outputs.runner_16, 'depot-') "
            "&& 'false' || 'true' }}"
        )
        expected = {
            ("ci-linux-host-slice.yml", "linux_host"): policy,
            ("ci-linux-runtime-slice.yml", "linux_runtime"): policy,
            ("ci-quality-slice.yml", "rust_clippy"): policy,
            ("ci-rust-tests-slice.yml", "rust_tests"): policy,
            ("ci-rust-tests-slice.yml", "safetensors_runtime_smoke"): policy,
            ("ci-windows-host-slice.yml", "windows_host"): policy,
            ("ci-windows-runtime-slice.yml", "windows_runtime"): policy,
            ("cache-warm-sccache.yml", "warm"): "false",
            ("depot-canary.yml", "runtime_seed"): "false",
            ("hf-download-smoke.yml", "hf_download_smoke"): "true",
            ("native-sdk-artifact.yml", "linux_native_sdk_artifact"): policy,
            ("native-sdk-artifact.yml", "macos_native_sdk_artifact"): policy,
            ("node-sdk-addon-artifact.yml", "linux_addon"): "true",
            ("node-sdk-addon-artifact.yml", "macos_addon"): "true",
            ("node-sdk-addon-artifact.yml", "windows_addon"): "true",
            ("release.yml", "build"): "true",
            ("release.yml", "build_native_runtime"): effective_release,
            ("release.yml", "build_native_runtime_linux_aarch64_cuda"): "true",
            ("release.yml", "build_native_runtime_linux_x86_64_cuda"): "true",
            ("release.yml", "build_native_runtime_linux_x86_64_rocm"): effective_release_runner_16,
            ("release.yml", "build_native_runtime_linux_x86_64_vulkan"): effective_release_runner_16,
            ("static-abi-artifact.yml", "static_abi_artifact"): policy,
            ("swift-sdk-artifact.yml", "swift_sdk_target"): policy,
            ("swift-sdk-artifact.yml", "swift_sdk_artifact"): policy,
        }
        actual: dict[tuple[str, str], str] = {}
        for workflow_path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8")) or {}
            for job_name, job in (workflow.get("jobs") or {}).items():
                for step in job.get("steps") or []:
                    if step.get("uses") != "./.github/actions/configure-sccache-gha":
                        continue
                    with_values = step.get("with") or {}
                    self.assertIn(
                        "allow_native_github_cache",
                        with_values,
                        f"{workflow_path.name}:{job_name} must set cache policy",
                    )
                    actual[(workflow_path.name, job_name)] = str(
                        with_values["allow_native_github_cache"],
                    )
        self.assertEqual(set(expected), set(actual))
        self.assertEqual(expected, actual)

    def run_capture(
        self,
        payload: dict[str, object],
        *,
        artifact_name: str = "sccache-test-1",
        sccache_error: str = "",
        cache_expectation: str = "opportunistic",
        minimum_hit_rate: str = "0",
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        fake_sccache = root / "sccache"
        fake_sccache.write_text(
            "#!/bin/sh\n"
            "if [ \"$#\" -eq 1 ] && [ \"$1\" = \"--show-stats\" ]; then\n"
            "  printf '%s\\n' "
            "'Cache location: https://human-stats-secret.example/private'\n"
            "elif [ \"$#\" -eq 3 ] && [ \"$1\" = \"--show-stats\" ] "
            "&& [ \"$2\" = \"--stats-format\" ] && [ \"$3\" = \"json\" ]; then\n"
            "  if [ -n \"$FAKE_SCCACHE_ERROR\" ]; then\n"
            "    printf '%s\\n' \"$FAKE_SCCACHE_ERROR\" >&2\n"
            "    exit 23\n"
            "  fi\n"
            "  printf '%s\\n' \"$FAKE_SCCACHE_JSON\"\n"
            "else\n"
            "  exit 2\n"
            "fi\n",
            encoding="utf-8",
        )
        fake_sccache.chmod(
            fake_sccache.stat().st_mode | stat.S_IXUSR,
        )
        stats_file = root / "evidence" / "sccache-stats.json"
        github_output = root / "github-output"
        result = subprocess.run(
            [
                sys.executable,
                str(CAPTURE),
                "--artifact-name",
                artifact_name,
                "--output",
                str(stats_file),
                "--github-output",
                str(github_output),
                "--cache-expectation",
                cache_expectation,
                "--minimum-hit-rate",
                minimum_hit_rate,
            ],
            env={
                **os.environ,
                "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                "FAKE_SCCACHE_JSON": json.dumps(payload),
                "FAKE_SCCACHE_ERROR": sccache_error,
            },
            check=False,
            capture_output=True,
            text=True,
        )
        return result, stats_file, github_output

    def test_capture_writes_only_sanitized_machine_readable_counters(self) -> None:
        payload = valid_payload()
        result, stats_file, github_output = self.run_capture(payload)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Human-readable sccache statistics", result.stdout)
        self.assertNotIn("human-stats-secret", result.stdout)
        self.assertEqual(
            json.loads(stats_file.read_text()),
            {
                "schema": "mesh-llm.sccache-stats",
                "schema_version": 2,
                "assessment": {
                    "expectation": "opportunistic",
                    "classification": "opportunistic",
                    "minimum_hit_rate": 0.0,
                    "hit_rate": 0.6,
                    "cache_requests": 10,
                    "passed": True,
                },
                "stats": {
                    "compile_requests": 12,
                    "requests_executed": 10,
                    "compilations": 4,
                    "cache_writes": 3,
                    "cache_read_errors": 0,
                    "cache_write_errors": 0,
                    "cache_hits": {"counts": {"total": 6}},
                    "cache_misses": {"counts": {"total": 4}},
                    "cache_errors": {"counts": {"total": 0}},
                },
            },
        )
        outputs = github_output.read_text(encoding="utf-8")
        self.assertIn("compile_requests=12", outputs)
        self.assertIn("requests_executed=10", outputs)
        self.assertIn("cache_hits=6", outputs)
        self.assertIn("cache_misses=4", outputs)
        self.assertIn("hit_rate=0.6", outputs)
        self.assertIn("cache_classification=opportunistic", outputs)
        self.assertIn("cache_passed=true", outputs)

    def test_raw_secrets_urls_and_paths_cannot_reach_logs_or_evidence(self) -> None:
        payload = valid_payload()
        stats = payload["stats"]
        self.assertIsInstance(stats, dict)
        cache_hits = stats["cache_hits"]
        self.assertIsInstance(cache_hits, dict)
        cache_hits["counts"] = {
            "/Users/private/cache/path?token=count-key-secret": 6,
        }
        stats["not_cached"] = {
            "/home/runner/private-source": 1,
            "https://stats-secret.example/cache": 2,
        }
        payload.update(
            {
                "cache_location": (
                    "WebDAV: https://cache-user:location-secret@cache.example"
                ),
                "basedirs": ["/home/runner/work/private-repository"],
                "version": "raw-version-secret",
                "url": "https://payload-secret.example/cache",
                "absolute_path": "/Users/private/sccache",
            },
        )

        result, stats_file, github_output = self.run_capture(payload)

        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(stats_file.read_text(encoding="utf-8"))
        self.assertEqual(evidence["stats"]["cache_hits"], {"counts": {"total": 6}})
        exposed_surface = "\n".join(
            (
                result.stdout,
                result.stderr,
                stats_file.read_text(encoding="utf-8"),
                github_output.read_text(encoding="utf-8"),
            ),
        )
        for forbidden in (
            "human-stats-secret",
            "count-key-secret",
            "private-source",
            "stats-secret",
            "cache_location",
            "basedirs",
            "location-secret",
            "raw-version-secret",
            "payload-secret",
            "/Users/private",
            "/home/runner/work/private-repository",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, exposed_surface)

    def test_sccache_error_output_cannot_leak_into_the_job_log(self) -> None:
        secret = "https://cache-user:stderr-secret@cache.example/private"
        result, stats_file, _ = self.run_capture(
            valid_payload(),
            sccache_error=secret,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(stats_file.exists())
        self.assertNotIn(secret, result.stdout)
        self.assertNotIn(secret, result.stderr)
        self.assertIn("exit code 23", result.stderr)

    def test_zero_compile_requests_warns_but_remains_valid_evidence(self) -> None:
        result, stats_file, _ = self.run_capture(
            valid_payload(compile_requests=0),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(stats_file.is_file())
        self.assertIn("::warning title=sccache reported zero compile requests", result.stdout)

    def test_zero_warm_floor_allows_an_observation_without_cache_requests(self) -> None:
        payload = valid_payload(compile_requests=0)
        stats = payload["stats"]
        self.assertIsInstance(stats, dict)
        stats["requests_executed"] = 0
        stats["compilations"] = 0
        stats["cache_writes"] = 0
        for name in ("cache_hits", "cache_misses"):
            counts = stats[name]
            self.assertIsInstance(counts, dict)
            counts["counts"] = {}

        result, stats_file, github_output = self.run_capture(
            payload,
            cache_expectation="warm",
            minimum_hit_rate="0",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        assessment = json.loads(stats_file.read_text())["assessment"]
        self.assertEqual(assessment["classification"], "warm-pass")
        self.assertIsNone(assessment["hit_rate"])
        self.assertIn("cache_passed=true", github_output.read_text())
        self.assertIn(
            "::warning title=sccache reported zero compile requests",
            result.stdout,
        )

    def test_positive_warm_floor_rejects_missing_cache_requests(self) -> None:
        payload = valid_payload(compile_requests=0)
        stats = payload["stats"]
        self.assertIsInstance(stats, dict)
        for name in ("cache_hits", "cache_misses"):
            counts = stats[name]
            self.assertIsInstance(counts, dict)
            counts["counts"] = {}

        result, stats_file, github_output = self.run_capture(
            payload,
            cache_expectation="warm",
            minimum_hit_rate="0.01",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        assessment = json.loads(stats_file.read_text())["assessment"]
        self.assertEqual(assessment["classification"], "warm-failure")
        self.assertIn("cache_passed=false", github_output.read_text())

    def test_warm_and_cold_observations_are_classified_separately(self) -> None:
        warm_result, warm_file, warm_output = self.run_capture(
            valid_payload(), cache_expectation="warm", minimum_hit_rate="0.80",
        )
        cold_result, cold_file, cold_output = self.run_capture(
            valid_payload(), cache_expectation="cold", minimum_hit_rate="0.80",
        )
        self.assertEqual(warm_result.returncode, 0, warm_result.stderr)
        self.assertEqual(cold_result.returncode, 0, cold_result.stderr)
        self.assertEqual(
            json.loads(warm_file.read_text())["assessment"]["classification"],
            "warm-failure",
        )
        self.assertIn("cache_passed=false", warm_output.read_text())
        self.assertEqual(
            json.loads(cold_file.read_text())["assessment"]["classification"],
            "cold",
        )
        self.assertIn("cache_passed=true", cold_output.read_text())

    def test_missing_or_invalid_counter_rejects_evidence(self) -> None:
        payload = valid_payload()
        stats = payload["stats"]
        self.assertIsInstance(stats, dict)
        del stats["cache_misses"]

        result, stats_file, _ = self.run_capture(payload)

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(stats_file.exists())
        self.assertIn("stats.cache_misses must be an object", result.stderr)

    def test_artifact_name_cannot_escape_the_evidence_namespace(self) -> None:
        result, stats_file, _ = self.run_capture(
            valid_payload(),
            artifact_name="../sccache-test",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(stats_file.exists())
        self.assertIn("artifact name must contain only", result.stderr)

    def test_composite_action_uploads_fourteen_day_json_evidence(self) -> None:
        action = (ACTION_DIR / "action.yml").read_text(encoding="utf-8")
        capture = CAPTURE.read_text(encoding="utf-8")

        self.assertIn("artifact_name:", action)
        self.assertIn("actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f", action)
        self.assertIn("retention-days: 14", action)
        self.assertIn("if-no-files-found: error", action)
        self.assertIn('"--show-stats", "--stats-format", "json"', capture)
        self.assertIn("REQUIRED_COUNTERS", capture)
        self.assertIn("REQUIRED_COUNT_MAPS", capture)

    def test_configure_action_resets_each_successful_server_route(self) -> None:
        configure = CONFIGURE_ACTION.read_text(encoding="utf-8")

        self.assertIn("['--zero-stats']", configure)
        self.assertEqual(configure.count("await resetStatistics("), 7)

    def test_remote_multilevel_writes_finish_before_ephemeral_job_exit(self) -> None:
        configure = CONFIGURE_ACTION.read_text(encoding="utf-8")

        self.assertEqual(
            configure.count(
                "core.exportVariable("
                "'SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY', 'all'"
                ")",
            ),
            2,
        )
        self.assertNotIn(
            "'SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY', 'ignore'",
            configure,
        )

    def test_pull_request_sccache_is_disk_only(self) -> None:
        configure = CONFIGURE_ACTION.read_text(encoding="utf-8")

        self.assertIn("DISPATCH_ORIGINAL_EVENT_NAME", configure)
        self.assertIn("effectiveEventName === 'pull_request'", configure)
        self.assertIn("effectiveEventName === 'pull_request_target'", configure)
        self.assertIn(
            "core.exportVariable('SCCACHE_GHA_RW_MODE', ghaRemoteMode)",
            configure,
        )
        self.assertIn("if (ghaRemoteMode === 'READ_ONLY')", configure)
        self.assertIn(
            "Pull-request trust context detected; using baked sccache "
            "with job-local disk only.",
            configure,
        )
        for path in (*WORKFLOWS.values(), HF_WORKFLOW, NATIVE_SDK_WORKFLOW):
            workflow = path.read_text(encoding="utf-8")
            if "uses: ./.github/actions/configure-sccache-gha" in workflow:
                self.assertNotIn("SCCACHE_WEBDAV_RW_MODE", workflow)
        self.assertNotIn("SCCACHE_WEBDAV_RW_MODE", configure)

    def test_pull_request_direct_sccache_users_are_reconfigured(self) -> None:
        workflows = (*WORKFLOWS.values(), HF_WORKFLOW, NATIVE_SDK_WORKFLOW)

        for path in workflows:
            lines = path.read_text(encoding="utf-8").splitlines()
            direct_users = [
                index
                for index, line in enumerate(lines)
                if "uses: mozilla-actions/sccache-action@" in line
            ]
            for index in direct_users:
                with self.subTest(workflow=path.name, line=index + 1):
                    next_step = next(
                        line.strip()
                        for line in lines[index + 1 :]
                        if line.strip()
                    )
                    self.assertEqual(
                        next_step,
                        "- uses: ./.github/actions/configure-sccache-gha",
                    )

    def test_swift_uses_trusted_main_seeded_dependency_cache(self) -> None:
        swift = WORKFLOWS["swift-sdk"].read_text(encoding="utf-8")

        self.assertIn(
            "uses: Swatinem/rust-cache@"
            "6323deb102c322ba6fcbdcafc7e3dddab59af2b6",
            swift,
        )
        self.assertIn("shared-key: swift-sdk", swift)
        self.assertIn("key: ${{ steps.native_toolchain.outputs.epoch }}", swift)
        self.assertIn('add-job-id-key: "false"', swift)
        self.assertIn(
            "save-if: ${{ github.event_name == 'push' "
            "&& github.ref == 'refs/heads/main' }}",
            swift,
        )

    def test_rust_sdk_cache_key_covers_smoke_compatibility_boundaries(self) -> None:
        workflow = yaml.safe_load(SDK_SMOKE_WORKFLOW.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["sdk_smoke"]["steps"]
        cache_steps = [
            step
            for step in steps
            if str(step.get("uses", "")).startswith("Swatinem/rust-cache@")
        ]
        self.assertEqual(len(cache_steps), 1)
        cache = cache_steps[0]
        self.assertEqual(cache["if"], "${{ inputs.sdk_kind == 'rust' }}")
        values = cache["with"]

        prefix = values["prefix-key"]
        for boundary in (
            "sdk-rust-cargo-v1",
            "env.SDK_RUST_TARGET",
            "env.SDK_RUST_IMAGE_DIGEST",
            "env.SDK_RUST_TOOLCHAIN_EPOCH",
            "env.SDK_RUST_PROFILE_LINKER",
            "hashFiles(",
            "'Cargo.lock'",
            "'.github/cache-version.txt'",
            "'.cargo/config.toml'",
            "'**/Cargo.toml'",
            "'scripts/ci-rust-sdk-smoke.sh'",
            "'scripts/ci-sdk-fixture.sh'",
            "'scripts/ci-prepare-native-runtime.sh'",
            "'scripts/package-sdk-console-assets.sh'",
            "'scripts/check-sdk-contract.sh'",
            "'scripts/verify-sdk-console-assets.sh'",
            "'.github/workflows/sdk-smoke.yml'",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, prefix)

        self.assertEqual(values["shared-key"], "ci-sdk-smoke-rust")
        self.assertEqual(values["cache-bin"], "false")
        self.assertIn(
            "github.ref == 'refs/heads/main'",
            values["save-if"],
        )
        self.assertNotIn("actions/cache", SDK_SMOKE_WORKFLOW.read_text(encoding="utf-8"))

    def test_linux_seed_producer_and_consumers_share_compatible_key(self) -> None:
        warmer_workflow = yaml.safe_load(SEED_WARMER.read_text(encoding="utf-8"))
        warmer_steps = warmer_workflow["jobs"]["warm"]["steps"]
        seed_steps = [step for step in warmer_steps if step.get("id") == "seed"]
        self.assertEqual(len(seed_steps), 1)
        seed_assignments = [
            match.group(1)
            for line in seed_steps[0]["run"].splitlines()
            if (match := re.fullmatch(r'\s*key="([^"\n]+)"\s*', line))
        ]
        self.assertEqual(len(seed_assignments), 1)
        expected_key = seed_assignments[0]
        self.assertIsNotNone(SEED_KEY_PATTERN.fullmatch(expected_key))

        cache_steps = [
            step for step in warmer_steps
            if str(step.get("uses", "")).startswith("actions/cache/")
        ]
        self.assertEqual(len(cache_steps), 2)
        for step in cache_steps:
            self.assertEqual(step["with"]["key"], "${{ steps.seed.outputs.key }}")

        consumers = (
            WORKFLOWS["quality"],
            WORKFLOWS["rust-tests"],
            WORKFLOWS["host"],
            WORKFLOWS["runtime"],
        )
        expected_restore_counts = {
            "ci-rust-tests-slice.yml": 2,
        }
        for path in consumers:
            with self.subTest(workflow=path.name):
                workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
                keys = [
                    step.get("with", {}).get("cache_key")
                    for job in workflow["jobs"].values()
                    for step in job.get("steps", [])
                    if step.get("uses") == "./.github/actions/restore-sccache-seed"
                ]
                self.assertEqual(
                    len(keys),
                    expected_restore_counts.get(path.name, 1),
                )
                for key in keys:
                    self.assertIsNotNone(SEED_KEY_PATTERN.fullmatch(key))
                    self.assertEqual(key, expected_key)
        warmer = SEED_WARMER.read_text(encoding="utf-8")
        self.assertIn("run: just ci-sccache-seed-build", warmer)
        self.assertNotIn(
            "run: cargo clippy --locked -p mesh-llm --all-targets -- -D warnings",
            warmer,
        )
        seed_recipe = (ROOT / "just" / "ci.just").read_text(encoding="utf-8")
        recipe_match = re.search(
            r"(?m)^ci-sccache-seed-build:\n(?P<body>(?:    .*\n)+)",
            seed_recipe,
        )
        self.assertIsNotNone(recipe_match)
        self.assertEqual(
            [line.strip() for line in recipe_match.group("body").splitlines()],
            [
                "cargo clippy --locked -p mesh-llm --all-targets -- -D warnings",
                "cargo test --locked -p mesh-llm-cli --no-run",
            ],
        )
        restore = (
            ROOT / ".github" / "actions" / "restore-sccache-seed" / "action.yml"
        ).read_text(encoding="utf-8")
        self.assertIn('echo "SCCACHE_CACHE_SIZE=2G" >> "$GITHUB_ENV"', restore)

    def test_runtime_seed_restore_is_deliberately_disabled(self) -> None:
        runtime = WORKFLOWS["runtime"].read_text(encoding="utf-8")
        self.assertIn('allow_trusted_seed: "false"', runtime)
        self.assertIn("uses: ./.github/actions/restore-sccache-seed", runtime)

    def test_instrumented_workflows_use_unique_evidence_artifacts(self) -> None:
        for workflow_name in INSTRUMENTED:
            path = WORKFLOWS[workflow_name]
            workflow = path.read_text(encoding="utf-8")
            with self.subTest(workflow=workflow_name):
                self.assertNotIn("Show sccache stats", workflow)
                self.assertIn(
                    "uses: ./.github/actions/capture-sccache-stats",
                    workflow,
                )
                names = [
                    line.split("artifact_name:", 1)[1].strip()
                    for line in workflow.splitlines()
                    if "artifact_name:" in line
                    and "sccache-" in line
                ]
                self.assertEqual(
                    workflow.count("uses: ./.github/actions/capture-sccache-stats"),
                    len(names),
                )
                self.assertEqual(len(names), len(set(names)))
                for artifact_name in names:
                    self.assertIn("${{ github.run_attempt }}", artifact_name)

    def test_safetensors_smoke_captures_cache_before_runtime_loop(self) -> None:
        workflow = yaml.safe_load(WORKFLOWS["rust-tests"].read_text(encoding="utf-8"))
        steps = workflow["jobs"]["safetensors_runtime_smoke"]["steps"]
        step_names = [step.get("name") for step in steps]

        build_index = step_names.index("Build and locate the SafeTensors smoke test")
        capture_index = step_names.index("Capture SafeTensors smoke cache evidence")
        exercise_index = step_names.index("Exercise every current direct-load quantization")
        self.assertEqual(capture_index, build_index + 1)
        self.assertLess(capture_index, exercise_index)


class SccacheStatsSummaryTests(unittest.TestCase):
    def write_evidence(
        self,
        path: Path,
        *,
        hits: int,
        misses: int,
        advanced_hits: int = 0,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "stats": {
                        "cache_hits": {
                            "counts": {"Rust": hits},
                            "adv_counts": {"Rust": advanced_hits},
                        },
                        "cache_misses": {
                            "counts": {"Rust": misses},
                            "adv_counts": {},
                        },
                    },
                },
            ),
            encoding="utf-8",
        )

    def run_summary(
        self,
        evidence: Path,
        *,
        minimum: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(SUMMARY),
            "--format",
            "json",
        ]
        if minimum is not None:
            command.extend(["--minimum-hit-rate", minimum])
        command.append(str(evidence))
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_offline_summary_aggregates_counts_without_advanced_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            self.write_evidence(
                evidence / "job-a" / "sccache-stats.json",
                hits=60,
                misses=10,
                advanced_hits=60,
            )
            self.write_evidence(
                evidence / "job-b" / "sccache-stats-warm.json",
                hits=20,
                misses=10,
                advanced_hits=20,
            )

            result = self.run_summary(evidence, minimum="0.80")

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["file_count"], 2)
        self.assertEqual(summary["cache_hits"], 80)
        self.assertEqual(summary["cache_misses"], 20)
        self.assertEqual(summary["hit_rate"], 0.8)
        self.assertTrue(summary["passed"])

    def test_offline_summary_fails_a_missed_hit_rate_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "sccache-stats.json"
            self.write_evidence(evidence, hits=79, misses=21)

            result = self.run_summary(evidence, minimum="0.80")

        self.assertEqual(result.returncode, 1)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["hit_rate"], 0.79)
        self.assertFalse(summary["passed"])

    def test_offline_summary_rejects_invalid_count_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "sccache-stats.json"
            self.write_evidence(evidence, hits=-1, misses=1)

            result = self.run_summary(evidence)

        self.assertEqual(result.returncode, 1)
        self.assertIn("negative counter", result.stderr)


if __name__ == "__main__":
    unittest.main()
