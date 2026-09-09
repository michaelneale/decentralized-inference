import unittest

import yaml
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "depot-canary.yml"


class DepotCanaryWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def _job_block(self, job_name: str) -> str:
        jobs = self.workflow.split("\njobs:\n", 1)[1]
        match = re.search(
            rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
            jobs,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise AssertionError(f"missing job {job_name}")
        return match.group("body")

    @staticmethod
    def _step_script(job: str, step_name: str) -> str:
        match = re.search(
            rf"^      - name: {re.escape(step_name)}\n(?P<body>.*?)(?=^      - name:|\Z)",
            job,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise AssertionError(f"missing step {step_name}")
        run = re.search(
            r"^        run: \|\n(?P<script>.*)",
            match.group("body"),
            flags=re.MULTILINE | re.DOTALL,
        )
        if run is None:
            raise AssertionError(f"step {step_name} has no run script")
        return dedent(run.group("script"))

    def _run_attestation(
        self,
        script: str,
        cache_url: str,
        results_url: str,
        *,
        runtime_token: str = "non-secret-runtime-token",
        path: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            "PATH": path or os.environ.get("PATH", "/usr/bin:/bin"),
            "ACTIONS_CACHE_URL": cache_url,
            "ACTIONS_RESULTS_URL": results_url,
            "ACTIONS_RUNTIME_TOKEN": runtime_token,
        }
        if runtime_token == "":
            environment.pop("ACTIONS_RUNTIME_TOKEN")
        return subprocess.run(
            ["/bin/bash", "-c", "set -euo pipefail\n" + script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_canary_has_no_code_or_credential_access(self) -> None:
        self.assertIn("permissions: {}", self.workflow)
        for job in ("runner", "seed_authority_marker", "verify_pr_write"):
            with self.subTest(job=job):
                self.assertNotIn("actions/checkout", self._job_block(job))
        self.assertNotIn("secrets.", self.workflow)
        self.assertNotIn("pull_request", self.workflow)
        self.assertNotIn("push:", self.workflow)

    def test_runtime_canary_checkout_is_hosted_and_read_only(self) -> None:
        jobs = yaml.safe_load(self.workflow)["jobs"]
        for name in ("runtime_seed", "runtime_seed_summary"):
            with self.subTest(job=name):
                job = jobs[name]
                self.assertEqual(job["runs-on"], "ubuntu-24.04")
                expected = {"contents": "read", "actions": "read"}
                if name == "runtime_seed":
                    expected["packages"] = "read"
                self.assertEqual(job["permissions"], expected)
                checkouts = [step for step in job["steps"]
                             if step.get("uses", "").startswith("actions/checkout@")]
                self.assertEqual(len(checkouts), 1)
                self.assertEqual(checkouts[0]["with"], {
                    "ref": "${{ github.sha }}", "persist-credentials": False,
                })

    def test_canary_covers_measured_depot_sizes(self) -> None:
        for runner in (
            "depot-ubuntu-24.04",
            "depot-ubuntu-24.04-4",
            "depot-ubuntu-24.04-8",
            "depot-ubuntu-24.04-16",
            "depot-ubuntu-24.04-arm",
            "depot-ubuntu-24.04-arm-8",
        ):
            with self.subTest(runner=runner):
                self.assertIn(f"- {runner}", self.workflow)
        self.assertIn("expected_arch=aarch64", self.workflow)
        self.assertIn('actual_arch="$(uname -m)"', self.workflow)

    def test_canary_sentinel_modes_are_single_runner_and_cache_bounded(self) -> None:
        self.assertIn("type: choice", self.workflow)
        self.assertIn("default: audit", self.workflow)
        for mode in ("audit", "seed", "verify-pr-write"):
            self.assertIn(f"          - {mode}", self.workflow)

        audit = self._job_block("runner")
        seed = self._job_block("seed_authority_marker")
        verify = self._job_block("verify_pr_write")

        self.assertIn("github.event.inputs.mode == 'audit'", audit)
        self.assertIn("github.event.inputs.mode == 'seed'", seed)
        self.assertIn("github.event.inputs.mode == 'verify-pr-write'", verify)
        for block in (seed, verify):
            with self.subTest(block=block.splitlines()[0]):
                self.assertIn("permissions: {}", block)
                self.assertIn("runs-on: depot-ubuntu-24.04", block)
                self.assertNotIn("actions/checkout", block)
                self.assertNotIn("secrets.", block)

        self.assertIn(
            "printf 'mesh-llm-depot-authority-marker-v1\\nsentinel_id=%s\\n'",
            seed,
        )
        self.assertIn(
            "uses: actions/cache/save@caa296126883cff596d87d8935842f9db880ef25",
            seed,
        )
        self.assertIn(
            "uses: actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25",
            seed,
        )
        self.assertNotIn("lookup-only: true", seed)
        self.assertIn("fail-on-cache-miss: true", seed)
        self.assertIn("Clear local marker before trusted restore", seed)
        self.assertIn("Validate trusted marker content", seed)
        self.assertIn("cmp -s", seed)
        self.assertIn(
            "key: ${{ steps.validate.outputs.seed_key }}",
            seed,
        )
        self.assertEqual(seed.count("path: .depot-authority-sentinel"), 2)
        self.assertIn(
            "key: ${{ steps.validate.outputs.poison_key }}",
            verify,
        )
        self.assertIn("path: .depot-authority-sentinel", verify)
        self.assertNotIn("lookup-only: true", verify)
        self.assertIn("fail-on-cache-miss: false", verify)
        self.assertIn("Clear local poison marker before trusted restore", verify)
        self.assertIn("Validate PR poison marker content on hit", verify)
        self.assertIn("cmp -s", verify)
        self.assertIn("PR poison marker was visible to trusted main", verify)

        for block in (seed, verify):
            self.assertIn("Attest provider-injected cache backend", block)
            self.assertIn("ACTIONS_CACHE_URL", block)
            self.assertIn("ACTIONS_RESULTS_URL", block)
            self.assertNotIn("ACTIONS_RUNTIME_TOKEN", block)
            self.assertIn("cache backend attestation failed", block)
            self.assertNotIn("${endpoint,,}", block)

    def test_sentinel_inputs_are_strict_and_derive_fixed_keys(self) -> None:
        seed = self.workflow.split("  seed_authority_marker:", 1)[1].split(
            "  verify_pr_write:", 1
        )[0]
        verify = self.workflow.split("  verify_pr_write:", 1)[1]

        def validation_script(block: str) -> str:
            run_marker = "        run: |\n"
            script = block.split(run_marker, 1)[1].split(
                "\n      - name:", 1
            )[0]
            return "set -euo pipefail\n" + dedent(script)

        def run_validation(
            script: str,
            sentinel_id: str,
            pr_number: str,
            configured_id: str = "0123456789abcdef0123456789abcdef",
            configured_ref: str = "refs/pull/42/merge",
        ) -> tuple[subprocess.CompletedProcess[str], str]:
            with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as output:
                environment = {
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "SENTINEL_ID": sentinel_id,
                    "PR_NUMBER": pr_number,
                    "CONFIGURED_SENTINEL_ID": configured_id,
                    "CONFIGURED_SENTINEL_REF": configured_ref,
                    "GITHUB_OUTPUT": output.name,
                }
                result = subprocess.run(
                    ["bash", "-c", script],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                output.seek(0)
                return result, output.read()

        valid_id = "0123456789abcdef0123456789abcdef"
        valid_pr = "42"
        seed_result, seed_output = run_validation(
            validation_script(seed), valid_id, valid_pr
        )
        self.assertEqual(seed_result.returncode, 0, seed_result.stderr)
        self.assertEqual(
            seed_output,
            "sentinel_id="
            f"{valid_id}\n"
            "seed_key=mesh-llm-depot-authority-seed-v1-"
            f"{valid_id}\n"
            "poison_key=mesh-llm-depot-authority-pr-v1-"
            f"{valid_id}-pr-{valid_pr}\n",
        )

        verify_result, verify_output = run_validation(
            validation_script(verify), valid_id, valid_pr
        )
        self.assertEqual(verify_result.returncode, 0, verify_result.stderr)
        self.assertEqual(
            verify_output,
            "sentinel_id="
            f"{valid_id}\n"
            "pr_number=42\n"
            "poison_key=mesh-llm-depot-authority-pr-v1-"
            f"{valid_id}-pr-{valid_pr}\n",
        )

        for sentinel_id in (
            "",
            "ABCDEF0123456789abcdef0123456789",
            "0" * 31,
            "0" * 33,
        ):
            with self.subTest(sentinel_id=sentinel_id):
                result, _ = run_validation(
                    validation_script(seed), sentinel_id, valid_pr
                )
                self.assertNotEqual(result.returncode, 0)

        for pr_number in ("", "0", "01", "+1", " 1", "1 ", "1" * 10):
            with self.subTest(pr_number=pr_number):
                result, _ = run_validation(
                    validation_script(seed), valid_id, pr_number
                )
                self.assertNotEqual(result.returncode, 0)

        for script_name, script in (("seed", seed), ("verify", verify)):
            with self.subTest(script=script_name, mismatch="id"):
                result, _ = run_validation(
                    validation_script(script),
                    valid_id,
                    valid_pr,
                    configured_id="fedcba9876543210fedcba9876543210",
                )
                self.assertNotEqual(result.returncode, 0)
            with self.subTest(script=script_name, mismatch="ref"):
                result, _ = run_validation(
                    validation_script(script),
                    valid_id,
                    valid_pr,
                    configured_ref="refs/pull/43/merge",
                )
                self.assertNotEqual(result.returncode, 0)
            with self.subTest(script=script_name, mismatch="number"):
                result, _ = run_validation(
                    validation_script(script),
                    valid_id,
                    "43",
                    configured_ref="refs/pull/42/merge",
                )
                self.assertNotEqual(result.returncode, 0)

    def test_all_cache_phases_attest_non_github_remote_backend(self) -> None:
        valid_cache = "http://cache.example.invalid:1234/cache"
        valid_results = "http://results.example.invalid:5678/results"
        scripts = {
            "seed": self._step_script(
                self._job_block("seed_authority_marker"),
                "Attest provider-injected cache backend",
            ),
            "verify-pr-write": self._step_script(
                self._job_block("verify_pr_write"),
                "Attest provider-injected cache backend",
            ),
        }
        for phase, script in scripts.items():
            with self.subTest(phase=phase, backend="valid"):
                result = self._run_attestation(script, valid_cache, valid_results)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, "")

            invalid_cases = (
                ("ACTIONS_CACHE_URL", "https://cache.example.invalid:1234/cache", "scheme"),
                ("ACTIONS_CACHE_URL", "http://actions.githubusercontent.com:1234/cache", "github"),
                ("ACTIONS_CACHE_URL", "http://localhost:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://127.0.0.1:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://127.0.0.0:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://127.1.2.3:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://127.255.255.255:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[::1]:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[::ffff:127.0.0.1]:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[::ffff:127.255.255.255]:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[::ffff:7f00:1]:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[0:0:0:0:0:ffff:127.0.0.1]:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[0:0:0:0:0:ffff:7f00:1]:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[0:0:0:0:0:0:0:1]:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[0::1]:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[0:0:0:0::1]:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[0::ffff:127.0.0.1]:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[0:0:0:0::ffff:127.0.0.1]:1234/cache", "loopback"),
                ("ACTIONS_CACHE_URL", "http://[not-an-ip]:1234/cache", "parser"),
                ("ACTIONS_CACHE_URL", "http://user@cache.example.invalid:1234/cache", "userinfo"),
                ("ACTIONS_CACHE_URL", "http://cache.example.invalid/cache", "port"),
                ("ACTIONS_CACHE_URL", "http://cache.example.invalid:0/cache", "port"),
                ("ACTIONS_CACHE_URL", "http://cache.example.invalid:65536/cache", "port"),
                ("ACTIONS_CACHE_URL", "http://cache.example.invalid:1234", "path"),
                ("ACTIONS_CACHE_URL", "http://cache.example.invalid:1234/cache with-space", "whitespace"),
                ("ACTIONS_RESULTS_URL", "https://results.example.invalid:5678/results", "scheme"),
                ("ACTIONS_RESULTS_URL", "http://results.example.invalid/results", "port"),
                ("ACTIONS_RESULTS_URL", "http://results.example.invalid:5678", "path"),
            )
            for endpoint_name, endpoint, reason in invalid_cases:
                with self.subTest(phase=phase, endpoint=endpoint_name, reason=reason):
                    cache_url = endpoint if endpoint_name == "ACTIONS_CACHE_URL" else valid_cache
                    results_url = endpoint if endpoint_name == "ACTIONS_RESULTS_URL" else valid_results
                    result = self._run_attestation(script, cache_url, results_url)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        f"cache backend attestation failed (variable={endpoint_name} reason={reason})",
                        result.stderr,
                    )
                    self.assertNotIn(endpoint, result.stderr)
                    self.assertNotIn("cache.example.invalid", result.stderr)
                    self.assertNotIn("results.example.invalid", result.stderr)
                    self.assertNotIn("actions.githubusercontent.com", result.stderr)
                    self.assertNotIn("non-secret-runtime-token", result.stderr)

            with self.subTest(phase=phase, token="missing"):
                result = self._run_attestation(
                    script,
                    valid_cache,
                    valid_results,
                    runtime_token="",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, "")

    def test_canary_fails_closed_on_cache_and_registry_injection(self) -> None:
        self.assertIn("forbidden_names=", self.workflow)
        for name in (
            "DEPOT_CACHE_TOKEN",
            "DEPOT_CACHE_URL",
            "DEPOT_CACHE_API_URL",
            "DEPOT_TOKEN",
            "DEPOT_REGISTRY_HOST",
            "DEPOT_REGISTRY_URL",
            "DEPOT_REGISTRY_PULL_TOKEN",
            "DEPOT_REGISTRY_TOKEN",
            "SCCACHE_WEBDAV_ENDPOINT",
            "SCCACHE_WEBDAV_TOKEN",
            "SCCACHE_WEBDAV_USERNAME",
            "SCCACHE_WEBDAV_PASSWORD",
        ):
            with self.subTest(name=name):
                self.assertIn(name, self.workflow)
        self.assertIn('for endpoint_name in ACTIONS_CACHE_URL ACTIONS_RESULTS_URL ACTIONS_RUNTIME_URL; do', self.workflow)
        self.assertIn('endpoint_host() {', self.workflow)
        self.assertIn('is_loopback_endpoint() {', self.workflow)
        self.assertIn('endpoint_scheme() {', self.workflow)
        self.assertIn('endpoint_authority_class() {', self.workflow)
        self.assertIn('endpoint_numeric_port() {', self.workflow)
        self.assertIn('endpoint_explicit_path() {', self.workflow)
        self.assertIn('report_endpoint_rejection() {', self.workflow)
        self.assertIn(r"\[::1\]", self.workflow)
        self.assertIn("numeric", self.workflow.lower())
        self.assertIn(
            'GitHub Actions endpoint rejected (variable=%s scheme=%s authority=%s numeric_port=%s explicit_path=%s)',
            self.workflow,
        )
        self.assertNotIn('GitHub Actions cache was transparently redirected to Depot ($endpoint_name)', self.workflow)
        self.assertNotIn('GitHub Actions endpoint contains URL userinfo ($endpoint_name)', self.workflow)
        self.assertNotIn('GitHub Actions endpoint is not GitHub-owned or loopback ($endpoint_name)', self.workflow)
        self.assertIn("ACTIONS_RESULTS_URL", self.workflow)
        self.assertIn("actions\\.githubusercontent\\.com", self.workflow)
        self.assertNotIn(",,}", self.workflow)
        self.assertIn(
            "printf '%s' \"$endpoint\" | tr '[:upper:]' '[:lower:]'",
            self.workflow,
        )
        self.assertIn('docker_auth_config="${DOCKER_AUTH_CONFIG:-}"', self.workflow)
        self.assertIn("Docker registry authentication is configured", self.workflow)
        self.assertIn(
            "Depot cache credentials/endpoints injected | no",
            self.workflow,
        )
        self.assertIn(
            "Depot registry credentials/config injected | no",
            self.workflow,
        )
        self.assertNotIn("expect_cache_hit", self.workflow)
        self.assertNotIn("actions/cache@", self.workflow)
        self.assertNotIn("sccache cc", self.workflow)
        self.assertNotIn("mozilla-actions/sccache-action@", self.workflow)
        self.assertNotIn("Exercise authenticated Depot sccache", self.workflow)
        self.assertIn('"$ImageOS" "$ImageVersion"', self.workflow)
        self.assertNotIn("printenv", self.workflow)

    def test_pr_audit_uses_portable_endpoint_and_docker_config_checks(self) -> None:
        action = (
            ROOT
            / ".github"
            / "actions"
            / "audit-depot-pr-isolation"
            / "action.yml"
        ).read_text(encoding="utf-8")
        self.assertNotIn(",,}", action)
        self.assertIn(
            "printf '%s' \"$endpoint\" | tr '[:upper:]' '[:lower:]'",
            action,
        )
        self.assertIn("depot_selected", action)
        self.assertIn("INPUT_DEPOT_SELECTED", action)
        self.assertIn('if [[ "$endpoint_lower" == *depot.dev* &&', action)
        self.assertIn('"$allow_native_github_cache" == "true"', action)
        self.assertIn("allow_native_github_cache", action)
        self.assertIn("allow_depot_remote_cache", action)
        self.assertIn("URL userinfo", action)
        self.assertIn('docker_config="${DOCKER_CONFIG:-${HOME:-}/.docker}/config.json"', action)
        self.assertIn('python_bin=""', action)
        self.assertIn('json.loads(raw, parse_constant=reject_constant)', action)
        self.assertIn('casefold()', action)
        self.assertIn('"auths", "credHelpers"', action)
        self.assertIn("Docker auth JSON is malformed", action)
        self.assertIn('depot_selected == "true"', action)

    def test_pr_audit_receives_central_runner_provider_selection(self) -> None:
        workflow_root = ROOT / ".github" / "workflows"
        audit_calls = 0
        for path in sorted(workflow_root.glob("*.yml")):
            lines = path.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines):
                if "audit-depot-pr-isolation@" not in line:
                    continue
                audit_calls += 1
                block = "\n".join(lines[line_number : line_number + 5])
                with self.subTest(path=path.name, line=line_number + 1):
                    self.assertIn("depot_selected:", block)
                    self.assertIn("startsWith(", block)
                    self.assertIn("needs.runner_policy.outputs.", block)
        self.assertEqual(audit_calls, 25)

    def test_endpoint_and_docker_auth_probes_fail_closed(self) -> None:
        action = (
            ROOT
            / ".github"
            / "actions"
            / "audit-depot-pr-isolation"
            / "action.yml"
        ).read_text(encoding="utf-8")
        action_run = action.split("      run: |\n", maxsplit=1)[1]
        action_script = dedent(action_run)
        runner_script = self._step_script(
            self._job_block("runner"),
            "Verify ephemeral runner resources",
        )
        cache_probe = re.search(
            r"^depot_selected=true\n(?P<body>.*?)(?=^\{\n|\Z)",
            runner_script,
            flags=re.MULTILINE | re.DOTALL,
        )
        if cache_probe is None:
            raise AssertionError("missing canary cache probe")
        canary_script = "set -euo pipefail\n" + cache_probe.group(0)

        def run_probe(
            script: str,
            cache_url: str,
            results_url: str,
            docker_auth_config: str | None = None,
            docker_config_content: str | None = None,
            depot_selected: str = "true",
            allow_native_github_cache: str | None = None,
            unset_endpoints: bool = False,
            **extra_environment: str,
        ) -> subprocess.CompletedProcess[str]:
            with (
                tempfile.TemporaryDirectory() as home,
                tempfile.TemporaryDirectory() as docker_config,
            ):
                if docker_config_content is not None:
                    (Path(docker_config) / "config.json").write_text(
                        docker_config_content,
                        encoding="utf-8",
                    )
                environment = {
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": home,
                    "DOCKER_CONFIG": docker_config,
                    "GITHUB_EVENT_NAME": "pull_request",
                    "INPUT_ORIGINAL_EVENT_NAME": "pull_request",
                    "INPUT_DEPOT_SELECTED": depot_selected,
                    "INPUT_ALLOW_NATIVE_GITHUB_CACHE": (
                        allow_native_github_cache
                        if allow_native_github_cache is not None
                        else "false" if depot_selected == "true" else "true"
                    ),
                    "INPUT_ALLOW_DEPOT_REMOTE_CACHE": "false",
                    "ACTIONS_CACHE_URL": cache_url,
                    "ACTIONS_RESULTS_URL": results_url,
                    "ACTIONS_RUNTIME_URL": "",
                    **extra_environment,
                }
                if unset_endpoints:
                    for endpoint_name in (
                        "ACTIONS_CACHE_URL",
                        "ACTIONS_RESULTS_URL",
                        "ACTIONS_RUNTIME_URL",
                    ):
                        environment.pop(endpoint_name, None)
                if docker_auth_config is not None:
                    environment["DOCKER_AUTH_CONFIG"] = docker_auth_config
                return subprocess.run(
                    ["bash", "-c", script],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )

        valid_endpoints = (
            (
                "https://actions.githubusercontent.com/cache",
                "https://results-receiver.actions.githubusercontent.com/results",
            ),
            (
                "https://actions.githubusercontent.com:443/cache",
                "https://results-receiver.actions.githubusercontent.com:8443/results?x=1#ok",
            ),
            (
                "HTTPS://ACTIONS.GITHUBUSERCONTENT.COM?cache=1",
                "https://results-receiver.actions.githubusercontent.com#results",
            ),
            (
                "http://[::1]:12345/_apis/artifactcache/",
                "HTTPS://LOCALHOST:12346/results",
            ),
        )
        hosted_endpoints = (
            (
                "http://127.0.0.1:12345/_apis/artifactcache/",
                "http://localhost:12346/",
            ),
        )
        invalid_endpoints = (
            (
                "https://actions.githubusercontent.com:443@attacker.example/",
                valid_endpoints[0][1],
            ),
            (
                "https://actions.githubusercontent.com@depot.dev/cache",
                valid_endpoints[0][1],
            ),
            (
                "http://actions.githubusercontent.com/cache",
                valid_endpoints[0][1],
            ),
            (
                "https://actions.githubusercontent.com.evil/cache",
                valid_endpoints[0][1],
            ),
            (
                "http://user@localhost:1234/cache",
                valid_endpoints[0][1],
            ),
            (
                "http://localhost.evil:1234/cache",
                valid_endpoints[0][1],
            ),
            (
                "https://cache.example.invalid/cache",
                valid_endpoints[0][1],
            ),
            (
                "http://[::1]:abc/cache",
                valid_endpoints[0][1],
            ),
            (
                "HTTPS://CACHE.DEPOT.DEV/cache",
                valid_endpoints[0][1],
            ),
        )
        absent_endpoints = (
            ("", ""),
            (valid_endpoints[0][0], ""),
            ("", valid_endpoints[0][1]),
        )
        for script_name, script in (
            ("audit action", action_script),
            ("Depot canary", canary_script),
        ):
            for endpoints in valid_endpoints:
                with self.subTest(script=script_name, endpoints=endpoints):
                    result = run_probe(script, *endpoints)
                    self.assertEqual(result.returncode, 0, result.stderr)
            if script_name == "audit action":
                with self.subTest(script=script_name, bounded_exception=True):
                    result = run_probe(
                        script,
                        "https://cache.depot.dev/cache",
                        "https://results.example.invalid/results",
                        allow_native_github_cache="true",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                with self.subTest(script=script_name, provider="malformed"):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        depot_selected="maybe",
                    )
                    self.assertNotEqual(result.returncode, 0)
                for endpoints in hosted_endpoints:
                    with self.subTest(
                        script=script_name,
                        hosted_endpoints=endpoints,
                    ):
                        result = run_probe(
                            script,
                            *endpoints,
                            depot_selected="false",
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
            for endpoints in absent_endpoints:
                with self.subTest(script=script_name, absent_endpoints=endpoints):
                    result = run_probe(script, *endpoints)
                    self.assertEqual(result.returncode, 0, result.stderr)
            with self.subTest(script=script_name, unset_endpoints=True):
                result = run_probe(
                    script,
                    "",
                    "",
                    unset_endpoints=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            for endpoints in invalid_endpoints:
                with self.subTest(script=script_name, endpoints=endpoints):
                    result = run_probe(script, *endpoints)
                    self.assertNotEqual(result.returncode, 0)
            if script_name == "Depot canary":
                diagnostic_cases = (
                    (
                        "unsupported host",
                        "https://cache.example.invalid/cache",
                        "https",
                        "other",
                        "absent",
                        "present",
                    ),
                    (
                        "Depot redirect",
                        "https://cache.depot.dev/cache",
                        "https",
                        "other",
                        "absent",
                        "present",
                    ),
                    (
                        "URL userinfo",
                        "https://user@attacker.example/cache",
                        "https",
                        "other",
                        "absent",
                        "present",
                    ),
                    (
                        "URL userinfo with numeric port",
                        "https://actions.githubusercontent.com:443@attacker.example/",
                        "https",
                        "other",
                        "absent",
                        "present",
                    ),
                    (
                        "numeric port",
                        "https://cache.example.invalid:8443/cache",
                        "https",
                        "other",
                        "present",
                        "present",
                    ),
                    (
                        "GitHub authority with HTTP",
                        "http://actions.githubusercontent.com/cache",
                        "http",
                        "github",
                        "absent",
                        "present",
                    ),
                    (
                        "other scheme",
                        "ftp://cache.example.invalid/cache",
                        "other",
                        "other",
                        "absent",
                        "present",
                    ),
                    (
                        "localhost authority",
                        "http://localhost/cache",
                        "http",
                        "localhost",
                        "absent",
                        "present",
                    ),
                    (
                        "IPv4 loopback authority",
                        "http://127.0.0.1:12345",
                        "http",
                        "127.0.0.1",
                        "present",
                        "absent",
                    ),
                    (
                        "IPv6 loopback authority",
                        "http://[::1]:65536/cache",
                        "http",
                        "ipv6-loopback",
                        "present",
                        "present",
                    ),
                )
                for endpoint_name in (
                    "ACTIONS_CACHE_URL",
                    "ACTIONS_RESULTS_URL",
                    "ACTIONS_RUNTIME_URL",
                ):
                    for (
                        diagnostic,
                        endpoint,
                        scheme,
                        authority,
                        numeric_port,
                        explicit_path,
                    ) in diagnostic_cases:
                        with self.subTest(
                            script=script_name,
                            endpoint_name=endpoint_name,
                            diagnostic=diagnostic,
                        ):
                            cache_url = endpoint if endpoint_name == "ACTIONS_CACHE_URL" else valid_endpoints[0][0]
                            results_url = endpoint if endpoint_name == "ACTIONS_RESULTS_URL" else valid_endpoints[0][1]
                            extra_environment = (
                                {"ACTIONS_RUNTIME_URL": endpoint}
                                if endpoint_name == "ACTIONS_RUNTIME_URL"
                                else {}
                            )
                            result = run_probe(
                                script,
                                cache_url,
                                results_url,
                                **extra_environment,
                            )
                            self.assertNotEqual(result.returncode, 0)
                            expected = (
                                "GitHub Actions endpoint rejected "
                                f"(variable={endpoint_name} scheme={scheme} "
                                f"authority={authority} numeric_port={numeric_port} "
                                f"explicit_path={explicit_path})"
                            )
                            self.assertIn(expected, result.stderr)
                            self.assertNotIn(endpoint, result.stderr)
                            for forbidden_fragment in (
                                "cache.example.invalid",
                                "cache.depot.dev",
                                "attacker.example",
                                "actions.githubusercontent.com",
                                "/cache",
                                "8443",
                                "65536",
                                "443",
                            ):
                                self.assertNotIn(forbidden_fragment, result.stderr)
            if script_name == "audit action":
                with self.subTest(
                    script=script_name,
                    policy="bounded native cache exception",
                ):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        INPUT_ALLOW_NATIVE_GITHUB_CACHE="true",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    malformed = run_probe(
                        script,
                        "ftp://cache.depot.dev/cache",
                        valid_endpoints[0][1],
                        allow_native_github_cache="true",
                    )
                    self.assertNotEqual(malformed.returncode, 0)
                    self.assertNotIn(
                        "ftp://cache.depot.dev/cache",
                        malformed.stderr,
                    )
                with self.subTest(script=script_name, policy="Depot remote cache enabled"):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        INPUT_ALLOW_DEPOT_REMOTE_CACHE="true",
                    )
                    self.assertNotEqual(result.returncode, 0)
                for endpoints in (
                    (
                        "https://cache.depot.dev/cache",
                        hosted_endpoints[0][1],
                    ),
                    (
                        hosted_endpoints[0][0],
                        "https://results.depot.dev/results",
                    ),
                ):
                    with self.subTest(
                        script=script_name,
                        depot_endpoint=endpoints,
                    ):
                        result = run_probe(
                            script,
                            *endpoints,
                            depot_selected="false",
                        )
                        self.assertNotEqual(result.returncode, 0)
                with self.subTest(script=script_name, userinfo="hosted"):
                    result = run_probe(
                        script,
                        *invalid_endpoints[0],
                        depot_selected="false",
                    )
                    self.assertNotEqual(result.returncode, 0)

        depot_auth = '{"auths":{"REGISTRY.DEPOT.DEV":{"auth":"secret"}}}'
        depot_config = '{"auths":{"registry.depot.dev":{"auth":"secret"}}}'
        escaped_depot_auth = r'{"auths":{"registry\u002eDEPOT\u002eDEV":{"auth":"secret"}}}'
        escaped_depot_helpers = r'{"credHelpers":{"REGISTRY\u002eDEPOT\u002eDEV":"secret"}}'
        safe_config = '{"auths":{"ghcr.io":{"auth":"secret"}}}'
        malformed_config = '{"auths":'
        for script_name, script in (
            ("audit action", action_script),
            ("Depot canary", canary_script),
        ):
            with self.subTest(script=script_name, auth="DOCKER_AUTH_CONFIG"):
                result = run_probe(
                    script,
                    *valid_endpoints[0],
                    docker_auth_config=depot_auth,
                )
                self.assertNotEqual(result.returncode, 0)
            with self.subTest(script=script_name, auth="config.json"):
                result = run_probe(
                    script,
                    *valid_endpoints[0],
                    docker_config_content=depot_config,
                )
                self.assertNotEqual(result.returncode, 0)
            for escaped_auth in (escaped_depot_auth, escaped_depot_helpers):
                with self.subTest(script=script_name, auth="escaped DOCKER_AUTH_CONFIG"):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        docker_auth_config=escaped_auth,
                    )
                    self.assertNotEqual(result.returncode, 0)
                with self.subTest(script=script_name, auth="escaped config.json"):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        docker_config_content=escaped_auth,
                    )
                    self.assertNotEqual(result.returncode, 0)
            with self.subTest(script=script_name, auth="any config.json"):
                result = run_probe(
                    script,
                    *valid_endpoints[0],
                    docker_config_content=safe_config,
                )
                self.assertNotEqual(result.returncode, 0)
            with self.subTest(script=script_name, auth="malformed DOCKER_AUTH_CONFIG"):
                result = run_probe(
                    script,
                    *valid_endpoints[0],
                    docker_auth_config=malformed_config,
                    depot_selected="false" if script_name == "audit action" else "true",
                )
                self.assertNotEqual(result.returncode, 0)
            with self.subTest(script=script_name, auth="malformed config.json"):
                result = run_probe(
                    script,
                    *valid_endpoints[0],
                    docker_config_content=malformed_config,
                    depot_selected="false" if script_name == "audit action" else "true",
                )
                self.assertNotEqual(result.returncode, 0)
            if script_name == "audit action":
                with self.subTest(script=script_name, provider="hosted", auth="DOCKER_AUTH_CONFIG"):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        docker_auth_config=safe_config,
                        depot_selected="false",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                with self.subTest(script=script_name, provider="hosted", auth="config.json"):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        docker_config_content=safe_config,
                        depot_selected="false",
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                with self.subTest(script=script_name, provider="hosted", auth="DOCKER_AUTH_CONFIG depot.dev"):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        docker_auth_config=depot_auth,
                        depot_selected="false",
                    )
                    self.assertNotEqual(result.returncode, 0)
                with self.subTest(script=script_name, provider="hosted", auth="config.json depot.dev"):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        docker_config_content=depot_config,
                        depot_selected="false",
                    )
                    self.assertNotEqual(result.returncode, 0)
                for escaped_auth in (escaped_depot_auth, escaped_depot_helpers):
                    with self.subTest(script=script_name, provider="hosted", auth="escaped DOCKER_AUTH_CONFIG"):
                        result = run_probe(
                            script,
                            *valid_endpoints[0],
                            docker_auth_config=escaped_auth,
                            depot_selected="false",
                        )
                        self.assertNotEqual(result.returncode, 0)
                    with self.subTest(script=script_name, provider="hosted", auth="escaped config.json"):
                        result = run_probe(
                            script,
                            *valid_endpoints[0],
                            docker_config_content=escaped_auth,
                            depot_selected="false",
                        )
                        self.assertNotEqual(result.returncode, 0)
                with self.subTest(script=script_name, provider="hosted", auth="malformed DOCKER_AUTH_CONFIG"):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        docker_auth_config=malformed_config,
                        depot_selected="false",
                    )
                    self.assertNotEqual(result.returncode, 0)
                with self.subTest(script=script_name, provider="hosted", auth="malformed config.json"):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        docker_config_content=malformed_config,
                        depot_selected="false",
                    )
                    self.assertNotEqual(result.returncode, 0)
            with self.subTest(script=script_name, auth="unset"):
                result = run_probe(script, *valid_endpoints[0])
                self.assertEqual(result.returncode, 0, result.stderr)

            if script_name == "audit action":
                with self.subTest(script=script_name, auth="DEPOT_CACHE_TOKEN"):
                    result = run_probe(
                        script,
                        *valid_endpoints[0],
                        DEPOT_CACHE_TOKEN="redacted-test-token",
                    )
                    self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
