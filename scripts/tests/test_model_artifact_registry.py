from __future__ import annotations

import hashlib
import fnmatch
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "ci" / "model-artifacts" / "registry.json"
MANIFESTS = ROOT / "ci" / "model-artifacts" / "manifests"
GENERATOR = ROOT / "scripts" / "generate-test-model-manifests.py"
RESOLVER = ROOT / "scripts" / "resolve-test-model-manifest.py"


class ModelArtifactRegistryTests(unittest.TestCase):
    def test_generated_manifests_are_current(self) -> None:
        subprocess.run(["python3", str(GENERATOR), "--check"], cwd=ROOT, check=True)

    def test_suite_manifests_match_registry_membership(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        registry_sha = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
        artifacts = {row["id"]: row for row in registry["artifacts"]}
        declared_suites = set(registry["suites"])
        generated_suites = set()

        for path in sorted(MANIFESTS.glob("*.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            suite = manifest["suite"]
            generated_suites.add(suite)
            self.assertEqual(manifest["registry_sha256"], registry_sha)
            expected_ids = [
                row["id"] for row in registry["artifacts"] if suite in row["suites"]
            ]
            self.assertEqual([row["id"] for row in manifest["artifacts"]], expected_ids)
            for generated in manifest["artifacts"]:
                source = artifacts[generated["id"]]["artifact"]
                self.assertEqual(generated["repo"], source["repo"])
                self.assertEqual(generated["revision"], source["revision"])
                self.assertEqual(generated["files"], [row["path"] for row in source["files"]])

        self.assertEqual(
            generated_suites,
            declared_suites - {"llama-family-certification"},
        )

    def test_suite_manifests_allow_every_executable_cadence(self) -> None:
        required = {
            "product-smoke": {"pull-request", "main", "release"},
            "product-integration-smoke": {"pull-request", "main", "release"},
            "scripted-binary-smoke": {"pull-request", "main", "release"},
            "sdk-smoke": {"pull-request", "main", "release"},
            "hf-download-smoke": {"pull-request", "main", "manual"},
            "openai-smoke": {"manual"},
            "skippy-correctness": {"pull-request", "main", "manual"},
            "safetensors-runtime-smoke": {"pull-request"},
            "skippy-ci-smoke": {"manual"},
            "skippy-parity": {"manual"},
            "competitive-benchmark": {"manual"},
            "radix-cache": {"manual"},
        }
        for suite, required_cadences in required.items():
            manifest = json.loads(
                (MANIFESTS / f"{suite}.json").read_text(encoding="utf-8")
            )
            for artifact in manifest["artifacts"]:
                with self.subTest(suite=suite, artifact=artifact["id"]):
                    self.assertTrue(required_cadences.issubset(artifact["cadences"]))

    def test_product_integration_manifest_is_the_pinned_dense_recurrent_pair(self) -> None:
        manifest = json.loads(
            (MANIFESTS / "product-integration-smoke.json").read_text(
                encoding="utf-8"
            )
        )
        artifacts = {row["id"]: row for row in manifest["artifacts"]}

        self.assertEqual(
            set(artifacts),
            {"smollm2-q8-inference", "family-granite-hybrid"},
        )
        self.assertEqual(
            artifacts["smollm2-q8-inference"]["model_ref"],
            "unsloth/SmolLM2-135M-Instruct-GGUF:Q8_0",
        )
        self.assertEqual(
            artifacts["family-granite-hybrid"]["model_ref"],
            "ibm-granite/granite-4.0-h-350m-GGUF:Q4_K_M",
        )
        for artifact in artifacts.values():
            self.assertEqual(len(artifact["files"]), 1)
            self.assertEqual(
                artifact["sha256"], artifact["file_integrity"][artifact["file"]]["blob_id"]
            )

    def test_family_manifest_is_generated_from_registry(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        family = json.loads(
            (ROOT / "ci" / "llama-canary" / "family-certified.json").read_text(
                encoding="utf-8"
            )
        )
        expected = [
            row["family"]
            for row in registry["artifacts"]
            if "llama-family-certification" in row["suites"]
        ]
        self.assertEqual([row["family"] for row in family["models"]], expected)
        self.assertEqual(family["policy"], registry["family_policy"])

    def test_opt_in_suite_configs_reference_registered_variants(self) -> None:
        manifests = {}
        for suite in ("competitive-benchmark", "radix-cache", "skippy-parity"):
            data = json.loads((MANIFESTS / f"{suite}.json").read_text(encoding="utf-8"))
            manifests[suite] = {row["id"]: row for row in data["artifacts"]}

        competitive = json.loads(
            (ROOT / "evals" / "skippy-competitive-benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        for model in competitive["models"]:
            artifact = manifests["competitive-benchmark"][model["artifact_id"]]
            self.assertEqual(model["repo"], artifact["repo"])
            self.assertEqual(model["revision"], artifact["revision"])
            self.assertEqual(model["filename"], artifact["file"])
            self.assertEqual(model["sha256"], artifact["sha256"])

        radix = json.loads(
            (ROOT / "evals" / "skippy-radix-cache-models.json").read_text(
                encoding="utf-8"
            )
        )
        for case in radix["cases"]:
            artifact_id = case.get("artifact_id")
            if artifact_id is None:
                continue
            artifact = manifests["radix-cache"][artifact_id]
            self.assertEqual(case["source"]["repo"], artifact["repo"])
            self.assertEqual(case["source"]["revision"], artifact["revision"])
            self.assertEqual(case["source"]["filename"], artifact["file"])

        parity = json.loads(
            (ROOT / "docs" / "skippy" / "llama-parity-candidates.json").read_text(
                encoding="utf-8"
            )
        )
        for candidate in parity["candidates"]:
            artifact_id = candidate.get("artifact_id")
            if artifact_id is None:
                continue
            artifact = manifests["skippy-parity"][artifact_id]
            self.assertEqual(candidate["repo"], artifact["repo"])
            patterns = candidate.get("include", "*.gguf")
            if isinstance(patterns, str):
                patterns = [patterns]
            self.assertTrue(
                all(any(fnmatch.fnmatch(name, pattern) for pattern in patterns) for name in artifact["files"])
            )

    def test_executable_manifest_consumers_declare_cadence(self) -> None:
        consumers = (
            ".github/actions/restore-smoke-inputs/action.yml",
            ".github/actions/restore-product-integration-inputs/action.yml",
            ".github/workflows/ci-rust-tests-slice.yml",
            "scripts/ci-hf-download-smoke.sh",
            "scripts/materialize-competitive-inputs.sh",
            "scripts/skippy-ci-smoke.sh",
            "scripts/skippy-openai-smoke.sh",
        )
        invocation = re.compile(r"resolve-test-model-manifest\.py")
        for relative in consumers:
            content = (ROOT / relative).read_text(encoding="utf-8")
            matches = list(invocation.finditer(content))
            self.assertTrue(matches, relative)
            for match in matches:
                with self.subTest(consumer=relative, offset=match.start()):
                    self.assertIn("--cadence", content[match.start() : match.start() + 500])

        parity = (ROOT / "scripts" / "download-skippy-parity-candidates.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('"manual" not in artifact.get("cadences", [])', parity)

    def test_resolver_prefixes_github_outputs_for_multi_fixture_consumers(self) -> None:
        manifest = MANIFESTS / "product-integration-smoke.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            result = subprocess.run(
                [
                    "python3", str(RESOLVER), str(manifest),
                    "--artifact-id", "smollm2-q8-inference",
                    "--cadence", "pull-request",
                    "--require-single-file",
                    "--github-output", str(output),
                    "--github-output-prefix", "dense_",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(result.stderr, "")
            self.assertIn("dense_file=SmolLM2-135M-Instruct-Q8_0.gguf", output.read_text())

    def test_smoke_identity_overrides_require_nonempty_values(self) -> None:
        skippy = (ROOT / "scripts" / "skippy-ci-smoke.sh").read_text(
            encoding="utf-8"
        )
        openai = (ROOT / "scripts" / "skippy-openai-smoke.sh").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("DENSE_MODEL_REPO+x", skippy)
        self.assertNotIn("RECURRENT_MODEL_REPO+x", skippy)
        self.assertNotIn("MODEL_REPO+x", openai)
        for variable in (
            "DENSE_MODEL_REPO",
            "DENSE_MODEL_FILE",
            "DENSE_MODEL_SELECTOR",
            "DENSE_MODEL_REVISION",
            "DENSE_MODEL_PATH",
        ):
            self.assertIn(f'${{{variable}:-}}', skippy)
        for variable in (
            "RECURRENT_MODEL_REPO",
            "RECURRENT_MODEL_FILE",
            "RECURRENT_MODEL_SELECTOR",
            "RECURRENT_MODEL_REVISION",
            "RECURRENT_MODEL_PATH",
        ):
            self.assertIn(f'${{{variable}:-}}', skippy)
        for variable in (
            "MODEL_REPO",
            "MODEL_FILE",
            "MODEL_SELECTOR",
            "MODEL_REVISION",
            "MODEL_PATH",
        ):
            self.assertIn(f'${{{variable}:-}}', openai)

    def test_hf_smoke_exports_selected_manifest_to_rust_tests(self) -> None:
        smoke = (ROOT / "scripts" / "ci-hf-download-smoke.sh").read_text(
            encoding="utf-8"
        )
        fixture = (ROOT / "crates" / "model-hf" / "tests" / "hf_download.rs").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'export MESH_HF_DOWNLOAD_TEST_MANIFEST="$MODEL_MANIFEST"', smoke
        )
        self.assertIn('std::env::var_os("MESH_HF_DOWNLOAD_TEST_MANIFEST")', fixture)
        self.assertNotIn("include_str!", fixture)

    def test_resolver_verifies_size_and_digest(self) -> None:
        payload = b"immutable fixture\n"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture.bin").write_bytes(payload)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "manifest_kind": "test-model-artifacts",
                        "artifacts": [
                            {
                                "id": "fixture",
                                "repo": "org/repo",
                                "revision": "a" * 40,
                                "selector": "fixture",
                                "model_ref": "org/repo:fixture",
                                "cadences": ["manual"],
                                "files": ["fixture.bin"],
                                "file_integrity": {
                                    "fixture.bin": {
                                        "size_bytes": len(payload),
                                        "blob_id": digest,
                                    }
                                },
                                "urls": ["https://example.invalid/fixture.bin"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "python3",
                    str(RESOLVER),
                    str(manifest),
                    "--cadence",
                    "manual",
                    "--verify-root",
                    str(root),
                ],
                check=True,
            )
            (root / "fixture.bin").write_bytes(b"tampered fixture\n")
            result = subprocess.run(
                [
                    "python3",
                    str(RESOLVER),
                    str(manifest),
                    "--cadence",
                    "manual",
                    "--verify-root",
                    str(root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mismatch", result.stderr)

    def test_resolver_rejects_undeclared_cadence(self) -> None:
        manifest = MANIFESTS / "product-smoke.json"
        result = subprocess.run(
            [
                "python3",
                str(RESOLVER),
                str(manifest),
                "--cadence",
                "manual",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not allowed at cadence", result.stderr)

    def test_resolver_single_file_mode_rejects_multipart_artifact(self) -> None:
        manifest = MANIFESTS / "hf-download-smoke.json"
        result = subprocess.run(
            [
                "python3",
                str(RESOLVER),
                str(manifest),
                "--artifact-id",
                "gemma3-bf16-metadata",
                "--cadence",
                "manual",
                "--require-single-file",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must contain exactly one file", result.stderr)


if __name__ == "__main__":
    unittest.main()
