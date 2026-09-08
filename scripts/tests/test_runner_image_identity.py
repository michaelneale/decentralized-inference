from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/runner-image-identity.py"
SPEC = importlib.util.spec_from_file_location("runner_image_identity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITY)


class RunnerImageIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(ROOT / ".github/workflows", self.root / ".github/workflows")
        for name in ("ci/runner-images.json", "ci/slices.yml", "ci/ownership.yml", "scripts/plan-ci.py", "just/ci.just"):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        shutil.copytree(ROOT / "ci/runner-image-evidence", self.root / "ci/runner-image-evidence")
        self.catalog = IDENTITY.read_json(self.root / "ci/runner-images.json")

    def replace(self, name: str, old: str, new: str, count: int = 1) -> None:
        path = self.root / name
        contents = path.read_text(encoding="utf-8")
        self.assertIn(old, contents)
        path.write_text(contents.replace(old, new, count), encoding="utf-8")

    def image(self, name: str) -> str:
        return self.catalog["images"][name]["reference"]

    def assert_drift(self, message: str) -> None:
        with self.assertRaisesRegex(IDENTITY.IdentityError, message):
            IDENTITY.check(self.catalog, self.root)

    def cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--root", str(self.root), *args],
            capture_output=True, text=True, check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    def test_current_consumers_and_actual_planner_agree_without_source_edits(self) -> None:
        paths = list((self.root / ".github/workflows").glob("*.yml")) + [self.root / "ci/slices.yml", self.root / "ci/ownership.yml"]
        before = {path: path.read_bytes() for path in paths}
        self.assertEqual(IDENTITY.check(self.catalog, self.root), {
            "images": 9, "roles": 32, "workflow_bindings": 33,
            "runtime_rows": 4, "seed_consumers": 6,
        })
        self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_historical_tools_provenance_and_seed_coverage_are_unknown(self) -> None:
        for image in (value for key, value in self.catalog["images"].items() if key not in ("public-ui", "public-browser")):
            self.assertIsNone(image["receipt"])
            self.assertIsNone(image["provenance"])
            self.assertNotIn("tools", image)
        self.assertIsNone(self.catalog["compiler_seed"]["workload_coverage"])
        result = self.cli("lookup", "release-ui-artifact", "--field", "receipt")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), None)

    def test_duplicate_json_keys_fail(self) -> None:
        path = self.root / "ci/runner-images.json"
        path.write_text('{"schema_version": 1, "schema_version": 1}')
        result = self.cli("validate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate JSON key", result.stderr)

    def test_catalog_rejects_mutable_reference(self) -> None:
        self.catalog["images"]["public-cpu"]["reference"] = IDENTITY.REPOSITORY + ":latest"
        self.assert_drift("immutable runner digest")

    def test_catalog_rejects_mismatched_epoch(self) -> None:
        self.catalog["images"]["public-cpu"]["native_toolchain_epoch"] = "different-epoch"
        self.assert_drift("epoch/digest mismatch")

    def test_catalog_rejects_unknown_image(self) -> None:
        self.catalog["consumer_roles"]["ui-quality"]["image_id"] = "missing"
        self.assert_drift("unknown image_id")

    def test_catalog_rejects_duplicate_consumer_binding(self) -> None:
        self.catalog["consumer_roles"]["extra-role"] = copy.deepcopy(self.catalog["consumer_roles"]["ui-quality"])
        self.assert_drift("duplicate consumer binding")

    def test_catalog_rejects_unverified_receipt_claim(self) -> None:
        self.catalog["images"]["public-cpu"]["receipt"] = {"path": "invented.json", "sha256": "a" * 64}
        self.assert_drift("receipt/provenance must be paired")

    def test_catalog_rejects_workflow_path_escape(self) -> None:
        self.catalog["consumer_roles"]["ui-quality"]["bindings"][0]["workflow"] = "../private.yml"
        self.assert_drift("invalid workflow")

    def test_ordinary_image_drift_fails(self) -> None:
        self.replace(".github/workflows/ci-quality-slice.yml", self.image("public-cpu"), self.image("public-web"))
        self.assert_drift("image reference drift")

    def test_conditional_container_opt_out_is_preserved(self) -> None:
        path = ".github/workflows/sdk-smoke.yml"
        self.replace(path, "inputs.sdk_kind != 'swift' &&", "inputs.sdk_kind == 'swift' &&")
        self.assert_drift("image reference drift")

    def test_distinct_release_rocm_pin_is_checked_read_only(self) -> None:
        self.assertNotEqual(self.image("public-rocm-ci"), self.image("public-rocm-release"))
        self.replace(".github/workflows/release.yml", self.image("public-rocm-release"), self.image("public-rocm-ci"))
        self.assert_drift("release-rocm.*image reference drift")

    def test_matrix_pins_cannot_swap_between_cuda_versions(self) -> None:
        path = self.root / ".github/workflows/release.yml"
        original = path.read_text()
        changed = original.replace(self.image("public-cuda12"), "PLACEHOLDER", 1)
        changed = changed.replace(self.image("public-cuda13"), self.image("public-cuda12"), 1)
        path.write_text(changed.replace("PLACEHOLDER", self.image("public-cuda13"), 1))
        self.assert_drift("release-cuda12-aarch64.*image reference drift")

    def test_matrix_epoch_consumer_cannot_ignore_declared_epoch(self) -> None:
        self.replace(".github/workflows/release.yml", "pinned_epoch: ${{ matrix.toolchain_epoch }}", "pinned_epoch: ignored")
        self.assert_drift("matrix epoch consumer drift")

    def test_new_unregistered_consumer_fails_census(self) -> None:
        path = self.root / ".github/workflows/ci-web-slice.yml"
        with path.open("a") as handle:
            handle.write("\n  extra_ui:\n    container:\n      image: " + self.image("public-web") + "\n")
        self.assert_drift("consumer census drift")

    def test_short_and_flow_container_syntax_cannot_escape_census(self) -> None:
        path = self.root / ".github/workflows/ci-web-slice.yml"
        original = path.read_text()
        reference = self.image("public-web")
        for declaration in ("container: " + reference, "container: {image: " + reference + "}"):
            with self.subTest(declaration=declaration):
                path.write_text(original + "\n  extra_ui:\n    " + declaration + "\n")
                self.assert_drift("unsupported runner image reference declaration")

    def test_yaml_workflow_extension_cannot_escape_census(self) -> None:
        path = self.root / ".github/workflows/extra-ui.yaml"
        path.write_text("name: Extra UI\non: workflow_dispatch\njobs:\n  extra_ui:\n"
                        "    container:\n      image: " + self.image("public-web") + "\n")
        self.assert_drift("consumer census drift")

    def test_registered_yaml_workflow_extension_is_checked(self) -> None:
        original, renamed = "ci-web-slice.yml", "ci-web-slice.yaml"
        (self.root / ".github/workflows" / original).rename(self.root / ".github/workflows" / renamed)
        for role in self.catalog["consumer_roles"].values():
            for binding in role["bindings"]:
                if binding["workflow"] == original:
                    binding["workflow"] = renamed
        self.assertEqual(IDENTITY.check(self.catalog, self.root)["workflow_bindings"], 33)
        self.replace(".github/workflows/" + renamed, self.image("public-web"), self.image("public-cpu"))
        self.assert_drift("image reference drift")

    def test_duplicate_image_field_fails_closed(self) -> None:
        image_line = "      image: " + self.image("public-web")
        self.replace(".github/workflows/ci-web-slice.yml", image_line, image_line + "\n" + image_line)
        self.assert_drift("exactly one image")

    def test_runtime_manifest_image_drift_fails(self) -> None:
        self.replace("ci/slices.yml", self.image("public-rocm-ci"), self.image("public-rocm-release"))
        self.assert_drift("linux-rocm: planner image drift")

    def test_runtime_manifest_epoch_drift_fails(self) -> None:
        epoch = self.catalog["images"]["public-cpu"]["native_toolchain_epoch"]
        self.replace("ci/slices.yml", epoch, "changed-native-epoch")
        self.assert_drift("linux-cpu: planner epoch drift")

    def test_actual_planner_output_is_checked_not_only_manifest(self) -> None:
        self.replace("scripts/plan-ci.py", "result.append(dict(mapping[row_id]))",
                     "result.append({**mapping[row_id], 'container_image': 'planner-replaced-image'} "
                     "if field == 'runtime_rows' and row_id == 'linux-cpu' else dict(mapping[row_id]))")
        self.assert_drift("linux-cpu: planner image drift")

    def test_runtime_workflow_must_consume_planned_image(self) -> None:
        self.replace(".github/workflows/ci-linux-runtime-slice.yml", "image: ${{ matrix.runtime.container_image }}", "image: ignored")
        self.assert_drift("runtime matrix image consumer drift")

    def test_consumer_seed_key_drift_fails(self) -> None:
        prefix = self.catalog["compiler_seed"]["key_prefix"]
        self.replace(".github/workflows/ci-rust-tests-slice.yml", prefix, prefix.replace("-v2-", "-v99-"))
        self.assert_drift("compiler seed key drift")

    def test_publisher_seed_recipe_hash_drift_fails(self) -> None:
        self.replace(".github/workflows/cache-warm-sccache.yml", "'Justfile', 'just/**'", "'Justfile', 'just/ci.just'")
        self.assert_drift("compiler seed key drift")

    def test_publisher_cannot_save_under_a_different_key(self) -> None:
        path = self.root / ".github/workflows/cache-warm-sccache.yml"
        original = path.read_text()
        prefix, final_key = original.rsplit("key: ${{ steps.seed.outputs.key }}", 1)
        path.write_text(prefix + "key: unrelated-cache-key" + final_key)
        self.assert_drift("publisher cache key drift")

    def test_literal_seed_key_comment_cannot_hide_a_different_restore_key(self) -> None:
        expression = IDENTITY.seed_expression(self.catalog)
        self.replace(".github/workflows/ci-linux-host-slice.yml", "cache_key: " + expression,
                     "# Previous key: " + expression + "\n          cache_key: unrelated-cache-key")
        self.assert_drift("compiler seed restore key drift")

    def test_restore_consumer_census_is_independent_of_key_literals(self) -> None:
        self.replace(".github/workflows/ci-linux-host-slice.yml", "uses: ./.github/actions/restore-sccache-seed", "uses: ./.github/actions/another-action")
        self.assert_drift("restore action census drift")

    def test_runtime_seed_guard_image_drift_fails(self) -> None:
        self.replace(".github/workflows/ci-linux-runtime-slice.yml", self.image("public-cpu"), self.image("public-web"))
        self.assert_drift("runtime seed guard container_image drift")

    def test_architecture_diagnostics_do_not_claim_workload_coverage(self) -> None:
        path = self.root / ".github/workflows/ci-linux-runtime-slice.yml"
        import re
        path.write_text(re.sub(r"matrix\.runtime\.architecture == '[^']+'", "matrix.runtime.architecture == 'fixture-mismatch'", path.read_text()))
        messages = IDENTITY.diagnose(self.catalog, self.root)
        self.assertTrue(any("fixture-mismatch" in message and "separate workload coverage decision" in message for message in messages))
        self.assertTrue(any("workload coverage is unqualified" in message for message in messages))

    def test_sdk_rust_epoch_stays_separate_from_native_epoch(self) -> None:
        sdk = self.catalog["sdk_rust"]
        native = self.catalog["images"]["public-cpu"]["native_toolchain_epoch"]
        self.assertNotEqual(sdk["toolchain_epoch"], native)
        self.replace(".github/workflows/sdk-smoke.yml", "SDK_RUST_TOOLCHAIN_EPOCH: " + sdk["toolchain_epoch"], "SDK_RUST_TOOLCHAIN_EPOCH: " + native)
        self.assert_drift("SDK Rust TOOLCHAIN_EPOCH drift")

    def test_sdk_rust_action_drift_fails(self) -> None:
        self.replace(".github/workflows/sdk-smoke.yml", self.catalog["sdk_rust"]["rust_action_ref"], "dtolnay/rust-toolchain@" + "a" * 40)
        self.assert_drift("SDK Rust action drift")

    def test_sdk_rust_cache_expression_must_use_epoch(self) -> None:
        self.replace(".github/workflows/sdk-smoke.yml", "env.SDK_RUST_TOOLCHAIN_EPOCH,", "'ignored-epoch',")
        self.assert_drift("SDK Rust cache key expression drift")

    def test_lookup_and_seed_key_keep_legacy_outputs(self) -> None:
        lookup = self.cli("lookup", "rust-clippy", "--field", "reference")
        self.assertEqual(lookup.returncode, 0, lookup.stderr)
        self.assertEqual(lookup.stdout.strip(), self.image("public-cpu"))
        result = self.cli("seed-key", "--recipe-hash", "a" * 64)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "mesh-llm-sccache-seed-linux-x86_64-img-8d93de6b-epoch-8d93de6b-v2-" + "a" * 64)

    def test_publisher_and_sdk_roles_require_single_bindings(self) -> None:
        for role_id, message in (
            (self.catalog["compiler_seed"]["publisher_role"], "publisher role must declare exactly one binding"),
            (self.catalog["sdk_rust"]["role"], "sdk_rust: role must declare exactly one binding"),
        ):
            with self.subTest(role=role_id):
                catalog = copy.deepcopy(self.catalog)
                extra = copy.deepcopy(catalog["consumer_roles"][role_id]["bindings"][0])
                extra["job"] = "additional_binding"
                catalog["consumer_roles"][role_id]["bindings"].append(extra)
                with self.assertRaisesRegex(IDENTITY.IdentityError, message):
                    IDENTITY.validate(catalog, self.root)

    def test_product_smoke_role_keeps_multiple_bindings(self) -> None:
        self.assertEqual(len(self.catalog["consumer_roles"]["product-smoke"]["bindings"]), 2)
        IDENTITY.validate(self.catalog, self.root)

    def test_diagnose_cli_rejects_missing_or_ambiguous_planner_row(self) -> None:
        row_id = self.catalog["compiler_seed"]["runtime_consumer"]["row_id"]
        row = {"id": row_id, "architecture": "amd64"}
        for rows in ([], [row, copy.deepcopy(row)]):
            with self.subTest(rows=rows):
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(IDENTITY, "planner_rows", return_value=rows), \
                     patch.object(sys, "argv", [str(SCRIPT), "--root", str(self.root), "diagnose"]), \
                     redirect_stdout(stdout), redirect_stderr(stderr):
                    result = IDENTITY.main()
                self.assertEqual(result, 1)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn(f"runner image identity: planner has no unique row {row_id}", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_failure_never_reports_success(self) -> None:
        for arguments in (("lookup", "missing-role"), ("seed-key", "--recipe-hash", "../../bad")):
            with self.subTest(arguments=arguments):
                result = self.cli(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("runner image identity:", result.stderr)


    def test_release_tag_branches_cannot_be_swapped_or_collapsed(self) -> None:
        path = ".github/workflows/ci-ui-artifact-slice.yml"
        original = (self.root / path).read_text()
        for replacement in ("inputs.release_tag == ''", "inputs.source_sha != ''"):
            with self.subTest(replacement=replacement):
                (self.root / path).write_text(original.replace("inputs.release_tag != '' &&", replacement + " &&"))
                self.assert_drift("conditional image reference drift")
        (self.root / path).write_text(original)
        del self.catalog["consumer_roles"]["release-ui-artifact"]
        self.assert_drift("complete UI artifact pair")

    def test_release_tag_selector_is_bounded_to_artifact_job(self) -> None:
        self.catalog["consumer_roles"]["ui-artifact"]["bindings"][0]["job"] = "ui_quality"
        self.assert_drift("complete UI artifact pair")

    def test_qualified_lean_receipts_match_retained_admission(self) -> None:
        for family in ("ui", "browser"):
            image = self.catalog["images"]["public-" + family]
            self.assertEqual(image["receipt"]["index_candidate_key"], "candidate-index-public-" + family)
            self.assertEqual(image["provenance"]["origin"]["run_id"], 34256062098)
            self.assertEqual(image["provenance"]["origin"]["run_attempt"], 1)
            self.assertEqual(image["provenance"]["validation"], "offline_binding_only")
        self.catalog["images"]["public-ui"]["receipt"]["index_candidate_key"] = "candidate-index-public-browser"
        self.assert_drift("image/index binding mismatch")

    def test_ordinary_and_release_artifact_workloads_keep_their_image_roles(self) -> None:
        self.assertEqual(self.catalog["consumer_roles"]["ui-artifact"]["image_id"], "public-ui")
        self.assertEqual(self.catalog["consumer_roles"]["release-ui-artifact"]["image_id"], "public-web")
        workflow = (self.root / ".github/workflows/ci-ui-artifact-slice.yml").read_text()
        self.assertIn("if: ${{ inputs.release_tag != '' }}", workflow)
        self.assertIn('scripts/release-version.sh "$RELEASE_TAG"', workflow)
        self.assertIn("run: pnpm run build", workflow)

    def test_unregistered_mutable_image_is_not_hidden_from_census(self) -> None:
        path = self.root / ".github/workflows/unregistered.yml"
        path.write_text("jobs:\n  extra:\n    container:\n      image: " + IDENTITY.REPOSITORY + ":public-latest\n")
        self.assert_drift("consumer census drift")

    def test_extra_unconditional_binding_on_conditional_job_fails_cleanly(self) -> None:
        extra = copy.deepcopy(self.catalog["consumer_roles"]["ui-artifact"])
        extra["bindings"][0]["matrix_selector"] = None
        self.catalog["consumer_roles"]["extra-artifact"] = extra
        (self.root / "ci/runner-images.json").write_text(json.dumps(self.catalog))
        result = self.cli("check")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("complete UI artifact pair", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_seed_guard_reference_outside_runtime_consumer_fails(self) -> None:
        guard = "          allow_trusted_seed: " + IDENTITY.REPOSITORY + ":unregistered\n"
        cases = {
            "other workflow": ("extra.yml", "jobs:\n  extra:\n    steps:\n      - uses: example/action@sha\n        with:\n" + guard),
            "other job": ("ci-linux-runtime-slice.yml", "  extra:\n    steps:\n      - uses: example/action@sha\n        with:\n" + guard),
            "outside jobs": ("ci-linux-runtime-slice.yml", "\nenv:\n" + guard),
        }
        for name, (workflow, addition) in cases.items():
            with self.subTest(case=name):
                path = self.root / ".github/workflows" / workflow
                original = path.read_text() if path.exists() else None
                path.write_text((original or "") + addition)
                result = self.cli("check")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("seed guard image references are limited", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                if original is None:
                    path.unlink()
                else:
                    path.write_text(original)

    def test_runtime_seed_canary_key_producer_cannot_be_overridden(self) -> None:
        path = self.root / ".github/workflows/depot-canary.yml"
        original = path.read_text()
        command = "python3 scripts/runtime-seed-canary.py preflight runtime-seed-evidence"
        mutations = {
            "override": original.replace(command, command + "; echo key=other >> $GITHUB_OUTPUT"),
            "missing": original.replace("id: seed_key", "id: other_key"),
            "duplicate": original + "      - name: duplicate key\n        id: seed_key\n        run: true\n",
            "unused expression": original.replace("CANARY_KEY:", "UNUSED_KEY:"),
        }
        for name, contents in mutations.items():
            with self.subTest(case=name):
                # Duplicate inside the canary job, not after the summary job.
                if name == "duplicate":
                    contents = original.replace("  runtime_seed_summary:", "      - name: duplicate key\n        id: seed_key\n        run: true\n\n  runtime_seed_summary:")
                path.write_text(contents)
                with self.assertRaises(IDENTITY.IdentityError):
                    IDENTITY.check(self.catalog, self.root)
        path.write_text(original)

if __name__ == "__main__":
    unittest.main()
