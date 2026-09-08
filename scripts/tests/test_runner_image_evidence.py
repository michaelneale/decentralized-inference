from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("evidence", ROOT / "scripts/runner-image-evidence.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.raw = (ROOT / "scripts/tests/fixtures/runner-image-evidence/synthetic-cohort.json").read_bytes()
        self.cohort = E.decode(self.raw)
        self.catalog = json.loads((ROOT / "ci/runner-images.json").read_text())
        self.image = self.catalog["images"]["public-cpu"]
        # Only test copies use the synthetic index; checked-in pins remain unchanged.
        self.image["reference"] = E.IMAGE + "@" + self.cohort["candidates"]["candidate-index-public-cpu"]["digest"]
        self.image["native_toolchain_epoch"] = "mesh-llm-cuda-runner-sha256-" + "8" * 64
        self.catalog["compiler_seed"]["key_prefix"] = "mesh-llm-sccache-seed-linux-x86_64-img-88888888-epoch-88888888-v2-"
        self.anchor = {
            "receipt": {
                "schema": 1,
                "cohort_sha256": E.hash_bytes(self.raw),
                "index_candidate_key": "candidate-index-public-cpu",
            },
            "provenance": {
                "schema": 1,
                "scope": "reviewed_producer_admission",
                "validation": "offline_binding_only",
                "cohort_sha256": E.hash_bytes(self.raw),
                "origin": self.cohort["origin"],
                "admission_validator_revision": "f" * 40,
            },
        }
        self.cohort_path = self.root / "input.json"
        self.cohort_path.write_bytes(self.raw)
        self.anchor_path = self.root / "anchor.json"
        self.anchor_path.write_text(json.dumps(self.anchor))

    def qualified(self):
        value = copy.deepcopy(self.image)
        value.update(copy.deepcopy(self.anchor))
        return value

    def test_actual_producer_bound_identity_and_fresh_proposal(self):
        E.validate_binding(self.qualified(), self.raw)
        before = copy.deepcopy(self.catalog)
        output = self.root / "proposal"
        E.bind(self.catalog, "public-cpu", self.cohort_path, self.anchor_path, output)
        self.assertEqual(self.catalog, before)
        proposed = json.loads((output / "ci/runner-images.json").read_text())
        E.validate_catalog(proposed, output)
        with self.assertRaisesRegex(ValueError, "must not exist"):
            E.bind(self.catalog, "public-cpu", self.cohort_path, self.anchor_path, output)

    def test_bytes_and_anchor_cannot_be_silently_substituted(self):
        with self.assertRaisesRegex(ValueError, "bytes"):
            E.validate_binding(self.qualified(), self.raw + b" ")
        for key in ("repository_id", "workflow_id", "run_id", "run_attempt", "mesh_revision", "runner_images_revision"):
            image = self.qualified()
            image["provenance"]["origin"][key] = 999 if type(image["provenance"]["origin"][key]) is int else "d" * 40
            with self.assertRaisesRegex(ValueError, "origin"):
                E.validate_binding(image, self.raw)
        image = self.qualified()
        image["provenance"]["scope"] = "attestation_verified"
        with self.assertRaisesRegex(ValueError, "trust claim"):
            E.validate_binding(image, self.raw)

    def test_binding_consistency_even_after_reviewed_hash_changes(self):
        index = ("candidates", "candidate-index-public-cpu")
        platform = ("platforms", "public-cpu-amd64")
        receipt = platform + ("receipt",)
        cases = {
            "wrong index digest": (index + ("digest",), "sha256:" + "f" * 64),
            "wrong family": (index + ("backend", "id"), "web"),
            "wrong normalized backend": (index + ("backend", "name"), "bogus"),
            "wrong architecture": (receipt + ("platform", "architecture"), "arm64"),
            "wrong source": (receipt + ("runtime", "source", "mesh_revision"), "f" * 40),
            "wrong manifest": (receipt + ("oci", "manifest", "digest"), "sha256:" + "f" * 64),
            "missing cache observations": (receipt + ("runtime", "cache"), None),
            "boolean platform schema": (platform + ("candidate", "schema"), True),
            "unknown platform field": (platform + ("candidate", "extra"), "unknown"),
            "unsupported cohort schema": (("schema",), 2),
            "unknown cohort field": (("extra",), "unknown"),
        }
        for name, (path, value) in cases.items():
            with self.subTest(case=name):
                cohort = copy.deepcopy(self.cohort)
                parent = cohort
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = value
                self.reject_reanchored(cohort)

    def reject_reanchored(self, cohort):
        raw = json.dumps(cohort).encode()
        image = self.qualified()
        image["receipt"]["cohort_sha256"] = E.hash_bytes(raw)
        image["provenance"]["cohort_sha256"] = E.hash_bytes(raw)
        with self.assertRaises((ValueError, KeyError)):
            E.validate_binding(image, raw)

    def test_missing_and_duplicate_platform_children_fail(self):
        for case in ("duplicate child", "missing child", "unlisted family platform"):
            with self.subTest(case=case):
                cohort = copy.deepcopy(self.cohort)
                children = cohort["candidates"]["candidate-index-public-cpu"]["children"]
                if case == "duplicate child":
                    children.append(copy.deepcopy(children[0]))
                elif case == "missing child":
                    children.clear()
                else:
                    cohort["platforms"]["public-cpu-arm64"] = copy.deepcopy(
                        cohort["platforms"]["public-cpu-amd64"]
                    )
                self.reject_reanchored(cohort)

    def test_immutable_anchor_conflict_leaves_no_proposal(self):
        self.image.update(copy.deepcopy(self.anchor))
        self.image["provenance"]["admission_validator_revision"] = "a" * 40
        output = self.root / "conflicting-proposal"
        with self.assertRaisesRegex(ValueError, "immutable binding conflict"):
            E.bind(self.catalog, "public-cpu", self.cohort_path, self.anchor_path, output)
        self.assertFalse(output.exists())

    def test_tool_and_cache_bytes_are_covered_by_anchor(self):
        for field in ("tools", "cache", "expected_tools", "dependencies"):
            changed = copy.deepcopy(self.cohort)
            changed["platforms"]["public-cpu-amd64"]["receipt"]["runtime"][field] = {}
            with self.assertRaisesRegex(ValueError, "bytes"):
                E.validate_binding(self.qualified(), json.dumps(changed).encode())

    def test_bounds_duplicates_and_symlinks(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":9007199254740992}', b'{"x":NaN}', b'[' * 65 + b'0' + b']' * 65, b' ' * (E.MAX_BYTES + 1)):
            with self.assertRaises(ValueError):
                E.decode(raw)
        (self.root / "ci").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            E.evidence_path(self.root, E.hash_bytes(self.raw))
        link = self.root / "link.json"
        link.symlink_to(self.cohort_path)
        with self.assertRaisesRegex(ValueError, "regular"):
            E.read_bytes(link)

    def test_sequential_proposals_retain_shared_and_distinct_evidence(self):
        for shared in (True, False):
            with self.subTest(shared=shared):
                first = self.root / ("first-" + str(shared))
                E.bind(self.catalog, "public-cpu", self.cohort_path, self.anchor_path, first)
                catalog = json.loads((first / "ci/runner-images.json").read_text())
                catalog["images"]["public-vulkan"]["reference"] = E.IMAGE + "@sha256:" + "7" * 64
                catalog["images"]["public-vulkan"]["native_toolchain_epoch"] = "mesh-llm-cuda-runner-sha256-" + "7" * 64
                raw = self.raw if shared else self.raw + b" "
                source = self.root / ("second-input-" + str(shared))
                source.write_bytes(raw)
                anchor = copy.deepcopy(self.anchor)
                anchor["receipt"]["index_candidate_key"] = "candidate-index-public-vulkan"
                anchor["receipt"]["cohort_sha256"] = anchor["provenance"]["cohort_sha256"] = E.hash_bytes(raw)
                anchor_path = self.root / ("second-anchor-" + str(shared))
                anchor_path.write_text(json.dumps(anchor))
                second = self.root / ("second-" + str(shared))
                E.bind(catalog, "public-vulkan", source, anchor_path, second, first)
                E.validate_catalog(json.loads((second / "ci/runner-images.json").read_text()), second)
                self.assertEqual(len(list((second / "ci/runner-image-evidence").iterdir())), 1 if shared else 2)

    def test_deep_json_cli_has_concise_error_without_traceback(self):
        (self.root / "ci").mkdir()
        (self.root / "ci/runner-images.json").write_bytes(b"[" * 2000 + b"0" + b"]" * 2000)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/runner-image-identity.py"), "--root", str(self.root), "validate"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("nesting", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_candidate_alias_cannot_replace_family_key(self):
        cohort = copy.deepcopy(self.cohort)
        cohort["candidates"]["alias"] = cohort["candidates"].pop("candidate-index-public-cpu")
        raw = json.dumps(cohort).encode()
        image = self.qualified()
        image["receipt"]["index_candidate_key"] = "alias"
        image["receipt"]["cohort_sha256"] = image["provenance"]["cohort_sha256"] = E.hash_bytes(raw)
        with self.assertRaisesRegex(ValueError, "key/family"):
            E.validate_binding(image, raw)

    def test_cli_bind_and_proposal_validate_use_explicit_root(self):
        (self.root / "ci").mkdir()
        catalog_path = self.root / "ci/runner-images.json"
        catalog_path.write_text(json.dumps(self.catalog))
        output = self.root / "cli-proposal"
        script = str(ROOT / "scripts/runner-image-identity.py")
        result = subprocess.run([sys.executable, script, "--root", str(self.root), "bind", "--image-id", "public-cpu", "--cohort", str(self.cohort_path), "--anchor", str(self.anchor_path), "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(catalog_path.read_text()), self.catalog)
        for args in (["validate"], ["lookup", "rust-clippy", "--field", "provenance"], ["seed-key", "--recipe-hash", "a" * 64]):
            result = subprocess.run([sys.executable, script, "--root", str(output), *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, script, "--root", str(self.root), "--catalog", str(output / "ci/runner-images.json"), "validate"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_runtime_row_cannot_require_missing_qualified_architecture(self):
        output = self.root / "platform-proposal"
        E.bind(self.catalog, "public-cpu", self.cohort_path, self.anchor_path, output)
        path = output / "ci/runner-images.json"
        catalog = json.loads(path.read_text())
        catalog["runtime_rows"]["linux-cpu"]["architecture"] = "arm64"
        path.write_text(json.dumps(catalog))
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/runner-image-identity.py"),
             "--root", str(output), "validate"], capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lacks runtime row platform", result.stderr)

    def test_bind_rejects_missing_runtime_platform_before_output(self):
        self.catalog["runtime_rows"]["linux-cpu"]["architecture"] = "arm64"
        output = self.root / "missing-platform-proposal"
        with self.assertRaisesRegex(ValueError, "lacks runtime row platform"):
            E.bind(self.catalog, "public-cpu", self.cohort_path, self.anchor_path, output)
        self.assertFalse(output.exists())

    def test_bound_compiler_seed_platform_is_independent_of_runtime_rows(self):
        catalog = copy.deepcopy(self.catalog)
        catalog["runtime_rows"] = {}
        with self.assertRaisesRegex(ValueError, "compiler seed platform"):
            E.require_platforms(catalog, {"public-cpu": {"arm64"}})

    def test_every_cli_command_rejects_missing_nonnull_evidence(self):
        self.image.update(self.anchor)
        (self.root / "ci").mkdir()
        (self.root / "ci/runner-images.json").write_text(json.dumps(self.catalog))
        for args in (["validate"], ["lookup", "rust-clippy"], ["seed-key", "--recipe-hash", 'a'*64], ["diagnose"], ["check"]):
            result = subprocess.run([sys.executable, str(ROOT / "scripts/runner-image-identity.py"), "--root", str(self.root), *args], capture_output=True, text=True, env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})
            self.assertNotEqual(result.returncode, 0, args)
            self.assertEqual(result.stdout, "")

if __name__ == "__main__":
    unittest.main()
