from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PLANNER = ROOT / "scripts" / "plan-family-battery.py"
MANIFEST = ROOT / "ci" / "llama-canary" / "family-certified.json"


class FamilyBatteryPlannerTests(unittest.TestCase):
    @staticmethod
    def _write_gguf(
        path: Path,
        block_count: int | None,
        embedding_length: int | None = 1024,
        architecture: str = "fixture",
        hyper_connection_count: int | None = None,
        embedding_length_out: int | None = None,
    ) -> int:
        def gguf_string(value: str) -> bytes:
            encoded = value.encode("utf-8")
            return struct.pack("<Q", len(encoded)) + encoded

        metadata = [("general.architecture", 8, architecture)]
        if block_count is not None:
            metadata.append((f"{architecture}.block_count", 4, block_count))
        if embedding_length is not None:
            metadata.append((f"{architecture}.embedding_length", 4, embedding_length))
        if hyper_connection_count is not None:
            metadata.append(
                (f"{architecture}.hyper_connection.count", 4, hyper_connection_count)
            )
        if embedding_length_out is not None:
            metadata.append(
                (f"{architecture}.embedding_length_out", 4, embedding_length_out)
            )

        payload = bytearray(b"GGUF")
        payload.extend(struct.pack("<IQQ", 3, 0, len(metadata)))
        for key, kind, value in metadata:
            payload.extend(gguf_string(key))
            payload.extend(struct.pack("<I", kind))
            if kind == 8:
                assert isinstance(value, str)
                payload.extend(gguf_string(value))
            else:
                assert isinstance(value, int)
                payload.extend(struct.pack("<I", value))
        path.write_bytes(payload)
        return len(payload)

    @classmethod
    def _materialize_cached_artifact(
        cls,
        root: Path,
        artifact: dict[str, object],
        block_counts: list[int | None],
        embedding_length: int = 1024,
        architecture: str = "fixture",
        hyper_connection_count: int | None = None,
        embedding_length_out: int | None = None,
    ) -> list[Path]:
        files = artifact["files"]
        assert isinstance(files, list)
        assert len(files) == len(block_counts)
        repo_dir = "models--" + str(artifact["repo"]).replace("/", "--")
        repo_root = root / "cache" / "hub" / repo_dir
        snapshot = repo_root / "snapshots" / str(artifact["revision"])
        integrity: dict[str, dict[str, object]] = {}
        paths = []
        for index, (relative, block_count) in enumerate(
            zip(files, block_counts, strict=True)
        ):
            blob_id = f"{index + 1:064x}"
            blob = repo_root / "blobs" / blob_id
            blob.parent.mkdir(parents=True, exist_ok=True)
            shard_width = embedding_length if block_count is not None else None
            size = cls._write_gguf(
                blob,
                block_count,
                shard_width,
                architecture,
                hyper_connection_count,
                embedding_length_out,
            )
            cached = snapshot / str(relative)
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.symlink_to(blob)
            integrity[str(relative)] = {"size_bytes": size, "blob_id": blob_id}
            paths.append(cached)
        artifact["file_integrity"] = integrity
        return paths

    def _run(
        self, manifest: Path = MANIFEST, *args: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(PLANNER), "--manifest", str(manifest), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_checked_in_policy_resolves_all_certified_models(self) -> None:
        result = self._run()
        self.assertEqual(0, result.returncode, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(77, plan["selected_family_count"])
        self.assertEqual(
            ["single-step", "chain", "state-handoff"],
            plan["required_certification_lanes"],
        )
        glm47 = next(
            model
            for model in plan["selected_models"]
            if model["family"] == "glm47-flash"
        )
        self.assertEqual(47, glm47["execution"]["layer_end"])
        self.assertEqual(0, glm47["execution"]["mtp_layers"])
        by_family = {model["family"]: model for model in plan["selected_models"]}
        expected_ranges = {
            "deepseek2": 27,
            "qwen3-moe": 48,
            "kimi-linear": 27,
            "mamba2": 64,
            "laguna": 40,
        }
        for family, layer_end in expected_ranges.items():
            with self.subTest(family=family):
                self.assertEqual(layer_end, by_family[family]["execution"]["layer_end"])
        self.assertEqual(4096, by_family["qwen3-vl"]["execution"]["activation_width"])
        self.assertEqual(600, by_family["qwen3-vl"]["resources"]["startup_timeout_secs"])
        qwen4exp = by_family["qwen4exp"]
        self.assertEqual(10240, qwen4exp["execution"]["activation_width"])
        self.assertEqual(4, qwen4exp["execution"]["boundary_sweep_period"])
        self.assertEqual(3, len(qwen4exp["artifact"]["files"]))
        self.assertEqual(16384, by_family["deepseek4"]["execution"]["activation_width"])

    def test_nightly_cadence_selects_cache_mechanism_sentinels(self) -> None:
        result = self._run(MANIFEST, "--cadence", "nightly")
        self.assertEqual(0, result.returncode, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual("nightly", plan["selected_cadence"])
        self.assertEqual(
            ["qwen3-dense", "falcon-h1", "qwen3-next", "mamba"],
            [model["family"] for model in plan["selected_models"]],
        )

    def test_cadence_and_explicit_family_selection_intersect(self) -> None:
        result = self._run(
            MANIFEST,
            "--cadence",
            "nightly",
            "--families",
            "mamba,qwen3-dense",
        )
        self.assertEqual(0, result.returncode, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(
            ["qwen3-dense", "mamba"],
            [model["family"] for model in plan["selected_models"]],
        )

    def test_empty_cadence_selection_fails_closed(self) -> None:
        result = self._run(MANIFEST, "--cadence", "rotating")
        self.assertEqual(2, result.returncode)
        self.assertIn("family selection produced no models", result.stderr)

    def test_mmproj_artifacts_resolve_and_cover_the_vision_families(self) -> None:
        result = self._run()
        self.assertEqual(0, result.returncode, result.stderr)
        plan = json.loads(result.stdout)
        with_mmproj = {
            model["family"]: model["mmproj_artifact"]
            for model in plan["selected_models"]
            if model.get("mmproj_artifact") is not None
        }
        self.assertEqual(
            {"gemma4", "lfm2-vl", "muse-glimmer", "qwen2-vl", "qwen3-vl"},
            set(with_mmproj),
        )
        for family, mmproj in with_mmproj.items():
            with self.subTest(family=family):
                self.assertEqual(1, len(mmproj["files"]))
                self.assertTrue(mmproj["file_integrity"])
                self.assertEqual(
                    set(mmproj["file_integrity"]), set(mmproj["files"])
                )
                self.assertRegex(
                    mmproj["files"][0], r"^mmproj"
                )

    def test_certified_model_requires_an_explicit_activation_width(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        del manifest["models"][0]["execution"]["activation_width"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self._run(path)
        self.assertEqual(2, result.returncode)
        self.assertIn("activation_width must be an integer", result.stderr)

    def test_certified_profile_cannot_drop_a_core_lane(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["policy"]["profiles"]["full"]["required_lanes"].remove(
            "state-handoff"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self._run(path)
        self.assertEqual(2, result.returncode)
        self.assertIn("must require exactly the three core lanes", result.stderr)

    def test_certified_profile_cannot_add_or_reorder_core_lanes(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        lanes = manifest["policy"]["profiles"]["package-oracle"]["required_lanes"]
        lanes.reverse()
        lanes.append("graph-parse")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self._run(path)
        self.assertEqual(2, result.returncode)
        self.assertIn("must require exactly the three core lanes", result.stderr)

    def test_duplicate_family_is_rejected(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["models"].append(copy.deepcopy(manifest["models"][0]))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self._run(path)
        self.assertEqual(2, result.returncode)
        self.assertIn("duplicate family", result.stderr)

    def test_cache_gate_requires_every_exact_revision_file(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            artifact = model["artifact"]
            cached = self._materialize_cached_artifact(root, artifact, [3])[0]
            manifest.write_text(json.dumps(source), encoding="utf-8")
            present = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
            self.assertEqual(0, present.returncode, present.stderr)
            cached.unlink()
            missing = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
        self.assertEqual(2, missing.returncode)
        self.assertIn("immutable family cache is incomplete", missing.stderr)
        self.assertIn(model["family"], missing.stderr)

    def test_cache_gate_rejects_runtime_range_drift_before_build(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            artifact = model["artifact"]
            self._materialize_cached_artifact(root, artifact, [4])
            manifest.write_text(json.dumps(source), encoding="utf-8")
            result = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
        self.assertEqual(2, result.returncode)
        self.assertIn("plans 3 runtime layers", result.stderr)
        self.assertIn("declares 4", result.stderr)

    def test_cache_gate_rejects_activation_width_drift_before_build(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            artifact = model["artifact"]
            self._materialize_cached_artifact(root, artifact, [3], 2048)
            manifest.write_text(json.dumps(source), encoding="utf-8")
            result = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
        self.assertEqual(2, result.returncode)
        self.assertIn("plans activation width 1024", result.stderr)
        self.assertIn("declares 2048", result.stderr)

    def test_cache_gate_derives_qwen4exp_hyper_connected_activation_width(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        model["execution"]["activation_width"] = 4096
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            self._materialize_cached_artifact(
                root,
                model["artifact"],
                [3],
                embedding_length=1024,
                architecture="qwen4exp",
                hyper_connection_count=4,
                embedding_length_out=4096,
            )
            manifest.write_text(json.dumps(source), encoding="utf-8")
            result = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_cache_gate_rejects_qwen4exp_output_width_drift(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        model["execution"]["activation_width"] = 4096
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            self._materialize_cached_artifact(
                root,
                model["artifact"],
                [3],
                embedding_length=1024,
                architecture="qwen4exp",
                hyper_connection_count=4,
                embedding_length_out=1024,
            )
            manifest.write_text(json.dumps(source), encoding="utf-8")
            result = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
        self.assertEqual(2, result.returncode)
        self.assertIn("embedding_length_out disagrees", result.stderr)

    def test_cache_gate_derives_dflash_hyper_connected_activation_width(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        model["execution"]["activation_width"] = 4096
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            self._materialize_cached_artifact(
                root,
                model["artifact"],
                [3],
                embedding_length=1024,
                architecture="dflash",
                hyper_connection_count=4,
            )
            manifest.write_text(json.dumps(source), encoding="utf-8")
            result = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_cache_gate_requires_qwen4exp_hyper_connection_count(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        model["execution"]["activation_width"] = 4096
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            self._materialize_cached_artifact(
                root,
                model["artifact"],
                [3],
                embedding_length=1024,
                architecture="qwen4exp",
            )
            manifest.write_text(json.dumps(source), encoding="utf-8")
            result = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
        self.assertEqual(2, result.returncode)
        self.assertIn("hyper_connection.count", result.stderr)

    def test_cache_gate_rejects_qwen4exp_activation_width_overflow(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        model["execution"]["activation_width"] = 0x80000000
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            self._materialize_cached_artifact(
                root,
                model["artifact"],
                [3],
                embedding_length=0x40000000,
                architecture="qwen4exp",
                hyper_connection_count=2,
            )
            manifest.write_text(json.dumps(source), encoding="utf-8")
            result = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
        self.assertEqual(2, result.returncode)
        self.assertIn("activation width exceeds i32", result.stderr)

    def test_cache_gate_checks_secondary_shard_metadata_and_blob_identity(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        model["artifact"]["files"] = ["model-00001.gguf", "model-00002.gguf"]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            cached = self._materialize_cached_artifact(
                root, model["artifact"], [3, 4]
            )
            manifest.write_text(json.dumps(source), encoding="utf-8")
            metadata_drift = self._run(
                manifest, "--check-cache", "--cache-root", str(root / "cache")
            )
            self.assertEqual(2, metadata_drift.returncode)
            self.assertIn("model-00002.gguf declares 4", metadata_drift.stderr)

            self._write_gguf(cached[1].resolve(), 3)
            model["artifact"]["file_integrity"]["model-00002.gguf"][
                "size_bytes"
            ] = cached[1].stat().st_size
            model["artifact"]["file_integrity"]["model-00002.gguf"][
                "blob_id"
            ] = "f" * 64
            manifest.write_text(json.dumps(source), encoding="utf-8")
            blob_drift = self._run(
                manifest, "--check-cache", "--cache-root", str(root / "cache")
            )
        self.assertEqual(2, blob_drift.returncode)
        self.assertIn("expected blob", blob_drift.stderr)

    def test_cache_gate_accepts_payload_only_secondary_shard(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        model["artifact"]["files"] = ["model-00001.gguf", "model-00002.gguf"]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            self._materialize_cached_artifact(root, model["artifact"], [3, None])
            manifest.write_text(json.dumps(source), encoding="utf-8")
            result = self._run(
                manifest, "--check-cache", "--cache-root", str(root / "cache")
            )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_cache_gate_requires_dimensions_in_at_least_one_shard(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["artifact"]["files"] = ["model-00001.gguf", "model-00002.gguf"]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            self._materialize_cached_artifact(root, model["artifact"], [None, None])
            manifest.write_text(json.dumps(source), encoding="utf-8")
            result = self._run(
                manifest, "--check-cache", "--cache-root", str(root / "cache")
            )
        self.assertEqual(2, result.returncode)
        self.assertIn("has no GGUF shard", result.stderr)

    def test_shards_are_deterministic_and_preserve_every_family_once(self) -> None:
        first = self._run(MANIFEST, "--shard-count", "4")
        second = self._run(MANIFEST, "--shard-count", "4")
        self.assertEqual(0, first.returncode, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        plan = json.loads(first.stdout)
        families = [
            family for shard in plan["shards"] for family in shard["families"]
        ]
        self.assertEqual(77, len(families))
        self.assertEqual(77, len(set(families)))
        self.assertEqual(4, len(plan["github_matrix"]["include"]))


if __name__ == "__main__":
    unittest.main()
