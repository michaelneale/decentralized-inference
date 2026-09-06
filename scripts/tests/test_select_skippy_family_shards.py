from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class FamilyShardSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = load(
            "select_skippy_family_shards",
            ROOT / "scripts" / "select-skippy-family-shards.py",
        )

    @staticmethod
    def series(*shards, version="0.3.0"):
        return {
            "schema_version": 1,
            "generator_version": version,
            "shards": list(shards),
        }

    @staticmethod
    def shard(digest, families, sources):
        return {"sha256": digest, "families": families, "sources": sources}

    def test_mapped_change_selects_family_and_cross_class_sentinels(self) -> None:
        base = self.series(self.shard("a", ["gemma2"], ["src/models/gemma2.cpp"]))
        current = self.series(self.shard("b", ["gemma2"], ["src/models/gemma2.cpp"]))
        result = self.selector.select(base, current, True)
        self.assertEqual(result["mode"], "targeted")
        self.assertEqual(
            set(result["families"]),
            {"gemma2", "qwen3-dense", "qwen3-moe", "mamba", "lfm2-vl"},
        )

    def test_unmapped_change_forces_full_battery(self) -> None:
        base = self.series(self.shard("a", [], ["src/models/new.cpp"]))
        current = self.series(self.shard("b", [], ["src/models/new.cpp"]))
        self.assertEqual(self.selector.select(base, current, True)["mode"], "full")

    def test_generator_change_forces_full_battery(self) -> None:
        shard = self.shard("a", ["gemma2"], ["src/models/gemma2.cpp"])
        result = self.selector.select(self.series(shard), self.series(shard, version="0.4.0"), True)
        self.assertEqual(result["mode"], "full")


class FamilyShardGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = load(
            "generate_skippy_family_patch",
            ROOT / "scripts" / "generate-skippy-family-patch.py",
        )

    def test_split_model_diff_preserves_complete_sections(self) -> None:
        diff = (
            "diff --git a/src/models/a.cpp b/src/models/a.cpp\n--- a/src/models/a.cpp\n+++ b/src/models/a.cpp\n@@ -1 +1 @@\n-a\n+b\n"
            "diff --git a/src/models/b.cpp b/src/models/b.cpp\n--- a/src/models/b.cpp\n+++ b/src/models/b.cpp\n@@ -1 +1 @@\n-c\n+d\n"
        )
        sections = self.generator.split_model_diff(diff)
        self.assertEqual(set(sections), {"src/models/a.cpp", "src/models/b.cpp"})
        self.assertEqual("".join(sections.values()), diff)


if __name__ == "__main__":
    unittest.main()
