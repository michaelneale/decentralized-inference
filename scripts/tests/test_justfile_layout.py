from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
IMPORTS: Final = (
    "just/build.just",
    "just/release-build.just",
    "just/skippy.just",
    "just/mesh.just",
    "just/release-bundle.just",
    "just/website-ui.just",
    "just/ci.just",
    "just/mesh-client.just",
    "just/utilities.just",
)
RECIPES_BY_FILE: Final = {
    "just/build.just": {
        "build", "build-dev", "build-linux", "build-mac", "build-runtime",
        "qa-logging-console-e2e", "with-lld",
    },
    "just/release-build.just": {
        "llama-build", "llama-prepare", "llama-prepare-latest", "release",
        "release-build", "release-build-aarch64", "release-build-aarch64-cuda",
        "release-build-cuda", "release-build-cuda-windows", "release-build-rocm",
        "release-build-rocm-windows", "release-build-vulkan",
        "release-build-vulkan-windows", "release-build-windows",
        "release-host-build", "release-host-build-windows", "release-runtime-build",
    },
    "just/skippy.just": {
        "bench-corpus", "competitive-benchmark-build", "family-certify",
        "metrics-server", "metrics-server-build",
        "skippy-native-full-replay", "skippy-native-tests", "skippy-openai-smoke",
        "skippy-quantize-build",
        "skippy-quantize-release-build", "skippy-quantize-standalone-build",
        "skippy-quantize-standalone-release-build", "skippy-wan-lab-build-bins",
        "spec-bench",
    },
    "just/mesh.just": {"bundle", "download-model", "mesh-join", "mesh-worker"},
    "just/release-bundle.just": {
        "check-env-mutation-contract", "check-release", "release-attestation",
        "release-bundle", "release-bundle-aarch64", "release-bundle-aarch64-cuda",
        "release-bundle-cuda", "release-bundle-cuda-windows", "release-bundle-rocm",
        "release-bundle-rocm-windows", "release-bundle-vulkan",
        "release-bundle-vulkan-windows", "release-bundle-windows",
    },
    "just/website-ui.just": {
        "cli-inventory-check", "crate-docs", "ui-dev", "ui-dev-public", "ui-test", "website-build",
        "website-clean", "website-dev",
    },
    "just/ci.just": {
        "ci-crate-lists", "ci-sccache-seed-build", "ci-shellcheck", "ci-validate",
        "no-console-print", "publish-crates", "test-all",
    },
    "just/mesh-client.just": {"auto", "mesh-client"},
    "just/utilities.just": {
        "cache-cargo-clean", "cache-cargo-metadata", "cache-prune",
        "cache-prune-dry-run", "cache-status", "clean", "diff",
        "docker-build-client", "docker-run-client", "llama-summary",
        "llama-update-pin", "stop", "test", "ui-clean",
    },
}
RECIPE_HEADER: Final = re.compile(r"^([A-Za-z_][\w-]*)(?:\s+[^:]*)?:(?!=)")


class JustfileLayoutTests(unittest.TestCase):
    def test_root_keeps_prelude_default_and_ordered_flat_imports(self) -> None:
        source = (ROOT / "Justfile").read_text(encoding="utf-8")
        imports = re.findall(r"(?m)^import '([^']+)'$", source)

        self.assertEqual(tuple(imports), IMPORTS)
        self.assertIn("# Distributed LLM Inference — build & run tasks", source)
        self.assertIn("default: build", source)
        self.assertNotRegex(source, r"(?m)^\s*(?:mod\b|import\?)")

    def test_each_recipe_stays_in_its_owning_import(self) -> None:
        for relative_path, expected in RECIPES_BY_FILE.items():
            with self.subTest(relative_path=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                actual = {
                    match.group(1)
                    for line in source.splitlines()
                    if (match := RECIPE_HEADER.match(line)) is not None
                }
                self.assertEqual(actual, expected)

    def test_only_with_lld_is_private_and_imports_create_no_modules(self) -> None:
        private_sources = {
            relative_path: (ROOT / relative_path).read_text(encoding="utf-8").count("[private]\n")
            for relative_path in IMPORTS
        }
        dump = json.loads(
            subprocess.check_output(
                ["just", "--dump", "--dump-format", "json"], cwd=ROOT, text=True
            )
        )

        self.assertEqual(private_sources["just/build.just"], 2)
        self.assertTrue(all(count == 0 for path, count in private_sources.items() if path != "just/build.just"))
        self.assertNotIn("with-lld", subprocess.check_output(["just", "--summary"], cwd=ROOT, text=True).split())
        self.assertEqual(dump["modules"], {})
        self.assertEqual(dump["first"], "default")


if __name__ == "__main__":
    unittest.main()
