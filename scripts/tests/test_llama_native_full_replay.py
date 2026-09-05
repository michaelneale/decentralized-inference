from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "build-llama.sh"

PRIVATE_TARGETS = {
    "skippy-graph-build-inputs",
    "skippy-hardware-application-probe",
    "skippy-model-fixture-generator",
    "skippy-model-loader-accounting",
    "skippy-noalloc-graph-planning",
    "skippy-renamed-multishard-planning",
    "skippy-stage-slice-plan",
}
LEGACY_TARGETS = {
    "test-skippy-activation-layout",
    "test-skippy-kv-cells-contiguous",
    "test-skippy-kv-page-export",
    "test-skippy-model-loader-accounting",
    "test-skippy-recurrent-state-roundtrip",
    "test-skippy-verify-checkpoint-retirement",
}


class LlamaNativeFullReplayTests(unittest.TestCase):
    def run_build(
        self, *, full_replay: bool, repeat_cached: bool = False
    ) -> list[dict[str, object]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            llama = root / "llama.cpp"
            build = root / "build"
            bin_dir = root / "bin"
            trace = root / "trace.jsonl"
            (llama / ".git").mkdir(parents=True)
            (llama / ".mesh-llm-patched-sha").write_text(
                "0123456789abcdef\n", encoding="utf-8"
            )
            bin_dir.mkdir()
            (bin_dir / "cmake").write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import json
                    import os
                    from pathlib import Path
                    import sys

                    with Path(os.environ["TRACE_FILE"]).open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"tool": "cmake", "args": sys.argv[1:]}) + "\\n")
                    if sys.argv[1:2] == ["--build"]:
                        build = Path(sys.argv[2])
                        for relative in (
                            "src/libllama.a",
                            "common/libllama-common.a",
                            "common/libllama-common-base.a",
                            "ggml/src/libggml.a",
                            "ggml/src/libggml-base.a",
                            "ggml/src/libggml-cpu.a",
                            "tools/mtmd/libmtmd.a",
                            "vendor/hash/libvendor-hash.a",
                        ):
                            path = build / relative
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.touch()
                        generator = build / "bin" / "skippy-model-fixture-generator"
                        generator.parent.mkdir(parents=True, exist_ok=True)
                        generator.write_text(
                            "#!/usr/bin/env python3\\n"
                            "import json, os, pathlib, sys\\n"
                            "with pathlib.Path(os.environ['TRACE_FILE']).open('a', encoding='utf-8') as handle:\\n"
                            "    handle.write(json.dumps({'tool': 'fixture', 'args': sys.argv[1:]}) + '\\\\n')\\n",
                            encoding="utf-8",
                        )
                        generator.chmod(0o755)
                    """
                ),
                encoding="utf-8",
            )
            (bin_dir / "git").write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    exit 0
                    """
                ),
                encoding="utf-8",
            )
            (bin_dir / "ctest").write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import json
                    import os
                    from pathlib import Path
                    import sys

                    with Path(os.environ["TRACE_FILE"]).open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"tool": "ctest", "args": sys.argv[1:]}) + "\\n")
                    """
                ),
                encoding="utf-8",
            )
            for tool in (bin_dir / "cmake", bin_dir / "ctest", bin_dir / "git"):
                tool.chmod(0o755)

            environment = os.environ.copy()
            environment.update(
                {
                    "LLAMA_WORKDIR": str(llama),
                    "LLAMA_STAGE_BUILD_DIR": str(build),
                    "LLAMA_STAGE_BACKEND": "cpu",
                    "LLAMA_STAGE_LINK_MODE": "static",
                    "LLAMA_STAGE_FORCE_BUILD": "1",
                    "LLAMA_STAGE_USE_SCCACHE": "0",
                    "TRACE_FILE": str(trace),
                    "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
                }
            )
            if full_replay:
                environment["LLAMA_STAGE_FULL_REPLAY"] = "ON"

            for run_number in range(2 if repeat_cached else 1):
                if run_number:
                    environment.pop("LLAMA_STAGE_FORCE_BUILD", None)
                subprocess.run(
                    [shutil.which("bash") or "bash", str(BUILD_SCRIPT)],
                    cwd=ROOT,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            return [json.loads(line) for line in trace.read_text().splitlines()]

    def test_default_build_keeps_standard_and_private_tests_disabled(self) -> None:
        trace = self.run_build(full_replay=False)
        configure = next(call for call in trace if call["args"][0] != "--build")
        build = next(call for call in trace if call["args"][0] == "--build")

        self.assertIn("-DLLAMA_BUILD_TESTS=OFF", configure["args"])
        self.assertIn("-DLLAMA_STAGE_BUILD_TESTS=OFF", configure["args"])
        self.assertTrue(PRIVATE_TARGETS.isdisjoint(build["args"]))
        self.assertTrue(LEGACY_TARGETS.isdisjoint(build["args"]))
        self.assertFalse(any(call["tool"] == "ctest" for call in trace))

    def test_full_replay_builds_and_runs_only_skippy_gates(self) -> None:
        trace = self.run_build(full_replay=True)
        configure = next(call for call in trace if call["args"][0] != "--build")
        build = next(call for call in trace if call["args"][0] == "--build")
        ctest = next(call for call in trace if call["tool"] == "ctest")
        fixtures = [call["args"] for call in trace if call["tool"] == "fixture"]

        self.assertIn("-DLLAMA_BUILD_TESTS=ON", configure["args"])
        self.assertIn("-DLLAMA_STAGE_BUILD_TESTS=ON", configure["args"])
        self.assertTrue(PRIVATE_TARGETS.issubset(build["args"]))
        self.assertTrue(LEGACY_TARGETS.issubset(build["args"]))
        self.assertNotIn("test-llama-archs", build["args"])
        self.assertEqual(
            [fixture[:4] for fixture in fixtures],
            [["--arch", "gemma", "--seed", "1"], ["--arch", "qwen2moe", "--seed", "1"]],
        )
        self.assertEqual(
            ctest["args"][ctest["args"].index("--fixture-exclude-any") + 1],
            "^generate-models$",
        )
        self.assertEqual(ctest["args"][-2:], ["-R", "^(skippy_|test-skippy-)"])
        self.assertIn("--timeout", ctest["args"])
        self.assertEqual(ctest["args"][ctest["args"].index("--timeout") + 1], "900")

    def test_cached_full_replay_rebuilds_and_runs_skippy_gates(self) -> None:
        trace = self.run_build(full_replay=True, repeat_cached=True)

        build_calls = [
            call for call in trace if call["tool"] == "cmake" and call["args"][0] == "--build"
        ]
        ctest_calls = [call for call in trace if call["tool"] == "ctest"]

        self.assertEqual(len(build_calls), 2)
        self.assertEqual(len(ctest_calls), 2)

    def test_cached_standard_build_keeps_cache_shortcut(self) -> None:
        trace = self.run_build(full_replay=False, repeat_cached=True)

        build_calls = [
            call for call in trace if call["tool"] == "cmake" and call["args"][0] == "--build"
        ]

        self.assertEqual(len(build_calls), 1)
        self.assertFalse(any(call["tool"] == "ctest" for call in trace))

    def test_just_recipe_uses_a_dedicated_forced_replay_build(self) -> None:
        recipe = (ROOT / "just" / "skippy.just").read_text(encoding="utf-8")
        replay = recipe.split(
            'skippy-native-full-replay backend="cpu":', maxsplit=1
        )[1].split("\n\n", maxsplit=1)[0]

        self.assertIn("scripts/prepare-llama.sh pinned", replay)
        self.assertIn("build-stage-full-replay-static-{{ backend }}", replay)
        self.assertIn("LLAMA_STAGE_FULL_REPLAY=ON", replay)
        self.assertIn("LLAMA_STAGE_FORCE_BUILD=1", replay)


if __name__ == "__main__":
    unittest.main()
