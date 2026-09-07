from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate-skippy-family-patch.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("skippy_family_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GenerateSkippyFamilyPatchTests(unittest.TestCase):
    def test_capture_uses_exact_utf8_bytes_without_newline_translation(self) -> None:
        generator = load_generator()
        with tempfile.TemporaryDirectory() as temporary:
            output = generator.run(
                [
                    "python3",
                    "-c",
                    "import sys; sys.stdout.buffer.write('before—after\\r\\n'.encode('utf-8'))",
                ],
                cwd=Path(temporary),
                capture=True,
            )

        self.assertEqual(output, "before—after\r\n")

    def test_write_utf8_emits_exact_lf_bytes(self) -> None:
        generator = load_generator()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "generated.patch"
            generator.write_utf8(path, "before—after\n")
            self.assertEqual(path.read_bytes(), "before—after\n".encode("utf-8"))

    def test_first_pass_rejects_pretransformed_builder(self) -> None:
        generator = load_generator()
        report = {
            "builders": [
                {
                    "file": "src/models/muse-glimmer.cpp",
                    "verdict": "already_transformed",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "first rewriter pass received pre-transformed"
            ):
                generator.validate_report(path, idempotence=False)

    def test_second_pass_accepts_pretransformed_builder(self) -> None:
        generator = load_generator()
        report = {
            "builders": [
                {
                    "file": "src/models/muse-glimmer.cpp",
                    "verdict": "already_transformed",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                generator.validate_report(path, idempotence=True), report
            )

    def test_first_pass_rejects_unsupported_builder(self) -> None:
        generator = load_generator()
        report = {
            "builders": [
                {
                    "file": "src/models/qwen4exp.cpp",
                    "verdict": "unsupported_shape",
                    "unsupported_reason": "unproven hyperconnection prelude",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "did not transform every decoder builder"
            ):
                generator.validate_report(path, idempotence=False)


if __name__ == "__main__":
    unittest.main()
