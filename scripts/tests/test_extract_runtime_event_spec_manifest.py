from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/extract-runtime-event-spec-manifest.py"


def load_extractor():
    spec = importlib.util.spec_from_file_location("runtime_event_spec_extractor", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RuntimeEventSpecManifestTests(unittest.TestCase):
    def test_extracts_all_required_bullets_with_continuations(self) -> None:
        extractor = load_extractor()

        bullets = extractor.extract_bullets(
            (ROOT / ".omo/specs/event-system.md").read_text(encoding="utf-8")
        )

        self.assertEqual(len(bullets), 137)
        self.assertEqual({bullet.section for bullet in bullets}, {f"8.{i}" for i in range(1, 16)})
        connection = next(bullet for bullet in bullets if bullet.section == "8.6" and bullet.ordinal == 13)
        self.assertEqual(
            connection.text,
            "upstream or downstream stage connection established/lost/recovered.",
        )

    def test_check_rejects_stale_output(self) -> None:
        extractor = load_extractor()

        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "event-system.md"
            spec.write_text(
                "\n".join(
                    [
                        *(
                            f"### 8.{index} Family {index}\n"
                            "Required events:\n"
                            f"- event_{index}"
                            for index in range(1, 16)
                        ),
                        "## 9. Later sections",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = Path(directory) / "manifest.json"
            output.write_text("{}\n", encoding="utf-8")
            with patch.object(
                sys,
                "argv",
                [
                    str(SCRIPT),
                    "--spec",
                    str(spec),
                    "--output",
                    str(output),
                    "--check",
                ],
            ):
                self.assertEqual(extractor.main(), 1)


if __name__ == "__main__":
    unittest.main()
