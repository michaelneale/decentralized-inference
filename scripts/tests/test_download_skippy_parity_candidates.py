from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "download-skippy-parity-candidates.sh"


class DownloadSkippyParityCandidatesTests(unittest.TestCase):
    def test_direct_candidate_revision_and_integrity_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            model = snapshot / "model.gguf"
            projector = snapshot / "projector.gguf"
            model.write_bytes(b"model-bytes")
            projector.write_bytes(b"projector-bytes")

            def record(path: Path) -> dict[str, object]:
                return {
                    "blob_id": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }

            manifest = root / "candidates.json"
            manifest.write_text(
                json.dumps(
                    {
                        "support_priority": {
                            "p0": {"families": ["vision-family"], "llama_models": []}
                        },
                        "candidates": [
                            {
                                "llama_model": "vision",
                                "family": "vision-family",
                                "status": "candidate_multimodal",
                                "repo": "owner/model",
                                "revision": "0123456789abcdef0123456789abcdef01234567",
                                "include": [model.name, projector.name],
                                "file_integrity": {
                                    model.name: record(model),
                                    projector.name: record(projector),
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            model_manifest = root / "model-manifest.json"
            model_manifest.write_text('{"artifacts": []}', encoding="utf-8")
            bindir = root / "bin"
            bindir.mkdir()
            hf = bindir / "hf"
            hf.write_text(
                "#!/bin/sh\n"
                "printf 'args=%s\\n' \"$*\"\n"
                f"printf 'path=%s\\n' '{snapshot}'\n",
                encoding="utf-8",
            )
            hf.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bindir}:{env['PATH']}",
                    "SKIPPY_PARITY_MANIFEST": str(manifest),
                    "SKIPPY_PARITY_MODEL_MANIFEST": str(model_manifest),
                }
            )
            result = subprocess.run(
                [str(SCRIPT), "--priority", "p0"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn(
                "args=download owner/model model.gguf projector.gguf --revision "
                "0123456789abcdef0123456789abcdef01234567",
                result.stdout,
            )
            self.assertIn(f"verified {model} ({model.stat().st_size} bytes)", result.stdout)
            self.assertIn(
                f"verified {projector} ({projector.stat().st_size} bytes)",
                result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
