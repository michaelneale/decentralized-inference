import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ui_distribution", ROOT / "scripts/ui-distribution.py")
UI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UI)
SOURCE = "a" * 40
TAG = "v0.74.0-rc.1"


class UiDistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dist = Path(self.temp.name)
        (self.dist / "assets").mkdir()
        (self.dist / "assets/app.js").write_text("console.log('release');")
        (self.dist / "index.html").write_text('<script type="module" src="/assets/app.js"></script>')

    def test_stamp_and_verify_exact_release(self):
        UI.stamp(self.dist, SOURCE, TAG)
        UI.verify(self.dist, SOURCE, TAG)
        manifest = json.loads((self.dist / UI.MANIFEST).read_text())
        self.assertEqual(set(manifest["files"]), {"index.html", "assets/app.js"})

    def test_wrong_source_and_version_are_rejected(self):
        UI.stamp(self.dist, SOURCE, TAG)
        for source, tag in (("b" * 40, TAG), (SOURCE, "v0.74.0")):
            with self.subTest(source=source, tag=tag), self.assertRaisesRegex(ValueError, "identity"):
                UI.verify(self.dist, source, tag)

    def test_changed_and_extra_files_are_rejected(self):
        UI.stamp(self.dist, SOURCE, TAG)
        (self.dist / "assets/app.js").write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksums"):
            UI.verify(self.dist, SOURCE, TAG)
        UI.stamp(self.dist, SOURCE, TAG)
        (self.dist / "assets/unexpected.js").write_text("extra")
        with self.assertRaisesRegex(ValueError, "checksums"):
            UI.verify(self.dist, SOURCE, TAG)

    def test_missing_module_and_windows_placeholder_are_rejected(self):
        (self.dist / "assets/app.js").unlink()
        with self.assertRaisesRegex(ValueError, "JavaScript"):
            UI.stamp(self.dist, SOURCE, TAG)
        (self.dist / "index.html").write_text("<html></html>")
        with self.assertRaisesRegex(ValueError, "JavaScript"):
            UI.stamp(self.dist, SOURCE, TAG)

    def test_missing_manifest_and_entry_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "release manifest"):
            UI.verify(self.dist, SOURCE, TAG)
        (self.dist / "index.html").unlink()
        with self.assertRaisesRegex(ValueError, "index.html"):
            UI.stamp(self.dist, SOURCE, TAG)

    def test_symlinks_are_rejected(self):
        (self.dist / "assets/link.js").symlink_to(self.dist / "assets/app.js")
        with self.assertRaisesRegex(ValueError, "symbolic"):
            UI.stamp(self.dist, SOURCE, TAG)

    def test_identity_shape_is_checked(self):
        for source, tag in (("main", TAG), (SOURCE, "main")):
            with self.subTest(source=source, tag=tag), self.assertRaises(ValueError):
                UI.stamp(self.dist, source, tag)

    def test_restore_action_verifies_with_python3_or_python(self):
        action = yaml.safe_load((ROOT / ".github/actions/restore-release-ui/action.yml").read_text())
        verify_script = action["runs"]["steps"][1]["run"]
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        UI.stamp(self.dist, SOURCE, TAG)
        for interpreters, selected in ((("python3", "python"), "python3"), (("python",), "python")):
            with self.subTest(interpreters=interpreters), tempfile.TemporaryDirectory() as temporary:
                workspace = Path(temporary)
                binaries = workspace / "bin"
                binaries.mkdir()
                marker = workspace / "interpreter"
                for interpreter in interpreters:
                    executable = binaries / interpreter
                    executable.write_text(
                        f"#!/bin/sh\nprintf '%s\\n' {interpreter} > \"$UI_PYTHON_MARKER\"\n"
                        f"exec {shlex.quote(sys.executable)} \"$@\"\n"
                    )
                    executable.chmod(0o755)
                (workspace / "scripts").mkdir()
                shutil.copyfile(ROOT / "scripts/ui-distribution.py", workspace / "scripts/ui-distribution.py")
                dist = workspace / "crates/mesh-llm-ui/dist"
                shutil.copytree(self.dist, dist)
                env = {
                    **os.environ,
                    "PATH": str(binaries),
                    "UI_SOURCE_SHA": SOURCE,
                    "UI_RELEASE_TAG": TAG,
                    "UI_PYTHON_MARKER": str(marker),
                }
                result = subprocess.run([bash, "-c", verify_script], cwd=workspace, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(marker.read_text().strip(), selected)
                (dist / "assets/app.js").write_text("tampered")
                result = subprocess.run([bash, "-c", verify_script], cwd=workspace, env=env, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("checksums", result.stderr)


if __name__ == "__main__":
    unittest.main()
