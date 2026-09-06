#!/usr/bin/env python3
"""Regression coverage for blobless llama.cpp patch preparation."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREPARE_LLAMA = ROOT / "scripts" / "prepare-llama.sh"


class PrepareLlamaTests(unittest.TestCase):
    def run_git(
        self, cwd: Path, *args: str, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-c", "commit.gpgsign=false", *args],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=capture_output,
        )

    def test_blobless_checkout_is_replaced_before_three_way_patch_application(
        self,
    ) -> None:
        """A complete checkout preserves three-way application of the patch queue."""
        with tempfile.TemporaryDirectory(prefix="mesh-llm-prepare-llama-") as temp_dir:
            root = Path(temp_dir)
            upstream = root / "upstream"
            author = root / "author"
            workdir = root / "workdir"
            patch_dir = root / "patches"
            pin_file = root / "upstream.txt"

            self.run_git(root, "init", "--initial-branch=master", str(upstream))
            self.run_git(upstream, "config", "user.name", "Test Author")
            self.run_git(upstream, "config", "user.email", "test@example.com")
            self.run_git(upstream, "config", "uploadpack.allowFilter", "true")
            (upstream / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
            self.run_git(upstream, "add", "sample.txt")
            self.run_git(upstream, "commit", "-m", "base")
            self.run_git(root, "clone", str(upstream), str(author))
            self.run_git(author, "config", "user.name", "Test Author")
            self.run_git(author, "config", "user.email", "test@example.com")
            (author / "sample.txt").write_text("patched\ntwo\nthree\n", encoding="utf-8")
            self.run_git(author, "commit", "-am", "local patch")
            patch_dir.mkdir()
            patch = self.run_git(
                author, "format-patch", "-1", "--stdout", capture_output=True
            ).stdout
            (patch_dir / "0001-local.patch").write_text(patch, encoding="utf-8")

            # Advance the pinned upstream on a disjoint line. The patch cannot
            # apply directly, so `git am --3way` must consult the patch's index
            # post-image, which exists only in the local author checkout.
            (upstream / "sample.txt").write_text("one\ntwo\nupstream\n", encoding="utf-8")
            self.run_git(upstream, "commit", "-am", "upstream change")
            pin_file.write_text(
                f"{self.run_git(upstream, 'rev-parse', 'HEAD', capture_output=True).stdout.strip()}\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    f"file://{upstream}",
                    str(workdir),
                ],
                cwd=root,
                check=True,
            )
            self.assertEqual(
                self.run_git(
                    workdir,
                    "config",
                    "--bool",
                    "--get",
                    "remote.origin.promisor",
                    capture_output=True,
                ).stdout.strip(),
                "true",
            )

            env = os.environ | {
                "LLAMA_UPSTREAM_URL": f"file://{upstream}",
                "LLAMA_WORKDIR": str(workdir),
                "LLAMA_PIN_FILE": str(pin_file),
                "LLAMA_PATCH_DIR": str(patch_dir),
                "LLAMA_GIT_MAX_ATTEMPTS": "1",
            }
            subprocess.run([str(PREPARE_LLAMA), "pinned"], cwd=ROOT, check=True, env=env)

            self.assertEqual(
                (workdir / "sample.txt").read_text(encoding="utf-8"),
                "patched\ntwo\nupstream\n",
            )
            promisor = subprocess.run(
                ["git", "config", "--get", "remote.origin.promisor"],
                cwd=workdir,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(promisor.returncode, 0)
            partial_clone = subprocess.run(
                ["git", "config", "--get", "extensions.partialClone"],
                cwd=workdir,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(partial_clone.returncode, 0)

    def test_duplicate_patch_sequence_is_rejected_before_application(self) -> None:
        """Patch queues must remain uniquely and contiguously numbered."""
        with tempfile.TemporaryDirectory(prefix="mesh-llm-prepare-sequence-") as temp_dir:
            root = Path(temp_dir)
            upstream = root / "upstream"
            workdir = root / "workdir"
            patch_dir = root / "patches"
            pin_file = root / "upstream.txt"

            self.run_git(root, "init", "--initial-branch=master", str(upstream))
            self.run_git(upstream, "config", "user.name", "Test Author")
            self.run_git(upstream, "config", "user.email", "test@example.com")
            (upstream / "sample.txt").write_text("base\n", encoding="utf-8")
            self.run_git(upstream, "add", "sample.txt")
            self.run_git(upstream, "commit", "-m", "base")
            self.run_git(root, "clone", str(upstream), str(workdir))
            pin_file.write_text(
                f"{self.run_git(upstream, 'rev-parse', 'HEAD', capture_output=True).stdout.strip()}\n",
                encoding="utf-8",
            )
            patch_dir.mkdir()
            (patch_dir / "0001-first.patch").write_text("first\n", encoding="utf-8")
            (patch_dir / "0001-second.patch").write_text("second\n", encoding="utf-8")

            env = os.environ | {
                "LLAMA_UPSTREAM_URL": str(upstream),
                "LLAMA_WORKDIR": str(workdir),
                "LLAMA_PIN_FILE": str(pin_file),
                "LLAMA_PATCH_DIR": str(patch_dir),
                "LLAMA_GIT_MAX_ATTEMPTS": "1",
            }
            result = subprocess.run(
                [str(PREPARE_LLAMA), "pinned"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "invalid llama patch sequence: expected 0002, found 0001",
                result.stderr,
            )

    def test_patch_application_has_deterministic_commit_identity(self) -> None:
        """Equivalent clean preparations produce one reusable patched SHA."""
        with tempfile.TemporaryDirectory(prefix="mesh-llm-prepare-identity-") as temp_dir:
            root = Path(temp_dir)
            upstream = root / "upstream"
            author = root / "author"
            patch_dir = root / "patches"
            pin_file = root / "upstream.txt"
            hook_dir = root / "hooks"
            hook_marker = root / "hook-ran"

            self.run_git(root, "init", "--initial-branch=master", str(upstream))
            self.run_git(upstream, "config", "user.name", "Patch Author")
            self.run_git(upstream, "config", "user.email", "author@example.com")
            (upstream / "sample.txt").write_text("base\n", encoding="utf-8")
            self.run_git(upstream, "add", "sample.txt")
            self.run_git(upstream, "commit", "-m", "base")
            pin_file.write_text(
                f"{self.run_git(upstream, 'rev-parse', 'HEAD', capture_output=True).stdout.strip()}\n",
                encoding="utf-8",
            )

            self.run_git(root, "clone", str(upstream), str(author))
            self.run_git(author, "config", "user.name", "Patch Author")
            self.run_git(author, "config", "user.email", "author@example.com")
            (author / "sample.txt").write_text("patched\n", encoding="utf-8")
            self.run_git(author, "commit", "-am", "local patch")
            patch_dir.mkdir()
            patch = self.run_git(
                author, "format-patch", "-1", "--stdout", capture_output=True
            ).stdout
            (patch_dir / "0001-local.patch").write_text(patch, encoding="utf-8")
            hook_dir.mkdir()
            for hook_name in ("applypatch-msg", "pre-applypatch"):
                hook = hook_dir / hook_name
                hook.write_text(
                    f"#!/bin/sh\ntouch '{hook_marker}'\nexit 1\n", encoding="utf-8"
                )
                hook.chmod(0o755)

            patched_shas = []
            for index, (committer_date, timezone) in enumerate(
                (
                    ("2001-01-01T00:00:00Z", "UTC"),
                    ("2031-01-01T00:00:00Z", "America/Toronto"),
                )
            ):
                workdir = root / f"workdir-{index}"
                env = os.environ | {
                    "GIT_AUTHOR_EMAIL": f"author-{index}@example.com",
                    "GIT_AUTHOR_NAME": f"Ambient Author {index}",
                    "GIT_COMMITTER_DATE": committer_date,
                    "GIT_COMMITTER_EMAIL": f"committer-{index}@example.com",
                    "GIT_COMMITTER_NAME": f"Ambient Committer {index}",
                    "GIT_CONFIG_COUNT": "2",
                    "GIT_CONFIG_KEY_0": "commit.gpgsign",
                    "GIT_CONFIG_VALUE_0": "true",
                    "GIT_CONFIG_KEY_1": "core.hooksPath",
                    "GIT_CONFIG_VALUE_1": str(hook_dir),
                    "TZ": timezone,
                    "LLAMA_UPSTREAM_URL": f"file://{upstream}",
                    "LLAMA_WORKDIR": str(workdir),
                    "LLAMA_PIN_FILE": str(pin_file),
                    "LLAMA_PATCH_DIR": str(patch_dir),
                    "LLAMA_GIT_MAX_ATTEMPTS": "1",
                }
                subprocess.run(
                    [str(PREPARE_LLAMA), "pinned"],
                    cwd=ROOT,
                    check=True,
                    env=env,
                )
                patched_shas.append(
                    (workdir / ".mesh-llm-patched-sha")
                    .read_text(encoding="utf-8")
                    .strip()
                )
                self.assertEqual(
                    (workdir / ".mesh-llm-prepare-schema")
                    .read_text(encoding="utf-8")
                    .strip(),
                    "4",
                )
                self.assertFalse(hook_marker.exists())

            self.assertEqual(patched_shas[0], patched_shas[1])


if __name__ == "__main__":
    unittest.main()
