from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


class RunnerSelectorMixin:
    """Shared runner-selector harness for CI contract test cases."""

    def run_runner_selector(
        self,
        *,
        event_name: str,
        ref: str,
        main_enabled: str,
        manual_enabled: str,
        original_event_name: str = "",
        repository: str = "Mesh-LLM/mesh-llm",
        head_repository: str | None = None,
        head_sha: str = "0123456789abcdef0123456789abcdef01234567",
        pr_enabled: str = "false",
        pr_canary_ref: str = "",
        pr_approved_ref: str = "",
        pr_approved_sha: str = "",
        force_hosted: str = "false",
        current_date: str = "2026-08-14",
    ) -> dict[str, str]:
        action = self.read_action("select-ci-runners")
        run_block = action.split("      run: |\n", maxsplit=1)[1]
        script = "\n".join(
            line[8:] if line.startswith("        ") else line
            for line in run_block.splitlines()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "github-output"
            bin_dir = Path(temp_dir) / "bin"
            bin_dir.mkdir()
            date = bin_dir / "date"
            date.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$SELECTOR_TEST_DATE\"\n",
                encoding="utf-8",
            )
            date.chmod(0o755)
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
                    "SELECTOR_TEST_DATE": current_date,
                    "GITHUB_OUTPUT": str(output),
                    "INPUT_EVENT_NAME": event_name,
                    "INPUT_ORIGINAL_EVENT_NAME": original_event_name,
                    "GITHUB_EVENT_NAME": event_name,
                    "INPUT_REPOSITORY": repository,
                    "INPUT_HEAD_REPOSITORY": head_repository or repository,
                    "INPUT_HEAD_SHA": head_sha,
                    "GITHUB_REPOSITORY": repository,
                    "INPUT_REF": ref,
                    "GITHUB_REF": ref,
                    "INPUT_DEPOT_MAIN_ENABLED": main_enabled,
                    "INPUT_DEPOT_PR_ENABLED": pr_enabled,
                    "INPUT_PR_CANARY_REF": pr_canary_ref,
                    "INPUT_PR_APPROVED_REF": pr_approved_ref,
                    "INPUT_PR_APPROVED_SHA": pr_approved_sha,
                    "INPUT_FORCE_HOSTED": force_hosted,
                    "INPUT_MANUAL_USE_DEPOT": manual_enabled,
                    "DISPATCH_ORIGINAL_EVENT_NAME": original_event_name,
                },
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return dict(
                line.split("=", maxsplit=1)
                for line in output.read_text(encoding="utf-8").splitlines()
            )
