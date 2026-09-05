"""Validates the executed-evidence invariant for the gated native-runtime
integration test.

`crates/skippy-runtime/tests/runtime_events_native.rs` writes an evidence
marker file when `MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE` is set. Defect D10
(`.omo/plans/event-system-fixes.md` task 11) was an evidence file whose
first line was `executed` even though the run was ungated and no native
symbol had been touched. This module encodes the fix's contract as a
standalone, independently-checkable invariant: a marker file that claims
`executed` as its first line must also show the two farthest real-work
checkpoints (`run_real_native_gate` reaching the model-open attempt, and the
reporter clear that follows it) somewhere in its body. A file that fails
this check is evidence of exactly the D10 shape: a claimed execution with no
proof any real native call happened.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class ExecutedEvidenceInvariantViolation(ValueError):
    """Raised when a marker file claims `executed` without the real-step lines."""


def check_executed_evidence_invariant(lines: list[str]) -> None:
    """Enforces: first line `executed` implies `model-open:` and
    `reporter-clear:` lines exist somewhere in the file.

    A marker file whose first line is not `executed` (e.g. a
    `blocked-when-ungated:` or `blocked:` marker) makes no execution claim,
    so it is unconstrained by this invariant -- only a file CLAIMING a real
    run has to PROVE one.
    """
    if not lines or lines[0].strip() != "executed":
        return
    has_model_open = any(line.startswith("model-open:") for line in lines)
    has_reporter_clear = any(line.startswith("reporter-clear:") for line in lines)
    missing = [
        prefix
        for prefix, present in (
            ("model-open:", has_model_open),
            ("reporter-clear:", has_reporter_clear),
        )
        if not present
    ]
    if missing:
        raise ExecutedEvidenceInvariantViolation(
            "evidence file's first line is 'executed' but is missing required "
            f"line prefix(es): {', '.join(missing)}"
        )


def check_executed_evidence_file(path: Path) -> None:
    """Reads `path` and enforces `check_executed_evidence_invariant` on its lines."""
    lines = path.read_text(encoding="utf-8").splitlines()
    check_executed_evidence_invariant(lines)


class ExecutedEvidenceInvariantTests(unittest.TestCase):
    def test_a_genuine_gated_run_satisfies_the_invariant(self) -> None:
        check_executed_evidence_invariant(
            [
                "executed",
                "exact-abi-admission: native runtime loaded (loader enforces "
                "exact major.minor.patch)",
                "capability-probe: confirmed=0x1 health_messages=0",
                "reporter-install: true",
                "model-open: single-part real model-open succeeded",
                "reporter-clear: returned (quiescent, no callback after clear "
                "observed)",
            ]
        )

    def test_blocked_when_ungated_evidence_is_not_constrained(self) -> None:
        check_executed_evidence_invariant(
            ["blocked-when-ungated: gate unset, no native symbol was touched"]
        )

    def test_blocked_gated_evidence_is_not_constrained(self) -> None:
        check_executed_evidence_invariant(
            [
                "blocked: MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR unset, required "
                "when MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST=1"
            ]
        )

    def test_empty_evidence_is_not_constrained(self) -> None:
        check_executed_evidence_invariant([])

    def test_the_original_d10_defect_shape_is_rejected(self) -> None:
        # The exact pre-fix bug: `executed` was written before the gate
        # check, so an ungated run's file had `executed` as its first line
        # with nothing else -- no model-open, no reporter-clear, because no
        # native call was ever made.
        with self.assertRaises(ExecutedEvidenceInvariantViolation) as raised:
            check_executed_evidence_invariant(
                [
                    "executed",
                    "blocked-when-ungated: gate unset, no native symbol was "
                    "touched",
                ]
            )
        message = str(raised.exception)
        self.assertIn("model-open:", message)
        self.assertIn("reporter-clear:", message)

    def test_missing_only_reporter_clear_is_rejected(self) -> None:
        with self.assertRaises(ExecutedEvidenceInvariantViolation) as raised:
            check_executed_evidence_invariant(
                [
                    "executed",
                    "model-open: single-part real model-open succeeded",
                ]
            )
        self.assertIn("reporter-clear:", str(raised.exception))

    def test_check_executed_evidence_file_rejects_a_hand_crafted_bad_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "ungated-but-claims-executed.txt"
            bad_file.write_text(
                "executed\n"
                "blocked-when-ungated: gate unset, no native symbol was "
                "touched\n",
                encoding="utf-8",
            )
            with self.assertRaises(ExecutedEvidenceInvariantViolation):
                check_executed_evidence_file(bad_file)

    def test_check_executed_evidence_file_accepts_a_genuine_gated_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            good_file = Path(tmp) / "gated-and-real.txt"
            good_file.write_text(
                "executed\n"
                "model-open: single-part real model-open succeeded\n"
                "reporter-clear: returned (quiescent, no callback after "
                "clear observed)\n",
                encoding="utf-8",
            )
            check_executed_evidence_file(good_file)  # must not raise


if __name__ == "__main__":
    unittest.main()
