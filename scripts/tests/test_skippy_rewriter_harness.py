from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "skippy-rewriter-harness.py"


def load_module():
    spec = importlib.util.spec_from_file_location("skippy_rewriter_harness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


PROOF = {
    "loop": {"var": "il", "start": "il_start", "end": "il_end"},
    "activation_in": "inpL",
    "activation_out": "cur",
    "embedding_owner": True,
    "output_owner": True,
    "terminal_predicates": ["il == il_end - 1 && inp_out_ids"],
    "nonlocal_exits": [],
}

EDITS = [{"kind": "insert", "file": "src/models/llama.cpp", "range": [1, 2], "text_ref": "a"}]


def make_report(builders, **overrides):
    report = {
        "schema_version": 0,
        "llama_cpp_commit": "cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
        "generator_version": "0.1.0",
        "builders": builders,
        "summary": {},
    }
    report.update(overrides)
    counts = {v: 0 for v in {"transformable", "already_transformed", "unsupported_shape", "error"}}
    for builder in builders:
        verdict = builder.get("verdict")
        if verdict in counts:
            counts[verdict] += 1
    report["summary"] = counts
    return report


class SkippyRewriterHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = load_module()

    def test_valid_transformable_report_passes(self) -> None:
        report = make_report(
            [{"file": "src/models/llama.cpp", "verdict": "transformable", "proof": PROOF, "edits": EDITS}]
        )
        self.assertEqual(self.harness.validate_report(report), [])

    def test_missing_proof_field_is_contract_violation(self) -> None:
        broken = {k: v for k, v in PROOF.items() if k != "activation_out"}
        report = make_report(
            [{"file": "src/models/llama.cpp", "verdict": "transformable", "proof": broken, "edits": EDITS}]
        )
        errors = self.harness.validate_report(report)
        self.assertTrue(any("activation_out" in e for e in errors))

    def test_unsupported_shape_requires_reason(self) -> None:
        report = make_report(
            [{"file": "src/models/rwkv7-base.cpp", "verdict": "unsupported_shape"}]
        )
        errors = self.harness.validate_report(report)
        self.assertTrue(any("unsupported_reason" in e for e in errors))

    def test_unknown_verdict_rejected(self) -> None:
        # The superseded vocabulary schema's 'understood' must NOT validate.
        report = make_report([{"file": "src/models/llama.cpp", "verdict": "understood"}])
        errors = self.harness.validate_report(report)
        self.assertTrue(any("unknown verdict" in e for e in errors))

    def test_already_transformed_carries_no_edits(self) -> None:
        report = make_report(
            [{"file": "src/models/qwen3.cpp", "verdict": "already_transformed", "edits": EDITS}]
        )
        errors = self.harness.validate_report(report)
        self.assertTrue(any("no edits" in e for e in errors))

    def test_duplicate_builder_record_rejected(self) -> None:
        record = {"file": "src/models/llama.cpp", "verdict": "transformable", "proof": PROOF, "edits": EDITS}
        report = make_report([record, dict(record)])
        errors = self.harness.validate_report(report)
        self.assertTrue(any("duplicate" in e for e in errors))

    def test_summary_must_match_builder_count(self) -> None:
        report = make_report(
            [{"file": "src/models/llama.cpp", "verdict": "transformable", "proof": PROOF, "edits": EDITS}]
        )
        report["summary"]["transformable"] = 2
        errors = self.harness.validate_report(report)
        self.assertTrue(any("summary counts" in e for e in errors))

    def test_idempotence_flags_second_run_transformable(self) -> None:
        report = make_report(
            [{"file": "src/models/llama.cpp", "verdict": "transformable", "proof": PROOF, "edits": EDITS}]
        )
        errors = self.harness.check_idempotence(report)
        self.assertTrue(any("idempotence" in e for e in errors))

    def test_idempotence_all_ready_tree(self) -> None:
        report = make_report(
            [{"file": "src/models/llama.cpp", "verdict": "already_transformed"}]
        )
        self.assertEqual(self.harness.check_idempotence(report), [])

    def _run_main(self, report, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            return self.harness.main(["--report", str(path), *sum(kwargs.items(), ())] if False else
                                     self._argv(path, **kwargs))

    @staticmethod
    def _argv(path, **kwargs):
        argv = ["--report", str(path)]
        for key, value in kwargs.items():
            argv += [f"--{key.replace('_', '-')}", value]
        return argv

    def test_cli_gates_on_patch_drift_when_policy_is_fail(self) -> None:
        report = make_report([{"file": "src/models/llama.cpp", "verdict": "already_transformed"}])
        code = self._run_main(report, patch_check="fail", patch_drift_gate="fail")
        self.assertEqual(code, 1)

    def test_cli_warns_on_patch_drift_when_policy_is_warn(self) -> None:
        report = make_report([{"file": "src/models/llama.cpp", "verdict": "already_transformed"}])
        code = self._run_main(report, patch_check="fail", patch_drift_gate="warn")
        self.assertEqual(code, 0)

    def test_cli_gates_on_compile_failure_from_day_one(self) -> None:
        report = make_report([{"file": "src/models/llama.cpp", "verdict": "already_transformed"}])
        code = self._run_main(report, compile_result="fail")
        self.assertEqual(code, 1)

    def test_cli_gates_on_graph_verifier_failure_from_day_one(self) -> None:
        report = make_report([{"file": "src/models/llama.cpp", "verdict": "already_transformed"}])
        code = self._run_main(report, graph_verify_result="fail")
        self.assertEqual(code, 1)

    def test_cli_accepts_clean_report(self) -> None:
        report = make_report([{"file": "src/models/llama.cpp", "verdict": "already_transformed"}])
        code = self._run_main(report)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
