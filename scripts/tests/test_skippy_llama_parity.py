from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "skippy-llama-parity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("skippy_llama_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class SkippyLlamaParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parity = load_module()

    def test_resolves_first_gguf_shard_for_split_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_cache = os.environ.get("HF_HUB_CACHE")
            cache_root = Path(tmp) / "hub"
            snapshot = (
                cache_root
                / "models--DevQuasar--CohereLabs.command-a-plus-05-2026-bf16-GGUF"
                / "snapshots"
                / "abc123"
            )
            snapshot.mkdir(parents=True)
            first_shard = (
                snapshot
                / "CohereLabs.command-a-plus-05-2026-bf16-Q4_K_M-00001-of-00009.gguf"
            )
            last_shard = (
                snapshot
                / "CohereLabs.command-a-plus-05-2026-bf16-Q4_K_M-00009-of-00009.gguf"
            )
            mmproj = snapshot / "mmproj-CohereLabs.command-a-plus-05-2026-bf16.gguf"
            first_shard.write_bytes(b"larger-first-shard")
            last_shard.write_bytes(b"x")
            mmproj.write_bytes(b"")
            os.environ["HF_HUB_CACHE"] = str(cache_root)
            try:
                resolved = self.parity.resolve_candidate_file(
                    {
                        "repo": "DevQuasar/CohereLabs.command-a-plus-05-2026-bf16-GGUF",
                        "include": "*Q4_K_M*.gguf",
                    }
                )
            finally:
                if previous_cache is None:
                    os.environ.pop("HF_HUB_CACHE", None)
                else:
                    os.environ["HF_HUB_CACHE"] = previous_cache

        self.assertEqual(resolved, first_shard)

    def test_runtime_slice_admission_rejects_architecture_allowlists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(
                    "if (model->arch != LLM_ARCH_LLAMA) { return SKIPPY_STATUS_UNSUPPORTED; }"
                ),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_requires_realized_contract_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source().replace(
                    "stage graph did not expose a stable input activation boundary",
                    "missing input boundary",
                ),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_rejects_diagnostic_literals_without_controls(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(controls=False), encoding="utf-8"
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 6)

    def test_runtime_slice_admission_rejects_detached_failure_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(detached_failure=True),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_rejects_commented_out_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(commented_control=True),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_rejects_inactive_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(inactive_controls=True),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_rejects_nested_invalid_argument_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(nested_invalid_failure=True),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_rejects_failure_after_early_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(early_return_failure=True),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_rejects_nested_boundary_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(nested_boundary_failure=True),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_rejects_unbraced_invalid_argument_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(unbraced_invalid_failure=True),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_rejects_unbraced_boundary_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(unbraced_boundary_failure=True),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 1)

    def test_runtime_slice_admission_rejects_raw_string_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(raw_string_controls=True),
                encoding="utf-8",
            )
            with patch("sys.stderr"):
                failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 6)

    def test_runtime_slice_admission_accepts_architecture_independent_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(self.runtime_slice_admission_source(), encoding="utf-8")

            failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 0)

    def test_runtime_slice_admission_allows_architecture_specific_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            llama_root = Path(tmp)
            source = llama_root / "src/skippy/model_loading.cpp"
            source.parent.mkdir(parents=True)
            source.write_text(
                self.runtime_slice_admission_source(
                    implementation=(
                        "if (model->arch == LLM_ARCH_GLM_DSA) { configure_graph(); }"
                    )
                ),
                encoding="utf-8",
            )

            failures = self.parity.validate_runtime_slice_admission(llama_root)

        self.assertEqual(failures, 0)

    @staticmethod
    def runtime_slice_admission_source(
        extra: str = "",
        implementation: str = "",
        controls: bool = True,
        detached_failure: bool = False,
        commented_control: bool = False,
        inactive_controls: bool = False,
        nested_invalid_failure: bool = False,
        nested_boundary_failure: bool = False,
        early_return_failure: bool = False,
        unbraced_invalid_failure: bool = False,
        unbraced_boundary_failure: bool = False,
        raw_string_controls: bool = False,
    ) -> str:
        checks = (
            "layer_end exceeds model layer count",
            "only the first runtime slice may include token embeddings",
            "the first runtime slice must include token embeddings",
            "only the final runtime slice may include output tensors",
            "stage graph did not expose a stable output activation boundary",
            "stage graph did not expose a stable input activation boundary",
        )
        invalid_argument_checks = (
            (
                "config->layer_end > n_layer",
                checks[0],
            ),
            (
                "config->include_embeddings && config->layer_start != 0 && !config->include_output",
                checks[1],
            ),
            (
                "config->layer_start == 0 && !config->include_embeddings",
                checks[2],
            ),
            (
                "config->include_output && config->layer_end != n_layer",
                checks[3],
            ),
        )
        def invalid_argument_failure(message: str) -> str:
            return (
                "llama_model_free(model); "
                f'const char * message = "{message}"; '
                "if (event_scope != nullptr) { "
                "event_scope->emit_failure(SKIPPY_STATUS_INVALID_ARGUMENT, message); } "
                "skippy_set_error(out_error, SKIPPY_STATUS_INVALID_ARGUMENT, message); "
                "return SKIPPY_STATUS_INVALID_ARGUMENT;"
            )

        control_lines = tuple(
            f"if ({guard}) {{ {invalid_argument_failure(message)} }}"
            for guard, message in invalid_argument_checks
        )
        if detached_failure:
            guard, message = invalid_argument_checks[0]
            control_lines = (
                f"if ({guard}) {{ }}",
                f"{{ {invalid_argument_failure(message)} }}",
                *control_lines[1:],
            )
        if commented_control:
            control_lines = (f"/* {control_lines[0]} */", *control_lines[1:])
        if nested_invalid_failure:
            guard, message = invalid_argument_checks[0]
            control_lines = (
                f"if ({guard}) {{ if (false) {{ {invalid_argument_failure(message)} }} }}",
                *control_lines[1:],
            )
        if early_return_failure:
            guard, message = invalid_argument_checks[0]
            control_lines = (
                f"if ({guard}) {{ return SKIPPY_STATUS_OK; "
                f"{invalid_argument_failure(message)} }}",
                *control_lines[1:],
            )
        if unbraced_invalid_failure:
            guard, message = invalid_argument_checks[0]
            control_lines = (
                f"if ({guard}) {{ if (false) {invalid_argument_failure(message)} }}",
                *control_lines[1:],
            )
        if controls:
            boundary_lines = (
                "if (!stage_model->ctx->get_activation_boundary(type, elements, bytes)) { "
                f'return fail_boundary_load("{checks[4]}"); }}',
                "if (!stage_model->ctx->get_input_activation_boundary(type, elements, bytes)) { "
                f'return fail_boundary_load("{checks[5]}"); }}',
            )
            if nested_boundary_failure:
                boundary_lines = (
                    "if (!stage_model->ctx->get_activation_boundary(type, elements, bytes)) { "
                    f'if (false) {{ return fail_boundary_load("{checks[4]}"); }} }}',
                    boundary_lines[1],
                )
            if unbraced_boundary_failure:
                boundary_lines = (
                    "if (!stage_model->ctx->get_activation_boundary(type, elements, bytes)) { "
                    f'if (false) return fail_boundary_load("{checks[4]}"); }}',
                    boundary_lines[1],
                )
        else:
            control_lines = ()
            boundary_lines = tuple(
                f'const char * diagnostic = "{check}";' for check in checks
            )
        if inactive_controls:
            control_lines = ("#if 0", "#if 1", *control_lines, "#endif", "#endif")
            boundary_lines = (
                "#if 0",
                "#if 1",
                *boundary_lines,
                "#endif",
                "#endif",
            )
        if raw_string_controls:
            fake_controls = "\n".join((*control_lines, *boundary_lines))
            extra = f'const char * fake = R"FAKE({fake_controls})FAKE";'
            control_lines = ()
            boundary_lines = ()
        return "\n".join(
            (
                "static enum skippy_status skippy_finish_model_open(",
                extra,
                *control_lines,
                "skippy_model * stage_model = new skippy_model{};",
                implementation,
                *boundary_lines,
                "enum skippy_status skippy_model_open_impl(",
            )
        )


class BoundaryRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parity = load_module()

    def test_unregistered_runnable_row_fails_without_reason(self):
        rows = [{"llama_model": "somearch", "status": "certified"}]
        failures = self.parity.validate_boundary_registration(rows, set())
        self.assertEqual(failures, 1)

    def test_certified_row_with_reason_fails_even_when_registered(self):
        rows = [
            {
                "llama_model": "somearch",
                "status": "certified",
                "unsupported_reason": "non-causal encoder",
            }
        ]
        # Certified rows may never carry an unsupported_reason, regardless
        # of hook state — that combination is a manifest error.
        self.assertEqual(self.parity.validate_boundary_registration(rows, {"somearch"}), 1)
        self.assertEqual(self.parity.validate_boundary_registration(rows, set()), 1)

    def test_unregistered_candidate_row_with_reason_passes(self):
        rows = [
            {
                "llama_model": "somearch",
                "status": "candidate",
                "unsupported_reason": "non-causal encoder",
            }
        ]
        self.assertEqual(self.parity.validate_boundary_registration(rows, set()), 0)

    def test_registered_row_with_reason_fails(self):
        rows = [
            {
                "llama_model": "somearch",
                "status": "certified",
                "unsupported_reason": "stale classification",
            }
        ]
        self.assertEqual(self.parity.validate_boundary_registration(rows, {"somearch"}), 1)

    def test_registered_row_without_reason_passes(self):
        rows = [{"llama_model": "somearch", "status": "certified"}]
        self.assertEqual(self.parity.validate_boundary_registration(rows, {"somearch"}), 0)

    def test_non_runnable_statuses_skipped(self):
        rows = [{"llama_model": "otherarch", "status": "implementation_base"}]
        self.assertEqual(self.parity.validate_boundary_registration(rows, set()), 0)

    def test_needs_boundary_registration_rows_lists_pending(self):
        rows = [
            {"llama_model": "aaa", "status": "certified"},
            {"llama_model": "bbb", "status": "certified"},
            {
                "llama_model": "ccc",
                "status": "certified",
                "unsupported_reason": "deliberate",
            },
            {"llama_model": "ddd", "status": "certified"},
        ]
        pending = self.parity.needs_boundary_registration_rows(rows, {"ddd"})
        self.assertEqual(pending, ["aaa", "bbb"])


class BoundaryRegisteredModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parity = load_module()

    def _models_dir(self, tmp: Path):
        models = tmp / "src" / "models"
        models.mkdir(parents=True)
        return models

    def test_both_hooks_required(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            models = self._models_dir(tmp)
            (models / "full.cpp").write_text("begin_block(inpL, il);\nend_block(cur, il);\n")
            (models / "half.cpp").write_text("begin_block(inpL, il);\n")
            registered = self.parity.boundary_registered_models(tmp)
        self.assertEqual(registered, {"full"})

    def test_missing_models_dir_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            self.assertEqual(self.parity.boundary_registered_models(Path(tmp_name)), set())


class ModelPinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parity = load_module()

    def test_valid_pin_passes(self):
        rows = [
            {
                "llama_model": "somearch",
                "status": "certified",
                "model_pin": {
                    "repo": "org/some-gguf",
                    "revision": "0" * 40,
                    "file": "model-Q4_K_M.gguf",
                    "size_bytes": 123,
                    "blob_sha256": "a" * 64,
                },
            }
        ]
        self.assertEqual(self.parity.validate_model_pins(rows), 0)

    def test_floating_pin_fails(self):
        rows = [
            {
                "llama_model": "somearch",
                "status": "certified",
                "model_pin": {"repo": "org/some-gguf"},
            }
        ]
        self.assertEqual(self.parity.validate_model_pins(rows), 1)

    def test_malformed_blob_fails(self):
        rows = [
            {
                "llama_model": "somearch",
                "status": "certified",
                "model_pin": {
                    "repo": "org/some-gguf",
                    "revision": "0" * 40,
                    "file": "model.gguf",
                    "size_bytes": 5,
                    "blob_sha256": "not-hex",
                },
            }
        ]
        self.assertEqual(self.parity.validate_model_pins(rows), 1)


class PinManifestJoinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parity = load_module()
        self.certified_dir = ROOT / "ci/llama-canary"
        self._original = (self.certified_dir / "family-certified.json").read_text(
            encoding="utf-8"
        )
        self.addCleanup(
            lambda: (self.certified_dir / "family-certified.json").write_text(
                self._original, encoding="utf-8"
            )
        )

    def _write_certified(self, models: list[dict]) -> None:
        import json

        (self.certified_dir / "family-certified.json").write_text(
            json.dumps({"schema_version": 1, "policy": {}, "models": models}),
            encoding="utf-8",
        )

    def _pin(self, size=10, blob="a" * 64, repo="org/some-gguf", revision="0" * 40):
        return {
            "llama_model": "somearch",
            "status": "certified",
            "model_pin": {
                "repo": repo,
                "revision": revision,
                "file": "model.gguf",
                "size_bytes": size,
                "blob_sha256": blob,
            },
        }

    def _artifact(self, size=10, blob="a" * 64, repo="org/some-gguf", revision="0" * 40):
        return {
            "artifact": {
                "repo": repo,
                "revision": revision,
                "file_integrity": {
                    "model.gguf": {"size_bytes": size, "blob_id": blob}
                },
            }
        }

    def test_matching_pin_joins(self):
        self._write_certified([self._artifact()])
        self.assertEqual(self.parity.validate_pin_manifest_join([self._pin()]), 0)

    def test_unknown_pin_fails(self):
        self._write_certified([])
        self.assertEqual(self.parity.validate_pin_manifest_join([self._pin()]), 1)

    def test_disagreeing_integrity_fails(self):
        self._write_certified([self._artifact(size=999)])
        self.assertEqual(self.parity.validate_pin_manifest_join([self._pin()]), 1)

    def test_pin_without_manifest_entry_passes_when_no_pin_rows(self):
        self._write_certified([])
        self.assertEqual(self.parity.validate_pin_manifest_join([]), 0)


if __name__ == "__main__":
    unittest.main()
