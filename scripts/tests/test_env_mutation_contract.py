from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check-env-mutation-contract.py"
AUDITED_FILE = "crates/model-hf/src/store/local.rs"
TODO = "// TODO: Audit that the environment access only happens in single-threaded code."


class EnvironmentMutationContractTests(unittest.TestCase):
    def run_checker(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_repository_census_is_serialized_or_explicitly_deferred(self) -> None:
        result = self.run_checker()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("35 Rust files", result.stdout)
        self.assertIn("200 mutation sites", result.stdout)
        self.assertIn("18 contract-audited files", result.stdout)

    def test_unregistered_mutation_file_is_rejected_by_repository_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "crates/new-crate/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                "unsafe { std::env::set_var(\"MESH_LLM_NEW_ENV\", \"1\") };\n",
                encoding="utf-8",
            )

            result = self.run_checker("--root", str(root))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unregistered process-environment mutation file", result.stderr)

    def test_unserialized_test_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / AUDITED_FILE
            source.parent.mkdir(parents=True)
            source.write_text(
                """#[cfg(test)]
mod tests {
    #[test]
    fn mutates_process_environment_without_serialization() {
        // SAFETY: this comment cannot replace the required test lock.
        unsafe { std::env::set_var(\"MESH_LLM_TEST_ENV\", \"1\") };
    }
}
""",
                encoding="utf-8",
            )

            result = self.run_checker("--root", str(root), "--file", AUDITED_FILE)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not covered by #[serial]", result.stderr)

    def test_serial_text_in_production_comment_is_not_an_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / AUDITED_FILE
            source.parent.mkdir(parents=True)
            source.write_text(
                """#[cfg(test)]
mod unrelated_tests {}

fn production_mutation() {
    // SAFETY: pretend the enclosing test contract is `#[serial]`.
    unsafe { std::env::set_var("MESH_LLM_TEST_ENV", "1") };
}
""",
                encoding="utf-8",
            )

            result = self.run_checker("--root", str(root), "--file", AUDITED_FILE)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not covered by #[serial]", result.stderr)

    def test_non_adjacent_safety_comment_does_not_cover_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / AUDITED_FILE
            source.parent.mkdir(parents=True)
            source.write_text(
                """#[cfg(test)]
mod tests {
    #[test]
    #[serial]
    fn mutation_with_distant_comment() {
        // SAFETY: this applies only to the first mutation.
        unsafe { std::env::set_var("FIRST", "1") };
        let _intervening_statement = true;
        unsafe { std::env::set_var("SECOND", "2") };
    }
}
""",
                encoding="utf-8",
            )

            result = self.run_checker("--root", str(root), "--file", AUDITED_FILE)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("test environment mutation needs a SAFETY comment", result.stderr)

    def test_deferred_mutation_requires_safety_comment_and_todo(self) -> None:
        deferred_file = "crates/skippy-runtime/src/logging.rs"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / deferred_file
            source.parent.mkdir(parents=True)
            source.write_text(
                f"""fn configure_runtime() {{
    {TODO}
    unsafe {{ std::env::set_var("RUNTIME", "1") }};
}}
""",
                encoding="utf-8",
            )

            result = self.run_checker("--root", str(root), "--file", deferred_file)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("needs adjacent SAFETY and audit TODO comments", result.stderr)

    def test_synchronous_bootstrap_mutation_rejects_audit_todo(self) -> None:
        bootstrap_file = (
            "crates/mesh-llm-host-runtime/src/inference/skippy/metal_pipeline_cache.rs"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / bootstrap_file
            source.parent.mkdir(parents=True)
            source.write_text(
                f"""fn configure_metal_pipeline_cache() {{
    // SAFETY: single-threaded bootstrap before the Tokio runtime.
    {TODO}
    unsafe {{ std::env::set_var("METAL_CACHE", "1") }};
}}
""",
                encoding="utf-8",
            )

            result = self.run_checker("--root", str(root), "--file", bootstrap_file)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stale environment audit TODO remains", result.stderr)

    def test_synchronous_bootstrap_mutation_requires_exact_boundary(self) -> None:
        bootstrap_file = (
            "crates/mesh-llm-host-runtime/src/inference/skippy/metal_pipeline_cache.rs"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / bootstrap_file
            source.parent.mkdir(parents=True)
            source.write_text(
                """fn configure_late() {
    // SAFETY: generic claim.
    unsafe { std::env::set_var("METAL_CACHE", "1") };
}
""",
                encoding="utf-8",
            )

            result = self.run_checker("--root", str(root), "--file", bootstrap_file)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "bootstrap mutation must remain in configure_metal_pipeline_cache",
                result.stderr,
            )


if __name__ == "__main__":
    unittest.main()
