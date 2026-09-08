//! Evidence-marker writer for gated native-runtime integration tests.
//!
//! `crates/skippy-runtime/tests/runtime_events_native.rs` needs to prove, to
//! a reader who never ran it, whether a specific gated step genuinely
//! executed. Defect D10 (`.omo/plans/event-system-fixes.md` task 11) was
//! exactly this contract broken: the marker file's `executed` line was
//! written before the test's own gate check ran, so an ungated (default)
//! `cargo test` produced a marker file that overclaimed a real native run.
//!
//! This module owns the one append-only write both the integration test and
//! its own unit tests call, so the file-I/O behavior -- create on first
//! write, append thereafter, and a strict no-op when unconfigured -- is
//! directly unit-testable without a real native runtime or process
//! environment mutation. The path is a plain parameter rather than an
//! environment lookup on purpose: `std::env::set_var`/`remove_var` are
//! `unsafe` as of Rust 1.82 because mutating process-global environment
//! state races under parallel test execution, so a testable writer must
//! take its destination as data, leaving the (single, non-parallel) caller
//! to resolve any environment variable.

use std::fs;
use std::io::Write;
use std::path::Path;

/// Appends `line` (plus a trailing newline) to the evidence file at `path`,
/// creating the file on first write and appending on every subsequent call.
/// A `None` path is a deliberate no-op: an evidence file is opt-in, so a run
/// with no configured destination must never create or touch anything on
/// disk.
pub fn write_evidence_marker(path: Option<&Path>, line: &str) {
    let Some(path) = path else {
        return;
    };
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|error| panic!("open evidence file {}: {error}", path.display()));
    writeln!(file, "{line}").expect("write evidence marker");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_path_touches_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let untouched = dir.path().join("evidence.txt");

        write_evidence_marker(None, "executed");

        assert!(
            !untouched.exists(),
            "write_evidence_marker(None, ..) must not create a file"
        );
    }

    #[test]
    fn some_path_creates_the_file_on_first_write() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("evidence.txt");
        assert!(!path.exists());

        write_evidence_marker(Some(&path), "first line");

        let contents = fs::read_to_string(&path).expect("read evidence file");
        assert_eq!(contents, "first line\n");
    }

    #[test]
    fn repeated_calls_append_in_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("evidence.txt");

        write_evidence_marker(Some(&path), "executed");
        write_evidence_marker(Some(&path), "model-open: ok");
        write_evidence_marker(Some(&path), "reporter-clear: ok");

        let contents = fs::read_to_string(&path).expect("read evidence file");
        assert_eq!(contents, "executed\nmodel-open: ok\nreporter-clear: ok\n");
    }

    #[test]
    fn an_existing_file_is_appended_to_not_truncated() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("evidence.txt");
        fs::write(&path, "pre-existing\n").expect("seed file");

        write_evidence_marker(Some(&path), "appended");

        let contents = fs::read_to_string(&path).expect("read evidence file");
        assert_eq!(contents, "pre-existing\nappended\n");
    }
}
