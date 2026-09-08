//! Real native-runtime integration test for Task 8's ABI admission +
//! capability-probe + reporter wiring. Gated behind
//! `MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST=1` so it never touches a native
//! symbol during an ordinary `cargo test`. An ungated run still executes
//! (it is never skipped), but it must never claim `executed`: it prints a
//! `BLOCKED: <prerequisite>` line to stdout, writes only a
//! `blocked-when-ungated: <prerequisite>` evidence marker, and exits 0.
//! `executed` is written only after `run_real_native_gate` has installed the
//! reporter, successfully opened a real model, observed structured production
//! callbacks, exercised unload when that capability is advertised, and cleared
//! the reporter, so a marker file that starts with `executed` reflects a
//! genuine run (review defect D10 -- see `.omo/plans/event-system-fixes.md`
//! task 11).
//!
//! The three prerequisites checked once the gate is set to `1` (the
//! `dynamic-native-runtime` feature, the native runtime bundle directory,
//! and the model path) each print their own `BLOCKED: <reason>` line before
//! panicking: a developer who explicitly opted into the real native gate
//! gets a loud, named failure for a misconfigured opt-in, not a silent
//! pass. Only the top-level gate-unset path is required to stay green,
//! since that is the path an ordinary `cargo test` run takes.

use std::env;
use std::path::PathBuf;

const GATE_ENV: &str = "MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST";
#[cfg(feature = "dynamic-native-runtime")]
const BUNDLE_DIR_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR";
#[cfg(feature = "dynamic-native-runtime")]
const MODEL_ENV: &str = "MESH_LLM_RUNTIME_EVENTS_MODEL";
const EVIDENCE_FILE_ENV: &str = "MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE";

/// Appends one evidence marker line, or does nothing when
/// `MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE` is unset. The actual file I/O is
/// `skippy_runtime::write_evidence_marker`, unit tested directly in
/// `crates/skippy-runtime/src/native_test_evidence.rs`.
fn write_marker(line: &str) {
    let path = env::var_os(EVIDENCE_FILE_ENV).map(PathBuf::from);
    skippy_runtime::write_evidence_marker(path.as_deref(), line);
}

#[test]
fn runtime_events_native_gate() {
    if env::var(GATE_ENV).ok().as_deref() != Some("1") {
        println!("BLOCKED: {GATE_ENV} unset");
        write_marker("blocked-when-ungated: gate unset, no native symbol was touched");
        return;
    }

    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        println!("BLOCKED: dynamic-native-runtime feature not enabled");
        write_marker("blocked: dynamic-native-runtime feature is not enabled for this run");
        panic!("{GATE_ENV}=1 requires the dynamic-native-runtime feature");
    }

    #[cfg(feature = "dynamic-native-runtime")]
    {
        run_real_native_gate();
    }
}

#[cfg(feature = "dynamic-native-runtime")]
fn run_real_native_gate() {
    let bundle_dir = env::var(BUNDLE_DIR_ENV).unwrap_or_else(|_| {
        println!("BLOCKED: {BUNDLE_DIR_ENV} unset");
        write_marker(&format!(
            "blocked: {BUNDLE_DIR_ENV} unset, required when {GATE_ENV}=1"
        ));
        panic!("{GATE_ENV}=1 requires {BUNDLE_DIR_ENV} to point at a dynamic native runtime")
    });
    let model_path = env::var(MODEL_ENV).unwrap_or_else(|_| {
        println!("BLOCKED: {MODEL_ENV} unset");
        write_marker(&format!(
            "blocked: {MODEL_ENV} unset, required when {GATE_ENV}=1"
        ));
        panic!("{GATE_ENV}=1 requires {MODEL_ENV} to name a readable model")
    });

    let bundle_dir = PathBuf::from(bundle_dir);
    let libraries = discover_libraries(&bundle_dir);
    assert!(
        !libraries.is_empty(),
        "no native runtime libraries found under {}",
        bundle_dir.display()
    );

    if !skippy_runtime::native_runtime_loaded() {
        unsafe { skippy_runtime::load_native_runtime_libraries(&libraries) }
            .expect("load native runtime libraries for the real ABI admission test");
    }

    let report = skippy_runtime::probe_capabilities();

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let structured_callbacks = Arc::new(AtomicUsize::new(0));
    let unload_callbacks = Arc::new(AtomicUsize::new(0));
    let structured_callbacks_sink = Arc::clone(&structured_callbacks);
    let unload_callbacks_sink = Arc::clone(&unload_callbacks);
    let installed = skippy_runtime::install_runtime_event_reporter(move |event| {
        let is_structured = matches!(
            event.kind,
            skippy_runtime::RuntimeEventKind::ModelLoadPhaseChanged
                | skippy_runtime::RuntimeEventKind::ModelLoadMemoryAllocated
                | skippy_runtime::RuntimeEventKind::ModelLoadTensorsOffloaded
                | skippy_runtime::RuntimeEventKind::ModelLoadTokenizerReady
                | skippy_runtime::RuntimeEventKind::ModelLoadAuxComponentReady
                | skippy_runtime::RuntimeEventKind::KvInitialized
                | skippy_runtime::RuntimeEventKind::KvPressureCrossed
                | skippy_runtime::RuntimeEventKind::KvPressureCleared
                | skippy_runtime::RuntimeEventKind::KvContextApproachingCapacity
                | skippy_runtime::RuntimeEventKind::KvContextCapacityExhausted
                | skippy_runtime::RuntimeEventKind::DeviceBackendInitialized
                | skippy_runtime::RuntimeEventKind::DeviceReady
                | skippy_runtime::RuntimeEventKind::DeviceDegraded
                | skippy_runtime::RuntimeEventKind::DeviceUnavailable
                | skippy_runtime::RuntimeEventKind::DeviceRecovered
                | skippy_runtime::RuntimeEventKind::DeviceLost
                | skippy_runtime::RuntimeEventKind::DeviceResourceAllocated
                | skippy_runtime::RuntimeEventKind::DeviceOutOfMemory
                | skippy_runtime::RuntimeEventKind::DeviceFallbackActivated
                | skippy_runtime::RuntimeEventKind::DiagnosticWarningRaised
                | skippy_runtime::RuntimeEventKind::DiagnosticWarningCleared
                | skippy_runtime::RuntimeEventKind::DiagnosticRecoverableFailure
                | skippy_runtime::RuntimeEventKind::DiagnosticFatalFailure
                | skippy_runtime::RuntimeEventKind::DiagnosticInvariantViolation
                | skippy_runtime::RuntimeEventKind::UnloadStarted
                | skippy_runtime::RuntimeEventKind::UnloadCompleted
                | skippy_runtime::RuntimeEventKind::UnloadFailed
                | skippy_runtime::RuntimeEventKind::UnloadForced
                | skippy_runtime::RuntimeEventKind::UnloadSessionDraining
        );
        if is_structured {
            structured_callbacks_sink.fetch_add(1, Ordering::Relaxed);
        }
        if matches!(
            event.kind,
            skippy_runtime::RuntimeEventKind::UnloadStarted
                | skippy_runtime::RuntimeEventKind::UnloadCompleted
                | skippy_runtime::RuntimeEventKind::UnloadFailed
                | skippy_runtime::RuntimeEventKind::UnloadForced
                | skippy_runtime::RuntimeEventKind::UnloadSessionDraining
        ) {
            unload_callbacks_sink.fetch_add(1, Ordering::Relaxed);
        }
    });
    assert!(
        installed,
        "runtime event reporter must install when the explicit native gate is enabled"
    );

    let config = skippy_runtime::RuntimeConfig::default();
    let model = match skippy_runtime::StageModel::open(&model_path, &config) {
        Ok(model) => model,
        Err(error) => {
            skippy_runtime::clear_runtime_event_reporter();
            panic!("real model-open failed with the reporter installed: {error}");
        }
    };

    let structured_count = structured_callbacks.load(Ordering::Relaxed);
    let unload_advertised = report.family_confirmed(skippy_ffi::FEATURE_UNLOAD_EVENTS);
    drop(model);
    let unload_count = unload_callbacks.load(Ordering::Relaxed);
    skippy_runtime::clear_runtime_event_reporter();

    assert!(
        structured_count > 0,
        "successful model-open must produce at least one structured production callback; \
         old model-open progress alone is insufficient"
    );
    if unload_advertised {
        assert!(
            unload_count > 0,
            "advertised unload-events support must produce an unload callback when the model is dropped"
        );
    }

    // Only after reporter installation, successful model-open, structured
    // production callbacks, and the unload exercise have all completed do we
    // claim that this opt-in path actually executed.
    write_marker("executed");
    write_marker(
        "exact-abi-admission: native runtime loaded (loader enforces exact major.minor.patch)",
    );
    write_marker(&format!(
        "capability-probe: confirmed={:#x} health_messages={}",
        report.confirmed,
        report.health_messages.len()
    ));
    write_marker("reporter-install: true");
    write_marker("model-open: single-part real model-open succeeded");
    write_marker(&format!(
        "structured-production-callbacks: {structured_count}"
    ));
    write_marker(&format!("unload-callbacks: {unload_count}"));
    write_marker("reporter-clear: returned");
}

/// Resolves the real installed-runtime layout: `MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR`
/// names the PARENT of one or more `<runtime-id>/{manifest.json,lib/*}`
/// subdirectories (see `dist/native-runtimes/README.md` and
/// `mesh-llm-runtime-install`'s own discovery convention), not a flat
/// directory of libraries. Prefers each candidate's own `manifest.json`
/// `runtime.libraries` ORDER — dependencies before the primary
/// `libllama.dylib` — over a lexicographic guess, since symbol-search order
/// in `skippy-ffi::dynamic::Symbols::load_paths` walks the list in reverse
/// and a naive alphabetical sort places `libllama*` before `libmtmd*`,
/// inverting the manifest's own dependency-then-primary contract.
#[cfg(feature = "dynamic-native-runtime")]
fn discover_libraries(bundle_dir: &std::path::Path) -> Vec<PathBuf> {
    if let Some(libraries) = libraries_from_flat_dir(bundle_dir) {
        return libraries;
    }
    let Ok(entries) = std::fs::read_dir(bundle_dir) else {
        return Vec::new();
    };
    let mut subdirs: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    subdirs.sort();
    for subdir in subdirs {
        if let Some(libraries) = libraries_from_manifest(&subdir) {
            return libraries;
        }
    }
    Vec::new()
}

#[cfg(feature = "dynamic-native-runtime")]
fn libraries_from_flat_dir(dir: &std::path::Path) -> Option<Vec<PathBuf>> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut libraries = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
            continue;
        };
        if matches!(extension, "dylib" | "so" | "dll") {
            libraries.push(path);
        }
    }
    if libraries.is_empty() {
        return None;
    }
    libraries.sort();
    Some(libraries)
}

#[cfg(feature = "dynamic-native-runtime")]
fn libraries_from_manifest(runtime_dir: &std::path::Path) -> Option<Vec<PathBuf>> {
    let manifest_path = runtime_dir.join("manifest.json");
    let manifest_text = std::fs::read_to_string(&manifest_path).ok()?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).ok()?;
    let entries = manifest
        .get("runtime")?
        .get("libraries")?
        .as_array()?
        .iter()
        .filter_map(|value| value.as_str());
    let libraries: Vec<PathBuf> = entries
        .map(|relative| runtime_dir.join(relative))
        .filter(|path| path.is_file())
        .collect();
    (!libraries.is_empty()).then_some(libraries)
}
