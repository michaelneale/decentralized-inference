//! Process-local runtime-event engine handle.
//!
//! Mirrors the existing `crate::logging_runtime_state()` pattern
//! (`crates/mesh-llm-host-runtime/src/lib.rs`): a replaceable global holder,
//! not a `OnceLock`, since embedded hosts and tests may start the runtime
//! more than once per process. Producers must treat an absent engine as
//! "the event system is inactive" and proceed with primary work unaffected
//! -- the accessors take only a short internal lock and never fail.

use std::sync::{Arc, LazyLock, RwLock};

use super::engine::RuntimeEventEngine;

static RUNTIME_EVENT_ENGINE: LazyLock<RwLock<Option<Arc<RuntimeEventEngine>>>> =
    LazyLock::new(|| RwLock::new(None));

/// Install the process-local runtime-event engine. Overwrites any
/// previously installed engine.
pub fn install_runtime_event_engine(engine: Arc<RuntimeEventEngine>) {
    *RUNTIME_EVENT_ENGINE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(engine);
}

/// The process-local runtime-event engine, if one has been installed.
/// `None` means the event system has not been started for this process;
/// callers must degrade to primary-work-only behavior, never fail.
#[must_use]
pub fn runtime_event_engine() -> Option<Arc<RuntimeEventEngine>> {
    RUNTIME_EVENT_ENGINE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Clear the process-local engine only when `engine` is still the exact
/// engine installed by the caller.
///
/// Runtime startup and shutdown can overlap with another embedded runtime in
/// the same process. The identity check prevents an older runtime's cleanup
/// from removing the newer runtime's engine. Callers must stop and await the
/// engine's owned driver before invoking this helper.
pub fn clear_runtime_event_engine_if_owned(engine: &Arc<RuntimeEventEngine>) -> bool {
    let mut installed = RUNTIME_EVENT_ENGINE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if installed
        .as_ref()
        .is_some_and(|current| Arc::ptr_eq(current, engine))
    {
        *installed = None;
        true
    } else {
        false
    }
}

/// Unconditionally clear the process-local runtime-event engine. Test-only;
/// production startup cleanup must use [`clear_runtime_event_engine_if_owned`]
/// so it cannot remove a newer runtime's replacement.
#[cfg(test)]
pub fn clear_runtime_event_engine() {
    *RUNTIME_EVENT_ENGINE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        clear_runtime_event_engine, clear_runtime_event_engine_if_owned,
        install_runtime_event_engine, runtime_event_engine,
    };
    use crate::runtime_events::engine::RuntimeEventEngine;

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn absent_engine_is_none() {
        clear_runtime_event_engine();
        assert!(runtime_event_engine().is_none());
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn installed_engine_is_retrievable() {
        clear_runtime_event_engine();
        install_runtime_event_engine(RuntimeEventEngine::new());
        assert!(runtime_event_engine().is_some());
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn installing_twice_replaces_the_previous_engine() {
        clear_runtime_event_engine();
        let first = RuntimeEventEngine::new();
        install_runtime_event_engine(first.clone());
        let second = RuntimeEventEngine::new();
        install_runtime_event_engine(second.clone());
        let installed = runtime_event_engine().expect("second engine installed");
        assert!(Arc::ptr_eq(&installed, &second));
        assert!(!Arc::ptr_eq(&installed, &first));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn owned_clear_does_not_remove_a_replacement_engine() {
        clear_runtime_event_engine();
        let first = RuntimeEventEngine::new();
        install_runtime_event_engine(first.clone());
        let replacement = RuntimeEventEngine::new();
        install_runtime_event_engine(replacement.clone());

        assert!(!clear_runtime_event_engine_if_owned(&first));
        let installed = runtime_event_engine().expect("replacement engine remains installed");
        assert!(Arc::ptr_eq(&installed, &replacement));

        assert!(clear_runtime_event_engine_if_owned(&replacement));
        assert!(runtime_event_engine().is_none());
    }
}
