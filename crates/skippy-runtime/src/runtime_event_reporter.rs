use std::ffi::c_void;
use std::mem;
use std::sync::{Mutex, OnceLock, PoisonError};

use skippy_ffi::{
    FEATURE_RUNTIME_EVENT_REPORTER, SkippyRuntimeEventReporterV1 as RawReporter,
    SkippyRuntimeEventV1 as RawRuntimeEvent,
};

use crate::capability_probe::probe_capabilities;
use crate::runtime_events::{RUNTIME_EVENT_V1_ABI_VERSION, RuntimeEvent};

type ReporterSink = Box<dyn FnMut(RuntimeEvent) + Send>;

static REPORTER_SINK: OnceLock<Mutex<Option<ReporterSink>>> = OnceLock::new();
static REPORTER_LIFECYCLE: OnceLock<Mutex<()>> = OnceLock::new();

fn sink_slot() -> &'static Mutex<Option<ReporterSink>> {
    REPORTER_SINK.get_or_init(|| Mutex::new(None))
}

fn lifecycle_slot() -> &'static Mutex<()> {
    REPORTER_LIFECYCLE.get_or_init(|| Mutex::new(()))
}

fn replace_sink(sink: Option<ReporterSink>) -> Option<ReporterSink> {
    let mut guard = sink_slot().lock().unwrap_or_else(PoisonError::into_inner);
    mem::replace(&mut *guard, sink)
}

/// Correlate-and-submit only: no formatting, logging, I/O, or blocking runs
/// on this native callback thread, matching the model-open trampoline's
/// contract in `runtime_events.rs`.
unsafe extern "C" fn runtime_reporter_trampoline(
    event: *const RawRuntimeEvent,
    _user_data: *mut c_void,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(event) = RuntimeEvent::from_raw_ptr(event) else {
            return;
        };
        let mut guard = sink_slot().lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(sink) = guard.as_mut() {
            sink(event);
        }
    }));
}

type SetReporterFn = unsafe extern "C" fn(*const RawReporter) -> skippy_ffi::Status;
type ClearReporterFn = unsafe extern "C" fn();

fn set_reporter_fn() -> Option<SetReporterFn> {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        skippy_ffi::skippy_set_runtime_event_reporter_fn()
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        static CACHE: OnceLock<Option<SetReporterFn>> = OnceLock::new();
        *CACHE.get_or_init(|| {
            #[cfg(unix)]
            {
                let symbol = unsafe {
                    libc::dlsym(
                        libc::RTLD_DEFAULT,
                        c"skippy_set_runtime_event_reporter".as_ptr(),
                    )
                };
                (!symbol.is_null())
                    .then(|| unsafe { std::mem::transmute::<*mut c_void, SetReporterFn>(symbol) })
            }
            #[cfg(not(unix))]
            {
                None
            }
        })
    }
}

fn clear_reporter_fn() -> Option<ClearReporterFn> {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        skippy_ffi::skippy_clear_runtime_event_reporter_fn()
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        static CACHE: OnceLock<Option<ClearReporterFn>> = OnceLock::new();
        *CACHE.get_or_init(|| {
            #[cfg(unix)]
            {
                let symbol = unsafe {
                    libc::dlsym(
                        libc::RTLD_DEFAULT,
                        c"skippy_clear_runtime_event_reporter".as_ptr(),
                    )
                };
                (!symbol.is_null())
                    .then(|| unsafe { std::mem::transmute::<*mut c_void, ClearReporterFn>(symbol) })
            }
            #[cfg(not(unix))]
            {
                None
            }
        })
    }
}

/// Installs the runtime-scoped (process-global) event reporter, gated on the
/// probed `runtime_event_reporter` family. Returns `false` without touching
/// native state when the family is unavailable or a symbol failed to
/// resolve, so a caller can fall back cleanly on an older runtime.
pub fn install_runtime_event_reporter<F>(sink: F) -> bool
where
    F: FnMut(RuntimeEvent) + Send + 'static,
{
    if !probe_capabilities().family_confirmed(FEATURE_RUNTIME_EVENT_REPORTER) {
        return false;
    }
    let Some(set_fn) = set_reporter_fn() else {
        return false;
    };

    install_runtime_event_reporter_with_setter(Box::new(sink), set_fn)
}

fn install_runtime_event_reporter_with_setter(sink: ReporterSink, set_fn: SetReporterFn) -> bool {
    // Serialize native install/clear operations independently of the callback
    // sink lock. The native setter may wait for callbacks to drain, so holding
    // the sink lock across it would deadlock a callback that needs to return
    // through the trampoline.
    let _lifecycle = lifecycle_slot()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let previous_sink = replace_sink(Some(sink));

    let reporter = RawReporter {
        abi_version: RUNTIME_EVENT_V1_ABI_VERSION,
        struct_size: mem::size_of::<RawReporter>() as u32,
        callback: Some(runtime_reporter_trampoline),
        user_data: std::ptr::null_mut(),
    };
    let status = unsafe { set_fn(&reporter) };
    if status == skippy_ffi::Status::Ok {
        drop(previous_sink);
        true
    } else {
        // Keep the previous sink on a failed native registration. The failed
        // replacement may have been observed by a callback while the setter
        // was running, so replace it only after that call has returned.
        let failed_sink = replace_sink(previous_sink);
        drop(failed_sink);
        false
    }
}

/// Clears the runtime-scoped event reporter. Blocks (via the native
/// `skippy_clear_runtime_event_reporter` quiescence contract) until every
/// in-flight callback has returned before this function returns, so no
/// callback fires into a dropped sink. A no-op when nothing was installed.
pub fn clear_runtime_event_reporter() {
    // The native clear call provides callback quiescence. Keep the sink alive
    // until it returns, but never hold the sink mutex while native code waits.
    // Dynamic symbol lookup requires a loaded runtime; the guard keeps this
    // public cleanup function a safe no-op before dynamic startup.
    let clear_fn = if skippy_ffi::native_runtime_loaded() {
        clear_reporter_fn()
    } else {
        None
    };
    clear_runtime_event_reporter_with_clearer(clear_fn);
}

fn clear_runtime_event_reporter_with_clearer(clear_fn: Option<ClearReporterFn>) {
    let _lifecycle = lifecycle_slot()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(clear_fn) = clear_fn {
        unsafe { clear_fn() };
    }
    let removed_sink = replace_sink(None);
    drop(removed_sink);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use skippy_ffi::{
        SkippyRuntimeEventCategory as RawRuntimeEventCategory,
        SkippyRuntimeEventEmitterKind as RawRuntimeEventEmitterKind,
        SkippyRuntimeEventFailureCode as RawRuntimeEventFailureCode,
        SkippyRuntimeEventKind as RawRuntimeEventKind,
        SkippyRuntimeEventProgressUnit as RawRuntimeEventProgressUnit,
        SkippyRuntimeEventV1 as RawRuntimeEvent,
    };

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn test_guard() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn raw_event() -> RawRuntimeEvent {
        RawRuntimeEvent {
            abi_version: 1,
            struct_size: mem::size_of::<RawRuntimeEvent>() as u32,
            category: RawRuntimeEventCategory::MODEL_OPEN,
            kind: RawRuntimeEventKind::MODEL_OPEN_STARTED,
            emitter: RawRuntimeEventEmitterKind::WORKER_THREAD,
            reserved0: 0,
            sequence: 1,
            timestamp_mono_ns: 1,
            model_id: 1,
            stage_id: 0,
            session_id: 0,
            progress_current: 0,
            progress_total: 0,
            progress_unit: RawRuntimeEventProgressUnit::NONE,
            failure_code: RawRuntimeEventFailureCode::NONE,
            status: skippy_ffi::Status::Ok,
            reserved1: 0,
            detail_ptr: std::ptr::null(),
            detail_len: 0,
            numeric_summary_0: 0,
            numeric_summary_1: 0,
            numeric_summary_2: 0,
            numeric_summary_3: 0,
        }
    }

    unsafe extern "C" fn successful_setter(_reporter: *const RawReporter) -> skippy_ffi::Status {
        skippy_ffi::Status::Ok
    }

    unsafe extern "C" fn failing_setter(_reporter: *const RawReporter) -> skippy_ffi::Status {
        skippy_ffi::Status::Error
    }

    unsafe extern "C" fn observing_clearer() {
        let event = raw_event();
        unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };
    }

    #[test]
    fn install_returns_false_without_a_confirmed_family() {
        let _test_guard = test_guard();
        replace_sink(None);
        // No native runtime is loaded in unit tests, so the family probe
        // always reports unconfirmed; install must refuse cleanly rather
        // than dereference an absent native symbol.
        assert!(!install_runtime_event_reporter(|_event| {}));
    }

    #[test]
    fn clear_is_a_safe_no_op_when_nothing_was_installed() {
        let _test_guard = test_guard();
        replace_sink(None);
        clear_runtime_event_reporter();
    }

    #[test]
    fn trampoline_recovers_after_a_panicking_sink() {
        let _test_guard = test_guard();
        replace_sink(None);
        let attempts = Arc::new(AtomicUsize::new(0));
        let later_calls = Arc::new(AtomicUsize::new(0));
        let attempts_handle = Arc::clone(&attempts);
        let later_calls_handle = Arc::clone(&later_calls);
        assert!(install_runtime_event_reporter_with_setter(
            Box::new(move |_event| {
                if attempts_handle.fetch_add(1, Ordering::SeqCst) == 0 {
                    panic!("reporter sink deliberately panics");
                }
                later_calls_handle.fetch_add(1, Ordering::SeqCst);
            }),
            successful_setter,
        ));

        let event = raw_event();
        unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };
        unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };
        assert_eq!(later_calls.load(Ordering::SeqCst), 1);
        replace_sink(None);
    }

    #[test]
    fn failed_install_restores_the_previous_sink() {
        let _test_guard = test_guard();
        replace_sink(None);

        let previous_calls = Arc::new(AtomicUsize::new(0));
        let replacement_calls = Arc::new(AtomicUsize::new(0));
        let previous_calls_handle = Arc::clone(&previous_calls);
        assert!(install_runtime_event_reporter_with_setter(
            Box::new(move |_event| {
                previous_calls_handle.fetch_add(1, Ordering::SeqCst);
            }),
            successful_setter,
        ));

        let replacement_calls_handle = Arc::clone(&replacement_calls);
        assert!(!install_runtime_event_reporter_with_setter(
            Box::new(move |_event| {
                replacement_calls_handle.fetch_add(1, Ordering::SeqCst);
            }),
            failing_setter,
        ));

        let event = raw_event();
        unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };
        assert_eq!(previous_calls.load(Ordering::SeqCst), 1);
        assert_eq!(replacement_calls.load(Ordering::SeqCst), 0);
        replace_sink(None);
    }

    #[test]
    fn clear_keeps_sink_until_native_clear_returns() {
        let _test_guard = test_guard();
        replace_sink(None);
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_handle = Arc::clone(&calls);
        assert!(install_runtime_event_reporter_with_setter(
            Box::new(move |_event| {
                calls_handle.fetch_add(1, Ordering::SeqCst);
            }),
            successful_setter,
        ));

        clear_runtime_event_reporter_with_clearer(Some(observing_clearer));
        let event = raw_event();
        unsafe { runtime_reporter_trampoline(&event, std::ptr::null_mut()) };
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
