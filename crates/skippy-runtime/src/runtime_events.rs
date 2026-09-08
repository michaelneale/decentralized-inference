use std::ffi::{c_char, c_void};
use std::mem;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use skippy_ffi::{
    Error as RawError, Model as RawModel, SkippyRuntimeEventReporterV1 as RawRuntimeEventReporter,
    SkippyRuntimeEventV1 as RawRuntimeEvent, Status,
};

mod wire_types;
pub use wire_types::{
    RuntimeEvent, RuntimeEventCategory, RuntimeEventEmitterKind, RuntimeEventFailureCode,
    RuntimeEventKind, RuntimeEventProgressUnit,
};

pub(crate) const RUNTIME_EVENT_V1_ABI_VERSION: u32 = 1;

/// Correlates every runtime event emitted during one native model-open call.
/// Callers of `open_with_events`/`open_from_parts_with_events` now supply
/// this explicitly (task 9): the boundary's shape did not change, only the
/// source moved from an internal mint to the call site, so a future
/// host-assigned identity (e.g. from `mesh-llm-host-runtime`'s runtime-event
/// engine) can be threaded in without another change here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationId(pub u64);

/// Default generator for callers with no host-assigned identity of their
/// own yet. Exposed so a caller's own id source can be swapped in later
/// without touching this crate again.
pub fn next_operation_id() -> OperationId {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    OperationId(NEXT.fetch_add(1, Ordering::Relaxed))
}

/// Sound ingress boundary for native callback threads: `Send + Sync` so a
/// shared handle can be called concurrently from more than one native worker
/// thread without any unsynchronized aliasing.
pub trait ModelOpenEventIngress: Send + Sync {
    fn submit(&self, operation_id: OperationId, event: RuntimeEvent);
}

/// Adapts an owned `FnMut` sink into a `Send + Sync` ingress by serializing
/// concurrent callback-thread access behind a mutex.
struct MutexIngress<F>(Mutex<F>);

impl<F> ModelOpenEventIngress for MutexIngress<F>
where
    F: FnMut(RuntimeEvent) + Send,
{
    fn submit(&self, _operation_id: OperationId, event: RuntimeEvent) {
        // A sink panic poisons this mutex while the FFI trampoline catches the
        // unwind. Recover the guard so one bad callback does not silently
        // disable every later event for the lifetime of this ingress.
        let mut sink = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        (sink)(event);
    }
}

pub(crate) type RawModelOpenWithEventsFn = unsafe extern "C" fn(
    path: *const c_char,
    config: *const skippy_ffi::RuntimeConfig,
    reporter: *const RawRuntimeEventReporter,
    out_model: *mut *mut RawModel,
    out_error: *mut *mut RawError,
) -> Status;

pub(crate) type RawModelOpenFromPartsWithEventsFn = unsafe extern "C" fn(
    paths: *const *const c_char,
    path_count: usize,
    config: *const skippy_ffi::RuntimeConfig,
    reporter: *const RawRuntimeEventReporter,
    out_model: *mut *mut RawModel,
    out_error: *mut *mut RawError,
) -> Status;

/// Data reachable from the native trampoline through an opaque `user_data`
/// pointer. Holds only `Copy`/`Arc` data so shared, immutable access from
/// concurrent native callback threads is sound; all mutation happens inside
/// the `Send + Sync` ingress behind its own synchronization.
struct ModelOpenEventBridge<'a> {
    operation_id: OperationId,
    ingress: Arc<dyn ModelOpenEventIngress + 'a>,
}

struct ModelOpenEventReporterRegistration<'a> {
    _bridge: Box<ModelOpenEventBridge<'a>>,
    reporter: RawRuntimeEventReporter,
}

impl<'a> ModelOpenEventReporterRegistration<'a> {
    fn new<F>(operation_id: OperationId, event_reporter: F) -> Self
    where
        F: FnMut(RuntimeEvent) + Send + 'a,
    {
        let mut bridge = Box::new(ModelOpenEventBridge {
            operation_id,
            ingress: Arc::new(MutexIngress(Mutex::new(event_reporter))),
        });
        let reporter = RawRuntimeEventReporter {
            abi_version: RUNTIME_EVENT_V1_ABI_VERSION,
            struct_size: mem::size_of::<RawRuntimeEventReporter>() as u32,
            callback: Some(model_open_event_trampoline),
            user_data: bridge.as_mut() as *mut ModelOpenEventBridge<'a> as *mut c_void,
        };
        Self {
            _bridge: bridge,
            reporter,
        }
    }

    fn reporter_ptr(&self) -> *const RawRuntimeEventReporter {
        &self.reporter
    }
}

/// Correlate-and-submit only: no formatting, logging, I/O, blocking, or
/// direct subscriber fan-out runs on this native callback thread.
unsafe extern "C" fn model_open_event_trampoline(
    event: *const RawRuntimeEvent,
    user_data: *mut c_void,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() {
            return;
        }
        let Some(event) = RuntimeEvent::from_raw_ptr(event) else {
            return;
        };
        // SAFETY: user_data was set by `ModelOpenEventReporterRegistration`
        // to a live `Box<ModelOpenEventBridge>` for the duration of the
        // registration; only a shared reference is taken, so concurrent
        // native worker-thread callbacks race only inside the ingress's own
        // synchronization, never on this pointer.
        let bridge = unsafe { &*(user_data as *const ModelOpenEventBridge<'_>) };
        bridge.ingress.submit(bridge.operation_id, event);
    }));
}

fn collect_model_open_events<OpenFn, EventFn>(
    operation_id: OperationId,
    open_fn: OpenFn,
    event_reporter: EventFn,
) -> (*mut RawModel, Status, *mut RawError)
where
    OpenFn:
        FnOnce(*const RawRuntimeEventReporter, *mut *mut RawModel, *mut *mut RawError) -> Status,
    EventFn: FnMut(RuntimeEvent) + Send,
{
    let registration = ModelOpenEventReporterRegistration::new(operation_id, event_reporter);
    let mut raw = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = open_fn(registration.reporter_ptr(), &mut raw, &mut error);
    (raw, status, error)
}

/// `operation_id` correlates every event this call emits. Task 9: the
/// caller now supplies it (see [`OperationId`]'s doc) instead of it being
/// minted inside this function.
pub(crate) fn run_model_open<OpenFn, OpenWithEventsFn>(
    operation_id: OperationId,
    open_fn: OpenFn,
    open_with_events_fn: OpenWithEventsFn,
    event_reporter: Option<&mut (dyn FnMut(RuntimeEvent) + Send)>,
    use_event_reporter: bool,
) -> (*mut RawModel, Status, *mut RawError)
where
    OpenFn: FnOnce(*mut *mut RawModel, *mut *mut RawError) -> Status,
    OpenWithEventsFn:
        FnOnce(*const RawRuntimeEventReporter, *mut *mut RawModel, *mut *mut RawError) -> Status,
{
    match (event_reporter, use_event_reporter) {
        (Some(event_reporter), true) => {
            collect_model_open_events(operation_id, open_with_events_fn, event_reporter)
        }
        _ => {
            let mut raw = ptr::null_mut();
            let mut error = ptr::null_mut();
            let status = open_fn(&mut raw, &mut error);
            (raw, status, error)
        }
    }
}

#[cfg(all(unix, not(feature = "dynamic-native-runtime")))]
fn lookup_model_open_with_events_symbol(name: &[u8]) -> Option<*mut c_void> {
    let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr().cast()) };
    (!symbol.is_null()).then_some(symbol)
}

#[cfg(all(not(unix), not(feature = "dynamic-native-runtime")))]
fn lookup_model_open_with_events_symbol(_name: &[u8]) -> Option<*mut c_void> {
    None
}

pub(crate) fn model_open_with_events_symbol() -> Option<RawModelOpenWithEventsFn> {
    static SYMBOL: OnceLock<Option<RawModelOpenWithEventsFn>> = OnceLock::new();
    *SYMBOL.get_or_init(|| {
        #[cfg(feature = "dynamic-native-runtime")]
        {
            skippy_ffi::skippy_model_open_with_events_fn()
        }
        #[cfg(not(feature = "dynamic-native-runtime"))]
        {
            lookup_model_open_with_events_symbol(b"skippy_model_open_with_events\0").map(
                |symbol| unsafe {
                    std::mem::transmute::<*mut c_void, RawModelOpenWithEventsFn>(symbol)
                },
            )
        }
    })
}

pub(crate) fn model_open_from_parts_with_events_symbol() -> Option<RawModelOpenFromPartsWithEventsFn>
{
    static SYMBOL: OnceLock<Option<RawModelOpenFromPartsWithEventsFn>> = OnceLock::new();
    *SYMBOL.get_or_init(|| {
        #[cfg(feature = "dynamic-native-runtime")]
        {
            skippy_ffi::skippy_model_open_from_parts_with_events_fn()
        }
        #[cfg(not(feature = "dynamic-native-runtime"))]
        {
            lookup_model_open_with_events_symbol(b"skippy_model_open_from_parts_with_events\0").map(
                |symbol| unsafe {
                    std::mem::transmute::<*mut c_void, RawModelOpenFromPartsWithEventsFn>(symbol)
                },
            )
        }
    })
}

// Gates purely on runtime-observable capability (native library loaded,
// feature bit advertised, `_with_events` symbols resolved) rather than a
// hardcoded ABI patch window. Exact-compatible loader probing is added by a
// later task; this function is the seam it extends.
pub(crate) fn model_open_events_supported() -> bool {
    skippy_ffi::native_runtime_loaded()
        && abi_features_bitmask()
            .is_some_and(|features| (features & skippy_ffi::FEATURE_RUNTIME_EVENTS) != 0)
        && model_open_with_events_symbol().is_some()
        && model_open_from_parts_with_events_symbol().is_some()
}

pub(crate) fn abi_features_bitmask() -> Option<u64> {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        skippy_ffi::skippy_abi_features_optional().map(|features| unsafe { features() })
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        Some(skippy_ffi::abi_features())
    }
}

#[cfg(test)]
pub(crate) mod tests;
#[cfg(test)]
mod tests_hardening;
