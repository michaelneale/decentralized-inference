use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use skippy_ffi::{
    SkippyRuntimeEventCategory as RawRuntimeEventCategory,
    SkippyRuntimeEventEmitterKind as RawRuntimeEventEmitterKind,
    SkippyRuntimeEventFailureCode as RawRuntimeEventFailureCode,
    SkippyRuntimeEventKind as RawRuntimeEventKind,
    SkippyRuntimeEventProgressUnit as RawRuntimeEventProgressUnit,
    SkippyRuntimeEventV1 as RawRuntimeEvent,
};

use super::{
    ModelOpenEventReporterRegistration, OperationId, RuntimeEvent, RuntimeEventKind, Status,
};

fn raw_event(kind: RawRuntimeEventKind, sequence: u64) -> RawRuntimeEvent {
    RawRuntimeEvent {
        abi_version: 1,
        struct_size: std::mem::size_of::<RawRuntimeEvent>() as u32,
        category: RawRuntimeEventCategory::MODEL_OPEN,
        kind,
        emitter: RawRuntimeEventEmitterKind::WORKER_THREAD,
        reserved0: 0,
        sequence,
        timestamp_mono_ns: sequence,
        model_id: 1,
        stage_id: 0,
        session_id: 0,
        progress_current: 0,
        progress_total: 0,
        progress_unit: RawRuntimeEventProgressUnit::NONE,
        failure_code: RawRuntimeEventFailureCode::NONE,
        status: Status::Ok,
        reserved1: 0,
        detail_ptr: std::ptr::null(),
        detail_len: 0,
        numeric_summary_0: 0,
        numeric_summary_1: 0,
        numeric_summary_2: 0,
        numeric_summary_3: 0,
    }
}

#[test]
fn from_raw_ptr_rejects_short_struct_before_reading_detail_fields() {
    let mut event = raw_event(RawRuntimeEventKind::MODEL_OPEN_STARTED, 1);
    // Below the BASE (pre-extension) layout, not just the full one -- see
    // the equivalent comment in runtime_events/tests.rs.
    event.struct_size = std::mem::offset_of!(RawRuntimeEvent, numeric_summary_0) as u32 - 1;
    assert!(RuntimeEvent::from_raw_ptr(&event).is_none());
}

#[test]
fn from_raw_ptr_rejects_oversized_detail_len() {
    let detail = b"x".repeat(16);
    let mut event = raw_event(RawRuntimeEventKind::MODEL_OPEN_PROGRESS, 2);
    event.detail_ptr = detail.as_ptr().cast();
    event.detail_len = u64::MAX;
    assert!(RuntimeEvent::from_raw_ptr(&event).is_none());
}

#[test]
fn from_raw_ptr_rejects_null_event() {
    assert!(RuntimeEvent::from_raw_ptr(std::ptr::null()).is_none());
}

#[test]
fn from_raw_ptr_enumerates_all_five_known_kinds() {
    let cases = [
        (
            RawRuntimeEventKind::MODEL_OPEN_STARTED,
            RuntimeEventKind::ModelOpenStarted,
        ),
        (
            RawRuntimeEventKind::MODEL_OPEN_PROGRESS,
            RuntimeEventKind::ModelOpenProgress,
        ),
        (
            RawRuntimeEventKind::BACKEND_DEVICE_SELECTED,
            RuntimeEventKind::BackendDeviceSelected,
        ),
        (
            RawRuntimeEventKind::MODEL_OPEN_FINISHED,
            RuntimeEventKind::ModelOpenFinished,
        ),
        (
            RawRuntimeEventKind::MODEL_OPEN_FAILED_HANDLED,
            RuntimeEventKind::ModelOpenFailedHandled,
        ),
    ];
    for (raw_kind, expected) in cases {
        let event = raw_event(raw_kind, 1);
        let decoded = RuntimeEvent::from_raw_ptr(&event).expect("known kind decodes");
        assert_eq!(decoded.kind, expected);
    }
}

#[test]
fn from_raw_ptr_preserves_unknown_kind_rather_than_dropping_it() {
    let event = raw_event(RawRuntimeEventKind(9999), 1);
    let decoded =
        RuntimeEvent::from_raw_ptr(&event).expect("unknown-but-well-formed event decodes");
    assert_eq!(decoded.kind, RuntimeEventKind::Unknown(9999));
}

#[test]
fn trampoline_catches_panicking_ingress_without_unwinding_across_ffi() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let later_calls = Arc::new(AtomicUsize::new(0));
    let attempts_handle = Arc::clone(&attempts);
    let later_calls_handle = Arc::clone(&later_calls);
    let registration =
        ModelOpenEventReporterRegistration::new(OperationId(1), move |_event: RuntimeEvent| {
            if attempts_handle.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("ingress deliberately panics");
            }
            later_calls_handle.fetch_add(1, Ordering::SeqCst);
        });
    let callback = registration
        .reporter_ptr()
        .cast::<super::RawRuntimeEventReporter>();
    let reporter = unsafe { &*callback };
    let callback = reporter.callback.expect("callback installed");
    let event = raw_event(RawRuntimeEventKind::MODEL_OPEN_STARTED, 1);
    // No panic should escape either call; the first panic poisons the ingress
    // mutex, but recovery in MutexIngress must preserve later callbacks.
    unsafe { callback(&event, reporter.user_data) };
    unsafe { callback(&event, reporter.user_data) };
    assert_eq!(later_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn trampoline_is_sound_under_concurrent_worker_thread_callbacks() {
    let received: Arc<Mutex<Vec<(OperationId, u64)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink_handle = Arc::clone(&received);
    let registration = ModelOpenEventReporterRegistration::new(OperationId(7), move |event| {
        sink_handle
            .lock()
            .expect("sink lock")
            .push((OperationId(7), event.sequence));
    });
    let callback_ptr = registration
        .reporter_ptr()
        .cast::<super::RawRuntimeEventReporter>();
    let reporter = unsafe { &*callback_ptr };
    let callback = reporter.callback.expect("callback installed");
    let user_data = reporter.user_data as usize;

    const PER_THREAD: u64 = 500;
    let barrier = Arc::new(Barrier::new(2));
    let mut handles = Vec::new();
    for thread_index in 0..2u64 {
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let user_data = user_data as *mut std::ffi::c_void;
            barrier.wait();
            for sequence in 0..PER_THREAD {
                let event = raw_event(
                    RawRuntimeEventKind::MODEL_OPEN_PROGRESS,
                    thread_index * PER_THREAD + sequence,
                );
                unsafe { callback(&event, user_data) };
            }
        }));
    }
    for handle in handles {
        handle.join().expect("worker thread callback loop");
    }

    let received = received.lock().expect("sink lock");
    assert_eq!(received.len(), (PER_THREAD * 2) as usize);
}

#[test]
fn ingress_reports_exact_saturation_callback_count() {
    let count = Arc::new(AtomicUsize::new(0));
    let count_handle = Arc::clone(&count);
    let registration = ModelOpenEventReporterRegistration::new(OperationId(3), move |_event| {
        count_handle.fetch_add(1, Ordering::SeqCst);
    });
    let callback_ptr = registration
        .reporter_ptr()
        .cast::<super::RawRuntimeEventReporter>();
    let reporter = unsafe { &*callback_ptr };
    let callback = reporter.callback.expect("callback installed");

    const CALLS: usize = 10_000;
    for sequence in 0..CALLS as u64 {
        let event = raw_event(RawRuntimeEventKind::MODEL_OPEN_PROGRESS, sequence);
        unsafe { callback(&event, reporter.user_data) };
    }

    assert_eq!(count.load(Ordering::SeqCst), CALLS);
}
