use std::mem;
use std::ptr;

use skippy_ffi::{
    SkippyRuntimeEventCategory as RawRuntimeEventCategory,
    SkippyRuntimeEventEmitterKind as RawRuntimeEventEmitterKind,
    SkippyRuntimeEventFailureCode as RawRuntimeEventFailureCode,
    SkippyRuntimeEventKind as RawRuntimeEventKind,
    SkippyRuntimeEventProgressUnit as RawRuntimeEventProgressUnit,
    SkippyRuntimeEventV1 as RawRuntimeEvent, Status,
};

/// Safety bound on a native event's `detail_len`. A malformed or hostile
/// value here must never drive an unbounded copy; anything past this is
/// rejected rather than trusted.
const MAX_DETAIL_BYTES: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEventCategory {
    ModelOpen,
    Backend,
    Session,
    Kv,
    Warning,
    Device,
    Diagnostic,
    Unload,
    Unknown(u32),
}

impl From<RawRuntimeEventCategory> for RuntimeEventCategory {
    fn from(value: RawRuntimeEventCategory) -> Self {
        match value {
            RawRuntimeEventCategory::MODEL_OPEN => Self::ModelOpen,
            RawRuntimeEventCategory::BACKEND => Self::Backend,
            RawRuntimeEventCategory::SESSION => Self::Session,
            RawRuntimeEventCategory::KV => Self::Kv,
            RawRuntimeEventCategory::WARNING => Self::Warning,
            RawRuntimeEventCategory::DEVICE => Self::Device,
            RawRuntimeEventCategory::DIAGNOSTIC => Self::Diagnostic,
            RawRuntimeEventCategory::UNLOAD => Self::Unload,
            RawRuntimeEventCategory(raw) => Self::Unknown(raw),
        }
    }
}

impl RuntimeEventCategory {
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::ModelOpen => RawRuntimeEventCategory::MODEL_OPEN.0,
            Self::Backend => RawRuntimeEventCategory::BACKEND.0,
            Self::Session => RawRuntimeEventCategory::SESSION.0,
            Self::Kv => RawRuntimeEventCategory::KV.0,
            Self::Warning => RawRuntimeEventCategory::WARNING.0,
            Self::Device => RawRuntimeEventCategory::DEVICE.0,
            Self::Diagnostic => RawRuntimeEventCategory::DIAGNOSTIC.0,
            Self::Unload => RawRuntimeEventCategory::UNLOAD.0,
            Self::Unknown(raw) => raw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEventKind {
    ModelOpenStarted,
    ModelOpenProgress,
    BackendDeviceSelected,
    ModelOpenFinished,
    ModelOpenFailedHandled,
    ModelLoadPhaseChanged,
    ModelLoadMemoryAllocated,
    ModelLoadTensorsOffloaded,
    ModelLoadTokenizerReady,
    ModelLoadAuxComponentReady,
    KvInitialized,
    KvPressureCrossed,
    KvPressureCleared,
    KvContextApproachingCapacity,
    KvContextCapacityExhausted,
    DeviceBackendInitialized,
    DeviceReady,
    DeviceDegraded,
    DeviceUnavailable,
    DeviceRecovered,
    DeviceLost,
    DeviceResourceAllocated,
    DeviceOutOfMemory,
    DeviceFallbackActivated,
    DiagnosticWarningRaised,
    DiagnosticWarningCleared,
    DiagnosticRecoverableFailure,
    DiagnosticFatalFailure,
    DiagnosticInvariantViolation,
    UnloadStarted,
    UnloadCompleted,
    UnloadFailed,
    UnloadForced,
    UnloadSessionDraining,
    Unknown(u32),
}

impl From<RawRuntimeEventKind> for RuntimeEventKind {
    fn from(value: RawRuntimeEventKind) -> Self {
        match value {
            RawRuntimeEventKind::MODEL_OPEN_STARTED => Self::ModelOpenStarted,
            RawRuntimeEventKind::MODEL_OPEN_PROGRESS => Self::ModelOpenProgress,
            RawRuntimeEventKind::BACKEND_DEVICE_SELECTED => Self::BackendDeviceSelected,
            RawRuntimeEventKind::MODEL_OPEN_FINISHED => Self::ModelOpenFinished,
            RawRuntimeEventKind::MODEL_OPEN_FAILED_HANDLED => Self::ModelOpenFailedHandled,
            RawRuntimeEventKind::MODEL_LOAD_PHASE_CHANGED => Self::ModelLoadPhaseChanged,
            RawRuntimeEventKind::MODEL_LOAD_MEMORY_ALLOCATED => Self::ModelLoadMemoryAllocated,
            RawRuntimeEventKind::MODEL_LOAD_TENSORS_OFFLOADED => Self::ModelLoadTensorsOffloaded,
            RawRuntimeEventKind::MODEL_LOAD_TOKENIZER_READY => Self::ModelLoadTokenizerReady,
            RawRuntimeEventKind::MODEL_LOAD_AUX_COMPONENT_READY => Self::ModelLoadAuxComponentReady,
            RawRuntimeEventKind::KV_INITIALIZED => Self::KvInitialized,
            RawRuntimeEventKind::KV_PRESSURE_CROSSED => Self::KvPressureCrossed,
            RawRuntimeEventKind::KV_PRESSURE_CLEARED => Self::KvPressureCleared,
            RawRuntimeEventKind::KV_CONTEXT_APPROACHING_CAPACITY => {
                Self::KvContextApproachingCapacity
            }
            RawRuntimeEventKind::KV_CONTEXT_CAPACITY_EXHAUSTED => Self::KvContextCapacityExhausted,
            RawRuntimeEventKind::DEVICE_BACKEND_INITIALIZED => Self::DeviceBackendInitialized,
            RawRuntimeEventKind::DEVICE_READY => Self::DeviceReady,
            RawRuntimeEventKind::DEVICE_DEGRADED => Self::DeviceDegraded,
            RawRuntimeEventKind::DEVICE_UNAVAILABLE => Self::DeviceUnavailable,
            RawRuntimeEventKind::DEVICE_RECOVERED => Self::DeviceRecovered,
            RawRuntimeEventKind::DEVICE_LOST => Self::DeviceLost,
            RawRuntimeEventKind::DEVICE_RESOURCE_ALLOCATED => Self::DeviceResourceAllocated,
            RawRuntimeEventKind::DEVICE_OUT_OF_MEMORY => Self::DeviceOutOfMemory,
            RawRuntimeEventKind::DEVICE_FALLBACK_ACTIVATED => Self::DeviceFallbackActivated,
            RawRuntimeEventKind::DIAGNOSTIC_WARNING_RAISED => Self::DiagnosticWarningRaised,
            RawRuntimeEventKind::DIAGNOSTIC_WARNING_CLEARED => Self::DiagnosticWarningCleared,
            RawRuntimeEventKind::DIAGNOSTIC_RECOVERABLE_FAILURE => {
                Self::DiagnosticRecoverableFailure
            }
            RawRuntimeEventKind::DIAGNOSTIC_FATAL_FAILURE => Self::DiagnosticFatalFailure,
            RawRuntimeEventKind::DIAGNOSTIC_INVARIANT_VIOLATION => {
                Self::DiagnosticInvariantViolation
            }
            RawRuntimeEventKind::UNLOAD_STARTED => Self::UnloadStarted,
            RawRuntimeEventKind::UNLOAD_COMPLETED => Self::UnloadCompleted,
            RawRuntimeEventKind::UNLOAD_FAILED => Self::UnloadFailed,
            RawRuntimeEventKind::UNLOAD_FORCED => Self::UnloadForced,
            RawRuntimeEventKind::UNLOAD_SESSION_DRAINING => Self::UnloadSessionDraining,
            RawRuntimeEventKind(raw) => Self::Unknown(raw),
        }
    }
}

impl RuntimeEventKind {
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::ModelOpenStarted => RawRuntimeEventKind::MODEL_OPEN_STARTED.0,
            Self::ModelOpenProgress => RawRuntimeEventKind::MODEL_OPEN_PROGRESS.0,
            Self::BackendDeviceSelected => RawRuntimeEventKind::BACKEND_DEVICE_SELECTED.0,
            Self::ModelOpenFinished => RawRuntimeEventKind::MODEL_OPEN_FINISHED.0,
            Self::ModelOpenFailedHandled => RawRuntimeEventKind::MODEL_OPEN_FAILED_HANDLED.0,
            Self::ModelLoadPhaseChanged => RawRuntimeEventKind::MODEL_LOAD_PHASE_CHANGED.0,
            Self::ModelLoadMemoryAllocated => RawRuntimeEventKind::MODEL_LOAD_MEMORY_ALLOCATED.0,
            Self::ModelLoadTensorsOffloaded => RawRuntimeEventKind::MODEL_LOAD_TENSORS_OFFLOADED.0,
            Self::ModelLoadTokenizerReady => RawRuntimeEventKind::MODEL_LOAD_TOKENIZER_READY.0,
            Self::ModelLoadAuxComponentReady => {
                RawRuntimeEventKind::MODEL_LOAD_AUX_COMPONENT_READY.0
            }
            Self::KvInitialized => RawRuntimeEventKind::KV_INITIALIZED.0,
            Self::KvPressureCrossed => RawRuntimeEventKind::KV_PRESSURE_CROSSED.0,
            Self::KvPressureCleared => RawRuntimeEventKind::KV_PRESSURE_CLEARED.0,
            Self::KvContextApproachingCapacity => {
                RawRuntimeEventKind::KV_CONTEXT_APPROACHING_CAPACITY.0
            }
            Self::KvContextCapacityExhausted => {
                RawRuntimeEventKind::KV_CONTEXT_CAPACITY_EXHAUSTED.0
            }
            Self::DeviceBackendInitialized => RawRuntimeEventKind::DEVICE_BACKEND_INITIALIZED.0,
            Self::DeviceReady => RawRuntimeEventKind::DEVICE_READY.0,
            Self::DeviceDegraded => RawRuntimeEventKind::DEVICE_DEGRADED.0,
            Self::DeviceUnavailable => RawRuntimeEventKind::DEVICE_UNAVAILABLE.0,
            Self::DeviceRecovered => RawRuntimeEventKind::DEVICE_RECOVERED.0,
            Self::DeviceLost => RawRuntimeEventKind::DEVICE_LOST.0,
            Self::DeviceResourceAllocated => RawRuntimeEventKind::DEVICE_RESOURCE_ALLOCATED.0,
            Self::DeviceOutOfMemory => RawRuntimeEventKind::DEVICE_OUT_OF_MEMORY.0,
            Self::DeviceFallbackActivated => RawRuntimeEventKind::DEVICE_FALLBACK_ACTIVATED.0,
            Self::DiagnosticWarningRaised => RawRuntimeEventKind::DIAGNOSTIC_WARNING_RAISED.0,
            Self::DiagnosticWarningCleared => RawRuntimeEventKind::DIAGNOSTIC_WARNING_CLEARED.0,
            Self::DiagnosticRecoverableFailure => {
                RawRuntimeEventKind::DIAGNOSTIC_RECOVERABLE_FAILURE.0
            }
            Self::DiagnosticFatalFailure => RawRuntimeEventKind::DIAGNOSTIC_FATAL_FAILURE.0,
            Self::DiagnosticInvariantViolation => {
                RawRuntimeEventKind::DIAGNOSTIC_INVARIANT_VIOLATION.0
            }
            Self::UnloadStarted => RawRuntimeEventKind::UNLOAD_STARTED.0,
            Self::UnloadCompleted => RawRuntimeEventKind::UNLOAD_COMPLETED.0,
            Self::UnloadFailed => RawRuntimeEventKind::UNLOAD_FAILED.0,
            Self::UnloadForced => RawRuntimeEventKind::UNLOAD_FORCED.0,
            Self::UnloadSessionDraining => RawRuntimeEventKind::UNLOAD_SESSION_DRAINING.0,
            Self::Unknown(raw) => raw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEventEmitterKind {
    Unknown,
    OpenThread,
    WorkerThread,
    Other(u32),
}

impl From<RawRuntimeEventEmitterKind> for RuntimeEventEmitterKind {
    fn from(value: RawRuntimeEventEmitterKind) -> Self {
        match value {
            RawRuntimeEventEmitterKind::UNKNOWN => Self::Unknown,
            RawRuntimeEventEmitterKind::OPEN_THREAD => Self::OpenThread,
            RawRuntimeEventEmitterKind::WORKER_THREAD => Self::WorkerThread,
            RawRuntimeEventEmitterKind(raw) => Self::Other(raw),
        }
    }
}

impl RuntimeEventEmitterKind {
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::Unknown => RawRuntimeEventEmitterKind::UNKNOWN.0,
            Self::OpenThread => RawRuntimeEventEmitterKind::OPEN_THREAD.0,
            Self::WorkerThread => RawRuntimeEventEmitterKind::WORKER_THREAD.0,
            Self::Other(raw) => raw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEventProgressUnit {
    None,
    Bytes,
    Items,
    Tensors,
    Steps,
    Unknown(u32),
}

impl From<RawRuntimeEventProgressUnit> for RuntimeEventProgressUnit {
    fn from(value: RawRuntimeEventProgressUnit) -> Self {
        match value {
            RawRuntimeEventProgressUnit::NONE => Self::None,
            RawRuntimeEventProgressUnit::BYTES => Self::Bytes,
            RawRuntimeEventProgressUnit::ITEMS => Self::Items,
            RawRuntimeEventProgressUnit::TENSORS => Self::Tensors,
            RawRuntimeEventProgressUnit::STEPS => Self::Steps,
            RawRuntimeEventProgressUnit(raw) => Self::Unknown(raw),
        }
    }
}

impl RuntimeEventProgressUnit {
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::None => RawRuntimeEventProgressUnit::NONE.0,
            Self::Bytes => RawRuntimeEventProgressUnit::BYTES.0,
            Self::Items => RawRuntimeEventProgressUnit::ITEMS.0,
            Self::Tensors => RawRuntimeEventProgressUnit::TENSORS.0,
            Self::Steps => RawRuntimeEventProgressUnit::STEPS.0,
            Self::Unknown(raw) => raw,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEventFailureCode {
    None,
    InvalidArgument,
    IoError,
    ModelError,
    RuntimeError,
    BackendError,
    Cancelled,
    InternalError,
    Unknown(u32),
}

impl From<RawRuntimeEventFailureCode> for RuntimeEventFailureCode {
    fn from(value: RawRuntimeEventFailureCode) -> Self {
        match value {
            RawRuntimeEventFailureCode::NONE => Self::None,
            RawRuntimeEventFailureCode::INVALID_ARGUMENT => Self::InvalidArgument,
            RawRuntimeEventFailureCode::IO_ERROR => Self::IoError,
            RawRuntimeEventFailureCode::MODEL_ERROR => Self::ModelError,
            RawRuntimeEventFailureCode::RUNTIME_ERROR => Self::RuntimeError,
            RawRuntimeEventFailureCode::BACKEND_ERROR => Self::BackendError,
            RawRuntimeEventFailureCode::CANCELLED => Self::Cancelled,
            RawRuntimeEventFailureCode::INTERNAL_ERROR => Self::InternalError,
            RawRuntimeEventFailureCode(raw) => Self::Unknown(raw),
        }
    }
}

impl RuntimeEventFailureCode {
    #[must_use]
    pub const fn raw(self) -> u32 {
        match self {
            Self::None => RawRuntimeEventFailureCode::NONE.0,
            Self::InvalidArgument => RawRuntimeEventFailureCode::INVALID_ARGUMENT.0,
            Self::IoError => RawRuntimeEventFailureCode::IO_ERROR.0,
            Self::ModelError => RawRuntimeEventFailureCode::MODEL_ERROR.0,
            Self::RuntimeError => RawRuntimeEventFailureCode::RUNTIME_ERROR.0,
            Self::BackendError => RawRuntimeEventFailureCode::BACKEND_ERROR.0,
            Self::Cancelled => RawRuntimeEventFailureCode::CANCELLED.0,
            Self::InternalError => RawRuntimeEventFailureCode::INTERNAL_ERROR.0,
            Self::Unknown(raw) => raw,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeEvent {
    pub abi_version: u32,
    pub struct_size: u32,
    pub category: RuntimeEventCategory,
    pub kind: RuntimeEventKind,
    pub emitter: RuntimeEventEmitterKind,
    pub reserved0: u32,
    pub sequence: u64,
    pub timestamp_mono_ns: u64,
    pub model_id: u64,
    pub stage_id: u64,
    pub session_id: u64,
    pub progress_current: u64,
    pub progress_total: u64,
    pub progress_unit: RuntimeEventProgressUnit,
    pub failure_code: RuntimeEventFailureCode,
    pub status: Status,
    pub reserved1: u32,
    pub detail_bytes: Vec<u8>,
    // `None` when the emitting native runtime's `struct_size` did not cover
    // these append-only fields (native ABI 0.1.44 or earlier, or a build
    // without SKIPPY_FEATURE_RUNTIME_EVENT_REPORTER) -- never read as zero.
    pub numeric_summary_0: Option<u64>,
    pub numeric_summary_1: Option<u64>,
    pub numeric_summary_2: Option<u64>,
    pub numeric_summary_3: Option<u64>,
}

/// Mirrors `RawRuntimeEvent`'s original (pre-extension) prefix exactly, field
/// for field. Used only to bound how much of the allocation is safe to read
/// as a full struct reference when `struct_size` covers the base layout but
/// not the four appended `numeric_summary_*` fields: casting to this type's
/// reference reads only the common initial sequence, never past what
/// `struct_size` proved is allocated.
#[repr(C)]
struct BaseRawRuntimeEvent {
    abi_version: u32,
    struct_size: u32,
    category: RawRuntimeEventCategory,
    kind: RawRuntimeEventKind,
    emitter: RawRuntimeEventEmitterKind,
    reserved0: u32,
    sequence: u64,
    timestamp_mono_ns: u64,
    model_id: u64,
    stage_id: u64,
    session_id: u64,
    progress_current: u64,
    progress_total: u64,
    progress_unit: RawRuntimeEventProgressUnit,
    failure_code: RawRuntimeEventFailureCode,
    status: Status,
    reserved1: u32,
    detail_ptr: *const std::ffi::c_char,
    detail_len: u64,
}

macro_rules! const_assert_eq {
    ($left:expr, $right:expr $(,)?) => {
        const _: () = assert!($left == $right);
    };
}

const_assert_eq!(
    mem::align_of::<BaseRawRuntimeEvent>(),
    mem::align_of::<RawRuntimeEvent>(),
);
const_assert_eq!(
    mem::size_of::<BaseRawRuntimeEvent>(),
    mem::offset_of!(RawRuntimeEvent, numeric_summary_0),
);

macro_rules! assert_shared_field_offsets {
    ($($field:ident),+ $(,)?) => {
        $(
            const_assert_eq!(
                mem::offset_of!(BaseRawRuntimeEvent, $field),
                mem::offset_of!(RawRuntimeEvent, $field),
            );
        )+
    };
}

assert_shared_field_offsets!(
    abi_version,
    struct_size,
    category,
    kind,
    emitter,
    reserved0,
    sequence,
    timestamp_mono_ns,
    model_id,
    stage_id,
    session_id,
    progress_current,
    progress_total,
    progress_unit,
    failure_code,
    status,
    reserved1,
    detail_ptr,
    detail_len,
);

impl RuntimeEvent {
    pub(crate) fn from_raw_ptr(event: *const RawRuntimeEvent) -> Option<Self> {
        if event.is_null() {
            return None;
        }
        // SAFETY: prefix-validate before any other field read. The
        // versioned-struct ABI contract guarantees every allocation covers
        // at least `struct_size`'s own offset; we read only that field via
        // `read_unaligned` and refuse to form a full struct reference until
        // it proves the allocation covers at least the base (pre-extension)
        // known layout.
        let struct_size = unsafe { ptr::read_unaligned(ptr::addr_of!((*event).struct_size)) };
        let base_size = mem::size_of::<BaseRawRuntimeEvent>();
        if (struct_size as usize) < base_size {
            return None;
        }
        let covers_extension = (struct_size as usize) >= mem::size_of::<RawRuntimeEvent>();

        // SAFETY: struct_size was just validated to cover at least
        // `base_size`, and `BaseRawRuntimeEvent` is `repr(C)` with the exact
        // same field prefix as `RawRuntimeEvent` (the C "common initial
        // sequence" pattern) -- reading through this narrower reference
        // never touches bytes past what struct_size proved is allocated,
        // regardless of whether the extension fields exist.
        let base = unsafe { &*event.cast::<BaseRawRuntimeEvent>() };
        let detail_len = usize::try_from(base.detail_len).ok()?;
        if detail_len > MAX_DETAIL_BYTES {
            return None;
        }
        let detail_bytes = if detail_len == 0 || base.detail_ptr.is_null() {
            Vec::new()
        } else {
            // SAFETY: detail_len is bound-checked above and detail_ptr is
            // non-null; the reporter contract guarantees this byte range is
            // valid and immutable for the callback's duration.
            unsafe { std::slice::from_raw_parts(base.detail_ptr.cast::<u8>(), detail_len) }.to_vec()
        };

        let (numeric_summary_0, numeric_summary_1, numeric_summary_2, numeric_summary_3) =
            if covers_extension {
                // SAFETY: struct_size covers the full extended layout, so
                // the caller's ABI contract guarantees this range is
                // initialized and in-bounds.
                let full = unsafe { &*event };
                (
                    Some(full.numeric_summary_0),
                    Some(full.numeric_summary_1),
                    Some(full.numeric_summary_2),
                    Some(full.numeric_summary_3),
                )
            } else {
                (None, None, None, None)
            };

        Some(Self {
            abi_version: base.abi_version,
            struct_size,
            category: base.category.into(),
            kind: base.kind.into(),
            emitter: base.emitter.into(),
            reserved0: base.reserved0,
            sequence: base.sequence,
            timestamp_mono_ns: base.timestamp_mono_ns,
            model_id: base.model_id,
            stage_id: base.stage_id,
            session_id: base.session_id,
            progress_current: base.progress_current,
            progress_total: base.progress_total,
            progress_unit: base.progress_unit.into(),
            failure_code: base.failure_code.into(),
            status: base.status,
            reserved1: base.reserved1,
            detail_bytes,
            numeric_summary_0,
            numeric_summary_1,
            numeric_summary_2,
            numeric_summary_3,
        })
    }
}
