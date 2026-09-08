use std::ffi::{c_char, c_int, c_void};

use crate::{
    ABI_VERSION_MAJOR, ABI_VERSION_MINOR, ABI_VERSION_PATCH, NativeMtpDraft, SamplingConfig,
};

pub const FEATURE_STAGE_PLAN: u64 = 1 << 17;
pub const FEATURE_BACKEND_DEVICES: u64 = 1 << 23;
pub const FEATURE_RUNTIME_EVENTS: u64 = 1 << 24;
pub const FEATURE_NATIVE_MTP_N1: u64 = 1 << 25;
pub const FEATURE_NGRAM_CACHE_DRAFT: u64 = 1 << 26;
pub const FEATURE_INKLING_MTP_MM: u64 = 1 << 27;
pub const FEATURE_ITERATION_BATCH: u64 = 1 << 28;
pub const FEATURE_ACTIVATION_BOUNDARY: u64 = 1 << 29;
pub const FEATURE_MODEL_SOURCE: u64 = 1 << 30;
pub const FEATURE_RUNTIME_EVENT_REPORTER: u64 = 1 << 31;
pub const FEATURE_MODEL_LOAD_EVENTS_V2: u64 = 1 << 32;
pub const FEATURE_KV_EVENTS: u64 = 1 << 33;
pub const FEATURE_DEVICE_EVENTS: u64 = 1 << 34;
pub const FEATURE_DIAGNOSTIC_EVENTS: u64 = 1 << 35;
pub const FEATURE_UNLOAD_EVENTS: u64 = 1 << 36;
pub const MODEL_TENSOR_SOURCE_V1_ABI_VERSION: u32 = 1;

pub type ModelReadTensorF32Callback = Option<
    unsafe extern "C" fn(
        tensor_name: *const c_char,
        destination: *mut f32,
        element_count: usize,
        out_message: *mut *const c_char,
        user_data: *mut c_void,
    ) -> Status,
>;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ModelImatrixEntryV1 {
    pub tensor_name: *const c_char,
    pub values: *const f32,
    pub value_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ModelTensorSourceV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub read_tensor_f32: ModelReadTensorF32Callback,
    pub user_data: *mut c_void,
    pub imatrix: *const ModelImatrixEntryV1,
    pub imatrix_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IterationRequest {
    pub session: *mut Session,
    pub token_ids: *const i32,
    pub token_count: usize,
    pub positions: *const i32,
    pub position_count: usize,
    pub sampling: *const SamplingConfig,
    pub input_desc: *const crate::ActivationDesc,
    pub input_payload: *const c_void,
    pub sample_last: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

/// Whether a native runtime reporting `version` can back this binary's ABI
/// bindings. Required symbol signatures and by-value struct layouts may change
/// between patches, so the loader requires an exact ABI match.
pub const fn runtime_abi_supported(version: AbiVersion) -> bool {
    version.major == ABI_VERSION_MAJOR
        && version.minor == ABI_VERSION_MINOR
        && version.patch == ABI_VERSION_PATCH
}

pub type LlamaLogCallback =
    Option<unsafe extern "C" fn(level: c_int, text: *const c_char, user_data: *mut c_void)>;
pub type MtmdProgressCallback =
    Option<unsafe extern "C" fn(progress: f32, user_data: *mut c_void) -> bool>;
pub type SkippyRuntimeEventCallback =
    Option<unsafe extern "C" fn(event: *const SkippyRuntimeEventV1, user_data: *mut c_void)>;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SkippyRuntimeEventCategory(pub u32);

impl SkippyRuntimeEventCategory {
    pub const MODEL_OPEN: Self = Self(1);
    pub const BACKEND: Self = Self(2);
    pub const SESSION: Self = Self(3);
    pub const KV: Self = Self(4);
    pub const WARNING: Self = Self(5);
    // Numeric values already emitted by `third_party/llama.cpp/patches/
    // 0060-0062-*`; naming them here changes no layout or ABI surface.
    pub const DEVICE: Self = Self(6);
    pub const DIAGNOSTIC: Self = Self(7);
    pub const UNLOAD: Self = Self(8);
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SkippyRuntimeEventKind(pub u32);

impl SkippyRuntimeEventKind {
    pub const MODEL_OPEN_STARTED: Self = Self(1);
    pub const MODEL_OPEN_PROGRESS: Self = Self(2);
    pub const BACKEND_DEVICE_SELECTED: Self = Self(3);
    pub const MODEL_OPEN_FINISHED: Self = Self(4);
    pub const MODEL_OPEN_FAILED_HANDLED: Self = Self(5);
    // Named mirrors of the `SKIPPY_RUNTIME_EVENT_KIND_*` literals already
    // emitted by `third_party/llama.cpp/patches/0058-0062-*`.
    pub const MODEL_LOAD_PHASE_CHANGED: Self = Self(100);
    pub const MODEL_LOAD_MEMORY_ALLOCATED: Self = Self(101);
    pub const MODEL_LOAD_TENSORS_OFFLOADED: Self = Self(102);
    pub const MODEL_LOAD_TOKENIZER_READY: Self = Self(103);
    pub const MODEL_LOAD_AUX_COMPONENT_READY: Self = Self(104);
    pub const KV_INITIALIZED: Self = Self(200);
    pub const KV_PRESSURE_CROSSED: Self = Self(201);
    pub const KV_PRESSURE_CLEARED: Self = Self(202);
    pub const KV_CONTEXT_APPROACHING_CAPACITY: Self = Self(203);
    pub const KV_CONTEXT_CAPACITY_EXHAUSTED: Self = Self(204);
    pub const DEVICE_BACKEND_INITIALIZED: Self = Self(300);
    pub const DEVICE_READY: Self = Self(301);
    pub const DEVICE_DEGRADED: Self = Self(302);
    pub const DEVICE_UNAVAILABLE: Self = Self(303);
    pub const DEVICE_RECOVERED: Self = Self(304);
    pub const DEVICE_LOST: Self = Self(305);
    pub const DEVICE_RESOURCE_ALLOCATED: Self = Self(306);
    pub const DEVICE_OUT_OF_MEMORY: Self = Self(307);
    pub const DEVICE_FALLBACK_ACTIVATED: Self = Self(308);
    pub const DIAGNOSTIC_WARNING_RAISED: Self = Self(400);
    pub const DIAGNOSTIC_WARNING_CLEARED: Self = Self(401);
    pub const DIAGNOSTIC_RECOVERABLE_FAILURE: Self = Self(402);
    pub const DIAGNOSTIC_FATAL_FAILURE: Self = Self(403);
    pub const DIAGNOSTIC_INVARIANT_VIOLATION: Self = Self(404);
    pub const UNLOAD_STARTED: Self = Self(500);
    pub const UNLOAD_COMPLETED: Self = Self(501);
    pub const UNLOAD_FAILED: Self = Self(502);
    pub const UNLOAD_FORCED: Self = Self(503);
    pub const UNLOAD_SESSION_DRAINING: Self = Self(504);
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SkippyRuntimeEventEmitterKind(pub u32);

impl SkippyRuntimeEventEmitterKind {
    pub const UNKNOWN: Self = Self(0);
    pub const OPEN_THREAD: Self = Self(1);
    pub const WORKER_THREAD: Self = Self(2);
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SkippyRuntimeEventProgressUnit(pub u32);

impl SkippyRuntimeEventProgressUnit {
    pub const NONE: Self = Self(0);
    pub const BYTES: Self = Self(1);
    pub const ITEMS: Self = Self(2);
    pub const TENSORS: Self = Self(3);
    pub const STEPS: Self = Self(4);
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SkippyRuntimeEventFailureCode(pub u32);

impl SkippyRuntimeEventFailureCode {
    pub const NONE: Self = Self(0);
    pub const INVALID_ARGUMENT: Self = Self(1);
    pub const IO_ERROR: Self = Self(2);
    pub const MODEL_ERROR: Self = Self(3);
    pub const RUNTIME_ERROR: Self = Self(4);
    pub const BACKEND_ERROR: Self = Self(5);
    pub const CANCELLED: Self = Self(6);
    pub const INTERNAL_ERROR: Self = Self(7);
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SkippyRuntimeEventV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub category: SkippyRuntimeEventCategory,
    pub kind: SkippyRuntimeEventKind,
    pub emitter: SkippyRuntimeEventEmitterKind,
    pub reserved0: u32,
    pub sequence: u64,
    pub timestamp_mono_ns: u64,
    pub model_id: u64,
    pub stage_id: u64,
    pub session_id: u64,
    pub progress_current: u64,
    pub progress_total: u64,
    pub progress_unit: SkippyRuntimeEventProgressUnit,
    pub failure_code: SkippyRuntimeEventFailureCode,
    pub status: Status,
    pub reserved1: u32,
    pub detail_ptr: *const c_char,
    pub detail_len: u64,
    // Append-only extension (native ABI 0.1.45+, `SKIPPY_FEATURE_RUNTIME_EVENT_REPORTER`).
    // Advertised by `struct_size` under the unchanged
    // `SKIPPY_RUNTIME_EVENT_V1_ABI_VERSION`; a caller reads these only when
    // `struct_size` covers them (see `skippy-runtime`'s `wire_types::from_raw_ptr`).
    pub numeric_summary_0: u64,
    pub numeric_summary_1: u64,
    pub numeric_summary_2: u64,
    pub numeric_summary_3: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SkippyRuntimeEventReporterV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub callback: SkippyRuntimeEventCallback,
    pub user_data: *mut c_void,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Status {
    Ok = 0,
    Error = 1,
    InvalidArgument = 2,
    Unsupported = 3,
    BufferTooSmall = 4,
    IoError = 5,
    ModelError = 6,
    RuntimeError = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum LoadMode {
    RuntimeSlice = 0,
    LayerPackage = 1,
    ArtifactSlice = 2,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MtpSource {
    #[default]
    Disabled = 0,
    Integrated = 1,
    External = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TensorRole {
    Unknown = 0,
    Metadata = 1,
    Tokenizer = 2,
    Embedding = 3,
    Layer = 4,
    FinalNorm = 5,
    Output = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ActivationDType {
    Unknown = 0,
    F32 = 1,
    F16 = 2,
    Bf16 = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ActivationLayout {
    Opaque = 0,
    TokenMajor = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum BackendDeviceType {
    Cpu = 0,
    Gpu = 1,
    IGpu = 2,
    Accel = 3,
    Meta = 4,
}

pub const BACKEND_DEVICE_CAP_ASYNC: u64 = 1 << 0;
pub const BACKEND_DEVICE_CAP_HOST_BUFFER: u64 = 1 << 1;
pub const BACKEND_DEVICE_CAP_BUFFER_FROM_HOST_PTR: u64 = 1 << 2;
pub const BACKEND_DEVICE_CAP_EVENTS: u64 = 1 << 3;

#[repr(C)]
pub struct Error {
    pub status: Status,
    pub message: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RuntimeConfig {
    pub stage_index: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub ctx_size: i32,
    pub lane_count: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub n_gpu_layers: i32,
    pub has_mmap_override: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub cache_type_k: i32,
    pub cache_type_v: i32,
    pub flash_attn_type: i32,
    pub load_mode: LoadMode,
    pub disable_repack: bool,
    pub use_mmap_prefetch: bool,
    pub use_mmap_buffer: bool,
    pub filter_tensors_on_load: bool,
    pub resident_tensor_names: *const *const c_char,
    pub resident_tensor_name_count: usize,
    pub include_embeddings: bool,
    pub include_output: bool,
    pub mtp_source: MtpSource,
    pub selected_backend_device: *const c_char,
    pub glm_dsa_policy_profile: i32,
    pub glm_dsa_policy_flags: u32,
    pub glm_dsa_short_prefill_max_tokens: i32,
    pub glm_dsa_direct_sparse_decode_max_top_k: i32,
    pub glm_dsa_dense_sparse_mask_max_bytes: u64,
    pub glm_dsa_compact_flash_min_kv: i32,
    pub kv_offload: i32,
    pub kv_unified: i32,
    pub swa_full: i32,
    pub op_offload: i32,
    pub no_host_buffer: bool,
    pub check_tensors: bool,
    pub use_direct_io: bool,
    pub has_main_gpu_override: bool,
    pub main_gpu: i32,
    pub split_mode: i32,
}

/// Sentinel for `RuntimeConfig` tri-state fields: keep the native derived default.
pub const TRISTATE_AUTO: i32 = -1;
/// Sentinel for `RuntimeConfig` tri-state fields: force false.
pub const TRISTATE_FALSE: i32 = 0;
/// Sentinel for `RuntimeConfig` tri-state fields: force true.
pub const TRISTATE_TRUE: i32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BackendDevice {
    pub version: u32,
    pub name: *const c_char,
    pub description: *const c_char,
    pub device_id: *const c_char,
    pub memory_free: u64,
    pub memory_total: u64,
    pub device_type: BackendDeviceType,
    pub caps: u64,
}

#[repr(C)]
pub struct Model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct NgramCache {
    _private: [u8; 0],
}

#[repr(C)]
pub struct Session {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ModelInfo {
    _private: [u8; 0],
}

pub type SkippyModelAttachMtpDraftModelFn = unsafe extern "C" fn(
    target_model: *mut Model,
    path: *const c_char,
    config: *const RuntimeConfig,
    out_error: *mut *mut Error,
) -> Status;

pub type SkippyDecodeStepSampledMtpFn = unsafe extern "C" fn(
    session: *mut Session,
    token_id: i32,
    sampling: *const SamplingConfig,
    out_predicted_token: *mut i32,
    max_draft_tokens: usize,
    out_mtp_draft: *mut NativeMtpDraft,
    out_error: *mut *mut Error,
) -> Status;

#[repr(C)]
pub struct SlicePlan {
    _private: [u8; 0],
}

pub type Opaque = c_void;
