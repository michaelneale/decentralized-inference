pub use skippy_ffi::Status;
#[cfg(test)]
pub(crate) use skippy_ffi::TensorRole;

mod activation;
mod checkpoint;
mod config;
mod devices;
mod error;
mod gguf_writer;
mod kv_pages;
mod logging;
mod media;
mod native;
mod native_mtp;
mod ngram;
pub mod package;
mod path_cstring;
mod runtime_events;
mod session;
mod types;

pub use activation::{DecodeFrameBatchRequest, IterationBatchPhase, IterationBatchRequest};
pub use checkpoint::{CheckpointQuantization, is_safetensors_checkpoint};
pub use config::{
    FlashAttentionType, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GlmDsaPolicy,
    LLAMA_SERVER_DEFAULT_N_BATCH, LLAMA_SERVER_DEFAULT_N_UBATCH, MtpSource, RuntimeConfig,
    SKIPPY_UNIFIED_KV_DEFAULT_N_BATCH, SplitMode, parse_cache_type,
};
pub use devices::{BackendDevice, BackendDeviceType, backend_devices};
pub(crate) use error::ensure_ok;
pub use gguf_writer::{
    ModelInfo, SlicePlan, write_gguf_from_parts, write_gguf_metadata_from_parts,
};
pub use logging::{
    LLAMA_LOG_LEVEL_DEBUG, NativeLogEvent, disable_verbose_native_logs, enable_verbose_native_logs,
    redirect_native_logs_to_file, register_filtered_native_logs, restore_native_logs,
    set_filtered_native_logs_enabled, suppress_native_logs, unregister_filtered_native_logs,
    write_native_log_note,
};
pub use native::{StageModel, StageModelReader};
pub use native_mtp::NativeMtpDraft;
pub use ngram::{Cache as NgramCache, NGRAM_CACHE_MAX_NGRAM};
pub use runtime_events::{
    RuntimeEvent, RuntimeEventCategory, RuntimeEventEmitterKind, RuntimeEventFailureCode,
    RuntimeEventKind, RuntimeEventProgressUnit,
};
pub use session::{DecodeBatchRequest, StageSession};
pub use skippy_ffi::LoadMode as RuntimeLoadMode;
pub use skippy_ffi::MAX_DRY_SEQUENCE_BREAKER_BYTES;
pub use skippy_ffi::{
    ACTIVATION_FLAG_GEMMA3N_ALTUP, ACTIVATION_SIDEBAND_TOKEN_IDS,
    ActivationDType as RuntimeActivationDType, ActivationLayout as RuntimeActivationLayout,
};
// KV page descriptor flags. Re-exported so callers can read a page's layout
// without taking a direct dependency on the raw ABI crate.
pub use skippy_ffi::{KV_PAGE_FLAG_HAS_K_IDX, KV_PAGE_FLAG_V_TRANSPOSED};
pub use types::{
    ActivationBoundaryDesc, ActivationDesc, ActivationFrame, ChatReasoningFormat,
    ChatTemplateJsonOptions, ChatTemplateJsonResult, ChatTemplateMessage, ChatTemplateOptions,
    DecodeFrameBatchOutput, DrySamplingConfig, GenerationSignalWindow, IterationBatchOutput,
    IterationSample, LoadedModelCapability, LogitBias, MAX_LOGIT_BIAS, MediaInput, MediaPrefill,
    MediaPrefillChunkFrame, MediaPrefillFrame, ModelStateKind, RuntimeKvPage, RuntimeKvPageDesc,
    SamplingConfig, TensorInfo, TokenSignal, XtcSamplingConfig,
};

#[cfg(feature = "dynamic-native-runtime")]
pub use skippy_ffi::{
    NativeRuntimeLoadError, load_native_runtime_libraries, load_native_runtime_library,
    native_runtime_loaded,
};

#[cfg(test)]
use error::format_skippy_error;

#[cfg(not(feature = "dynamic-native-runtime"))]
pub fn native_runtime_loaded() -> bool {
    true
}

#[cfg(not(feature = "dynamic-native-runtime"))]
/// No-op for statically linked Skippy runtime builds.
///
/// # Safety
///
/// Static builds resolve the native ABI at process link/load time, so this
/// function does not dereference the supplied path or mutate loader state.
pub unsafe fn load_native_runtime_library(
    _path: impl AsRef<std::path::Path>,
) -> Result<(), skippy_ffi::NativeRuntimeLoadError> {
    Ok(())
}

#[cfg(not(feature = "dynamic-native-runtime"))]
/// No-op for statically linked Skippy runtime builds.
///
/// # Safety
///
/// Static builds resolve the native ABI at process link/load time, so this
/// function does not dereference the supplied paths or mutate loader state.
pub unsafe fn load_native_runtime_libraries<I, P>(
    _paths: I,
) -> Result<(), skippy_ffi::NativeRuntimeLoadError>
where
    I: IntoIterator<Item = P>,
    P: AsRef<std::path::Path>,
{
    Ok(())
}

#[cfg(test)]
include!("tests.rs");
