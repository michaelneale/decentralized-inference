#[cfg(feature = "dynamic-runtime")]
mod dynamic_library;

// Keep these constants in the facade: packaging and SDK tooling reads this file
// without compiling the crate to determine native-runtime compatibility.
pub const ABI_VERSION_MAJOR: u32 = 0;
pub const ABI_VERSION_MINOR: u32 = 1;
pub const ABI_VERSION_PATCH: u32 = 53;

mod abi;
mod activation;
#[cfg(feature = "dynamic-runtime")]
mod dynamic;
mod model;
mod multimodal;
mod runtime;
mod sampling;
mod stage_plan;
mod state;
#[cfg(not(feature = "dynamic-runtime"))]
mod static_bindings;

#[cfg(test)]
mod tests;

pub use abi::{
    AbiVersion, ActivationDType, ActivationLayout, BACKEND_DEVICE_CAP_ASYNC,
    BACKEND_DEVICE_CAP_BUFFER_FROM_HOST_PTR, BACKEND_DEVICE_CAP_EVENTS,
    BACKEND_DEVICE_CAP_HOST_BUFFER, BackendDevice, BackendDeviceType, Error,
    FEATURE_ACTIVATION_BOUNDARY, FEATURE_BACKEND_DEVICES, FEATURE_DEVICE_EVENTS,
    FEATURE_DIAGNOSTIC_EVENTS, FEATURE_INKLING_MTP_MM, FEATURE_ITERATION_BATCH, FEATURE_KV_EVENTS,
    FEATURE_MODEL_LOAD_EVENTS_V2, FEATURE_MODEL_SOURCE, FEATURE_NATIVE_MTP_N1,
    FEATURE_NGRAM_CACHE_DRAFT, FEATURE_RUNTIME_EVENT_REPORTER, FEATURE_RUNTIME_EVENTS,
    FEATURE_STAGE_PLAN, FEATURE_UNLOAD_EVENTS, IterationRequest, LlamaLogCallback, LoadMode,
    MODEL_TENSOR_SOURCE_V1_ABI_VERSION, Model, ModelImatrixEntryV1, ModelInfo,
    ModelReadTensorF32Callback, ModelTensorSourceV1, MtmdProgressCallback, MtpSource, NgramCache,
    Opaque, RuntimeConfig, Session, SkippyDecodeStepSampledMtpFn, SkippyModelAttachMtpDraftModelFn,
    SkippyRuntimeEventCallback, SkippyRuntimeEventCategory, SkippyRuntimeEventEmitterKind,
    SkippyRuntimeEventFailureCode, SkippyRuntimeEventKind, SkippyRuntimeEventProgressUnit,
    SkippyRuntimeEventReporterV1, SkippyRuntimeEventV1, SlicePlan, Status, TRISTATE_AUTO,
    TRISTATE_FALSE, TRISTATE_TRUE, TensorRole, runtime_abi_supported,
};
pub use activation::{
    ACTIVATION_FLAG_GEMMA3N_ALTUP, ACTIVATION_FLAG_INKLING_MTP_EMBD, ACTIVATION_SIDEBAND_TOKEN_IDS,
    ActivationBoundaryDesc, ActivationDesc, LogitBias, TensorInfo,
};
pub use model::{
    GgmlType, LlamaFileType, LlamaModelImatrixData, LlamaModelKvOverride, LlamaModelKvOverrideType,
    LlamaModelKvOverrideValue, LlamaModelQuantizeParams, LlamaModelTensorOverride,
};
pub use multimodal::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdDecoderPos, MtmdHelperBitmapWrapper,
    MtmdHelperInitOpt, MtmdHelperVideo, MtmdHelperVideoInitParams, MtmdInputChunkType,
    MtmdInputChunks, MtmdInputText,
};
pub use runtime::{
    NativeRuntimeLoadError, abi_features, llama_model_is_diffusion, llama_model_is_hybrid,
    llama_model_is_recurrent, try_abi_features,
};
pub use sampling::{
    GenerationSignalWindow, MAX_DRY_SEQUENCE_BREAKER_BYTES, MAX_DRY_SEQUENCE_BREAKERS,
    MAX_SAMPLERS, NATIVE_MTP_MAX_DRAFT_TOKENS, NativeMtpDraft, SamplingConfig, TokenSignal,
};
pub use stage_plan::{
    STAGE_PLAN_DESC_V1_ABI_VERSION, STAGE_PLAN_MAX_DIMS, STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION,
    STAGE_PLAN_STATE_DESC_V1_ABI_VERSION, STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION,
    STAGE_PLANNER_CONFIG_V1_ABI_VERSION, STAGE_PLANNER_PROFILE_V1_ABI_VERSION,
    STAGE_PLANNER_TENSOR_V1_ABI_VERSION, StagePlan, StagePlanDescV1, StagePlanProfileDescV1,
    StagePlanStateAccess, StagePlanStateDescV1, StagePlanStateKind, StagePlanStringRefV1,
    StagePlanValueDescV1, StagePlanValueKind, StagePlanner, StagePlannerConfigV1,
    StagePlannerProfileV1, StagePlannerTensorV1,
};
pub use state::{
    KV_PAGE_CODEC_ISWA_COMPOSITE_V1, KV_PAGE_CODEC_SINGLE_V1, KV_PAGE_FLAG_HAS_K_IDX,
    KV_PAGE_FLAG_V_TRANSPOSED, KvPageComponentDesc, KvPageDesc,
};

#[cfg(not(feature = "dynamic-runtime"))]
pub use runtime::{
    load_native_runtime_libraries, load_native_runtime_library, native_runtime_loaded,
    skippy_decode_step_sampled_mtp_fn, skippy_model_attach_mtp_draft_model_fn,
};

#[cfg(feature = "dynamic-runtime")]
pub use runtime::skippy_abi_features;

#[cfg(feature = "dynamic-runtime")]
pub use dynamic::{
    ggml_log_set, llama_log_set, llama_model_quantize, llama_model_quantize_default_params,
    load_native_runtime_libraries, load_native_runtime_library, mtmd_bitmap_free,
    mtmd_context_params_default, mtmd_decode_use_mrope, mtmd_default_marker, mtmd_free,
    mtmd_helper_bitmap_init_from_buf, mtmd_helper_eval_chunk_single, mtmd_helper_eval_chunks,
    mtmd_helper_get_n_pos, mtmd_helper_get_n_tokens, mtmd_helper_image_get_decoder_pos,
    mtmd_helper_init_opt_default, mtmd_helper_log_set, mtmd_helper_video_free, mtmd_init_from_file,
    mtmd_input_chunk_get_n_tokens, mtmd_input_chunk_get_tokens_image,
    mtmd_input_chunk_get_tokens_text, mtmd_input_chunk_get_type, mtmd_input_chunks_free,
    mtmd_input_chunks_get, mtmd_input_chunks_init, mtmd_input_chunks_size, mtmd_tokenize,
    native_runtime_loaded, skippy_abi_features_optional, skippy_apply_chat_template_json,
    skippy_backend_device_at, skippy_backend_device_count, skippy_decode_batch_sampled,
    skippy_decode_step_frame_batch_sampled, skippy_decode_step_frame_sampled,
    skippy_decode_step_frame_sampled_mtp, skippy_decode_step_sampled,
    skippy_decode_step_sampled_mtp, skippy_decode_step_sampled_mtp_fn, skippy_detokenize,
    skippy_error_free, skippy_export_full_state, skippy_export_kv_page,
    skippy_export_recurrent_state, skippy_export_state, skippy_import_full_state,
    skippy_import_kv_page, skippy_import_recurrent_state, skippy_import_state,
    skippy_iteration_batch_sampled, skippy_model_attach_mtp_draft_model_fn, skippy_model_free,
    skippy_model_info_free, skippy_model_info_open, skippy_model_info_tensor_at,
    skippy_model_info_tensor_count, skippy_model_input_activation_boundary,
    skippy_model_llama_model, skippy_model_open, skippy_model_open_from_parts,
    skippy_model_open_from_parts_with_events_fn, skippy_model_open_from_source,
    skippy_model_open_with_events_fn, skippy_model_output_activation_boundary,
    skippy_ngram_cache_append, skippy_ngram_cache_create, skippy_ngram_cache_draft,
    skippy_ngram_cache_free, skippy_ngram_cache_reset, skippy_parse_chat_response_json,
    skippy_prefill_chunk, skippy_prefill_chunk_frame, skippy_prefill_chunk_frame_sampled,
    skippy_prefill_chunk_frame_sampled_with_positions, skippy_prefill_chunk_frame_with_positions,
    skippy_retire_verify_checkpoint, skippy_session_batch_size,
    skippy_session_begin_external_decode, skippy_session_configure_chat_sampling,
    skippy_session_copy_output_activation_frame, skippy_session_create,
    skippy_session_create_from_resident_prefix, skippy_session_drop_sequence,
    skippy_session_end_external_decode, skippy_session_free, skippy_session_last_token_signal,
    skippy_session_llama_context, skippy_session_memory_used_cells, skippy_session_position,
    skippy_session_reset, skippy_session_restore_prefix, skippy_session_sample_current,
    skippy_session_save_prefix, skippy_session_sequence_id, skippy_session_set_position,
    skippy_session_signal_window, skippy_slice_plan_add_layer_range, skippy_slice_plan_create,
    skippy_slice_plan_free, skippy_stage_plan_describe_v1, skippy_stage_plan_free,
    skippy_stage_plan_profile_at_v1, skippy_stage_plan_resident_tensor_at_v1,
    skippy_stage_plan_state_at_v1, skippy_stage_plan_string_v1,
    skippy_stage_plan_validate_chain_v1, skippy_stage_plan_value_at_v1,
    skippy_stage_planner_create_v1, skippy_stage_planner_free, skippy_stage_planner_realize_v1,
    skippy_token_is_eog, skippy_tokenize, skippy_trim_session, skippy_verify_tokens,
    skippy_verify_tokens_frame_sampled, skippy_write_gguf_from_parts, skippy_write_slice_gguf,
};

#[cfg(feature = "dynamic-runtime")]
pub use dynamic::{
    skippy_clear_runtime_event_reporter_fn, skippy_set_runtime_event_reporter_fn, symbol_present,
};

#[cfg(not(feature = "dynamic-runtime"))]
pub use static_bindings::{
    ggml_log_set, llama_log_set, llama_model_quantize, llama_model_quantize_default_params,
    mtmd_bitmap_free, mtmd_context_params_default, mtmd_decode_use_mrope, mtmd_default_marker,
    mtmd_free, mtmd_helper_bitmap_init_from_buf, mtmd_helper_eval_chunk_single,
    mtmd_helper_eval_chunks, mtmd_helper_get_n_pos, mtmd_helper_get_n_tokens,
    mtmd_helper_image_get_decoder_pos, mtmd_helper_init_opt_default, mtmd_helper_log_set,
    mtmd_helper_video_free, mtmd_init_from_file, mtmd_input_chunk_get_n_tokens,
    mtmd_input_chunk_get_tokens_image, mtmd_input_chunk_get_tokens_text, mtmd_input_chunk_get_type,
    mtmd_input_chunks_free, mtmd_input_chunks_get, mtmd_input_chunks_init, mtmd_input_chunks_size,
    mtmd_tokenize, skippy_abi_features, skippy_apply_chat_template_json, skippy_backend_device_at,
    skippy_backend_device_count, skippy_decode_batch_sampled,
    skippy_decode_step_frame_batch_sampled, skippy_decode_step_frame_sampled,
    skippy_decode_step_frame_sampled_mtp, skippy_decode_step_sampled,
    skippy_decode_step_sampled_mtp, skippy_detokenize, skippy_error_free, skippy_export_full_state,
    skippy_export_kv_page, skippy_export_recurrent_state, skippy_export_state,
    skippy_import_full_state, skippy_import_kv_page, skippy_import_recurrent_state,
    skippy_import_state, skippy_iteration_batch_sampled, skippy_model_attach_mtp_draft_model,
    skippy_model_free, skippy_model_info_free, skippy_model_info_open, skippy_model_info_tensor_at,
    skippy_model_info_tensor_count, skippy_model_input_activation_boundary,
    skippy_model_llama_model, skippy_model_open, skippy_model_open_from_parts,
    skippy_model_open_from_source, skippy_model_output_activation_boundary,
    skippy_ngram_cache_append, skippy_ngram_cache_create, skippy_ngram_cache_draft,
    skippy_ngram_cache_free, skippy_ngram_cache_reset, skippy_parse_chat_response_json,
    skippy_prefill_chunk, skippy_prefill_chunk_frame, skippy_prefill_chunk_frame_sampled,
    skippy_prefill_chunk_frame_sampled_with_positions, skippy_prefill_chunk_frame_with_positions,
    skippy_retire_verify_checkpoint, skippy_session_batch_size,
    skippy_session_begin_external_decode, skippy_session_configure_chat_sampling,
    skippy_session_copy_output_activation_frame, skippy_session_create,
    skippy_session_create_from_resident_prefix, skippy_session_drop_sequence,
    skippy_session_end_external_decode, skippy_session_free, skippy_session_last_token_signal,
    skippy_session_llama_context, skippy_session_memory_used_cells, skippy_session_position,
    skippy_session_reset, skippy_session_restore_prefix, skippy_session_sample_current,
    skippy_session_save_prefix, skippy_session_sequence_id, skippy_session_set_position,
    skippy_session_signal_window, skippy_slice_plan_add_layer_range, skippy_slice_plan_create,
    skippy_slice_plan_free, skippy_stage_plan_describe_v1, skippy_stage_plan_free,
    skippy_stage_plan_profile_at_v1, skippy_stage_plan_resident_tensor_at_v1,
    skippy_stage_plan_state_at_v1, skippy_stage_plan_string_v1,
    skippy_stage_plan_validate_chain_v1, skippy_stage_plan_value_at_v1,
    skippy_stage_planner_create_v1, skippy_stage_planner_free, skippy_stage_planner_realize_v1,
    skippy_token_is_eog, skippy_tokenize, skippy_trim_session, skippy_verify_tokens,
    skippy_verify_tokens_frame_sampled, skippy_write_gguf_from_parts, skippy_write_slice_gguf,
};
