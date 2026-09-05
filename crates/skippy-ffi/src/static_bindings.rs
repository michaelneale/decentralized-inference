use std::ffi::{c_char, c_int, c_void};

use crate::{
    ActivationBoundaryDesc, ActivationDesc, BackendDevice, Error, GenerationSignalWindow,
    IterationRequest, KvPageDesc, LlamaLogCallback, LlamaModelQuantizeParams, Model, ModelInfo,
    ModelTensorSourceV1, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdDecoderPos,
    MtmdHelperBitmapWrapper, MtmdHelperInitOpt, MtmdHelperVideo, MtmdInputChunkType,
    MtmdInputChunks, MtmdInputText, NativeMtpDraft, NgramCache, Opaque, RuntimeConfig,
    SamplingConfig, Session, SlicePlan, StagePlan, StagePlanDescV1, StagePlanProfileDescV1,
    StagePlanStateDescV1, StagePlanStringRefV1, StagePlanValueDescV1, StagePlanValueKind,
    StagePlanner, StagePlannerConfigV1, Status, TensorInfo, TokenSignal,
};

unsafe extern "C" {
    pub fn llama_log_set(log_callback: LlamaLogCallback, user_data: *mut c_void);

    pub fn ggml_log_set(log_callback: LlamaLogCallback, user_data: *mut c_void);

    pub fn llama_model_quantize_default_params() -> LlamaModelQuantizeParams;

    pub fn llama_model_quantize(
        fname_inp: *const c_char,
        fname_out: *const c_char,
        params: *const LlamaModelQuantizeParams,
    ) -> u32;

    pub fn skippy_abi_features() -> u64;

    pub fn skippy_error_free(error: *mut Error);

    pub fn skippy_ngram_cache_create(
        ngram_min: u16,
        ngram_max: u16,
        out_cache: *mut *mut NgramCache,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_ngram_cache_free(cache: *mut NgramCache);

    pub fn skippy_ngram_cache_reset(
        cache: *mut NgramCache,
        token_ids: *const i32,
        token_count: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_ngram_cache_append(
        cache: *mut NgramCache,
        token_ids: *const i32,
        token_count: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_ngram_cache_draft(
        cache: *mut NgramCache,
        continuation_prefix: *const i32,
        continuation_prefix_count: usize,
        max_draft_tokens: u16,
        output_tokens: *mut i32,
        output_token_capacity: usize,
        out_token_count: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_backend_device_count(out_count: *mut usize, out_error: *mut *mut Error)
    -> Status;

    pub fn skippy_backend_device_at(
        index: usize,
        out_device: *mut BackendDevice,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_open(
        path: *const c_char,
        config: *const RuntimeConfig,
        out_model: *mut *mut Model,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_open_from_parts(
        paths: *const *const c_char,
        path_count: usize,
        config: *const RuntimeConfig,
        out_model: *mut *mut Model,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_open_from_source(
        metadata_gguf: *const c_void,
        metadata_gguf_size: usize,
        source: *const ModelTensorSourceV1,
        quantization_ftype: i32,
        config: *const RuntimeConfig,
        out_model: *mut *mut Model,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_attach_mtp_draft_model(
        target_model: *mut Model,
        path: *const c_char,
        config: *const RuntimeConfig,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_free(model: *mut Model, out_error: *mut *mut Error) -> Status;

    pub fn skippy_model_llama_model(model: *const Model) -> *const Opaque;

    pub fn llama_model_is_recurrent(model: *const Opaque) -> bool;

    pub fn llama_model_is_hybrid(model: *const Opaque) -> bool;

    pub fn llama_model_is_diffusion(model: *const Opaque) -> bool;

    pub fn skippy_model_output_activation_boundary(
        model: *const Model,
        out_desc: *mut ActivationBoundaryDesc,
    ) -> bool;

    pub fn skippy_model_input_activation_boundary(
        model: *const Model,
        out_desc: *mut ActivationBoundaryDesc,
    ) -> bool;

    pub fn skippy_session_create(
        model: *mut Model,
        out_session: *mut *mut Session,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_create_from_resident_prefix(
        model: *mut Model,
        cache_seq_id: i32,
        token_ids: *const i32,
        token_count: usize,
        out_session: *mut *mut Session,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_llama_context(session: *mut Session) -> *mut Opaque;

    pub fn skippy_session_position(session: *const Session) -> i32;

    pub fn skippy_session_batch_size(session: *const Session) -> i32;

    pub fn skippy_session_sequence_id(session: *const Session) -> i32;

    pub fn skippy_session_begin_external_decode(
        session: *mut Session,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_end_external_decode(
        session: *mut Session,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_set_position(
        session: *mut Session,
        n_past: i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_sample_current(
        session: *mut Session,
        sampling: *const SamplingConfig,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_configure_chat_sampling(
        session: *mut Session,
        sampling: *const SamplingConfig,
        metadata_json: *const c_char,
        prompt_token_count: u64,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_reset(session: *mut Session, out_error: *mut *mut Error) -> Status;

    pub fn skippy_session_free(session: *mut Session, out_error: *mut *mut Error) -> Status;

    pub fn skippy_prefill_chunk(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        input_activations: *const c_void,
        input_activation_bytes: usize,
        output_activations: *mut c_void,
        output_activation_capacity: usize,
        out_output_activation_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_verify_tokens(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        output_tokens: *mut i32,
        output_token_capacity: usize,
        out_token_count: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_decode_step_sampled(
        session: *mut Session,
        token_id: i32,
        sampling: *const SamplingConfig,
        input_activation: *const c_void,
        input_activation_bytes: usize,
        output_activation: *mut c_void,
        output_activation_capacity: usize,
        out_output_activation_bytes: *mut usize,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_decode_step_sampled_mtp(
        session: *mut Session,
        token_id: i32,
        sampling: *const SamplingConfig,
        out_predicted_token: *mut i32,
        max_draft_tokens: usize,
        out_mtp_draft: *mut NativeMtpDraft,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_decode_batch_sampled(
        sessions: *const *mut Session,
        token_ids: *const i32,
        sampling: *const *const SamplingConfig,
        request_count: usize,
        out_predicted_tokens: *mut i32,
        predicted_token_capacity: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_iteration_batch_sampled(
        requests: *const IterationRequest,
        request_count: usize,
        output_descs: *mut ActivationDesc,
        output_payloads: *const *mut c_void,
        output_payload_capacities: *const usize,
        out_output_payload_bytes: *mut usize,
        out_sampled_request_indexes: *mut usize,
        out_predicted_tokens: *mut i32,
        sampled_output_capacity: usize,
        out_sampled_output_count: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_prefill_chunk_frame(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_prefill_chunk_frame_sampled(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        sampling: *const SamplingConfig,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_prefill_chunk_frame_with_positions(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        positions: *const i32,
        position_count: usize,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_prefill_chunk_frame_sampled_with_positions(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        positions: *const i32,
        position_count: usize,
        sampling: *const SamplingConfig,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_verify_tokens_frame_sampled(
        session: *mut Session,
        token_ids: *const i32,
        token_count: usize,
        sampling: *const SamplingConfig,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        output_tokens: *mut i32,
        output_token_capacity: usize,
        out_token_count: *mut usize,
        max_draft_tokens: usize,
        out_mtp_draft: *mut NativeMtpDraft,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_decode_step_frame_sampled(
        session: *mut Session,
        token_id: i32,
        sampling: *const SamplingConfig,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_predicted_token: *mut i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_decode_step_frame_sampled_mtp(
        session: *mut Session,
        token_id: i32,
        sampling: *const SamplingConfig,
        input_desc: *const ActivationDesc,
        input_payload: *const c_void,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_predicted_token: *mut i32,
        max_draft_tokens: usize,
        out_mtp_draft: *mut NativeMtpDraft,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_decode_step_frame_batch_sampled(
        sessions: *const *mut Session,
        token_ids: *const i32,
        sampling: *const *const SamplingConfig,
        input_descs: *const *const ActivationDesc,
        input_payloads: *const *const c_void,
        output_descs: *mut ActivationDesc,
        output_payloads: *const *mut c_void,
        output_payload_capacities: *const usize,
        out_output_payload_bytes: *mut usize,
        out_predicted_tokens: *mut i32,
        predicted_token_capacity: usize,
        request_count: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_copy_output_activation_frame(
        session: *mut Session,
        token_count: usize,
        output_desc: *mut ActivationDesc,
        output_payload: *mut c_void,
        output_payload_capacity: usize,
        out_output_payload_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_last_token_signal(
        session: *mut Session,
        out_signal: *mut TokenSignal,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_signal_window(
        session: *mut Session,
        window_tokens: u32,
        out_window: *mut GenerationSignalWindow,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_trim_session(
        session: *mut Session,
        token_count: u64,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_retire_verify_checkpoint(
        session: *mut Session,
        token_start: u64,
        token_count: u64,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_export_state(
        session: *mut Session,
        layer_start: i32,
        layer_end: i32,
        output: *mut c_void,
        output_capacity: usize,
        out_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_import_state(
        session: *mut Session,
        layer_start: i32,
        layer_end: i32,
        input: *const c_void,
        input_bytes: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_export_full_state(
        session: *mut Session,
        layer_start: i32,
        layer_end: i32,
        output: *mut c_void,
        output_capacity: usize,
        out_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_import_full_state(
        session: *mut Session,
        layer_start: i32,
        layer_end: i32,
        input: *const c_void,
        input_bytes: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_export_kv_page(
        session: *mut Session,
        layer_start: i32,
        layer_end: i32,
        token_start: u64,
        token_count: u64,
        out_desc: *mut KvPageDesc,
        output: *mut c_void,
        output_capacity: usize,
        out_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_import_kv_page(
        session: *mut Session,
        desc: *const KvPageDesc,
        input: *const c_void,
        input_bytes: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_export_recurrent_state(
        session: *mut Session,
        output: *mut c_void,
        output_capacity: usize,
        out_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_import_recurrent_state(
        session: *mut Session,
        input: *const c_void,
        input_bytes: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_save_prefix(
        session: *mut Session,
        cache_seq_id: i32,
        token_count: u64,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_restore_prefix(
        session: *mut Session,
        cache_seq_id: i32,
        token_ids: *const i32,
        token_count: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_memory_used_cells(
        session: *mut Session,
        out_used_cells: *mut u64,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_session_drop_sequence(
        session: *mut Session,
        seq_id: i32,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_tokenize(
        model: *mut Model,
        text: *const c_char,
        add_special: bool,
        output_tokens: *mut i32,
        output_token_capacity: usize,
        out_token_count: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_detokenize(
        model: *mut Model,
        tokens: *const i32,
        token_count: usize,
        output_text: *mut c_char,
        output_text_capacity: usize,
        out_text_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_token_is_eog(
        model: *mut Model,
        token_id: i32,
        out_is_eog: *mut bool,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_apply_chat_template_json(
        model: *mut Model,
        messages_json: *const c_char,
        tools_json: *const c_char,
        tool_choice_json: *const c_char,
        add_assistant: bool,
        override_enable_thinking: bool,
        enable_thinking: bool,
        parallel_tool_calls: bool,
        reasoning_format: *const c_char,
        chat_template_kwargs: *const c_char,
        chat_template: *const c_char,
        use_jinja: bool,
        grammar: *const c_char,
        json_schema: *const c_char,
        skip_chat_parsing: bool,
        output_text: *mut c_char,
        output_text_capacity: usize,
        out_text_bytes: *mut usize,
        output_metadata_json: *mut c_char,
        output_metadata_json_capacity: usize,
        out_metadata_json_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_parse_chat_response_json(
        generated_text: *const c_char,
        metadata_json: *const c_char,
        is_partial: bool,
        output_message_json: *mut c_char,
        output_message_json_capacity: usize,
        out_message_json_bytes: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_info_open(
        path: *const c_char,
        out_info: *mut *mut ModelInfo,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_info_free(info: *mut ModelInfo, out_error: *mut *mut Error) -> Status;

    pub fn skippy_model_info_tensor_count(
        info: *mut ModelInfo,
        out_count: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_model_info_tensor_at(
        info: *mut ModelInfo,
        index: usize,
        out_tensor: *mut TensorInfo,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_slice_plan_create(
        info: *mut ModelInfo,
        out_plan: *mut *mut SlicePlan,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_slice_plan_free(plan: *mut SlicePlan, out_error: *mut *mut Error) -> Status;

    pub fn skippy_slice_plan_add_layer_range(
        plan: *mut SlicePlan,
        stage_index: i32,
        layer_start: i32,
        layer_end: i32,
        include_embeddings: bool,
        include_output: bool,
        include_per_layer_token_embd: bool,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_write_slice_gguf(
        info: *mut ModelInfo,
        plan: *const SlicePlan,
        stage_index: i32,
        output_path: *const c_char,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_write_gguf_from_parts(
        input_paths: *const *const c_char,
        input_count: usize,
        output_path: *const c_char,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_stage_planner_create_v1(
        config: *const StagePlannerConfigV1,
        out_planner: *mut *mut StagePlanner,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_stage_planner_free(planner: *mut StagePlanner);

    pub fn skippy_stage_planner_realize_v1(
        planner: *const StagePlanner,
        layer_start: i32,
        layer_end: i32,
        out_plan: *mut *mut StagePlan,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_stage_plan_free(plan: *mut StagePlan);

    pub fn skippy_stage_plan_describe_v1(
        plan: *const StagePlan,
        out_desc: *mut StagePlanDescV1,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_stage_plan_profile_at_v1(
        plan: *const StagePlan,
        index: usize,
        out_desc: *mut StagePlanProfileDescV1,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_stage_plan_resident_tensor_at_v1(
        plan: *const StagePlan,
        index: usize,
        out_desc: *mut StagePlanValueDescV1,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_stage_plan_value_at_v1(
        plan: *const StagePlan,
        profile_index: usize,
        kind: StagePlanValueKind,
        index: usize,
        out_desc: *mut StagePlanValueDescV1,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_stage_plan_state_at_v1(
        plan: *const StagePlan,
        profile_index: usize,
        index: usize,
        out_desc: *mut StagePlanStateDescV1,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_stage_plan_string_v1(
        plan: *const StagePlan,
        reference: StagePlanStringRefV1,
        out_data: *mut *const c_char,
        out_length: *mut usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn skippy_stage_plan_validate_chain_v1(
        plans: *const *const StagePlan,
        plan_count: usize,
        out_error: *mut *mut Error,
    ) -> Status;

    pub fn mtmd_default_marker() -> *const c_char;

    pub fn mtmd_helper_log_set(log_callback: LlamaLogCallback, user_data: *mut c_void);

    pub fn mtmd_context_params_default() -> MtmdContextParams;

    pub fn mtmd_init_from_file(
        mmproj_fname: *const c_char,
        text_model: *const Opaque,
        ctx_params: MtmdContextParams,
    ) -> *mut MtmdContext;

    pub fn mtmd_free(ctx: *mut MtmdContext);

    pub fn mtmd_helper_init_opt_default() -> MtmdHelperInitOpt;

    pub fn mtmd_helper_bitmap_init_from_buf(
        ctx: *mut MtmdContext,
        buf: *const u8,
        len: usize,
        placeholder: bool,
        opt: MtmdHelperInitOpt,
    ) -> MtmdHelperBitmapWrapper;

    pub fn mtmd_helper_video_free(video: *mut MtmdHelperVideo);

    pub fn mtmd_bitmap_free(bitmap: *mut MtmdBitmap);

    pub fn mtmd_input_chunks_init() -> *mut MtmdInputChunks;

    pub fn mtmd_input_chunks_free(chunks: *mut MtmdInputChunks);

    pub fn mtmd_tokenize(
        ctx: *mut MtmdContext,
        output: *mut MtmdInputChunks,
        text: *const MtmdInputText,
        bitmaps: *const *const MtmdBitmap,
        n_bitmaps: usize,
    ) -> c_int;

    pub fn mtmd_helper_get_n_tokens(chunks: *const MtmdInputChunks) -> usize;

    pub fn mtmd_helper_get_n_pos(chunks: *const MtmdInputChunks) -> i32;

    pub fn mtmd_input_chunks_size(chunks: *const MtmdInputChunks) -> usize;

    pub fn mtmd_input_chunks_get(chunks: *const MtmdInputChunks, index: usize) -> *const Opaque;

    pub fn mtmd_decode_use_mrope(ctx: *const MtmdContext) -> bool;

    pub fn mtmd_input_chunk_get_type(chunk: *const Opaque) -> MtmdInputChunkType;

    pub fn mtmd_input_chunk_get_n_tokens(chunk: *const Opaque) -> usize;

    pub fn mtmd_input_chunk_get_tokens_text(
        chunk: *const Opaque,
        out_count: *mut usize,
    ) -> *const i32;

    pub fn mtmd_input_chunk_get_tokens_image(chunk: *const Opaque) -> *const Opaque;

    pub fn mtmd_helper_image_get_decoder_pos(
        image: *const Opaque,
        pos_0: i32,
        out_pos: *mut MtmdDecoderPos,
    );

    pub fn mtmd_helper_eval_chunks(
        ctx: *mut MtmdContext,
        lctx: *mut Opaque,
        chunks: *const MtmdInputChunks,
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        logits_last: bool,
        new_n_past: *mut i32,
    ) -> c_int;

    pub fn mtmd_helper_eval_chunk_single(
        ctx: *mut MtmdContext,
        lctx: *mut Opaque,
        chunk: *const Opaque,
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        logits_last: bool,
        new_n_past: *mut i32,
    ) -> c_int;
}
