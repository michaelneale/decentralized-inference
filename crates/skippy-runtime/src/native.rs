use std::ffi::CString;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use skippy_ffi::Model as RawModel;

use crate::error::{ensure_ok, free_error};
use crate::logging::write_native_log_note;
use crate::media::MediaProjector;
use crate::path_cstring::path_to_cstring;
use crate::runtime_events;
use crate::session::StageSession;
use crate::{
    ActivationBoundaryDesc, ChatReasoningFormat, ChatTemplateJsonOptions, ChatTemplateJsonResult,
    ChatTemplateMessage, ChatTemplateOptions, LoadedModelCapability, ModelStateKind, RuntimeConfig,
    RuntimeEvent, Status,
};

pub struct StageModel {
    inner: Arc<StageModelInner>,
    pub(crate) media: Option<MediaProjector>,
}

struct StageModelInner {
    raw: *mut RawModel,
    include_output: bool,
    capability: Option<LoadedModelCapability>,
}

/// A read-only model handle for vocabulary operations that do not touch a
/// session or its mutable inference context.
#[derive(Clone)]
pub struct StageModelReader {
    inner: Arc<StageModelInner>,
}

// The native model and vocabulary are immutable after loading. Session/context
// mutation is owned by separate handles and remains externally serialized.
unsafe impl Send for StageModelInner {}
unsafe impl Sync for StageModelInner {}

// The experimental C ABI owns synchronization internally for model/session use.
// Rust stage-server access is additionally serialized behind a Mutex.
unsafe impl Send for StageModel {}

fn classify_model_state(recurrent: bool, hybrid: bool, diffusion: bool) -> ModelStateKind {
    if diffusion {
        ModelStateKind::Diffusion
    } else if hybrid {
        ModelStateKind::Hybrid
    } else if recurrent {
        ModelStateKind::Recurrent
    } else {
        ModelStateKind::Dense
    }
}

fn capability_from_state_probes(
    recurrent: Option<bool>,
    hybrid: Option<bool>,
    diffusion: Option<bool>,
) -> Option<LoadedModelCapability> {
    Some(LoadedModelCapability {
        state_kind: classify_model_state(recurrent?, hybrid?, diffusion?),
    })
}

fn loaded_model_capability(raw: *mut RawModel) -> Option<LoadedModelCapability> {
    let model = unsafe { skippy_ffi::skippy_model_llama_model(raw) };
    if model.is_null() {
        return None;
    }
    capability_from_state_probes(
        unsafe { skippy_ffi::llama_model_is_recurrent(model) },
        unsafe { skippy_ffi::llama_model_is_hybrid(model) },
        unsafe { skippy_ffi::llama_model_is_diffusion(model) },
    )
}

impl StageModel {
    pub fn new_dummy() -> Self {
        Self {
            inner: Arc::new(StageModelInner {
                raw: std::ptr::null_mut(),
                include_output: true,
                capability: None,
            }),
            media: None,
        }
    }

    pub fn output_activation_boundary(&self) -> Option<ActivationBoundaryDesc> {
        let mut raw = skippy_ffi::ActivationBoundaryDesc::default();
        let present = unsafe {
            skippy_ffi::skippy_model_output_activation_boundary(self.inner.raw, &mut raw)
        };
        present.then(|| raw.into())
    }

    pub fn input_activation_boundary(&self) -> Option<ActivationBoundaryDesc> {
        let mut raw = skippy_ffi::ActivationBoundaryDesc::default();
        let present =
            unsafe { skippy_ffi::skippy_model_input_activation_boundary(self.inner.raw, &mut raw) };
        present.then(|| raw.into())
    }

    fn from_opened_raw(
        raw: *mut RawModel,
        config: &RuntimeConfig,
        null_handle_message: &'static str,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(anyhow!(null_handle_message));
        }
        let capability = loaded_model_capability(raw);
        let media = config
            .projector_path
            .as_deref()
            .map(|projector_path| MediaProjector::open(projector_path, raw, config))
            .transpose()?;
        Ok(Self {
            inner: Arc::new(StageModelInner {
                raw,
                include_output: config.include_output,
                capability,
            }),
            media,
        })
    }

    fn open_path_with_optional_event_reporter(
        path: impl AsRef<Path>,
        config: &RuntimeConfig,
        operation_id: runtime_events::OperationId,
        event_reporter: Option<&mut (dyn FnMut(RuntimeEvent) + Send)>,
    ) -> Result<Self> {
        let path = path.as_ref();
        if crate::checkpoint::is_safetensors_checkpoint(path) {
            if event_reporter.is_some() {
                write_native_log_note(
                    "SafeTensors source loading does not yet emit native model-open events",
                );
            }
            return Self::open_safetensors(path, config.checkpoint_quantization, config);
        }
        let use_events = event_reporter.is_some() && runtime_events::model_open_events_supported();
        let begin_label = if use_events {
            "skippy_model_open_with_events begin"
        } else {
            "skippy_model_open begin"
        };
        let end_label = if use_events {
            "skippy_model_open_with_events returned"
        } else {
            "skippy_model_open returned"
        };
        let null_handle_message = if use_events {
            "skippy_model_open_with_events returned a null handle"
        } else {
            "skippy_model_open returned a null handle"
        };
        write_native_log_note(format!(
            "{begin_label} path={} {}",
            path.display(),
            config.native_log_summary()
        ));
        let path = path_to_cstring(path, "model path")?;
        let raw_config = config.as_raw()?;
        #[cfg(not(test))]
        let (raw, status, error) = runtime_events::run_model_open(
            operation_id,
            |out_model, out_error| unsafe {
                skippy_ffi::skippy_model_open(path.as_ptr(), &raw_config.raw, out_model, out_error)
            },
            |reporter, out_model, out_error| unsafe {
                let open_with_events_symbol = runtime_events::model_open_with_events_symbol()
                    .expect("runtime-event symbol availability checked before use");
                open_with_events_symbol(
                    path.as_ptr(),
                    &raw_config.raw,
                    reporter,
                    out_model,
                    out_error,
                )
            },
            event_reporter,
            use_events,
        );
        #[cfg(test)]
        let (raw, status, error) = {
            debug_assert!(event_reporter.is_none());
            runtime_events::run_model_open(
                operation_id,
                |out_model, out_error| unsafe {
                    skippy_ffi::skippy_model_open(
                        path.as_ptr(),
                        &raw_config.raw,
                        out_model,
                        out_error,
                    )
                },
                |_reporter, _out_model, _out_error| {
                    unreachable!("test builds do not link _with_events model-open symbols")
                },
                None,
                false,
            )
        };
        write_native_log_note(format!("{end_label} status={status:?}"));
        ensure_ok(status, error)?;
        Self::from_opened_raw(raw, config, null_handle_message)
    }

    fn open_parts_with_optional_event_reporter(
        paths: &[impl AsRef<Path>],
        config: &RuntimeConfig,
        operation_id: runtime_events::OperationId,
        event_reporter: Option<&mut (dyn FnMut(RuntimeEvent) + Send)>,
    ) -> Result<Self> {
        if paths.is_empty() {
            return Err(anyhow!("at least one GGUF part path is required"));
        }
        let use_events = event_reporter.is_some() && runtime_events::model_open_events_supported();
        let begin_label = if use_events {
            "skippy_model_open_from_parts_with_events begin"
        } else {
            "skippy_model_open_from_parts begin"
        };
        let end_label = if use_events {
            "skippy_model_open_from_parts_with_events returned"
        } else {
            "skippy_model_open_from_parts returned"
        };
        let null_handle_message = if use_events {
            "skippy_model_open_from_parts_with_events returned a null handle"
        } else {
            "skippy_model_open_from_parts returned a null handle"
        };
        let path_list = paths
            .iter()
            .map(|path| path.as_ref().display().to_string())
            .collect::<Vec<_>>()
            .join(",");
        write_native_log_note(format!(
            "{begin_label} parts={} {}",
            path_list,
            config.native_log_summary()
        ));
        let paths = paths
            .iter()
            .map(|path| path_to_cstring(path.as_ref(), "part path"))
            .collect::<Result<Vec<_>>>()?;
        let path_ptrs = paths.iter().map(|path| path.as_ptr()).collect::<Vec<_>>();
        let raw_config = config.as_raw()?;
        #[cfg(not(test))]
        let (raw, status, error) = runtime_events::run_model_open(
            operation_id,
            |out_model, out_error| unsafe {
                skippy_ffi::skippy_model_open_from_parts(
                    path_ptrs.as_ptr(),
                    path_ptrs.len(),
                    &raw_config.raw,
                    out_model,
                    out_error,
                )
            },
            |reporter, out_model, out_error| unsafe {
                let open_from_parts_with_events_symbol =
                    runtime_events::model_open_from_parts_with_events_symbol()
                        .expect("runtime-event symbol availability checked before use");
                open_from_parts_with_events_symbol(
                    path_ptrs.as_ptr(),
                    path_ptrs.len(),
                    &raw_config.raw,
                    reporter,
                    out_model,
                    out_error,
                )
            },
            event_reporter,
            use_events,
        );
        #[cfg(test)]
        let (raw, status, error) = {
            debug_assert!(event_reporter.is_none());
            runtime_events::run_model_open(
                operation_id,
                |out_model, out_error| unsafe {
                    skippy_ffi::skippy_model_open_from_parts(
                        path_ptrs.as_ptr(),
                        path_ptrs.len(),
                        &raw_config.raw,
                        out_model,
                        out_error,
                    )
                },
                |_reporter, _out_model, _out_error| {
                    unreachable!(
                        "test builds do not link _with_events model-open-from-parts symbols"
                    )
                },
                None,
                false,
            )
        };
        write_native_log_note(format!("{end_label} status={status:?}"));
        ensure_ok(status, error)?;
        Self::from_opened_raw(raw, config, null_handle_message)
    }

    pub fn open(path: impl AsRef<Path>, config: &RuntimeConfig) -> Result<Self> {
        Self::open_path_with_optional_event_reporter(
            path,
            config,
            runtime_events::next_operation_id(),
            None,
        )
    }

    /// Opens an official Hugging Face SafeTensors checkpoint without writing an
    /// intermediate GGUF. Quantization, when requested, happens per tensor as
    /// llama.cpp allocates the model's destination buffers.
    pub fn open_safetensors(
        source: impl AsRef<Path>,
        quantization: crate::CheckpointQuantization,
        config: &RuntimeConfig,
    ) -> Result<Self> {
        let raw = crate::checkpoint::open_safetensors(source.as_ref(), quantization, config)?;
        Self::from_opened_raw(
            raw,
            config,
            "skippy_model_open_from_source returned a null handle",
        )
    }

    /// `operation_id` correlates every event this call emits with its
    /// caller's own operation identity (task 9: caller-supplied, not minted
    /// inside this crate -- see [`runtime_events::OperationId`]'s doc).
    pub fn open_with_events(
        path: impl AsRef<Path>,
        config: &RuntimeConfig,
        operation_id: runtime_events::OperationId,
        event_reporter: &mut (dyn FnMut(RuntimeEvent) + Send),
    ) -> Result<Self> {
        #[cfg(test)]
        {
            let _ = event_reporter;
            Self::open_path_with_optional_event_reporter(path, config, operation_id, None)
        }

        #[cfg(not(test))]
        Self::open_path_with_optional_event_reporter(
            path,
            config,
            operation_id,
            Some(event_reporter),
        )
    }

    pub fn open_from_parts(paths: &[impl AsRef<Path>], config: &RuntimeConfig) -> Result<Self> {
        Self::open_parts_with_optional_event_reporter(
            paths,
            config,
            runtime_events::next_operation_id(),
            None,
        )
    }

    /// See [`Self::open_with_events`]'s `operation_id` doc.
    pub fn open_from_parts_with_events(
        paths: &[impl AsRef<Path>],
        config: &RuntimeConfig,
        operation_id: runtime_events::OperationId,
        event_reporter: &mut (dyn FnMut(RuntimeEvent) + Send),
    ) -> Result<Self> {
        #[cfg(test)]
        {
            let _ = event_reporter;
            Self::open_parts_with_optional_event_reporter(paths, config, operation_id, None)
        }

        #[cfg(not(test))]
        Self::open_parts_with_optional_event_reporter(
            paths,
            config,
            operation_id,
            Some(event_reporter),
        )
    }

    pub fn attach_mtp_draft_model(
        &mut self,
        path: impl AsRef<Path>,
        config: &RuntimeConfig,
    ) -> Result<()> {
        let inner = Arc::get_mut(&mut self.inner).ok_or_else(|| {
            anyhow!("cannot attach MTP draft model while model readers are active")
        })?;
        if inner.raw.is_null() {
            return Err(anyhow!("cannot attach MTP draft model to a null model"));
        }
        let attach_symbol = skippy_ffi::skippy_model_attach_mtp_draft_model_fn()
            .ok_or_else(|| anyhow!("native runtime does not support external MTP draft models"))?;
        let path = path.as_ref();
        write_native_log_note(format!(
            "skippy_model_attach_mtp_draft_model begin path={} {}",
            path.display(),
            config.native_log_summary()
        ));
        let path = path_to_cstring(path, "MTP draft model path")?;
        let raw_config = config.as_raw()?;
        let mut error = ptr::null_mut();
        let status =
            unsafe { attach_symbol(inner.raw, path.as_ptr(), &raw_config.raw, &mut error) };
        write_native_log_note(format!(
            "skippy_model_attach_mtp_draft_model returned status={status:?}"
        ));
        ensure_ok(status, error)
    }

    pub fn create_session(&self) -> Result<StageSession> {
        write_native_log_note("skippy_session_create begin");
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_session_create(self.inner.raw, &mut raw, &mut error) };
        write_native_log_note(format!("skippy_session_create returned status={status:?}"));
        ensure_ok(status, error)?;
        if raw.is_null() {
            return Err(anyhow!("skippy_session_create returned a null handle"));
        }
        Ok(StageSession {
            raw,
            token_count: 0,
            include_output: self.inner.include_output,
        })
    }

    pub fn create_session_from_resident_prefix(
        &self,
        cache_seq_id: i32,
        token_ids: &[i32],
    ) -> Result<StageSession> {
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_create_from_resident_prefix(
                self.inner.raw,
                cache_seq_id,
                token_ids.as_ptr(),
                token_ids.len(),
                &mut raw,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        if raw.is_null() {
            return Err(anyhow!(
                "skippy_session_create_from_resident_prefix returned a null handle"
            ));
        }
        Ok(StageSession {
            raw,
            token_count: u64::try_from(token_ids.len()).context("token count exceeds u64")?,
            include_output: self.inner.include_output,
        })
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        tokenize(self.inner.raw, text, add_special)
    }

    /// Tokenize without allocating a token buffer larger than `max_tokens`.
    ///
    /// The common case uses one native call with an optimistic buffer. If the
    /// tokenizer needs more space, the ABI reports the exact required count;
    /// counts above the bound return `Ok(None)` before the retry allocation.
    pub fn tokenize_bounded(
        &self,
        text: &str,
        add_special: bool,
        max_tokens: usize,
    ) -> Result<Option<Vec<i32>>> {
        tokenize_bounded(self.inner.raw, text, add_special, max_tokens)
    }

    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.detokenize_bytes(tokens)?).into_owned())
    }

    pub fn detokenize_bytes(&self, tokens: &[i32]) -> Result<Vec<u8>> {
        detokenize_bytes(self.inner.raw, tokens)
    }

    pub fn token_is_eog(&self, token: i32) -> Result<bool> {
        token_is_eog(self.inner.raw, token)
    }

    pub fn reader(&self) -> StageModelReader {
        StageModelReader {
            inner: Arc::clone(&self.inner),
        }
    }

    pub fn capability(&self) -> Option<&LoadedModelCapability> {
        self.inner.capability.as_ref()
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatTemplateMessage],
        add_assistant: bool,
    ) -> Result<String> {
        self.apply_chat_template_with_options(
            messages,
            ChatTemplateOptions {
                add_assistant,
                enable_thinking: None,
                reasoning_format: None,
                ..ChatTemplateOptions::default()
            },
        )
    }

    pub fn apply_chat_template_with_options(
        &self,
        messages: &[ChatTemplateMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        let messages_json = serde_json::to_string(
            &messages
                .iter()
                .map(|message| {
                    serde_json::json!({
                        "role": message.role,
                        "content": message.content,
                    })
                })
                .collect::<Vec<_>>(),
        )?;
        let rendered = self.apply_chat_template_json(
            &messages_json,
            ChatTemplateJsonOptions {
                add_assistant: options.add_assistant,
                enable_thinking: options.enable_thinking,
                reasoning_format: options.reasoning_format,
                chat_template_kwargs: options.chat_template_kwargs,
                chat_template: options.chat_template,
                use_jinja: options.use_jinja,
                grammar: options.grammar,
                json_schema: options.json_schema,
                skip_chat_parsing: options.skip_chat_parsing,
                ..ChatTemplateJsonOptions::default()
            },
        )?;
        Ok(rendered.prompt)
    }

    pub fn apply_chat_template_json(
        &self,
        messages_json: &str,
        options: ChatTemplateJsonOptions,
    ) -> Result<ChatTemplateJsonResult> {
        apply_chat_template_json(self.inner.raw, messages_json, options)
    }

    pub fn parse_chat_response_json(
        &self,
        generated_text: &str,
        metadata_json: &str,
        is_partial: bool,
    ) -> Result<String> {
        parse_chat_response_json_native(generated_text, metadata_json, is_partial)
    }
}

fn parse_chat_response_json_native(
    generated_text: &str,
    metadata_json: &str,
    is_partial: bool,
) -> Result<String> {
    let initial_capacity =
        optimistic_chat_parse_capacity(generated_text.len(), metadata_json.len());
    let generated_text =
        CString::new(generated_text).context("generated text contains an interior NUL byte")?;
    let metadata_json = CString::new(metadata_json)
        .context("chat template metadata contains an interior NUL byte")?;

    let mut output = vec![0_u8; initial_capacity];
    let mut bytes = output.len();
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_parse_chat_response_json(
            generated_text.as_ptr(),
            metadata_json.as_ptr(),
            is_partial,
            output.as_mut_ptr().cast(),
            output.len(),
            &mut bytes,
            &mut error,
        )
    };
    if status == Status::Ok {
        ensure_ok(status, error)?;
        output.truncate(bytes);
        return String::from_utf8(output).context("parsed chat response is not valid UTF-8");
    }
    if status != Status::BufferTooSmall {
        ensure_ok(status, error)?;
    }
    free_error(error);

    output.resize(bytes.max(1), 0);
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_parse_chat_response_json(
            generated_text.as_ptr(),
            metadata_json.as_ptr(),
            is_partial,
            output.as_mut_ptr().cast(),
            output.len(),
            &mut bytes,
            &mut error,
        )
    };
    ensure_ok(status, error)?;
    output.truncate(bytes);
    String::from_utf8(output).context("parsed chat response is not valid UTF-8")
}

const OPTIMISTIC_OUTPUT_HEADROOM: usize = 4 * 1024;

fn optimistic_token_capacity(text_bytes: usize, max_tokens: usize) -> usize {
    text_bytes.div_ceil(2).saturating_add(8).min(max_tokens)
}

fn optimistic_chat_prompt_capacity(
    messages_bytes: usize,
    tools_bytes: usize,
    kwargs_bytes: usize,
) -> usize {
    messages_bytes
        .saturating_add(tools_bytes)
        .saturating_add(kwargs_bytes)
        .saturating_add(OPTIMISTIC_OUTPUT_HEADROOM)
}

fn optimistic_chat_metadata_capacity(tools_bytes: usize, kwargs_bytes: usize) -> usize {
    tools_bytes
        .saturating_mul(2)
        .saturating_add(kwargs_bytes)
        .saturating_add(OPTIMISTIC_OUTPUT_HEADROOM)
}

fn optimistic_chat_parse_capacity(generated_bytes: usize, metadata_bytes: usize) -> usize {
    generated_bytes
        .saturating_add(metadata_bytes)
        .saturating_add(OPTIMISTIC_OUTPUT_HEADROOM)
}

fn chat_template_json_result(prompt: Vec<u8>, metadata: Vec<u8>) -> Result<ChatTemplateJsonResult> {
    Ok(ChatTemplateJsonResult {
        prompt: String::from_utf8(prompt).context("chat template output is not valid UTF-8")?,
        metadata_json: String::from_utf8(metadata)
            .context("chat template metadata is not valid UTF-8")?,
    })
}

impl StageModelReader {
    pub fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        tokenize(self.inner.raw, text, add_special)
    }

    /// Tokenize without allocating a token buffer larger than `max_tokens`;
    /// see [`StageModel::tokenize_bounded`].
    pub fn tokenize_bounded(
        &self,
        text: &str,
        add_special: bool,
        max_tokens: usize,
    ) -> Result<Option<Vec<i32>>> {
        tokenize_bounded(self.inner.raw, text, add_special, max_tokens)
    }

    pub fn apply_chat_template_json(
        &self,
        messages_json: &str,
        options: ChatTemplateJsonOptions,
    ) -> Result<ChatTemplateJsonResult> {
        apply_chat_template_json(self.inner.raw, messages_json, options)
    }

    pub fn parse_chat_response_json(
        &self,
        generated_text: &str,
        metadata_json: &str,
        is_partial: bool,
    ) -> Result<String> {
        parse_chat_response_json_native(generated_text, metadata_json, is_partial)
    }

    pub fn detokenize_bytes(&self, tokens: &[i32]) -> Result<Vec<u8>> {
        detokenize_bytes(self.inner.raw, tokens)
    }

    pub fn token_is_eog(&self, token: i32) -> Result<bool> {
        token_is_eog(self.inner.raw, token)
    }
}

fn detokenize_bytes(raw: *mut RawModel, tokens: &[i32]) -> Result<Vec<u8>> {
    let mut bytes = 0usize;
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_detokenize(
            raw,
            tokens.as_ptr(),
            tokens.len(),
            ptr::null_mut(),
            0,
            &mut bytes,
            &mut error,
        )
    };
    if status != Status::BufferTooSmall && status != Status::Ok {
        ensure_ok(status, error)?;
    } else {
        free_error(error);
    }

    let mut output = vec![0_u8; bytes.max(1)];
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_detokenize(
            raw,
            tokens.as_ptr(),
            tokens.len(),
            output.as_mut_ptr().cast(),
            output.len(),
            &mut bytes,
            &mut error,
        )
    };
    ensure_ok(status, error)?;
    output.truncate(bytes);
    Ok(output)
}

fn token_is_eog(raw: *mut RawModel, token: i32) -> Result<bool> {
    let mut is_eog = false;
    let mut error = ptr::null_mut();
    let status = unsafe { skippy_ffi::skippy_token_is_eog(raw, token, &mut is_eog, &mut error) };
    ensure_ok(status, error)?;
    Ok(is_eog)
}

fn tokenize(raw: *mut RawModel, text: &str, add_special: bool) -> Result<Vec<i32>> {
    tokenize_bounded(raw, text, add_special, usize::MAX)?
        .ok_or_else(|| anyhow!("tokenizer output exceeds the requested limit"))
}

fn tokenize_bounded(
    raw: *mut RawModel,
    text: &str,
    add_special: bool,
    max_tokens: usize,
) -> Result<Option<Vec<i32>>> {
    let initial_capacity = optimistic_token_capacity(text.len(), max_tokens);
    let text = CString::new(text).context("text contains an interior NUL byte")?;
    let mut tokens = vec![0_i32; initial_capacity];
    let mut count = 0usize;
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_tokenize(
            raw,
            text.as_ptr(),
            add_special,
            tokens.as_mut_ptr(),
            tokens.len(),
            &mut count,
            &mut error,
        )
    };
    if status == Status::Ok {
        ensure_ok(status, error)?;
        tokens.truncate(count);
        return Ok(Some(tokens));
    }
    if status != Status::BufferTooSmall {
        ensure_ok(status, error)?;
    }
    free_error(error);

    if count > max_tokens {
        return Ok(None);
    }

    tokens.resize(count, 0);
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_tokenize(
            raw,
            text.as_ptr(),
            add_special,
            tokens.as_mut_ptr(),
            tokens.len(),
            &mut count,
            &mut error,
        )
    };
    if status == Status::BufferTooSmall {
        free_error(error);
        return Ok(None);
    }
    ensure_ok(status, error)?;
    tokens.truncate(count);
    Ok(Some(tokens))
}

fn apply_chat_template_json(
    raw: *mut RawModel,
    messages_json: &str,
    options: ChatTemplateJsonOptions,
) -> Result<ChatTemplateJsonResult> {
    let prompt_capacity = optimistic_chat_prompt_capacity(
        messages_json.len(),
        options.tools_json.as_deref().map_or(0, str::len),
        options.chat_template_kwargs.as_deref().map_or(0, str::len),
    );
    let metadata_capacity = optimistic_chat_metadata_capacity(
        options.tools_json.as_deref().map_or(0, str::len),
        options.chat_template_kwargs.as_deref().map_or(0, str::len),
    );
    let messages_json =
        CString::new(messages_json).context("messages JSON contains an interior NUL byte")?;
    let tools_json = options
        .tools_json
        .as_deref()
        .map(CString::new)
        .transpose()
        .context("tools JSON contains an interior NUL byte")?;
    let tool_choice_json = options
        .tool_choice_json
        .as_deref()
        .map(CString::new)
        .transpose()
        .context("tool choice JSON contains an interior NUL byte")?;
    let tools_ptr = tools_json
        .as_ref()
        .map(|value| value.as_ptr())
        .unwrap_or(ptr::null());
    let tool_choice_ptr = tool_choice_json
        .as_ref()
        .map(|value| value.as_ptr())
        .unwrap_or(ptr::null());
    let reasoning_format = options
        .reasoning_format
        .map(ChatReasoningFormat::parser_name)
        .map(CString::new)
        .transpose()
        .context("reasoning format contains an interior NUL byte")?;
    let reasoning_format_ptr = reasoning_format
        .as_ref()
        .map(|value| value.as_ptr())
        .unwrap_or(ptr::null());
    let chat_template_kwargs = options
        .chat_template_kwargs
        .as_deref()
        .map(CString::new)
        .transpose()
        .context("chat template kwargs contain an interior NUL byte")?;
    let chat_template_kwargs_ptr = chat_template_kwargs
        .as_ref()
        .map(|value| value.as_ptr())
        .unwrap_or(ptr::null());
    let chat_template = optional_c_string(options.chat_template.as_deref(), "chat template")?;
    let grammar = optional_c_string(options.grammar.as_deref(), "grammar")?;
    let json_schema = optional_c_string(options.json_schema.as_deref(), "JSON schema")?;
    let chat_template_ptr = optional_c_string_ptr(&chat_template);
    let grammar_ptr = optional_c_string_ptr(&grammar);
    let json_schema_ptr = optional_c_string_ptr(&json_schema);

    let mut prompt = vec![0_u8; prompt_capacity];
    let mut metadata = vec![0_u8; metadata_capacity];
    let mut prompt_bytes = prompt.len();
    let mut metadata_bytes = metadata.len();
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_apply_chat_template_json(
            raw,
            messages_json.as_ptr(),
            tools_ptr,
            tool_choice_ptr,
            options.add_assistant,
            options.enable_thinking.is_some(),
            options.enable_thinking.unwrap_or(true),
            options.parallel_tool_calls,
            reasoning_format_ptr,
            chat_template_kwargs_ptr,
            chat_template_ptr,
            options.use_jinja,
            grammar_ptr,
            json_schema_ptr,
            options.skip_chat_parsing,
            prompt.as_mut_ptr().cast(),
            prompt.len(),
            &mut prompt_bytes,
            metadata.as_mut_ptr().cast(),
            metadata.len(),
            &mut metadata_bytes,
            &mut error,
        )
    };
    if status == Status::Ok {
        ensure_ok(status, error)?;
        prompt.truncate(prompt_bytes);
        metadata.truncate(metadata_bytes);
        return chat_template_json_result(prompt, metadata);
    }
    if status != Status::BufferTooSmall {
        ensure_ok(status, error)?;
    }
    free_error(error);

    prompt.resize(prompt_bytes.max(1), 0);
    metadata.resize(metadata_bytes.max(1), 0);
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_apply_chat_template_json(
            raw,
            messages_json.as_ptr(),
            tools_ptr,
            tool_choice_ptr,
            options.add_assistant,
            options.enable_thinking.is_some(),
            options.enable_thinking.unwrap_or(true),
            options.parallel_tool_calls,
            reasoning_format_ptr,
            chat_template_kwargs_ptr,
            chat_template_ptr,
            options.use_jinja,
            grammar_ptr,
            json_schema_ptr,
            options.skip_chat_parsing,
            prompt.as_mut_ptr().cast(),
            prompt.len(),
            &mut prompt_bytes,
            metadata.as_mut_ptr().cast(),
            metadata.len(),
            &mut metadata_bytes,
            &mut error,
        )
    };
    ensure_ok(status, error)?;
    prompt.truncate(prompt_bytes);
    metadata.truncate(metadata_bytes);
    chat_template_json_result(prompt, metadata)
}

impl Drop for StageModelInner {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = skippy_ffi::skippy_model_free(self.raw, ptr::null_mut());
            }
        }
    }
}

fn optional_c_string(value: Option<&str>, field: &str) -> Result<Option<CString>> {
    value
        .map(CString::new)
        .transpose()
        .with_context(|| format!("{field} contains an interior NUL byte"))
}

fn optional_c_string_ptr(value: &Option<CString>) -> *const std::ffi::c_char {
    value
        .as_ref()
        .map(|value| value.as_ptr())
        .unwrap_or(ptr::null())
}

impl Drop for StageModel {
    fn drop(&mut self) {
        self.media.take();
    }
}

#[cfg(test)]
mod output_capacity_tests {
    use super::{
        ModelStateKind, OPTIMISTIC_OUTPUT_HEADROOM, capability_from_state_probes,
        classify_model_state, optimistic_chat_metadata_capacity, optimistic_chat_parse_capacity,
        optimistic_chat_prompt_capacity, optimistic_token_capacity,
    };

    #[test]
    fn loaded_model_flags_classify_state_without_family_names() {
        assert_eq!(
            classify_model_state(false, false, false),
            ModelStateKind::Dense
        );
        assert_eq!(
            classify_model_state(true, false, false),
            ModelStateKind::Recurrent
        );
        assert_eq!(
            classify_model_state(true, true, false),
            ModelStateKind::Hybrid
        );
        assert_eq!(
            classify_model_state(true, true, true),
            ModelStateKind::Diffusion
        );
    }

    #[test]
    fn missing_native_state_probe_fails_capability_closed() {
        assert!(capability_from_state_probes(None, Some(false), Some(false)).is_none());
        assert!(capability_from_state_probes(Some(false), None, Some(false)).is_none());
        assert!(capability_from_state_probes(Some(false), Some(false), None).is_none());
        assert_eq!(
            capability_from_state_probes(Some(true), Some(true), Some(false))
                .expect("all native probes are present")
                .state_kind,
            ModelStateKind::Hybrid
        );
    }

    #[test]
    fn token_capacity_is_optimistic_but_never_exceeds_bound() {
        assert_eq!(optimistic_token_capacity(1_000, usize::MAX), 508);
        assert_eq!(optimistic_token_capacity(1_000, 100), 100);
        assert_eq!(optimistic_token_capacity(0, 0), 0);
    }

    #[test]
    fn chat_capacities_include_inputs_and_retry_headroom() {
        assert_eq!(
            optimistic_chat_prompt_capacity(100, 20, 5),
            125 + OPTIMISTIC_OUTPUT_HEADROOM
        );
        assert_eq!(
            optimistic_chat_metadata_capacity(20, 5),
            45 + OPTIMISTIC_OUTPUT_HEADROOM
        );
        assert_eq!(
            optimistic_chat_parse_capacity(100, 25),
            125 + OPTIMISTIC_OUTPUT_HEADROOM
        );
    }
}
