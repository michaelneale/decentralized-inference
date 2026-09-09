use std::collections::BTreeSet;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::fs::{File, OpenOptions};
use std::io::{LineWriter, Write};
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use tokio::sync::mpsc;

/// GGML_LLAMA_LOG_LEVEL values (set before llama_backend_init).
/// 0=silent, 1=error, 2=warn, 3=info (default), 4=debug.
pub const LLAMA_LOG_LEVEL_DEBUG: &str = "4";

static NATIVE_LOG_FILE: OnceLock<Mutex<Option<LineWriter<File>>>> = OnceLock::new();

/// Channel sender for filtered native log messages.
/// Messages matching key patterns (backend init, model load, VRAM, KV cache, tokenizer) are sent here.
static NATIVE_LOG_FILTERED_TX: OnceLock<Mutex<Option<mpsc::UnboundedSender<NativeLogEvent>>>> =
    OnceLock::new();

static NATIVE_LOG_AGGREGATOR: OnceLock<Mutex<NativeLogAggregator>> = OnceLock::new();
static NATIVE_LOG_FORWARDING_ENABLED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
static NATIVE_LOG_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn native_log_test_guard() -> std::sync::MutexGuard<'static, ()> {
    NATIVE_LOG_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeLogEvent {
    pub message: String,
    pub category: &'static str,
    pub params: Vec<(String, Value)>,
}

#[derive(Debug, Default)]
struct ProgressTracker {
    total: Option<usize>,
    completed: usize,
    next_percent: usize,
}

impl ProgressTracker {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn set_total(&mut self, total: usize) {
        self.total = Some(total);
        self.completed = 0;
        self.next_percent = 10;
    }

    fn advance(
        &mut self,
        delta: usize,
        category: &'static str,
        label: &'static str,
        unit: &'static str,
    ) -> Vec<NativeLogEvent> {
        let Some(total) = self.total else {
            return Vec::new();
        };
        if total == 0 {
            return Vec::new();
        }

        self.completed = self.completed.saturating_add(delta).min(total);
        let mut events = Vec::new();
        while self.next_percent <= 100 && self.completed * 100 >= total * self.next_percent {
            events.push(NativeLogEvent {
                message: format!(
                    "{label} {}% ({}/{} {unit})",
                    self.next_percent, self.completed, total
                ),
                category,
                params: Vec::new(),
            });
            self.next_percent += 10;
        }
        events
    }

    fn is_complete(&self) -> bool {
        matches!(self.total, Some(total) if total > 0 && self.completed >= total)
    }
}

#[derive(Debug, Default)]
struct ModelMetadataHighlights {
    architecture: Option<String>,
    name: Option<String>,
    model_type: Option<String>,
    size_label: Option<String>,
    context_length: Option<String>,
    block_count: Option<String>,
    embedding_length: Option<String>,
    feed_forward_length: Option<String>,
    attention_heads: Option<String>,
    attention_heads_kv: Option<String>,
    tokenizer_model: Option<String>,
    tokenizer_pre: Option<String>,
}

impl ModelMetadataHighlights {
    fn apply(&mut self, key: &str, value: &str) {
        let value = value.trim().trim_matches('"').to_string();
        if value.is_empty() {
            return;
        }

        match key {
            "general.architecture" => self.architecture = Some(value),
            "general.name" => self.name = Some(value),
            "general.type" => self.model_type = Some(value),
            "general.size_label" => self.size_label = Some(value),
            "tokenizer.ggml.model" => self.tokenizer_model = Some(value),
            "tokenizer.ggml.pre" => self.tokenizer_pre = Some(value),
            _ if key.ends_with(".context_length") => self.context_length = Some(value),
            _ if key.ends_with(".block_count") => self.block_count = Some(value),
            _ if key.ends_with(".embedding_length") => self.embedding_length = Some(value),
            _ if key.ends_with(".feed_forward_length") => self.feed_forward_length = Some(value),
            _ if key.ends_with(".attention.head_count") => self.attention_heads = Some(value),
            _ if key.ends_with(".attention.head_count_kv") => self.attention_heads_kv = Some(value),
            _ => {}
        }
    }

    fn summary_params(&self) -> Vec<(String, Value)> {
        let mut params = Vec::new();
        if let Some(value) = &self.architecture {
            params.push(("architecture".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.name {
            params.push(("name".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.model_type {
            params.push(("type".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.size_label {
            params.push(("size".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.context_length {
            params.push(("ctx".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.block_count {
            params.push(("blocks".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.embedding_length {
            params.push(("embed".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.feed_forward_length {
            params.push(("ffn".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.attention_heads {
            params.push(("heads".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.attention_heads_kv {
            params.push(("kv_heads".to_string(), json_value_from_text(value)));
        }
        if let Some(value) = &self.tokenizer_model {
            params.push(("tokenizer".to_string(), Value::String(value.clone())));
        }
        if let Some(value) = &self.tokenizer_pre {
            params.push(("tokenizer_pre".to_string(), Value::String(value.clone())));
        }
        params
    }
}

#[derive(Debug, Default)]
struct NativeLogAggregator {
    metadata_progress: ProgressTracker,
    tensor_progress: ProgressTracker,
    layer_assign_progress: ProgressTracker,
    kv_cache_progress: ProgressTracker,
    metadata_in_dump: bool,
    metadata_summary_emitted: bool,
    metadata_highlights: ModelMetadataHighlights,
    tensor_groups: Vec<(String, usize)>,
    tensor_groups_emitted: bool,
    kv_layers_seen: BTreeSet<usize>,
}

fn native_log_file() -> &'static Mutex<Option<LineWriter<File>>> {
    NATIVE_LOG_FILE.get_or_init(|| Mutex::new(None))
}

fn native_log_aggregator() -> &'static Mutex<NativeLogAggregator> {
    NATIVE_LOG_AGGREGATOR.get_or_init(|| Mutex::new(NativeLogAggregator::default()))
}

/// Register a channel receiver for filtered native log messages.
/// Returns the receiver end; call this once before model loading begins.
pub fn register_filtered_native_logs() -> mpsc::UnboundedReceiver<NativeLogEvent> {
    let (tx, rx) = mpsc::unbounded_channel();
    NATIVE_LOG_FILTERED_TX
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .replace(tx);
    if let Ok(mut aggregator) = native_log_aggregator().lock() {
        aggregator.reset();
    }
    rx
}

pub fn unregister_filtered_native_logs() {
    if let Some(sender) = NATIVE_LOG_FILTERED_TX.get() {
        sender.lock().unwrap().take();
    }
    if let Ok(mut aggregator) = native_log_aggregator().lock() {
        aggregator.reset();
    }
}

pub fn set_filtered_native_logs_enabled(enabled: bool) {
    NATIVE_LOG_FORWARDING_ENABLED.store(enabled, Ordering::Relaxed);
}

impl NativeLogAggregator {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn reset_model_loading_state(&mut self) {
        self.metadata_progress.reset();
        self.tensor_progress.reset();
        self.layer_assign_progress.reset();
        self.kv_cache_progress.reset();
        self.metadata_in_dump = false;
        self.metadata_summary_emitted = false;
        self.metadata_highlights = ModelMetadataHighlights::default();
        self.tensor_groups.clear();
        self.tensor_groups_emitted = false;
        self.kv_layers_seen.clear();
    }

    fn process_line(&mut self, line: &str) -> Vec<NativeLogEvent> {
        let s = line.trim();
        if s.is_empty() {
            return Vec::new();
        }

        let mut events = Vec::new();
        let metadata_kv = parse_metadata_kv_line(s);
        let tensor_summary = parse_tensor_type_summary(s);
        if metadata_kv.is_none() {
            events.extend(self.flush_metadata_summary());
        }
        if tensor_summary.is_none() {
            events.extend(self.flush_tensor_group_summary());
        }

        if let Some((metadata_rows, tensor_rows)) = parse_loaded_metadata_counts(s) {
            self.reset_model_loading_state();
            self.metadata_progress.set_total(metadata_rows);
            self.tensor_progress.set_total(tensor_rows);
            events.push(NativeLogEvent {
                message: format!(
                    "model load plan: metadata rows={metadata_rows}, tensor rows={tensor_rows}"
                ),
                category: "model",
                params: Vec::new(),
            });
            return events;
        }

        if let Some((key, value)) = metadata_kv {
            self.metadata_in_dump = true;
            self.metadata_highlights.apply(key, value);
            events.extend(
                self.metadata_progress
                    .advance(1, "model", "metadata", "rows"),
            );
            return events;
        }

        if let Some((tensor_type, count)) = tensor_summary {
            self.record_tensor_group(tensor_type, count);
            events.extend(
                self.tensor_progress
                    .advance(count, "model", "tensors", "tensors"),
            );
            if self.tensor_progress.is_complete() {
                events.extend(self.flush_tensor_group_summary());
            }
            return events;
        }

        if let Some(layers) = parse_kv_cache_layers_total(s) {
            self.kv_cache_progress.set_total(layers);
            self.kv_layers_seen.clear();
            events.push(NativeLogEvent {
                message: format!("kv cache plan: layer rows={layers}"),
                category: "kv_cache",
                params: Vec::new(),
            });
            return events;
        }

        if let Some(layer_index) = parse_kv_cache_layer_index(s) {
            if self.kv_layers_seen.insert(layer_index) {
                events.extend(
                    self.kv_cache_progress
                        .advance(1, "kv_cache", "kv cache", "layers"),
                );
            }
            return events;
        }

        if let Some(layer_index) = parse_layer_assign_index(s) {
            if self.layer_assign_progress.total.is_none()
                && let Some(total) = self
                    .metadata_highlights
                    .block_count
                    .as_deref()
                    .and_then(|s| s.parse::<usize>().ok())
            {
                self.layer_assign_progress.set_total(total);
            }
            let new_completed = layer_index + 1;
            if new_completed > self.layer_assign_progress.completed {
                let delta = new_completed - self.layer_assign_progress.completed;
                events.extend(
                    self.layer_assign_progress
                        .advance(delta, "model", "layers", "layers"),
                );
            }
            return events;
        }

        if should_suppress_native_log_line(s) {
            return events;
        }

        if let Some(event) = summarize_native_log_line(s) {
            events.push(event);
        }

        events
    }

    fn flush_metadata_summary(&mut self) -> Vec<NativeLogEvent> {
        if !self.metadata_in_dump || self.metadata_summary_emitted {
            return Vec::new();
        }
        self.metadata_in_dump = false;
        self.metadata_summary_emitted = true;
        let params = self.metadata_highlights.summary_params();
        if params.is_empty() {
            Vec::new()
        } else {
            vec![NativeLogEvent {
                message: "Reading model metadata...".to_string(),
                category: "model",
                params,
            }]
        }
    }

    fn record_tensor_group(&mut self, tensor_type: &str, count: usize) {
        let tensor_type = canonical_tensor_group_key(tensor_type);
        if let Some((_, existing_count)) = self
            .tensor_groups
            .iter_mut()
            .find(|(existing_type, _)| existing_type == &tensor_type)
        {
            *existing_count = count;
        } else {
            self.tensor_groups.push((tensor_type, count));
        }
        self.tensor_groups_emitted = false;
    }

    fn flush_tensor_group_summary(&mut self) -> Vec<NativeLogEvent> {
        if self.tensor_groups.is_empty() || self.tensor_groups_emitted {
            return Vec::new();
        }
        self.tensor_groups_emitted = true;
        vec![NativeLogEvent {
            message: "Reading tensor groups...".to_string(),
            category: "model",
            params: self
                .tensor_groups
                .iter()
                .map(|(tensor_type, count)| (tensor_type.clone(), Value::from(*count as u64)))
                .collect(),
        }]
    }
}

fn json_value_from_text(value: &str) -> Value {
    value
        .parse::<u64>()
        .map(Value::from)
        .unwrap_or_else(|_| Value::String(value.to_string()))
}

fn canonical_tensor_group_key(tensor_type: &str) -> String {
    let trimmed = tensor_type.trim();
    if trimmed.eq_ignore_ascii_case("q4_k") {
        "q4_K".to_string()
    } else if trimmed.eq_ignore_ascii_case("q5_k") {
        "q5_K".to_string()
    } else {
        trimmed.to_string()
    }
}

fn should_suppress_native_log_line(line: &str) -> bool {
    line.starts_with("llama_model_loader:") && (line.contains(": - kv") || line.contains("- kv"))
        || (line.starts_with("clip_model_loader:") && line.contains(": tensor["))
        || line.contains("tokenizer.ggml.tokens arr")
        || line.contains("tokenizer.ggml.merges arr")
        || line.contains("tokenizer.ggml.token_type arr")
        || line.starts_with("print_info:")
        || (line.starts_with("llama_kv_cache:")
            && (line.contains(": filtered") || line.contains(": dev =")))
}

fn summarize_native_log_line(line: &str) -> Option<NativeLogEvent> {
    if let Some((category, params)) = cpu_offload_diagnostic_params(line) {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category,
            params,
        });
    }

    let lower = line.to_ascii_lowercase();
    if line.contains("backend_init")
        || line.contains("llama_backend_init")
        || line.contains("GGML_CUDA")
        || line.contains("GGML_HIP")
        || line.contains("GGML_ROCM")
        || ((lower.contains("cuda")
            || lower.contains("hip")
            || lower.contains("rocm")
            || lower.contains("metal"))
            && (lower.contains("init") || lower.contains("device") || lower.contains("backend")))
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "backend",
            params: Vec::new(),
        });
    }

    if line.contains(".gguf loaded")
        || line.starts_with("llm_load_print_meta")
        || line.starts_with("llm_load_tensors")
        || (line.contains("loading model") && !line.contains("clip_model"))
        || (line.contains("loaded model") && !line.starts_with("llama_model_loader:"))
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "model",
            params: Vec::new(),
        });
    }

    if line.starts_with("llama_context: n_ubatch") || line.starts_with("llama_context: flash_attn")
    {
        // Forward the resolved micro-batch size and flash-attention mode so a live
        // deployment can prove which values the runtime actually constructed with.
        // These lines come from the llama_context parameter dump
        // (llama-context.cpp, `n_ubatch = ...` / `flash_attn = ...`).
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "runtime",
            params: Vec::new(),
        });
    }

    if line.contains("VRAM")
        || line.contains("vram")
        || line.contains("mem_alloc")
        || line.contains("_Mapped model buffer size")
        || (line.contains("GPU") && line.contains("memory"))
        || line.contains("compute buffer size")
        || line.contains("scratch buffer")
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "memory",
            params: Vec::new(),
        });
    }

    if line.starts_with("llama_kv_cache:")
        && (line.contains("buffer size") || line.contains("size = ") || line.contains("attn_rot"))
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "kv_cache",
            params: Vec::new(),
        });
    }

    if line.starts_with("init_tokenizer:")
        || line.starts_with("load: special tokens cache size")
        || line.starts_with("load: token to piece cache size")
    {
        return Some(NativeLogEvent {
            message: line.to_string(),
            category: "tokenizer",
            params: Vec::new(),
        });
    }

    None
}

fn cpu_offload_diagnostic_params(line: &str) -> Option<(&'static str, Vec<(String, Value)>)> {
    let lower = line.to_ascii_lowercase();
    let (category, surface) = if lower.contains("cpu_mapped model buffer size") {
        ("memory", "model_buffer")
    } else if lower.contains("cpu kv buffer size") {
        ("kv_cache", "kv_buffer")
    } else if lower.contains("cpu compute buffer size") {
        ("memory", "compute_buffer")
    } else {
        return None;
    };
    Some((
        category,
        vec![
            (
                "offload_device".to_string(),
                Value::String("CPU".to_string()),
            ),
            (
                "offload_surface".to_string(),
                Value::String(surface.to_string()),
            ),
        ],
    ))
}

fn parse_loaded_metadata_counts(line: &str) -> Option<(usize, usize)> {
    let (_, remainder) = line.split_once("loaded meta data with ")?;
    let (metadata_rows, remainder) = remainder.split_once(" key-value pairs and ")?;
    let metadata_rows = metadata_rows.trim().parse().ok()?;
    let (tensor_rows, _) = remainder.split_once(" tensors")?;
    let tensor_rows = tensor_rows.trim().parse().ok()?;
    Some((metadata_rows, tensor_rows))
}

fn parse_metadata_kv_line(line: &str) -> Option<(&str, &str)> {
    if !line.starts_with("llama_model_loader:") || !line.contains("- kv") {
        return None;
    }
    let (_, remainder) = line.split_once(": - kv")?;
    let (_, remainder) = remainder.split_once(':')?;
    let remainder = remainder.trim();
    let (lhs, value) = remainder.split_once(" = ")?;
    let key = lhs.split_whitespace().next()?;
    Some((key, value.trim()))
}

fn parse_tensor_type_summary(line: &str) -> Option<(&str, usize)> {
    if !line.starts_with("llama_model_loader:") || !line.contains("- type") {
        return None;
    }
    let (_, remainder) = line.split_once("- type")?;
    let remainder = remainder.trim();
    let (tensor_type, count_and_suffix) = remainder.split_once(':')?;
    let count = count_and_suffix.split_whitespace().next()?.parse().ok()?;
    Some((tensor_type.trim(), count))
}

fn parse_layer_assign_index(line: &str) -> Option<usize> {
    if !line.starts_with("load_tensors: layer") || !line.contains("assigned to device") {
        return None;
    }
    let (_, remainder) = line.split_once("load_tensors: layer")?;
    let digits = remainder
        .trim_start()
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn parse_kv_cache_layers_total(line: &str) -> Option<usize> {
    if !line.starts_with("llama_kv_cache:") || !line.contains(" layers") {
        return None;
    }
    let prefix = line.split_once(" layers")?.0;
    let digits = prefix
        .chars()
        .rev()
        .skip_while(|ch| ch.is_whitespace())
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn parse_kv_cache_layer_index(line: &str) -> Option<usize> {
    if !line.starts_with("llama_kv_cache: layer") {
        return None;
    }
    let (_, remainder) = line.split_once("layer")?;
    let digits = remainder
        .trim_start()
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn flush_native_log_writer<W: Write>(writer: &mut Option<LineWriter<W>>) {
    if let Some(writer) = writer.as_mut() {
        let _ = writer.flush();
    }
}

fn sanitize_native_log_note(note: &str) -> String {
    note.chars()
        .map(|ch| if matches!(ch, '\n' | '\r') { ' ' } else { ch })
        .collect()
}

pub fn write_native_log_note(note: impl AsRef<str>) {
    let note = sanitize_native_log_note(note.as_ref());
    if let Ok(mut guard) = native_log_file().lock()
        && let Some(writer) = guard.as_mut()
    {
        let _ = writeln!(writer, "mesh-llm: {note}");
        let _ = writer.flush();
    }
    forward_native_log_note(note);
}

fn forward_native_log_note(note: String) {
    if !NATIVE_LOG_FORWARDING_ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let event = NativeLogEvent {
        message: format!("mesh-llm: {note}"),
        category: "model",
        params: Vec::new(),
    };
    if let Some(sender) = NATIVE_LOG_FILTERED_TX.get()
        && let Ok(guard) = sender.lock()
        && let Some(tx) = guard.as_ref()
    {
        let _ = tx.send(event);
    }
}

fn clear_native_log_file() {
    if let Ok(mut guard) = native_log_file().lock() {
        flush_native_log_writer(&mut guard);
        *guard = None;
    }
}

fn set_native_log_callback(callback: skippy_ffi::LlamaLogCallback) {
    if !skippy_ffi::native_runtime_loaded() {
        return;
    }
    unsafe {
        skippy_ffi::llama_log_set(callback, ptr::null_mut());
        skippy_ffi::ggml_log_set(callback, ptr::null_mut());
        skippy_ffi::mtmd_helper_log_set(callback, ptr::null_mut());
    }
}

pub fn redirect_native_logs_to_file(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    options.mode(0o600);

    let file = options
        .open(path)
        .with_context(|| format!("open skippy native log file {}", path.display()))?;
    let mut guard = native_log_file()
        .lock()
        .map_err(|_| anyhow!("native log file mutex poisoned"))?;
    flush_native_log_writer(&mut guard);
    *guard = Some(LineWriter::new(file));
    drop(guard);

    set_native_log_callback(Some(write_native_log));

    Ok(())
}

pub fn suppress_native_logs() {
    clear_native_log_file();
    set_native_log_callback(Some(discard_native_log));
}

pub fn restore_native_logs() {
    clear_native_log_file();
    set_native_log_callback(None);
}

/// Enable verbose llama.cpp logging. Call before `llama_backend_init()` / model loading.
/// Sets GGML_LLAMA_LOG_LEVEL=4 so LLAMA_LOG_DEBUG macros produce output.
pub fn enable_verbose_native_logs() {
    // SAFETY: UNSAFE CONTRACT — callers must invoke this before concurrent
    // runtime work can access the process environment. The API does not yet
    // enforce that startup boundary; retain the audit TODO below.
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe { std::env::set_var("GGML_LLAMA_LOG_LEVEL", LLAMA_LOG_LEVEL_DEBUG) };
}

/// Disable verbose llama.cpp logging (restore default level).
pub fn disable_verbose_native_logs() {
    // SAFETY: UNSAFE CONTRACT — callers must invoke this before concurrent
    // runtime work can access the process environment. The API does not yet
    // enforce that startup boundary; retain the audit TODO below.
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe { std::env::remove_var("GGML_LLAMA_LOG_LEVEL") };
}

unsafe extern "C" fn write_native_log(_level: c_int, text: *const c_char, _user_data: *mut c_void) {
    if text.is_null() {
        return;
    }

    let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
    if let Ok(mut guard) = native_log_file().lock()
        && let Some(writer) = guard.as_mut()
    {
        let _ = writer.write_all(bytes);
    }

    // Also send aggregated messages through the channel when runtime forwarding is enabled.
    if !NATIVE_LOG_FORWARDING_ENABLED.load(Ordering::Relaxed) {
        return;
    }

    if let Ok(text_str) = core::str::from_utf8(bytes) {
        let events = match native_log_aggregator().lock() {
            Ok(mut aggregator) => aggregator.process_line(text_str.trim()),
            _ => Vec::new(),
        };
        if let Some(tx) = NATIVE_LOG_FILTERED_TX.get()
            && let Ok(guard) = tx.lock()
            && let Some(ref sender) = *guard
        {
            for event in events {
                let _ = sender.send(event);
            }
        }
    }
}

unsafe extern "C" fn discard_native_log(
    _level: c_int,
    _text: *const c_char,
    _user_data: *mut c_void,
) {
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        env,
        ffi::CString,
        fs, ptr,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::{SystemTime, UNIX_EPOCH},
    };
    use tokio::sync::mpsc::error::TryRecvError;

    mod native_log {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/tests/native_log.rs"
        ));
    }

    struct FlushCountingWriter {
        flush_count: Arc<AtomicUsize>,
    }

    impl Write for FlushCountingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.flush_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn native_log_writer_flush_helper_explicitly_flushes_line_writer() {
        let flush_count = Arc::new(AtomicUsize::new(0));
        let writer = FlushCountingWriter {
            flush_count: flush_count.clone(),
        };
        let mut writer = Some(LineWriter::new(writer));
        writer
            .as_mut()
            .expect("writer should exist")
            .write_all(b"buffered native log line\n")
            .expect("write to buffered test writer should succeed");

        flush_native_log_writer(&mut writer);

        assert_eq!(flush_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn native_log_writer_flushes_newline_and_partial_line() -> anyhow::Result<()> {
        let _native_log_guard = native_log_test_guard();

        struct RestoreNativeLogs;

        impl Drop for RestoreNativeLogs {
            fn drop(&mut self) {
                restore_native_logs();
            }
        }

        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!(
            "skippy-native-log-buffer-test-{}-{nanos}.log",
            std::process::id()
        ));
        let _guard = RestoreNativeLogs;
        redirect_native_logs_to_file(&path)?;

        let message = CString::new("buffered native log line\n")?;
        unsafe {
            write_native_log(0, message.as_ptr(), ptr::null_mut());
        }

        let contents = fs::read_to_string(&path)?;
        restore_native_logs();

        fs::remove_file(&path)?;
        assert_eq!(contents, "buffered native log line\n");

        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!(
            "skippy-native-log-partial-line-test-{}-{nanos}.log",
            std::process::id()
        ));
        let _guard = RestoreNativeLogs;
        redirect_native_logs_to_file(&path)?;

        let message = CString::new("partial native log line")?;
        unsafe {
            write_native_log(0, message.as_ptr(), ptr::null_mut());
        }
        restore_native_logs();

        let contents = fs::read_to_string(&path)?;
        fs::remove_file(&path)?;
        assert_eq!(contents, "partial native log line");
        Ok(())
    }

    #[test]
    fn native_log_note_writes_sanitized_flushed_context() -> anyhow::Result<()> {
        let _native_log_guard = native_log_test_guard();

        struct RestoreNativeLogs;

        impl Drop for RestoreNativeLogs {
            fn drop(&mut self) {
                restore_native_logs();
            }
        }

        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = env::temp_dir().join(format!(
            "skippy-native-log-note-test-{}-{nanos}.log",
            std::process::id()
        ));
        let _guard = RestoreNativeLogs;
        redirect_native_logs_to_file(&path)?;

        write_native_log_note("native call begin\nwith context");

        let contents = fs::read_to_string(&path)?;
        restore_native_logs();
        fs::remove_file(&path)?;

        assert!(
            contents.ends_with("mesh-llm: native call begin with context\n"),
            "unexpected native log contents: {contents:?}"
        );
        assert!(
            !contents.contains("native call begin\nwith context"),
            "native log note was not sanitized: {contents:?}"
        );
        Ok(())
    }

    #[test]
    fn aggregator_preserves_backend_summary_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert_eq!(
            aggregator.process_line("backend_init succeeded"),
            vec![NativeLogEvent {
                message: "backend_init succeeded".to_string(),
                category: "backend",
                params: Vec::new(),
            }]
        );
        assert_eq!(
            aggregator.process_line("llama_backend_init: GGML_CUDA"),
            vec![NativeLogEvent {
                message: "llama_backend_init: GGML_CUDA".to_string(),
                category: "backend",
                params: Vec::new(),
            }]
        );
        assert_eq!(
            aggregator.process_line("llama_backend_init: GGML_HIP backend initialized"),
            vec![NativeLogEvent {
                message: "llama_backend_init: GGML_HIP backend initialized".to_string(),
                category: "backend",
                params: Vec::new(),
            }]
        );
        assert_eq!(
            aggregator.process_line("llama_backend_init: GGML_ROCM backend initialized"),
            vec![NativeLogEvent {
                message: "llama_backend_init: GGML_ROCM backend initialized".to_string(),
                category: "backend",
                params: Vec::new(),
            }]
        );
    }

    #[test]
    fn aggregator_forwards_llama_context_config_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert_eq!(
            aggregator.process_line("llama_context: n_ubatch      = 512"),
            vec![NativeLogEvent {
                message: "llama_context: n_ubatch      = 512".to_string(),
                category: "runtime",
                params: Vec::new(),
            }]
        );
        assert_eq!(
            aggregator.process_line("llama_context: flash_attn    = enabled"),
            vec![NativeLogEvent {
                message: "llama_context: flash_attn    = enabled".to_string(),
                category: "runtime",
                params: Vec::new(),
            }]
        );
        assert!(
            aggregator
                .process_line("llama_context: n_ctx         = 8192")
                .is_empty()
        );
        assert!(
            aggregator
                .process_line("llama_context: causal_attn   = 1")
                .is_empty()
        );
    }

    #[test]
    fn aggregator_ignores_non_backend_cuda_mentions() {
        let mut aggregator = NativeLogAggregator::default();
        assert!(
            aggregator
                .process_line("CUDA kernel launch for attention")
                .is_empty()
        );
        assert!(aggregator.process_line("offloading to CUDA").is_empty());
    }

    #[test]
    fn aggregator_builds_metadata_summary_and_progress() {
        let mut aggregator = NativeLogAggregator::default();
        assert_eq!(
                aggregator.process_line(
                    "llama_model_loader: loaded meta data with 10 key-value pairs and 100 tensors from model.gguf (version GGUF V3)"
                ),
                vec![NativeLogEvent {
                    message: "model load plan: metadata rows=10, tensor rows=100".to_string(),
                    category: "model",
                    params: Vec::new(),
                }]
            );

        for (idx, line) in [
            "llama_model_loader: - kv   0: general.architecture str = qwen35",
            "llama_model_loader: - kv   1: general.name str = Qwen 3.5 4B",
            "llama_model_loader: - kv   2: general.type str = model",
            "llama_model_loader: - kv   3: general.size_label str = 4B",
            "llama_model_loader: - kv   4: qwen35.context_length u32 = 40960",
            "llama_model_loader: - kv   5: qwen35.block_count u32 = 36",
            "llama_model_loader: - kv   6: qwen35.embedding_length u32 = 2560",
            "llama_model_loader: - kv   7: qwen35.feed_forward_length u32 = 9728",
            "llama_model_loader: - kv   8: qwen35.attention.head_count u32 = 32",
            "llama_model_loader: - kv   9: qwen35.attention.head_count_kv u32 = 8",
        ]
        .iter()
        .enumerate()
        {
            let events = aggregator.process_line(line);
            assert!(
                events
                    .iter()
                    .any(|event| event.message.contains(&format!("{}%", (idx + 1) * 10))),
                "expected {}% metadata progress in {:?}",
                (idx + 1) * 10,
                events
            );
        }

        let flush_events = aggregator.process_line("llm_load_print_meta: version = 3");
        assert!(
            flush_events
                .iter()
                .any(|event| event.message == "llm_load_print_meta: version = 3")
        );
        assert!(
            flush_events
                .iter()
                .any(|event| event.message == "Reading model metadata...")
        );
        assert!(flush_events.iter().any(|event| {
            event.params.iter().any(|(key, value)| {
                key == "architecture" && value == &Value::String("qwen35".to_string())
            })
        }));
    }

    #[test]
    fn aggregator_emits_tensor_progress_from_type_summaries() {
        let mut aggregator = NativeLogAggregator::default();
        aggregator.process_line(
                "llama_model_loader: loaded meta data with 46 key-value pairs and 100 tensors from model.gguf (version GGUF V3)",
            );

        let first = aggregator.process_line("llama_model_loader: - type  f32:  30 tensors");
        assert!(
            first
                .iter()
                .any(|event| event.message.contains("tensors 10%"))
        );
        assert!(
            first
                .iter()
                .any(|event| event.message.contains("tensors 30%"))
        );

        let second = aggregator.process_line("llama_model_loader: - type q4_k:  70 tensors");
        assert!(
            second
                .iter()
                .any(|event| event.message.contains("tensors 100%"))
        );
        assert!(second.iter().any(|event| {
            event.message == "Reading tensor groups..."
                && event
                    .params
                    .iter()
                    .any(|(key, value)| key == "f32" && value == &Value::from(30_u64))
                && event
                    .params
                    .iter()
                    .any(|(key, value)| key == "q4_K" && value == &Value::from(70_u64))
        }));
    }

    #[test]
    fn aggregator_preserves_memory_summary_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert_eq!(
            aggregator.process_line("VRAM used: 12.4 GB"),
            vec![NativeLogEvent {
                message: "VRAM used: 12.4 GB".to_string(),
                category: "memory",
                params: Vec::new(),
            }]
        );
    }

    #[test]
    fn aggregator_tags_cpu_offload_evidence_without_capacity_facts() {
        let mut aggregator = NativeLogAggregator::default();
        let model_buffer =
            aggregator.process_line("load_tensors:   CPU_Mapped model buffer size = 47492.37 MiB");
        assert_eq!(
            model_buffer,
            vec![NativeLogEvent {
                message: "load_tensors:   CPU_Mapped model buffer size = 47492.37 MiB".to_string(),
                category: "memory",
                params: vec![
                    (
                        "offload_device".to_string(),
                        Value::String("CPU".to_string())
                    ),
                    (
                        "offload_surface".to_string(),
                        Value::String("model_buffer".to_string())
                    ),
                ],
            }]
        );
        assert_no_capacity_params(&model_buffer);

        assert_eq!(
            aggregator.process_line("llama_kv_cache:        CPU KV buffer size =  3264.00 MiB"),
            vec![NativeLogEvent {
                message: "llama_kv_cache:        CPU KV buffer size =  3264.00 MiB".to_string(),
                category: "kv_cache",
                params: vec![
                    (
                        "offload_device".to_string(),
                        Value::String("CPU".to_string())
                    ),
                    (
                        "offload_surface".to_string(),
                        Value::String("kv_buffer".to_string())
                    ),
                ],
            }]
        );
        assert_eq!(
            aggregator.process_line("sched_reserve:        CPU compute buffer size =   856.29 MiB"),
            vec![NativeLogEvent {
                message: "sched_reserve:        CPU compute buffer size =   856.29 MiB".to_string(),
                category: "memory",
                params: vec![
                    (
                        "offload_device".to_string(),
                        Value::String("CPU".to_string())
                    ),
                    (
                        "offload_surface".to_string(),
                        Value::String("compute_buffer".to_string())
                    ),
                ],
            }]
        );
    }

    fn assert_no_capacity_params(events: &[NativeLogEvent]) {
        const CAPACITY_KEYS: &[&str] = &[
            "backend_device",
            "capacity_gb",
            "gpu_count",
            "gpu_vram",
            "vram_bytes",
        ];
        assert!(events.iter().all(|event| {
            event
                .params
                .iter()
                .all(|(key, _)| !CAPACITY_KEYS.contains(&key.as_str()))
        }));
    }

    #[test]
    fn aggregator_tracks_kv_cache_layer_progress_without_double_counting() {
        let mut aggregator = NativeLogAggregator::default();
        let plan = aggregator.process_line(
                "llama_kv_cache: size = 4096.00 MiB (131072 cells,   8 layers,  2/1 seqs), K (f16): 2048.00 MiB, V (f16): 2048.00 MiB",
            );
        assert_eq!(
            plan,
            vec![NativeLogEvent {
                message: "kv cache plan: layer rows=8".to_string(),
                category: "kv_cache",
                params: Vec::new(),
            }]
        );

        let first = aggregator.process_line("llama_kv_cache: layer   0: filtered");
        assert!(
            first
                .iter()
                .any(|event| event.message.contains("kv cache 10%"))
        );

        let duplicate = aggregator.process_line("llama_kv_cache: layer   0: dev = MTL0");
        assert!(duplicate.is_empty());

        for layer in 1..8 {
            aggregator.process_line(&format!("llama_kv_cache: layer   {layer}: filtered"));
        }

        let summary = aggregator.process_line("llama_kv_cache: attn_rot = 128");
        assert_eq!(
            summary,
            vec![NativeLogEvent {
                message: "llama_kv_cache: attn_rot = 128".to_string(),
                category: "kv_cache",
                params: Vec::new(),
            }]
        );
    }

    #[test]
    fn aggregator_preserves_tokenizer_summary_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert_eq!(
            aggregator.process_line("init_tokenizer: initializing tokenizer for type 2"),
            vec![NativeLogEvent {
                message: "init_tokenizer: initializing tokenizer for type 2".to_string(),
                category: "tokenizer",
                params: Vec::new(),
            }]
        );
    }

    #[test]
    fn aggregator_suppresses_print_info_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert!(
            aggregator
                .process_line("print_info: n_vocab               = 248320")
                .is_empty()
        );
    }

    #[test]
    fn aggregator_rejects_empty_and_whitespace_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert!(aggregator.process_line("").is_empty());
        assert!(aggregator.process_line("   ").is_empty());
    }

    #[test]
    fn aggregator_suppresses_raw_noise_lines() {
        let mut aggregator = NativeLogAggregator::default();
        assert!(
            aggregator
                .process_line(
                    "clip_model_loader: tensor[0]: n_dims = 1, name = v.blk.0.attn_out.bias"
                )
                .is_empty()
        );
        assert!(
            aggregator
                .process_line("tokenizer.ggml.tokens arr[str,248320] = [\"!\", ...]")
                .is_empty()
        );
    }

    #[test]
    fn parse_layer_assign_index_extracts_layer_number() {
        assert_eq!(
            parse_layer_assign_index("load_tensors: layer   0 assigned to device CUDA0"),
            Some(0)
        );
        assert_eq!(
            parse_layer_assign_index("load_tensors: layer  63 assigned to device CUDA0"),
            Some(63)
        );
        assert_eq!(
            parse_layer_assign_index("load_tensors: layer   5 assigned to device CPU"),
            Some(5)
        );
        assert_eq!(
            parse_layer_assign_index("llm_load_tensors: offloaded 64/65 layers"),
            None
        );
        assert_eq!(
            parse_layer_assign_index("load_tensors: layer   0 computation graph"),
            None
        );
    }

    #[test]
    fn aggregator_tracks_layer_assign_progress_using_block_count() {
        let mut aggregator = NativeLogAggregator::default();

        aggregator.process_line(
                "llama_model_loader: loaded meta data with 10 key-value pairs and 100 tensors from model.gguf (version GGUF V3)"
            );
        for line in [
            "llama_model_loader: - kv   0: general.architecture str = qwen35",
            "llama_model_loader: - kv   1: general.name str = Qwen 3.5 4B",
            "llama_model_loader: - kv   2: general.type str = model",
            "llama_model_loader: - kv   3: general.size_label str = 4B",
            "llama_model_loader: - kv   4: qwen35.context_length u32 = 40960",
            "llama_model_loader: - kv   5: qwen35.block_count u32 = 4",
            "llama_model_loader: - kv   6: qwen35.embedding_length u32 = 2560",
            "llama_model_loader: - kv   7: qwen35.feed_forward_length u32 = 9728",
            "llama_model_loader: - kv   8: qwen35.attention.head_count u32 = 32",
            "llama_model_loader: - kv   9: qwen35.attention.head_count_kv u32 = 8",
        ] {
            aggregator.process_line(line);
        }
        aggregator.process_line("llm_load_print_meta: version = 3");

        let e0 = aggregator.process_line("load_tensors: layer   0 assigned to device CUDA0");
        assert!(e0.iter().any(|event| event.message.contains("layers 10%")));
        assert!(e0.iter().any(|event| event.message.contains("layers 20%")));

        let e1 = aggregator.process_line("load_tensors: layer   1 assigned to device CUDA0");
        assert!(e1.iter().any(|event| event.message.contains("layers 50%")));

        let e2 = aggregator.process_line("load_tensors: layer   2 assigned to device CUDA0");
        assert!(e2.iter().any(|event| event.message.contains("layers 70%")));

        let e3 = aggregator.process_line("load_tensors: layer   3 assigned to device CUDA0");
        let pcts: Vec<&str> = e3
            .iter()
            .filter_map(|ev| {
                if ev.message.contains("layers") && ev.message.contains('%') {
                    Some(ev.message.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            pcts.iter().any(|m| m.contains("100%")),
            "expected layers 100% at final layer, got {:?}",
            pcts
        );
    }
}
