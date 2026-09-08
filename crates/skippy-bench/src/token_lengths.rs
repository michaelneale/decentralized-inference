use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use skippy_runtime::{
    ChatTemplateMessage, ChatTemplateOptions, MtpSource, RuntimeConfig, RuntimeLoadMode,
    StageModel, suppress_native_logs,
};

use crate::cli::TokenLengthsArgs;

#[derive(Debug)]
struct PromptCase {
    sequence: usize,
    prompt_id: Option<String>,
    family: Option<String>,
    length_bucket: Option<String>,
    prompt_chars: usize,
    messages: Vec<ChatTemplateMessage>,
}

#[derive(Debug, Serialize)]
struct TokenLengthRow {
    sequence: usize,
    prompt_id: Option<String>,
    family: Option<String>,
    length_bucket: Option<String>,
    prompt_chars: usize,
    rendered_chars: usize,
    prompt_tokens: usize,
    generation_limit: u32,
    requested_tokens: usize,
    ctx_size: u32,
    fits_context: bool,
}

#[derive(Debug, Serialize)]
struct TokenLengthSummary {
    model_path: String,
    prompt_corpus: String,
    output_tsv: String,
    ctx_size: u32,
    generation_limit: u32,
    enable_thinking: bool,
    row_count: usize,
    fits_context: usize,
    exceeds_context: usize,
    prompt_tokens_min: Option<usize>,
    prompt_tokens_p50: Option<usize>,
    prompt_tokens_p95: Option<usize>,
    prompt_tokens_p99: Option<usize>,
    prompt_tokens_max: Option<usize>,
    requested_tokens_max: Option<usize>,
}

pub fn token_lengths(args: TokenLengthsArgs) -> Result<()> {
    suppress_native_logs();

    if args.ctx_size == 0 {
        anyhow::bail!("--ctx-size must be greater than zero");
    }
    if args.generation_limit == 0 {
        anyhow::bail!("--generation-limit must be greater than zero");
    }
    if args.layer_end == 0 {
        anyhow::bail!("--layer-end must be greater than zero");
    }

    let cases = prompt_cases(&args.prompt_corpus)?;
    let model = StageModel::open(
        &args.model_path,
        &RuntimeConfig {
            stage_index: 0,
            layer_start: 0,
            layer_end: args.layer_end,
            ctx_size: args.ctx_size,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: args.n_gpu_layers,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_runtime::SplitMode::Auto,
            selected_backend_device: None,
            cache_type_k: skippy_runtime::GGML_TYPE_F16,
            cache_type_v: skippy_runtime::GGML_TYPE_F16,
            flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
            load_mode: RuntimeLoadMode::RuntimeSlice,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_runtime::GlmDsaPolicy::Auto,
            include_embeddings: true,
            include_output: false,
            mtp_source: MtpSource::Disabled,
            filter_tensors_on_load: true,
            resident_tensor_names: Vec::new(),
            checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
            checkpoint_imatrix: None,
            checkpoint_imatrix_sha256: None,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
        },
    )
    .with_context(|| format!("open tokenizer model {}", args.model_path.display()))?;

    let mut rows = Vec::with_capacity(cases.len());
    for prompt_case in &cases {
        let rendered = model
            .apply_chat_template_with_options(
                &prompt_case.messages,
                ChatTemplateOptions {
                    add_assistant: true,
                    enable_thinking: Some(args.enable_thinking),
                    reasoning_format: None,
                    ..ChatTemplateOptions::default()
                },
            )
            .with_context(|| {
                format!(
                    "apply chat template for row {} ({})",
                    prompt_case.sequence,
                    prompt_case.prompt_id.as_deref().unwrap_or("no-id")
                )
            })?;
        let tokens = model.tokenize(&rendered, true).with_context(|| {
            format!(
                "tokenize row {} ({})",
                prompt_case.sequence,
                prompt_case.prompt_id.as_deref().unwrap_or("no-id")
            )
        })?;
        let requested_tokens = tokens.len().saturating_add(args.generation_limit as usize);
        rows.push(TokenLengthRow {
            sequence: prompt_case.sequence,
            prompt_id: prompt_case.prompt_id.clone(),
            family: prompt_case.family.clone(),
            length_bucket: prompt_case.length_bucket.clone(),
            prompt_chars: prompt_case.prompt_chars,
            rendered_chars: rendered.chars().count(),
            prompt_tokens: tokens.len(),
            generation_limit: args.generation_limit,
            requested_tokens,
            ctx_size: args.ctx_size,
            fits_context: requested_tokens <= args.ctx_size as usize,
        });
    }

    write_tsv(&args.output_tsv, &rows)?;
    let summary = summarize(&args, &rows);
    let summary_json = serde_json::to_vec_pretty(&summary)?;
    if let Some(path) = args.summary_json.as_ref() {
        fs::write(path, &summary_json)
            .with_context(|| format!("write summary JSON {}", path.display()))?;
    }
    println!("{}", String::from_utf8(summary_json)?);
    Ok(())
}

fn prompt_cases(path: &Path) -> Result<Vec<PromptCase>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut cases = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<Value>(line).with_context(|| {
            format!("parse JSONL line {} in {}", line_index + 1, path.display())
        })?;
        cases.push(
            prompt_case_from_value(cases.len(), &value).with_context(|| {
                format!("read corpus line {} in {}", line_index + 1, path.display())
            })?,
        );
    }
    if cases.is_empty() {
        anyhow::bail!("prompt corpus is empty: {}", path.display());
    }
    Ok(cases)
}

fn prompt_case_from_value(sequence: usize, value: &Value) -> Result<PromptCase> {
    let prompt_id = value
        .get("id")
        .or_else(|| value.get("prompt_id"))
        .and_then(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .or_else(|| value.as_i64().map(|id| id.to_string()))
        });
    let family = value
        .get("family")
        .or_else(|| value.get("category"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let length_bucket = value
        .get("length_bucket")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let messages = if let Some(messages) = value.get("messages").and_then(Value::as_array) {
        messages_from_value(messages)?
    } else if let Some(prompt) = value.get("prompt").and_then(Value::as_str) {
        vec![ChatTemplateMessage::new("user", prompt)]
    } else if let Some(turns) = value.get("turns").and_then(Value::as_array) {
        let prompt = turns
            .iter()
            .find_map(Value::as_str)
            .context("turns did not contain a string prompt")?;
        vec![ChatTemplateMessage::new("user", prompt)]
    } else if let Some(prompt) = value.as_str() {
        vec![ChatTemplateMessage::new("user", prompt)]
    } else {
        anyhow::bail!("JSONL row must include prompt, turns, or messages");
    };
    let prompt_chars = messages
        .iter()
        .map(|message| message.content.chars().count())
        .sum();
    Ok(PromptCase {
        sequence,
        prompt_id,
        family,
        length_bucket,
        prompt_chars,
        messages,
    })
}

fn messages_from_value(messages: &[Value]) -> Result<Vec<ChatTemplateMessage>> {
    let mut rendered = Vec::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .context("message is missing string role")?;
        let content = message
            .get("content")
            .map(message_content_to_text)
            .transpose()?
            .unwrap_or_default();
        rendered.push(ChatTemplateMessage::new(role, content));
    }
    if rendered.is_empty() {
        anyhow::bail!("messages array is empty");
    }
    Ok(rendered)
}

fn message_content_to_text(content: &Value) -> Result<String> {
    if let Some(text) = content.as_str() {
        return Ok(text.to_string());
    }
    if let Some(parts) = content.as_array() {
        let text = parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(text);
    }
    if content.is_null() {
        return Ok(String::new());
    }
    Ok(content.to_string())
}

fn write_tsv(path: &Path, rows: &[TokenLengthRow]) -> Result<()> {
    let mut output = String::from(
        "sequence\tprompt_id\tfamily\tlength_bucket\tprompt_chars\trendered_chars\tprompt_tokens\tgeneration_limit\trequested_tokens\tctx_size\tfits_context\n",
    );
    for row in rows {
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.sequence,
            tsv_cell(row.prompt_id.as_deref()),
            tsv_cell(row.family.as_deref()),
            tsv_cell(row.length_bucket.as_deref()),
            row.prompt_chars,
            row.rendered_chars,
            row.prompt_tokens,
            row.generation_limit,
            row.requested_tokens,
            row.ctx_size,
            row.fits_context
        ));
    }
    fs::write(path, output).with_context(|| format!("write token length TSV {}", path.display()))
}

fn tsv_cell(value: Option<&str>) -> String {
    value
        .unwrap_or("")
        .replace('\t', " ")
        .replace(['\r', '\n'], " ")
}

fn summarize(args: &TokenLengthsArgs, rows: &[TokenLengthRow]) -> TokenLengthSummary {
    let mut prompt_tokens = rows.iter().map(|row| row.prompt_tokens).collect::<Vec<_>>();
    let mut requested_tokens = rows
        .iter()
        .map(|row| row.requested_tokens)
        .collect::<Vec<_>>();
    prompt_tokens.sort_unstable();
    requested_tokens.sort_unstable();
    let fits_context = rows.iter().filter(|row| row.fits_context).count();
    TokenLengthSummary {
        model_path: args.model_path.display().to_string(),
        prompt_corpus: args.prompt_corpus.display().to_string(),
        output_tsv: args.output_tsv.display().to_string(),
        ctx_size: args.ctx_size,
        generation_limit: args.generation_limit,
        enable_thinking: args.enable_thinking,
        row_count: rows.len(),
        fits_context,
        exceeds_context: rows.len().saturating_sub(fits_context),
        prompt_tokens_min: prompt_tokens.first().copied(),
        prompt_tokens_p50: percentile(&prompt_tokens, 0.50),
        prompt_tokens_p95: percentile(&prompt_tokens, 0.95),
        prompt_tokens_p99: percentile(&prompt_tokens, 0.99),
        prompt_tokens_max: prompt_tokens.last().copied(),
        requested_tokens_max: requested_tokens.last().copied(),
    }
}

fn percentile(values: &[usize], percentile: f64) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values.get(index).copied()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use serde_json::json;

    #[test]
    fn parses_plain_prompt_as_user_message() {
        let value = json!({
            "id": "row-1",
            "family": "coding",
            "length_bucket": "short",
            "prompt": "Fix this bug."
        });

        let case = prompt_case_from_value(0, &value).unwrap();

        assert_eq!(case.prompt_id.as_deref(), Some("row-1"));
        assert_eq!(case.family.as_deref(), Some("coding"));
        assert_eq!(case.length_bucket.as_deref(), Some("short"));
        assert_eq!(case.messages[0].role, "user");
        assert_eq!(case.messages[0].content, "Fix this bug.");
    }

    #[test]
    fn preserves_chat_messages_and_text_parts() {
        let value = json!({
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": [
                    {"type": "text", "text": "First part."},
                    {"type": "image_url", "image_url": {"url": "ignored"}},
                    {"type": "text", "text": "Second part."}
                ]}
            ]
        });

        let case = prompt_case_from_value(0, &value).unwrap();

        assert_eq!(case.messages.len(), 2);
        assert_eq!(case.messages[0].role, "system");
        assert_eq!(case.messages[0].content, "You are terse.");
        assert_eq!(case.messages[1].role, "user");
        assert_eq!(case.messages[1].content, "First part.\nSecond part.");
    }

    #[test]
    fn summarizes_context_fit() {
        let args = TokenLengthsArgs {
            model_path: PathBuf::from("model.gguf"),
            prompt_corpus: PathBuf::from("corpus.jsonl"),
            ctx_size: 10,
            generation_limit: 4,
            layer_end: 40,
            n_gpu_layers: 0,
            enable_thinking: false,
            output_tsv: PathBuf::from("prompt-lengths.tsv"),
            summary_json: None,
        };
        let rows = vec![
            TokenLengthRow {
                sequence: 0,
                prompt_id: None,
                family: None,
                length_bucket: None,
                prompt_chars: 5,
                rendered_chars: 7,
                prompt_tokens: 3,
                generation_limit: 4,
                requested_tokens: 7,
                ctx_size: 10,
                fits_context: true,
            },
            TokenLengthRow {
                sequence: 1,
                prompt_id: None,
                family: None,
                length_bucket: None,
                prompt_chars: 10,
                rendered_chars: 12,
                prompt_tokens: 8,
                generation_limit: 4,
                requested_tokens: 12,
                ctx_size: 10,
                fits_context: false,
            },
        ];

        let summary = summarize(&args, &rows);

        assert_eq!(summary.row_count, 2);
        assert_eq!(summary.fits_context, 1);
        assert_eq!(summary.exceeds_context, 1);
        assert_eq!(summary.prompt_tokens_max, Some(8));
        assert_eq!(summary.requested_tokens_max, Some(12));
    }
}
