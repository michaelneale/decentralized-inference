//! mesh-llm: distributed LLM inference.
//!
//! Three ways to use:
//!
//! - `Engine::solo(path)` — load a model in-process, direct GPU inference
//! - `Engine::connect(url)` — connect to a running mesh-llm (or any OpenAI-compatible server)
//! - `Engine::auto(path)` — connect to mesh if available, fall back to solo
//!
//! All three return the same `Engine` with the same `.chat()` / `.chat_blocking()` API.
//!
//! ```no_run
//! use mesh_llm::Engine;
//!
//! # tokio_test::block_on(async {
//! let engine = Engine::auto("~/.models/Qwen2.5-3B-Instruct-Q4_K_M.gguf").await?;
//! let mut stream = engine.chat(&[("user", "What is 2+2?")]);
//! while let Some(token) = stream.next().await {
//!     print!("{}", token);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # });
//! ```

use std::path::Path;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

// Re-export llama-cpp-2 for consumers who need lower-level access
pub use llama_cpp_2;

/// Chat message: (role, content).
pub type Message<'a> = (&'a str, &'a str);

/// Token usage statistics.
#[derive(Clone, Debug, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Token stream from a generation request.
pub struct TokenStream {
    rx: mpsc::Receiver<String>,
    usage_rx: tokio::sync::oneshot::Receiver<Usage>,
    usage: Option<Usage>,
}

impl TokenStream {
    /// Receive the next token piece, or None when generation is done.
    pub async fn next(&mut self) -> Option<String> {
        match self.rx.recv().await {
            Some(tok) => Some(tok),
            None => {
                if self.usage.is_none() {
                    self.usage = self.usage_rx.try_recv().ok();
                }
                None
            }
        }
    }

    /// Collect all tokens into a single string.
    pub async fn collect(mut self) -> String {
        let mut out = String::new();
        while let Some(tok) = self.next().await {
            out.push_str(&tok);
        }
        out
    }

    /// Token usage (available after stream completes).
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }
}

// ══════════════════════════════════════════════════════════════════════
// Engine — unified inference API
// ══════════════════════════════════════════════════════════════════════

/// Inference engine. Wraps either in-process llama.cpp or a remote server.
/// Clone is cheap — all clones share the same backend.
#[derive(Clone)]
pub struct Engine {
    backend: Arc<Backend>,
}

enum Backend {
    Solo(Mutex<SoloInner>),
    Remote {
        base_url: String,
        client: reqwest::Client,
        model: Option<String>,
    },
}

impl Engine {
    /// Load a model for solo (in-process) inference.
    pub fn solo(model_path: impl AsRef<Path>) -> Result<Self, String> {
        Self::solo_with_ctx(model_path, 8192)
    }

    /// Load a model with a specific context size.
    pub fn solo_with_ctx(model_path: impl AsRef<Path>, n_ctx: u32) -> Result<Self, String> {
        let inner = SoloInner::load(model_path.as_ref(), n_ctx)?;
        let model_name = model_path
            .as_ref()
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        eprintln!("mesh-llm: loaded {} (solo, n_ctx={})", model_name, inner.n_ctx);
        Ok(Engine {
            backend: Arc::new(Backend::Solo(Mutex::new(inner))),
        })
    }

    /// Connect to a running mesh-llm instance (or any OpenAI-compatible server).
    pub fn connect(base_url: &str, model: Option<&str>) -> Result<Self, String> {
        let client = reqwest::Client::new();
        let base = base_url.trim_end_matches('/').to_string();
        eprintln!(
            "mesh-llm: connected to {} (model: {})",
            base,
            model.unwrap_or("default")
        );
        Ok(Engine {
            backend: Arc::new(Backend::Remote {
                base_url: base,
                client,
                model: model.map(|s| s.to_string()),
            }),
        })
    }

    /// Try to connect to a local mesh-llm instance, fall back to solo.
    pub async fn auto(model_path: impl AsRef<Path>) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
            .map_err(|e| e.to_string())?;

        match client.get("http://localhost:9337/v1/models").send().await {
            Ok(resp) if resp.status().is_success() => {
                eprintln!("mesh-llm: found running mesh on localhost:9337");
                Self::connect("http://localhost:9337", None)
            }
            _ => {
                eprintln!("mesh-llm: no mesh found, loading solo");
                Self::solo(model_path)
            }
        }
    }

    /// Send a chat completion request. Returns a stream of token pieces.
    pub fn chat(&self, messages: &[Message<'_>]) -> TokenStream {
        self.chat_with_tools(messages, None)
    }

    /// Send a chat completion with tools (OpenAI-format JSON array string).
    pub fn chat_with_tools(&self, messages: &[Message<'_>], tools_json: Option<&str>) -> TokenStream {
        let (tx, rx) = mpsc::channel(256);
        let (usage_tx, usage_rx) = tokio::sync::oneshot::channel();
        let backend = self.backend.clone();
        let msgs: Vec<(String, String)> = messages
            .iter()
            .map(|(r, c)| (r.to_string(), c.to_string()))
            .collect();
        let tools = tools_json.map(|s| s.to_string());

        tokio::spawn(async move {
            match &*backend {
                Backend::Solo(_) => {
                    let backend2 = backend.clone();
                    let tx2 = tx;
                    tokio::task::spawn_blocking(move || {
                        if let Backend::Solo(ref inner) = *backend2 {
                            let mut eng = inner.blocking_lock();
                            match eng.generate(&msgs, tools.as_deref(), 4096) {
                                Ok((pieces, usage)) => {
                                    for piece in pieces {
                                        if tx2.blocking_send(piece).is_err() {
                                            break;
                                        }
                                    }
                                    let _ = usage_tx.send(usage);
                                }
                                Err(e) => {
                                    let _ = tx2.blocking_send(format!("[error: {e}]"));
                                }
                            }
                        }
                    })
                    .await
                    .ok();
                }
                Backend::Remote {
                    base_url,
                    client,
                    model,
                } => {
                    let body = build_request_body(&msgs, model.as_deref(), tools.as_deref(), true);
                    match stream_sse(client, base_url, &body, &tx).await {
                        Ok(usage) => {
                            let _ = usage_tx.send(usage);
                        }
                        Err(e) => {
                            let _ = tx.send(format!("[error: {e}]")).await;
                        }
                    }
                }
            }
        });

        TokenStream { rx, usage_rx, usage: None }
    }

    /// Non-streaming chat completion. Returns the full response text.
    pub async fn chat_blocking(&self, messages: &[Message<'_>]) -> Result<String, String> {
        match &*self.backend {
            Backend::Solo(_) => {
                let backend = self.backend.clone();
                let msgs: Vec<(String, String)> = messages
                    .iter()
                    .map(|(r, c)| (r.to_string(), c.to_string()))
                    .collect();
                tokio::task::spawn_blocking(move || {
                    if let Backend::Solo(ref inner) = *backend {
                        let mut eng = inner.blocking_lock();
                        let (pieces, _usage) = eng.generate(&msgs, None, 4096)?;
                        Ok(pieces.join(""))
                    } else {
                        unreachable!()
                    }
                })
                .await
                .map_err(|e| format!("task error: {e}"))?
            }
            Backend::Remote {
                base_url,
                client,
                model,
            } => {
                let msgs: Vec<(String, String)> = messages
                    .iter()
                    .map(|(r, c)| (r.to_string(), c.to_string()))
                    .collect();
                let body = build_request_body(&msgs, model.as_deref(), None, false);
                let resp = client
                    .post(format!("{base_url}/v1/chat/completions"))
                    .header("Content-Type", "application/json")
                    .body(body)
                    .send()
                    .await
                    .map_err(|e| format!("request failed: {e}"))?;

                let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
                json["choices"][0]["message"]["content"]
                    .as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| format!("unexpected response: {json}"))
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// Remote helpers
// ══════════════════════════════════════════════════════════════════════

fn build_request_body(
    messages: &[(String, String)],
    model: Option<&str>,
    tools: Option<&str>,
    stream: bool,
) -> String {
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|(r, c)| serde_json::json!({"role": r, "content": c}))
        .collect();
    let mut req = serde_json::json!({
        "model": model.unwrap_or("default"),
        "messages": msgs,
        "stream": stream,
    });
    if stream {
        req["stream_options"] = serde_json::json!({"include_usage": true});
    }
    if let Some(tools_str) = tools {
        if let Ok(tools_val) = serde_json::from_str::<serde_json::Value>(tools_str) {
            req["tools"] = tools_val;
        }
    }
    req.to_string()
}

async fn stream_sse(
    client: &reqwest::Client,
    base_url: &str,
    body: &str,
    tx: &mpsc::Sender<String>,
) -> Result<Usage, String> {
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("server returned {}", resp.status()));
    }

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut usage = Usage::default();

    use tokio_stream::StreamExt;
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| e.to_string())?;
        buf.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(line_end) = buf.find('\n') {
            let line = buf[..line_end].trim_end_matches('\r').to_string();
            buf = buf[line_end + 1..].to_string();

            if line == "data: [DONE]" {
                return Ok(usage);
            }
            if let Some(json_str) = line.strip_prefix("data: ") {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(content) = v["choices"][0]["delta"]["content"].as_str() {
                        if !content.is_empty() && tx.send(content.to_string()).await.is_err() {
                            return Ok(usage);
                        }
                    }
                    if let Some(u) = v.get("usage") {
                        usage.prompt_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as usize;
                        usage.completion_tokens = u["completion_tokens"].as_u64().unwrap_or(0) as usize;
                        usage.total_tokens = u["total_tokens"].as_u64().unwrap_or(0) as usize;
                    }
                }
            }
        }
    }
    Ok(usage)
}

// ══════════════════════════════════════════════════════════════════════
// Solo backend — llama-cpp-2 safe wrappers
// ══════════════════════════════════════════════════════════════════════

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;

struct SoloInner {
    backend: LlamaBackend,
    model: LlamaModel,
    template: LlamaChatTemplate,
    n_ctx: u32,
}

// LlamaBackend and LlamaModel are Send but not marked as such in all versions
unsafe impl Send for SoloInner {}

impl SoloInner {
    fn load(path: &Path, n_ctx: u32) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("model not found: {}", path.display()));
        }

        let backend = LlamaBackend::init().map_err(|e| format!("backend init: {e}"))?;
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, path, &params)
            .map_err(|e| format!("load model: {e}"))?;

        let template = match model.chat_template(None) {
            Ok(t) => t,
            Err(_) => LlamaChatTemplate::new("chatml")
                .map_err(|e| format!("fallback template: {e}"))?,
        };

        let actual_ctx = model.n_ctx_train().min(n_ctx);

        Ok(SoloInner {
            backend,
            model,
            template,
            n_ctx: actual_ctx,
        })
    }

    fn generate(
        &mut self,
        messages: &[(String, String)],
        tools_json: Option<&str>,
        max_tokens: usize,
    ) -> Result<(Vec<String>, Usage), String> {
        // Build chat messages
        let chat_messages: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|(r, c)| LlamaChatMessage::new(r.clone(), c.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("message error: {e}"))?;

        // Apply template — with tools if provided
        let (prompt, additional_stops) = if tools_json.is_some() {
            // Use tools-aware template (jinja + grammar)
            let result = self
                .model
                .apply_chat_template_with_tools_oaicompat(
                    &self.template,
                    &chat_messages,
                    tools_json,
                    None,
                    true,
                )
                .map_err(|e| format!("template with tools: {e}"))?;
            (result.prompt, result.additional_stops)
        } else {
            // Basic template — no tools
            let prompt = self
                .model
                .apply_chat_template(&self.template, &chat_messages, true)
                .map_err(|e| format!("template: {e}"))?;
            (prompt, Vec::new())
        };

        // Tokenize
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Never)
            .map_err(|e| format!("tokenize: {e}"))?;
        let n_prompt = tokens.len();

        if tokens.is_empty() {
            return Err("empty prompt".into());
        }

        let effective_ctx = (n_prompt + max_tokens).min(self.n_ctx as usize);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(effective_ctx as u32));

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("create context: {e}"))?;

        // Prefill
        let n_batch = ctx.n_batch() as usize;
        for chunk in tokens.chunks(n_batch) {
            let mut batch = LlamaBatch::get_one(chunk)
                .map_err(|e| format!("batch: {e}"))?;
            ctx.decode(&mut batch)
                .map_err(|e| format!("prefill: {e}"))?;
        }

        // Generate
        let mut sampler = build_sampler();
        let effective_max = max_tokens.min(effective_ctx.saturating_sub(n_prompt));
        let mut pieces = Vec::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut generated_text = String::new();

        for _ in 0..effective_max {
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)
                .map_err(|e| format!("detokenize: {e}"))?;

            if !piece.is_empty() {
                generated_text.push_str(&piece);
                pieces.push(piece);
            }

            // Check additional stop sequences
            if additional_stops
                .iter()
                .any(|stop| !stop.is_empty() && generated_text.ends_with(stop))
            {
                break;
            }

            let next = [token];
            let mut batch = LlamaBatch::get_one(&next)
                .map_err(|e| format!("batch: {e}"))?;
            ctx.decode(&mut batch)
                .map_err(|e| format!("decode: {e}"))?;
        }

        let n_gen = pieces.len();
        Ok((
            pieces,
            Usage {
                prompt_tokens: n_prompt,
                completion_tokens: n_gen,
                total_tokens: n_prompt + n_gen,
            },
        ))
    }
}

fn build_sampler() -> LlamaSampler {
    LlamaSampler::chain_simple([
        LlamaSampler::temp(0.7),
        LlamaSampler::top_p(0.9, 1),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::dist(rand::random::<u32>()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_message_type() {
        let _msg: Message = ("user", "hello");
    }
}
