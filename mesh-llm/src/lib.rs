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

mod llama_ffi;

use llama_ffi::*;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

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
                // Stream done — grab usage
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
    /// In-process: direct llama.cpp FFI.
    Solo(Mutex<SoloInner>),
    /// Remote: HTTP client to an OpenAI-compatible endpoint.
    Remote {
        base_url: String,
        client: reqwest::Client,
        model: Option<String>,
    },
}

impl Engine {
    // ── Solo: in-process inference ──

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

    // ── Connect: remote server ──

    /// Connect to a running mesh-llm instance (or any OpenAI-compatible server).
    ///
    /// ```no_run
    /// # tokio_test::block_on(async {
    /// let engine = mesh_llm::Engine::connect("http://localhost:9337", None)?;
    /// # Ok::<(), String>(())
    /// # });
    /// ```
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

    // ── Auto: mesh if available, else solo ──

    /// Try to connect to a local mesh-llm instance, fall back to solo.
    ///
    /// Checks `localhost:9337` for a running mesh. If found, uses it.
    /// Otherwise loads the model in-process.
    pub async fn auto(model_path: impl AsRef<Path>) -> Result<Self, String> {
        // Quick probe: is mesh-llm running?
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

    // ── Inference API (same for all backends) ──

    /// Send a chat completion request. Returns a stream of token pieces.
    pub fn chat(&self, messages: &[Message<'_>]) -> TokenStream {
        let (tx, rx) = mpsc::channel(256);
        let (usage_tx, usage_rx) = tokio::sync::oneshot::channel();
        let backend = self.backend.clone();
        let msgs: Vec<(String, String)> = messages
            .iter()
            .map(|(r, c)| (r.to_string(), c.to_string()))
            .collect();

        tokio::spawn(async move {
            match &*backend {
                Backend::Solo(_) => {
                    let backend2 = backend.clone();
                    let tx2 = tx;
                    tokio::task::spawn_blocking(move || {
                        if let Backend::Solo(ref inner) = *backend2 {
                            let mut eng = inner.blocking_lock();
                            match eng.generate(&msgs, 4096) {
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
                    let body = build_request_body(&msgs, model.as_deref(), true);
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
                        let (pieces, _usage) = eng.generate(&msgs, 4096)?;
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
                let body = build_request_body(&msgs, model.as_deref(), false);
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
    req
    .to_string()
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
                    // llama-server sends usage in the final chunk
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
// Solo backend — direct llama.cpp FFI
// ══════════════════════════════════════════════════════════════════════

struct SoloInner {
    model: *mut llama_model,
    ctx: *mut llama_context,
    vocab: *const llama_vocab,
    sampler: *mut llama_sampler,
    chat_template: Option<CString>,
    n_ctx: u32,
}

unsafe impl Send for SoloInner {}

impl SoloInner {
    fn load(path: &Path, n_ctx: u32) -> Result<Self, String> {
        let path_cstr =
            CString::new(path.to_str().ok_or("invalid path")?).map_err(|e| e.to_string())?;

        unsafe {
            llama_backend_init();

            let mparams = llama_model_default_params();
            let model = llama_model_load_from_file(path_cstr.as_ptr(), mparams);
            if model.is_null() {
                return Err(format!("failed to load: {}", path.display()));
            }

            let mut cparams = llama_context_default_params();
            cparams.n_ctx = n_ctx;
            let ctx = llama_init_from_model(model, cparams);
            if ctx.is_null() {
                llama_model_free(model);
                return Err("failed to create context".into());
            }

            let vocab = llama_model_get_vocab(model);
            let actual_ctx = llama_n_ctx(ctx);

            let tmpl_ptr = llama_model_chat_template(model, std::ptr::null());
            let chat_template = if !tmpl_ptr.is_null() {
                Some(CString::new(CStr::from_ptr(tmpl_ptr).to_bytes()).unwrap())
            } else {
                None
            };

            let sparams = llama_sampler_chain_default_params();
            let sampler = llama_sampler_chain_init(sparams);
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7));
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(rand::random::<u32>()));

            Ok(SoloInner {
                model,
                ctx,
                vocab,
                sampler,
                chat_template,
                n_ctx: actual_ctx,
            })
        }
    }

    fn generate(
        &mut self,
        messages: &[(String, String)],
        max_tokens: usize,
    ) -> Result<(Vec<String>, Usage), String> {
        let prompt = self.apply_chat_template(messages)?;
        let tokens = self.tokenize(&prompt, false);
        let n_prompt = tokens.len();

        if tokens.is_empty() {
            return Err("empty prompt".into());
        }
        if n_prompt >= self.n_ctx as usize {
            return Err(format!(
                "prompt ({n_prompt}) exceeds context ({})",
                self.n_ctx
            ));
        }

        // Clear KV cache
        unsafe {
            let mem = llama_get_memory(self.ctx);
            if !mem.is_null() {
                llama_memory_clear(mem, true);
            }
        }

        // Prefill
        let batch = unsafe { llama_batch_get_one(tokens.as_ptr(), tokens.len() as i32) };
        if unsafe { llama_decode(self.ctx, batch) } != 0 {
            return Err("prefill decode failed".into());
        }

        // Generate
        let effective_max = max_tokens.min((self.n_ctx as usize).saturating_sub(n_prompt));
        let mut pieces = Vec::new();
        let mut n_gen = 0usize;

        for _ in 0..effective_max {
            let token = unsafe { llama_sampler_sample(self.sampler, self.ctx, -1) };

            if unsafe { llama_vocab_is_eog(self.vocab, token) } {
                break;
            }

            n_gen += 1;
            pieces.push(self.token_to_str(token));

            let batch = unsafe { llama_batch_get_one(&token as *const _, 1) };
            if unsafe { llama_decode(self.ctx, batch) } != 0 {
                return Err("decode failed".into());
            }
        }

        Ok((pieces, Usage {
            prompt_tokens: n_prompt,
            completion_tokens: n_gen,
            total_tokens: n_prompt + n_gen,
        }))
    }

    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String, String> {
        let c_roles: Vec<CString> = messages
            .iter()
            .map(|(r, _)| CString::new(r.as_str()).unwrap())
            .collect();
        let c_contents: Vec<CString> = messages
            .iter()
            .map(|(_, c)| CString::new(c.as_str()).unwrap())
            .collect();
        let c_msgs: Vec<llama_chat_message> = c_roles
            .iter()
            .zip(c_contents.iter())
            .map(|(r, c)| llama_chat_message {
                role: r.as_ptr(),
                content: c.as_ptr(),
            })
            .collect();

        let tmpl_ptr = self
            .chat_template
            .as_ref()
            .map(|t| t.as_ptr())
            .unwrap_or(std::ptr::null());

        let needed = unsafe {
            llama_chat_apply_template(
                tmpl_ptr,
                c_msgs.as_ptr(),
                c_msgs.len(),
                true,
                std::ptr::null_mut(),
                0,
            )
        };

        if needed < 0 {
            let mut prompt = String::new();
            for (role, content) in messages {
                prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
            prompt.push_str("<|im_start|>assistant\n");
            return Ok(prompt);
        }

        let mut buf = vec![0u8; (needed + 1) as usize];
        let written = unsafe {
            llama_chat_apply_template(
                tmpl_ptr,
                c_msgs.as_ptr(),
                c_msgs.len(),
                true,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
            )
        };
        if written < 0 {
            return Err("chat template failed".into());
        }
        buf.truncate(written as usize);
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn tokenize(&self, text: &str, add_special: bool) -> Vec<llama_token> {
        let ctext = CString::new(text).unwrap();
        let mut tokens = vec![0i32; text.len() + 128];
        let n = unsafe {
            llama_tokenize(
                self.vocab,
                ctext.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
                true,
            )
        };
        if n < 0 {
            tokens.resize((-n) as usize, 0);
            let n2 = unsafe {
                llama_tokenize(
                    self.vocab,
                    ctext.as_ptr(),
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    add_special,
                    true,
                )
            };
            tokens.truncate(n2.max(0) as usize);
        } else {
            tokens.truncate(n as usize);
        }
        tokens
    }

    fn token_to_str(&self, token: llama_token) -> String {
        unsafe {
            let mut buf = vec![0u8; 256];
            let n = llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                0,
                false,
            );
            if n < 0 {
                buf.resize((-n) as usize, 0);
                let n2 = llama_token_to_piece(
                    self.vocab,
                    token,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    0,
                    false,
                );
                buf.truncate(n2.max(0) as usize);
            } else {
                buf.truncate(n as usize);
            }
            String::from_utf8_lossy(&buf).into_owned()
        }
    }
}

impl Drop for SoloInner {
    fn drop(&mut self) {
        unsafe {
            if !self.sampler.is_null() {
                llama_sampler_free(self.sampler);
            }
            if !self.ctx.is_null() {
                llama_free(self.ctx);
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_message_type() {
        let _msg: Message = ("user", "hello");
    }
}
