use anyhow::{anyhow, Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use clap::{Parser, Subcommand};
use mesh_llm_plugin::{
    json_schema_tool, plugin_server_info, PluginMetadata, PluginRuntime, PluginStartupPolicy,
    SimplePlugin, ToolRouter,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    cell::RefCell,
    convert::Infallible,
    net::SocketAddr,
    os::unix::fs::symlink,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    net::TcpListener,
    sync::{mpsc, Mutex},
};
use tokio_stream::wrappers::ReceiverStream;

#[cfg(feature = "native-mlx")]
mod gemma2;

#[cfg(feature = "native-mlx")]
mod gemma3_text;

#[cfg(feature = "native-mlx")]
mod mistral;

#[cfg(feature = "native-mlx")]
mod quantized;

#[cfg(feature = "native-mlx")]
use mlx_lm::{
    cache::ConcatKeyValueCache,
    models::{llama, qwen3},
};
#[cfg(feature = "native-mlx")]
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Conversation, Tokenizer,
};
#[cfg(feature = "native-mlx")]
use mlx_rs::{
    module::Module,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

const PLUGIN_ID: &str = "mlx-native";

thread_local! {
    static ACTIVE_PROFILE: RefCell<Option<RequestProfile>> = const { RefCell::new(None) };
}

#[derive(Parser, Debug)]
#[command(name = "mesh-llm-mlx")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// MLX model directory.
    #[arg(long)]
    model: PathBuf,

    /// Local HTTP address for the OpenAI-compatible endpoint.
    #[arg(long, default_value = "127.0.0.1:0")]
    listen: SocketAddr,

    /// Plugin id to advertise to mesh-llm.
    #[arg(long, default_value = PLUGIN_ID)]
    plugin_id: String,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run only the local OpenAI-compatible HTTP server.
    Serve,
}

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<EngineState>>,
}

struct EngineState {
    model_path: PathBuf,
    model_id: String,
    listen_addr: SocketAddr,
    backend: BackendState,
}

#[cfg_attr(feature = "native-mlx", allow(dead_code))]
enum BackendState {
    #[cfg(feature = "native-mlx")]
    Ready(NativeBackend),
    Disabled {
        reason: String,
    },
}

#[cfg(feature = "native-mlx")]
struct NativeBackend {
    family: ModelFamily,
    engine: Box<dyn FamilyAdapter + Send>,
}

#[cfg(feature = "native-mlx")]
struct LlamaEngine {
    tokenizer: Tokenizer,
    chat_template: Option<String>,
    model: llama::Model,
    stop_ids: Vec<u32>,
    prompt_cache: Option<PromptCacheEntry>,
    _compat_dir: Option<tempfile::TempDir>,
}

#[cfg(feature = "native-mlx")]
struct Gemma2Engine {
    tokenizer: Tokenizer,
    chat_template: Option<String>,
    model: gemma2::Model,
    stop_ids: Vec<u32>,
}

#[cfg(feature = "native-mlx")]
struct Gemma3TextEngine {
    tokenizer: Tokenizer,
    chat_template: Option<String>,
    model: gemma3_text::Model,
    stop_ids: Vec<u32>,
}

#[cfg(feature = "native-mlx")]
struct MistralEngine {
    tokenizer: Tokenizer,
    chat_template: Option<String>,
    model: mistral::Model,
    stop_ids: Vec<u32>,
}

#[cfg(feature = "native-mlx")]
struct Qwen3Engine {
    tokenizer: Tokenizer,
    chat_template: String,
    model: qwen3::Model,
    stop_ids: Vec<u32>,
    prompt_cache: Option<PromptCacheEntry>,
}

#[cfg(feature = "native-mlx")]
#[derive(Clone)]
struct PromptCacheEntry {
    prompt_ids: Vec<u32>,
    prefill_cache: Vec<Option<ConcatKeyValueCache>>,
    prefill_logits: Array,
}

#[cfg(feature = "native-mlx")]
type GenerationResult = Result<(String, &'static str, usize, usize), ApiError>;

#[cfg(feature = "native-mlx")]
type StreamingGenerationResult = Result<GenerationStats, ApiError>;

#[cfg(feature = "native-mlx")]
#[derive(Debug, Clone, Copy)]
struct GenerationStats {
    finish_reason: &'static str,
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[cfg(feature = "native-mlx")]
#[derive(Debug, Default, Clone, Copy)]
struct DecodeTimings {
    first_token_ms: Option<f64>,
    decode_ms: f64,
}

#[cfg(feature = "native-mlx")]
#[derive(Debug)]
struct RequestProfile {
    route: &'static str,
    model_id: String,
    family: &'static str,
    streaming: bool,
    started_at: Instant,
    render_ms: f64,
    build_ms: f64,
    encode_ms: f64,
    reused_prompt_tokens: usize,
    first_token_ms: Option<f64>,
    first_chunk_ms: Option<f64>,
    decode_ms: f64,
}

#[cfg(feature = "native-mlx")]
trait FamilyAdapter {
    fn generate_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult;
    fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult;
    fn stream_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult;
    fn stream_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult;
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
struct EmptyArgs {}

#[derive(Debug, Serialize)]
struct StatusSummary {
    plugin: String,
    model_id: String,
    model_path: String,
    listen_addr: String,
    backend: String,
    healthy: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct ModelsSummary {
    data: Vec<ModelInfo>,
    object: &'static str,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    capabilities: Vec<&'static str>,
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct CompletionChoice {
    index: u32,
    text: String,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: u32,
    message: ChatResponseMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct ChatResponseMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct CompletionStreamResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionStreamChoice>,
}

#[derive(Debug, Serialize)]
struct CompletionStreamChoice {
    index: u32,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionStreamResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatStreamChoice>,
}

#[derive(Debug, Serialize)]
struct ChatStreamChoice {
    index: u32,
    delta: ChatStreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Default, Serialize)]
struct ChatStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

#[cfg(feature = "native-mlx")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelFamily {
    Gemma2,
    Gemma3Text,
    Llama,
    Mistral,
    Qwen2,
    Qwen3,
}

#[cfg(feature = "native-mlx")]
struct FamilySpec {
    family: ModelFamily,
    model_types: &'static [&'static str],
    backend_name: &'static str,
    #[cfg(test)]
    test_model_path_env_var: &'static str,
    #[cfg(test)]
    default_test_model_dir: Option<&'static str>,
    #[cfg(test)]
    test_repo_env_var: &'static str,
    #[cfg(test)]
    default_test_repo_id: Option<&'static str>,
}

#[cfg(feature = "native-mlx")]
const FAMILY_SPECS: &[FamilySpec] = &[
    FamilySpec {
        family: ModelFamily::Gemma2,
        model_types: &["gemma2"],
        backend_name: "mesh-llm-mlx:gemma2",
        #[cfg(test)]
        test_model_path_env_var: "MESH_LLM_MLX_TEST_GEMMA2_MODEL",
        #[cfg(test)]
        default_test_model_dir: None,
        #[cfg(test)]
        test_repo_env_var: "MESH_LLM_MLX_TEST_GEMMA2_REPO",
        #[cfg(test)]
        default_test_repo_id: Some("mlx-community/gemma-2-2b-it-4bit"),
    },
    FamilySpec {
        family: ModelFamily::Gemma3Text,
        model_types: &["gemma3_text"],
        backend_name: "mesh-llm-mlx:gemma3_text",
        #[cfg(test)]
        test_model_path_env_var: "MESH_LLM_MLX_TEST_GEMMA3_TEXT_MODEL",
        #[cfg(test)]
        default_test_model_dir: None,
        #[cfg(test)]
        test_repo_env_var: "MESH_LLM_MLX_TEST_GEMMA3_TEXT_REPO",
        #[cfg(test)]
        default_test_repo_id: Some("mlx-community/gemma-3-270m-it-4bit"),
    },
    FamilySpec {
        family: ModelFamily::Llama,
        model_types: &["llama"],
        backend_name: "mlx-rs:llama",
        #[cfg(test)]
        test_model_path_env_var: "MESH_LLM_MLX_TEST_LLAMA_MODEL",
        #[cfg(test)]
        default_test_model_dir: None,
        #[cfg(test)]
        test_repo_env_var: "MESH_LLM_MLX_TEST_LLAMA_REPO",
        #[cfg(test)]
        default_test_repo_id: Some("mlx-community/Llama-3.2-1B-Instruct-bf16"),
    },
    FamilySpec {
        family: ModelFamily::Mistral,
        model_types: &["mistral"],
        backend_name: "mesh-llm-mlx:mistral",
        #[cfg(test)]
        test_model_path_env_var: "MESH_LLM_MLX_TEST_MISTRAL_MODEL",
        #[cfg(test)]
        default_test_model_dir: None,
        #[cfg(test)]
        test_repo_env_var: "MESH_LLM_MLX_TEST_MISTRAL_REPO",
        #[cfg(test)]
        default_test_repo_id: Some("mlx-community/Mistral-7B-Instruct-v0.2-4bit"),
    },
    FamilySpec {
        family: ModelFamily::Qwen2,
        model_types: &["qwen2"],
        backend_name: "mlx-rs:qwen2",
        #[cfg(test)]
        test_model_path_env_var: "MESH_LLM_MLX_TEST_QWEN2_MODEL",
        #[cfg(test)]
        default_test_model_dir: None,
        #[cfg(test)]
        test_repo_env_var: "MESH_LLM_MLX_TEST_QWEN2_REPO",
        #[cfg(test)]
        default_test_repo_id: Some("mlx-community/Qwen2.5-0.5B-Instruct-bf16"),
    },
    FamilySpec {
        family: ModelFamily::Qwen3,
        model_types: &["qwen3"],
        backend_name: "mlx-rs:qwen3",
        #[cfg(test)]
        test_model_path_env_var: "MESH_LLM_MLX_TEST_QWEN3_MODEL",
        #[cfg(test)]
        default_test_model_dir: Some("Qwen3-0.6B-bf16"),
        #[cfg(test)]
        test_repo_env_var: "MESH_LLM_MLX_TEST_QWEN3_REPO",
        #[cfg(test)]
        default_test_repo_id: Some("mlx-community/Qwen3-0.6B-bf16"),
    },
];

#[cfg(feature = "native-mlx")]
impl ModelFamily {
    fn spec(self) -> &'static FamilySpec {
        FAMILY_SPECS
            .iter()
            .find(|spec| spec.family == self)
            .expect("supported model family should have a spec")
    }

    fn primary_model_type(self) -> &'static str {
        self.spec()
            .model_types
            .first()
            .copied()
            .expect("supported model family should advertise at least one model_type")
    }
}

#[cfg(feature = "native-mlx")]
fn profiling_enabled() -> bool {
    std::env::var_os("MESH_LLM_MLX_PROFILE").is_some()
}

#[cfg(feature = "native-mlx")]
fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(feature = "native-mlx")]
fn begin_request_profile(
    route: &'static str,
    model_id: &str,
    family: &'static str,
    streaming: bool,
) {
    if !profiling_enabled() {
        return;
    }
    ACTIVE_PROFILE.with(|slot| {
        *slot.borrow_mut() = Some(RequestProfile {
            route,
            model_id: model_id.to_string(),
            family,
            streaming,
            started_at: Instant::now(),
            render_ms: 0.0,
            build_ms: 0.0,
            encode_ms: 0.0,
            reused_prompt_tokens: 0,
            first_token_ms: None,
            first_chunk_ms: None,
            decode_ms: 0.0,
        });
    });
}

#[cfg(feature = "native-mlx")]
fn add_profile_stage(stage: &'static str, elapsed: Duration) {
    if !profiling_enabled() {
        return;
    }
    let elapsed_ms = duration_ms(elapsed);
    ACTIVE_PROFILE.with(|slot| {
        if let Some(profile) = slot.borrow_mut().as_mut() {
            match stage {
                "render" => profile.render_ms += elapsed_ms,
                "build" => profile.build_ms += elapsed_ms,
                "encode" => profile.encode_ms += elapsed_ms,
                _ => {}
            }
        }
    });
}

#[cfg(feature = "native-mlx")]
fn set_profile_decode_timings(timings: DecodeTimings) {
    if !profiling_enabled() {
        return;
    }
    ACTIVE_PROFILE.with(|slot| {
        if let Some(profile) = slot.borrow_mut().as_mut() {
            profile.first_token_ms = timings.first_token_ms;
            profile.decode_ms = timings.decode_ms;
        }
    });
}

#[cfg(feature = "native-mlx")]
fn set_profile_reused_prompt_tokens(reused_prompt_tokens: usize) {
    if !profiling_enabled() {
        return;
    }
    ACTIVE_PROFILE.with(|slot| {
        if let Some(profile) = slot.borrow_mut().as_mut() {
            profile.reused_prompt_tokens = reused_prompt_tokens;
        }
    });
}

#[cfg(feature = "native-mlx")]
fn mark_profile_first_chunk() {
    if !profiling_enabled() {
        return;
    }
    ACTIVE_PROFILE.with(|slot| {
        if let Some(profile) = slot.borrow_mut().as_mut() {
            if profile.first_chunk_ms.is_none() {
                profile.first_chunk_ms = Some(duration_ms(profile.started_at.elapsed()));
            }
        }
    });
}

#[cfg(feature = "native-mlx")]
fn finish_request_profile(
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &'static str,
) {
    if !profiling_enabled() {
        return;
    }
    ACTIVE_PROFILE.with(|slot| {
        if let Some(profile) = slot.borrow_mut().take() {
            let total_ms = duration_ms(profile.started_at.elapsed());
            let first_token = profile
                .first_token_ms
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "n/a".to_string());
            let first_chunk = profile
                .first_chunk_ms
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "n/a".to_string());
            eprintln!(
                "mlx-profile route={} family={} model={} stream={} encode_ms={:.1} render_ms={:.1} build_ms={:.1} ttft_ms={} reused_prompt_tokens={} first_chunk_ms={} decode_ms={:.1} total_ms={:.1} prompt_tokens={} completion_tokens={} finish_reason={}",
                profile.route,
                profile.family,
                profile.model_id,
                profile.streaming,
                profile.encode_ms,
                profile.render_ms,
                profile.build_ms,
                first_token,
                profile.reused_prompt_tokens,
                first_chunk,
                profile.decode_ms,
                total_ms,
                prompt_tokens,
                completion_tokens,
                finish_reason,
            );
        }
    });
}

#[cfg(feature = "native-mlx")]
fn clear_request_profile() {
    if !profiling_enabled() {
        return;
    }
    ACTIVE_PROFILE.with(|slot| {
        slot.borrow_mut().take();
    });
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    model_type: Option<String>,
}

fn main() -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to create tokio runtime")?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
    let cli = Cli::parse();
    let state = initialize_state(&cli).await?;
    let listen_addr = state.inner.lock().await.listen_addr;

    let server = tokio::spawn(run_http_server(state.clone(), listen_addr));

    match cli.command {
        Some(Command::Serve) => {
            server.await??;
            Ok(())
        }
        None => {
            PluginRuntime::run(build_plugin(cli.plugin_id, state)).await?;
            Ok(())
        }
    }
}

async fn initialize_state(cli: &Cli) -> Result<AppState> {
    let configured_model_path = cli.model.clone();
    let model_path = cli
        .model
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", cli.model.display()))?;
    let listener = TcpListener::bind(cli.listen)
        .await
        .with_context(|| format!("failed to bind {}", cli.listen))?;
    let listen_addr = listener
        .local_addr()
        .context("failed to read bound address")?;
    drop(listener);

    let model_id = model_id_from_configured_path(&configured_model_path)?;
    let backend = build_backend_state(&model_path).with_context(|| {
        format!(
            "failed to initialize native backend for {}",
            model_path.display()
        )
    })?;

    Ok(AppState {
        inner: Arc::new(Mutex::new(EngineState {
            model_path,
            model_id,
            listen_addr,
            backend,
        })),
    })
}

fn build_plugin(plugin_id: String, state: AppState) -> SimplePlugin {
    let health_state = state.clone();
    let status_state = state.clone();
    let models_state = state.clone();

    SimplePlugin::new(
        PluginMetadata::new(
            plugin_id,
            env!("CARGO_PKG_VERSION"),
            plugin_server_info(
                "mesh-mlx",
                env!("CARGO_PKG_VERSION"),
                "MLX Worker",
                "Runs a local OpenAI-compatible HTTP endpoint backed by native MLX.",
                Some(
                    "Use mlx-native.status to discover the local HTTP port and mlx-native.models to inspect the served model.",
                ),
            ),
        )
        .with_capabilities(vec!["local-http:openai".into(), "backend:mlx".into()])
        .with_startup_policy(PluginStartupPolicy::Any),
    )
    .with_tool_router(tool_router(status_state, models_state))
    .with_health(move |_context| {
        let state = health_state.clone();
        Box::pin(async move {
            let status = build_status_summary(&state).await;
            if status.healthy {
                Ok(format!("ok {}", status.listen_addr))
            } else {
                Ok(format!("degraded {}", status.detail))
            }
        })
    })
}

fn tool_router(status_state: AppState, models_state: AppState) -> ToolRouter {
    let mut router = ToolRouter::new();

    router.add_json_default(
        json_schema_tool::<EmptyArgs>("mlx-native.status", "Show local MLX backend status."),
        move |_params: EmptyArgs, _context| {
            let state = status_state.clone();
            Box::pin(async move { Ok(json!({ "status": build_status_summary(&state).await })) })
        },
    );

    router.add_json_default(
        json_schema_tool::<EmptyArgs>(
            "mlx-native.models",
            "List models served by the local MLX backend.",
        ),
        move |_params: EmptyArgs, _context| {
            let state = models_state.clone();
            Box::pin(async move { Ok(json!({ "models": build_models_summary(&state).await })) })
        },
    );

    router
}

async fn run_http_server(state: AppState, listen_addr: SocketAddr) -> Result<()> {
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/completions", post(completions_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state);

    let listener = TcpListener::bind(listen_addr)
        .await
        .with_context(|| format!("failed to bind {}", listen_addr))?;
    axum::serve(listener, app)
        .await
        .context("native MLX HTTP server exited unexpectedly")
}

async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    Json(build_status_summary(&state).await)
}

async fn models_handler(State(state): State<AppState>) -> impl IntoResponse {
    Json(build_models_summary(&state).await)
}

async fn completions_handler(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    if request.stream.unwrap_or(false) {
        return stream_completions_response(state, request).await;
    }

    let mut guard = state.inner.lock().await;
    guard.validate_requested_model(request.model.as_deref())?;
    #[cfg(feature = "native-mlx")]
    begin_request_profile(
        "/v1/completions",
        &guard.model_id,
        guard.backend_family_label(),
        false,
    );
    let (text, finish_reason, prompt_tokens, completion_tokens) = match guard.generate_completion(
        &request.prompt,
        request.max_tokens.unwrap_or(128),
        request.temperature.unwrap_or(0.0),
    ) {
        Ok(result) => result,
        Err(err) => {
            #[cfg(feature = "native-mlx")]
            clear_request_profile();
            return Err(err);
        }
    };
    #[cfg(feature = "native-mlx")]
    finish_request_profile(prompt_tokens, completion_tokens, finish_reason);

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", now_millis()),
        object: "text_completion",
        created: now_secs(),
        model: guard.model_id.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text,
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}

async fn chat_completions_handler(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if request.stream.unwrap_or(false) {
        return stream_chat_completions_response(state, request).await;
    }

    let mut guard = state.inner.lock().await;
    guard.validate_requested_model(request.model.as_deref())?;
    #[cfg(feature = "native-mlx")]
    begin_request_profile(
        "/v1/chat/completions",
        &guard.model_id,
        guard.backend_family_label(),
        false,
    );
    let (text, finish_reason, prompt_tokens, completion_tokens) = match guard.generate_chat(
        &request.messages,
        request.max_tokens.unwrap_or(128),
        request.temperature.unwrap_or(0.0),
    ) {
        Ok(result) => result,
        Err(err) => {
            #[cfg(feature = "native-mlx")]
            clear_request_profile();
            return Err(err);
        }
    };
    #[cfg(feature = "native-mlx")]
    finish_request_profile(prompt_tokens, completion_tokens, finish_reason);

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", now_millis()),
        object: "chat.completion",
        created: now_secs(),
        model: guard.model_id.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                role: "assistant",
                content: text,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}

fn json_sse_event<T: Serialize>(payload: &T) -> Result<Event, ApiError> {
    let data = serde_json::to_string(payload).map_err(|err| ApiError::internal(err.to_string()))?;
    Ok(Event::default().data(data))
}

fn done_sse_event() -> Event {
    Event::default().data("[DONE]")
}

fn completion_stream_chunk(
    id: &str,
    created: u64,
    model: &str,
    text: String,
) -> Result<Event, ApiError> {
    json_sse_event(&CompletionStreamResponse {
        id: id.to_string(),
        object: "text_completion",
        created,
        model: model.to_string(),
        choices: vec![CompletionStreamChoice {
            index: 0,
            text,
            finish_reason: None,
        }],
    })
}

fn completion_stream_finish(
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &'static str,
) -> Result<Event, ApiError> {
    json_sse_event(&CompletionStreamResponse {
        id: id.to_string(),
        object: "text_completion",
        created,
        model: model.to_string(),
        choices: vec![CompletionStreamChoice {
            index: 0,
            text: String::new(),
            finish_reason: Some(finish_reason),
        }],
    })
}

fn chat_stream_role_chunk(id: &str, created: u64, model: &str) -> Result<Event, ApiError> {
    json_sse_event(&ChatCompletionStreamResponse {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                role: Some("assistant"),
                content: None,
            },
            finish_reason: None,
        }],
    })
}

fn chat_stream_content_chunk(
    id: &str,
    created: u64,
    model: &str,
    content: String,
) -> Result<Event, ApiError> {
    json_sse_event(&ChatCompletionStreamResponse {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta {
                role: None,
                content: Some(content),
            },
            finish_reason: None,
        }],
    })
}

fn chat_stream_finish(
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &'static str,
) -> Result<Event, ApiError> {
    json_sse_event(&ChatCompletionStreamResponse {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta::default(),
            finish_reason: Some(finish_reason),
        }],
    })
}

async fn stream_completions_response(
    state: AppState,
    request: CompletionRequest,
) -> Result<Response, ApiError> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(16);
    let inner = state.inner.clone();
    let stream_id = format!("cmpl-{}", now_millis());
    let created = now_secs();

    tokio::task::spawn_blocking(move || {
        let mut guard = inner.blocking_lock();
        let model_id = guard.model_id.clone();
        let requested_model = request.model.clone();
        let prompt = request.prompt.clone();
        let max_tokens = request.max_tokens.unwrap_or(128);
        let temperature = request.temperature.unwrap_or(0.0);

        let send = |event: Result<Event, ApiError>| -> Result<(), ApiError> {
            tx.blocking_send(Ok(event?))
                .map_err(|_| ApiError::internal("stream receiver dropped"))?;
            Ok(())
        };

        let result: Result<(), ApiError> = (|| {
            guard.validate_requested_model(requested_model.as_deref())?;
            #[cfg(feature = "native-mlx")]
            begin_request_profile(
                "/v1/completions",
                &model_id,
                guard.backend_family_label(),
                true,
            );
            let mut on_chunk = |chunk: String| -> Result<(), ApiError> {
                if chunk.is_empty() {
                    return Ok(());
                }
                #[cfg(feature = "native-mlx")]
                mark_profile_first_chunk();
                send(completion_stream_chunk(
                    &stream_id, created, &model_id, chunk,
                ))
            };
            let stats = guard.stream_completion(&prompt, max_tokens, temperature, &mut on_chunk)?;
            #[cfg(feature = "native-mlx")]
            finish_request_profile(
                stats.prompt_tokens,
                stats.completion_tokens,
                stats.finish_reason,
            );
            send(completion_stream_finish(
                &stream_id,
                created,
                &model_id,
                stats.finish_reason,
            ))?;
            send(Ok(done_sse_event()))?;
            Ok(())
        })();

        if let Err(err) = result {
            #[cfg(feature = "native-mlx")]
            clear_request_profile();
            let _ = tx.blocking_send(Ok(Event::default().data(
                serde_json::to_string(&ErrorBody { error: err.message })
                    .unwrap_or_else(|_| "{\"error\":\"streaming failed\"}".to_string()),
            )));
            let _ = tx.blocking_send(Ok(done_sse_event()));
        }
    });

    Ok(Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response())
}

async fn stream_chat_completions_response(
    state: AppState,
    request: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(16);
    let inner = state.inner.clone();
    let stream_id = format!("chatcmpl-{}", now_millis());
    let created = now_secs();

    tokio::task::spawn_blocking(move || {
        let mut guard = inner.blocking_lock();
        let model_id = guard.model_id.clone();
        let requested_model = request.model.clone();
        let messages = request.messages;
        let max_tokens = request.max_tokens.unwrap_or(128);
        let temperature = request.temperature.unwrap_or(0.0);

        let send = |event: Result<Event, ApiError>| -> Result<(), ApiError> {
            tx.blocking_send(Ok(event?))
                .map_err(|_| ApiError::internal("stream receiver dropped"))?;
            Ok(())
        };

        let result: Result<(), ApiError> = (|| {
            guard.validate_requested_model(requested_model.as_deref())?;
            #[cfg(feature = "native-mlx")]
            begin_request_profile(
                "/v1/chat/completions",
                &model_id,
                guard.backend_family_label(),
                true,
            );
            send(chat_stream_role_chunk(&stream_id, created, &model_id))?;
            let mut on_chunk = |chunk: String| -> Result<(), ApiError> {
                if chunk.is_empty() {
                    return Ok(());
                }
                #[cfg(feature = "native-mlx")]
                mark_profile_first_chunk();
                send(chat_stream_content_chunk(
                    &stream_id, created, &model_id, chunk,
                ))
            };
            let stats = guard.stream_chat(&messages, max_tokens, temperature, &mut on_chunk)?;
            #[cfg(feature = "native-mlx")]
            finish_request_profile(
                stats.prompt_tokens,
                stats.completion_tokens,
                stats.finish_reason,
            );
            send(chat_stream_finish(
                &stream_id,
                created,
                &model_id,
                stats.finish_reason,
            ))?;
            send(Ok(done_sse_event()))?;
            Ok(())
        })();

        if let Err(err) = result {
            #[cfg(feature = "native-mlx")]
            clear_request_profile();
            let _ = tx.blocking_send(Ok(Event::default().data(
                serde_json::to_string(&ErrorBody { error: err.message })
                    .unwrap_or_else(|_| "{\"error\":\"streaming failed\"}".to_string()),
            )));
            let _ = tx.blocking_send(Ok(done_sse_event()));
        }
    });

    Ok(Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response())
}

impl EngineState {
    #[cfg(feature = "native-mlx")]
    fn backend_family_label(&self) -> &'static str {
        match &self.backend {
            BackendState::Ready(model) => model.family.primary_model_type(),
            BackendState::Disabled { .. } => "disabled",
        }
    }

    fn validate_requested_model(&self, requested: Option<&str>) -> Result<(), ApiError> {
        if let Some(requested) = requested {
            if requested != self.model_id {
                return Err(ApiError::bad_request(format!(
                    "unknown model '{}'; expected '{}'",
                    requested, self.model_id
                )));
            }
        }
        Ok(())
    }

    fn generate_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<(String, &'static str, usize, usize), ApiError> {
        match &mut self.backend {
            #[cfg(feature = "native-mlx")]
            BackendState::Ready(model) => {
                model
                    .engine
                    .generate_completion(prompt, max_tokens, temperature)
            }
            BackendState::Disabled { reason } => Err(ApiError::service_unavailable(reason.clone())),
        }
    }

    fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<(String, &'static str, usize, usize), ApiError> {
        match &mut self.backend {
            #[cfg(feature = "native-mlx")]
            BackendState::Ready(model) => {
                model
                    .engine
                    .generate_chat(messages, max_tokens, temperature)
            }
            BackendState::Disabled { reason } => Err(ApiError::service_unavailable(reason.clone())),
        }
    }

    fn stream_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        match &mut self.backend {
            #[cfg(feature = "native-mlx")]
            BackendState::Ready(model) => {
                model
                    .engine
                    .stream_completion(prompt, max_tokens, temperature, on_chunk)
            }
            BackendState::Disabled { reason } => Err(ApiError::service_unavailable(reason.clone())),
        }
    }

    fn stream_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        match &mut self.backend {
            #[cfg(feature = "native-mlx")]
            BackendState::Ready(model) => {
                model
                    .engine
                    .stream_chat(messages, max_tokens, temperature, on_chunk)
            }
            BackendState::Disabled { reason } => Err(ApiError::service_unavailable(reason.clone())),
        }
    }
}

#[cfg(feature = "native-mlx")]
impl FamilyAdapter for LlamaEngine {
    fn generate_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult {
        let prompt_ids = encode_prompt_ids(&mut self.tokenizer, prompt)?;
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (text, finish_reason, completion_tokens) = generate_llama_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            &mut self.prompt_cache,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok((text, finish_reason, prompt_ids.len(), completion_tokens))
    }

    fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult {
        let prompt_ids =
            render_chat_prompt(&mut self.tokenizer, self.chat_template.clone(), messages)
                .map_err(ApiError::bad_request)?;
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (text, finish_reason, completion_tokens) = generate_llama_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            &mut self.prompt_cache,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok((text, finish_reason, prompt_ids.len(), completion_tokens))
    }

    fn stream_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        let prompt_ids = encode_prompt_ids(&mut self.tokenizer, prompt)?;
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (finish_reason, completion_tokens) = stream_llama_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            on_chunk,
            &mut self.prompt_cache,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok(GenerationStats {
            finish_reason,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
        })
    }

    fn stream_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        let prompt_ids =
            render_chat_prompt(&mut self.tokenizer, self.chat_template.clone(), messages)
                .map_err(ApiError::bad_request)?;
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (finish_reason, completion_tokens) = stream_llama_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            on_chunk,
            &mut self.prompt_cache,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok(GenerationStats {
            finish_reason,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
        })
    }
}

#[cfg(feature = "native-mlx")]
impl FamilyAdapter for Gemma2Engine {
    fn generate_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult {
        let prompt_ids = render_chat_prompt(
            &mut self.tokenizer,
            self.chat_template.clone(),
            &[ChatMessage {
                role: "user".into(),
                content: serde_json::Value::String(prompt.to_string()),
            }],
        )
        .map_err(ApiError::bad_request)?;
        let prompt_ids = ensure_leading_bos_token(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (text, finish_reason, completion_tokens) = generate_gemma2_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok((text, finish_reason, prompt_ids.len(), completion_tokens))
    }

    fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult {
        let prompt_ids =
            render_chat_prompt(&mut self.tokenizer, self.chat_template.clone(), messages)
                .map_err(ApiError::bad_request)?;
        let prompt_ids = ensure_leading_bos_token(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (text, finish_reason, completion_tokens) = generate_gemma2_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok((text, finish_reason, prompt_ids.len(), completion_tokens))
    }

    fn stream_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        let prompt_ids = render_chat_prompt(
            &mut self.tokenizer,
            self.chat_template.clone(),
            &[ChatMessage {
                role: "user".into(),
                content: serde_json::Value::String(prompt.to_string()),
            }],
        )
        .map_err(ApiError::bad_request)?;
        let prompt_ids = ensure_leading_bos_token(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (finish_reason, completion_tokens) = stream_gemma2_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            on_chunk,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok(GenerationStats {
            finish_reason,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
        })
    }

    fn stream_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        let prompt_ids =
            render_chat_prompt(&mut self.tokenizer, self.chat_template.clone(), messages)
                .map_err(ApiError::bad_request)?;
        let prompt_ids = ensure_leading_bos_token(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (finish_reason, completion_tokens) = stream_gemma2_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            on_chunk,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok(GenerationStats {
            finish_reason,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
        })
    }
}

#[cfg(feature = "native-mlx")]
impl FamilyAdapter for Gemma3TextEngine {
    fn generate_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult {
        let prompt_ids = render_chat_prompt(
            &mut self.tokenizer,
            self.chat_template.clone(),
            &[ChatMessage {
                role: "user".into(),
                content: serde_json::Value::String(prompt.to_string()),
            }],
        )
        .map_err(ApiError::bad_request)?;
        let prompt_ids = ensure_leading_bos_token(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (text, finish_reason, completion_tokens) = generate_gemma3_text_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok((text, finish_reason, prompt_ids.len(), completion_tokens))
    }

    fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult {
        let prompt_ids =
            render_chat_prompt(&mut self.tokenizer, self.chat_template.clone(), messages)
                .map_err(ApiError::bad_request)?;
        let prompt_ids = ensure_leading_bos_token(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (text, finish_reason, completion_tokens) = generate_gemma3_text_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok((text, finish_reason, prompt_ids.len(), completion_tokens))
    }

    fn stream_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        let prompt_ids = render_chat_prompt(
            &mut self.tokenizer,
            self.chat_template.clone(),
            &[ChatMessage {
                role: "user".into(),
                content: serde_json::Value::String(prompt.to_string()),
            }],
        )
        .map_err(ApiError::bad_request)?;
        let prompt_ids = ensure_leading_bos_token(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (finish_reason, completion_tokens) = stream_gemma3_text_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            on_chunk,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok(GenerationStats {
            finish_reason,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
        })
    }

    fn stream_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        let prompt_ids =
            render_chat_prompt(&mut self.tokenizer, self.chat_template.clone(), messages)
                .map_err(ApiError::bad_request)?;
        let prompt_ids = ensure_leading_bos_token(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (finish_reason, completion_tokens) = stream_gemma3_text_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            on_chunk,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok(GenerationStats {
            finish_reason,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
        })
    }
}

#[cfg(feature = "native-mlx")]
impl FamilyAdapter for MistralEngine {
    fn generate_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult {
        let prompt_ids = encode_prompt_ids(&mut self.tokenizer, prompt)?;
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (text, finish_reason, completion_tokens) = generate_mistral_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok((text, finish_reason, prompt_ids.len(), completion_tokens))
    }

    fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult {
        let prompt_ids =
            render_chat_prompt(&mut self.tokenizer, self.chat_template.clone(), messages)
                .map_err(ApiError::bad_request)?;
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (text, finish_reason, completion_tokens) = generate_mistral_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok((text, finish_reason, prompt_ids.len(), completion_tokens))
    }

    fn stream_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        let prompt_ids = encode_prompt_ids(&mut self.tokenizer, prompt)?;
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (finish_reason, completion_tokens) = stream_mistral_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            on_chunk,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok(GenerationStats {
            finish_reason,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
        })
    }

    fn stream_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        let prompt_ids =
            render_chat_prompt(&mut self.tokenizer, self.chat_template.clone(), messages)
                .map_err(ApiError::bad_request)?;
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (finish_reason, completion_tokens) = stream_mistral_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            on_chunk,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok(GenerationStats {
            finish_reason,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
        })
    }
}

#[cfg(feature = "native-mlx")]
impl FamilyAdapter for Qwen3Engine {
    fn generate_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult {
        let prompt_ids =
            render_qwen3_completion_prompt(&mut self.tokenizer, self.chat_template.clone(), prompt)
                .map_err(ApiError::bad_request)?;
        let prompt_ids = prepare_qwen3_prompt_ids(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (text, finish_reason, completion_tokens) = generate_qwen3_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            &mut self.prompt_cache,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok((
            sanitize_qwen3_output(&text),
            finish_reason,
            prompt_ids.len(),
            completion_tokens,
        ))
    }

    fn generate_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
    ) -> GenerationResult {
        let prompt_ids = render_chat_prompt(
            &mut self.tokenizer,
            Some(self.chat_template.clone()),
            messages,
        )
        .map_err(ApiError::bad_request)?;
        let prompt_ids = prepare_qwen3_prompt_ids(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (text, finish_reason, completion_tokens) = generate_qwen3_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            &mut self.prompt_cache,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok((
            sanitize_qwen3_output(&text),
            finish_reason,
            prompt_ids.len(),
            completion_tokens,
        ))
    }

    fn stream_completion(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        let prompt_ids =
            render_qwen3_completion_prompt(&mut self.tokenizer, self.chat_template.clone(), prompt)
                .map_err(ApiError::bad_request)?;
        let prompt_ids = prepare_qwen3_prompt_ids(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (finish_reason, completion_tokens) = stream_qwen3_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            on_chunk,
            &mut self.prompt_cache,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok(GenerationStats {
            finish_reason,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
        })
    }

    fn stream_chat(
        &mut self,
        messages: &[ChatMessage],
        max_tokens: usize,
        temperature: f32,
        on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    ) -> StreamingGenerationResult {
        let prompt_ids = render_chat_prompt(
            &mut self.tokenizer,
            Some(self.chat_template.clone()),
            messages,
        )
        .map_err(ApiError::bad_request)?;
        let prompt_ids = prepare_qwen3_prompt_ids(&self.tokenizer, prompt_ids);
        let prompt_tokens =
            build_prompt_array(&prompt_ids).map_err(|err| ApiError::internal(err.to_string()))?;
        let (finish_reason, completion_tokens) = stream_qwen3_tokens(
            &mut self.model,
            &mut self.tokenizer,
            &self.stop_ids,
            &prompt_ids,
            &prompt_tokens,
            max_tokens,
            temperature,
            on_chunk,
            &mut self.prompt_cache,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;

        Ok(GenerationStats {
            finish_reason,
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
        })
    }
}

#[cfg(feature = "native-mlx")]
fn build_backend_state(model_path: &Path) -> Result<BackendState> {
    let family = detect_model_family(model_path)?;
    let engine: Box<dyn FamilyAdapter + Send> = match family {
        ModelFamily::Gemma2 => Box::new(load_gemma2_engine(model_path)?),
        ModelFamily::Gemma3Text => Box::new(load_gemma3_text_engine(model_path)?),
        ModelFamily::Llama => Box::new(load_llama_engine(model_path)?),
        ModelFamily::Mistral => Box::new(load_mistral_engine(model_path)?),
        ModelFamily::Qwen2 => Box::new(load_qwen2_engine(model_path)?),
        ModelFamily::Qwen3 => Box::new(load_qwen3_engine(model_path)?),
    };
    Ok(BackendState::Ready(NativeBackend { family, engine }))
}

#[cfg(not(feature = "native-mlx"))]
fn build_backend_state(_model_path: &Path) -> Result<BackendState> {
    Ok(BackendState::Disabled {
        reason: "mesh-llm-mlx was built without --features native-mlx".into(),
    })
}

#[cfg(feature = "native-mlx")]
fn detect_model_family(model_path: &Path) -> Result<ModelFamily> {
    let config_path = model_path.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: ModelConfig = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    parse_model_family(config.model_type.as_deref())
}

#[cfg(feature = "native-mlx")]
fn parse_model_family(model_type: Option<&str>) -> Result<ModelFamily> {
    match model_type {
        Some(model_type) => FAMILY_SPECS
            .iter()
            .find(|spec| {
                spec.model_types
                    .iter()
                    .any(|candidate| candidate == &model_type)
            })
            .map(|spec| spec.family)
            .ok_or_else(|| anyhow!("unsupported model_type '{}'", model_type)),
        None => Err(anyhow!("config.json is missing model_type")),
    }
}

#[cfg(feature = "native-mlx")]
fn load_llama_engine(model_path: &Path) -> Result<LlamaEngine> {
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|err| anyhow!(err.to_string()))?;
    let model = llama::load_llama_model(model_path)?;
    let stop_ids = stop_token_ids(model_path, &tokenizer)?;
    Ok(LlamaEngine {
        tokenizer,
        chat_template: load_chat_template(model_path)?,
        model,
        stop_ids,
        prompt_cache: None,
        _compat_dir: None,
    })
}

#[cfg(feature = "native-mlx")]
fn load_gemma2_engine(model_path: &Path) -> Result<Gemma2Engine> {
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|err| anyhow!(err.to_string()))?;
    let model = gemma2::load_model(model_path)?;
    let stop_ids = stop_token_ids(model_path, &tokenizer)?;
    Ok(Gemma2Engine {
        tokenizer,
        chat_template: load_chat_template(model_path)?,
        model,
        stop_ids,
    })
}

#[cfg(feature = "native-mlx")]
fn load_gemma3_text_engine(model_path: &Path) -> Result<Gemma3TextEngine> {
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|err| anyhow!(err.to_string()))?;
    let model = gemma3_text::load_model(model_path)?;
    let stop_ids = stop_token_ids(model_path, &tokenizer)?;
    Ok(Gemma3TextEngine {
        tokenizer,
        chat_template: load_chat_template(model_path)?,
        model,
        stop_ids,
    })
}

#[cfg(feature = "native-mlx")]
fn load_mistral_engine(model_path: &Path) -> Result<MistralEngine> {
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|err| anyhow!(err.to_string()))?;
    let model = mistral::load_model(model_path)?;
    let stop_ids = stop_token_ids(model_path, &tokenizer)?;
    Ok(MistralEngine {
        tokenizer,
        chat_template: load_chat_template(model_path)?,
        model,
        stop_ids,
    })
}

#[cfg(feature = "native-mlx")]
fn load_qwen2_engine(model_path: &Path) -> Result<LlamaEngine> {
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|err| anyhow!(err.to_string()))?;
    let (model, compat_dir) = load_llama_model_with_compat(model_path, CompatConfig::Qwen2)?;
    let stop_ids = stop_token_ids(model_path, &tokenizer)?;
    Ok(LlamaEngine {
        tokenizer,
        chat_template: load_chat_template(model_path)?,
        model,
        stop_ids,
        prompt_cache: None,
        _compat_dir: Some(compat_dir),
    })
}

fn load_qwen3_engine(model_path: &Path) -> Result<Qwen3Engine> {
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
        .map_err(|err| anyhow!(err.to_string()))?;
    let chat_template =
        load_model_chat_template_from_file(model_path.join("tokenizer_config.json"))?
            .ok_or_else(|| anyhow!("tokenizer_config.json is missing chat_template"))?;
    let model = qwen3::load_qwen3_model(model_path)?;
    let stop_ids = stop_token_ids(model_path, &tokenizer)?;
    Ok(Qwen3Engine {
        tokenizer,
        chat_template: normalize_qwen3_chat_template(chat_template),
        model,
        stop_ids,
        prompt_cache: None,
    })
}

#[cfg(feature = "native-mlx")]
#[derive(Clone, Copy)]
enum CompatConfig {
    Qwen2,
}

#[cfg(feature = "native-mlx")]
fn load_llama_model_with_compat(
    model_path: &Path,
    compat: CompatConfig,
) -> Result<(llama::Model, tempfile::TempDir)> {
    let config_path = model_path.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let mut config: serde_json::Value =
        serde_json::from_str(&raw).context("failed to parse compat config.json")?;
    inject_llama_compat_fields(&mut config, compat)?;

    let compat_dir = tempfile::tempdir().context("failed to create compat dir")?;
    for entry in std::fs::read_dir(model_path)
        .with_context(|| format!("failed to read {}", model_path.display()))?
    {
        let entry = entry?;
        let file_name = entry.file_name();
        if file_name == "config.json" {
            continue;
        }
        symlink(entry.path(), compat_dir.path().join(file_name)).with_context(|| {
            format!(
                "failed to create symlink in compat dir for {}",
                entry.path().display()
            )
        })?;
    }

    std::fs::write(
        compat_dir.path().join("config.json"),
        serde_json::to_vec_pretty(&config).context("failed to serialize compat config")?,
    )
    .context("failed to write compat config")?;

    let model = llama::load_llama_model(compat_dir.path())?;
    Ok((model, compat_dir))
}

#[cfg(feature = "native-mlx")]
fn inject_llama_compat_fields(config: &mut serde_json::Value, compat: CompatConfig) -> Result<()> {
    let object = config
        .as_object_mut()
        .ok_or_else(|| anyhow!("compat config.json must be a JSON object"))?;
    if object.contains_key("head_dim") {
        return Ok(());
    }

    let hidden_size = object
        .get("hidden_size")
        .and_then(|value| value.as_i64())
        .ok_or_else(|| anyhow!("compat config.json is missing hidden_size"))?;
    let num_attention_heads = object
        .get("num_attention_heads")
        .and_then(|value| value.as_i64())
        .ok_or_else(|| anyhow!("compat config.json is missing num_attention_heads"))?;
    if num_attention_heads == 0 || hidden_size % num_attention_heads != 0 {
        return Err(anyhow!(
            "compat hidden_size {} is not divisible by num_attention_heads {}",
            hidden_size,
            num_attention_heads
        ));
    }

    object.insert(
        "head_dim".into(),
        serde_json::Value::Number((hidden_size / num_attention_heads).into()),
    );
    match compat {
        CompatConfig::Qwen2 => {}
    }
    Ok(())
}

#[cfg(feature = "native-mlx")]
fn load_chat_template(model_path: &Path) -> Result<Option<String>> {
    let tokenizer_config_path = model_path.join("tokenizer_config.json");
    if !tokenizer_config_path.exists() {
        return Ok(None);
    }
    load_model_chat_template_from_file(tokenizer_config_path).map_err(Into::into)
}

#[cfg(feature = "native-mlx")]
fn build_prompt_array(ids: &[u32]) -> Result<Array> {
    let started = Instant::now();
    let prompt = Array::from(ids).index(NewAxis);
    add_profile_stage("build", started.elapsed());
    Ok(prompt)
}

#[cfg(feature = "native-mlx")]
fn decode_token_chunk(tokenizer: &mut Tokenizer, token_id: u32) -> Result<String> {
    tokenizer
        .decode(&[token_id], true)
        .map_err(|err| anyhow!(err.to_string()))
}

#[cfg(feature = "native-mlx")]
fn encode_prompt_ids(tokenizer: &mut Tokenizer, prompt: &str) -> Result<Vec<u32>, ApiError> {
    let started = Instant::now();
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|err| ApiError::bad_request(err.to_string()))?;
    add_profile_stage("encode", started.elapsed());
    Ok(encoding.get_ids().to_vec())
}

#[cfg(feature = "native-mlx")]
fn prefill_llama_prompt(
    model: &mut llama::Model,
    prompt_tokens: &Array,
) -> Result<(Array, Vec<Option<ConcatKeyValueCache>>)> {
    let mut cache = Vec::<Option<ConcatKeyValueCache>>::new();
    let input = llama::ModelInput {
        inputs: prompt_tokens,
        mask: None,
        cache: &mut cache,
    };
    let logits = model.forward(input)?;
    let logits = logits.index((.., -1, ..));
    eval([&logits])?;
    Ok((logits, cache))
}

#[cfg(feature = "native-mlx")]
fn common_prefix_len(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(l, r)| l == r)
        .count()
}

#[cfg(feature = "native-mlx")]
fn extend_llama_prefill_one_token_at_a_time(
    model: &mut llama::Model,
    prompt_ids: &[u32],
    start_at: usize,
    mut cache: Vec<Option<ConcatKeyValueCache>>,
) -> Result<(Array, Vec<Option<ConcatKeyValueCache>>)> {
    let mut last_logits = None;
    for token_id in &prompt_ids[start_at..] {
        let chunk_tokens = build_prompt_array(&[*token_id])?;
        let input = llama::ModelInput {
            inputs: &chunk_tokens,
            mask: None,
            cache: &mut cache,
        };
        let logits = model.forward(input)?;
        let logits = logits.index((.., -1, ..));
        eval([&logits])?;
        last_logits = Some(logits.clone());
    }
    let prefill_logits =
        last_logits.ok_or_else(|| anyhow!("llama prompt must contain at least one token"))?;
    Ok((prefill_logits, cache))
}

#[cfg(feature = "native-mlx")]
fn get_or_create_llama_prefill(
    model: &mut llama::Model,
    prompt_ids: &[u32],
    prompt_tokens: &Array,
    prompt_cache: &mut Option<PromptCacheEntry>,
) -> Result<(Array, Vec<Option<ConcatKeyValueCache>>, usize)> {
    if let Some(entry) = prompt_cache.as_ref() {
        let reused_prompt_tokens = common_prefix_len(&entry.prompt_ids, prompt_ids);
        if entry.prompt_ids == prompt_ids {
            return Ok((
                entry.prefill_logits.clone(),
                entry.prefill_cache.clone(),
                entry.prompt_ids.len(),
            ));
        }

        if reused_prompt_tokens == entry.prompt_ids.len() && reused_prompt_tokens < prompt_ids.len()
        {
            let (prefill_logits, prefill_cache) = extend_llama_prefill_one_token_at_a_time(
                model,
                prompt_ids,
                reused_prompt_tokens,
                entry.prefill_cache.clone(),
            )?;
            *prompt_cache = Some(PromptCacheEntry {
                prompt_ids: prompt_ids.to_vec(),
                prefill_cache: prefill_cache.clone(),
                prefill_logits: prefill_logits.clone(),
            });
            return Ok((prefill_logits, prefill_cache, reused_prompt_tokens));
        }
    }

    let (prefill_logits, prefill_cache) = prefill_llama_prompt(model, prompt_tokens)?;
    *prompt_cache = Some(PromptCacheEntry {
        prompt_ids: prompt_ids.to_vec(),
        prefill_cache: prefill_cache.clone(),
        prefill_logits: prefill_logits.clone(),
    });
    Ok((prefill_logits, prefill_cache, 0))
}

#[cfg(feature = "native-mlx")]
fn prefill_qwen3_prompt(
    model: &mut qwen3::Model,
    prompt_tokens: &Array,
) -> Result<(Array, Vec<Option<ConcatKeyValueCache>>)> {
    let mut cache = Vec::<Option<ConcatKeyValueCache>>::new();
    let input = qwen3::ModelInput {
        inputs: prompt_tokens,
        mask: None,
        cache: &mut cache,
    };
    let logits = model.forward(input)?;
    let logits = logits.index((.., -1, ..));
    eval([&logits])?;
    Ok((logits, cache))
}

#[cfg(feature = "native-mlx")]
fn extend_qwen3_prefill_one_token_at_a_time(
    model: &mut qwen3::Model,
    prompt_ids: &[u32],
    start_at: usize,
    mut cache: Vec<Option<ConcatKeyValueCache>>,
) -> Result<(Array, Vec<Option<ConcatKeyValueCache>>)> {
    let mut last_logits = None;
    for token_id in &prompt_ids[start_at..] {
        let chunk_tokens = build_prompt_array(&[*token_id])?;
        let input = qwen3::ModelInput {
            inputs: &chunk_tokens,
            mask: None,
            cache: &mut cache,
        };
        let logits = model.forward(input)?;
        let logits = logits.index((.., -1, ..));
        eval([&logits])?;
        last_logits = Some(logits.clone());
    }
    let prefill_logits =
        last_logits.ok_or_else(|| anyhow!("qwen3 prompt must contain at least one token"))?;
    Ok((prefill_logits, cache))
}

#[cfg(feature = "native-mlx")]
fn get_or_create_qwen3_prefill(
    model: &mut qwen3::Model,
    prompt_ids: &[u32],
    prompt_tokens: &Array,
    prompt_cache: &mut Option<PromptCacheEntry>,
) -> Result<(Array, Vec<Option<ConcatKeyValueCache>>, usize)> {
    if let Some(entry) = prompt_cache.as_ref() {
        let reused_prompt_tokens = common_prefix_len(&entry.prompt_ids, prompt_ids);
        if entry.prompt_ids == prompt_ids {
            return Ok((
                entry.prefill_logits.clone(),
                entry.prefill_cache.clone(),
                entry.prompt_ids.len(),
            ));
        }

        if reused_prompt_tokens == entry.prompt_ids.len() && reused_prompt_tokens < prompt_ids.len()
        {
            let (prefill_logits, prefill_cache) = extend_qwen3_prefill_one_token_at_a_time(
                model,
                prompt_ids,
                reused_prompt_tokens,
                entry.prefill_cache.clone(),
            )?;
            *prompt_cache = Some(PromptCacheEntry {
                prompt_ids: prompt_ids.to_vec(),
                prefill_cache: prefill_cache.clone(),
                prefill_logits: prefill_logits.clone(),
            });
            return Ok((prefill_logits, prefill_cache, reused_prompt_tokens));
        }
    }

    let (prefill_logits, prefill_cache) = prefill_qwen3_prompt(model, prompt_tokens)?;
    *prompt_cache = Some(PromptCacheEntry {
        prompt_ids: prompt_ids.to_vec(),
        prefill_cache: prefill_cache.clone(),
        prefill_logits: prefill_logits.clone(),
    });
    Ok((prefill_logits, prefill_cache, 0))
}

#[cfg(feature = "native-mlx")]
fn generate_llama_tokens(
    model: &mut llama::Model,
    tokenizer: &mut Tokenizer,
    stop_ids: &[u32],
    prompt_ids: &[u32],
    prompt_tokens: &Array,
    max_tokens: usize,
    temperature: f32,
    prompt_cache: &mut Option<PromptCacheEntry>,
) -> Result<(String, &'static str, usize)> {
    let decode_started = Instant::now();
    let mut output_ids = Vec::new();
    let mut finish_reason = "length";
    let (prefill_logits, mut cache, reused_prompt_tokens) =
        get_or_create_llama_prefill(model, prompt_ids, prompt_tokens, prompt_cache)?;
    set_profile_reused_prompt_tokens(reused_prompt_tokens);
    let mut y = llama::sample(&prefill_logits, temperature)?;
    let first_token_ms = Some(duration_ms(decode_started.elapsed()));
    let mut token_id = y.item::<u32>();
    if stop_ids.contains(&token_id) {
        finish_reason = "stop";
    } else {
        output_ids.push(token_id);
    }

    while output_ids.len() < max_tokens && finish_reason != "stop" {
        let inputs = y.index((.., NewAxis));
        let input = llama::ModelInput {
            inputs: &inputs,
            mask: None,
            cache: &mut cache,
        };
        let logits = model.forward(input)?;
        y = llama::sample(&logits.index((.., -1, ..)), temperature)?;
        token_id = y.item::<u32>();
        if stop_ids.contains(&token_id) {
            finish_reason = "stop";
            break;
        }
        output_ids.push(token_id);
    }

    let text = tokenizer
        .decode(&output_ids, true)
        .map_err(|err| anyhow!(err.to_string()))?;
    set_profile_decode_timings(DecodeTimings {
        first_token_ms,
        decode_ms: duration_ms(decode_started.elapsed()),
    });
    Ok((text, finish_reason, output_ids.len()))
}

#[cfg(feature = "native-mlx")]
fn stream_llama_tokens(
    model: &mut llama::Model,
    tokenizer: &mut Tokenizer,
    stop_ids: &[u32],
    prompt_ids: &[u32],
    prompt_tokens: &Array,
    max_tokens: usize,
    temperature: f32,
    on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    prompt_cache: &mut Option<PromptCacheEntry>,
) -> Result<(&'static str, usize)> {
    let decode_started = Instant::now();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";
    let mut output_ids = Vec::new();
    let mut emitted_text = String::new();
    let (prefill_logits, mut cache, reused_prompt_tokens) =
        get_or_create_llama_prefill(model, prompt_ids, prompt_tokens, prompt_cache)?;
    set_profile_reused_prompt_tokens(reused_prompt_tokens);
    let mut y = llama::sample(&prefill_logits, temperature)?;
    let first_token_ms = Some(duration_ms(decode_started.elapsed()));
    let mut token_id = y.item::<u32>();
    if !stop_ids.contains(&token_id) {
        completion_tokens += 1;
        output_ids.push(token_id);
        let decoded = tokenizer
            .decode(&output_ids, true)
            .map_err(|err| anyhow!(err.to_string()))?;
        if let Some(chunk) = emitted_suffix(&emitted_text, &decoded) {
            emitted_text = decoded;
            on_chunk(chunk).map_err(|err| anyhow!(err.message))?;
        }
    } else {
        finish_reason = "stop";
    }

    while completion_tokens < max_tokens && finish_reason != "stop" {
        let inputs = y.index((.., NewAxis));
        let input = llama::ModelInput {
            inputs: &inputs,
            mask: None,
            cache: &mut cache,
        };
        let logits = model.forward(input)?;
        y = llama::sample(&logits.index((.., -1, ..)), temperature)?;
        token_id = y.item::<u32>();
        if stop_ids.contains(&token_id) {
            finish_reason = "stop";
            break;
        }
        completion_tokens += 1;
        output_ids.push(token_id);
        let decoded = tokenizer
            .decode(&output_ids, true)
            .map_err(|err| anyhow!(err.to_string()))?;
        if let Some(chunk) = emitted_suffix(&emitted_text, &decoded) {
            emitted_text = decoded;
            on_chunk(chunk).map_err(|err| anyhow!(err.message))?;
        }
    }

    set_profile_decode_timings(DecodeTimings {
        first_token_ms,
        decode_ms: duration_ms(decode_started.elapsed()),
    });
    Ok((finish_reason, completion_tokens))
}

#[cfg(feature = "native-mlx")]
fn generate_gemma2_tokens(
    model: &mut gemma2::Model,
    tokenizer: &mut Tokenizer,
    stop_ids: &[u32],
    prompt_tokens: &Array,
    max_tokens: usize,
    temperature: f32,
) -> Result<(String, &'static str, usize)> {
    let decode_started = Instant::now();
    let mut first_token_ms = None;
    let mut cache = Vec::<Option<ConcatKeyValueCache>>::new();
    let mut output_ids = Vec::new();
    let mut finish_reason = "length";
    let generate =
        gemma2::Generate::<ConcatKeyValueCache>::new(model, &mut cache, temperature, prompt_tokens);

    for (token, _) in generate.zip(0..max_tokens) {
        let token = token?;
        if first_token_ms.is_none() {
            first_token_ms = Some(duration_ms(decode_started.elapsed()));
        }
        let token_id = token.item::<u32>();
        if stop_ids.contains(&token_id) {
            finish_reason = "stop";
            break;
        }
        output_ids.push(token_id);
    }

    let text = tokenizer
        .decode(&output_ids, true)
        .map_err(|err| anyhow!(err.to_string()))?;
    set_profile_decode_timings(DecodeTimings {
        first_token_ms,
        decode_ms: duration_ms(decode_started.elapsed()),
    });
    Ok((text, finish_reason, output_ids.len()))
}

#[cfg(feature = "native-mlx")]
fn stream_gemma2_tokens(
    model: &mut gemma2::Model,
    tokenizer: &mut Tokenizer,
    stop_ids: &[u32],
    prompt_tokens: &Array,
    max_tokens: usize,
    temperature: f32,
    on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
) -> Result<(&'static str, usize)> {
    let decode_started = Instant::now();
    let mut first_token_ms = None;
    let mut cache = Vec::<Option<ConcatKeyValueCache>>::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";
    let generate =
        gemma2::Generate::<ConcatKeyValueCache>::new(model, &mut cache, temperature, prompt_tokens);

    for (token, _) in generate.zip(0..max_tokens) {
        let token = token?;
        if first_token_ms.is_none() {
            first_token_ms = Some(duration_ms(decode_started.elapsed()));
        }
        let token_id = token.item::<u32>();
        if stop_ids.contains(&token_id) {
            finish_reason = "stop";
            break;
        }
        completion_tokens += 1;
        let chunk = decode_token_chunk(tokenizer, token_id)?;
        if !chunk.is_empty() {
            on_chunk(chunk).map_err(|err| anyhow!(err.message))?;
        }
    }

    set_profile_decode_timings(DecodeTimings {
        first_token_ms,
        decode_ms: duration_ms(decode_started.elapsed()),
    });
    Ok((finish_reason, completion_tokens))
}

#[cfg(feature = "native-mlx")]
fn generate_gemma3_text_tokens(
    model: &mut gemma3_text::Model,
    tokenizer: &mut Tokenizer,
    stop_ids: &[u32],
    prompt_tokens: &Array,
    max_tokens: usize,
    temperature: f32,
) -> Result<(String, &'static str, usize)> {
    let decode_started = Instant::now();
    let mut first_token_ms = None;
    let mut cache = Vec::new();
    let mut output_ids = Vec::new();
    let mut finish_reason = "length";
    let generate = gemma3_text::Generate::new(model, &mut cache, temperature, prompt_tokens);

    for (token, _) in generate.zip(0..max_tokens) {
        let token = token?;
        if first_token_ms.is_none() {
            first_token_ms = Some(duration_ms(decode_started.elapsed()));
        }
        let token_id = token.item::<u32>();
        if stop_ids.contains(&token_id) {
            finish_reason = "stop";
            break;
        }
        output_ids.push(token_id);
    }

    let text = tokenizer
        .decode(&output_ids, true)
        .map_err(|err| anyhow!(err.to_string()))?;
    set_profile_decode_timings(DecodeTimings {
        first_token_ms,
        decode_ms: duration_ms(decode_started.elapsed()),
    });
    Ok((text, finish_reason, output_ids.len()))
}

#[cfg(feature = "native-mlx")]
fn stream_gemma3_text_tokens(
    model: &mut gemma3_text::Model,
    tokenizer: &mut Tokenizer,
    stop_ids: &[u32],
    prompt_tokens: &Array,
    max_tokens: usize,
    temperature: f32,
    on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
) -> Result<(&'static str, usize)> {
    let decode_started = Instant::now();
    let mut first_token_ms = None;
    let mut cache = Vec::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";
    let generate = gemma3_text::Generate::new(model, &mut cache, temperature, prompt_tokens);

    for (token, _) in generate.zip(0..max_tokens) {
        let token = token?;
        if first_token_ms.is_none() {
            first_token_ms = Some(duration_ms(decode_started.elapsed()));
        }
        let token_id = token.item::<u32>();
        if stop_ids.contains(&token_id) {
            finish_reason = "stop";
            break;
        }
        completion_tokens += 1;
        let chunk = decode_token_chunk(tokenizer, token_id)?;
        if !chunk.is_empty() {
            on_chunk(chunk).map_err(|err| anyhow!(err.message))?;
        }
    }

    set_profile_decode_timings(DecodeTimings {
        first_token_ms,
        decode_ms: duration_ms(decode_started.elapsed()),
    });
    Ok((finish_reason, completion_tokens))
}

#[cfg(feature = "native-mlx")]
fn generate_mistral_tokens(
    model: &mut mistral::Model,
    tokenizer: &mut Tokenizer,
    stop_ids: &[u32],
    prompt_tokens: &Array,
    max_tokens: usize,
    temperature: f32,
) -> Result<(String, &'static str, usize)> {
    let decode_started = Instant::now();
    let mut first_token_ms = None;
    let mut cache = Vec::<Option<ConcatKeyValueCache>>::new();
    let mut output_ids = Vec::new();
    let mut finish_reason = "length";
    let generate = mistral::Generate::<ConcatKeyValueCache>::new(
        model,
        &mut cache,
        temperature,
        prompt_tokens,
    );

    for (token, _) in generate.zip(0..max_tokens) {
        let token = token?;
        if first_token_ms.is_none() {
            first_token_ms = Some(duration_ms(decode_started.elapsed()));
        }
        let token_id = token.item::<u32>();
        if stop_ids.contains(&token_id) {
            finish_reason = "stop";
            break;
        }
        output_ids.push(token_id);
    }

    let text = tokenizer
        .decode(&output_ids, true)
        .map_err(|err| anyhow!(err.to_string()))?;
    set_profile_decode_timings(DecodeTimings {
        first_token_ms,
        decode_ms: duration_ms(decode_started.elapsed()),
    });
    Ok((text, finish_reason, output_ids.len()))
}

#[cfg(feature = "native-mlx")]
fn stream_mistral_tokens(
    model: &mut mistral::Model,
    tokenizer: &mut Tokenizer,
    stop_ids: &[u32],
    prompt_tokens: &Array,
    max_tokens: usize,
    temperature: f32,
    on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
) -> Result<(&'static str, usize)> {
    let decode_started = Instant::now();
    let mut first_token_ms = None;
    let mut cache = Vec::<Option<ConcatKeyValueCache>>::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";
    let generate = mistral::Generate::<ConcatKeyValueCache>::new(
        model,
        &mut cache,
        temperature,
        prompt_tokens,
    );

    for (token, _) in generate.zip(0..max_tokens) {
        let token = token?;
        if first_token_ms.is_none() {
            first_token_ms = Some(duration_ms(decode_started.elapsed()));
        }
        let token_id = token.item::<u32>();
        if stop_ids.contains(&token_id) {
            finish_reason = "stop";
            break;
        }
        completion_tokens += 1;
        let chunk = decode_token_chunk(tokenizer, token_id)?;
        if !chunk.is_empty() {
            on_chunk(chunk).map_err(|err| anyhow!(err.message))?;
        }
    }

    set_profile_decode_timings(DecodeTimings {
        first_token_ms,
        decode_ms: duration_ms(decode_started.elapsed()),
    });
    Ok((finish_reason, completion_tokens))
}

#[cfg(feature = "native-mlx")]
fn generate_qwen3_tokens(
    model: &mut qwen3::Model,
    tokenizer: &mut Tokenizer,
    stop_ids: &[u32],
    prompt_ids: &[u32],
    prompt_tokens: &Array,
    max_tokens: usize,
    temperature: f32,
    prompt_cache: &mut Option<PromptCacheEntry>,
) -> Result<(String, &'static str, usize)> {
    let decode_started = Instant::now();
    let mut output_ids = Vec::new();
    let mut finish_reason = "length";
    let (prefill_logits, mut cache, reused_prompt_tokens) =
        get_or_create_qwen3_prefill(model, prompt_ids, prompt_tokens, prompt_cache)?;
    set_profile_reused_prompt_tokens(reused_prompt_tokens);
    let mut y = qwen3::sample(&prefill_logits, temperature)?;
    let first_token_ms = Some(duration_ms(decode_started.elapsed()));
    let mut token_id = y.item::<u32>();
    if stop_ids.contains(&token_id) {
        finish_reason = "stop";
    } else {
        output_ids.push(token_id);
    }

    while output_ids.len() < max_tokens && finish_reason != "stop" {
        let inputs = y.index((.., NewAxis));
        let input = qwen3::ModelInput {
            inputs: &inputs,
            mask: None,
            cache: &mut cache,
        };
        let logits = model.forward(input)?;
        y = qwen3::sample(&logits.index((.., -1, ..)), temperature)?;
        token_id = y.item::<u32>();
        if stop_ids.contains(&token_id) {
            finish_reason = "stop";
            break;
        }
        output_ids.push(token_id);
    }

    let text = tokenizer
        .decode(&output_ids, true)
        .map_err(|err| anyhow!(err.to_string()))?;
    set_profile_decode_timings(DecodeTimings {
        first_token_ms,
        decode_ms: duration_ms(decode_started.elapsed()),
    });
    Ok((text, finish_reason, output_ids.len()))
}

#[cfg(feature = "native-mlx")]
fn stream_qwen3_tokens(
    model: &mut qwen3::Model,
    tokenizer: &mut Tokenizer,
    stop_ids: &[u32],
    prompt_ids: &[u32],
    prompt_tokens: &Array,
    max_tokens: usize,
    temperature: f32,
    on_chunk: &mut dyn FnMut(String) -> Result<(), ApiError>,
    prompt_cache: &mut Option<PromptCacheEntry>,
) -> Result<(&'static str, usize)> {
    let decode_started = Instant::now();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";
    let mut output_ids = Vec::new();
    let mut in_thinking = false;
    let (prefill_logits, mut cache, reused_prompt_tokens) =
        get_or_create_qwen3_prefill(model, prompt_ids, prompt_tokens, prompt_cache)?;
    set_profile_reused_prompt_tokens(reused_prompt_tokens);
    let mut y = qwen3::sample(&prefill_logits, temperature)?;
    let first_token_ms = Some(duration_ms(decode_started.elapsed()));
    let think_start = "<think>";
    let think_end = "</think>";
    let mut token_id = y.item::<u32>();
    if !stop_ids.contains(&token_id) {
        output_ids.push(token_id);
        completion_tokens += 1;
        let chunk = decode_token_chunk(tokenizer, token_id)?;
        if chunk == think_start {
            in_thinking = true;
        } else if chunk == think_end {
            in_thinking = false;
        } else if !in_thinking && !chunk.is_empty() {
            on_chunk(chunk).map_err(|err| anyhow!(err.message))?;
        }
    } else {
        finish_reason = "stop";
    }

    while completion_tokens < max_tokens && finish_reason != "stop" {
        let inputs = y.index((.., NewAxis));
        let input = qwen3::ModelInput {
            inputs: &inputs,
            mask: None,
            cache: &mut cache,
        };
        let logits = model.forward(input)?;
        y = qwen3::sample(&logits.index((.., -1, ..)), temperature)?;
        token_id = y.item::<u32>();
        if stop_ids.contains(&token_id) {
            finish_reason = "stop";
            break;
        }
        output_ids.push(token_id);
        completion_tokens += 1;
        let chunk = decode_token_chunk(tokenizer, token_id)?;
        if chunk == think_start {
            in_thinking = true;
        } else if chunk == think_end {
            in_thinking = false;
        } else if !in_thinking && !chunk.is_empty() {
            on_chunk(chunk).map_err(|err| anyhow!(err.message))?;
        }
    }

    set_profile_decode_timings(DecodeTimings {
        first_token_ms,
        decode_ms: duration_ms(decode_started.elapsed()),
    });
    Ok((finish_reason, completion_tokens))
}

#[cfg(feature = "native-mlx")]
fn render_chat_prompt(
    tokenizer: &mut Tokenizer,
    chat_template: Option<String>,
    messages: &[ChatMessage],
) -> Result<Vec<u32>, String> {
    let started = Instant::now();
    #[derive(Clone, Serialize)]
    #[serde(rename_all = "lowercase")]
    enum Role {
        System,
        User,
        Assistant,
    }

    let mut conversations = Vec::with_capacity(messages.len());
    for message in messages {
        let role = match message.role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => return Err(format!("unsupported role '{}'", other)),
        };
        let content = message
            .content
            .as_str()
            .ok_or_else(|| "only string message content is supported".to_string())?;
        conversations.push(Conversation {
            role,
            content: content.to_string(),
        });
    }

    let template = if let Some(chat_template) = chat_template {
        chat_template
    } else {
        let mut rendered = String::new();
        for message in &conversations {
            match message.role {
                Role::System => {
                    rendered.push_str(
                        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
                    );
                    rendered.push_str(&message.content);
                    rendered.push_str("<|eot_id|>");
                }
                Role::User => {
                    if rendered.is_empty() {
                        rendered.push_str("<|begin_of_text|>");
                    }
                    rendered.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
                    rendered.push_str(&message.content);
                    rendered.push_str("<|eot_id|>");
                }
                Role::Assistant => {
                    if rendered.is_empty() {
                        rendered.push_str("<|begin_of_text|>");
                    }
                    rendered.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                    rendered.push_str(&message.content);
                    rendered.push_str("<|eot_id|>");
                }
            }
        }
        rendered.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        let encoding = tokenizer
            .encode(rendered, false)
            .map_err(|err| err.to_string())?;
        add_profile_stage("render", started.elapsed());
        return Ok(encoding.get_ids().to_vec());
    };

    let encodings = tokenizer
        .apply_chat_template_and_encode(
            template,
            ApplyChatTemplateArgs {
                conversations: vec![conversations.into()],
                documents: None,
                model_id: "local",
                chat_template_id: None,
                add_generation_prompt: Some(true),
                continue_final_message: None,
            },
        )
        .map_err(|err| err.to_string())?;

    let result = encodings
        .iter()
        .flat_map(|encoding| encoding.get_ids().iter().copied())
        .collect();
    add_profile_stage("render", started.elapsed());
    Ok(result)
}

#[cfg(feature = "native-mlx")]
fn render_qwen3_completion_prompt(
    tokenizer: &mut Tokenizer,
    chat_template: String,
    prompt: &str,
) -> Result<Vec<u32>, String> {
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: serde_json::Value::String(prompt.to_string()),
    }];
    render_chat_prompt(tokenizer, Some(chat_template), &messages)
}

#[cfg(feature = "native-mlx")]
fn normalize_qwen3_chat_template(chat_template: String) -> String {
    chat_template.replace(
        "if enable_thinking is defined and enable_thinking is false",
        "if true",
    )
}

#[cfg(feature = "native-mlx")]
// Adapted from exo's MLX prompt repair flow. See NOTICE-exo.md.
fn prepare_qwen3_prompt_ids(tokenizer: &Tokenizer, prompt_ids: Vec<u32>) -> Vec<u32> {
    fix_unmatched_think_end_tokens(tokenizer, prompt_ids)
}

#[cfg(feature = "native-mlx")]
// Adapted from exo's MLX prompt repair flow. See NOTICE-exo.md.
fn fix_unmatched_think_end_tokens(tokenizer: &Tokenizer, prompt_ids: Vec<u32>) -> Vec<u32> {
    let think_start_id = tokenizer.token_to_id("<think>");
    let think_end_id = tokenizer.token_to_id("</think>");
    match (think_start_id, think_end_id) {
        (Some(think_start_id), Some(think_end_id)) => {
            repair_unmatched_balanced_tokens(&prompt_ids, think_start_id, think_end_id)
        }
        _ => prompt_ids,
    }
}

#[cfg(feature = "native-mlx")]
fn repair_unmatched_balanced_tokens(ids: &[u32], start_id: u32, end_id: u32) -> Vec<u32> {
    let mut result = Vec::with_capacity(ids.len());
    let mut depth = 0usize;
    for &id in ids {
        if id == start_id {
            depth += 1;
        } else if id == end_id {
            if depth == 0 {
                result.push(start_id);
            } else {
                depth -= 1;
            }
        }
        result.push(id);
    }
    result
}

#[cfg(feature = "native-mlx")]
fn sanitize_qwen3_output(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.starts_with("<think>") && !trimmed.contains("</think>") {
        return String::new();
    }
    let trimmed = cleanup_qwen3_labels(&truncate_repetition_loop(trimmed));
    if let Some(rest) = trimmed.strip_prefix("<think>") {
        if let Some((_, answer)) = rest.split_once("</think>") {
            return cleanup_qwen3_labels(&truncate_repetition_loop(answer.trim()));
        }
    }
    trimmed
}

#[cfg(feature = "native-mlx")]
fn truncate_repetition_loop(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 4 {
        return text.to_string();
    }

    let mut repeat_run = 1usize;
    for idx in 1..words.len() {
        if words[idx] == words[idx - 1] {
            repeat_run += 1;
            if repeat_run >= 3 {
                if idx > 1 {
                    let prefix = words[..idx - 1].join(" ");
                    if !prefix.is_empty() {
                        return prefix;
                    }
                }
                return text.to_string();
            }
        } else {
            repeat_run = 1;
        }
    }

    text.to_string()
}

#[cfg(feature = "native-mlx")]
fn trim_leading_label_word(text: &str) -> String {
    let trimmed = text.trim_start();
    for prefix in ["Instructions", "Answer", "Question"] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let rest = rest.trim_start_matches(|ch: char| ch.is_whitespace() || ch == ':');
            if !rest.is_empty() {
                return rest.to_string();
            }
        }
    }
    trimmed.to_string()
}

#[cfg(feature = "native-mlx")]
fn trim_trailing_label_word(text: &str) -> String {
    let trimmed = text.trim_end();
    for suffix in ["Instructions", "Answer", "Question"] {
        if let Some(prefix) = trimmed.strip_suffix(suffix) {
            let prefix = prefix.trim_end();
            if !prefix.is_empty() {
                return prefix.to_string();
            }
        }
    }
    trimmed.to_string()
}

#[cfg(feature = "native-mlx")]
fn cleanup_qwen3_labels(text: &str) -> String {
    let mut current = text.trim().to_string();
    loop {
        let next = trim_trailing_label_word(&trim_leading_label_word(&current));
        if next == current {
            return next;
        }
        current = next;
    }
}

#[cfg(feature = "native-mlx")]
fn emitted_suffix(previous: &str, current: &str) -> Option<String> {
    if current.is_empty() {
        return None;
    }
    current.strip_prefix(previous).and_then(|suffix| {
        if suffix.is_empty() {
            None
        } else {
            Some(suffix.to_string())
        }
    })
}

#[cfg(feature = "native-mlx")]
fn ensure_leading_bos_token(tokenizer: &Tokenizer, mut prompt_ids: Vec<u32>) -> Vec<u32> {
    if let Some(bos_id) = tokenizer.token_to_id("<bos>") {
        if prompt_ids.first().copied() != Some(bos_id) {
            prompt_ids.insert(0, bos_id);
        }
    }
    prompt_ids
}

#[cfg(feature = "native-mlx")]
fn stop_token_ids(model_path: &Path, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    let config_path = model_path.join("tokenizer_config.json");
    let value = if config_path.exists() {
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse {}", config_path.display()))?
    } else {
        let special_tokens_map_path = model_path.join("special_tokens_map.json");
        if special_tokens_map_path.exists() {
            let raw = std::fs::read_to_string(&special_tokens_map_path)
                .with_context(|| format!("failed to read {}", special_tokens_map_path.display()))?;
            let special_tokens_map: serde_json::Value =
                serde_json::from_str(&raw).with_context(|| {
                    format!("failed to parse {}", special_tokens_map_path.display())
                })?;
            json!({
                "eos_token": special_tokens_map.get("eos_token").cloned().unwrap_or(serde_json::Value::Null),
                "model_type": detect_model_type(model_path).unwrap_or_default(),
            })
        } else {
            json!({
                "model_type": detect_model_type(model_path).unwrap_or_default(),
            })
        }
    };

    let mut ids = Vec::new();
    collect_stop_token_ids(&value["eos_token_id"], &mut ids);
    collect_stop_token_value_ids(tokenizer, &value["eos_token"], &mut ids);
    collect_known_family_stop_ids(
        value
            .pointer("/model_type")
            .and_then(|value| value.as_str()),
        tokenizer,
        &mut ids,
    );

    if ids.is_empty() {
        match value
            .pointer("/model_type")
            .and_then(|value| value.as_str())
        {
            Some("llama") => ids.extend([128001, 128009]),
            _ => {}
        }
    }

    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

#[cfg(feature = "native-mlx")]
fn collect_known_family_stop_ids(
    model_type: Option<&str>,
    tokenizer: &Tokenizer,
    ids: &mut Vec<u32>,
) {
    match model_type {
        Some("qwen2") | Some("qwen3") => {
            for token in ["<|im_end|>", "<|endoftext|>"] {
                if let Some(id) = tokenizer.token_to_id(token) {
                    ids.push(id);
                }
            }
        }
        _ => {}
    }
}

#[cfg(feature = "native-mlx")]
fn detect_model_type(model_path: &Path) -> Result<String> {
    let config_path = model_path.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: ModelConfig = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    config
        .model_type
        .ok_or_else(|| anyhow!("config.json is missing model_type"))
}

#[cfg(feature = "native-mlx")]
fn collect_stop_token_ids(value: &serde_json::Value, ids: &mut Vec<u32>) {
    match value {
        serde_json::Value::Number(number) => {
            if let Some(id) = number.as_u64().and_then(|value| u32::try_from(value).ok()) {
                ids.push(id);
            }
        }
        serde_json::Value::Array(array) => {
            for value in array {
                collect_stop_token_ids(value, ids);
            }
        }
        _ => {}
    }
}

#[cfg(feature = "native-mlx")]
fn collect_stop_token_value_ids(
    tokenizer: &Tokenizer,
    value: &serde_json::Value,
    ids: &mut Vec<u32>,
) {
    match value {
        serde_json::Value::String(token) => {
            if let Some(id) = tokenizer.token_to_id(token) {
                ids.push(id);
            }
        }
        serde_json::Value::Object(object) => {
            if let Some(token) = object.get("content").and_then(|value| value.as_str()) {
                if let Some(id) = tokenizer.token_to_id(token) {
                    ids.push(id);
                }
            }
        }
        serde_json::Value::Array(array) => {
            for value in array {
                collect_stop_token_value_ids(tokenizer, value, ids);
            }
        }
        _ => {}
    }
}

async fn build_status_summary(state: &AppState) -> StatusSummary {
    let guard = state.inner.lock().await;
    let (healthy, backend, detail) = match &guard.backend {
        #[cfg(feature = "native-mlx")]
        BackendState::Ready(model) => {
            let backend_name = model.family.spec().backend_name.to_string();
            let detail = format!("ready ({})", model.family.primary_model_type());
            (true, backend_name, detail)
        }
        BackendState::Disabled { reason } => (false, "disabled".to_string(), reason.clone()),
    };

    StatusSummary {
        plugin: PLUGIN_ID.to_string(),
        model_id: guard.model_id.clone(),
        model_path: guard.model_path.display().to_string(),
        listen_addr: guard.listen_addr.to_string(),
        backend,
        healthy,
        detail,
    }
}

async fn build_models_summary(state: &AppState) -> ModelsSummary {
    let guard = state.inner.lock().await;
    ModelsSummary {
        data: vec![ModelInfo {
            capabilities: vec!["text"],
            id: guard.model_id.clone(),
            object: "model",
            owned_by: "mesh-llm-mlx",
        }],
        object: "list",
    }
}

fn model_dir_name(path: &Path) -> Result<&str> {
    path.file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow!("invalid model directory name"))
}

fn model_id_from_configured_path(path: &Path) -> Result<String> {
    if let Some(model_id) = huggingface_snapshot_model_id(path) {
        return Ok(model_id);
    }
    Ok(model_dir_name(path)?.to_string())
}

fn huggingface_snapshot_model_id(path: &Path) -> Option<String> {
    let snapshots_dir = path.parent()?;
    if snapshots_dir.file_name()?.to_str()? != "snapshots" {
        return None;
    }

    let repo_dir = snapshots_dir.parent()?.file_name()?.to_str()?;
    let repo_id = repo_dir.strip_prefix("models--")?.replace("--", "/");
    repo_id.rsplit('/').next().map(ToString::to_string)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::api::sync::Api;
    use std::{env, path::PathBuf};

    #[cfg(feature = "native-mlx")]
    #[test]
    fn parse_model_family_supports_gemma2() {
        assert_eq!(
            parse_model_family(Some("gemma2")).unwrap(),
            ModelFamily::Gemma2
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn parse_model_family_supports_gemma3_text() {
        assert_eq!(
            parse_model_family(Some("gemma3_text")).unwrap(),
            ModelFamily::Gemma3Text
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn parse_model_family_supports_llama() {
        assert_eq!(
            parse_model_family(Some("llama")).unwrap(),
            ModelFamily::Llama
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn parse_model_family_supports_qwen3() {
        assert_eq!(
            parse_model_family(Some("qwen3")).unwrap(),
            ModelFamily::Qwen3
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn parse_model_family_supports_mistral() {
        assert_eq!(
            parse_model_family(Some("mistral")).unwrap(),
            ModelFamily::Mistral
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn parse_model_family_supports_qwen2() {
        assert_eq!(
            parse_model_family(Some("qwen2")).unwrap(),
            ModelFamily::Qwen2
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn every_supported_family_has_test_fixture_metadata() {
        for spec in FAMILY_SPECS {
            assert!(
                !spec.backend_name.is_empty(),
                "{:?} should advertise a backend name",
                spec.family
            );
            assert!(
                !spec.test_model_path_env_var.is_empty(),
                "{:?} should define a test model path env var",
                spec.family
            );
            assert!(
                !spec.test_repo_env_var.is_empty(),
                "{:?} should define a test repo env var",
                spec.family
            );
        }
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn parse_model_family_rejects_unknown_family() {
        let err = parse_model_family(Some("gemma")).unwrap_err();
        assert!(err.to_string().contains("unsupported model_type 'gemma'"));
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn parse_model_family_requires_model_type() {
        let err = parse_model_family(None).unwrap_err();
        assert!(err
            .to_string()
            .contains("config.json is missing model_type"));
    }

    #[test]
    fn model_dir_name_returns_last_path_component() {
        let path = Path::new("/tmp/models/Qwen3-0.6B-bf16");
        assert_eq!(model_dir_name(path).unwrap(), "Qwen3-0.6B-bf16");
    }

    #[test]
    fn model_id_from_regular_path_uses_directory_name() {
        let path = Path::new("/tmp/models/Llama-3.2-1B-Instruct-bf16");
        assert_eq!(
            model_id_from_configured_path(path).unwrap(),
            "Llama-3.2-1B-Instruct-bf16"
        );
    }

    #[test]
    fn model_id_from_huggingface_snapshot_path_uses_repo_name() {
        let path = Path::new(
            "/Users/test/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-bf16/snapshots/56d07e766edd7159fbe12ed12d9cf114bf38bf1e",
        );
        assert_eq!(
            model_id_from_configured_path(path).unwrap(),
            "Qwen2.5-0.5B-Instruct-bf16"
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn repair_unmatched_balanced_tokens_inserts_missing_start_before_end() {
        assert_eq!(
            repair_unmatched_balanced_tokens(&[10, 20, 30], 20, 30),
            vec![10, 20, 30]
        );
        assert_eq!(
            repair_unmatched_balanced_tokens(&[10, 30, 40], 20, 30),
            vec![10, 20, 30, 40]
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn sanitize_qwen3_output_hides_open_think_block() {
        assert_eq!(sanitize_qwen3_output("<think>"), "");
        assert_eq!(sanitize_qwen3_output("<think>draft"), "");
        assert_eq!(
            sanitize_qwen3_output("<think>draft</think> final answer"),
            "final answer"
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn sanitize_qwen3_output_strips_leading_and_trailing_labels() {
        assert_eq!(sanitize_qwen3_output("Answer: hello"), "hello");
        assert_eq!(
            sanitize_qwen3_output("hello Instructions Instructions"),
            "hello"
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    #[ignore = "requires a real local Gemma2 MLX model"]
    fn real_gemma2_parameter_key_coverage() {
        use mlx_rs::module::ModuleParameters;

        let path =
            resolve_or_download_test_model(ModelFamily::Gemma2).expect("resolve gemma2 test model");
        let Some(path) = path else {
            eprintln!("skipping real_gemma2_parameter_key_coverage: no local model configured");
            return;
        };

        let args = crate::gemma2::get_model_args(&path).expect("load gemma2 config");
        let quantization = crate::quantized::checkpoint_quantization(&path)
            .expect("inspect gemma2 checkpoint quantization");
        let model = crate::gemma2::Model::new(args, &quantization).expect("build gemma2 model");
        let params = model.parameters().flatten();
        let param_set: std::collections::HashSet<_> = params
            .keys()
            .map(|key: &std::rc::Rc<str>| key.to_string())
            .collect();

        let weights_path = path.join("model.safetensors");
        let weight_set: std::collections::HashSet<_> = if weights_path.exists() {
            mlx_rs::Array::load_safetensors(&weights_path)
                .expect("load gemma2 safetensors")
                .keys()
                .map(|key| key.to_string())
                .collect()
        } else {
            let raw = std::fs::read_to_string(path.join("model.safetensors.index.json"))
                .expect("read gemma2 weights index");
            let index: crate::quantized::WeightMap =
                serde_json::from_str(&raw).expect("parse gemma2 weights index");
            index.weight_map.keys().cloned().collect()
        };
        let normalized_weight_set: std::collections::HashSet<_> = weight_set
            .into_iter()
            .map(|key| {
                if key.ends_with(".weight") {
                    let compat_key = key.replacen(".weight", ".inner.weight", 1);
                    if param_set.contains(&compat_key) {
                        return compat_key;
                    }
                }
                key
            })
            .collect();

        let unloaded: Vec<_> = normalized_weight_set
            .difference(&param_set)
            .cloned()
            .collect();
        let missing: Vec<_> = param_set
            .difference(&normalized_weight_set)
            .cloned()
            .collect();

        assert!(
            unloaded.is_empty(),
            "gemma2 weights missing matching params, sample: {:?}",
            unloaded.iter().take(10).collect::<Vec<_>>()
        );
        assert!(
            missing.is_empty(),
            "gemma2 params missing matching weights, sample: {:?}",
            missing.iter().take(10).collect::<Vec<_>>()
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    #[ignore = "requires a real local Gemma3 Text MLX model"]
    fn real_gemma3_text_inference_smoke() {
        let path = resolve_or_download_test_model(ModelFamily::Gemma3Text)
            .expect("resolve gemma3_text test model");
        let Some(path) = path else {
            eprintln!("skipping real_gemma3_text_inference_smoke: no local model configured");
            return;
        };

        let mut engine = load_gemma3_text_engine(&path).expect("load gemma3_text engine");
        let (text, finish_reason, prompt_tokens, completion_tokens) = engine
            .generate_chat(
                &[ChatMessage {
                    role: "user".into(),
                    content: serde_json::Value::String("Say hello in one short sentence.".into()),
                }],
                24,
                0.0,
            )
            .expect("gemma3_text inference");

        assert!(
            !text.trim().is_empty(),
            "gemma3_text output should not be empty"
        );
        assert!(
            prompt_tokens > 0,
            "gemma3_text prompt token count should be positive"
        );
        assert!(
            completion_tokens > 0,
            "gemma3_text completion token count should be positive"
        );
        assert!(
            matches!(finish_reason, "length" | "stop"),
            "unexpected finish reason: {finish_reason}"
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    #[ignore = "requires a real local Gemma2 MLX model"]
    fn real_gemma2_inference_smoke() {
        let path =
            resolve_or_download_test_model(ModelFamily::Gemma2).expect("resolve gemma2 test model");
        let Some(path) = path else {
            eprintln!("skipping real_gemma2_inference_smoke: no local model configured");
            return;
        };

        let mut engine = load_gemma2_engine(&path).expect("load gemma2 engine");
        let (text, finish_reason, prompt_tokens, completion_tokens) = engine
            .generate_chat(
                &[ChatMessage {
                    role: "user".into(),
                    content: serde_json::Value::String("Say hello in one short sentence.".into()),
                }],
                24,
                0.0,
            )
            .expect("gemma2 inference");

        assert!(!text.trim().is_empty(), "gemma2 output should not be empty");
        assert!(
            prompt_tokens > 0,
            "gemma2 prompt token count should be positive"
        );
        assert!(
            completion_tokens > 0,
            "gemma2 completion token count should be positive"
        );
        assert!(
            matches!(finish_reason, "length" | "stop"),
            "unexpected finish reason: {finish_reason}"
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    #[ignore = "requires a real local Llama MLX model"]
    fn real_llama_inference_smoke() {
        let path =
            resolve_or_download_test_model(ModelFamily::Llama).expect("resolve llama test model");
        let Some(path) = path else {
            eprintln!("skipping real_llama_inference_smoke: no local model configured");
            return;
        };

        let mut engine = load_llama_engine(&path).expect("load llama engine");
        let (text, finish_reason, prompt_tokens, completion_tokens) = engine
            .generate_completion("Say hello in one short sentence.", 24, 0.0)
            .expect("llama inference");

        assert!(!text.trim().is_empty(), "llama output should not be empty");
        assert!(
            prompt_tokens > 0,
            "llama prompt token count should be positive"
        );
        assert!(
            completion_tokens > 0,
            "llama completion token count should be positive"
        );
        assert!(
            matches!(finish_reason, "length" | "stop"),
            "unexpected finish reason: {finish_reason}"
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    #[ignore = "requires a real local Llama MLX model"]
    fn real_llama_prefix_cache_extension_smoke() {
        let path =
            resolve_or_download_test_model(ModelFamily::Llama).expect("resolve llama test model");
        let Some(path) = path else {
            eprintln!(
                "skipping real_llama_prefix_cache_extension_smoke: no local model configured"
            );
            return;
        };

        let mut engine = load_llama_engine(&path).expect("load llama engine");

        let base_prompt =
            "Write exactly three short words about startup speed, separated by spaces.";
        let extended_prompt = "Write exactly three short words about startup speed, separated by spaces. Then add one more short word about latency.";

        let base_ids = encode_prompt_ids(&mut engine.tokenizer, base_prompt)
            .expect("encode llama base prompt");
        let base_tokens = build_prompt_array(&base_ids).expect("build llama base prompt array");
        let (_, _, reused_base) = get_or_create_llama_prefill(
            &mut engine.model,
            &base_ids,
            &base_tokens,
            &mut engine.prompt_cache,
        )
        .expect("prefill base prompt");
        assert_eq!(
            reused_base, 0,
            "first prompt should not reuse cached tokens"
        );

        let extended_ids = encode_prompt_ids(&mut engine.tokenizer, extended_prompt)
            .expect("encode llama extended prompt");
        let extended_tokens =
            build_prompt_array(&extended_ids).expect("build llama extended prompt array");
        let (_, _, reused_extended) = get_or_create_llama_prefill(
            &mut engine.model,
            &extended_ids,
            &extended_tokens,
            &mut engine.prompt_cache,
        )
        .expect("prefill extended prompt");
        assert_eq!(
            reused_extended,
            base_ids.len(),
            "extended prompt should reuse the full cached base prompt"
        );

        let (_, _, reused_repeat) = get_or_create_llama_prefill(
            &mut engine.model,
            &extended_ids,
            &extended_tokens,
            &mut engine.prompt_cache,
        )
        .expect("prefill repeated extended prompt");
        assert_eq!(
            reused_repeat,
            extended_ids.len(),
            "exact repeated prompt should reuse the full cached prompt"
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    #[ignore = "requires a real local Qwen3 MLX model"]
    fn real_qwen3_exact_repeat_cache_smoke() {
        let path =
            resolve_or_download_test_model(ModelFamily::Qwen3).expect("resolve qwen3 test model");
        let Some(path) = path else {
            eprintln!("skipping real_qwen3_exact_repeat_cache_smoke: no local model configured");
            return;
        };

        let mut engine = load_qwen3_engine(&path).expect("load qwen3 engine");

        let prompt = "Write exactly three short words about startup speed, separated by spaces.";

        let prompt_ids = render_qwen3_completion_prompt(
            &mut engine.tokenizer,
            engine.chat_template.clone(),
            prompt,
        )
        .map(|ids| prepare_qwen3_prompt_ids(&engine.tokenizer, ids))
        .expect("encode qwen3 prompt");
        let prompt_tokens = build_prompt_array(&prompt_ids).expect("build qwen3 prompt array");
        let (_, _, reused_base) = get_or_create_qwen3_prefill(
            &mut engine.model,
            &prompt_ids,
            &prompt_tokens,
            &mut engine.prompt_cache,
        )
        .expect("prefill base prompt");
        assert_eq!(
            reused_base, 0,
            "first prompt should not reuse cached tokens"
        );

        let (_, _, reused_repeat) = get_or_create_qwen3_prefill(
            &mut engine.model,
            &prompt_ids,
            &prompt_tokens,
            &mut engine.prompt_cache,
        )
        .expect("prefill repeated prompt");
        assert_eq!(
            reused_repeat,
            prompt_ids.len(),
            "exact repeated prompt should reuse the full cached prompt"
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    #[ignore = "requires a real local Mistral MLX model"]
    fn real_mistral_inference_smoke() {
        let path = resolve_or_download_test_model(ModelFamily::Mistral)
            .expect("resolve mistral test model");
        let Some(path) = path else {
            eprintln!("skipping real_mistral_inference_smoke: no local model configured");
            return;
        };

        let mut engine = load_mistral_engine(&path).expect("load mistral engine");
        let (text, finish_reason, prompt_tokens, completion_tokens) = engine
            .generate_completion("Say hello in one short sentence.", 24, 0.0)
            .expect("mistral inference");

        assert!(
            !text.trim().is_empty(),
            "mistral output should not be empty"
        );
        assert!(
            prompt_tokens > 0,
            "mistral prompt token count should be positive"
        );
        assert!(
            completion_tokens > 0,
            "mistral completion token count should be positive"
        );
        assert!(
            matches!(finish_reason, "length" | "stop"),
            "unexpected finish reason: {finish_reason}"
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    #[ignore = "requires a real local Qwen2 MLX model"]
    fn real_qwen2_inference_smoke() {
        let path =
            resolve_or_download_test_model(ModelFamily::Qwen2).expect("resolve qwen2 test model");
        let Some(path) = path else {
            eprintln!("skipping real_qwen2_inference_smoke: no local model configured");
            return;
        };

        let mut engine = load_qwen2_engine(&path).expect("load qwen2 engine");
        let (text, finish_reason, prompt_tokens, completion_tokens) = engine
            .generate_chat(
                &[ChatMessage {
                    role: "user".into(),
                    content: serde_json::Value::String("Say hello in one short sentence.".into()),
                }],
                24,
                0.0,
            )
            .expect("qwen2 inference");

        assert!(!text.trim().is_empty(), "qwen2 output should not be empty");
        assert!(
            prompt_tokens > 0,
            "qwen2 prompt token count should be positive"
        );
        assert!(
            completion_tokens > 0,
            "qwen2 completion token count should be positive"
        );
        assert!(
            matches!(finish_reason, "length" | "stop"),
            "unexpected finish reason: {finish_reason}"
        );
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    #[ignore = "requires a real local Qwen3 MLX model"]
    fn real_qwen3_inference_smoke() {
        let path =
            resolve_or_download_test_model(ModelFamily::Qwen3).expect("resolve qwen3 test model");
        let Some(path) = path else {
            eprintln!("skipping real_qwen3_inference_smoke: no local model configured");
            return;
        };

        let mut engine = load_qwen3_engine(&path).expect("load qwen3 engine");
        let (text, finish_reason, prompt_tokens, completion_tokens) = engine
            .generate_chat(
                &[ChatMessage {
                    role: "user".into(),
                    content: serde_json::Value::String("Say hello in one short sentence.".into()),
                }],
                24,
                0.0,
            )
            .expect("qwen3 inference");

        assert!(!text.trim().is_empty(), "qwen3 output should not be empty");
        assert!(
            prompt_tokens > 0,
            "qwen3 prompt token count should be positive"
        );
        assert!(
            completion_tokens > 0,
            "qwen3 completion token count should be positive"
        );
        assert!(
            matches!(finish_reason, "length" | "stop"),
            "unexpected finish reason: {finish_reason}"
        );
    }

    #[cfg(feature = "native-mlx")]
    fn resolve_test_model_path(env_var: &str, default_model_dir: Option<&str>) -> Option<PathBuf> {
        if let Ok(path) = env::var(env_var) {
            let path = PathBuf::from(path);
            if path.exists() {
                return Some(path);
            }
        }

        let default_model_dir = default_model_dir?;
        let default = env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".models/mlx").join(default_model_dir))?;
        default.exists().then_some(default)
    }

    #[cfg(feature = "native-mlx")]
    fn resolve_or_download_test_model(family: ModelFamily) -> Result<Option<PathBuf>> {
        let spec = family.spec();
        if let Some(path) =
            resolve_test_model_path(spec.test_model_path_env_var, spec.default_test_model_dir)
        {
            return Ok(Some(path));
        }

        let repo_id = match env::var(spec.test_repo_env_var) {
            Ok(repo_id) => repo_id,
            Err(_) => match spec.default_test_repo_id {
                Some(repo_id) => repo_id.to_string(),
                None => return Ok(None),
            },
        };
        if let Some(path) = resolve_cached_repo_snapshot(&repo_id) {
            return Ok(Some(path));
        }
        download_model_repo(&repo_id).map(Some)
    }

    #[cfg(feature = "native-mlx")]
    fn resolve_cached_repo_snapshot(repo_id: &str) -> Option<PathBuf> {
        let cache_root = env::var_os("HF_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|home| home.join(".cache/huggingface"))
            })
            .map(|root| root.join("hub"))?;
        let repo_dir_name = format!("models--{}", repo_id.replace('/', "--"));
        let snapshots_dir = cache_root.join(repo_dir_name).join("snapshots");
        let mut snapshots = std::fs::read_dir(&snapshots_dir)
            .ok()?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        snapshots.sort();
        snapshots
            .into_iter()
            .rev()
            .find(|snapshot| snapshot_looks_complete(snapshot))
    }

    #[cfg(feature = "native-mlx")]
    fn snapshot_looks_complete(snapshot: &Path) -> bool {
        if snapshot.join("model.safetensors").exists() {
            return true;
        }
        let index_path = snapshot.join("model.safetensors.index.json");
        if !index_path.exists() {
            return false;
        }
        let Ok(raw) = std::fs::read_to_string(index_path) else {
            return false;
        };
        let Ok(index) = serde_json::from_str::<crate::quantized::WeightMap>(&raw) else {
            return false;
        };
        index
            .weight_map
            .values()
            .all(|filename| snapshot.join(filename).exists())
    }

    #[cfg(feature = "native-mlx")]
    fn download_model_repo(repo_id: &str) -> Result<PathBuf> {
        let api = Api::new().map_err(|err| anyhow!("failed to create hf api: {err}"))?;
        let repo = api.model(repo_id.to_string());
        let info = repo
            .info()
            .map_err(|err| anyhow!("failed to inspect repo {repo_id}: {err}"))?;

        let mut config_path = None;
        for sibling in info.siblings {
            let path = repo.download(&sibling.rfilename).map_err(|err| {
                anyhow!("failed to download {repo_id}/{}: {err}", sibling.rfilename)
            })?;
            if sibling.rfilename == "config.json" {
                config_path = Some(path);
            }
        }

        let config_path =
            config_path.ok_or_else(|| anyhow!("repo {repo_id} is missing config.json"))?;
        config_path
            .parent()
            .map(PathBuf::from)
            .ok_or_else(|| anyhow!("failed to resolve downloaded snapshot directory for {repo_id}"))
    }
}
