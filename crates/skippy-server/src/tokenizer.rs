use std::{
    sync::atomic::{AtomicBool, Ordering},
    sync::{Arc, Mutex},
};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use mesh_native_serving_plugin_api as native_plugin_api;
use serde::{Deserialize, Serialize};
use skippy_protocol::StageConfig;
use skippy_tokenizer::{
    EncodeRequest, EncodeResponse, InputPiece, SpecialTokenPolicy, TokenizeBatchItem,
    TokenizeRequest, TokenizeResponse, Tokenizer, TokenizerError, TokenizerIdentity,
    TokenizerLimits,
};
pub use skippy_tokenizer::{
    MAX_TOKENIZE_BATCH_INPUT_BYTES, MAX_TOKENIZE_BATCH_SIZE, MAX_TOKENIZE_INPUT_BYTES,
    MAX_TOKENIZE_TOKENS, TOKENIZER_VERSION,
};

use crate::runtime_state::RuntimeState;

mod binding;

use binding::{inventory_from_stage, tokenizer_binding_digest};

pub type TokenizerCapabilityError = TokenizerError;

pub(crate) fn tokenizer_identity_from_stage(
    stage_index: u32,
    model_id: &str,
    source_model_sha256: Option<&str>,
) -> Result<TokenizerIdentity, TokenizerCapabilityError> {
    if stage_index != 0 {
        return Err(TokenizerCapabilityError::StageZeroRequired);
    }
    if model_id.trim().is_empty() {
        return Err(TokenizerCapabilityError::IdentityUnavailable);
    }
    let source_model_sha256 = source_model_sha256
        .filter(|value| is_sha256(value))
        .ok_or(TokenizerCapabilityError::IdentityUnavailable)?
        .to_ascii_lowercase();
    Ok(TokenizerIdentity {
        model_id: model_id.to_owned(),
        tokenizer_id: format!("gguf-source-sha256:{source_model_sha256}"),
        source_model_sha256,
        tokenizer_version: Some(TOKENIZER_VERSION.to_owned()),
        stage_index,
        serving_profile: Some("stage-zero".to_owned()),
    })
}

fn identity_matches(expected: &TokenizerIdentity, actual: &TokenizerIdentity) -> bool {
    expected.model_id == actual.model_id
        && expected.source_model_sha256 == actual.source_model_sha256
        && expected.tokenizer_id == actual.tokenizer_id
        && expected.stage_index == actual.stage_index
        && expected
            .tokenizer_version
            .as_ref()
            .is_none_or(|value| actual.tokenizer_version.as_ref() == Some(value))
        && expected
            .serving_profile
            .as_ref()
            .is_none_or(|value| actual.serving_profile.as_ref() == Some(value))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

trait TokenizerSource: Send + Sync {
    fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        max_tokens: usize,
    ) -> Result<Vec<i32>, TokenizerCapabilityError>;
    fn encode(
        &self,
        pieces: &[InputPiece],
        max_tokens: usize,
    ) -> Result<Vec<i32>, TokenizerCapabilityError>;
    fn token_pieces(&self, token_ids: &[i32]) -> Result<Vec<Vec<u8>>, TokenizerCapabilityError>;
}

struct LoadedStageZeroTokenizer {
    runtime: Arc<Mutex<RuntimeState>>,
    active: Arc<AtomicBool>,
    #[cfg(test)]
    initial_check_signal: Option<Arc<std::sync::Barrier>>,
}

impl LoadedStageZeroTokenizer {
    fn ensure_active(&self) -> Result<(), TokenizerCapabilityError> {
        if self.active.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(TokenizerCapabilityError::RuntimeUnavailable)
        }
    }
}

impl TokenizerSource for LoadedStageZeroTokenizer {
    fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        max_tokens: usize,
    ) -> Result<Vec<i32>, TokenizerCapabilityError> {
        self.ensure_active()?;
        #[cfg(test)]
        if let Some(signal) = &self.initial_check_signal {
            signal.wait();
        }
        let runtime =
            self.runtime
                .lock()
                .map_err(|_| TokenizerCapabilityError::BackendFailure {
                    message: "runtime lock poisoned".to_owned(),
                })?;
        self.ensure_active()?;
        let tokens = runtime
            .model
            .tokenize_bounded(text, add_special, max_tokens)
            .map_err(|error| TokenizerCapabilityError::BackendFailure {
                message: error.to_string(),
            })?;
        tokens.ok_or(TokenizerCapabilityError::TooManyTokens { limit: max_tokens })
    }

    fn token_pieces(&self, token_ids: &[i32]) -> Result<Vec<Vec<u8>>, TokenizerCapabilityError> {
        self.ensure_active()?;
        let runtime =
            self.runtime
                .lock()
                .map_err(|_| TokenizerCapabilityError::BackendFailure {
                    message: "runtime lock poisoned".to_owned(),
                })?;
        self.ensure_active()?;
        token_ids
            .iter()
            .map(|token_id| {
                runtime.model.detokenize_bytes(&[*token_id]).map_err(|_| {
                    TokenizerCapabilityError::BackendFailure {
                        message: "detokenization failed".to_owned(),
                    }
                })
            })
            .collect()
    }

    fn encode(
        &self,
        pieces: &[InputPiece],
        max_tokens: usize,
    ) -> Result<Vec<i32>, TokenizerCapabilityError> {
        let mut bytes = Vec::new();
        for piece in pieces {
            match piece {
                InputPiece::Bytes(piece_bytes) => bytes.extend_from_slice(piece_bytes),
                InputPiece::Control { .. } => {
                    return Err(TokenizerCapabilityError::UnsupportedInput {
                        reason: "native control descriptor is unsupported by this backend"
                            .to_owned(),
                    });
                }
            }
        }
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        if max_tokens == 0 {
            return Err(TokenizerCapabilityError::TooManyTokens { limit: 0 });
        }
        let text = std::str::from_utf8(&bytes).map_err(|_| {
            TokenizerCapabilityError::UnsupportedInput {
                reason: "input is not valid UTF-8".to_owned(),
            }
        })?;
        if text.as_bytes().contains(&0) {
            return Err(TokenizerCapabilityError::UnsupportedInput {
                reason: "input contains an interior NUL byte".to_owned(),
            });
        }
        self.tokenize(text, false, max_tokens)
    }
}

#[derive(Clone)]
pub struct TokenizerCapability {
    identity: TokenizerIdentity,
    source: Arc<dyn TokenizerSource>,
    inventory: Option<Arc<native_plugin_api::TokenizerInventory>>,
    binding_digest: Option<[u8; 32]>,
}

impl TokenizerCapability {
    pub fn from_stage_zero(
        config: &StageConfig,
        runtime: Arc<Mutex<RuntimeState>>,
    ) -> Result<Self, TokenizerCapabilityError> {
        Self::from_stage_zero_with_lifecycle(config, runtime, Arc::new(AtomicBool::new(true)))
    }

    pub(crate) fn from_stage_zero_with_lifecycle(
        config: &StageConfig,
        runtime: Arc<Mutex<RuntimeState>>,
        active: Arc<AtomicBool>,
    ) -> Result<Self, TokenizerCapabilityError> {
        let identity = tokenizer_identity_from_stage(
            config.stage_index,
            &config.model_id,
            config.source_model_sha256.as_deref(),
        )?;
        let source: Arc<dyn TokenizerSource> = Arc::new(LoadedStageZeroTokenizer {
            runtime,
            active,
            #[cfg(test)]
            initial_check_signal: None,
        });
        let inventory = inventory_from_stage(config, &identity, source.as_ref()).map(Arc::new);
        let binding_digest = inventory
            .as_deref()
            .map(|inventory| tokenizer_binding_digest(&identity, inventory));
        Ok(Self {
            identity,
            source,
            inventory,
            binding_digest,
        })
    }

    pub fn identity(&self) -> &TokenizerIdentity {
        &self.identity
    }

    /// A fully materialized, immutable native vocabulary, when the stage was
    /// bound from a readable source GGUF. Inventory construction occurs while
    /// binding the model, never from the decode proposal path.
    pub fn inventory(&self) -> Option<&native_plugin_api::TokenizerInventory> {
        self.inventory.as_deref()
    }

    /// A stable digest of the bound identity, inventory, and encode behavior.
    /// It is absent when the model did not expose a complete native inventory.
    pub fn binding_digest(&self) -> Option<[u8; 32]> {
        self.binding_digest
    }

    pub fn limits(&self) -> TokenizerLimits {
        <Self as Tokenizer>::limits(self)
    }

    pub fn encode(
        &self,
        request: EncodeRequest,
    ) -> Result<EncodeResponse, TokenizerCapabilityError> {
        <Self as Tokenizer>::encode(self, request)
    }

    pub fn tokenize(
        &self,
        request: TokenizeRequest,
    ) -> Result<TokenizeResponse, TokenizerCapabilityError> {
        self.tokenize_batch(&[request])?
            .into_iter()
            .next()
            .expect("single-request tokenizer batch must return one item")
            .result
    }
}

impl Tokenizer for TokenizerCapability {
    fn identity(&self) -> &TokenizerIdentity {
        &self.identity
    }

    fn limits(&self) -> TokenizerLimits {
        TokenizerLimits::default()
    }

    fn tokenize_batch(
        &self,
        requests: &[TokenizeRequest],
    ) -> Result<Vec<TokenizeBatchItem>, TokenizerCapabilityError> {
        let limits = self.limits();
        if requests.len() > limits.max_batch_size {
            return Err(TokenizerCapabilityError::BatchTooLarge {
                limit: limits.max_batch_size,
            });
        }
        let batch_input_bytes = requests
            .iter()
            .map(|request| request.text.len())
            .try_fold(0usize, usize::checked_add)
            .ok_or(TokenizerCapabilityError::BatchInputTooLarge {
                limit: limits.max_batch_input_bytes,
            })?;
        if batch_input_bytes > limits.max_batch_input_bytes {
            return Err(TokenizerCapabilityError::BatchInputTooLarge {
                limit: limits.max_batch_input_bytes,
            });
        }

        let items = requests
            .iter()
            .enumerate()
            .map(|(request_index, request)| {
                let result = self.tokenize_one(request, limits);
                TokenizeBatchItem {
                    request_index,
                    result,
                }
            })
            .collect();
        Ok(items)
    }

    fn encode(&self, request: EncodeRequest) -> Result<EncodeResponse, TokenizerCapabilityError> {
        if !identity_matches(&request.expected_identity, &self.identity) {
            return Err(TokenizerCapabilityError::IdentityMismatch {
                expected: Box::new(request.expected_identity),
                actual: Box::new(self.identity.clone()),
            });
        }
        let limits = self.limits();
        if request.pieces.len() > skippy_tokenizer::MAX_TOKENIZE_PIECES {
            return Err(TokenizerCapabilityError::TooManyPieces {
                limit: skippy_tokenizer::MAX_TOKENIZE_PIECES,
            });
        }
        let input_bytes =
            request
                .total_input_bytes()
                .ok_or(TokenizerCapabilityError::InputTooLarge {
                    limit: limits.max_input_bytes,
                })?;
        if input_bytes > limits.max_input_bytes {
            return Err(TokenizerCapabilityError::InputTooLarge {
                limit: limits.max_input_bytes,
            });
        }
        let token_ids = self
            .source
            .encode(&request.pieces, limits.max_output_tokens)?;
        if token_ids.len() > limits.max_output_tokens {
            return Err(TokenizerCapabilityError::TooManyTokens {
                limit: limits.max_output_tokens,
            });
        }
        Ok(EncodeResponse {
            identity: self.identity.clone(),
            token_ids,
        })
    }
}

impl TokenizerCapability {
    fn tokenize_one(
        &self,
        request: &TokenizeRequest,
        limits: TokenizerLimits,
    ) -> Result<TokenizeResponse, TokenizerCapabilityError> {
        if !identity_matches(&request.expected_identity, &self.identity) {
            return Err(TokenizerCapabilityError::IdentityMismatch {
                expected: Box::new(request.expected_identity.clone()),
                actual: Box::new(self.identity.clone()),
            });
        }
        if request.text.len() > limits.max_input_bytes {
            return Err(TokenizerCapabilityError::InputTooLarge {
                limit: limits.max_input_bytes,
            });
        }
        let token_ids = self.source.tokenize(
            &request.text,
            request.special_tokens == SpecialTokenPolicy::Add,
            limits.max_output_tokens,
        )?;
        if token_ids.len() > limits.max_output_tokens {
            return Err(TokenizerCapabilityError::TooManyTokens {
                limit: limits.max_output_tokens,
            });
        }
        let token_pieces = request
            .include_token_pieces
            .then(|| self.source.token_pieces(&token_ids))
            .transpose()?;
        Ok(TokenizeResponse {
            identity: self.identity.clone(),
            token_ids,
            token_pieces,
        })
    }
}

#[derive(Debug, Serialize)]
struct TokenizerErrorBody {
    error: &'static str,
}

#[derive(Deserialize)]
struct HttpTokenizeRequest {
    expected_identity: TokenizerIdentity,
    text: String,
    #[serde(default)]
    special_tokens: SpecialTokenPolicy,
    #[serde(default)]
    add_special: Option<bool>,
    #[serde(default)]
    include_token_pieces: bool,
}

impl HttpTokenizeRequest {
    fn into_request(self) -> TokenizeRequest {
        let special_tokens = self.add_special.map_or(self.special_tokens, |add_special| {
            if add_special {
                SpecialTokenPolicy::Add
            } else {
                SpecialTokenPolicy::Omit
            }
        });
        TokenizeRequest {
            expected_identity: self.expected_identity,
            text: self.text,
            special_tokens,
            include_token_pieces: self.include_token_pieces,
        }
    }
}

struct TokenizerHttpError(TokenizerCapabilityError);

impl IntoResponse for TokenizerHttpError {
    fn into_response(self) -> Response {
        let status = match self.0 {
            TokenizerCapabilityError::InputTooLarge { .. }
            | TokenizerCapabilityError::TooManyPieces { .. }
            | TokenizerCapabilityError::BatchInputTooLarge { .. }
            | TokenizerCapabilityError::BatchTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            TokenizerCapabilityError::TooManyTokens { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            TokenizerCapabilityError::UnsupportedInput { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            TokenizerCapabilityError::BackendFailure { .. } => StatusCode::INTERNAL_SERVER_ERROR,
            TokenizerCapabilityError::StageZeroRequired
            | TokenizerCapabilityError::UnsupportedStage { .. }
            | TokenizerCapabilityError::IdentityUnavailable
            | TokenizerCapabilityError::IdentityMismatch { .. } => StatusCode::CONFLICT,
            TokenizerCapabilityError::RuntimeUnavailable => StatusCode::SERVICE_UNAVAILABLE,
        };
        (
            status,
            Json(TokenizerErrorBody {
                error: self.0.code(),
            }),
        )
            .into_response()
    }
}

/// Skippy's tokenizer extension for the product OpenAI endpoint.
///
/// The capability must come from the same already-loaded stage-0 runtime used
/// for generation. This router owns the only HTTP tokenizer route; the stage
/// transport server deliberately does not expose it.
pub(crate) fn tokenizer_http_router(capability: TokenizerCapability) -> Router {
    Router::new()
        .route("/v1/tokenize", post(tokenize_entrypoint))
        .with_state(capability)
}

async fn tokenize_entrypoint(
    State(capability): State<TokenizerCapability>,
    Json(request): Json<HttpTokenizeRequest>,
) -> Result<Json<TokenizeResponse>, TokenizerHttpError> {
    tokio::task::spawn_blocking(move || capability.tokenize(request.into_request()))
        .await
        .map_err(|_| {
            TokenizerHttpError(TokenizerCapabilityError::BackendFailure {
                message: "tokenizer task failed".to_owned(),
            })
        })?
        .map(Json)
        .map_err(TokenizerHttpError)
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Barrier, atomic::AtomicBool},
        thread,
        time::Duration,
    };

    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode, header::CONTENT_TYPE},
    };
    use skippy_protocol::LoadMode;
    use tower::ServiceExt;

    use super::*;

    const SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn stage_config(source_model_path: Option<String>, model_path: Option<String>) -> StageConfig {
        StageConfig {
            run_id: "run".to_owned(),
            topology_id: "topology".to_owned(),
            model_id: "model".to_owned(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path,
            projector_path: None,
            stage_id: "stage-0".to_owned(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 1_024,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_owned(),
            cache_type_v: "f16".to_owned(),
            flash_attn_type: Default::default(),
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: false,
            resident_tensor_names: Vec::new(),
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_owned(),
            upstream: None,
            downstream: None,
            ..StageConfig::default()
        }
    }

    struct RecordingTokenizer {
        tokens: Vec<i32>,
    }

    struct UnreachableOpenAiBackend;

    #[async_trait::async_trait]
    impl openai_frontend::OpenAiBackend for UnreachableOpenAiBackend {
        async fn models(&self) -> openai_frontend::OpenAiResult<Vec<openai_frontend::ModelObject>> {
            unreachable!("tokenizer requests must not enter the generic OpenAI backend")
        }

        async fn chat_completion(
            &self,
            _request: openai_frontend::ChatCompletionRequest,
        ) -> openai_frontend::OpenAiResult<openai_frontend::ChatCompletionResponse> {
            unreachable!("tokenizer requests must not enter the generic OpenAI backend")
        }

        async fn chat_completion_stream(
            &self,
            _request: openai_frontend::ChatCompletionRequest,
            _context: openai_frontend::OpenAiRequestContext,
        ) -> openai_frontend::OpenAiResult<openai_frontend::ChatCompletionStream> {
            unreachable!("tokenizer requests must not enter the generic OpenAI backend")
        }
    }

    impl TokenizerSource for RecordingTokenizer {
        fn tokenize(
            &self,
            _text: &str,
            add_special: bool,
            max_tokens: usize,
        ) -> Result<Vec<i32>, TokenizerCapabilityError> {
            let mut tokens = self.tokens.clone();
            if add_special {
                tokens.insert(0, 1);
            }
            if tokens.len() > max_tokens {
                return Err(TokenizerCapabilityError::TooManyTokens { limit: max_tokens });
            }
            Ok(tokens)
        }

        fn token_pieces(
            &self,
            token_ids: &[i32],
        ) -> Result<Vec<Vec<u8>>, TokenizerCapabilityError> {
            Ok(token_ids
                .iter()
                .map(|token_id| token_id.to_string().into_bytes())
                .collect())
        }

        fn encode(
            &self,
            pieces: &[InputPiece],
            max_tokens: usize,
        ) -> Result<Vec<i32>, TokenizerCapabilityError> {
            if pieces
                .iter()
                .any(|piece| matches!(piece, InputPiece::Control { .. }))
            {
                return Err(TokenizerCapabilityError::UnsupportedInput {
                    reason: "recording source does not support controls".to_owned(),
                });
            }
            for piece in pieces {
                let InputPiece::Bytes(bytes) = piece else {
                    unreachable!();
                };
                std::str::from_utf8(bytes).map_err(|_| {
                    TokenizerCapabilityError::UnsupportedInput {
                        reason: "recording source only accepts UTF-8".to_owned(),
                    }
                })?;
            }
            self.tokenize("", false, max_tokens)
        }
    }

    fn identity() -> TokenizerIdentity {
        tokenizer_identity_from_stage(0, "model", Some(SHA256)).unwrap()
    }

    fn capability(tokens: Vec<i32>) -> (TokenizerCapability, Arc<RecordingTokenizer>) {
        let source = Arc::new(RecordingTokenizer { tokens });
        (
            TokenizerCapability {
                identity: identity(),
                source: source.clone(),
                inventory: None,
                binding_digest: None,
            },
            source,
        )
    }

    fn request(text: String) -> TokenizeRequest {
        TokenizeRequest {
            expected_identity: identity(),
            text,
            special_tokens: SpecialTokenPolicy::Omit,
            include_token_pieces: false,
        }
    }

    #[test]
    fn queued_tokenization_rechecks_lifecycle_after_runtime_lock() {
        let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
        let active = Arc::new(AtomicBool::new(true));
        let source = Arc::new(LoadedStageZeroTokenizer {
            runtime: Arc::clone(&runtime),
            active: Arc::clone(&active),
            initial_check_signal: Some(Arc::new(Barrier::new(2))),
        });
        let held = runtime.lock().expect("runtime lock");
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let request_source = Arc::clone(&source);
        thread::spawn(move || {
            result_tx
                .send(request_source.tokenize("hello", false, 8))
                .expect("send tokenizer result");
        });

        source
            .initial_check_signal
            .as_ref()
            .expect("test synchronization signal")
            .wait();
        active.store(false, Ordering::Release);
        drop(held);

        assert_eq!(
            result_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("receive tokenizer result")
                .unwrap_err(),
            TokenizerCapabilityError::RuntimeUnavailable
        );
    }

    async fn post_tokenize(capability: TokenizerCapability, request: &TokenizeRequest) -> Response {
        tokenizer_http_router(capability)
            .oneshot(
                Request::post("/v1/tokenize")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn response_json(response: Response) -> serde_json::Value {
        serde_json::from_slice(
            &to_bytes(response.into_body(), 64 * 1024 * 1024)
                .await
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn tokenization_returns_source_tokens() {
        let (capability, _) = capability(vec![4, 5]);
        let mut request = request("hello".to_string());
        request.special_tokens = SpecialTokenPolicy::Add;
        let response = capability.tokenize(request).unwrap();
        assert_eq!(response.token_ids, vec![1, 4, 5]);
    }

    #[test]
    fn encode_preserves_identity_and_rejects_non_utf8_without_fallback() {
        let (capability, _) = capability(vec![4, 5]);
        let response = capability
            .encode(EncodeRequest::bytes(identity(), b"hello".to_vec()))
            .unwrap();
        assert_eq!(response.identity, identity());
        assert_eq!(response.token_ids, vec![4, 5]);

        assert!(matches!(
            capability.encode(EncodeRequest::bytes(identity(), vec![0xff])),
            Err(TokenizerCapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn encode_enforces_the_input_bound_before_calling_the_source() {
        let (capability, _) = capability(vec![4, 5]);
        let request = EncodeRequest::new(
            identity(),
            vec![InputPiece::Bytes(vec![
                b'a';
                skippy_tokenizer::MAX_TOKENIZE_INPUT_BYTES
                    + 1
            ])],
        );
        assert_eq!(
            capability.encode(request).unwrap_err(),
            TokenizerCapabilityError::InputTooLarge {
                limit: skippy_tokenizer::MAX_TOKENIZE_INPUT_BYTES
            }
        );
    }

    #[test]
    fn encode_rejects_opaque_controls_when_backend_cannot_preserve_them() {
        let (capability, _) = capability(vec![4, 5]);
        let request = EncodeRequest::new(
            identity(),
            vec![
                InputPiece::Bytes(b"before".to_vec()),
                InputPiece::Control {
                    descriptor: vec![0xff, 0x00],
                },
                InputPiece::Bytes(b"after".to_vec()),
            ],
        );
        assert!(matches!(
            capability.encode(request),
            Err(TokenizerCapabilityError::UnsupportedInput { .. })
        ));
    }

    #[test]
    fn batch_results_keep_request_indexes_and_attribute_identity_errors() {
        let (capability, _) = capability(vec![4, 5]);
        let mut mismatched = request("second".to_owned());
        mismatched.expected_identity.model_id = "other-model".to_owned();
        let expected_identity = mismatched.expected_identity.clone();

        let results = capability
            .tokenize_batch(&[request("first".to_owned()), mismatched.clone()])
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].request_index, 0);
        assert!(results[0].result.is_ok());
        assert_eq!(results[1].request_index, 1);
        assert_eq!(
            results[1].result.as_ref().unwrap_err().clone(),
            TokenizerCapabilityError::IdentityMismatch {
                expected: Box::new(expected_identity),
                actual: Box::new(identity()),
            }
        );
    }

    #[test]
    fn batch_size_and_input_limits_are_rejected_before_tokenization() {
        let (capability, _) = capability(vec![4]);
        let too_many = (0..=MAX_TOKENIZE_BATCH_SIZE)
            .map(|index| request(index.to_string()))
            .collect::<Vec<_>>();
        assert_eq!(
            capability.tokenize_batch(&too_many).unwrap_err(),
            TokenizerCapabilityError::BatchTooLarge {
                limit: MAX_TOKENIZE_BATCH_SIZE
            }
        );

        let too_many_bytes = (0..=MAX_TOKENIZE_BATCH_INPUT_BYTES / MAX_TOKENIZE_INPUT_BYTES)
            .map(|_| request("x".repeat(MAX_TOKENIZE_INPUT_BYTES)))
            .collect::<Vec<_>>();
        assert_eq!(
            capability.tokenize_batch(&too_many_bytes).unwrap_err(),
            TokenizerCapabilityError::BatchInputTooLarge {
                limit: MAX_TOKENIZE_BATCH_INPUT_BYTES
            }
        );
    }

    #[test]
    fn bounds_input_and_output() {
        let (input_bounded, _) = capability(Vec::new());
        let error = input_bounded
            .tokenize(request("x".repeat(MAX_TOKENIZE_INPUT_BYTES + 1)))
            .unwrap_err();
        assert_eq!(
            error,
            TokenizerCapabilityError::InputTooLarge {
                limit: MAX_TOKENIZE_INPUT_BYTES
            }
        );

        let (output_bounded, _) = capability(vec![7; MAX_TOKENIZE_TOKENS + 1]);
        let error = output_bounded
            .tokenize(request("x".to_string()))
            .unwrap_err();
        assert_eq!(
            error,
            TokenizerCapabilityError::TooManyTokens {
                limit: MAX_TOKENIZE_TOKENS
            }
        );
    }

    #[test]
    fn identity_is_authoritative_and_fail_closed() {
        assert_eq!(
            tokenizer_identity_from_stage(1, "model", Some(SHA256)).unwrap_err(),
            TokenizerCapabilityError::StageZeroRequired
        );
        assert_eq!(
            tokenizer_identity_from_stage(0, "model", None).unwrap_err(),
            TokenizerCapabilityError::IdentityUnavailable
        );
        let (capability, _) = capability(Vec::new());
        let mut request = request("x".to_string());
        request.expected_identity.model_id = "another-model".to_string();
        let expected_identity = request.expected_identity.clone();
        assert_eq!(
            capability.tokenize(request).unwrap_err(),
            TokenizerCapabilityError::IdentityMismatch {
                expected: Box::new(expected_identity),
                actual: Box::new(identity()),
            }
        );
    }

    #[test]
    fn legacy_identity_without_optional_provenance_fields_remains_usable() {
        let (capability, _) = capability(vec![4, 5]);
        let mut request = request("legacy".to_string());
        request.expected_identity.tokenizer_version = None;
        request.expected_identity.serving_profile = None;

        assert_eq!(capability.tokenize(request).unwrap().token_ids, vec![4, 5]);
    }

    #[test]
    fn source_gguf_path_falls_back_when_source_model_path_is_unavailable() {
        let temp_dir = tempfile::tempdir().expect("create temporary GGUF directory");
        let model_path = temp_dir.path().join("model.gguf");
        std::fs::write(&model_path, []).expect("create temporary GGUF");
        let config = stage_config(
            Some(temp_dir.path().join("missing.gguf").display().to_string()),
            Some(model_path.display().to_string()),
        );

        assert_eq!(
            binding::source_gguf_path(&config),
            Some(model_path.as_path())
        );
    }

    #[test]
    fn optional_pieces_align_one_to_one_with_exact_token_ids() {
        let (capability, _) = capability(vec![4, 29, 8]);
        let mut request = request("hello".to_string());
        request.include_token_pieces = true;
        let response = capability.tokenize(request).unwrap();
        assert_eq!(response.token_ids, vec![4, 29, 8]);
        assert_eq!(
            response.token_pieces.unwrap(),
            vec![b"4".to_vec(), b"29".to_vec(), b"8".to_vec()]
        );
    }

    #[tokio::test]
    async fn openai_tokenizer_route_preserves_the_exact_wire_contract() {
        let (capability, _) = capability(vec![4, 29, 8]);
        let mut request = request("hello".to_owned());
        request.include_token_pieces = true;

        let response =
            crate::embedded::openai_backend_router(Arc::new(UnreachableOpenAiBackend), capability)
                .oneshot(
                    Request::post("/v1/tokenize")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(serde_json::to_vec(&request).unwrap()))
                        .unwrap(),
                )
                .await
                .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response_json(response).await,
            serde_json::json!({
                "identity": identity(),
                "token_ids": [4, 29, 8],
                "token_pieces": [[52], [50, 57], [56]],
            })
        );
    }

    #[tokio::test]
    async fn openai_tokenizer_route_rejects_identity_mismatch() {
        let (capability, _) = capability(vec![4, 5]);
        let mut request = request("hello".to_owned());
        request.expected_identity.model_id = "wrong-model".to_owned();

        let response = post_tokenize(capability, &request).await;

        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(response).await,
            serde_json::json!({"error": "identity_mismatch"})
        );
    }

    #[tokio::test]
    async fn openai_tokenizer_route_enforces_input_and_output_bounds() {
        let (input_bounded, _) = capability(Vec::new());
        let input_response = post_tokenize(
            input_bounded,
            &request("x".repeat(MAX_TOKENIZE_INPUT_BYTES + 1)),
        )
        .await;
        assert_eq!(input_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            response_json(input_response).await,
            serde_json::json!({"error": "input_too_large"})
        );

        let (output_bounded, _) = capability(vec![7; MAX_TOKENIZE_TOKENS + 1]);
        let output_response = post_tokenize(output_bounded, &request("x".to_owned())).await;
        assert_eq!(output_response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            response_json(output_response).await,
            serde_json::json!({"error": "too_many_tokens"})
        );
    }
}
