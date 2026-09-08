use crate::frontend::NativeMtpStats;
use crate::frontend::util::now_unix_millis;
use crate::frontend::util::stable_wire_id;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::CompletionRequest;
use skippy_protocol::binary::StageReplyStats;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;
use uuid::Uuid;

pub(in crate::frontend) static OPENAI_GENERATION_COUNTER: AtomicU64 = AtomicU64::new(1);
static OPENAI_PROCESS_NONCE: OnceLock<String> = OnceLock::new();

/// Sentinel meaning "no caller-specified max completion length; let the
/// request consume the entire remaining context window when the client
/// also omits max_tokens".
///
/// This is opt-in and should only be wired in by callers that have made
/// a deliberate decision to allow unbounded chat completions. The
/// embedded mesh-llm wiring uses [`DEFAULT_EMBEDDED_MAX_TOKENS`] instead.
pub const CONTEXT_BUDGET_MAX_TOKENS: u32 = u32::MAX;

/// Default max completion tokens for embedded mesh-llm chat serving when
/// the client omits max_tokens. Bounded so that an adversarial or
/// non-terminating generation cannot run for the full context window.
/// Clients can still request more by sending max_tokens explicitly, up
/// to the remaining context budget.
///
/// When the configured context window is smaller than this value, the
/// request is silently clamped to whatever remaining budget exists
/// rather than rejected — see `GenerationTokenLimit::resolve`.
pub const DEFAULT_EMBEDDED_MAX_TOKENS: u32 = 4096;
pub(in crate::frontend) const GENERATION_RETRY_AFTER_SECS: u64 = 1;
pub(in crate::frontend) const MAX_EXACT_REPLAY_TOKENS: usize = 8;

#[derive(Clone)]
pub(in crate::frontend) struct OpenAiGenerationIds {
    pub(in crate::frontend) session_label: String,
    pub(in crate::frontend) session_id: u64,
    pub(in crate::frontend) request_id: u64,
    pub(in crate::frontend) request_started_at: Instant,
    pub(in crate::frontend) agent_session_id: Option<Box<str>>,
    pub(in crate::frontend) agent_session_trusted: bool,
    pub(in crate::frontend) cache: OpenAiCacheHints,
    /// Raw UUID bytes of the OpenAI frontend's `x-request-id`, when this
    /// generation was admitted through the OpenAI HTTP boundary. `None` for
    /// non-frontend callers (tests, direct embedding without a frontend
    /// request). Carried onto `GenerationStart`/`GenerationReceipt` so the
    /// runtime-event adapter can correlate a generation under the same
    /// request root Task 11's OpenAI observer reserves.
    pub(in crate::frontend) frontend_request_id: Option<[u8; 16]>,
}

impl OpenAiGenerationIds {
    pub(in crate::frontend) fn new_with_trust(
        cache: OpenAiCacheHints,
        agent_session_id: Option<&str>,
        agent_session_trusted: bool,
        frontend_request_id: Option<[u8; 16]>,
    ) -> Self {
        let request_started_at = Instant::now();
        let sequence = OPENAI_GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let process_nonce = process_nonce();
        // Only identity supplied by the configured trusted transport header is
        // allowed to bind multiple requests to one native KV session. Request
        // payload metadata and protocol-level conversation IDs remain isolated.
        let trusted_session_id = agent_session_id
            .filter(|_| agent_session_trusted)
            .map(|id| stable_wire_id(&[b"openai-agent-session", id.as_bytes()]));
        let session_label = trusted_session_id
            .map(|id| format!("openai-agent-session-{id}"))
            .unwrap_or_else(|| {
                format!(
                    "openai-session-{}-{process_nonce}-{sequence}",
                    now_unix_millis()
                )
            });
        Self {
            session_id: trusted_session_id
                .unwrap_or_else(|| stable_wire_id(&[session_label.as_bytes()])),
            request_id: request_id(&session_label, sequence, process_nonce),
            session_label,
            request_started_at,
            agent_session_id: agent_session_id.map(Into::into),
            agent_session_trusted: agent_session_trusted && agent_session_id.is_some(),
            cache,
            frontend_request_id,
        }
    }

    pub(in crate::frontend) fn session_id_string(&self) -> String {
        self.session_id.to_string()
    }

    pub(in crate::frontend) fn request_id_string(&self) -> String {
        self.request_id.to_string()
    }
}

fn process_nonce() -> &'static str {
    OPENAI_PROCESS_NONCE.get_or_init(|| Uuid::new_v4().simple().to_string())
}

fn request_id(session_label: &str, sequence: u64, process_nonce: &str) -> u64 {
    let request_sequence = sequence.to_string();
    stable_wire_id(&[
        session_label.as_bytes(),
        b"request",
        process_nonce.as_bytes(),
        request_sequence.as_bytes(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trusted_agent_session_reuses_native_session_but_not_request_id() {
        let first = OpenAiGenerationIds::new_with_trust(
            OpenAiCacheHints::default(),
            Some("agent-42"),
            true,
            None,
        );
        let second = OpenAiGenerationIds::new_with_trust(
            OpenAiCacheHints::default(),
            Some("agent-42"),
            true,
            None,
        );

        assert!(first.session_label.starts_with("openai-agent-session-"));
        assert!(!first.session_label.contains("agent-42"));
        assert_eq!(first.session_label, second.session_label);
        assert_eq!(first.session_id, second.session_id);
        assert_ne!(first.request_id, second.request_id);
    }

    #[test]
    fn requests_without_agent_session_get_fresh_native_sessions() {
        let first =
            OpenAiGenerationIds::new_with_trust(OpenAiCacheHints::default(), None, false, None);
        let second =
            OpenAiGenerationIds::new_with_trust(OpenAiCacheHints::default(), None, false, None);

        assert!(first.session_label.contains(process_nonce()));
        assert_ne!(first.session_id, second.session_id);
        assert_ne!(first.request_id, second.request_id);
    }

    #[test]
    fn repeated_untrusted_agent_sessions_get_fresh_native_sessions() {
        let first = OpenAiGenerationIds::new_with_trust(
            OpenAiCacheHints::default(),
            Some("conversation-7"),
            false,
            None,
        );
        let second = OpenAiGenerationIds::new_with_trust(
            OpenAiCacheHints::default(),
            Some("conversation-7"),
            false,
            None,
        );

        assert_eq!(first.agent_session_id.as_deref(), Some("conversation-7"));
        assert_eq!(second.agent_session_id.as_deref(), Some("conversation-7"));
        assert!(!first.agent_session_trusted);
        assert!(!second.agent_session_trusted);
        assert_ne!(first.session_label, second.session_label);
        assert_ne!(first.session_id, second.session_id);
        assert_ne!(first.request_id, second.request_id);
    }

    #[test]
    fn request_ids_differ_for_replica_equivalent_sequences() {
        assert_ne!(
            request_id("openai-agent-session-agent-42", 7, "process-a"),
            request_id("openai-agent-session-agent-42", 7, "process-b")
        );
    }
}

#[derive(Clone, Default)]
pub(in crate::frontend) struct OpenAiCacheHints {
    pub(in crate::frontend) prompt_cache_key: Option<String>,
    pub(in crate::frontend) prompt_cache_retention: Option<String>,
}

impl OpenAiCacheHints {
    pub(in crate::frontend) fn from_chat_request(request: &ChatCompletionRequest) -> Self {
        Self {
            prompt_cache_key: request
                .prompt_cache_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string),
            prompt_cache_retention: request
                .prompt_cache_retention
                .map(prompt_cache_retention_label)
                .map(ToString::to_string),
        }
    }

    pub(in crate::frontend) fn from_completion_request(request: &CompletionRequest) -> Self {
        Self {
            prompt_cache_key: request
                .prompt_cache_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string),
            prompt_cache_retention: request
                .prompt_cache_retention
                .map(prompt_cache_retention_label)
                .map(ToString::to_string),
        }
    }

    pub(in crate::frontend) fn namespace(&self) -> Option<String> {
        self.prompt_cache_key
            .as_ref()
            .map(|key| format!("openai:prompt_cache_key:{key}"))
    }
}

pub(in crate::frontend) fn prompt_cache_retention_label(
    retention: openai_frontend::PromptCacheRetention,
) -> &'static str {
    match retention {
        openai_frontend::PromptCacheRetention::InMemory => "in_memory",
        openai_frontend::PromptCacheRetention::TwentyFourHours => "24h",
    }
}

pub(in crate::frontend) struct GenerationCacheStats {
    pub(in crate::frontend) status: &'static str,
    pub(in crate::frontend) cached_prompt_tokens: u32,
    pub(in crate::frontend) matched_prefix_tokens: u32,
    pub(in crate::frontend) suffix_prefill_tokens: u32,
    pub(in crate::frontend) hit_kind: Option<&'static str>,
    pub(in crate::frontend) native_mtp_stats: NativeMtpStats,
    pub(in crate::frontend) native_mtp_decode_telemetry:
        Option<crate::frontend::native_mtp::NativeMtpDecodeTelemetry>,
    pub(in crate::frontend) verify_window_pipeline_stats:
        Option<crate::frontend::decode_scheduler::VerifyWindowPipelineStats>,
    pub(in crate::frontend) speculative_stats:
        Option<crate::frontend::speculative::OpenAiSpeculativeStats>,
    pub(in crate::frontend) prompt_ms: f64,
    pub(in crate::frontend) predicted_ms: f64,
    pub(in crate::frontend) queue_wait_ms: f64,
    pub(in crate::frontend) restore_ms: f64,
}

impl Default for GenerationCacheStats {
    fn default() -> Self {
        Self {
            status: "disabled",
            cached_prompt_tokens: 0,
            matched_prefix_tokens: 0,
            suffix_prefill_tokens: 0,
            hit_kind: None,
            native_mtp_stats: NativeMtpStats::default(),
            native_mtp_decode_telemetry: None,
            verify_window_pipeline_stats: None,
            speculative_stats: None,
            prompt_ms: 0.0,
            predicted_ms: 0.0,
            queue_wait_ms: 0.0,
            restore_ms: 0.0,
        }
    }
}

pub(in crate::frontend) struct ChainPrefixRestore {
    pub(in crate::frontend) restored_tokens: usize,
    pub(in crate::frontend) stats: StageReplyStats,
}
