use super::cache_cost::CacheCostObservation;
use crate::inference::election;
use crate::logging::OpenAiRouteObserver;
use crate::network::openai::request_normalize::ResponseAdapter;
use crate::network::openai::response_quality::{self, ResponseQualityFailure};
use crate::network::target_health::TargetHealthOutcome;
use mesh_llm_events::logging::events::TokenUsage;
use mesh_llm_events::logging::identifiers::RequestId;

/// Detect an OpenAI-shaped error body carried as an SSE `data:` frame.
///
/// Embedded backends report mid-stream failures this way: HTTP stays 200 and
/// the terminal failure is a `{"error": {...}}` JSON frame. Recognizing it
/// lets stream relays distinguish a failed stream from a completed one in the
/// operations log instead of silently counting a contentless stream as
/// delivered.
pub(in crate::network::openai::response) fn sse_data_frame_is_openai_error(data: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
        return false;
    };
    value
        .get("error")
        .is_some_and(|error| error.is_object() || error.is_string())
}

#[derive(Clone, Copy)]
pub(in crate::network::openai) struct RouteAttemptLoggingContext<'a> {
    pub(in crate::network::openai) request_id: RequestId,
    pub(in crate::network::openai) retry_policy: ResponseRetryPolicy,
    pub(in crate::network::openai) response_adapter: ResponseAdapter,
    pub(in crate::network::openai) route_observer: OpenAiRouteObserver<'a>,
    /// Hex-encoded `EndpointId` to echo back as `x-mesh-served-by` on
    /// delivery. See `RouteModelRequestContext::served_by_header`.
    pub(in crate::network::openai) served_by: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::network::openai) enum RouteAttemptResult {
    Delivered {
        status_code: u16,
        usage: Option<TokenUsage>,
        cache_cost: Option<CacheCostObservation>,
    },
    RetryableTimeout,
    RetryableUnavailable,
    RetryableContextOverflow,
    RetryableResponseQuality(ResponseQualityFailure),
    CommittedStreamFailure {
        status_code: u16,
    },
    ClientDisconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::network::openai) struct ResponseRetryPolicy {
    pub(in crate::network::openai) context_overflow: bool,
    pub(in crate::network::openai) response_quality: bool,
}

impl ResponseRetryPolicy {
    pub(in crate::network::openai) fn next_target_available(available: bool) -> Self {
        Self {
            context_overflow: available,
            response_quality: available,
        }
    }
}

pub(in crate::network::openai) fn route_attempt_result_label(
    result: &RouteAttemptResult,
) -> &'static str {
    match result {
        RouteAttemptResult::Delivered { .. } => "delivered",
        RouteAttemptResult::RetryableTimeout => "retryable_timeout",
        RouteAttemptResult::RetryableUnavailable => "retryable_unavailable",
        RouteAttemptResult::RetryableContextOverflow => "retryable_context_overflow",
        RouteAttemptResult::RetryableResponseQuality(_) => "retryable_response_quality",
        RouteAttemptResult::CommittedStreamFailure { .. } => "committed_stream_failure",
        RouteAttemptResult::ClientDisconnected => "client_disconnected",
    }
}

pub(in crate::network::openai) fn target_health_outcome_for_attempt(
    result: &RouteAttemptResult,
) -> TargetHealthOutcome {
    match result {
        RouteAttemptResult::Delivered { status_code, .. } if (200..300).contains(status_code) => {
            TargetHealthOutcome::Success
        }
        RouteAttemptResult::Delivered { status_code, .. } if (500..600).contains(status_code) => {
            TargetHealthOutcome::Unavailable
        }
        RouteAttemptResult::Delivered { .. } => TargetHealthOutcome::Rejected,
        RouteAttemptResult::RetryableTimeout => TargetHealthOutcome::Timeout,
        RouteAttemptResult::RetryableUnavailable => TargetHealthOutcome::Unavailable,
        RouteAttemptResult::RetryableContextOverflow => TargetHealthOutcome::ContextOverflow,
        RouteAttemptResult::RetryableResponseQuality(_) => TargetHealthOutcome::Rejected,
        RouteAttemptResult::CommittedStreamFailure { .. } => TargetHealthOutcome::Unavailable,
        RouteAttemptResult::ClientDisconnected => TargetHealthOutcome::ClientDisconnected,
    }
}

fn is_disconnect_kind(kind: std::io::ErrorKind) -> bool {
    matches!(
        kind,
        std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::NotConnected
            | std::io::ErrorKind::UnexpectedEof
    )
}

pub(in crate::network::openai::response) fn is_client_disconnect_error(
    err: &anyhow::Error,
) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .map(|io_err| is_disconnect_kind(io_err.kind()))
            .unwrap_or(false)
    })
}

pub(in crate::network::openai) fn is_timeout_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .map(|io_err| io_err.kind() == std::io::ErrorKind::TimedOut)
            .unwrap_or(false)
            || cause.is::<tokio::time::error::Elapsed>()
    })
}

fn response_message_text(json: &serde_json::Value) -> Option<String> {
    fn value_to_text(value: &serde_json::Value) -> Option<String> {
        match value {
            serde_json::Value::String(text) => Some(text.clone()),
            serde_json::Value::Object(map) => map
                .get("message")
                .and_then(value_to_text)
                .or_else(|| map.get("error").and_then(value_to_text)),
            _ => None,
        }
    }

    value_to_text(json)
}

pub(in crate::network::openai::response) fn is_retryable_context_overflow_response(
    body: &[u8],
) -> bool {
    let text = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|json| response_message_text(&json))
        .unwrap_or_else(|| String::from_utf8_lossy(body).to_string())
        .to_ascii_lowercase();

    let mentions_context = [
        "context", "n_ctx", "ctx", "prompt", "token", "slot", "window",
    ]
    .into_iter()
    .any(|needle| text.contains(needle));
    let mentions_limit = [
        "exceed",
        "overflow",
        "too long",
        "too many",
        "greater than",
        "longer than",
        "limit",
        "maximum",
    ]
    .into_iter()
    .any(|needle| text.contains(needle));

    mentions_context && mentions_limit
}

/// Parse only upstream-reported token usage. Chat Completions and Responses
/// use different names for prompt and completion counts; both map into the
/// shared contract without deriving or estimating missing values.
pub(in crate::network::openai) fn parse_token_usage_from_json_body(
    body: &[u8],
) -> Option<TokenUsage> {
    let json = serde_json::from_slice::<serde_json::Value>(body).ok()?;
    let usage = json.get("usage")?;
    let cached_prompt_tokens = usage
        .get("prompt_tokens_details")
        .or_else(|| usage.get("input_tokens_details"))
        .and_then(|details| details.get("cached_tokens"))
        .and_then(serde_json::Value::as_u64);
    TokenUsage::from_counts(
        usage
            .get("prompt_tokens")
            .or_else(|| usage.get("input_tokens"))
            .and_then(serde_json::Value::as_u64),
        usage
            .get("completion_tokens")
            .or_else(|| usage.get("output_tokens"))
            .and_then(serde_json::Value::as_u64),
        usage
            .get("total_tokens")
            .and_then(serde_json::Value::as_u64),
    )
    .map(|usage| usage.with_cached_prompt_tokens(cached_prompt_tokens))
}

pub(in crate::network::openai::response) fn retryable_quality_result(
    body: &[u8],
    policy: ResponseRetryPolicy,
) -> Option<RouteAttemptResult> {
    if !policy.response_quality {
        return None;
    }
    let failure = response_quality::failure_from_json_body(body)?;
    tracing::warn!(
        reason = failure.label(),
        "API proxy: upstream returned retryable low-quality success response before commit"
    );
    Some(RouteAttemptResult::RetryableResponseQuality(failure))
}

fn delivered_attempt_outcome(status_code: u16) -> crate::network::metrics::AttemptOutcome {
    match status_code {
        200..=299 => crate::network::metrics::AttemptOutcome::Success,
        400..=499 => crate::network::metrics::AttemptOutcome::Rejected,
        500..=599 => crate::network::metrics::AttemptOutcome::Unavailable,
        _ => crate::network::metrics::AttemptOutcome::Rejected,
    }
}

pub(in crate::network::openai) fn request_outcome_for_status(
    status_code: u16,
    service: crate::network::metrics::RequestService,
) -> crate::network::metrics::RequestOutcome {
    match status_code {
        200..=299 => crate::network::metrics::RequestOutcome::Success(service),
        _ => crate::network::metrics::RequestOutcome::Rejected(service),
    }
}

pub(in crate::network::openai::response) fn retryable_route_result_from_error(
    err: &anyhow::Error,
) -> RouteAttemptResult {
    if is_timeout_error(err) {
        RouteAttemptResult::RetryableTimeout
    } else {
        RouteAttemptResult::RetryableUnavailable
    }
}

pub(in crate::network::openai) fn attempt_outcome_for_result(
    result: &RouteAttemptResult,
) -> crate::network::metrics::AttemptOutcome {
    match result {
        RouteAttemptResult::Delivered { status_code, .. } => {
            delivered_attempt_outcome(*status_code)
        }
        RouteAttemptResult::RetryableTimeout => crate::network::metrics::AttemptOutcome::Timeout,
        RouteAttemptResult::RetryableUnavailable => {
            crate::network::metrics::AttemptOutcome::Unavailable
        }
        RouteAttemptResult::RetryableContextOverflow => {
            crate::network::metrics::AttemptOutcome::ContextOverflow
        }
        RouteAttemptResult::RetryableResponseQuality(_) => {
            crate::network::metrics::AttemptOutcome::Rejected
        }
        RouteAttemptResult::CommittedStreamFailure { .. } => {
            crate::network::metrics::AttemptOutcome::Unavailable
        }
        RouteAttemptResult::ClientDisconnected => {
            crate::network::metrics::AttemptOutcome::Unavailable
        }
    }
}

pub(in crate::network::openai) fn completion_tokens_for_result(
    result: &RouteAttemptResult,
) -> Option<u64> {
    match result {
        RouteAttemptResult::Delivered { usage, .. } => {
            usage.and_then(|usage| usage.completion_tokens)
        }
        _ => None,
    }
}

pub(in crate::network::openai) fn request_service_for_target(
    target: &election::InferenceTarget,
) -> crate::network::metrics::RequestService {
    match target {
        election::InferenceTarget::Local(_) => crate::network::metrics::RequestService::Local,
        election::InferenceTarget::Remote(_) | election::InferenceTarget::None => {
            crate::network::metrics::RequestService::Remote
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::openai::response_quality::ResponseQualityFailure;
    use std::time::Duration;

    #[test]
    fn test_route_attempt_result_label_values() {
        assert_eq!(
            route_attempt_result_label(&RouteAttemptResult::Delivered {
                status_code: 200,
                usage: None,
                cache_cost: None,
            }),
            "delivered"
        );
        assert_eq!(
            route_attempt_result_label(&RouteAttemptResult::RetryableTimeout),
            "retryable_timeout"
        );
        assert_eq!(
            route_attempt_result_label(&RouteAttemptResult::RetryableUnavailable),
            "retryable_unavailable"
        );
        assert_eq!(
            route_attempt_result_label(&RouteAttemptResult::RetryableContextOverflow),
            "retryable_context_overflow"
        );
        assert_eq!(
            route_attempt_result_label(&RouteAttemptResult::RetryableResponseQuality(
                ResponseQualityFailure::EmptyAssistantOutput
            )),
            "retryable_response_quality"
        );
        assert_eq!(
            route_attempt_result_label(&RouteAttemptResult::ClientDisconnected),
            "client_disconnected"
        );
    }

    #[test]
    fn test_target_health_outcome_for_attempt_values() {
        assert_eq!(
            target_health_outcome_for_attempt(&RouteAttemptResult::Delivered {
                status_code: 200,
                usage: None,
                cache_cost: None,
            }),
            TargetHealthOutcome::Success
        );
        assert_eq!(
            target_health_outcome_for_attempt(&RouteAttemptResult::Delivered {
                status_code: 503,
                usage: None,
                cache_cost: None,
            }),
            TargetHealthOutcome::Unavailable
        );
        assert_eq!(
            target_health_outcome_for_attempt(&RouteAttemptResult::Delivered {
                status_code: 400,
                usage: None,
                cache_cost: None,
            }),
            TargetHealthOutcome::Rejected
        );
        assert_eq!(
            target_health_outcome_for_attempt(&RouteAttemptResult::RetryableContextOverflow),
            TargetHealthOutcome::ContextOverflow
        );
        assert_eq!(
            target_health_outcome_for_attempt(&RouteAttemptResult::RetryableResponseQuality(
                ResponseQualityFailure::LengthFinishReason
            )),
            TargetHealthOutcome::Rejected
        );
        assert_eq!(
            target_health_outcome_for_attempt(&RouteAttemptResult::RetryableTimeout),
            TargetHealthOutcome::Timeout
        );
    }
    #[test]
    fn token_usage_parser_supports_chat_responses_and_absence() {
        let chat = serde_json::json!({
            "usage": {
                "prompt_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens": 3,
                "total_tokens": 8
            }
        });
        let responses = serde_json::json!({
            "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}
        });

        assert_eq!(
            parse_token_usage_from_json_body(chat.to_string().as_bytes()),
            Some(TokenUsage {
                prompt_tokens: Some(5),
                cached_prompt_tokens: Some(4),
                completion_tokens: Some(3),
                total_tokens: Some(8),
            })
        );
        assert_eq!(
            parse_token_usage_from_json_body(responses.to_string().as_bytes()),
            Some(TokenUsage {
                prompt_tokens: Some(5),
                cached_prompt_tokens: None,
                completion_tokens: Some(4),
                total_tokens: Some(9),
            })
        );
        assert_eq!(
            parse_token_usage_from_json_body(br#"{"usage":{"prompt_tokens":"five"}}"#),
            None
        );
        assert_eq!(
            parse_token_usage_from_json_body(
                br#"{"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":7}}"#
            ),
            None
        );
        assert_eq!(parse_token_usage_from_json_body(br#"{"usage":{"prompt_tokens":18446744073709551615,"completion_tokens":1,"total_tokens":0}}"#), None);
    }

    #[tokio::test]
    async fn test_is_timeout_error_accepts_concrete_timeout_types_only() {
        let io_timeout = anyhow::Error::from(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "socket timed out",
        ));
        let elapsed_timeout = anyhow::Error::from(
            tokio::time::timeout(Duration::from_millis(1), std::future::pending::<()>())
                .await
                .unwrap_err(),
        );
        let generic_timeout_text = anyhow::anyhow!("context timeout budget exceeded");

        assert!(is_timeout_error(&io_timeout));
        assert!(is_timeout_error(&elapsed_timeout));
        assert!(!is_timeout_error(&generic_timeout_text));
    }
    #[test]
    fn test_is_retryable_context_overflow_response_detects_llama_style_message() {
        let body = br#"{"error":{"message":"prompt tokens exceed context window (n_ctx=4096)"}}"#;
        assert!(is_retryable_context_overflow_response(body));
        assert!(!is_retryable_context_overflow_response(
            br#"{"error":{"message":"missing required field: messages"}}"#
        ));
    }
}
