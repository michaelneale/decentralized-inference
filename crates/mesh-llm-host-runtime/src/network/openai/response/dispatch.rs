use super::common::{ResponseRetryPolicy, RouteAttemptResult, is_client_disconnect_error};
use super::json_adaptation::{
    relay_normalized_chat_completion_json, relay_translated_responses_json,
};
use super::probe::{ResponseProbe, try_parse_response_headers};
use super::relay::{relay_error_response, relay_success_response};
use super::stream_translation::{
    relay_normalized_chat_completion_stream, relay_translated_responses_stream,
};
use crate::logging::OpenAiRouteObserver;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::request_normalize::ResponseAdapter;
use anyhow::{Result, anyhow};
use mesh_llm_events::logging::identifiers::RequestId;
use tokio::io::AsyncRead;

pub(in crate::network::openai::response) struct RelayAttemptContext<'a> {
    pub(in crate::network::openai::response) request_id: RequestId,
    pub(in crate::network::openai::response) disconnect_message: &'a str,
    pub(in crate::network::openai::response) commit_message: &'a str,
    pub(in crate::network::openai::response) served_by: Option<&'a str>,
    pub(in crate::network::openai::response) route_observer: OpenAiRouteObserver<'a>,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::network::openai::response) async fn relay_probed_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    _request_id: RequestId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<RouteAttemptResult> {
    if let Some(result) = relay_adapted_response(
        tcp_stream,
        reader,
        probe.clone(),
        retry_policy,
        response_adapter,
        served_by,
        route_observer,
    )
    .await?
    {
        return Ok(result);
    }

    if retry_policy.context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }
    if !(200..300).contains(&probe.status_code) {
        return relay_error_response(tcp_stream, reader, probe, route_observer).await;
    }

    let parsed = try_parse_response_headers(&probe.buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    relay_success_response(
        tcp_stream,
        reader,
        probe,
        parsed,
        retry_policy,
        served_by,
        route_observer,
    )
    .await
}

async fn relay_adapted_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<Option<RouteAttemptResult>> {
    match response_adapter {
        ResponseAdapter::OpenAiChatCompletionsJson => Ok(Some(
            relay_normalized_chat_completion_json(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
            )
            .await?,
        )),
        ResponseAdapter::OpenAiChatCompletionsStream => Ok(Some(
            relay_normalized_chat_completion_stream(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
            )
            .await?,
        )),
        ResponseAdapter::OpenAiResponsesJson => Ok(Some(
            relay_translated_responses_json(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
            )
            .await?,
        )),
        ResponseAdapter::OpenAiResponsesStream => Ok(Some(
            relay_translated_responses_stream(
                tcp_stream,
                reader,
                probe,
                retry_policy,
                served_by,
                route_observer,
            )
            .await?,
        )),
        ResponseAdapter::None => Ok(None),
    }
}

pub(in crate::network::openai::response) async fn relay_attempted_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    context: RelayAttemptContext<'_>,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
) -> RouteAttemptResult {
    let status_code = probe.status_code;
    match relay_probed_response(
        tcp_stream,
        reader,
        probe,
        context.request_id,
        retry_policy,
        response_adapter,
        context.served_by,
        context.route_observer,
    )
    .await
    {
        Ok(result) => result,
        Err(err) => {
            if is_streaming_response_adapter(response_adapter) {
                if is_client_disconnect_error(&err) {
                    context.route_observer.stream_cancelled();
                } else {
                    context.route_observer.stream_error("stream_relay_failed");
                }
            }
            if is_client_disconnect_error(&err) {
                tracing::info!("{}", context.disconnect_message);
                return RouteAttemptResult::ClientDisconnected;
            }
            tracing::debug!("{}: {err}", context.commit_message);
            if is_streaming_response_adapter(response_adapter) {
                return RouteAttemptResult::CommittedStreamFailure { status_code: 200 };
            }
            RouteAttemptResult::Delivered {
                status_code,
                usage: None,
                cache_cost: None,
            }
        }
    }
}

const fn is_streaming_response_adapter(response_adapter: ResponseAdapter) -> bool {
    matches!(
        response_adapter,
        ResponseAdapter::OpenAiChatCompletionsStream | ResponseAdapter::OpenAiResponsesStream
    )
}
