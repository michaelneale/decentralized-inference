// Streaming chat-completion request/response handling for benchmark trials.
// Owns the SSE parsing used to measure time-to-first-token (TTFT) and
// decode-only throughput without changing the historical `decode_tok_s`
// definition or its `request_ms` timing.

use std::io::{BufRead, Read};
use std::time::{Duration, Instant};

// Epsilon guarding the decode-only-throughput denominator from a
// vanishingly small (but strictly positive) decode interval. Defined once;
// every computation of `decode_only_tok_s` must go through
// `decode_only_tok_s` below rather than inlining this value.
pub(crate) const DECODE_ONLY_TOK_S_EPSILON_SECS: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TrialChatOutcome {
    pub(crate) completion_tokens: u64,
    pub(crate) ttft_ms: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct StreamingChatOutcome {
    completion_tokens: u64,
    ttft: Option<Duration>,
}

// `completion_tokens / max(total_request_elapsed - ttft, epsilon)`. Null
// (never zero) when `ttft_ms` is unknown or the decode interval
// (`total_elapsed_ms - ttft_ms`) is zero or negative; a genuinely broken
// measurement must not be silently reported as a valid (or zero) rate.
pub(crate) fn decode_only_tok_s(
    completion_tokens: u64,
    total_elapsed_ms: f64,
    ttft_ms: Option<f64>,
) -> Option<f64> {
    let ttft_ms = ttft_ms?;
    let interval_secs = (total_elapsed_ms - ttft_ms) / 1000.0;
    if interval_secs <= 0.0 {
        return None;
    }
    Some(completion_tokens as f64 / interval_secs.max(DECODE_ONLY_TOK_S_EPSILON_SECS))
}

pub(crate) fn send_chat_request(
    client: &reqwest::blocking::Client,
    port: u16,
    prompt: &str,
    max_tokens: u32,
) -> anyhow::Result<TrialChatOutcome> {
    let started = Instant::now();
    let response = client
        .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .send()?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text()?;
        anyhow::bail!("chat completion failed with HTTP {status}: {body}");
    }
    let outcome = parse_streaming_chat_response(response, started)?;
    Ok(TrialChatOutcome {
        completion_tokens: outcome.completion_tokens,
        ttft_ms: outcome.ttft.map(|duration| duration.as_secs_f64() * 1000.0),
    })
}

// Parses an SSE chat-completion stream, measuring time-to-first non-empty
// content delta relative to `started` and returning terminal usage.
// Malformed individual chunks are skipped, not fatal; a stream that ends
// ([DONE] or disconnect/EOF) without ever producing terminal usage is an
// error, matching the historical non-streaming behavior of requiring
// `usage.completion_tokens`.
fn parse_streaming_chat_response<R: Read>(
    reader: R,
    started: Instant,
) -> anyhow::Result<StreamingChatOutcome> {
    let mut buffered = std::io::BufReader::new(reader);
    let mut ttft: Option<Duration> = None;
    let mut completion_tokens: Option<u64> = None;
    let mut line = String::new();
    loop {
        line.clear();
        let bytes_read = buffered.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        let Some(payload) = sse_data_payload(&line) else {
            continue;
        };
        if payload == "[DONE]" {
            break;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if ttft.is_none()
            && let Some(content) = first_choice_delta_content(&value)
            && !content.is_empty()
        {
            ttft = Some(started.elapsed());
        }
        if let Some(tokens) = terminal_usage_completion_tokens(&value) {
            completion_tokens = Some(tokens);
        }
    }
    let completion_tokens = completion_tokens
        .ok_or_else(|| anyhow::anyhow!("chat completion stream ended without terminal usage"))?;
    Ok(StreamingChatOutcome {
        completion_tokens,
        ttft,
    })
}

fn sse_data_payload(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.strip_prefix("data:")?.trim())
}

fn first_choice_delta_content(value: &serde_json::Value) -> Option<&str> {
    value
        .get("choices")?
        .as_array()?
        .first()?
        .get("delta")?
        .get("content")?
        .as_str()
}

fn terminal_usage_completion_tokens(value: &serde_json::Value) -> Option<u64> {
    value.get("usage")?.get("completion_tokens")?.as_u64()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ChunkedReader {
        data: std::io::Cursor<Vec<u8>>,
        chunk_size: usize,
    }

    impl ChunkedReader {
        fn new(data: impl Into<Vec<u8>>, chunk_size: usize) -> Self {
            Self {
                data: std::io::Cursor::new(data.into()),
                chunk_size,
            }
        }
    }

    impl Read for ChunkedReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let cap = buf.len().min(self.chunk_size);
            self.data.read(&mut buf[..cap])
        }
    }

    struct ErroringAfterPrefixReader {
        prefix: std::io::Cursor<Vec<u8>>,
        exhausted: bool,
    }

    impl ErroringAfterPrefixReader {
        fn new(prefix: impl Into<Vec<u8>>) -> Self {
            Self {
                prefix: std::io::Cursor::new(prefix.into()),
                exhausted: false,
            }
        }
    }

    impl Read for ErroringAfterPrefixReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.exhausted {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "simulated read timeout",
                ));
            }
            let read = self.prefix.read(buf)?;
            if read == 0 {
                self.exhausted = true;
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "simulated read timeout",
                ));
            }
            Ok(read)
        }
    }

    fn content_chunk(text: &str) -> String {
        format!(r#"data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}"#)
    }

    fn usage_chunk(completion_tokens: u64) -> String {
        format!(
            r#"data: {{"choices":[],"usage":{{"prompt_tokens":3,"completion_tokens":{completion_tokens},"total_tokens":{completion_tokens}}}}}"#
        )
    }

    #[test]
    fn first_content_sets_ttft() {
        let stream = format!(
            "{}\n\n{}\n\ndata: [DONE]\n\n",
            content_chunk("hello"),
            usage_chunk(2)
        );
        let outcome = parse_streaming_chat_response(
            std::io::Cursor::new(stream.into_bytes()),
            Instant::now(),
        )
        .expect("valid stream parses");
        assert_eq!(outcome.completion_tokens, 2);
        assert!(outcome.ttft.is_some());
    }

    #[test]
    fn usage_is_read_from_terminal_chunk() {
        let stream = format!(
            "{}\n\n{}\n\ndata: [DONE]\n\n",
            content_chunk("hi"),
            usage_chunk(7)
        );
        let outcome = parse_streaming_chat_response(
            std::io::Cursor::new(stream.into_bytes()),
            Instant::now(),
        )
        .expect("valid stream parses");
        assert_eq!(outcome.completion_tokens, 7);
    }

    #[test]
    fn empty_deltas_do_not_count_as_first_content() {
        let stream = concat!(
            r#"data: {"choices":[{"delta":{}}]}"#,
            "\n\n",
            r#"data: {"choices":[{"delta":{"content":""}}]}"#,
            "\n\n",
        )
        .to_string()
            + &usage_chunk(4)
            + "\n\ndata: [DONE]\n\n";
        let outcome = parse_streaming_chat_response(
            std::io::Cursor::new(stream.into_bytes()),
            Instant::now(),
        )
        .expect("stream with only empty deltas still completes");
        assert_eq!(outcome.completion_tokens, 4);
        assert!(outcome.ttft.is_none(), "empty deltas must never set ttft");
    }

    #[test]
    fn done_sentinel_terminates_the_stream() {
        let stream = format!(
            "{}\n\n{}\n\ndata: [DONE]\n\nTRAILING GARBAGE THAT MUST BE IGNORED",
            content_chunk("hi"),
            usage_chunk(1)
        );
        let outcome = parse_streaming_chat_response(
            std::io::Cursor::new(stream.into_bytes()),
            Instant::now(),
        )
        .expect("trailing bytes after [DONE] must not be read");
        assert_eq!(outcome.completion_tokens, 1);
    }

    #[test]
    fn malformed_json_chunk_is_skipped_not_fatal() {
        let stream = format!(
            "data: {{not valid json\n\n{}\n\n{}\n\ndata: [DONE]\n\n",
            content_chunk("hi"),
            usage_chunk(3)
        );
        let outcome = parse_streaming_chat_response(
            std::io::Cursor::new(stream.into_bytes()),
            Instant::now(),
        )
        .expect("one malformed chunk must not fail the whole stream");
        assert_eq!(outcome.completion_tokens, 3);
        assert!(outcome.ttft.is_some());
    }

    #[test]
    fn null_json_payload_does_not_panic() {
        let stream = format!(
            "data: null\n\n{}\n\n{}\n\ndata: [DONE]\n\n",
            content_chunk("hi"),
            usage_chunk(5)
        );
        let outcome = parse_streaming_chat_response(
            std::io::Cursor::new(stream.into_bytes()),
            Instant::now(),
        )
        .expect("a literal JSON null chunk must not panic or fail the stream");
        assert_eq!(outcome.completion_tokens, 5);
    }

    #[test]
    fn split_chunks_reassemble_full_json_events() {
        let stream = format!(
            "{}\n\n{}\n\ndata: [DONE]\n\n",
            content_chunk("hello world"),
            usage_chunk(9)
        );
        let reader = ChunkedReader::new(stream.into_bytes(), 3);
        let outcome = parse_streaming_chat_response(reader, Instant::now())
            .expect("bytes fragmented across many small reads still parse");
        assert_eq!(outcome.completion_tokens, 9);
        assert!(outcome.ttft.is_some());
    }

    #[test]
    fn missing_terminal_usage_is_error_not_zero() {
        let stream = format!("{}\n\ndata: [DONE]\n\n", content_chunk("hi"));
        let error = parse_streaming_chat_response(
            std::io::Cursor::new(stream.into_bytes()),
            Instant::now(),
        )
        .expect_err("a stream with no usage chunk must error, not return 0 tokens");
        assert!(error.to_string().contains("terminal usage"));
    }

    #[test]
    fn disconnect_mid_stream_before_usage_is_error() {
        // EOF (clean disconnect) before `[DONE]`/usage ever arrived.
        let stream = content_chunk("partial");
        let error = parse_streaming_chat_response(
            std::io::Cursor::new(stream.into_bytes()),
            Instant::now(),
        )
        .expect_err("an early disconnect without usage must error, not succeed");
        assert!(error.to_string().contains("terminal usage"));
    }

    #[test]
    fn timeout_before_first_content_is_an_error() {
        let error = parse_streaming_chat_response(
            ErroringAfterPrefixReader::new(Vec::new()),
            Instant::now(),
        )
        .expect_err("a read timeout before any content must surface as an error");
        assert!(error.to_string().to_lowercase().contains("timeout"));
    }

    #[test]
    fn decode_only_tok_s_is_null_when_ttft_is_unknown() {
        assert_eq!(decode_only_tok_s(100, 2000.0, None), None);
    }

    #[test]
    fn decode_only_tok_s_is_null_on_zero_decode_interval_never_zero() {
        assert_eq!(decode_only_tok_s(100, 1000.0, Some(1000.0)), None);
    }

    #[test]
    fn decode_only_tok_s_is_null_on_negative_decode_interval_never_zero() {
        assert_eq!(decode_only_tok_s(100, 1000.0, Some(1200.0)), None);
    }

    #[test]
    fn decode_only_tok_s_clamps_a_tiny_positive_interval_to_epsilon() {
        // interval = 1e-10s, well below DECODE_ONLY_TOK_S_EPSILON_SECS.
        let rate = decode_only_tok_s(100, 1000.0000001, Some(1000.0))
            .expect("a strictly-positive interval, however tiny, must be a real value");
        assert!(rate.is_finite());
        assert_eq!(rate, 100.0 / DECODE_ONLY_TOK_S_EPSILON_SECS);
    }

    #[test]
    fn decode_only_tok_s_computes_the_normal_case() {
        let rate = decode_only_tok_s(150, 2000.0, Some(500.0)).expect("normal case is Some");
        assert!((rate - 100.0).abs() < 1e-9);
    }
}
