use crate::mesh;
use crate::plugin;
use anyhow::{Context, Result, anyhow, bail};
use mesh_llm_events::logging::identifiers::RequestId;
use serde::Deserialize;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use super::request_normalize::{
    ResponseAdapter, normalize_openai_compat_request, resolve_request_object_references,
};
use super::routing_rank::descriptor_for_model;

pub(crate) const MAX_HEADER_BYTES: usize = 64 * 1024;
/// Private lifecycle ownership assertion used only on trusted mesh forwarding.
///
/// This header is removed from every parsed inbound request before the request
/// is forwarded. Raw mesh ingress adds it only after claiming the matching
/// lifecycle parent, so ordinary API clients cannot opt into target-owner
/// suppression by sending it themselves.
pub(crate) const RAW_LIFECYCLE_OWNER_HEADER: &str = "x-mesh-llm-raw-lifecycle";
/// Force remote-mesh dispatch to exactly one peer (fail closed if it doesn't
/// serve the requested model). See `ingress.rs`'s remote-mesh routing.
pub(crate) const MESH_TARGET_HEADER: &str = "x-mesh-target";
/// Remove one or more peers from the remote-mesh candidate set before
/// selection. Comma-separated within one header value.
pub(crate) const MESH_EXCLUDE_HEADER: &str = "x-mesh-exclude";
pub(super) const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_OBJECT_UPLOAD_BODY_BYTES: usize = 64 * 1024 * 1024;
const MAX_CHUNKED_WIRE_BYTES: usize = MAX_BODY_BYTES * 6 + 64 * 1024;
const MAX_OBJECT_UPLOAD_CHUNKED_WIRE_BYTES: usize = MAX_OBJECT_UPLOAD_BODY_BYTES * 6 + 64 * 1024;
pub(super) const MAX_HEADERS: usize = 64;
const CRLF: &[u8] = b"\r\n";
const LF: &[u8] = b"\n";
const CRLF_HEADER_TERMINATOR: &[u8] = b"\r\n\r\n";
const LF_HEADER_TERMINATOR: &[u8] = b"\n\n";

#[derive(Debug, Clone, Copy)]
pub(super) struct HttpReadLimits {
    pub(super) max_header_bytes: usize,
    pub(super) max_body_bytes: usize,
    pub(super) max_chunked_wire_bytes: usize,
}

const HTTP_READ_LIMITS: HttpReadLimits = HttpReadLimits {
    max_header_bytes: MAX_HEADER_BYTES,
    max_body_bytes: MAX_BODY_BYTES,
    max_chunked_wire_bytes: MAX_CHUNKED_WIRE_BYTES,
};

/// Parsed header metadata extracted via httparse.
struct ParsedHeaders {
    header_end: usize,
    method: String,
    path: String,
    request_id: RequestId,
    content_length: Option<usize>,
    is_chunked: bool,
    expects_continue: bool,
    correlation_id: Option<String>,
}

/// The bounded metadata available after HTTP headers parse, even if consuming
/// or normalizing the body later fails. It is deliberately body/header-value
/// free so an error response can be attached to the real ingress lifecycle
/// without retaining raw request data.
#[derive(Clone, Debug)]
pub(crate) struct ParsedOpenAiRequestContext {
    pub(crate) request_id: RequestId,
    pub(crate) client_path: String,
}

/// A request-reader failure with optional safe lifecycle context.
///
/// `context` exists only after complete bounded headers were parsed. Callers
/// must not manufacture a logging request for failures before that boundary.
#[derive(Debug)]
pub(crate) struct OpenAiRequestReadError {
    error: anyhow::Error,
    context: Option<ParsedOpenAiRequestContext>,
}

impl OpenAiRequestReadError {
    fn before_headers(error: anyhow::Error) -> Self {
        Self {
            error,
            context: None,
        }
    }

    fn after_headers(error: anyhow::Error, parsed: &ParsedHeaders) -> Self {
        Self {
            error,
            context: Some(ParsedOpenAiRequestContext {
                request_id: parsed.request_id,
                client_path: parsed.path.clone(),
            }),
        }
    }

    pub(crate) fn context(&self) -> Option<&ParsedOpenAiRequestContext> {
        self.context.as_ref()
    }

    fn into_error(self) -> anyhow::Error {
        self.error
    }
}

impl std::fmt::Display for OpenAiRequestReadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(formatter)
    }
}

#[derive(Debug)]
pub struct BufferedHttpRequest {
    pub raw: Vec<u8>,
    pub method: String,
    pub path: String,
    pub client_path: String,
    /// One canonical UUID selected before any host OpenAI forwarding.
    ///
    /// The raw request is rebuilt with exactly this header, so local, remote,
    /// and plugin routes receive the same metadata without retaining payloads.
    pub request_id: RequestId,
    pub body_json: Option<serde_json::Value>,
    pub(super) body_json_attempted: bool,
    pub(super) body_bytes: Option<Vec<u8>>,
    pub body_len_bytes: usize,
    pub completion_tokens: Option<u32>,
    pub stream: Option<bool>,
    pub model_name: Option<String>,
    pub request_object_request_ids: Vec<String>,
    pub response_adapter: ResponseAdapter,
    pub correlation_id: Option<String>,
}

impl BufferedHttpRequest {
    /// Whether this is the product tokenizer capability route.
    ///
    /// This deliberately requires the exact method and path. Tokenization is
    /// not a generation request and must never inherit chat routing behavior.
    pub fn is_tokenize_request(&self) -> bool {
        is_tokenize_request(&self.method, &self.path)
    }

    pub fn ensure_body_json(&mut self) {
        if self.body_json.is_none() && !self.body_json_attempted {
            self.body_json = self
                .body_bytes
                .as_deref()
                .and_then(|body| serde_json::from_slice(body).ok())
                .or_else(|| parse_json_body_from_http_request(&self.raw));
            self.body_json_attempted = true;
        }
    }

    /// The stabilized capsule client nonce and origin marker carried on the
    /// already-rebuilt `raw` request.
    ///
    /// `finalize_forwarded_request` resolves the nonce once at ingress and
    /// re-stamps `x-capsule-client-nonce` (and, only when this frontend minted
    /// the value, `x-capsule-nonce-origin`) into `raw`. Paths that rebuild the
    /// forwarded request from a parsed body instead of forwarding `raw`
    /// byte-for-byte (the pipeline / MoA strong-model proxy) must read the
    /// stabilized value back out here so the outbound request carries the same
    /// nonce every downstream reader expects, rather than dropping it.
    pub fn capsule_nonce_headers(&self) -> (Option<String>, Option<String>) {
        capsule_nonce_headers_from_raw(&self.raw)
    }

    /// Raw (unparsed) values of the `x-mesh-target` / `x-mesh-exclude` mesh
    /// routing headers, read back off the already-buffered raw request.
    ///
    /// Every occurrence of each header name is returned verbatim, including
    /// duplicates — the router (not this parser) decides whether more than
    /// one `x-mesh-target` value is an error. These headers are opaque to
    /// this layer: no endpoint-id parsing happens here. A header value with
    /// non-UTF-8 bytes is rejected outright rather than silently dropped, so
    /// an attacker can't smuggle a routing decision past invalid bytes.
    pub fn mesh_routing_header_values(&self) -> Result<(Vec<String>, Vec<String>), String> {
        let target = header_values_from_raw(&self.raw, MESH_TARGET_HEADER)
            .map_err(|()| format!("{MESH_TARGET_HEADER} header contains invalid UTF-8"))?;
        let exclude = header_values_from_raw(&self.raw, MESH_EXCLUDE_HEADER)
            .map_err(|()| format!("{MESH_EXCLUDE_HEADER} header contains invalid UTF-8"))?;
        Ok((target, exclude))
    }

    /// The only semantic request media kind trusted by artifact capture.
    ///
    /// This derives from the closed OpenAI ingress route vocabulary and a
    /// successfully parsed JSON body. It intentionally never consults raw
    /// client headers, whose arbitrary values are not a logging contract.
    pub(crate) fn artifact_request_media_kind(&self) -> Option<&'static str> {
        let path = self
            .client_path
            .split('?')
            .next()
            .unwrap_or(&self.client_path);
        matches!(
            path,
            "/v1/chat/completions" | "/v1/completions" | "/v1/responses"
        )
        .then_some(())
        .filter(|()| {
            self.body_bytes
                .as_deref()
                .is_some_and(|body| serde_json::from_slice::<serde_json::Value>(body).is_ok())
        })
        .map(|()| "application/json")
    }

    /// Assert that this request is owned by the raw mesh lifecycle parent.
    ///
    /// The assertion is added after parsing and lifecycle registration, never
    /// copied from client input. It is deliberately kept in the forwarded
    /// bytes so the authenticated target tunnel can avoid creating a second
    /// parent for this one-hop request.
    pub(crate) fn mark_raw_lifecycle_owned(&mut self) {
        let Some(header_end) = self.raw.windows(4).position(|window| window == b"\r\n\r\n") else {
            return;
        };
        if self.raw[..header_end]
            .split(|byte| *byte == b'\r' || *byte == b'\n')
            .any(|line| {
                line.split(|byte| *byte == b':').next().is_some_and(|name| {
                    name.eq_ignore_ascii_case(RAW_LIFECYCLE_OWNER_HEADER.as_bytes())
                })
            })
        {
            return;
        }
        let marker = format!(
            "{RAW_LIFECYCLE_OWNER_HEADER}: {}\r\n",
            self.request_id.as_uuid()
        );
        // Keep the forwarded request within the same bounded header contract
        // as ordinary client input. If there is no room for the assertion,
        // leave it absent so the target safely retains frontend ownership.
        if header_end.saturating_add(4).saturating_add(marker.len()) > MAX_HEADER_BYTES {
            return;
        }
        self.raw
            .splice(header_end + 2..header_end + 2, marker.into_bytes());
    }
}

#[derive(Debug, Default, Deserialize)]
struct RequestMetadata {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
    #[serde(default)]
    n_predict: Option<u32>,
    #[serde(default)]
    expected_identity: Option<RequestExpectedIdentity>,
}

#[derive(Debug, Default, Deserialize)]
struct RequestExpectedIdentity {
    #[serde(default)]
    model_id: Option<String>,
}

struct RequestRewriteOutcome {
    body_json: Option<serde_json::Value>,
    request_object_request_ids: Vec<String>,
    request_path: String,
    response_adapter: ResponseAdapter,
    rewritten_body: Option<Vec<u8>>,
}

// ── Request parsing ──

/// Read and buffer one HTTP request for routing decisions.
///
/// This reads complete headers plus the full request body when body framing is
/// known via `Content-Length` or `Transfer-Encoding: chunked`. The raw request
/// bytes are preserved so the chosen upstream sees the original payload.
pub async fn read_http_request<S>(stream: &mut S) -> Result<BufferedHttpRequest>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    read_http_request_with_limits(stream, HTTP_READ_LIMITS, None).await
}

/// Variant for host ingress boundaries that need to bind locally generated
/// error responses to a safely established request lifecycle.
pub(crate) async fn read_http_request_with_plugin_manager_with_context<S>(
    stream: &mut S,
    plugin_manager: Option<&plugin::PluginManager>,
) -> std::result::Result<BufferedHttpRequest, OpenAiRequestReadError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    read_http_request_with_limits_with_context(stream, HTTP_READ_LIMITS, plugin_manager).await
}

pub(super) async fn read_http_request_with_limits<S>(
    stream: &mut S,
    limits: HttpReadLimits,
    plugin_manager: Option<&plugin::PluginManager>,
) -> Result<BufferedHttpRequest>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    read_http_request_with_limits_with_context(stream, limits, plugin_manager)
        .await
        .map_err(OpenAiRequestReadError::into_error)
}

async fn read_http_request_with_limits_with_context<S>(
    stream: &mut S,
    limits: HttpReadLimits,
    plugin_manager: Option<&plugin::PluginManager>,
) -> std::result::Result<BufferedHttpRequest, OpenAiRequestReadError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut raw = Vec::with_capacity(8192);
    let parsed = read_until_headers_parsed(stream, &mut raw, limits.max_header_bytes)
        .await
        .map_err(OpenAiRequestReadError::before_headers)?;
    let body_limits = body_limits_for_path(&parsed.path, limits);
    let header_end = parsed.header_end;
    let body = read_buffered_request_body(stream, &mut raw, &parsed, header_end, body_limits)
        .await
        .map_err(|error| OpenAiRequestReadError::after_headers(error, &parsed))?;

    let tokenize_request = is_tokenize_request(&parsed.method, &parsed.path);
    let metadata = if body.is_empty() {
        None
    } else if tokenize_request {
        Some(
            serde_json::from_slice::<RequestMetadata>(&body)
                .context("parse /v1/tokenize request metadata")
                .map_err(|error| OpenAiRequestReadError::after_headers(error, &parsed))?,
        )
    } else {
        serde_json::from_slice::<RequestMetadata>(&body).ok()
    };
    let requires_json_transform =
        request_requires_json_transform(&parsed.path, &body, plugin_manager.is_some());
    let rewrite = rewrite_request_body_for_forwarding(
        &parsed.path,
        &body,
        plugin_manager,
        requires_json_transform,
    )
    .await
    .map_err(|error| OpenAiRequestReadError::after_headers(error, &parsed))?;
    let mut response_adapter = rewrite.response_adapter;
    if response_adapter == ResponseAdapter::None
        && parsed.path.split('?').next().unwrap_or(&parsed.path) == "/v1/chat/completions"
    {
        response_adapter = if metadata.as_ref().and_then(|value| value.stream) == Some(true) {
            ResponseAdapter::OpenAiChatCompletionsStream
        } else {
            ResponseAdapter::OpenAiChatCompletionsJson
        };
    }
    let model_name = if tokenize_request {
        Some(
            metadata
                .as_ref()
                .and_then(|value| value.expected_identity.as_ref())
                .and_then(|identity| identity.model_id.as_deref())
                .filter(|model_id| !model_id.is_empty())
                .context("/v1/tokenize requires non-empty expected_identity.model_id")
                .map_err(|error| OpenAiRequestReadError::after_headers(error, &parsed))?
                .to_owned(),
        )
    } else {
        metadata.as_ref().and_then(|value| value.model.clone())
    };
    let completion_tokens = metadata.as_ref().and_then(|value| {
        value
            .max_completion_tokens
            .or(value.max_tokens)
            .or(value.max_output_tokens)
            .or(value.n_predict)
    });
    let raw = finalize_forwarded_request(
        raw,
        header_end,
        parsed.expects_continue,
        Some(&rewrite.request_path),
        rewrite.rewritten_body.as_deref(),
        parsed.request_id,
    )
    .map_err(|error| OpenAiRequestReadError::after_headers(error, &parsed))?;
    let body_len_bytes = body.len();
    let body_bytes = if body.is_empty() { None } else { Some(body) };

    Ok(BufferedHttpRequest {
        raw,
        method: parsed.method,
        client_path: parsed.path,
        path: rewrite.request_path,
        body_json: rewrite.body_json,
        body_json_attempted: requires_json_transform,
        body_bytes,
        body_len_bytes,
        completion_tokens,
        stream: metadata.as_ref().and_then(|value| value.stream),
        model_name,
        request_object_request_ids: rewrite.request_object_request_ids,
        response_adapter,
        request_id: parsed.request_id,
        correlation_id: parsed.correlation_id,
    })
}

async fn read_buffered_request_body<S>(
    stream: &mut S,
    raw: &mut Vec<u8>,
    parsed: &ParsedHeaders,
    header_end: usize,
    body_limits: HttpReadLimits,
) -> Result<Vec<u8>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    if parsed.is_chunked {
        return read_chunked_request_body(stream, raw, parsed, header_end, body_limits).await;
    }
    if let Some(content_length) = parsed.content_length {
        return read_fixed_length_request_body(
            stream,
            raw,
            parsed,
            header_end,
            content_length,
            body_limits,
        )
        .await;
    }
    raw.truncate(header_end);
    Ok(Vec::new())
}

async fn read_chunked_request_body<S>(
    stream: &mut S,
    raw: &mut Vec<u8>,
    parsed: &ParsedHeaders,
    header_end: usize,
    body_limits: HttpReadLimits,
) -> Result<Vec<u8>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut sent_continue = false;
    loop {
        if let Some((consumed, decoded)) =
            try_decode_chunked_body(&raw[header_end..], body_limits.max_body_bytes)?
        {
            raw.truncate(header_end + consumed);
            return Ok(decoded);
        }
        if !sent_continue && parsed.expects_continue {
            stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
            sent_continue = true;
        }
        read_more(stream, raw).await?;
        if raw.len().saturating_sub(header_end) > body_limits.max_chunked_wire_bytes {
            bail!(
                "HTTP chunked wire body exceeds {} bytes",
                body_limits.max_chunked_wire_bytes
            );
        }
    }
}

async fn read_fixed_length_request_body<S>(
    stream: &mut S,
    raw: &mut Vec<u8>,
    parsed: &ParsedHeaders,
    header_end: usize,
    content_length: usize,
    body_limits: HttpReadLimits,
) -> Result<Vec<u8>>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    if content_length > body_limits.max_body_bytes {
        bail!("HTTP body exceeds {} bytes", body_limits.max_body_bytes);
    }
    let body_end = header_end + content_length;
    let mut sent_continue = false;
    while raw.len() < body_end {
        if !sent_continue && parsed.expects_continue && content_length > 0 {
            stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
            sent_continue = true;
        }
        read_more(stream, raw).await?;
    }
    raw.truncate(body_end);
    Ok(raw[header_end..body_end].to_vec())
}

async fn rewrite_request_body_for_forwarding(
    path: &str,
    body: &[u8],
    plugin_manager: Option<&plugin::PluginManager>,
    requires_json_transform: bool,
) -> Result<RequestRewriteOutcome> {
    let mut outcome = RequestRewriteOutcome {
        body_json: None,
        request_object_request_ids: Vec::new(),
        request_path: path.to_string(),
        response_adapter: ResponseAdapter::None,
        rewritten_body: None,
    };
    if !requires_json_transform {
        return Ok(outcome);
    }

    outcome.body_json = serde_json::from_slice(body).ok();
    let Some(body_json) = outcome.body_json.as_mut() else {
        return Ok(outcome);
    };

    let normalization = normalize_openai_compat_request(path, body_json)?;
    let mut changed = normalization.changed;
    if let Some(rewritten_path) = normalization.rewritten_path {
        outcome.request_path = rewritten_path;
    }
    outcome.response_adapter = normalization.response_adapter;
    if let Some(plugin_manager) = plugin_manager {
        let resolved_request_ids =
            resolve_request_object_references(&outcome.request_path, body_json, plugin_manager)
                .await?;
        if !resolved_request_ids.is_empty() {
            outcome.request_object_request_ids = resolved_request_ids;
            changed = true;
        }
    }
    if changed {
        outcome.rewritten_body = Some(
            serde_json::to_vec(body_json)
                .context("serialize normalized OpenAI-compatible request body")?,
        );
    }
    Ok(outcome)
}

fn body_limits_for_path(path: &str, default: HttpReadLimits) -> HttpReadLimits {
    let path_only = path.split('?').next().unwrap_or(path);
    if path_only == "/api/objects" {
        HttpReadLimits {
            max_header_bytes: default.max_header_bytes,
            max_body_bytes: MAX_OBJECT_UPLOAD_BODY_BYTES,
            max_chunked_wire_bytes: MAX_OBJECT_UPLOAD_CHUNKED_WIRE_BYTES,
        }
    } else {
        default
    }
}

fn finalize_forwarded_request(
    mut raw: Vec<u8>,
    header_end: usize,
    strip_expect: bool,
    rewritten_path: Option<&str>,
    rewritten_body: Option<&[u8]>,
    request_id: RequestId,
) -> Result<Vec<u8>> {
    let original_body = raw.split_off(header_end);
    // Re-parse with httparse so we iterate over validated header structs.
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut req = httparse::Request::new(&mut headers_buf);
    let _ = req.parse(&raw).context("re-parse headers for forwarding")?;

    let method = req.method.unwrap_or("GET");
    let path = rewritten_path.unwrap_or_else(|| req.path.unwrap_or("/"));
    let version = req.version.unwrap_or(1);

    // Resolve the capsule client nonce once here, at the single point every
    // forwarded request is rebuilt — so local, remote, plugin-endpoint, and
    // every retry/hop path receives the identical, stabilized value regardless
    // of which downstream code forwards `request.raw`. Mirrors how the
    // canonical `x-request-id` is resolved and re-stamped below.
    let (client_nonce, client_nonce_origin) = client_nonce_from_headers(req.headers);

    let mut rebuilt = format!("{method} {path} HTTP/1.{version}\r\n");

    for header in req.headers.iter() {
        let name = header.name;
        if name.eq_ignore_ascii_case("connection") {
            continue;
        }
        if name.eq_ignore_ascii_case("x-request-id") {
            continue;
        }
        // Strip every inbound capsule nonce header on every ingress path. A
        // caller must never be able to smuggle a forged `x-capsule-nonce-origin`
        // marker (which asserts *this* frontend minted the value) or a duplicate
        // nonce through the raw proxy; both are re-stamped from the resolved,
        // validated value below.
        if name.eq_ignore_ascii_case(openai_frontend::lifecycle::CLIENT_NONCE_HEADER.as_str())
            || name.eq_ignore_ascii_case(
                openai_frontend::lifecycle::CLIENT_NONCE_ORIGIN_HEADER.as_str(),
            )
        {
            continue;
        }
        if name.eq_ignore_ascii_case(RAW_LIFECYCLE_OWNER_HEADER) {
            continue;
        }
        if strip_expect && name.eq_ignore_ascii_case("expect") {
            continue;
        }
        if rewritten_body.is_some()
            && (name.eq_ignore_ascii_case("content-length")
                || name.eq_ignore_ascii_case("transfer-encoding"))
        {
            continue;
        }
        let value = std::str::from_utf8(header.value).unwrap_or("");
        rebuilt.push_str(&format!("{name}: {value}\r\n"));
    }
    if let Some(body) = rewritten_body {
        rebuilt.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    rebuilt.push_str(&format!("x-request-id: {}\r\n", request_id.as_uuid()));
    rebuilt.push_str(&format!(
        "{}: {}\r\n",
        openai_frontend::lifecycle::CLIENT_NONCE_HEADER.as_str(),
        client_nonce,
    ));
    if let Some(origin) = client_nonce_origin {
        rebuilt.push_str(&format!(
            "{}: {origin}\r\n",
            openai_frontend::lifecycle::CLIENT_NONCE_ORIGIN_HEADER.as_str(),
        ));
    }

    // The proxy buffers exactly one request for routing, so force a single-request
    // connection contract upstream instead of reusing the client connection blindly.
    rebuilt.push_str("Connection: close\r\n\r\n");

    let mut forwarded = rebuilt.into_bytes();
    forwarded.extend_from_slice(rewritten_body.unwrap_or(&original_body));
    Ok(forwarded)
}

/// Read from the stream until httparse can fully parse the request headers.
/// Returns parsed metadata; `buf` contains all bytes read so far (headers +
/// any trailing body bytes that arrived in the same read).
async fn read_until_headers_parsed<S>(
    stream: &mut S,
    buf: &mut Vec<u8>,
    max_header_bytes: usize,
) -> Result<ParsedHeaders>
where
    S: AsyncRead + Unpin,
{
    loop {
        let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
        let mut req = httparse::Request::new(&mut headers_buf);
        match req.parse(buf) {
            Ok(httparse::Status::Complete(header_end)) => {
                let method = req.method.unwrap_or("GET").to_string();
                let path = req.path.unwrap_or("/").to_string();

                let mut content_length = None;
                let mut is_chunked = false;
                let mut expects_continue = false;
                let mut correlation_id = None;

                for header in req.headers.iter() {
                    if header.name.eq_ignore_ascii_case("content-length") {
                        let val = std::str::from_utf8(header.value)
                            .context("invalid Content-Length encoding")?;
                        content_length = Some(
                            val.trim()
                                .parse::<usize>()
                                .with_context(|| format!("invalid Content-Length: {val}"))?,
                        );
                    } else if header.name.eq_ignore_ascii_case("transfer-encoding") {
                        let val = std::str::from_utf8(header.value).unwrap_or("");
                        is_chunked = val
                            .split(',')
                            .any(|part| part.trim().eq_ignore_ascii_case("chunked"));
                    } else if header.name.eq_ignore_ascii_case("expect") {
                        let val = std::str::from_utf8(header.value).unwrap_or("");
                        expects_continue = val
                            .split(',')
                            .any(|part| part.trim().eq_ignore_ascii_case("100-continue"));
                    } else if header.name.eq_ignore_ascii_case("x-correlation-id")
                        || header.name.eq_ignore_ascii_case("x-request-id")
                        || header.name.eq_ignore_ascii_case("correlation-id")
                    {
                        correlation_id =
                            Some(std::str::from_utf8(header.value).unwrap_or("").to_string());
                    }
                }

                // RFC 7230 §3.3.3: if both Transfer-Encoding and Content-Length
                // are present, Transfer-Encoding wins and Content-Length is ignored.
                if is_chunked {
                    content_length = None;
                }

                return Ok(ParsedHeaders {
                    header_end,
                    method,
                    path,
                    request_id: request_id_from_headers(req.headers),
                    content_length,
                    is_chunked,
                    expects_continue,
                    correlation_id,
                });
            }
            Ok(httparse::Status::Partial) => {
                if buf.len() >= max_header_bytes {
                    bail!("HTTP headers exceed {max_header_bytes} bytes");
                }
                read_more(stream, buf).await?;
            }
            Err(e) => bail!("HTTP parse error: {e}"),
        }
    }
}

/// Parse the canonical request ID from a complete, bounded HTTP header prefix.
///
/// This never generates an identifier: tunnel ingress must fail open when the
/// trusted forwarded header is absent, malformed, or duplicated.
pub(crate) fn canonical_request_id_from_header_prefix(prefix: &[u8]) -> Option<RequestId> {
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut request = httparse::Request::new(&mut headers_buf);
    match request.parse(prefix) {
        Ok(httparse::Status::Complete(_)) => canonical_request_id_from_headers(request.headers),
        Ok(httparse::Status::Partial) | Err(_) => None,
    }
}

pub(crate) fn http_header_terminator(prefix: &[u8]) -> Option<(usize, &'static [u8])> {
    let crlf = prefix
        .windows(CRLF_HEADER_TERMINATOR.len())
        .position(|window| window == CRLF_HEADER_TERMINATOR)
        .map(|offset| (offset + CRLF_HEADER_TERMINATOR.len(), CRLF));
    let lf = prefix
        .windows(LF_HEADER_TERMINATOR.len())
        .position(|window| window == LF_HEADER_TERMINATOR)
        .map(|offset| (offset + LF_HEADER_TERMINATOR.len(), LF));

    match (crlf, lf) {
        (Some(crlf), Some(lf)) => Some(if crlf.0 <= lf.0 { crlf } else { lf }),
        (Some(terminator), None) | (None, Some(terminator)) => Some(terminator),
        (None, None) => None,
    }
}

pub(crate) fn ensure_canonical_request_id_in_header_prefix(
    mut prefix: Vec<u8>,
) -> (Vec<u8>, Option<RequestId>) {
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut request = httparse::Request::new(&mut headers_buf);
    let Ok(httparse::Status::Complete(header_end)) = request.parse(&prefix) else {
        return (prefix, None);
    };
    if header_end > MAX_HEADER_BYTES {
        return (prefix, None);
    }

    let request_id_header_count = request
        .headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case("x-request-id"))
        .count();
    if let Some(request_id) = canonical_request_id_from_headers(request.headers) {
        return (prefix, Some(request_id));
    }
    if request_id_header_count != 0 || request.headers.len() >= MAX_HEADERS {
        return (prefix, None);
    }

    let Some((terminator_end, line_ending)) = http_header_terminator(&prefix[..header_end]) else {
        return (prefix, None);
    };
    if terminator_end != header_end {
        return (prefix, None);
    }

    let request_id = RequestId::new();
    let mut header = format!("x-request-id: {}", request_id.as_uuid()).into_bytes();
    header.extend_from_slice(line_ending);
    if header_end.saturating_add(header.len()) > MAX_HEADER_BYTES {
        return (prefix, None);
    }
    let insertion_offset = header_end - line_ending.len();
    prefix.splice(insertion_offset..insertion_offset, header);
    (prefix, Some(request_id))
}

/// Parse the private raw-lifecycle assertion from a complete, bounded HTTP
/// header prefix. The marker is accepted only once and only when its UUID
/// exactly matches the one canonical `x-request-id` header.
pub(crate) fn raw_lifecycle_owner_from_header_prefix(prefix: &[u8]) -> Option<RequestId> {
    let parsed = canonical_request_id_from_header_prefix(prefix)?;
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut request = httparse::Request::new(&mut headers_buf);
    let httparse::Status::Complete(_) = request.parse(prefix).ok()? else {
        return None;
    };
    let mut markers = request
        .headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case(RAW_LIFECYCLE_OWNER_HEADER));
    let marker = markers.next()?;
    if markers.next().is_some() {
        return None;
    }
    let marker_id = std::str::from_utf8(marker.value)
        .ok()
        .and_then(openai_frontend::parse_request_id)?;
    (marker_id == parsed).then_some(parsed)
}

fn request_id_from_headers(headers: &[httparse::Header<'_>]) -> RequestId {
    canonical_request_id_from_headers(headers).unwrap_or_default()
}

fn canonical_request_id_from_headers(headers: &[httparse::Header<'_>]) -> Option<RequestId> {
    let request_id_values = headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case("x-request-id"))
        .map(|header| std::str::from_utf8(header.value).ok());
    openai_frontend::parse_single_request_id(request_id_values)
}

/// Resolve the capsule client nonce for a forwarded request, using the same
/// single-valid-UUIDv4 acceptance rule as the axum frontend ingress.
///
/// Returns the nonce value to stamp and, only when this ingress minted it, the
/// trusted origin marker to stamp alongside it. A forwarded (client-supplied)
/// nonce is returned with no origin marker, so a caller can never make a value
/// it chose look as though this frontend minted it.
/// Read the stabilized `x-capsule-client-nonce` / `x-capsule-nonce-origin`
/// headers back out of an already-rebuilt raw HTTP request.
///
/// Only the request-header block is scanned. Used by request-rebuilding proxy
/// paths (pipeline / MoA) that would otherwise emit an outbound request with no
/// nonce because they construct it from a parsed JSON body rather than
/// forwarding `raw`.
fn capsule_nonce_headers_from_raw(raw: &[u8]) -> (Option<String>, Option<String>) {
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap_or(raw.len());
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut req = httparse::Request::new(&mut headers_buf);
    if req
        .parse(&raw[..header_end.saturating_add(4).min(raw.len())])
        .is_err()
    {
        return (None, None);
    }
    let nonce_header = openai_frontend::lifecycle::CLIENT_NONCE_HEADER.as_str();
    let origin_header = openai_frontend::lifecycle::CLIENT_NONCE_ORIGIN_HEADER.as_str();
    let find = |name: &str| {
        req.headers
            .iter()
            .find(|header| header.name.eq_ignore_ascii_case(name))
            .and_then(|header| std::str::from_utf8(header.value).ok())
            .map(str::to_string)
    };
    (find(nonce_header), find(origin_header))
}

/// Every value of a given header name, read back off an already-rebuilt raw
/// HTTP request. Only the request-header block is scanned. Order matches the
/// wire order; duplicates are returned as separate entries. `Err(())` means
/// at least one occurrence of `name` had non-UTF-8 bytes -- the caller must
/// reject the request rather than silently drop that occurrence.
fn header_values_from_raw(raw: &[u8], name: &str) -> Result<Vec<String>, ()> {
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map_or(raw.len(), |pos| pos);
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut req = httparse::Request::new(&mut headers_buf);
    if req
        .parse(&raw[..header_end.saturating_add(4).min(raw.len())])
        .is_err()
    {
        return Ok(Vec::new());
    }
    req.headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case(name))
        .map(|header| {
            std::str::from_utf8(header.value)
                .map(|value| value.trim().to_string())
                .map_err(|_| ())
        })
        .collect()
}

fn client_nonce_from_headers(headers: &[httparse::Header<'_>]) -> (String, Option<&'static str>) {
    let nonce_header = openai_frontend::lifecycle::CLIENT_NONCE_HEADER.as_str();
    let inbound = headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case(nonce_header))
        .map(|header| std::str::from_utf8(header.value).ok());
    match openai_frontend::parse_single_client_nonce(inbound) {
        Some(value) => (
            value
                .to_str()
                .expect("a parsed UUIDv4 nonce is always ASCII")
                .to_string(),
            None,
        ),
        None => (
            openai_frontend::lifecycle::generate_client_nonce()
                .to_str()
                .expect("a minted UUIDv4 nonce is always ASCII")
                .to_string(),
            Some(openai_frontend::lifecycle::CLIENT_NONCE_ORIGIN_FRONTEND),
        ),
    }
}

async fn read_more<S: AsyncRead + Unpin>(stream: &mut S, buf: &mut Vec<u8>) -> Result<()> {
    let mut chunk = [0u8; 8192];
    let n = stream.read(&mut chunk).await?;
    if n == 0 {
        bail!("unexpected EOF while reading HTTP request");
    }
    buf.extend_from_slice(&chunk[..n]);
    Ok(())
}

fn try_decode_chunked_body(buf: &[u8], max_body_bytes: usize) -> Result<Option<(usize, Vec<u8>)>> {
    let mut pos = 0usize;
    let mut decoded = Vec::new();

    loop {
        let Some(line_end_rel) = buf[pos..].windows(2).position(|window| window == b"\r\n") else {
            return Ok(None);
        };
        let line_end = pos + line_end_rel;
        let size_line = std::str::from_utf8(&buf[pos..line_end]).context("invalid chunk header")?;
        let size_text = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_text, 16)
            .with_context(|| format!("invalid chunk size: {size_text}"))?;
        pos = line_end + 2;

        if size == 0 {
            if buf.len() < pos + 2 {
                return Ok(None);
            }
            if &buf[pos..pos + 2] == b"\r\n" {
                return Ok(Some((pos + 2, decoded)));
            }
            let Some(trailer_end_rel) = buf[pos..]
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
            else {
                return Ok(None);
            };
            return Ok(Some((pos + trailer_end_rel + 4, decoded)));
        }

        if buf.len() < pos + size + 2 {
            return Ok(None);
        }
        decoded.extend_from_slice(&buf[pos..pos + size]);
        pos += size;

        if &buf[pos..pos + 2] != b"\r\n" {
            return Err(anyhow!("invalid chunk terminator"));
        }
        pos += 2;

        if decoded.len() > max_body_bytes {
            bail!("HTTP chunked body exceeds {max_body_bytes} bytes");
        }
    }
}

fn request_requires_json_transform(path: &str, body: &[u8], plugin_manager_present: bool) -> bool {
    openai_frontend::request_body_requires_json_normalization(path, body)
        || (plugin_manager_present
            && path.split('?').next().unwrap_or(path) == "/v1/chat/completions"
            && std::str::from_utf8(body).ok().is_some_and(|body_text| {
                body_text.contains("mesh://blob/")
                    || body_text.contains("\"blob_token\"")
                    || body_text.contains("\"mesh_token\"")
                    || body_text.contains("\"input_audio\"")
                    || body_text.contains("\"input_image\"")
            }))
}

pub(super) fn parse_json_body_from_http_request(raw: &[u8]) -> Option<serde_json::Value> {
    let header_end = raw.windows(4).position(|window| window == b"\r\n\r\n")? + 4;
    serde_json::from_slice(&raw[header_end..]).ok()
}

/// Inject `"mesh_hooks": true/false` into the JSON body of an HTTP request.
///
/// Inserts the field right after the opening `{` in the body, then rebuilds
/// the Content-Length header to match.
pub fn inject_mesh_hooks_flag(raw: &mut Vec<u8>, enabled: bool) {
    let Some(header_end) = raw.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4) else {
        return;
    };
    let body = &raw[header_end..];
    let Some(brace) = body.iter().position(|&b| b == b'{') else {
        return;
    };

    // Build new body with mesh_hooks injected after opening brace
    let fragment = if enabled {
        &b"\"mesh_hooks\":true,"[..]
    } else {
        &b"\"mesh_hooks\":false,"[..]
    };
    let mut new_body = Vec::with_capacity(body.len() + fragment.len());
    new_body.extend_from_slice(&body[..brace + 1]);
    new_body.extend_from_slice(fragment);
    new_body.extend_from_slice(&body[brace + 1..]);

    // Rebuild headers with correct Content-Length
    let headers = std::str::from_utf8(&raw[..header_end - 4]).unwrap_or("");
    let mut rebuilt = String::new();
    for line in headers.split("\r\n") {
        if line.to_ascii_lowercase().starts_with("content-length:") {
            rebuilt.push_str(&format!("Content-Length: {}", new_body.len()));
        } else {
            rebuilt.push_str(line);
        }
        rebuilt.push_str("\r\n");
    }
    rebuilt.push_str("\r\n");

    let mut result = rebuilt.into_bytes();
    result.extend_from_slice(&new_body);
    *raw = result;
}

/// Rewrite the JSON body `model` field and rebuild Content-Length.
pub fn rewrite_model_field(request: &mut BufferedHttpRequest, model: &str) {
    let Some(header_end) = request
        .raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)
    else {
        return;
    };

    let Ok(mut body) = serde_json::from_slice::<serde_json::Value>(&request.raw[header_end..])
    else {
        return;
    };
    let Some(object) = body.as_object_mut() else {
        return;
    };

    object.insert(
        "model".to_string(),
        serde_json::Value::String(model.to_string()),
    );
    let Ok(new_body) = serde_json::to_vec(&body) else {
        return;
    };

    let headers = std::str::from_utf8(&request.raw[..header_end - 4]).unwrap_or("");
    let mut rebuilt = String::new();
    for line in headers.split("\r\n") {
        if line.to_ascii_lowercase().starts_with("content-length:") {
            rebuilt.push_str(&format!("Content-Length: {}", new_body.len()));
        } else {
            rebuilt.push_str(line);
        }
        rebuilt.push_str("\r\n");
    }
    rebuilt.push_str("\r\n");

    let mut raw = rebuilt.into_bytes();
    raw.extend_from_slice(&new_body);

    request.raw = raw;
    request.body_len_bytes = new_body.len();
    request.body_bytes = Some(new_body);
    request.body_json = Some(body);
    request.body_json_attempted = true;
    request.model_name = Some(model.to_string());
}

pub fn is_models_list_request(method: &str, path: &str) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    method == "GET" && (path == "/v1/models" || path == "/models")
}

/// Legacy lifecycle paths formerly handled by the peer-reachable OpenAI
/// ingress. They remain recognizable only so callers can return an explicit
/// compatibility response instead of routing them as inference.
pub fn is_legacy_lifecycle_path(path: &str) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    matches!(path, "/mesh/load" | "/mesh/drop")
}

fn is_tokenize_request(method: &str, path: &str) -> bool {
    method == "POST" && path == "/v1/tokenize"
}

pub fn pipeline_request_supported(path: &str, body: &serde_json::Value) -> bool {
    let path = path.split('?').next().unwrap_or(path);
    path == "/v1/chat/completions"
        && body
            .get("messages")
            .map(|messages| messages.is_array())
            .unwrap_or(false)
}

pub fn rewrite_public_model_alias(
    request: &mut BufferedHttpRequest,
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
) {
    let Some(requested) = request.model_name.as_deref() else {
        return;
    };
    if request.is_tokenize_request() {
        // Tokenizer identity is authoritative end to end. Rewriting only the
        // routing key would send a different expected identity to the target
        // and correctly fail its authority check. Require an exact served
        // identity instead.
        return;
    }
    if requested == "auto" || models.iter().any(|model| model == requested) {
        return;
    }
    let Some(internal) = internal_model_for_public_id(requested, models, descriptors) else {
        return;
    };
    rewrite_model_field(request, &internal);
}

fn internal_model_for_public_id(
    requested: &str,
    models: &[String],
    descriptors: &[mesh::ServedModelDescriptor],
) -> Option<String> {
    let (requested_base, requested_profile) =
        crate::network::openai::ingress::parse_model_with_profile(requested);

    models.iter().find_map(|model| {
        let (model_base, model_profile) =
            crate::network::openai::ingress::parse_model_with_profile(model);
        let descriptor = descriptor_for_model(descriptors, model_base);
        let public_id = public_model_id(model_base, descriptor, model_profile);
        if public_id == requested {
            return Some(model.clone());
        }
        let (public_base, _public_profile) =
            crate::network::openai::ingress::parse_model_with_profile(&public_id);
        if public_base == requested_base && requested_profile.is_empty() {
            return Some(model.clone());
        }
        None
    })
}

pub(super) fn public_model_id(
    model_name: &str,
    descriptor: Option<&mesh::ServedModelDescriptor>,
    profile: &str,
) -> String {
    // A descriptor with an `artifact` field has enough information to
    // produce a public ID that round-trips to the same model. Without
    // it, the HuggingFace path collapses to just the repo name and
    // silently drops the quant-tag suffix the resolver needs (PR #566
    // review feedback — "some IDs in /v1/models dropped quant
    // suffixes"). Only use the descriptor-derived id when it can be
    // lossless; otherwise prefer the on-disk file (authoritative for
    // local models), and finally the internal model_name (which
    // always carries the quant suffix our resolver knows how to
    // route).
    let base_id = if let Some(descriptor) = descriptor
        && descriptor_can_produce_lossless_id(&descriptor.identity)
        && let Some(id) = public_model_id_from_identity(&descriptor.identity)
    {
        id
    } else if let Some(id) = public_model_id_from_local_path(model_name) {
        id
    } else {
        model_name.to_string()
    };

    // Append profile suffix for non-default profiles
    if profile.is_empty() {
        base_id
    } else {
        format!("{}#{}", base_id, profile)
    }
}

/// A descriptor identity carries enough information for
/// `public_model_id_from_identity` to produce an ID that round-trips
/// to the same model. For HuggingFace that means the `artifact` field
/// (the GGUF file name) is present so the quant selector can be
/// derived. Catalog identities always carry a `canonical_ref` with the
/// selector baked in.
fn descriptor_can_produce_lossless_id(identity: &mesh::ServedModelIdentity) -> bool {
    match identity.source_kind {
        mesh::ModelSourceKind::HuggingFace => identity.artifact.is_some(),
        mesh::ModelSourceKind::Catalog => identity.canonical_ref.is_some(),
        mesh::ModelSourceKind::LocalGguf
        | mesh::ModelSourceKind::DirectUrl
        | mesh::ModelSourceKind::Unknown => false,
    }
}

fn public_model_id_from_identity(identity: &mesh::ServedModelIdentity) -> Option<String> {
    match identity.source_kind {
        mesh::ModelSourceKind::HuggingFace => identity
            .repository
            .as_deref()
            .and_then(|repo| public_huggingface_model_ref(repo, identity.artifact.as_deref()))
            .or_else(|| {
                identity
                    .canonical_ref
                    .as_deref()
                    .and_then(|model_ref| model_ref::ModelRef::parse(model_ref).ok())
                    .map(|model_ref| model_ref.display_id())
            }),
        mesh::ModelSourceKind::Catalog => identity
            .canonical_ref
            .as_deref()
            .and_then(|model_ref| model_ref::ModelRef::parse(model_ref).ok())
            .map(|model_ref| model_ref.display_id()),
        mesh::ModelSourceKind::LocalGguf
        | mesh::ModelSourceKind::DirectUrl
        | mesh::ModelSourceKind::Unknown => None,
    }
}

fn public_model_id_from_local_path(model_name: &str) -> Option<String> {
    let path = crate::models::find_model_path(model_name);
    if !path.is_file() {
        return None;
    }
    if path.extension().and_then(|extension| extension.to_str()) != Some("gguf") {
        return None;
    }
    Some(crate::models::model_ref_for_path(&path))
}

fn public_huggingface_model_ref(repo: &str, artifact: Option<&str>) -> Option<String> {
    // `artifact` can be either a GGUF filename (e.g. `Falcon-Q4_K_M.gguf`)
    // or an already-extracted quant selector (e.g. `Q4_K_M` or
    // `qwen2.5-3b-instruct-q4_k_m`, when the descriptor was built from
    // a parsed `ModelRef::selector`). Handle both — if the artifact
    // looks like a quant selector use it directly; otherwise try to
    // pull a selector out of the filename.
    let selector = artifact.and_then(|a| {
        model_ref::quant_selector_from_gguf_file(a)
            .or_else(|| (!a.is_empty() && !a.ends_with(".gguf")).then(|| a.to_string()))
    });
    Some(model_ref::format_model_ref(repo, None, selector.as_deref()))
}

#[cfg(test)]
#[path = "request_parse_tests.rs"]
mod tests;
