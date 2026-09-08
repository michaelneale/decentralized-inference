//! Forwarded request construction for the proxy path.
//!
//! This module owns rebuilding an HTTP/1.x request before forwarding it to a
//! peer or local inference target: parsing the Connection header to discover
//! nominated hop-by-hop headers, rejecting framing-nominated Connection options
//! per RFC 7230 §6.1, stripping omitted headers, and (for peer forwards) the
//! caller credential headers.
//!
//! Behavior here is byte-faithful: header field values are preserved as raw
//! octets, never coerced through UTF-8.

use anyhow::{Context, Result, bail};

use super::request_parse::MAX_HEADERS;
use super::response::is_valid_header_name;

/// Parse every `Connection` header on the request and return its nominated
/// hop-by-hop header names.
///
/// Tokens are validated against HTTP/1.x token syntax (RFC 7230 §3.2.6
/// `tchar`): every character must be in the complete tchar set, so
/// separators, control characters, whitespace, and non-ASCII bytes all reject
/// the request. Any malformed value rejects the entire request. Tokens are
/// returned verbatim; callers compare case-insensitively via
/// `eq_ignore_ascii_case` so neither this helper nor its caller pays for a
/// per-token `to_lowercase()` allocation.
fn collect_connection_nominated_tokens<'a>(
    headers: &'a [httparse::Header<'a>],
) -> Result<Vec<&'a str>> {
    let mut nominated = Vec::new();
    for h in headers {
        if !h.name.eq_ignore_ascii_case("connection") {
            continue;
        }

        // Connection values must be valid HTTP/1.x token syntax — ASCII only.
        // Reject the entire request when invalid bytes or malformed tokens appear.
        let value = std::str::from_utf8(h.value)
            .context("Connection header contains non-UTF-8 bytes; reject as malformed")?;

        for tok in value.split(',') {
            // RFC 7230 §3.2.3: OWS is SP and HTAB only. Trimming just those
            // means non-ASCII whitespace (e.g. NBSP) is never silently
            // swallowed at a token edge — it fails the tchar check below.
            let t = tok.trim_matches(|c| c == ' ' || c == '\t');
            if t.is_empty() {
                continue;
            }
            // Every character must be in the complete HTTP tchar set
            // (RFC 7230 §3.2.6). Reuse the canonical tchar predicate so this
            // parser and the response writers share one definition; it
            // rejects separators (';', ':', ...), control characters, space,
            // and any non-ASCII byte.
            if !is_valid_header_name(t) {
                bail!(
                    "Connection header contains malformed token '{}'; reject as malformed",
                    value
                );
            }
            nominated.push(t);
        }
    }
    Ok(nominated)
}

/// Reject framing-related hop-by-hop fields nominated via Connection.
///
/// Per RFC 7230 §6.1, `content-length` and `transfer-encoding` MUST NOT be
/// used as Connection options because they are framing-related hop-by-hop
/// fields with special semantics. Reject malformed requests before removing
/// headers or forwarding the wire body.
fn reject_framing_nominated(nominated: &[&str]) -> Result<()> {
    if nominated
        .iter()
        .any(|n| n.eq_ignore_ascii_case("content-length"))
    {
        bail!(
            "Connection header nominates 'Content-Length'; reject as protocol violation (RFC 7230 §6.1)"
        );
    }

    if nominated
        .iter()
        .any(|n| n.eq_ignore_ascii_case("transfer-encoding"))
    {
        bail!(
            "Connection header nominates 'Transfer-Encoding'; reject as protocol violation (RFC 7230 §6.1)"
        );
    }

    Ok(())
}

pub(super) fn finalize_forwarded_request(
    raw: &[u8],
    strip_expect: bool,
    rewritten_path: Option<&str>,
    rewritten_body: Option<&[u8]>,
    omitted_headers: &[&str],
) -> Result<Vec<u8>> {
    // Re-parse with httparse so we iterate over validated header structs.
    let mut headers_buf = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut req = httparse::Request::new(&mut headers_buf);
    let header_end = match req.parse(raw).context("re-parse headers for forwarding")? {
        httparse::Status::Complete(header_end) => header_end,
        httparse::Status::Partial => bail!("incomplete HTTP headers for forwarding"),
    };

    let method = req.method.unwrap_or("GET");
    let path = rewritten_path.unwrap_or_else(|| req.path.unwrap_or("/"));
    let version = req.version.unwrap_or(1);

    let mut rebuilt = Vec::new();
    rebuilt.extend_from_slice(format!("{method} {path} HTTP/1.{version}\r\n").as_bytes());

    // RFC 7230 §6.1: parse Connection header to find nominated hop-by-hop
    // headers, and reject framing-related nominations before forwarding.
    let connection_nominated = collect_connection_nominated_tokens(req.headers)?;
    reject_framing_nominated(&connection_nominated)?;

    for header in req.headers.iter() {
        let name = header.name;
        // Compare case-insensitively against each ASCII header name without per-header allocation.
        let is_connection_nominated = connection_nominated
            .iter()
            .any(|n| n.eq_ignore_ascii_case(name));
        if name.eq_ignore_ascii_case("connection")
            || is_connection_nominated
            || omitted_headers
                .iter()
                .any(|omitted| name.eq_ignore_ascii_case(omitted))
        {
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
        // Preserve header values as raw bytes — HTTP/1.x field-values are arbitrary
        // octets, not UTF-8 strings. Write name (ASCII) + ": " separator + raw
        // value, then terminate with CRLF.
        rebuilt.extend_from_slice(name.as_ref());
        rebuilt.push(b':');
        rebuilt.push(b' ');
        rebuilt.extend_from_slice(header.value);
        rebuilt.extend_from_slice(b"\r\n");
    }
    if let Some(body) = rewritten_body {
        rebuilt.extend_from_slice(format!("Content-Length: {}\r\n", body.len()).as_bytes());
    }

    // The proxy buffers exactly one request for routing, so force a single-request connection contract upstream instead of reusing the client connection blindly.
    rebuilt.extend_from_slice(b"Connection: close\r\n\r\n");

    let mut forwarded = rebuilt;
    forwarded.extend_from_slice(rewritten_body.unwrap_or(&raw[header_end..]));
    Ok(forwarded)
}

/// Rebuild a request for a remote peer without forwarding ingress credentials
/// or the routing headers the ROUTING node already consumed.
///
/// `x-mesh-target` / `x-mesh-exclude` express the client's routing intent to
/// the node the client is talking to; they must never survive onto a
/// peer-to-peer hop. Without this, a peer that re-enters `route_request`
/// still carrying the original target/exclude can re-route the request
/// itself -- and since nothing here carries a hop count or visited-peer
/// list, a stale advertisement can bounce a request between peers with no
/// bound (CodeRabbit / erlich, PR #1671 round 2).
pub(super) fn prepare_peer_forwarded_request(raw: &[u8]) -> Result<Vec<u8>> {
    const OMITTED_ON_PEER_FORWARD: &[&str] = &[
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        super::request_parse::MESH_TARGET_HEADER,
        super::request_parse::MESH_EXCLUDE_HEADER,
    ];
    finalize_forwarded_request(raw, false, None, None, OMITTED_ON_PEER_FORWARD)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_forwarding_strips_credentials_and_preserves_headers_and_body() {
        let body = br#"{"model":"test","input":"Authorization: keep in body"}"#;
        let mut raw = format!(
            "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\naUtHoRiZaTiOn: Bearer caller-secret\r\nX-Trace-Id: trace-123\r\nPROXY-AUTHORIZATION: Basic proxy-secret\r\nX-API-Key: anthropic-secret\r\napi-key: azure-secret\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();
        raw.extend_from_slice(body);

        let forwarded = prepare_peer_forwarded_request(&raw).unwrap();

        let mut expected = format!(
            "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nX-Trace-Id: trace-123\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        expected.extend_from_slice(body);
        assert_eq!(forwarded, expected);
    }

    /// Regression (CodeRabbit / erlich, PR #1671 round 2): routing headers
    /// must never survive a peer-to-peer hop, or a peer that re-enters
    /// `route_request` can re-route the request again with no hop bound.
    #[test]
    fn peer_forwarding_strips_mesh_routing_headers() {
        let body = b"{}";
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nx-mesh-target: aabbcc\r\nx-mesh-exclude: ddeeff,001122\r\nX-Keep: yes\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let mut raw = raw.into_bytes();
        raw.extend_from_slice(body);

        let forwarded = prepare_peer_forwarded_request(&raw).unwrap();
        let text = String::from_utf8_lossy(&forwarded);

        assert!(
            !text.to_ascii_lowercase().contains("x-mesh-target"),
            "x-mesh-target must not reach the peer: {text}"
        );
        assert!(
            !text.to_ascii_lowercase().contains("x-mesh-exclude"),
            "x-mesh-exclude must not reach the peer: {text}"
        );
        assert!(
            text.contains("X-Keep: yes"),
            "unrelated headers must survive: {text}"
        );
    }

    #[test]
    fn peer_forwarding_rejects_incomplete_headers() {
        let raw = b"GET /v1/models HTTP/1.1\r\nAuthorization: Bearer caller-secret\r\n";

        assert!(prepare_peer_forwarded_request(raw).is_err());
    }

    #[test]
    fn finalize_strips_connection_nominated_headers() {
        let raw = b"GET /v1/chat/completions HTTP/1.1\r\nHost: localhost:9337\r\nConnection: TE, Upgrade\r\nTE: trailers\r\nUpgrade: h2c\r\nX-Keep: yes\r\nContent-Length: 0\r\n\r\n";
        let result = finalize_forwarded_request(raw, false, None, None, &[]).unwrap();
        let text = String::from_utf8_lossy(&result);

        // Proxy adds exactly one Connection header of its own.
        assert_eq!(text.matches("Connection:").count(), 1);

        // TE and Upgrade nominated by Connection must be stripped (RFC 7230 §6.1 fix)
        assert!(
            !text.contains("TE:"),
            "expected TE stripped via Connection nomination"
        );
        assert!(
            !text.contains("Upgrade:"),
            "expected Upgrade stripped via Connection nomination"
        );

        // Non-nominated headers preserved, proxy adds its own Connection: close
        assert!(text.contains("X-Keep: yes"), "expected X-Keep preserved");
    }

    #[test]
    fn finalize_connection_nominated_case_insensitive() {
        let raw = b"GET / HTTP/1.1\r\nHost: x\r\nConnection: te, upgrade\r\nte: trailers\r\nUpgrade: h2c\r\nContent-Length: 0\r\n\r\n";
        let result = finalize_forwarded_request(raw, false, None, None, &[]).unwrap();
        let text = String::from_utf8_lossy(&result);

        assert!(!text.contains("te:"), "expected te stripped");
        assert!(
            !text.to_lowercase().contains("upgrade"),
            "expected Upgrade stripped case-insensitively"
        );
    }

    /// Regression: RFC 7230 §6.1 forbids nominating Content-Length via Connection.
    /// The request MUST be rejected before header removal or body forwarding.
    #[test]
    fn finalize_rejects_connection_nominating_content_length() {
        let raw =
            b"GET / HTTP/1.1\r\nHost: x\r\nConnection: Content-Length\r\nContent-Length: 0\r\n\r\n";
        let result = finalize_forwarded_request(raw, false, None, None, &[]);
        assert!(
            result.is_err(),
            "nominating Content-Length via Connection must be rejected, got: {:?}",
            result
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Content-Length"),
            "error should name Content-Length framing violation, got: {msg}"
        );
    }

    /// Regression: RFC 7230 §6.1 forbids nominating Transfer-Encoding via Connection.
    /// The request MUST be rejected before header removal or body forwarding.
    #[test]
    fn finalize_rejects_connection_nominating_transfer_encoding() {
        let raw = b"GET / HTTP/1.1\r\nHost: x\r\nConnection: Transfer-Encoding\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n";
        let result = finalize_forwarded_request(raw, false, None, None, &[]);
        assert!(
            result.is_err(),
            "nominating Transfer-Encoding via Connection must be rejected, got: {:?}",
            result
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Transfer-Encoding"),
            "error should name Transfer-Encoding framing violation, got: {msg}"
        );
    }

    /// Regression: header field values are arbitrary octets (RFC 7230 §3.2).
    /// A non-UTF-8 value on an ordinary header must be preserved verbatim, not
    /// coerced to an empty string.
    #[test]
    fn finalize_preserves_opaque_non_utf8_field_value() {
        // X-Custom carries a raw ISO-8859-1 byte 0xE9 ('é') which is invalid UTF-8.
        let raw: &[u8] =
            b"GET / HTTP/1.1\r\nHost: x\r\nX-Custom: caf\xE9\r\nContent-Length: 0\r\n\r\n";
        let result = finalize_forwarded_request(raw, false, None, None, &[]).unwrap();
        // The raw 0xE9 byte must appear unchanged in the rebuilt request.
        assert!(
            result.windows(1).any(|w| w == [0xE9]),
            "opaque non-UTF-8 field value must be preserved as raw bytes, rebuilt: {:?}",
            String::from_utf8_lossy(&result)
        );
        assert!(
            result
                .windows(b"X-Custom: ".len())
                .any(|w| w == b"X-Custom: "),
            "header name + value must be present in the rebuilt request"
        );
    }

    /// Regression: a malformed Connection value containing invalid bytes must be
    /// rejected wholesale, not partially filtered. A high-bit byte is invalid in
    /// HTTP/1.x token syntax and must fail the request.
    #[test]
    fn finalize_rejects_malformed_connection_value() {
        // 0xFF inside the Connection token list — invalid ASCII token syntax.
        let raw: &[u8] = b"GET / HTTP/1.1\r\nHost: x\r\nConnection: keep-alive, \xFF\r\nContent-Length: 0\r\n\r\n";
        let result = finalize_forwarded_request(raw, false, None, None, &[]);
        assert!(
            result.is_err(),
            "malformed Connection value with invalid bytes must be rejected, got: {:?}",
            result
        );
    }

    /// Regression: a Connection value that is valid UTF-8 but contains an ASCII
    /// control char is still malformed token syntax and must be rejected.
    #[test]
    fn finalize_rejects_connection_control_char() {
        let raw =
            b"GET / HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\x01\r\nContent-Length: 0\r\n\r\n";
        let result = finalize_forwarded_request(raw, false, None, None, &[]);
        assert!(
            result.is_err(),
            "Connection with control character must be rejected, got: {:?}",
            result
        );
    }

    /// Regression: a Connection value that is valid UTF-8 but contains a
    /// non-ASCII character (high-bit byte that decodes cleanly, e.g. a UTF-8
    /// multibyte sequence) is still invalid HTTP/1.x token syntax and must be
    /// rejected. This exercises the non-ASCII token validation branch rather
    /// than the invalid-UTF-8 path covered by
    /// `finalize_rejects_malformed_connection_value`.
    #[test]
    fn finalize_rejects_connection_non_ascii_token() {
        // 'é' encodes as 0xC3 0xA9 in UTF-8: valid UTF-8, non-ASCII token syntax.
        let raw: &[u8] =
            b"GET / HTTP/1.1\r\nHost: x\r\nConnection: keep-alive, caf\xC3\xA9\r\nContent-Length: 0\r\n\r\n";
        let result = finalize_forwarded_request(raw, false, None, None, &[]);
        assert!(
            result.is_err(),
            "Connection with non-ASCII but valid-UTF-8 token must be rejected, got: {:?}",
            result
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("malformed token"),
            "error should report the malformed token, got: {msg}"
        );
    }

    /// Regression: Connection options must be tokens (RFC 7230 §3.2.6).
    /// Separator characters such as ';' are not tchars and must reject the
    /// request instead of being accepted as part of a token.
    #[test]
    fn finalize_rejects_connection_separator_char() {
        let raw =
            b"GET / HTTP/1.1\r\nHost: x\r\nConnection: keep-alive;foo\r\nContent-Length: 0\r\n\r\n";
        let result = finalize_forwarded_request(raw, false, None, None, &[]);
        assert!(
            result.is_err(),
            "Connection option containing a separator must be rejected, got: {:?}",
            result
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("malformed token"),
            "error should report the malformed token, got: {msg}"
        );
    }

    /// Regression: whitespace around Connection options is OWS, i.e. only SP
    /// and HTAB (RFC 7230 §3.2.3). Non-ASCII whitespace such as NBSP must not
    /// be trimmed at a token edge; it must fail the tchar check instead of
    /// being silently accepted.
    #[test]
    fn finalize_rejects_connection_non_ascii_whitespace() {
        // NBSP (U+00A0) between the comma and the next option: valid UTF-8,
        // but neither SP/HTAB nor a tchar.
        let raw: &[u8] =
            b"GET / HTTP/1.1\r\nHost: x\r\nConnection: keep-alive, \xC2\xA0close\r\nContent-Length: 0\r\n\r\n";
        let result = finalize_forwarded_request(raw, false, None, None, &[]);
        assert!(
            result.is_err(),
            "Connection with non-ASCII whitespace must be rejected, got: {:?}",
            result
        );
    }
}
