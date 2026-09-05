//! Canonical v1 cursor: `rt1:<process-instance-uuid>:<sequence>`.
//!
//! Transport mirrors the existing typed log cursor / `Last-Event-ID`
//! merging in `api/routes/logs/events/query.rs` (header validated
//! independently from query; header wins when both are present), but this
//! is a distinct grammar and storage domain — it shares only the merge
//! shape, never the logs replay cursor type itself.

use mesh_llm_runtime_event_contracts::ProcessInstanceId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Cursor {
    pub(super) process_instance: ProcessInstanceId,
    pub(super) sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CursorError {
    Malformed,
}

impl Cursor {
    pub(super) const fn new(process_instance: ProcessInstanceId, sequence: u64) -> Self {
        Self {
            process_instance,
            sequence,
        }
    }

    pub(super) fn parse(value: &str) -> Result<Self, CursorError> {
        let rest = value.strip_prefix("rt1:").ok_or(CursorError::Malformed)?;
        let (uuid_part, seq_part) = rest.split_once(':').ok_or(CursorError::Malformed)?;
        let uuid = uuid::Uuid::parse_str(uuid_part).map_err(|_| CursorError::Malformed)?;
        if uuid.hyphenated().to_string() != uuid_part {
            return Err(CursorError::Malformed);
        }
        let sequence = parse_canonical_u64(seq_part)?;
        Ok(Self {
            process_instance: ProcessInstanceId::from_uuid(uuid),
            sequence,
        })
    }

    pub(super) fn encode(self) -> String {
        format!("rt1:{}:{}", self.process_instance.as_uuid(), self.sequence)
    }
}

fn parse_canonical_u64(value: &str) -> Result<u64, CursorError> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(CursorError::Malformed);
    }
    if value.len() > 1 && value.starts_with('0') {
        return Err(CursorError::Malformed);
    }
    value.parse().map_err(|_| CursorError::Malformed)
}

/// Resolve the request cursor from `Last-Event-ID` and `?cursor=`. Each is
/// validated independently, so a malformed value in either rejects with
/// `Malformed`; when both are present and valid, the header wins.
pub(super) fn resolve(path: &str, raw_request: &[u8]) -> Result<Option<Cursor>, CursorError> {
    let header = header_cursor(raw_request)?;
    let query = query_cursor(path)?;
    Ok(header.or(query))
}

fn header_cursor(raw_request: &[u8]) -> Result<Option<Cursor>, CursorError> {
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut request = httparse::Request::new(&mut headers);
    if !matches!(
        request.parse(raw_request),
        Ok(httparse::Status::Complete(_))
    ) {
        return Ok(None);
    }
    let mut values = request
        .headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case("last-event-id"));
    let Some(header) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(CursorError::Malformed);
    }
    let value = std::str::from_utf8(header.value).map_err(|_| CursorError::Malformed)?;
    Cursor::parse(value.trim()).map(Some)
}

fn query_cursor(path: &str) -> Result<Option<Cursor>, CursorError> {
    let Some((_, query)) = path.split_once('?') else {
        return Ok(None);
    };
    let mut found = None;
    for (key, value) in url::form_urlencoded::parse(query.as_bytes()) {
        if key == "cursor" {
            if found.is_some() {
                return Err(CursorError::Malformed);
            }
            found = Some(Cursor::parse(value.as_ref())?);
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    const UUID: &str = "0195f000-0000-7000-8000-000000000001";

    #[test]
    fn canonical_cursor_round_trips() {
        let cursor = Cursor::parse(&format!("rt1:{UUID}:42")).expect("valid cursor");
        assert_eq!(cursor.sequence, 42);
        assert_eq!(cursor.encode(), format!("rt1:{UUID}:42"));
    }

    #[test]
    fn zero_sequence_is_canonical() {
        assert!(Cursor::parse(&format!("rt1:{UUID}:0")).is_ok());
    }

    #[test]
    fn malformed_cursors_are_rejected() {
        for value in [
            "rt1:not-a-uuid:1",
            "rt2:0195f000-0000-7000-8000-000000000001:1",
            &format!("rt1:{UUID}:"),
            &format!("rt1:{UUID}:01"),
            &format!("rt1:{UUID}:-1"),
            &format!("rt1:{UUID}:1.0"),
            &format!("rt1:{UUID}"),
            &format!("rt1:{}:1", UUID.to_ascii_uppercase()),
            &format!("rt1:{{{UUID}}}:1"),
            "",
        ] {
            assert!(Cursor::parse(value).is_err(), "must reject {value:?}");
        }
    }

    #[test]
    fn header_wins_over_query_when_both_present() {
        let raw = format!(
            "GET /api/runtime/events/v1 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\nLast-Event-ID: rt1:{UUID}:9\r\n\r\n"
        );
        let resolved = resolve(
            &format!("/api/runtime/events/v1?cursor=rt1%3A{UUID}%3A3"),
            raw.as_bytes(),
        )
        .expect("resolves")
        .expect("cursor present");
        assert_eq!(resolved.sequence, 9);
    }

    #[test]
    fn query_cursor_used_when_no_header_present() {
        let raw = b"GET /api/runtime/events/v1 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n";
        let resolved = resolve(
            &format!("/api/runtime/events/v1?cursor=rt1%3A{UUID}%3A3"),
            raw,
        )
        .expect("resolves")
        .expect("cursor present");
        assert_eq!(resolved.sequence, 3);
    }

    #[test]
    fn no_cursor_present_resolves_to_none() {
        let raw = b"GET /api/runtime/events/v1 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n";
        assert_eq!(resolve("/api/runtime/events/v1", raw), Ok(None));
    }

    #[test]
    fn malformed_header_cursor_rejects_even_with_a_valid_query_cursor() {
        let raw = b"GET /api/runtime/events/v1 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\nLast-Event-ID: garbage\r\n\r\n";
        assert_eq!(
            resolve(
                &format!("/api/runtime/events/v1?cursor=rt1%3A{UUID}%3A3"),
                raw
            ),
            Err(CursorError::Malformed)
        );
    }

    #[test]
    fn malformed_query_cursor_rejects_even_without_a_header() {
        let raw = b"GET /api/runtime/events/v1 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n";
        assert_eq!(
            resolve("/api/runtime/events/v1?cursor=garbage", raw),
            Err(CursorError::Malformed)
        );
    }

    #[test]
    fn duplicate_query_cursor_is_rejected() {
        let raw = b"GET /api/runtime/events/v1 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n";
        assert_eq!(
            resolve(
                &format!("/api/runtime/events/v1?cursor=rt1%3A{UUID}%3A1&cursor=rt1%3A{UUID}%3A2"),
                raw
            ),
            Err(CursorError::Malformed)
        );
    }
}
