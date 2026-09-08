use std::net::{IpAddr, SocketAddr};

pub(crate) fn requires_trusted_local_access(method: &str, path: &str) -> bool {
    // Log history is local operator data. Keep this broad path classification
    // at the server boundary so all current and future `/api/logs/**` routes
    // are rejected before dispatch can touch the query facade or store.
    if path == "/api/logs" || path.starts_with("/api/logs/") {
        return true;
    }
    if path == "/mcp"
        || path.starts_with("/api/plugins")
        || (method == "POST"
            && (path == "/mesh/hook"
                || path == "/api/objects"
                || path.starts_with("/api/objects/")))
    {
        return true;
    }

    matches!(
        (method, path),
        ("GET", "/api/runtime/events/v1")
            | ("GET", "/api/runtime/endpoints")
            | ("GET", "/api/runtime/config-schema")
            | ("GET", "/api/runtime/config-control-state")
            | ("GET", "/api/runtime/control-bootstrap")
            | ("GET", "/api/runtime/intents")
            | ("GET", "/api/runtime/activity")
            | ("POST", "/api/runtime/control/get-config")
            | ("POST", "/api/runtime/control/scan-refresh")
            | ("POST", "/api/runtime/control/refresh-inventory")
            | ("POST", "/api/runtime/control/apply-config")
            | ("POST", "/api/runtime/control/load-model")
            | ("POST", "/api/runtime/control/unload-model")
            | ("POST", "/api/runtime/control/ensure-model")
            | ("POST", "/api/runtime/control/drain-model")
            | ("POST", "/api/runtime/config/validate")
            | ("POST", "/api/runtime/pick-directory")
            | ("POST", "/api/runtime/mesh-guardrails")
            | ("POST", "/api/runtime/models")
            | ("POST", "/api/model-interests")
            | ("PUT", "/api/runtime/activity/override")
    ) || (method == "DELETE"
        && (path == "/api/runtime/activity/override"
            || path.starts_with("/api/runtime/models/")
            || path.starts_with("/api/runtime/instances/")
            || path.starts_with("/api/model-interests/")))
}

pub(crate) fn is_trusted_local_request(
    peer_addr: Option<SocketAddr>,
    origin: Option<&str>,
    host: Option<&str>,
) -> bool {
    let Some(peer_addr) = peer_addr else {
        return false;
    };
    if !is_loopback_ip(peer_addr.ip()) {
        return false;
    }

    host.is_none_or(is_trusted_local_authority) && origin.is_none_or(is_trusted_local_origin)
}

pub(crate) fn request_origin(raw_request: &[u8]) -> Result<Option<&str>, ()> {
    request_header(raw_request, "origin")
}

pub(crate) fn request_host(raw_request: &[u8]) -> Result<Option<&str>, ()> {
    request_header(raw_request, "host")
}

pub(crate) fn request_header<'a>(raw_request: &'a [u8], name: &str) -> Result<Option<&'a str>, ()> {
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut request = httparse::Request::new(&mut headers);
    match request.parse(raw_request).map_err(|_| ())? {
        httparse::Status::Complete(_) => {}
        httparse::Status::Partial => return Err(()),
    }
    request
        .headers
        .iter()
        .find(|header| header.name.eq_ignore_ascii_case(name))
        .map(|header| std::str::from_utf8(header.value).map_err(|_| ()))
        .transpose()
}

fn is_loopback_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ip.is_loopback(),
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip
                    .to_ipv4_mapped()
                    .is_some_and(|mapped| mapped.is_loopback())
        }
    }
}

fn is_trusted_local_origin(origin: &str) -> bool {
    let Ok(origin) = url::Url::parse(origin.trim()) else {
        return false;
    };
    if !matches!(origin.scheme(), "http" | "https") {
        return false;
    }
    origin
        .host_str()
        .is_some_and(|host| host.eq_ignore_ascii_case("localhost") || is_loopback_host(host))
}

fn is_trusted_local_authority(authority: &str) -> bool {
    let Ok(authority) = authority.trim().parse::<http::uri::Authority>() else {
        return false;
    };
    let host = authority.host();
    host.eq_ignore_ascii_case("localhost") || is_loopback_host(host)
}

fn is_loopback_host(host: &str) -> bool {
    host.trim_matches(['[', ']'])
        .parse::<IpAddr>()
        .is_ok_and(is_loopback_ip)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitive_management_routes_require_trusted_local_access() {
        for (method, path) in [
            ("GET", "/api/logs/requests"),
            ("POST", "/api/logs/requests/export"),
            ("POST", "/mcp"),
            ("GET", "/api/plugins"),
            ("POST", "/api/plugins/agents/tools/run"),
            ("POST", "/api/objects"),
            ("POST", "/api/objects/complete"),
            ("POST", "/mesh/hook"),
            ("GET", "/api/runtime/events/v1"),
            ("GET", "/api/runtime/endpoints"),
            ("POST", "/api/runtime/models"),
            ("DELETE", "/api/runtime/models/qwen"),
            ("DELETE", "/api/runtime/instances/instance-1"),
            ("POST", "/api/runtime/mesh-guardrails"),
            ("POST", "/api/model-interests"),
            ("DELETE", "/api/model-interests/qwen"),
        ] {
            assert!(
                requires_trusted_local_access(method, path),
                "{method} {path} must be local-only"
            );
        }
    }

    #[test]
    fn read_only_status_routes_remain_network_accessible() {
        for path in [
            "/api/status",
            "/api/models",
            "/api/runtime",
            "/api/events",
            "/api/discover",
        ] {
            assert!(
                !requires_trusted_local_access("GET", path),
                "GET {path} should remain readable"
            );
        }
    }

    #[test]
    fn non_loopback_callers_are_rejected() {
        assert!(!is_trusted_local_request(
            Some(SocketAddr::from(([192, 0, 2, 10], 40123))),
            None,
            Some("localhost:3131")
        ));
    }

    #[test]
    fn browser_requests_from_untrusted_origins_are_rejected() {
        assert!(!is_trusted_local_request(
            Some(SocketAddr::from(([127, 0, 0, 1], 40123))),
            Some("https://attacker.example"),
            Some("localhost:3131")
        ));
        assert!(!is_trusted_local_request(
            Some(SocketAddr::from(([127, 0, 0, 1], 40123))),
            Some("null"),
            Some("localhost:3131")
        ));
        assert!(!is_trusted_local_request(
            Some(SocketAddr::from(([127, 0, 0, 1], 40123))),
            None,
            Some("attacker.example")
        ));
    }

    #[test]
    fn origin_header_is_extracted_case_insensitively() {
        let request = b"POST /mcp HTTP/1.1\r\nHost: localhost\r\noRiGiN: https://attacker.example\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(
            request_origin(request),
            Ok(Some("https://attacker.example"))
        );
    }

    #[test]
    fn malformed_security_header_is_rejected() {
        let request = b"POST /mcp HTTP/1.1\r\nHost: localhost\r\nOrigin: https://local\xff\r\n\r\n";
        assert_eq!(request_origin(request), Err(()));
    }

    #[test]
    fn trusted_local_callers_are_allowed() {
        let peer = Some(SocketAddr::from(([127, 0, 0, 1], 40123)));
        assert!(is_trusted_local_request(peer, None, Some("localhost:3131")));
        assert!(is_trusted_local_request(
            peer,
            Some("http://localhost:3131"),
            Some("localhost:3131")
        ));
        assert!(is_trusted_local_request(
            peer,
            Some("http://127.0.0.1:3131"),
            Some("127.0.0.1:3131")
        ));
        assert!(is_trusted_local_request(
            Some(SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 1], 40123))),
            Some("http://[::1]:3131"),
            Some("[::1]:3131")
        ));
    }
}
