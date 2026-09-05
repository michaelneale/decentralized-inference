//! Request-local management API lifecycle observation.

use std::sync::{
    Arc,
    atomic::{AtomicU16, Ordering},
};

use mesh_llm_events::logging::identifiers::RequestId;

use crate::logging::ManagementRequestLifecycle;

tokio::task_local! {
    static MANAGEMENT_LIFECYCLE: Arc<ManagementLifecycleContext>;
}

pub(super) struct ManagementLifecycleContext {
    lifecycle: ManagementRequestLifecycle,
    status: AtomicU16,
}

impl ManagementLifecycleContext {
    fn new(lifecycle: ManagementRequestLifecycle) -> Self {
        Self {
            lifecycle,
            status: AtomicU16::new(0),
        }
    }

    fn record_status(&self, status: u16) {
        self.status.store(status, Ordering::Release);
    }

    fn finish(&self) {
        let status = self.status.load(Ordering::Acquire);
        if status == 0 {
            self.lifecycle.fail_dispatch();
        } else {
            self.lifecycle.finish_status(status);
        }
    }
}

pub(super) fn request_id_from_raw(raw: &[u8]) -> RequestId {
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut request = httparse::Request::new(&mut headers);
    let Ok(httparse::Status::Complete(_)) = request.parse(raw) else {
        return RequestId::default();
    };
    let mut request_id_headers = request
        .headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case("x-request-id"));
    let Some(header) = request_id_headers.next() else {
        return RequestId::default();
    };
    if request_id_headers.next().is_some() {
        return RequestId::default();
    }
    std::str::from_utf8(header.value)
        .ok()
        .and_then(openai_frontend::parse_request_id)
        .unwrap_or_default()
}

pub(super) fn eligible_management_route(method: &str, path: &str) -> bool {
    is_mutation_method(method)
        && path != "/models"
        && !path.starts_with("/v1/")
        && path != "/api/chat"
        && path != "/api/responses"
        && path != "/api/objects"
        && !path.starts_with("/api/objects/")
        && path != "/mesh/hook"
        && path != "/api/logs"
        && !path.starts_with("/api/logs/")
        && (path.starts_with("/api/") || path == "/mcp")
}

fn is_mutation_method(method: &str) -> bool {
    matches!(method, "POST" | "PUT" | "PATCH" | "DELETE")
}

pub(super) fn method_route_label(method: &str, path: &str) -> &'static str {
    match (method, path) {
        ("GET", "/api/status") => "management_get_status",
        ("GET", "/api/models") => "management_get_models",
        ("GET", "/api/events") => "management_get_events",
        ("GET", "/api/runtime/events") => "management_get_runtime_events",
        ("GET", "/api/runtime/events/v1") => "management_get_runtime_events_v1",
        ("GET", _) => "management_get_other",
        ("POST", _) => "management_post",
        ("PUT", _) => "management_put",
        ("PATCH", _) => "management_patch",
        ("DELETE", _) => "management_delete",
        _ => "management_other",
    }
}

pub(super) async fn scope<F, T>(lifecycle: ManagementRequestLifecycle, future: F) -> T
where
    F: std::future::Future<Output = T>,
{
    let context = Arc::new(ManagementLifecycleContext::new(lifecycle));
    let result = MANAGEMENT_LIFECYCLE
        .scope(Arc::clone(&context), future)
        .await;
    context.finish();
    result
}

pub(super) fn record_response_status(status: u16) {
    let _ = MANAGEMENT_LIFECYCLE.try_with(|context| context.record_status(status));
}

pub(super) fn response_request_id_header() -> Option<String> {
    MANAGEMENT_LIFECYCLE
        .try_with(|context| context.lifecycle.request_id().as_uuid().to_string())
        .ok()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use mesh_llm_events::logging::identifiers::RequestId;

    use super::*;

    fn lifecycle(
        request_id: RequestId,
    ) -> (
        Arc<crate::logging::LoggingService>,
        ManagementRequestLifecycle,
    ) {
        let service = Arc::new(crate::logging::LoggingService::new_disabled(
            Default::default(),
        ));
        let lifecycle = ManagementRequestLifecycle::register(
            Arc::clone(&service),
            request_id,
            "management_get_status",
        );
        (service, lifecycle)
    }

    #[tokio::test]
    async fn valid_request_id_is_propagated_and_status_completes_once() {
        let raw = b"GET /api/status HTTP/1.1\r\nHost: localhost\r\nx-request-id: 00000000-0000-4000-8000-000000000001\r\n\r\n";
        let request_id = request_id_from_raw(raw);
        let (service, lifecycle) = lifecycle(request_id);

        scope(lifecycle, async {
            assert_eq!(
                response_request_id_header().as_deref(),
                Some("00000000-0000-4000-8000-000000000001")
            );
            record_response_status(200);
        })
        .await;

        let records = service.bus_ref().replay_window().records;
        assert_eq!(
            records
                .iter()
                .filter(|record| record.entry.payload.contains("admitted"))
                .count(),
            1
        );
        assert_eq!(
            records
                .iter()
                .filter(|record| record.entry.payload.contains("completed"))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn invalid_or_missing_request_ids_are_replaced_and_rejections_are_terminal() {
        for raw in [
            b"GET /api/unknown HTTP/1.1\r\nHost: localhost\r\nx-request-id: not-a-uuid\r\n\r\n"
                .as_slice(),
            b"GET /api/unknown HTTP/1.1\r\nHost: localhost\r\n\r\n".as_slice(),
        ] {
            let request_id = request_id_from_raw(raw);
            assert!(uuid::Uuid::parse_str(&request_id.as_uuid().to_string()).is_ok());
            let (service, lifecycle) = lifecycle(request_id);
            scope(lifecycle, async { record_response_status(404) }).await;
            assert_eq!(
                service
                    .bus_ref()
                    .replay_window()
                    .records
                    .iter()
                    .filter(|record| record.entry.payload.contains("rejected"))
                    .count(),
                1
            );
        }
    }

    #[test]
    fn duplicate_request_ids_are_replaced_instead_of_selecting_the_first() {
        let first = "00000000-0000-4000-8000-000000000051";
        let second = "00000000-0000-4000-8000-000000000052";
        let raw = format!(
            "POST /api/runtime/config/validate HTTP/1.1\r\nHost: localhost\r\nx-request-id: {first}\r\nX-Request-Id: {second}\r\nContent-Length: 0\r\n\r\n"
        );

        let request_id = request_id_from_raw(raw.as_bytes());

        assert_ne!(request_id.as_uuid().to_string(), first);
        assert_ne!(request_id.as_uuid().to_string(), second);
    }

    #[tokio::test]
    async fn trusted_local_and_server_failures_use_bounded_terminal_states() {
        let (_, rejected) = lifecycle(RequestId::new());
        scope(rejected, async { record_response_status(403) }).await;

        let (service, failed) = lifecycle(RequestId::new());
        scope(failed, async { record_response_status(500) }).await;
        assert!(
            service
                .bus_ref()
                .replay_window()
                .records
                .iter()
                .any(|record| record.entry.payload.contains("management_http_failed"))
        );
    }

    #[tokio::test]
    async fn aborting_a_scoped_management_task_terminalizes_it_as_dropped() {
        let request_id = RequestId::new();
        let (service, lifecycle) = lifecycle(request_id);
        let task = tokio::spawn(scope(lifecycle, std::future::pending::<()>()));
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;

        let summary = service
            .registry_ref()
            .get_recent(&request_id.as_uuid().to_string())
            .expect("aborted scoped request is terminalized");
        assert_eq!(summary.state, "dropped");
        assert_eq!(
            service
                .bus_ref()
                .replay_window()
                .records
                .iter()
                .filter(|record| record.entry.payload.contains("dropped"))
                .count(),
            1
        );
    }

    #[test]
    fn management_lifecycle_routes_are_mutation_only() {
        let mutation_methods = ["POST", "PUT", "PATCH", "DELETE"];
        let observation_methods = ["GET", "HEAD", "OPTIONS", "TRACE", "CONNECT"];
        let management_paths = ["/api/runtime/control", "/api/plugins/install", "/mcp"];
        let excluded_paths = [
            "/v1/chat/completions",
            "/models",
            "/api/chat",
            "/api/responses",
            "/api/objects",
            "/api/objects/item",
            "/api/logs",
            "/api/logs/requests",
            "/mesh/hook",
        ];

        for method in mutation_methods {
            for path in management_paths {
                assert!(
                    eligible_management_route(method, path),
                    "{method} {path} must be eligible"
                );
            }
        }
        for method in observation_methods {
            for path in management_paths {
                assert!(
                    !eligible_management_route(method, path),
                    "{method} {path} must be excluded"
                );
            }
        }
        for method in mutation_methods.into_iter().chain(observation_methods) {
            for path in excluded_paths {
                assert!(
                    !eligible_management_route(method, path),
                    "{method} {path} must stay excluded"
                );
            }
        }
    }

    #[test]
    fn patch_management_routes_keep_their_method_label() {
        assert_eq!(
            method_route_label("PATCH", "/api/plugins/demo/web-ui/config"),
            "management_patch"
        );
    }

    #[test]
    fn runtime_events_v1_route_has_its_own_label_distinct_from_the_legacy_route() {
        assert_eq!(
            method_route_label("GET", "/api/runtime/events/v1"),
            "management_get_runtime_events_v1"
        );
        assert_eq!(
            method_route_label("GET", "/api/runtime/events"),
            "management_get_runtime_events"
        );
    }
}
