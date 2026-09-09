use super::cancellation::{CancelUpstream, cancel_upstream_if_client_disconnected};
use super::common::{
    ResponseRetryPolicy, RouteAttemptLoggingContext, RouteAttemptResult,
    retryable_route_result_from_error,
};
use super::dispatch::{RelayAttemptContext, relay_attempted_response};
use super::probe::{ResponseProbe, probe_http_response, probe_http_response_local};
use crate::logging::OpenAiRouteObserver;
use crate::mesh;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::forwarded_request::prepare_peer_forwarded_request;
use crate::network::openai::request_normalize::ResponseAdapter;
use anyhow::Result;
use mesh_llm_events::logging::identifiers::RequestId;
use std::future::Future;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

pub(in crate::network::openai) async fn route_local_attempt(
    node: &mesh::Node,
    tcp_stream: &mut ClientStream,
    port: u16,
    prefetched: &[u8],
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    let RouteAttemptLoggingContext {
        request_id,
        retry_policy,
        response_adapter,
        route_observer,
        served_by,
    } = logging;
    let Ok((_instance_request, mut upstream)) = acquire_local_attempt_upstream(node, port).await
    else {
        return RouteAttemptResult::RetryableUnavailable;
    };
    let _inflight = node.begin_inflight_request();
    let _ = upstream.set_nodelay(true);
    if let Err(err) = forward_buffered_request(&mut upstream, prefetched).await {
        tracing::warn!(
            "API proxy: failed to forward buffered request to local OpenAI surface on {port}: {err}"
        );
        return RouteAttemptResult::RetryableUnavailable;
    }
    route_local_attempt_after_forward(
        tcp_stream,
        &mut upstream,
        port,
        request_id,
        retry_policy,
        response_adapter,
        served_by,
        route_observer,
    )
    .await
}

async fn acquire_local_attempt_upstream(
    node: &mesh::Node,
    port: u16,
) -> Result<(Option<crate::runtime::InstanceRequestGuard>, TcpStream), ()> {
    let instance_request = node
        .begin_runtime_instance_request(port)
        .await
        .map_err(|error| {
            tracing::debug!(%error, port, "local runtime instance rejected new work");
        })?;
    let upstream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .map_err(|error| {
            tracing::warn!("API proxy: can't reach local OpenAI surface on {port}: {error}");
        })?;
    Ok((instance_request, upstream))
}

#[allow(clippy::too_many_arguments)]
async fn route_local_attempt_after_forward<U: AsyncRead + Unpin + CancelUpstream>(
    tcp_stream: &mut ClientStream,
    upstream: &mut U,
    port: u16,
    request_id: RequestId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteAttemptResult {
    match probe_with_downstream_disconnect(tcp_stream, probe_http_response_local(upstream)).await {
        ProbeOutcome::ClientDisconnected => {
            cancel_upstream_if_client_disconnected(RouteAttemptResult::ClientDisconnected, upstream)
                .await
        }
        ProbeOutcome::Response(Ok(probe)) => {
            let result = relay_attempted_response(
                tcp_stream,
                upstream,
                probe,
                RelayAttemptContext {
                    request_id,
                    disconnect_message: "API proxy (local): downstream client disconnected during relay",
                    commit_message: "API proxy (local) ended after commit",
                    served_by,
                    route_observer,
                },
                retry_policy,
                response_adapter,
            )
            .await;
            cancel_upstream_if_client_disconnected(result, upstream).await
        }
        ProbeOutcome::Response(Err(err)) => {
            tracing::warn!(
                "API proxy: failed to read local response from OpenAI surface on {port}: {err}"
            );
            retryable_route_result_from_error(&err)
        }
    }
}

enum ProbeOutcome {
    ClientDisconnected,
    Response(Result<ResponseProbe>),
}

/// Race the complete upstream response probe against a non-consuming
/// downstream close check. Pipelined TCP bytes leave the watcher pending, while
/// a TCP reset or QUIC STOP_SENDING reports a client disconnect promptly.
async fn probe_with_downstream_disconnect<F>(client: &ClientStream, probe: F) -> ProbeOutcome
where
    F: Future<Output = Result<ResponseProbe>>,
{
    tokio::pin!(probe);
    tokio::select! {
        biased;
        disconnected = client.wait_for_response_disconnect() => {
            debug_assert!(disconnected);
            ProbeOutcome::ClientDisconnected
        }
        response = &mut probe => ProbeOutcome::Response(response),
    }
}

#[allow(clippy::too_many_arguments)]
async fn route_remote_attempt_after_forward<R: AsyncRead + Unpin + CancelUpstream>(
    tcp_stream: &mut ClientStream,
    quic_recv: &mut R,
    host_id: iroh::EndpointId,
    request_id: RequestId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteAttemptResult {
    match probe_with_downstream_disconnect(tcp_stream, probe_http_response(quic_recv)).await {
        ProbeOutcome::ClientDisconnected => {
            cancel_upstream_if_client_disconnected(
                RouteAttemptResult::ClientDisconnected,
                quic_recv,
            )
            .await
        }
        ProbeOutcome::Response(Ok(probe)) => {
            let result = relay_attempted_response(
                tcp_stream,
                quic_recv,
                probe,
                RelayAttemptContext {
                    request_id,
                    disconnect_message: "API proxy (remote): downstream client disconnected during relay",
                    commit_message: "API proxy (remote) ended after commit",
                    served_by,
                    route_observer,
                },
                retry_policy,
                response_adapter,
            )
            .await;
            cancel_upstream_if_client_disconnected(result, quic_recv).await
        }
        ProbeOutcome::Response(Err(err)) => {
            tracing::warn!(
                "API proxy: failed to read response from host {}: {err}",
                host_id.fmt_short()
            );
            retryable_route_result_from_error(&err)
        }
    }
}

pub(in crate::network::openai) async fn route_remote_attempt(
    node: &mesh::Node,
    tcp_stream: &mut ClientStream,
    host_id: iroh::EndpointId,
    prefetched: &[u8],
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    let RouteAttemptLoggingContext {
        request_id,
        retry_policy,
        response_adapter,
        route_observer,
        served_by,
    } = logging;
    let (mut quic_send, mut quic_recv) = match node.open_http_tunnel(host_id).await {
        Ok(tunnel) => tunnel,
        Err(err) => {
            tracing::warn!(
                "API proxy: can't tunnel to host {}: {err}",
                host_id.fmt_short()
            );
            return retryable_route_result_from_error(&err);
        }
    };

    if forward_peer_request(&mut quic_send, host_id, prefetched)
        .await
        .is_err()
    {
        return RouteAttemptResult::RetryableUnavailable;
    }

    route_remote_attempt_after_forward(
        tcp_stream,
        &mut quic_recv,
        host_id,
        request_id,
        retry_policy,
        response_adapter,
        served_by,
        route_observer,
    )
    .await
}

async fn forward_peer_request(
    quic_send: &mut iroh::endpoint::SendStream,
    host_id: iroh::EndpointId,
    prefetched: &[u8],
) -> Result<(), ()> {
    // Caller credentials are meaningful at ingress, not on a remote peer.
    let peer_request = match prepare_peer_forwarded_request(prefetched) {
        Ok(request) => request,
        Err(err) => {
            tracing::warn!(
                "API proxy: refusing to forward malformed request to host {}: {err}",
                host_id.fmt_short()
            );
            return Err(());
        }
    };

    if let Err(err) = quic_send.write_all(&peer_request).await {
        tracing::warn!(
            "API proxy: failed to forward buffered request to host {}: {err}",
            host_id.fmt_short()
        );
        return Err(());
    }

    Ok(())
}

async fn forward_buffered_request<W: AsyncWrite + Unpin>(
    upstream: &mut W,
    prefetched: &[u8],
) -> std::io::Result<()> {
    upstream.write_all(prefetched).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll};
    use tokio::io::AsyncReadExt;
    use tokio::io::ReadBuf;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio::time::{Duration, timeout};

    /// A real silent duplex pipe as the upstream half of `CancelUpstream`.
    struct DuplexUpstream {
        inner: tokio::io::DuplexStream,
        cancels: Arc<AtomicUsize>,
    }

    impl AsyncRead for DuplexUpstream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            Pin::new(&mut self.inner).poll_read(cx, buf)
        }
    }

    impl CancelUpstream for DuplexUpstream {
        async fn cancel(&mut self) {
            self.cancels.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// An upstream that hands out a fixed script of reads, then records whether
    /// the route arm cancelled it.
    struct ScriptedUpstream {
        steps: std::collections::VecDeque<Result<Vec<u8>, io::ErrorKind>>,
        cancels: Arc<AtomicUsize>,
    }

    impl ScriptedUpstream {
        fn new(steps: Vec<Result<Vec<u8>, io::ErrorKind>>) -> Self {
            Self {
                steps: steps.into(),
                cancels: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl AsyncRead for ScriptedUpstream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<io::Result<()>> {
            match self.steps.pop_front() {
                Some(Ok(bytes)) => {
                    buf.put_slice(&bytes);
                    Poll::Ready(Ok(()))
                }
                Some(Err(kind)) => Poll::Ready(Err(io::Error::new(kind, "scripted failure"))),
                None => Poll::Ready(Ok(())),
            }
        }
    }

    impl CancelUpstream for ScriptedUpstream {
        async fn cancel(&mut self) {
            self.cancels.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Both route arms must cancel their silent upstream as soon as the real
    /// downstream socket is reset, without waiting for the response probe.
    #[tokio::test]
    async fn a_local_attempt_cancels_the_upstream_when_the_client_disconnects() {
        let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_address = upstream_listener.local_addr().unwrap();
        let (upstream_ready_tx, upstream_ready_rx) = oneshot::channel();
        let upstream_task = tokio::spawn(async move {
            let (mut upstream_peer, _) = upstream_listener.accept().await.unwrap();
            upstream_ready_tx.send(()).unwrap();
            let mut bytes = [0u8; 1];
            timeout(Duration::from_secs(2), upstream_peer.read(&mut bytes))
                .await
                .expect("local cancellation must reach the upstream socket")
                .expect("reading the cancelled upstream socket must succeed")
        });

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            let mut upstream = TcpStream::connect(upstream_address).await.unwrap();
            route_local_attempt_after_forward(
                &mut client,
                &mut upstream,
                0,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        });
        let client_socket = TcpStream::connect(address).await.unwrap();
        upstream_ready_rx.await.unwrap();
        client_socket.set_zero_linger().unwrap();
        drop(client_socket);

        let result = timeout(Duration::from_secs(2), task)
            .await
            .expect("local route must notice a reset promptly")
            .unwrap();

        assert_eq!(result, RouteAttemptResult::ClientDisconnected);
        assert_eq!(
            upstream_task.await.unwrap(),
            0,
            "local cancellation must shut down the concrete TCP upstream"
        );
    }

    #[tokio::test]
    async fn a_remote_attempt_cancels_the_peer_tunnel_when_the_client_disconnects() {
        let (_upstream_writer, upstream_reader) = tokio::io::duplex(64 * 1024);
        let cancels = Arc::new(AtomicUsize::new(0));
        let mut upstream = DuplexUpstream {
            inner: upstream_reader,
            cancels: Arc::clone(&cancels),
        };
        let host_id = iroh::SecretKey::generate().public();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            route_remote_attempt_after_forward(
                &mut client,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        });
        let client_socket = TcpStream::connect(address).await.unwrap();
        client_socket.set_zero_linger().unwrap();
        drop(client_socket);

        let result = timeout(Duration::from_secs(2), task)
            .await
            .expect("remote route must notice a reset promptly")
            .unwrap();

        assert_eq!(result, RouteAttemptResult::ClientDisconnected);
        assert_eq!(cancels.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_quic_downstream_reset_cancels_the_peer_tunnel_during_probe() {
        const TEST_ALPN: &[u8] = b"mesh-llm/routing-cancellation-test/1";

        let server = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(iroh::SecretKey::generate())
            .alpns(vec![TEST_ALPN.to_vec()])
            .relay_mode(iroh::endpoint::RelayMode::Disabled)
            .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
            .unwrap()
            .bind()
            .await
            .unwrap();
        let server_endpoint = server.clone();
        let cancels = Arc::new(AtomicUsize::new(0));
        let route_cancels = Arc::clone(&cancels);
        let host_id = iroh::SecretKey::generate().public();
        let route = tokio::spawn(async move {
            let incoming = server_endpoint.accept().await.expect("connection arrives");
            let connection = incoming.await.expect("connection negotiates");
            let (send, recv) = connection.accept_bi().await.expect("stream arrives");
            let mut downstream = ClientStream::from_quic_with_prefix(recv, send, Vec::new());
            let (_upstream_writer, upstream_reader) = tokio::io::duplex(64 * 1024);
            let mut upstream = DuplexUpstream {
                inner: upstream_reader,
                cancels: route_cancels,
            };
            route_remote_attempt_after_forward(
                &mut downstream,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        });

        let client = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(iroh::SecretKey::generate())
            .relay_mode(iroh::endpoint::RelayMode::Disabled)
            .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
            .unwrap()
            .bind()
            .await
            .unwrap();
        let connection = client.connect(server.addr(), TEST_ALPN).await.unwrap();
        let (_request_send, mut response_recv) = connection.open_bi().await.unwrap();
        response_recv.stop(42u32.into()).unwrap();

        let result = timeout(Duration::from_secs(2), route)
            .await
            .expect("QUIC reset must interrupt the pending response probe")
            .unwrap();
        assert_eq!(result, RouteAttemptResult::ClientDisconnected);
        assert_eq!(cancels.load(Ordering::SeqCst), 1);

        client.close().await;
        server.close().await;
    }

    #[tokio::test]
    async fn pipelined_downstream_bytes_are_not_consumed_by_disconnect_watch() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (downstream, _) = listener.accept().await.unwrap();
            let mut downstream: ClientStream = downstream.into();
            assert!(
                timeout(
                    Duration::from_millis(100),
                    downstream.wait_for_response_disconnect()
                )
                .await
                .is_err(),
                "pipelined bytes are not a disconnect"
            );
            let mut pipelined = [0u8; 1];
            timeout(
                Duration::from_secs(1),
                downstream.read_exact(&mut pipelined),
            )
            .await
            .expect("pipelined byte must remain available after probing")
            .unwrap();
            pipelined
        });

        let mut client = TcpStream::connect(address).await.unwrap();
        client.write_all(b"x").await.unwrap();

        assert_eq!(task.await.unwrap(), [b'x']);
    }

    /// The counterpart: a normal delivery must not cancel anything.
    #[tokio::test]
    async fn a_remote_attempt_that_delivers_leaves_the_peer_tunnel_alone() {
        let body = "x".repeat(8);
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let mut upstream = ScriptedUpstream::new(vec![Ok(header.into_bytes())]);
        let cancels = Arc::clone(&upstream.cancels);
        let host_id = iroh::SecretKey::generate().public();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            route_remote_attempt_after_forward(
                &mut client,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        });
        let mut socket = TcpStream::connect(address).await.unwrap();
        let mut relayed = Vec::new();
        socket.read_to_end(&mut relayed).await.unwrap();

        let result = task.await.unwrap();

        assert!(matches!(result, RouteAttemptResult::Delivered { .. }));
        assert_eq!(cancels.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_write_half_closed_downstream_still_receives_the_upstream_response() {
        let body = "half-close response";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let mut upstream = ScriptedUpstream::new(vec![Ok(response.into_bytes())]);
        let host_id = iroh::SecretKey::generate().public();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            route_remote_attempt_after_forward(
                &mut client,
                &mut upstream,
                host_id,
                RequestId::new(),
                ResponseRetryPolicy::next_target_available(false),
                ResponseAdapter::None,
                None,
                OpenAiRouteObserver::default(),
            )
            .await
        });

        let mut socket = TcpStream::connect(address).await.unwrap();
        socket.shutdown().await.unwrap();
        let mut relayed = Vec::new();
        timeout(Duration::from_secs(1), socket.read_to_end(&mut relayed))
            .await
            .expect("write-half-closed client should still receive a response")
            .unwrap();
        let result = timeout(Duration::from_secs(1), task)
            .await
            .expect("route should finish after relaying the response")
            .unwrap();

        assert!(matches!(result, RouteAttemptResult::Delivered { .. }));
        assert!(String::from_utf8_lossy(&relayed).contains(body));
    }

    #[tokio::test]
    async fn local_and_remote_forwarding_preserve_the_canonical_request_id_bytes() {
        const REQUEST: &[u8] = b"POST /v1/chat/completions HTTP/1.1\r\nx-request-id: 4c3ca94d-bc1f-4759-912d-f4f6d77d5515\r\nContent-Length: 2\r\n\r\n{}";

        for _route in ["local", "remote"] {
            let (mut upstream, mut received) = tokio::io::duplex(REQUEST.len());
            let forwarding = tokio::spawn(async move {
                forward_buffered_request(&mut upstream, REQUEST)
                    .await
                    .unwrap();
            });
            let mut forwarded = Vec::new();
            received.read_to_end(&mut forwarded).await.unwrap();
            forwarding.await.unwrap();

            assert_eq!(forwarded, REQUEST);
        }
    }
}
