use super::*;
use crate::inference::pipeline;
use crate::network::router;
use crate::plugin;
use crate::plugins::blobstore::BlobStore;
use base64::Engine;
use rmcp::model::ErrorCode;
use serde_json::json;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{oneshot, watch};

async fn spawn_api_proxy_test_harness(
    targets: election::ModelTargets,
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let (addr, handle, _) = spawn_api_proxy_test_harness_with_affinity(targets).await;
    (addr, handle)
}

async fn spawn_api_proxy_test_harness_with_affinity(
    targets: election::ModelTargets,
) -> (
    SocketAddr,
    tokio::task::JoinHandle<()>,
    affinity::AffinityRouter,
) {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (_target_tx, target_rx) = watch::channel(targets);
    let affinity = affinity::AffinityRouter::default();
    let handle = tokio::spawn(api_proxy(
        node,
        addr.port(),
        target_rx,
        Some(listener),
        false,
        affinity.clone(),
    ));
    (addr, handle, affinity)
}

async fn spawn_api_proxy_test_harness_with_contexts(
    targets: election::ModelTargets,
    contexts: &[(&str, u32)],
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    for (model, context_length) in contexts {
        node.set_model_runtime_context_length(model, Some(*context_length))
            .await;
    }
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (_target_tx, target_rx) = watch::channel(targets);
    let handle = tokio::spawn(api_proxy(
        node,
        addr.port(),
        target_rx,
        Some(listener),
        false,
        affinity::AffinityRouter::default(),
    ));
    (addr, handle)
}

async fn spawn_api_proxy_test_harness_with_plugin_manager(
    targets: election::ModelTargets,
    plugin_manager: plugin::PluginManager,
) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    node.set_plugin_manager(plugin_manager).await;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (_target_tx, target_rx) = watch::channel(targets);
    let handle = tokio::spawn(api_proxy(
        node,
        addr.port(),
        target_rx,
        Some(listener),
        false,
        affinity::AffinityRouter::default(),
    ));
    (addr, handle)
}

#[derive(Clone)]
struct BlobstoreTestBridge {
    plugin_name: String,
    store: BlobStore,
}

#[derive(Clone, Default)]
struct NoopTestBridge;

impl BlobstoreTestBridge {
    fn error_response(message: impl Into<String>) -> plugin::proto::ErrorResponse {
        plugin::proto::ErrorResponse {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: message.into(),
            data_json: String::new(),
        }
    }
}

impl plugin::PluginRpcBridge for NoopTestBridge {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        _params_json: String,
    ) -> plugin::BridgeFuture<Result<plugin::RpcResult, plugin::proto::ErrorResponse>> {
        Box::pin(async move {
            Err(plugin::proto::ErrorResponse {
                code: ErrorCode::METHOD_NOT_FOUND.0,
                message: format!("Noop test bridge cannot handle {plugin_name}:{method}"),
                data_json: String::new(),
            })
        })
    }

    fn handle_notification(
        &self,
        _plugin_name: String,
        _method: String,
        _params_json: String,
    ) -> plugin::BridgeFuture<()> {
        Box::pin(async {})
    }
}

impl plugin::PluginRpcBridge for BlobstoreTestBridge {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> plugin::BridgeFuture<Result<plugin::RpcResult, plugin::proto::ErrorResponse>> {
        let expected_plugin_name = self.plugin_name.clone();
        let store = self.store.clone();
        Box::pin(async move {
            if plugin_name != expected_plugin_name {
                return Err(Self::error_response(format!(
                    "Unsupported test plugin '{}'",
                    plugin_name
                )));
            }

            if method == "tools/call" {
                let request: mesh_llm_plugin::OperationRequest = serde_json::from_str(&params_json)
                    .map_err(|err| Self::error_response(err.to_string()))?;
                let result_json = match request.name.as_str() {
                    crate::plugins::blobstore::PUT_REQUEST_OBJECT_TOOL => {
                        let request: crate::plugins::blobstore::PutRequestObjectRequest =
                            serde_json::from_value(request.arguments)
                                .map_err(|err| Self::error_response(err.to_string()))?;
                        let response = store
                            .put_request_object(request)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        let value = serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        serde_json::to_string(&rmcp::model::CallToolResult::structured(value))
                            .map_err(|err| Self::error_response(err.to_string()))?
                    }
                    crate::plugins::blobstore::GET_REQUEST_OBJECT_TOOL => {
                        let request: crate::plugins::blobstore::GetRequestObjectRequest =
                            serde_json::from_value(request.arguments)
                                .map_err(|err| Self::error_response(err.to_string()))?;
                        let response = store
                            .get_request_object(request)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        let value = serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        serde_json::to_string(&rmcp::model::CallToolResult::structured(value))
                            .map_err(|err| Self::error_response(err.to_string()))?
                    }
                    crate::plugins::blobstore::COMPLETE_REQUEST_TOOL
                    | crate::plugins::blobstore::ABORT_REQUEST_TOOL => {
                        let request: crate::plugins::blobstore::FinishRequestRequest =
                            serde_json::from_value(request.arguments)
                                .map_err(|err| Self::error_response(err.to_string()))?;
                        let response = store
                            .finish_request(&request.request_id)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        let value = serde_json::to_value(response)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                        serde_json::to_string(&rmcp::model::CallToolResult::structured(value))
                            .map_err(|err| Self::error_response(err.to_string()))?
                    }
                    _ => {
                        return Err(Self::error_response(format!(
                            "Unsupported blobstore tool '{}'",
                            request.name
                        )));
                    }
                };
                return Ok(plugin::RpcResult { result_json });
            }

            let result_json = match method.as_str() {
                crate::plugins::blobstore::PUT_REQUEST_OBJECT_METHOD => {
                    let request: crate::plugins::blobstore::PutRequestObjectRequest =
                        serde_json::from_str(&params_json)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .put_request_object(request)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&response)
                        .map_err(|err| Self::error_response(err.to_string()))?
                }
                crate::plugins::blobstore::GET_REQUEST_OBJECT_METHOD => {
                    let request: crate::plugins::blobstore::GetRequestObjectRequest =
                        serde_json::from_str(&params_json)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .get_request_object(request)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&response)
                        .map_err(|err| Self::error_response(err.to_string()))?
                }
                crate::plugins::blobstore::COMPLETE_REQUEST_METHOD => {
                    let request: crate::plugins::blobstore::FinishRequestRequest =
                        serde_json::from_str(&params_json)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .finish_request(&request.request_id)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&response)
                        .map_err(|err| Self::error_response(err.to_string()))?
                }
                crate::plugins::blobstore::ABORT_REQUEST_METHOD => {
                    let request: crate::plugins::blobstore::FinishRequestRequest =
                        serde_json::from_str(&params_json)
                            .map_err(|err| Self::error_response(err.to_string()))?;
                    let response = store
                        .finish_request(&request.request_id)
                        .map_err(|err| Self::error_response(err.to_string()))?;
                    serde_json::to_string(&response)
                        .map_err(|err| Self::error_response(err.to_string()))?
                }
                _ => {
                    return Err(Self::error_response(format!(
                        "Unsupported blobstore RPC '{}'",
                        method
                    )));
                }
            };

            Ok(plugin::RpcResult { result_json })
        })
    }

    fn handle_notification(
        &self,
        _plugin_name: String,
        _method: String,
        _params_json: String,
    ) -> plugin::BridgeFuture<()> {
        Box::pin(async {})
    }
}

fn temp_blobstore_root(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "mesh-llm-runtime-proxy-{name}-{}",
        rand::random::<u64>()
    ))
}

async fn start_blobstore_plugin_manager() -> (plugin::PluginManager, std::path::PathBuf) {
    start_blobstore_plugin_manager_for(
        plugin::BLOBSTORE_PLUGIN_ID,
        vec!["internal:blobstore".into(), "object-store.v1".into()],
    )
    .await
}

async fn start_blobstore_plugin_manager_for(
    plugin_name: &str,
    capabilities: Vec<String>,
) -> (plugin::PluginManager, std::path::PathBuf) {
    let root = temp_blobstore_root("blobstore");
    let bridge = BlobstoreTestBridge {
        plugin_name: plugin_name.to_string(),
        store: BlobStore::new(root.clone()),
    };
    let plugin_manager = plugin::PluginManager::for_test_bridge(&[plugin_name], Arc::new(bridge));
    let mut manifests = HashMap::new();
    manifests.insert(
        plugin_name.to_string(),
        mesh_llm_plugin::proto::PluginManifest {
            capabilities,
            ..Default::default()
        },
    );
    plugin_manager
        .set_test_manifests(manifests.into_iter().collect())
        .await;
    (plugin_manager, root)
}

async fn start_inference_endpoint_plugin_manager(
    address: String,
    models: Vec<String>,
) -> plugin::PluginManager {
    let plugin_manager = plugin::PluginManager::for_test_bridge(&[], Arc::new(NoopTestBridge));
    plugin_manager
        .set_test_inference_endpoints(vec![plugin::InferenceEndpointRoute {
            plugin_name: "endpoint-plugin".into(),
            endpoint_id: "endpoint-plugin".into(),
            address,
            models,
        }])
        .await;
    plugin_manager
}

async fn spawn_capturing_upstream(
    response_body: &str,
) -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
    spawn_status_upstream("200 OK", response_body).await
}

async fn spawn_status_upstream(
    status: &str,
    response_body: &str,
) -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let status = status.to_string();
    let response = response_body.to_string();
    let (request_tx, request_rx) = oneshot::channel();
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let raw = read_raw_http_request(&mut stream).await;
        let _ = request_tx.send(raw);

        let resp = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response.len(),
            response
        );
        stream.write_all(resp.as_bytes()).await.unwrap();
        let _ = stream.shutdown().await;
    });
    (port, request_rx, handle)
}

async fn spawn_held_upstream(
    response_body: &str,
) -> (
    u16,
    Arc<std::sync::atomic::AtomicUsize>,
    Arc<tokio::sync::Semaphore>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let response = response_body.to_string();
    let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let task_accepted = Arc::clone(&accepted);
    let task_release = Arc::clone(&release);
    let handle = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let response = response.clone();
            let accepted = Arc::clone(&task_accepted);
            let release = Arc::clone(&task_release);
            tokio::spawn(async move {
                let _raw = read_raw_http_request(&mut stream).await;
                accepted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let _permit = release.acquire().await.expect("release semaphore");
                let reply = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                );
                stream.write_all(reply.as_bytes()).await.unwrap();
                let _ = stream.shutdown().await;
            });
        }
    });
    (port, accepted, release, handle)
}

async fn spawn_streaming_upstream(
    content_type: &str,
    chunks: Vec<(Duration, Vec<u8>)>,
) -> (u16, oneshot::Receiver<Vec<u8>>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let content_type = content_type.to_string();
    let (request_tx, request_rx) = oneshot::channel();
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let raw = read_raw_http_request(&mut stream).await;
        let _ = request_tx.send(raw);

        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        );
        if stream.write_all(header.as_bytes()).await.is_err() {
            return;
        }

        for (delay, chunk) in chunks {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let chunk_header = format!("{:x}\r\n", chunk.len());
            if stream.write_all(chunk_header.as_bytes()).await.is_err() {
                return;
            }
            if stream.write_all(&chunk).await.is_err() {
                return;
            }
            if stream.write_all(b"\r\n").await.is_err() {
                return;
            }
        }

        let _ = stream.write_all(b"0\r\n\r\n").await;
        let _ = stream.shutdown().await;
    });
    (port, request_rx, handle)
}

async fn read_raw_http_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut raw = Vec::new();
    loop {
        let mut chunk = [0u8; 8192];
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "unexpected EOF while reading test request");
        raw.extend_from_slice(&chunk[..n]);

        let Some(header_end) = find_header_end(&raw) else {
            continue;
        };
        let headers = std::str::from_utf8(&raw[..header_end]).unwrap();

        if header_has_token(headers, "transfer-encoding", "chunked") {
            if raw[header_end..]
                .windows(5)
                .any(|window| window == b"0\r\n\r\n")
            {
                return raw;
            }
            continue;
        }

        if let Some(content_length) = content_length(headers) {
            if raw.len() >= header_end + content_length {
                raw.truncate(header_end + content_length);
                return raw;
            }
            continue;
        }

        raw.truncate(header_end);
        return raw;
    }
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|idx| idx + 4)
}

fn header_value<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines().skip(1).find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if key.trim().eq_ignore_ascii_case(name) {
            Some(value.trim())
        } else {
            None
        }
    })
}

fn header_has_token(headers: &str, name: &str, token: &str) -> bool {
    header_value(headers, name)
        .map(|value| {
            value
                .split(',')
                .any(|part| part.trim().eq_ignore_ascii_case(token))
        })
        .unwrap_or(false)
}

fn content_length(headers: &str) -> Option<usize> {
    header_value(headers, "content-length")?.parse().ok()
}

fn local_targets(entries: &[(&str, u16)]) -> election::ModelTargets {
    let mut targets = election::ModelTargets::default();
    targets.targets = entries
        .iter()
        .map(|(model, port)| {
            (
                (*model).to_string(),
                vec![election::InferenceTarget::Local(*port)],
            )
        })
        .collect::<HashMap<_, _>>();
    targets
}

fn unavailable_targets(models: &[&str]) -> election::ModelTargets {
    let mut targets = election::ModelTargets::default();
    targets.targets = models
        .iter()
        .map(|model| ((*model).to_string(), vec![election::InferenceTarget::None]))
        .collect();
    targets
}

fn single_model_targets(model: &str, ports: &[u16]) -> election::ModelTargets {
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        model.to_string(),
        ports
            .iter()
            .copied()
            .map(election::InferenceTarget::Local)
            .collect(),
    );
    targets
}

fn build_chunked_request(path: &str, body: &[u8], chunks: &[usize]) -> Vec<u8> {
    let mut out = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
    )
    .into_bytes();
    let mut pos = 0usize;
    for &chunk_len in chunks {
        let end = pos + chunk_len;
        out.extend_from_slice(format!("{chunk_len:x}\r\n").as_bytes());
        out.extend_from_slice(&body[pos..end]);
        out.extend_from_slice(b"\r\n");
        pos = end;
    }
    out.extend_from_slice(b"0\r\n\r\n");
    out
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

async fn read_until_contains(stream: &mut TcpStream, needle: &[u8], timeout: Duration) -> Vec<u8> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut response = Vec::new();
    while !contains_bytes(&response, needle) {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        assert!(
            !remaining.is_zero(),
            "timed out waiting for {:?} in response: {}",
            String::from_utf8_lossy(needle),
            String::from_utf8_lossy(&response)
        );
        let mut chunk = [0u8; 8192];
        let n = tokio::time::timeout(remaining, stream.read(&mut chunk))
            .await
            .expect("timed out waiting for response bytes")
            .unwrap();
        assert!(n > 0, "unexpected EOF while waiting for response bytes");
        response.extend_from_slice(&chunk[..n]);
    }
    response
}

async fn send_request_and_read_response(addr: SocketAddr, parts: Vec<Vec<u8>>) -> String {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    for part in parts {
        stream.write_all(&part).await.unwrap();
    }
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    String::from_utf8(response).unwrap()
}

include!("basic.rs");
include!("routing.rs");
