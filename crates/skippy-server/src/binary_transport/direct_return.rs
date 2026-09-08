use std::{
    collections::HashMap,
    io::{self, Read},
    net::{IpAddr, SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
        mpsc::RecvTimeoutError,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::{
    StageConfig, StageTopology,
    binary::{
        STAGE_WIRE_FIXED_HEADER_BYTES, StageReply, StageStateHeader, StageWireMessage,
        WireMessageKind, WireReplyKind, read_stage_message, recv_ready, recv_reply, send_ready,
        send_reply_message, write_stage_message,
    },
};

use super::socket::{connect_downstream_socket, downstream_source_ip, resolve_downstream_endpoint};
use super::stage_execution::{
    consume_optional_client_ready_hello, send_client_ready_hello_if_enabled,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PredictionReturnKey {
    request_id: u64,
    session_id: u64,
}

impl PredictionReturnKey {
    pub(crate) fn new(request_id: u64, session_id: u64) -> Self {
        Self {
            request_id,
            session_id,
        }
    }
}

pub struct PredictionReturnHub {
    waiters: Mutex<HashMap<PredictionReturnKey, mpsc::Sender<Result<StageReply, String>>>>,
}

// Return sinks normally wait only until the matching generation reaches the
// final stage. Bound unmatched opens so they cannot retain sockets indefinitely
// without limit; a rejected preferred sink uses the existing reverse fallback.
const MAX_PENDING_PREDICTION_RETURN_SINKS: usize = 64;

#[derive(Default)]
pub(crate) struct PredictionReturnSinks {
    streams: Mutex<HashMap<PredictionReturnKey, TcpStream>>,
}

impl Default for PredictionReturnHub {
    fn default() -> Self {
        Self {
            waiters: Mutex::new(HashMap::new()),
        }
    }
}

pub struct PredictionReturnListener {
    shutdown: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    hub: Arc<PredictionReturnHub>,
}

impl PredictionReturnListener {
    pub fn start(bind_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr)
            .with_context(|| format!("bind direct prediction return listener {bind_addr}"))?;
        listener
            .set_nonblocking(true)
            .context("set direct prediction return listener nonblocking")?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_shutdown = shutdown.clone();
        let hub = Arc::new(PredictionReturnHub::default());
        let thread_hub = hub.clone();
        let thread = thread::spawn(move || {
            while !thread_shutdown.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        if let Err(error) = stream.set_nonblocking(false) {
                            eprintln!(
                                "direct prediction return connection failed: set blocking: {error}"
                            );
                            continue;
                        }
                        let hub = thread_hub.clone();
                        thread::spawn(move || {
                            if let Err(error) = handle_prediction_return_connection(hub, stream) {
                                eprintln!("direct prediction return connection failed: {error:#}");
                            }
                        });
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(error) => {
                        eprintln!("direct prediction return listener failed: {error}");
                        break;
                    }
                }
            }
        });
        Ok(Self {
            shutdown,
            thread: Some(thread),
            hub,
        })
    }

    pub fn hub(&self) -> Arc<PredictionReturnHub> {
        self.hub.clone()
    }
}

impl Drop for PredictionReturnListener {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn handle_prediction_return_connection(
    hub: Arc<PredictionReturnHub>,
    mut stream: TcpStream,
) -> Result<()> {
    consume_optional_client_ready_hello(&mut stream)
        .context("consume optional direct prediction return client ready hello")?;
    send_ready(&mut stream).context("send direct prediction return ready")?;
    let open = read_prediction_return_open(&mut stream)?;
    hub.handle_return_connection(open, stream)
}

fn read_prediction_return_open(stream: &mut TcpStream) -> Result<StageWireMessage> {
    let mut header = [0_u8; STAGE_WIRE_FIXED_HEADER_BYTES];
    stream
        .read_exact(&mut header)
        .context("read direct prediction return open header")?;
    let read_i32 = |offset: usize| {
        let mut bytes = [0_u8; 4];
        bytes.copy_from_slice(&header[offset..offset + 4]);
        i32::from_le_bytes(bytes)
    };

    let kind = WireMessageKind::try_from(read_i32(0))
        .context("parse direct prediction return open kind")?;
    if kind != WireMessageKind::PredictionReturnOpen {
        bail!("expected prediction return open message");
    }

    // Prediction-return opens are fixed routing headers. Reject fields that
    // would make the generic decoder read or allocate a variable-length body.
    if [4, 8, 12, 16]
        .into_iter()
        .any(|offset| read_i32(offset) != 0)
    {
        bail!("noncanonical prediction return open message");
    }

    let open = read_stage_message(io::Cursor::new(header), 0)
        .context("parse direct prediction return open")?;
    if open.state != StageStateHeader::new(kind) {
        bail!("noncanonical prediction return open message");
    }
    Ok(open)
}

impl PredictionReturnHub {
    pub(crate) fn register(
        self: &Arc<Self>,
        request_id: u64,
        session_id: u64,
    ) -> Result<PredictionReturnReceiver> {
        let key = PredictionReturnKey::new(request_id, session_id);
        let (sender, receiver) = mpsc::channel();
        self.waiters
            .lock()
            .map_err(|_| anyhow!("prediction return hub lock poisoned"))?
            .insert(key, sender);
        Ok(PredictionReturnReceiver {
            key,
            hub: self.clone(),
            receiver,
        })
    }

    pub(crate) fn unregister(&self, key: PredictionReturnKey) {
        if let Ok(mut waiters) = self.waiters.lock() {
            waiters.remove(&key);
        }
    }

    pub(crate) fn handle_return_connection(
        &self,
        open: StageWireMessage,
        stream: TcpStream,
    ) -> Result<()> {
        if open.kind != WireMessageKind::PredictionReturnOpen {
            bail!("expected prediction return open message");
        }
        let key = PredictionReturnKey::new(open.request_id, open.session_id);
        self.handle_return_stream(key, stream)
    }

    fn handle_return_stream(&self, key: PredictionReturnKey, mut stream: TcpStream) -> Result<()> {
        let sender = self
            .waiters
            .lock()
            .map_err(|_| anyhow!("prediction return hub lock poisoned"))?
            .get(&key)
            .cloned()
            .ok_or_else(|| anyhow!("no prediction return waiter for request {}", key.request_id))?;
        loop {
            match recv_reply(&mut stream) {
                Ok(reply) => {
                    if sender.send(Ok(reply)).is_err() {
                        return Ok(());
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                    let _ = sender.send(Err(format!(
                        "direct prediction return closed before the next reply: {error}"
                    )));
                    return Ok(());
                }
                Err(error) => {
                    let _ = sender.send(Err(error.to_string()));
                    return Err(error).context("read direct prediction return");
                }
            }
        }
    }
}

pub(crate) struct PredictionReturnReceiver {
    key: PredictionReturnKey,
    hub: Arc<PredictionReturnHub>,
    receiver: mpsc::Receiver<Result<StageReply, String>>,
}

impl PredictionReturnReceiver {
    pub(crate) fn attach_opened_stream(&self, stream: TcpStream) {
        let hub = self.hub.clone();
        let key = self.key;
        thread::spawn(move || {
            if let Err(error) = hub.handle_return_stream(key, stream) {
                eprintln!("direct prediction return reader failed: {error:#}");
            }
        });
    }

    pub(crate) fn recv_expected_timeout(
        &self,
        expected: WireReplyKind,
        timeout: Duration,
    ) -> Result<Option<StageReply>> {
        let reply = match self.receiver.recv_timeout(timeout) {
            Ok(Ok(reply)) => reply,
            Ok(Err(error)) => return Err(anyhow!(error)),
            Err(RecvTimeoutError::Timeout) => return Ok(None),
            Err(RecvTimeoutError::Disconnected) => {
                return Err(anyhow!("prediction return channel disconnected"));
            }
        };
        validate_expected_reply(reply, std::slice::from_ref(&expected)).map(Some)
    }

    /// Event-driven bounded wait: blocks on the return channel and wakes the
    /// moment a reply is delivered. Returns `None` on timeout.
    pub(crate) fn recv_one_of_timeout(
        &self,
        expected: &[WireReplyKind],
        timeout: Duration,
    ) -> Result<Option<StageReply>> {
        let reply = match self.receiver.recv_timeout(timeout) {
            Ok(Ok(reply)) => reply,
            Ok(Err(error)) => return Err(anyhow!(error)),
            Err(RecvTimeoutError::Timeout) => return Ok(None),
            Err(RecvTimeoutError::Disconnected) => {
                return Err(anyhow!("prediction return channel disconnected"));
            }
        };
        validate_expected_reply(reply, expected).map(Some)
    }
}

fn validate_expected_reply(reply: StageReply, expected: &[WireReplyKind]) -> Result<StageReply> {
    if !expected.contains(&reply.kind) {
        bail!(
            "expected one of {expected:?} from direct prediction return, got {:?}",
            reply.kind
        );
    }
    Ok(reply)
}

impl Drop for PredictionReturnReceiver {
    fn drop(&mut self) {
        self.hub.unregister(self.key);
    }
}

impl PredictionReturnSinks {
    pub(crate) fn insert_opened_sink(
        &self,
        open: StageWireMessage,
        stream: TcpStream,
    ) -> Result<()> {
        if open.kind != WireMessageKind::PredictionReturnOpen {
            bail!("expected prediction return open message");
        }
        let key = PredictionReturnKey::new(open.request_id, open.session_id);
        let mut streams = self
            .streams
            .lock()
            .map_err(|_| anyhow!("prediction return sinks lock poisoned"))?;
        if !streams.contains_key(&key) && streams.len() >= MAX_PENDING_PREDICTION_RETURN_SINKS {
            bail!("too many pending prediction return sinks");
        }
        streams.insert(key, stream);
        Ok(())
    }

    pub(crate) fn take_wait(
        &self,
        request_id: u64,
        session_id: u64,
        timeout: Duration,
    ) -> Result<Option<TcpStream>> {
        let key = PredictionReturnKey::new(request_id, session_id);
        let started = std::time::Instant::now();
        loop {
            if let Some(stream) = self
                .streams
                .lock()
                .map_err(|_| anyhow!("prediction return sinks lock poisoned"))?
                .remove(&key)
            {
                return Ok(Some(stream));
            }
            if started.elapsed() >= timeout {
                return Ok(None);
            }
            thread::sleep(Duration::from_millis(2));
        }
    }

    pub(crate) fn remove(&self, request_id: u64, session_id: u64) {
        let key = PredictionReturnKey::new(request_id, session_id);
        if let Ok(mut streams) = self.streams.lock() {
            streams.remove(&key);
        }
    }
}

/// Read timeout for the return-sink ready handshake. `recv_ready` is a blocking
/// `read_exact`; without this a stalled downstream connection hangs the open
/// forever, which mid-generation blocks the request from ever falling back to
/// the upstream reply. Cleared afterwards so the sink's normal reads stay
/// blocking.
///
/// Budget sizing (20s): over a WAN mesh the return sink connects to a LOCAL
/// bridge alias, but the remote `ready` byte only arrives after the bridge
/// COLD-establishes a fresh stage QUIC connection (up to ~10s) and the remote
/// inbound handler then dials its local binary server. A 5s budget timed out
/// during that cold setup (observed EAGAIN on a healthy ~26ms WAN split), even
/// though the pooled forward lanes — which get a 20s initial connect budget and
/// are then reused — succeeded on the same bridge. Matching the forward-lane
/// budget lets the cold return path complete instead of failing to the slower
/// upstream-reply fallback.
///
/// This is a *single bounded deadline*, not a retry budget: the sink is opened
/// on the generation hot path, and `connect_downstream_socket` already bounds
/// the connect itself, so wrapping this in an outer retry only compounds the
/// worst-case stall (see PR #1011 review).
const RETURN_SINK_READY_READ_TIMEOUT: Duration = Duration::from_secs(20);

/// Connect to `return_addr`, complete the ready handshake, and send the
/// prediction-return open message. Single bounded attempt — on failure the
/// caller falls back to the upstream reply path.
fn open_return_sink_once(
    return_addr: SocketAddr,
    source_ip: Option<IpAddr>,
    request_id: u64,
    session_id: u64,
    not_ready_context: &'static str,
) -> Result<TcpStream> {
    let mut stream = connect_downstream_socket(return_addr, source_ip, Duration::from_secs(2))
        .map_err(|error| anyhow!(error))?;
    stream.set_nodelay(true).ok();
    send_client_ready_hello_if_enabled(&mut stream)
        .context("send prediction return client ready hello")?;
    // Bound the ready handshake read. `recv_ready` is a blocking `read_exact`;
    // without a timeout a stalled downstream connection hangs the return-sink
    // open forever, blocking generation from falling back to the upstream reply.
    // A single short deadline (no outer retry) fails fast to that fallback.
    // Both the set and the clear are propagated: if the set fails, `recv_ready`
    // would be unbounded (defeating the fix); if the clear fails, the handshake
    // timeout would leak into the sink's later reads.
    stream
        .set_read_timeout(Some(RETURN_SINK_READY_READ_TIMEOUT))
        .context("set prediction return ready read timeout")?;
    let ready = recv_ready(&mut stream).context(not_ready_context);
    stream
        .set_read_timeout(None)
        .context("clear prediction return ready read timeout")?;
    ready?;
    write_stage_message(
        &mut stream,
        &prediction_return_open_message(request_id, session_id),
    )
    .context("open prediction return stream")?;
    Ok(stream)
}

pub(crate) fn open_prediction_return_stream(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    request_id: u64,
    session_id: u64,
    _timeout_secs: u64,
) -> Result<TcpStream> {
    let endpoint = driver_stage_endpoint(config, topology)?;
    let source_ip = downstream_source_ip(config)?;
    let return_addr = resolve_downstream_endpoint(endpoint, source_ip)?;
    open_return_sink_once(
        return_addr,
        source_ip,
        request_id,
        session_id,
        "prediction return sink did not become ready",
    )
    .with_context(|| format!("connect direct prediction return sink at {endpoint}"))
}

pub(crate) fn open_downstream_prediction_return_stream(
    config: &StageConfig,
    request_id: u64,
    session_id: u64,
) -> Result<TcpStream> {
    let downstream = config
        .downstream
        .as_ref()
        .ok_or_else(|| anyhow!("direct prediction return requires downstream stage"))?;
    let endpoint = strip_tcp_prefix(&downstream.endpoint);
    let source_ip = downstream_source_ip(config)?;
    let return_addr = resolve_downstream_endpoint(endpoint, source_ip)?;
    open_return_sink_once(
        return_addr,
        source_ip,
        request_id,
        session_id,
        "downstream prediction return sink did not become ready",
    )
    .with_context(|| format!("connect downstream prediction return sink at {endpoint}"))
}

pub(crate) fn send_direct_prediction_return(
    stream: &mut TcpStream,
    reply: StageReply,
) -> Result<()> {
    send_reply_message(stream, &reply).context("send direct prediction return")
}

fn driver_stage_endpoint<'a>(
    config: &'a StageConfig,
    topology: Option<&'a StageTopology>,
) -> Result<&'a str> {
    if let Some(topology) = topology {
        return driver_stage_endpoint_from_topology(topology);
    }
    if let Some(upstream) = config
        .upstream
        .as_ref()
        .filter(|upstream| upstream.stage_index == 0)
    {
        return Ok(strip_tcp_prefix(&upstream.endpoint));
    }
    Err(anyhow!("direct prediction return requires topology"))
}

fn driver_stage_endpoint_from_topology(topology: &StageTopology) -> Result<&str> {
    topology
        .stages
        .iter()
        .find(|stage| stage.stage_index == 0)
        .map(|stage| strip_tcp_prefix(&stage.endpoint))
        .ok_or_else(|| anyhow!("topology does not contain driver-facing stage 0"))
}

fn strip_tcp_prefix(endpoint: &str) -> &str {
    endpoint.strip_prefix("tcp://").unwrap_or(endpoint)
}

fn prediction_return_open_message(request_id: u64, session_id: u64) -> StageWireMessage {
    StageWireMessage {
        kind: WireMessageKind::PredictionReturnOpen,
        pos_start: 0,
        token_count: 0,
        state: StageStateHeader::new(WireMessageKind::PredictionReturnOpen),
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::{
        recv_ready, recv_reply, send_reply_predicted_with_stats, state_flags,
    };
    use std::io::Write;

    #[test]
    fn handle_return_connection_delivers_reply_to_registered_waiter() {
        let request_id = 17;
        let session_id = 23;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        let open = prediction_return_open_message(request_id, session_id);
        let handle = {
            let hub = hub.clone();
            thread::spawn(move || hub.handle_return_connection(open, server))
        };

        send_reply_predicted_with_stats(&mut client, 42, Default::default()).unwrap();

        let reply = receiver
            .recv_expected_timeout(WireReplyKind::PredictedToken, Duration::from_secs(1))
            .unwrap()
            .expect("prediction return reply");
        assert_eq!(reply.predicted, 42);
        drop(client);
        handle.join().unwrap().unwrap();
    }

    #[test]
    fn prediction_return_open_rejects_metadata_before_reading_its_body() {
        let hub = Arc::new(PredictionReturnHub::default());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let handle = thread::spawn(move || {
            let _ = result_tx.send(handle_prediction_return_connection(hub, server));
        });

        recv_ready(&mut client).unwrap();

        let mut header = Vec::new();
        write_stage_message(&mut header, &prediction_return_open_message(1, 2)).unwrap();
        assert_eq!(header.len(), STAGE_WIRE_FIXED_HEADER_BYTES);
        let state_flags_offset = 5 * 4 + 3 * 4;
        header[state_flags_offset..state_flags_offset + 4]
            .copy_from_slice(&state_flags::CHAT_SAMPLING_METADATA.to_le_bytes());
        client.write_all(&header).unwrap();
        client.flush().unwrap();

        let result = result_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("return open must be rejected before its metadata body is read");
        assert!(result.is_err());
        drop(client);
        handle.join().unwrap();
    }

    #[test]
    fn blocking_prediction_return_receive_times_out_without_polling() {
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(53, 59).unwrap();
        let started = std::time::Instant::now();

        assert!(
            receiver
                .recv_expected_timeout(WireReplyKind::PredictedTokens, Duration::from_millis(10),)
                .unwrap()
                .is_none()
        );
        assert!(started.elapsed() >= Duration::from_millis(8));
    }

    #[test]
    fn closed_prediction_return_wakes_waiter_with_error() {
        let request_id = 61;
        let session_id = 67;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();
        let listener = TcpListener::bind("localhost:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let open = prediction_return_open_message(request_id, session_id);
        let handle = {
            let hub = hub.clone();
            thread::spawn(move || hub.handle_return_connection(open, server))
        };

        drop(client);

        let error = receiver
            .recv_expected_timeout(WireReplyKind::PredictedToken, Duration::from_secs(1))
            .expect_err("closed prediction return must wake the waiter");
        assert!(error.to_string().contains("closed before the next reply"));
        handle.join().unwrap().unwrap();
    }

    #[test]
    fn direct_prediction_return_preserves_typed_native_mtp_draft() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();

        let reply = StageReply {
            kind: WireReplyKind::PredictedToken,
            predicted: 42,
            predicted_tokens: vec![42],
            native_mtp_draft: Some(skippy_protocol::binary::StageNativeMtpDraft {
                token_ids: vec![43],
                proposal_compute_us: 123,
            }),
            window: skippy_protocol::binary::StageReplyWindow { window_id: 7 },
            stats: Default::default(),
        };
        send_direct_prediction_return(&mut server, reply).unwrap();

        let received = recv_reply(&mut client).unwrap();
        assert_eq!(received.kind, WireReplyKind::PredictedToken);
        assert_eq!(received.predicted, 42);
        assert_eq!(received.predicted_tokens, vec![42]);
        assert_eq!(
            received.native_mtp_draft,
            Some(skippy_protocol::binary::StageNativeMtpDraft {
                token_ids: vec![43],
                proposal_compute_us: 123,
            })
        );
        assert_eq!(received.window.window_id, 7);
    }

    #[test]
    fn prediction_return_sinks_store_upstream_opened_streams() {
        let request_id = 31;
        let session_id = 37;
        let sinks = PredictionReturnSinks::default();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();

        sinks
            .insert_opened_sink(
                prediction_return_open_message(request_id, session_id),
                server,
            )
            .unwrap();

        let stream = sinks
            .take_wait(request_id, session_id, Duration::from_millis(1))
            .unwrap()
            .expect("registered prediction return sink");
        assert_eq!(stream.peer_addr().unwrap(), client.local_addr().unwrap());
    }

    #[test]
    fn prediction_return_sinks_enforce_and_release_the_pending_limit() {
        let sinks = PredictionReturnSinks::default();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();

        for request_id in 0..MAX_PENDING_PREDICTION_RETURN_SINKS as u64 {
            let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let (server, _) = listener.accept().unwrap();
            sinks
                .insert_opened_sink(prediction_return_open_message(request_id, 1), server)
                .unwrap();
            drop(client);
        }

        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let error = sinks
            .insert_opened_sink(prediction_return_open_message(u64::MAX, 1), server)
            .expect_err("pending prediction return sink limit must be enforced");
        assert!(error.to_string().contains("too many pending"));
        assert_eq!(
            sinks.streams.lock().unwrap().len(),
            MAX_PENDING_PREDICTION_RETURN_SINKS
        );
        drop(client);

        sinks.remove(0, 1);
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        sinks
            .insert_opened_sink(prediction_return_open_message(u64::MAX, 1), server)
            .expect("released capacity must admit the next sink");
        assert_eq!(
            sinks.streams.lock().unwrap().len(),
            MAX_PENDING_PREDICTION_RETURN_SINKS
        );
        drop(client);
    }

    #[test]
    fn prediction_return_sinks_remove_abandoned_streams() {
        let request_id = 41;
        let session_id = 43;
        let sinks = PredictionReturnSinks::default();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();

        sinks
            .insert_opened_sink(
                prediction_return_open_message(request_id, session_id),
                server,
            )
            .unwrap();
        sinks.remove(request_id, session_id);

        assert!(
            sinks
                .take_wait(request_id, session_id, Duration::from_millis(1))
                .unwrap()
                .is_none()
        );
        drop(client);
    }
}
