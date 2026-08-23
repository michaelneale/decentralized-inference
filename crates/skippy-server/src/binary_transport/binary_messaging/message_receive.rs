use anyhow::{Context, Result};
use skippy_protocol::binary::{StageWireMessage, read_stage_message};
use std::io;
use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;

use super::stale_discard::StaleDiscardRegistry;

static BINARY_SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_connection_session_id() -> u64 {
    BINARY_SESSION_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Messages the reader may hold parsed ahead of execution. The coordinator
/// admits at most `MAX_VERIFY_WINDOW_PIPELINE_DEPTH` verify windows per
/// request and retires each one with a control message, so this covers the
/// whole admitted backlog: the reader never blocks on a stale window while a
/// `DiscardStaleWindows` for it is still unread in the socket.
pub(super) const INBOUND_LOOKAHEAD_MESSAGES: usize =
    2 * skippy_protocol::MAX_VERIFY_WINDOW_PIPELINE_DEPTH;

/// Reads upstream messages on a dedicated thread so the executor can run a
/// buffered message while later ones are already parsed. This is what lets a
/// `DiscardStaleWindows` control message take effect before the buffered
/// stale verify windows behind it get executed: the reader records the
/// discard range in the shared registry the moment it reads the message.
///
/// Dropping the reader shuts the socket down and joins the thread, so a
/// handler that exits on a local error while the peer keeps its side open
/// does not leak a blocked thread and its cloned descriptor.
pub(super) struct InboundMessageReader {
    receiver: mpsc::Receiver<io::Result<StageWireMessage>>,
    stream: TcpStream,
    thread: Option<thread::JoinHandle<()>>,
}

impl Drop for InboundMessageReader {
    fn drop(&mut self) {
        // Unblock a pending `read_stage_message`; errors here only mean the
        // socket is already closed.
        let _ = self.stream.shutdown(Shutdown::Both);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub(super) fn spawn_message_reader(
    upstream: &TcpStream,
    activation_width: i32,
    capacity: usize,
    registry: Arc<StaleDiscardRegistry>,
) -> Result<InboundMessageReader> {
    let mut reader = upstream
        .try_clone()
        .context("clone upstream stream for inbound message reader")?;
    let stream = upstream
        .try_clone()
        .context("clone upstream stream for inbound reader shutdown")?;
    let (sender, receiver) = mpsc::sync_channel(capacity.max(INBOUND_LOOKAHEAD_MESSAGES));
    let thread = thread::spawn(move || {
        loop {
            match read_stage_message(&mut reader, activation_width) {
                Ok(message) => {
                    if message.kind.is_stale_window_discard() {
                        registry.record_message(&message);
                    }
                    if sender.send(Ok(message)).is_err() {
                        return;
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(error));
                    return;
                }
            }
        }
    });
    Ok(InboundMessageReader {
        receiver,
        stream,
        thread: Some(thread),
    })
}

impl InboundMessageReader {
    /// Mirrors `receive_next_message`'s EOF classification: a clean EOF before
    /// any traffic is a normal connection close, anything else is an error.
    pub(super) fn next(
        &self,
        first_message: Option<StageWireMessage>,
        pending_prefill_replies: usize,
        observed_message_count: usize,
    ) -> Result<Option<StageWireMessage>> {
        if first_message.is_some() {
            return Ok(first_message);
        }
        match self.receiver.recv() {
            Ok(Ok(message)) => Ok(Some(message)),
            Ok(Err(error))
                if error.kind() == io::ErrorKind::UnexpectedEof
                    && pending_prefill_replies == 0
                    && observed_message_count == 0 =>
            {
                Ok(None)
            }
            Ok(Err(error)) => Err(error).context("read binary stage message"),
            // The reader thread is gone without a final error; treat it as a
            // closed connection.
            Err(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::{
        StageStateHeader, WireActivationDType, WireMessageKind, write_stage_message,
    };
    use std::net::TcpListener;
    use std::time::{Duration, Instant};

    fn control_message(kind: WireMessageKind, tokens: Vec<i32>) -> StageWireMessage {
        StageWireMessage {
            kind,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(kind, WireActivationDType::F32),
            request_id: 7,
            session_id: 9,
            sampling: None,
            chat_sampling_metadata: None,
            tokens,
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn connected_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let address = listener.local_addr().expect("local addr");
        let client = TcpStream::connect(address).expect("connect");
        let (server, _) = listener.accept().expect("accept");
        (client, server)
    }

    #[test]
    fn discard_is_recorded_behind_a_backlog_larger_than_the_execution_queue() {
        let (mut peer, upstream) = connected_pair();
        let registry = Arc::new(StaleDiscardRegistry::default());
        // Execution queue of one; the reader must still look past a full
        // admitted backlog without anything being dequeued.
        let reader = spawn_message_reader(&upstream, 4, 1, registry.clone()).expect("spawn");

        for _ in 0..skippy_protocol::MAX_VERIFY_WINDOW_PIPELINE_DEPTH {
            let stale = control_message(WireMessageKind::Stop, Vec::new());
            write_stage_message(&mut peer, &stale, WireActivationDType::F32).expect("write");
        }
        let discard = control_message(WireMessageKind::DiscardStaleWindows, vec![3, 9]);
        write_stage_message(&mut peer, &discard, WireActivationDType::F32).expect("write");

        let deadline = Instant::now() + Duration::from_secs(5);
        while !registry.is_discarded(7, 9, 5) {
            assert!(
                Instant::now() < deadline,
                "discard must be recorded while the stale backlog is still queued"
            );
            thread::sleep(Duration::from_millis(5));
        }
        drop(reader);
    }

    #[test]
    fn dropping_the_reader_joins_the_thread_while_the_peer_stays_open() {
        let (peer, upstream) = connected_pair();
        let registry = Arc::new(StaleDiscardRegistry::default());
        let reader = spawn_message_reader(&upstream, 4, 1, registry).expect("spawn");

        let (done, dropped) = mpsc::channel();
        thread::spawn(move || {
            drop(reader);
            let _ = done.send(());
        });
        dropped
            .recv_timeout(Duration::from_secs(5))
            .expect("drop must shut the socket down and join the blocked reader thread");
        drop(peer);
    }
}
