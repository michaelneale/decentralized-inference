use anyhow::{Context, Result};
use skippy_protocol::binary::{StageWireMessage, read_stage_message};
use std::io;
use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use super::stale_discard::StaleDiscardRegistry;

static BINARY_SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_connection_session_id() -> u64 {
    BINARY_SESSION_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Messages the reader may hold parsed ahead of execution. The coordinator
/// admits at most `MAX_VERIFY_WINDOW_PIPELINE_DEPTH` verify windows per
/// request and retires each one with a control message, so this covers the
/// whole admitted backlog by count.
///
/// A discard therefore usually overtakes the stale windows queued ahead of
/// it. It is not guaranteed to: `INBOUND_LOOKAHEAD_BYTES` binds first for
/// wide frames (a full 64-window backlog of `MAX_STAGE_FRAME_BYTES` frames is
/// far past the byte ceiling), and the reader then parks with the discard
/// still unread. That degrades to executing the stale tail — today's cost
/// without this path — rather than deadlocking, because the executor keeps
/// draining the queue.
pub(super) const INBOUND_LOOKAHEAD_MESSAGES: usize =
    2 * skippy_protocol::MAX_VERIFY_WINDOW_PIPELINE_DEPTH;

/// Byte ceiling for the same queue, and the per-connection bound on what a
/// misbehaving peer can make this process buffer: reading ahead moves frames
/// out of the kernel socket buffer into userspace, so the message count alone
/// bounds nothing useful for memory. A `DiscardStaleWindows` frame is ~100
/// bytes and overtakes a 32 MiB backlog exactly as reliably as a larger one,
/// so this is sized for the smallest backlog that preserves the property.
pub(super) const INBOUND_LOOKAHEAD_BYTES: usize = 32 * 1024 * 1024;

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
    receiver: Option<mpsc::Receiver<io::Result<StageWireMessage>>>,
    stream: TcpStream,
    thread: Option<thread::JoinHandle<()>>,
    queued_bytes: Arc<AtomicUsize>,
    stopped: Arc<AtomicBool>,
}

impl Drop for InboundMessageReader {
    fn drop(&mut self) {
        // Disconnect the channel first: a reader blocked in `send` on a full
        // lookahead queue is not woken by the socket shutdown.
        // Release a reader parked on the byte ceiling: it is waiting on the
        // executor, which is not coming back, and neither the receiver drop
        // nor the socket shutdown would wake it.
        self.stopped.store(true, Ordering::Release);
        drop(self.receiver.take());
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
    let queued_bytes = Arc::new(AtomicUsize::new(0));
    let reader_queued_bytes = queued_bytes.clone();
    let stopped = Arc::new(AtomicBool::new(false));
    let reader_stopped = stopped.clone();
    let thread = thread::spawn(move || {
        loop {
            // Back off while the parsed backlog is over the byte ceiling; the
            // executor decrements as it takes messages off the queue.
            while reader_queued_bytes.load(Ordering::Acquire) >= INBOUND_LOOKAHEAD_BYTES {
                if reader_stopped.load(Ordering::Acquire) {
                    return;
                }
                thread::sleep(Duration::from_millis(1));
            }
            match read_stage_message(&mut reader, activation_width) {
                Ok(message) => {
                    if message.kind.is_stale_window_discard() {
                        registry.record_message(&message);
                    }
                    let message_bytes = message.estimated_wire_bytes();
                    reader_queued_bytes.fetch_add(message_bytes, Ordering::AcqRel);
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
        receiver: Some(receiver),
        stream,
        thread: Some(thread),
        queued_bytes,
        stopped,
    })
}

impl InboundMessageReader {
    /// EOF classification: a clean EOF before any traffic is a normal
    /// connection close, anything else is an error.
    pub(super) fn next(
        &self,
        first_message: Option<StageWireMessage>,
        pending_prefill_replies: usize,
        observed_message_count: usize,
    ) -> Result<Option<StageWireMessage>> {
        if first_message.is_some() {
            return Ok(first_message);
        }
        let receiver = self
            .receiver
            .as_ref()
            .expect("inbound receiver present until drop");
        match receiver.recv() {
            Ok(Ok(message)) => {
                // Saturating, not a load-then-subtract: the counter must
                // never wrap, or the reader parks on the ceiling forever.
                let bytes = message.estimated_wire_bytes();
                self.queued_bytes
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                        Some(queued.saturating_sub(bytes))
                    })
                    .ok();
                Ok(Some(message))
            }
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
    fn dropping_the_reader_completes_while_it_is_parked_on_the_byte_ceiling() {
        let (mut peer, upstream) = connected_pair();
        let registry = Arc::new(StaleDiscardRegistry::default());
        let reader = spawn_message_reader(&upstream, 4, 1, registry).expect("spawn");

        // Park the reader: pretend the executor is holding the whole byte
        // budget, so the backoff loop is the only thing running.
        reader
            .queued_bytes
            .store(INBOUND_LOOKAHEAD_BYTES, Ordering::Release);
        write_stage_message(
            &mut peer,
            &control_message(WireMessageKind::Stop, Vec::new()),
            WireActivationDType::F32,
        )
        .expect("write");
        thread::sleep(Duration::from_millis(50));

        let (done, dropped) = mpsc::channel();
        thread::spawn(move || {
            drop(reader);
            let _ = done.send(());
        });
        dropped
            .recv_timeout(Duration::from_secs(5))
            .expect("a reader parked on the byte ceiling must still be released on drop");
        drop(peer);
    }

    #[test]
    fn dropping_the_reader_completes_while_the_lookahead_channel_is_full() {
        let (mut peer, upstream) = connected_pair();
        let registry = Arc::new(StaleDiscardRegistry::default());
        let reader = spawn_message_reader(&upstream, 4, 1, registry).expect("spawn");

        // Fill the lookahead queue and leave the reader blocked in `send`.
        for _ in 0..(INBOUND_LOOKAHEAD_MESSAGES + 4) {
            let message = control_message(WireMessageKind::Stop, Vec::new());
            write_stage_message(&mut peer, &message, WireActivationDType::F32).expect("write");
        }
        thread::sleep(Duration::from_millis(100));

        let (done, dropped) = mpsc::channel();
        thread::spawn(move || {
            drop(reader);
            let _ = done.send(());
        });
        dropped
            .recv_timeout(Duration::from_secs(5))
            .expect("dropping the receiver must unblock the queued send before the join");
        drop(peer);
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
