use anyhow::{Context, Result};
use skippy_protocol::binary::{StageWireMessage, read_stage_message};
use std::io;
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;

use super::stale_discard::StaleDiscardRegistry;

static BINARY_SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_connection_session_id() -> u64 {
    BINARY_SESSION_COUNTER.fetch_add(1, Ordering::Relaxed)
}


/// Reads upstream messages on a dedicated thread so the executor can run a
/// buffered message while later ones are already parsed. This is what lets a
/// `DiscardStaleWindows` control message take effect before the buffered
/// stale verify windows behind it get executed: the reader records the
/// discard range in the shared registry the moment it reads the message.
pub(super) struct InboundMessageReader {
    receiver: mpsc::Receiver<io::Result<StageWireMessage>>,
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
    let (sender, receiver) = mpsc::sync_channel(capacity.max(1));
    thread::spawn(move || {
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
    Ok(InboundMessageReader { receiver })
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
