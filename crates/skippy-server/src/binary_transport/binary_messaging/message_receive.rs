use super::ConnectionWorkerControl;
use anyhow::{Context, Result};
use skippy_protocol::{
    StageConfig,
    binary::{StageWireMessage, read_stage_message_for_codec_policy},
};
use std::io;
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};

static BINARY_SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

pub(super) fn next_connection_session_id() -> u64 {
    BINARY_SESSION_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub(super) fn receive_next_message(
    upstream: &mut TcpStream,
    worker_control: &ConnectionWorkerControl,
    activation_width: i32,
    config: &StageConfig,
    first_message: Option<StageWireMessage>,
    pending_prefill_replies: usize,
    observed_message_count: usize,
) -> Result<Option<StageWireMessage>> {
    if first_message.is_some() {
        return Ok(first_message);
    }
    if !worker_control
        .wait_for_readable(upstream)
        .context("wait for the next binary stage message")?
    {
        // Shutdown was requested while the connection was idle: end the
        // worker cleanly instead of blocking in a read that a socket
        // shutdown cannot interrupt on Windows (#1538).
        return Ok(None);
    }
    match read_stage_message_for_codec_policy(
        upstream,
        activation_width,
        config.activation_codec,
        config.activation_codec_policy,
    ) {
        Ok(message) => Ok(Some(message)),
        Err(error)
            if error.kind() == io::ErrorKind::UnexpectedEof
                && pending_prefill_replies == 0
                && observed_message_count == 0 =>
        {
            Ok(None)
        }
        Err(error) => Err(error).context("read binary stage message"),
    }
}
