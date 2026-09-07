use crate::binary_transport::AsyncForwardReceipt;
use crate::binary_transport::AsyncForwarder;
use crate::binary_transport::BinaryStageExecutionOptions;
use crate::binary_transport::PredictionReturnReceiver;
use crate::binary_transport::forwarded_stage_message_timed;
use crate::binary_transport::run_binary_stage_message;
use crate::binary_transport::stage_output_activation_capacity;
use crate::binary_transport::write_stage_message_conditioned;
use crate::frontend::generation::EmbeddedExecutionStats;
use crate::frontend::generation::EmbeddedLocalOutput;
use crate::frontend::generation::EmbeddedStageExecution;
use crate::frontend::generation::EmbeddedStageZeroGeneration;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::stage_reply_timeout;
use crate::frontend::util::ms_to_us;
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::openai_io_error;
use crate::frontend::wire_messages::{discard_stale_windows_message, retire_verify_window_message};
use crate::telemetry::now_unix_nanos;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde_json::json;
use skippy_protocol::binary::StageReply;
use skippy_protocol::binary::StageReplyStats;
use skippy_protocol::binary::StageWireMessage;
use skippy_protocol::binary::WireMessageKind;
use skippy_protocol::binary::WireReplyKind;
use skippy_protocol::binary::recv_reply;
use std::net::TcpStream;
use std::time::Duration;
use std::time::Instant;

const DIRECT_RETURN_FALLBACK_POLL: Duration = Duration::from_millis(10);
// A dead downstream tunnel can leave both the persistent lane and the direct
// return reader open without producing EOF. Bound every reply wait so the
// request reaches lane replacement and session teardown instead of occupying
// a generation permit indefinitely. This is deliberately much larger than a
// normal WAN verify traversal while remaining shorter than the HTTP client's
// request timeout.

/// Identifies a contiguous stale verify-window range for one request.
pub(super) struct StaleWindowDiscard {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) min_window_id: i32,
    pub(super) max_window_id: i32,
}

pub(super) struct VerifyRetirement {
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) token_start: usize,
    pub(super) token_count: usize,
}

pub(super) struct DispatchedEmbeddedStage {
    started: Instant,
    stats: StageReplyStats,
    execution: EmbeddedExecutionStats,
    message_kind: WireMessageKind,
    token_count: i32,
    forward_receipt: Option<AsyncForwardReceipt>,
}

impl StageOpenAiBackend {
    pub(super) fn retire_verify_window(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        async_forwarder: Option<&mut AsyncForwarder>,
        session_key: &str,
        retirement: VerifyRetirement,
    ) -> OpenAiResult<()> {
        let scheduler_session_key = session_key.to_string();
        self.iteration_scheduler
            .execute_runtime("embedded-verify-retire", move |runtime| {
                runtime
                    .retire_verify_checkpoint(
                        &scheduler_session_key,
                        retirement.token_start as u64,
                        retirement.token_count as u64,
                    )
                    .map_err(openai_backend_error)
            })?;
        let message = retire_verify_window_message(
            retirement.request_id,
            retirement.session_id,
            retirement.token_start,
            retirement.token_count,
        )?;
        if let Some(forwarder) = async_forwarder {
            forwarder
                .send_tracked(
                    message,
                    request.downstream_wire_condition,
                    self.openai_attrs(request.ids),
                )
                .map_err(openai_backend_error)?
                .finish()
                .map_err(openai_backend_error)?;
        } else {
            write_stage_message_conditioned(
                downstream,
                &message,
                request.downstream_wire_condition,
            )
            .map_err(openai_io_error)?;
        }
        Ok(())
    }

    /// Sends a stale-window discard downstream without waiting for the
    /// write receipt: the message queues behind the already-dispatched stale
    /// windows, and blocking here would stall recovery for the whole stale
    /// tail's wire time.
    pub(super) fn discard_stale_windows(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        async_forwarder: Option<&mut AsyncForwarder>,
        discard: StaleWindowDiscard,
    ) -> OpenAiResult<()> {
        let message = discard_stale_windows_message(
            discard.request_id,
            discard.session_id,
            discard.min_window_id,
            discard.max_window_id,
        )?;
        if let Some(forwarder) = async_forwarder {
            forwarder
                .send(
                    message,
                    request.downstream_wire_condition,
                    self.openai_attrs(request.ids),
                )
                .map_err(openai_backend_error)?;
        } else {
            write_stage_message_conditioned(
                downstream,
                &message,
                request.downstream_wire_condition,
            )
            .map_err(openai_io_error)?;
        }
        Ok(())
    }

    pub(super) fn execute_embedded_stage_message(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
        expected_reply: WireReplyKind,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        let dispatched = self.dispatch_embedded_stage_message(
            request,
            downstream,
            session_key,
            message,
            token_ids,
            None,
        )?;
        self.complete_dispatched_stage_message(request, downstream, dispatched, expected_reply)
    }

    pub(super) fn dispatch_embedded_stage_message(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        message: &StageWireMessage,
        token_ids: &[i32],
        async_forwarder: Option<&mut AsyncForwarder>,
    ) -> OpenAiResult<DispatchedEmbeddedStage> {
        let started = Instant::now();
        let stats = StageReplyStats::default();
        let stage0_timer = PhaseTimer::start();
        let target_token_count = message.authoritative_session_position();
        let output_capacity = stage_output_activation_capacity(
            request.config,
            message.token_count,
            request.activation_width,
        )
        .map_err(openai_backend_error)?;
        let scheduler_session_key = session_key.to_string();
        let scheduler_message = message.clone();
        let scheduler_token_ids = token_ids.to_vec();
        let native_mtp_enabled = request.native_mtp_enabled;
        let native_mtp_max_tokens = request.speculative.native_mtp.max_draft_tokens;
        let scheduler_outcome = self.iteration_scheduler.execute_runtime_timed(
            "embedded-stage-execute",
            move |runtime| {
                let align = target_token_count
                    .map(|target_token_count| {
                        runtime
                            .align_session_to_token_count_if_ahead(
                                &scheduler_session_key,
                                target_token_count,
                            )
                            .map_err(openai_backend_error)
                    })
                    .transpose()?
                    .flatten();
                let output = run_binary_stage_message(
                    runtime,
                    &scheduler_session_key,
                    &scheduler_message,
                    &scheduler_token_ids,
                    None,
                    BinaryStageExecutionOptions::new(false, output_capacity, native_mtp_enabled)
                        .with_native_mtp_max_tokens(native_mtp_max_tokens),
                )
                .map_err(openai_backend_error)?
                .2;
                Ok((align, output))
            },
        )?;
        let (align, stage_output) = scheduler_outcome.value;
        if let Some(align) = align {
            let mut attrs = self.openai_attrs(request.ids);
            attrs.insert(
                "llama_stage.session_auto_align_before_tokens".to_string(),
                json!(align.before_token_count),
            );
            attrs.insert(
                "llama_stage.session_auto_align_after_tokens".to_string(),
                json!(align.after_token_count),
            );
            self.telemetry
                .emit_debug("stage.openai_session_auto_align", attrs);
        }
        let output = EmbeddedLocalOutput {
            output: stage_output,
            runtime_lock_wait_ms: scheduler_outcome.runtime_lock_wait_ms,
            runtime_lock_hold_ms: scheduler_outcome.runtime_lock_hold_ms,
        };
        let stage0_compute_ms = stage0_timer.elapsed_ms();
        if self.telemetry.is_debug_enabled() {
            let mut attrs = self.openai_attrs(request.ids);
            attrs.insert(
                "llama_stage.message_kind".to_string(),
                json!(format!("{:?}", message.kind)),
            );
            attrs.insert(
                "llama_stage.token_count".to_string(),
                json!(message.token_count),
            );
            if let Some(window_id) = message.verify_window_id() {
                attrs.insert("llama_stage.verify_window_id".to_string(), json!(window_id));
            }
            self.telemetry.emit_debug_span(
                "stage.openai_stage0_llama_decode",
                attrs,
                stage0_timer.start_unix_nanos,
                now_unix_nanos() as u64,
            );
        }
        let forwarded = forwarded_stage_message_timed(
            request.config,
            message,
            &output.output,
            request.activation_width,
        )
        .map_err(openai_backend_error)?;
        let forward_activation_bytes = forwarded.message.activation.len();
        let write_timer = PhaseTimer::start();
        let forward_receipt = if let Some(forwarder) = async_forwarder {
            Some(
                forwarder
                    .send_tracked(
                        forwarded.message,
                        request.downstream_wire_condition,
                        self.openai_attrs(request.ids),
                    )
                    .map_err(openai_backend_error)?,
            )
        } else {
            write_stage_message_conditioned(
                &mut *downstream,
                &forwarded.message,
                request.downstream_wire_condition,
            )
            .map_err(openai_io_error)?;
            None
        };
        let forward_write_ms = write_timer.elapsed_ms();
        Ok(DispatchedEmbeddedStage {
            started,
            stats,
            execution: EmbeddedExecutionStats {
                stage0_compute_ms,
                runtime_lock_wait_ms: output.runtime_lock_wait_ms,
                runtime_lock_hold_ms: output.runtime_lock_hold_ms,
                activation_encode_ms: forwarded.activation_encode_ms,
                output_activation_bytes: output.output.payload.len(),
                forward_activation_bytes,
                forward_write_ms,
                downstream_wait_ms: 0.0,
            },
            message_kind: message.kind,
            token_count: message.token_count,
            forward_receipt,
        })
    }

    pub(super) fn complete_dispatched_stage_message(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        dispatched: DispatchedEmbeddedStage,
        expected_reply: WireReplyKind,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        self.complete_dispatched_stage_message_with_return(
            request,
            downstream,
            dispatched,
            expected_reply,
            false,
        )
    }

    pub(super) fn complete_dispatched_stage_message_direct(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        dispatched: DispatchedEmbeddedStage,
        expected_reply: WireReplyKind,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        self.complete_dispatched_stage_message_with_return(
            request,
            downstream,
            dispatched,
            expected_reply,
            true,
        )
    }

    fn complete_dispatched_stage_message_with_return(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        mut dispatched: DispatchedEmbeddedStage,
        expected_reply: WireReplyKind,
        require_direct_return: bool,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        if let Some(receipt) = dispatched.forward_receipt.take() {
            dispatched.execution.forward_write_ms =
                receipt.finish().map_err(openai_backend_error)?;
        }
        let wait_timer = PhaseTimer::start();
        let reply = if require_direct_return {
            receive_direct_prediction_return(request.prediction_return.as_ref(), expected_reply)?
        } else {
            receive_embedded_stage_reply(
                downstream,
                request.prediction_return.as_ref(),
                expected_reply,
            )?
        };
        dispatched.execution.downstream_wait_ms = wait_timer.elapsed_ms();
        dispatched.stats.merge(reply.stats);
        if dispatched.message_kind == WireMessageKind::VerifyWindow {
            dispatched.stats.verify_window_compute_us +=
                ms_to_us(dispatched.execution.stage0_compute_ms);
            dispatched.stats.verify_window_forward_write_us +=
                ms_to_us(dispatched.execution.forward_write_ms);
            dispatched.stats.verify_window_downstream_wait_us +=
                ms_to_us(dispatched.execution.downstream_wait_ms);
            dispatched.stats.verify_window_total_us +=
                ms_to_us(dispatched.started.elapsed().as_secs_f64() * 1000.0);
            dispatched.stats.verify_window_stage_count += 1;
            dispatched.stats.verify_window_request_count += 1;
            dispatched.stats.verify_window_token_count += i64::from(dispatched.token_count.max(0));
            dispatched.stats.verify_window_max_tokens = dispatched
                .stats
                .verify_window_max_tokens
                .max(i64::from(dispatched.token_count.max(0)));
        }
        Ok(EmbeddedStageExecution {
            reply: StageReply {
                stats: dispatched.stats,
                ..reply
            },
            stats: dispatched.execution,
            elapsed_ms: dispatched.started.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

fn receive_direct_prediction_return(
    prediction_return: Option<&PredictionReturnReceiver>,
    expected_reply: WireReplyKind,
) -> OpenAiResult<StageReply> {
    let prediction_return = prediction_return.ok_or_else(|| {
        OpenAiError::backend("direct prediction return was required but is not configured")
    })?;
    prediction_return
        .recv_expected_timeout(expected_reply, stage_reply_timeout())
        .map_err(openai_backend_error)?
        .ok_or_else(|| {
            OpenAiError::backend(format!(
                "timed out waiting for {expected_reply:?} reply from direct prediction return"
            ))
        })
}

pub(crate) fn receive_embedded_stage_reply(
    downstream: &mut TcpStream,
    prediction_return: Option<&PredictionReturnReceiver>,
    expected_reply: WireReplyKind,
) -> OpenAiResult<StageReply> {
    receive_embedded_stage_reply_one_of(
        downstream,
        prediction_return,
        std::slice::from_ref(&expected_reply),
    )
}

pub(crate) fn receive_embedded_stage_reply_one_of(
    downstream: &mut TcpStream,
    prediction_return: Option<&PredictionReturnReceiver>,
    expected_replies: &[WireReplyKind],
) -> OpenAiResult<StageReply> {
    if expected_replies.is_empty() {
        return Err(OpenAiError::backend(
            "at least one expected stage reply kind is required",
        ));
    }
    let Some(prediction_return) = prediction_return else {
        return receive_downstream_stage_reply_one_of(downstream, expected_replies);
    };
    poll_direct_or_downstream_reply(downstream, prediction_return, expected_replies)
}

fn poll_direct_or_downstream_reply(
    downstream: &mut TcpStream,
    prediction_return: &PredictionReturnReceiver,
    expected_replies: &[WireReplyKind],
) -> OpenAiResult<StageReply> {
    poll_direct_or_downstream_reply_with_timeouts(
        downstream,
        prediction_return,
        expected_replies,
        DIRECT_RETURN_FALLBACK_POLL,
        DIRECT_RETURN_PEEK_TIMEOUT,
        stage_reply_timeout(),
    )
}

// The fallback poll and availability-peek intervals are injectable so tests can
// widen them: a wide fallback poll proves the wake is event-driven rather than
// quantized to the poll (without depending on sub-poll scheduler latency), and a
// wide peek interval proves the peek is bounded by the remaining reply deadline
// rather than overrunning it by a fixed interval.
fn poll_direct_or_downstream_reply_with_timeouts(
    downstream: &mut TcpStream,
    prediction_return: &PredictionReturnReceiver,
    expected_replies: &[WireReplyKind],
    fallback_poll: Duration,
    peek_timeout: Duration,
    reply_timeout: Duration,
) -> OpenAiResult<StageReply> {
    let mut timeout_restore = DirectReturnFallbackTimeout::install(downstream, peek_timeout)?;
    let started = Instant::now();
    let timeout_error = || {
        OpenAiError::backend(format!(
            "timed out waiting for one of {expected_replies:?} from direct return or downstream"
        ))
    };
    // The direct-return sink is the standard reply path, so block on its
    // channel and wake the moment the reply lands instead of sampling it
    // between bounded peeks: sampling quantized every reply wait to the poll
    // interval and cost a uniform 0..poll of added latency per token. The
    // tunnelled downstream fallback is checked without blocking each slice,
    // so a fallback reply is still detected within one poll interval.
    let result = loop {
        let remaining = reply_timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            break Err(timeout_error());
        }
        if let Some(reply) = prediction_return
            .recv_one_of_timeout(expected_replies, remaining.min(fallback_poll))
            .map_err(openai_backend_error)?
        {
            break Ok(reply);
        }
        // The channel wait above may have consumed the deadline; recompute the
        // budget and bound the availability peek by it so the whole wait honours
        // `reply_timeout` rather than overrunning by a fixed peek interval.
        let remaining = reply_timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            break Err(timeout_error());
        }
        downstream
            .set_read_timeout(Some(remaining.min(peek_timeout)))
            .map_err(openai_io_error)?;
        if downstream_reply_available(downstream)? {
            // `peek` only proves that the first byte has arrived. Tunnelled
            // replies may be fragmented, so decode the complete frame under
            // the remainder of the bounded fallback deadline rather than the
            // short read timeout used for the availability check.
            let remaining = reply_timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                break Err(timeout_error());
            }
            downstream
                .set_read_timeout(Some(remaining))
                .map_err(openai_io_error)?;
            break receive_downstream_stage_reply_one_of(downstream, expected_replies);
        }
    };
    timeout_restore.restore()?;
    result
}

fn downstream_reply_available(downstream: &TcpStream) -> OpenAiResult<bool> {
    let mut byte = [0u8; 1];
    match downstream.peek(&mut byte) {
        Ok(0) => Err(OpenAiError::backend("downstream closed before stage reply")),
        Ok(_) => Ok(true),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ) =>
        {
            Ok(false)
        }
        Err(error) => Err(openai_io_error(error)),
    }
}

/// Availability peeks run under a short read timeout rather than nonblocking
/// mode. `O_NONBLOCK` lives on the shared open-file description, so setting it
/// on this socket would also make writes on any `try_clone()` handle — such as
/// an `AsyncForwarder` worker forwarding activations to the next stage — fail
/// with `WouldBlock` once the send buffer fills. A read timeout (`SO_RCVTIMEO`)
/// affects receives only, so a concurrent cloned writer keeps blocking
/// semantics.
const DIRECT_RETURN_PEEK_TIMEOUT: Duration = Duration::from_millis(1);

struct DirectReturnFallbackTimeout {
    downstream: TcpStream,
    previous_timeout: Option<Duration>,
    restored: bool,
}

impl DirectReturnFallbackTimeout {
    /// Give the downstream socket a short read timeout so availability peeks
    /// return promptly while the reply wait blocks on the direct-return channel,
    /// without toggling nonblocking mode on the shared file description.
    fn install(downstream: &TcpStream, peek_timeout: Duration) -> OpenAiResult<Self> {
        let previous_timeout = downstream.read_timeout().map_err(openai_io_error)?;
        let restore_stream = downstream.try_clone().map_err(openai_io_error)?;
        downstream
            .set_read_timeout(Some(peek_timeout))
            .map_err(openai_io_error)?;
        Ok(Self {
            downstream: restore_stream,
            previous_timeout,
            restored: false,
        })
    }

    fn restore(&mut self) -> OpenAiResult<()> {
        self.downstream
            .set_read_timeout(self.previous_timeout)
            .map_err(openai_io_error)?;
        self.restored = true;
        Ok(())
    }
}

impl Drop for DirectReturnFallbackTimeout {
    fn drop(&mut self) {
        if !self.restored {
            let _ = self.downstream.set_read_timeout(self.previous_timeout);
        }
    }
}

fn receive_downstream_stage_reply_one_of(
    downstream: &mut TcpStream,
    expected_replies: &[WireReplyKind],
) -> OpenAiResult<StageReply> {
    let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
    if !expected_replies.contains(&reply.kind) {
        return Err(OpenAiError::backend(format!(
            "expected one of {expected_replies:?} from downstream, got {:?}",
            reply.kind
        )));
    }
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_transport::PredictionReturnHub;
    use skippy_protocol::binary::StageStateHeader;
    use std::net::TcpListener;
    use std::sync::Arc;

    #[test]
    fn embedded_stage_reply_accepts_fused_restore_hits_and_misses_from_direct_return() {
        assert_eq!(
            receive_direct_reply_one_of(
                WireReplyKind::PredictedToken,
                &[WireReplyKind::PredictedToken, WireReplyKind::Ack],
            ),
            WireReplyKind::PredictedToken
        );
        assert_eq!(
            receive_direct_reply_one_of(
                WireReplyKind::Ack,
                &[WireReplyKind::PredictedToken, WireReplyKind::Ack],
            ),
            WireReplyKind::Ack
        );
    }

    fn receive_direct_reply_one_of(
        reply_kind: WireReplyKind,
        expected_replies: &[WireReplyKind],
    ) -> WireReplyKind {
        let request_id = 17;
        let session_id = 23;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();
        let (mut direct_client, direct_server) = tcp_pair();
        let hub_thread = {
            let hub = hub.clone();
            std::thread::spawn(move || {
                hub.handle_return_connection(
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
                    },
                    direct_server,
                )
            })
        };
        skippy_protocol::binary::send_reply_message(
            &mut direct_client,
            &StageReply {
                kind: reply_kind,
                predicted: 0,
                predicted_tokens: Vec::new(),
                native_mtp_draft: None,
                window: Default::default(),
                stats: StageReplyStats::default(),
            },
        )
        .unwrap();
        let (mut downstream, _downstream_peer) = tcp_pair();
        let reply =
            receive_embedded_stage_reply_one_of(&mut downstream, Some(&receiver), expected_replies)
                .unwrap();
        drop(direct_client);
        hub_thread.join().unwrap().unwrap();
        reply.kind
    }

    fn tcp_pair() -> (TcpStream, TcpStream) {
        connected_stream_pair()
    }

    fn connected_stream_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        (client, server)
    }

    #[test]
    fn fallback_guard_keeps_a_cloned_writer_blocking() {
        use std::io::{Read, Write};
        // AsyncForwarder clones the downstream socket and writes frames from a
        // separate worker thread. The fallback-wait guard must not toggle
        // O_NONBLOCK on the shared open-file description, or those writes would
        // fail with WouldBlock once the send buffer fills instead of blocking.
        // A read timeout (SO_RCVTIMEO) affects receives only, so the writer is
        // unaffected. Regression for michaelneale's review on #1575.
        const PAYLOAD: usize = 8 * 1024 * 1024;
        let (downstream, peer) = connected_stream_pair();
        let writer = downstream.try_clone().unwrap();

        // Peer starts reading only after a delay, forcing the send buffer to
        // fill so a correct blocking write_all must wait rather than error.
        let reader = std::thread::spawn(move || {
            let mut peer = peer;
            std::thread::sleep(Duration::from_millis(100));
            let mut sink = vec![0u8; 1 << 16];
            let mut total = 0usize;
            while total < PAYLOAD {
                match peer.read(&mut sink) {
                    Ok(0) => break,
                    Ok(n) => total += n,
                    Err(error) => panic!("peer read failed: {error}"),
                }
            }
            total
        });

        let guard =
            DirectReturnFallbackTimeout::install(&downstream, DIRECT_RETURN_PEEK_TIMEOUT).unwrap();
        let mut writer = writer;
        // Must block until the peer drains, not fail with WouldBlock (os err 35).
        writer
            .write_all(&vec![0u8; PAYLOAD])
            .expect("cloned writer must keep blocking semantics under the guard");
        drop(guard);
        drop(writer);
        drop(downstream);
        assert_eq!(reader.join().unwrap(), PAYLOAD);
    }

    #[test]
    fn direct_return_fallback_timeout_restores_on_drop_after_early_exit() {
        let (stream, _peer) = connected_stream_pair();
        stream
            .set_read_timeout(Some(Duration::from_millis(123)))
            .unwrap();
        let effective_original = stream.read_timeout().unwrap();

        {
            let _restore =
                DirectReturnFallbackTimeout::install(&stream, DIRECT_RETURN_PEEK_TIMEOUT).unwrap();
            // The short read timeout makes an availability peek on a silent
            // socket return promptly instead of blocking out the original read
            // timeout. A timed-out peek surfaces as WouldBlock on Unix and
            // TimedOut on Windows.
            let started = Instant::now();
            let mut byte = [0u8; 1];
            let peek = stream.peek(&mut byte);
            assert!(matches!(
                peek,
                Err(ref error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    )
            ));
            assert!(started.elapsed() < Duration::from_millis(50));
        }

        // Drop restores blocking mode and the original read timeout.
        assert_eq!(stream.read_timeout().unwrap(), effective_original);
        let started = Instant::now();
        let mut byte = [0u8; 1];
        let peek = stream.peek(&mut byte);
        assert!(matches!(
            peek,
            Err(ref error) if matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut)
        ));
        assert!(started.elapsed() >= Duration::from_millis(100));
    }

    #[test]
    fn direct_return_reply_wakes_the_wait_immediately() {
        // Far wider than the production 10ms fallback poll so an event-driven
        // wake and a polling wake sit orders of magnitude apart.
        const WIDE_FALLBACK_POLL: Duration = Duration::from_millis(1000);
        let request_id = 71;
        let session_id = 73;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();
        let (mut downstream, _peer) = connected_stream_pair();

        // Feed replies through the real return-stream reader: the sink writer
        // half sends framed replies, the attached reader delivers them to the
        // hub channel the wait blocks on.
        let (sink_writer, sink_reader) = connected_stream_pair();
        receiver.attach_opened_stream(sink_reader);
        let sink_writer = Arc::new(std::sync::Mutex::new(sink_writer));

        let mut max_wake_ms = 0.0_f64;
        for _ in 0..5 {
            let writer = sink_writer.clone();
            let sent_at = Arc::new(std::sync::Mutex::new(None::<Instant>));
            let sent_at_writer = sent_at.clone();
            let sender = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                let mut stream = writer.lock().unwrap();
                *sent_at_writer.lock().unwrap() = Some(Instant::now());
                skippy_protocol::binary::send_reply_message(
                    &mut *stream,
                    &StageReply {
                        kind: WireReplyKind::PredictedToken,
                        predicted: 5,
                        predicted_tokens: vec![5],
                        native_mtp_draft: None,
                        window: Default::default(),
                        stats: StageReplyStats::default(),
                    },
                )
                .unwrap();
            });

            // Inject a deliberately wide fallback poll so the two behaviours are
            // far apart: a polling wait would take up to WIDE_FALLBACK_POLL to
            // notice the reply, while an event-driven wait wakes on channel
            // delivery. Asserting the wake lands well inside that interval proves
            // it is event-driven without pinning an absolute sub-poll latency
            // that a loaded CI runner cannot honour.
            let reply = poll_direct_or_downstream_reply_with_timeouts(
                &mut downstream,
                &receiver,
                &[WireReplyKind::PredictedToken],
                WIDE_FALLBACK_POLL,
                DIRECT_RETURN_PEEK_TIMEOUT,
                stage_reply_timeout(),
            )
            .unwrap();
            let received_at = Instant::now();
            sender.join().unwrap();
            assert_eq!(reply.predicted, 5);
            let wake_ms = received_at
                .duration_since(sent_at.lock().unwrap().unwrap())
                .as_secs_f64()
                * 1000.0;
            max_wake_ms = max_wake_ms.max(wake_ms);
        }
        // Event-driven wake fires on channel delivery; a polling wait would
        // quantize to WIDE_FALLBACK_POLL (1000ms). The 100ms bound sits an order
        // of magnitude below the poll yet far above channel-delivery + scheduler
        // latency on a loaded runner, so it fails a polling regression without
        // flaking on a correct implementation.
        assert!(
            max_wake_ms < 100.0,
            "direct return wake took {max_wake_ms:.2}ms with a {}ms fallback poll; reply wait is polling, not event-driven",
            WIDE_FALLBACK_POLL.as_millis()
        );
    }

    #[test]
    fn direct_return_fallback_wait_stays_within_reply_deadline() {
        const REPLY_TIMEOUT: Duration = Duration::from_millis(50);
        // A short fallback poll guarantees the loop reaches an availability peek
        // with time still on the clock, and a peek interval far larger than the
        // deadline means an unbounded peek would overrun by ~2s. Bounding the
        // peek by the remaining budget keeps the whole wait near REPLY_TIMEOUT.
        const SHORT_FALLBACK_POLL: Duration = Duration::from_millis(10);
        const WIDE_PEEK_POLL: Duration = Duration::from_secs(2);
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(81, 83).unwrap();
        let (mut downstream, _peer) = connected_stream_pair();

        let started = Instant::now();
        let error = poll_direct_or_downstream_reply_with_timeouts(
            &mut downstream,
            &receiver,
            &[WireReplyKind::PredictedToken],
            SHORT_FALLBACK_POLL,
            WIDE_PEEK_POLL,
            REPLY_TIMEOUT,
        )
        .unwrap_err();
        let elapsed = started.elapsed();

        assert!(error.to_string().contains("timed out waiting for one of"));
        assert!(elapsed >= REPLY_TIMEOUT);
        assert!(
            elapsed < Duration::from_millis(400),
            "reply wait took {elapsed:?} with a {REPLY_TIMEOUT:?} deadline and {WIDE_PEEK_POLL:?} peek interval; the availability peek is not bounded by the remaining budget"
        );
        assert_eq!(downstream.read_timeout().unwrap(), None);
    }

    #[test]
    fn direct_return_fallback_accepts_fragmented_downstream_reply() {
        use std::io::Write;

        let request_id = 91;
        let session_id = 92;
        let hub = Arc::new(PredictionReturnHub::default());
        let receiver = hub.register(request_id, session_id).unwrap();
        let (mut downstream, mut downstream_peer) = connected_stream_pair();
        let mut bytes = Vec::new();
        skippy_protocol::binary::send_reply_message(
            &mut bytes,
            &StageReply {
                kind: WireReplyKind::PredictedTokens,
                predicted: 0,
                predicted_tokens: vec![17, 23],
                native_mtp_draft: None,
                window: Default::default(),
                stats: StageReplyStats::default(),
            },
        )
        .unwrap();
        let writer = std::thread::spawn(move || {
            downstream_peer.write_all(&bytes[..1]).unwrap();
            downstream_peer.flush().unwrap();
            std::thread::sleep(DIRECT_RETURN_FALLBACK_POLL * 3);
            downstream_peer.write_all(&bytes[1..]).unwrap();
        });

        let reply = receive_embedded_stage_reply_one_of(
            &mut downstream,
            Some(&receiver),
            &[WireReplyKind::PredictedTokens],
        )
        .unwrap();

        assert_eq!(reply.predicted_tokens, vec![17, 23]);
        assert_eq!(downstream.read_timeout().unwrap(), None);
        writer.join().unwrap();
    }
}
