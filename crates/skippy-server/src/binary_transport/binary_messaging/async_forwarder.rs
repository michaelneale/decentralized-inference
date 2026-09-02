use crate::binary_transport::WireCondition;
use crate::binary_transport::stage_execution::elapsed_ms;
use crate::binary_transport::write_stage_message_after_propagation;
use crate::telemetry::Telemetry;
use crate::telemetry::now_unix_nanos;
use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use serde_json::Value;
use serde_json::json;
use skippy_protocol::binary::StageWireMessage;
use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::net::TcpStream;
use std::sync::mpsc;
use std::sync::mpsc::RecvTimeoutError;
use std::sync::mpsc::TryRecvError;
use std::thread;
use std::time::Duration;
use std::time::Instant;

const ASYNC_FORWARD_TERMINAL_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) struct AsyncForwarder {
    sender: Option<mpsc::SyncSender<AsyncForwardJob>>,
    pending: VecDeque<AsyncForwardReceipt>,
    writer: Option<thread::JoinHandle<()>>,
}

impl Drop for AsyncForwarder {
    /// Queued frames must not still be on the wire after the request that
    /// owns them returns: a persistent lane is handed back for reuse, and a
    /// teardown `Stop` written through another clone of the same socket
    /// would interleave with a frame this forwarder is still writing.
    /// Dropping the sender ends the writer loop once its queue drains, and
    /// the join makes that ordering observable to the caller.
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

pub(crate) struct AsyncForwardReceipt {
    receiver: mpsc::Receiver<Result<f64, String>>,
}

struct AsyncForwardJob {
    message: StageWireMessage,
    condition: WireCondition,
    attrs: BTreeMap<String, Value>,
    done: mpsc::Sender<Result<f64, String>>,
    enqueued_at: Instant,
    enqueued_unix_nanos: u64,
}

impl AsyncForwarder {
    pub(crate) fn new(
        downstream: &TcpStream,
        telemetry: Telemetry,
        queue_capacity: usize,
    ) -> Result<Self> {
        let mut writer = downstream
            .try_clone()
            .context("clone downstream stream for async activation forwarding")?;
        writer
            .set_write_timeout(Some(ASYNC_FORWARD_TERMINAL_TIMEOUT))
            .context("set async activation forward write timeout")?;
        let (sender, receiver) = mpsc::sync_channel::<AsyncForwardJob>(queue_capacity.max(1));
        let writer_thread =
            thread::spawn(move || run_forwarder(&mut writer, &receiver, &telemetry));
        Ok(Self {
            sender: Some(sender),
            pending: VecDeque::new(),
            writer: Some(writer_thread),
        })
    }

    pub(crate) fn send(
        &mut self,
        message: StageWireMessage,
        condition: WireCondition,
        attrs: BTreeMap<String, Value>,
    ) -> Result<()> {
        let receipt = self.send_tracked(message, condition, attrs)?;
        self.pending.push_back(receipt);
        Ok(())
    }

    pub(crate) fn send_tracked(
        &mut self,
        message: StageWireMessage,
        condition: WireCondition,
        attrs: BTreeMap<String, Value>,
    ) -> Result<AsyncForwardReceipt> {
        self.reap_completed()?;
        let (done, receiver) = mpsc::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| anyhow!("async activation forwarder stopped"))?
            .send(AsyncForwardJob {
                message,
                condition,
                attrs,
                done,
                enqueued_at: Instant::now(),
                enqueued_unix_nanos: now_unix_nanos() as u64,
            })
            .map_err(|_| anyhow!("async activation forwarder stopped"))?;
        Ok(AsyncForwardReceipt { receiver })
    }

    fn reap_completed(&mut self) -> Result<()> {
        loop {
            let Some(receiver) = self.pending.front() else {
                return Ok(());
            };
            match receiver.try_finish() {
                Ok(Some(_write_ms)) => {
                    self.pending.pop_front();
                }
                Ok(None) => return Ok(()),
                Err(error) => {
                    self.pending.pop_front();
                    return Err(error);
                }
            }
        }
    }

    pub(crate) fn flush(&mut self) -> Result<()> {
        while let Some(receiver) = self.pending.pop_front() {
            receiver.finish()?;
        }
        Ok(())
    }
}

fn run_forwarder(
    writer: &mut TcpStream,
    receiver: &mpsc::Receiver<AsyncForwardJob>,
    telemetry: &Telemetry,
) {
    while let Ok(job) = receiver.recv() {
        let wait = time_until_ready(&job);
        if !wait.is_zero() {
            thread::sleep(wait);
        }
        forward_job(writer, telemetry, job);
    }
}

fn time_until_ready(job: &AsyncForwardJob) -> std::time::Duration {
    let ready_at = job.enqueued_at + job.condition.propagation_delay();
    ready_at.saturating_duration_since(Instant::now())
}

fn forward_job(writer: &mut TcpStream, telemetry: &Telemetry, job: AsyncForwardJob) {
    let result = write_stage_message_after_propagation(writer, &job.message, job.condition)
        .context("async forward activation frame downstream")
        .map(|()| elapsed_ms(job.enqueued_at))
        .map_err(|error| format!("{error:#}"));
    let write_end_unix_nanos = now_unix_nanos() as u64;
    let mut attrs = job.attrs;
    attrs.insert(
        "llama_stage.forward_write_ms".to_string(),
        json!(elapsed_ms(job.enqueued_at)),
    );
    telemetry.emit_debug_span(
        "stage.binary_downstream_write",
        attrs,
        job.enqueued_unix_nanos,
        write_end_unix_nanos,
    );
    let _ = job.done.send(result);
}

impl AsyncForwardReceipt {
    pub(crate) fn finish(self) -> Result<f64> {
        self.finish_with_timeout(ASYNC_FORWARD_TERMINAL_TIMEOUT)
    }

    fn finish_with_timeout(self, timeout: Duration) -> Result<f64> {
        match self.receiver.recv_timeout(timeout) {
            Ok(result) => result.map_err(|error| anyhow!(error)),
            Err(RecvTimeoutError::Timeout) => {
                Err(anyhow!("timed out waiting for async activation forward"))
            }
            Err(RecvTimeoutError::Disconnected) => {
                Err(anyhow!("async activation forwarder dropped result"))
            }
        }
    }

    fn try_finish(&self) -> Result<Option<f64>> {
        match self.receiver.try_recv() {
            Ok(Ok(write_ms)) => Ok(Some(write_ms)),
            Ok(Err(error)) => Err(anyhow!(error)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                Err(anyhow!("async activation forwarder dropped result"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use skippy_protocol::binary::{StageStateHeader, WireMessageKind, read_stage_message};

    use super::*;
    use crate::binary_transport::stage_execution::prefix_cache_test_config;
    use crate::telemetry::TelemetryLevel;

    fn message(kind: WireMessageKind, pos_start: i32) -> StageWireMessage {
        StageWireMessage {
            kind,
            pos_start,
            token_count: if kind == WireMessageKind::RetireVerifyWindow {
                4
            } else {
                0
            },
            state: StageStateHeader::new(kind),
            request_id: 1,
            session_id: 2,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    /// A delayed discard must be fully written before a teardown `Stop` that
    /// goes out through a different clone of the same socket, otherwise the
    /// two frames interleave and poison a lane that is handed back for reuse.
    #[test]
    fn a_delayed_discard_lands_before_a_teardown_stop_on_another_clone() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(address).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let telemetry = Telemetry::new(None, 1, prefix_cache_test_config(), TelemetryLevel::Off);
        let mut forwarder = AsyncForwarder::new(&client, telemetry, 8).unwrap();
        // 250ms of simulated propagation: without the drop-time join, the
        // teardown write below wins the race and the frames interleave.
        let condition = WireCondition::new(250.0, None).unwrap();

        forwarder
            .send(
                message(WireMessageKind::DiscardStaleWindows, 11),
                condition,
                BTreeMap::new(),
            )
            .unwrap();
        drop(forwarder);

        write_stage_message_after_propagation(
            &mut client,
            &message(WireMessageKind::Stop, 22),
            WireCondition::new(0.0, None).unwrap(),
        )
        .unwrap();

        let first = read_stage_message(&mut server, 4).unwrap();
        let second = read_stage_message(&mut server, 4).unwrap();

        assert_eq!(first.kind, WireMessageKind::DiscardStaleWindows);
        assert_eq!(first.pos_start, 11);
        assert_eq!(second.kind, WireMessageKind::Stop);
        assert_eq!(second.pos_start, 22);
    }

    /// The lane-reuse half of the same property: after a delayed discard and
    /// the teardown `Stop`, the socket must be clean enough to serve the next
    /// request. A frame left half-written by the torn-down forwarder would be
    /// read as the next request's header, so this asserts both frames arrive
    /// whole and in order and that the next request's traffic follows them
    /// undisturbed on the same socket.
    #[test]
    fn a_reused_lane_carries_the_next_request_after_a_delayed_teardown() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let mut lane = TcpStream::connect(address).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let telemetry = Telemetry::new(None, 1, prefix_cache_test_config(), TelemetryLevel::Off);
        let mut forwarder = AsyncForwarder::new(&lane, telemetry, 8).unwrap();
        let delayed = WireCondition::new(250.0, None).unwrap();
        let immediate = WireCondition::new(0.0, None).unwrap();

        // First request: a discard still in flight when the request ends.
        forwarder
            .send(
                message(WireMessageKind::DiscardStaleWindows, 31),
                delayed,
                BTreeMap::new(),
            )
            .unwrap();
        drop(forwarder);
        write_stage_message_after_propagation(
            &mut lane,
            &message(WireMessageKind::Stop, 32),
            immediate,
        )
        .unwrap();

        // The pool hands the same socket to the next request.
        write_stage_message_after_propagation(
            &mut lane,
            &message(WireMessageKind::VerifyWindow, 41),
            immediate,
        )
        .unwrap();

        let frames = (0..3)
            .map(|_| read_stage_message(&mut server, 4).unwrap())
            .collect::<Vec<_>>();
        let observed = frames
            .iter()
            .map(|frame| (frame.kind, frame.pos_start))
            .collect::<Vec<_>>();

        assert_eq!(
            observed,
            vec![
                (WireMessageKind::DiscardStaleWindows, 31),
                (WireMessageKind::Stop, 32),
                (WireMessageKind::VerifyWindow, 41),
            ],
            "a lane returned to the pool must carry the next request intact"
        );
    }

    #[test]
    fn retirement_receipt_orders_all_prior_verify_writes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = TcpStream::connect(address).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let telemetry = Telemetry::new(None, 1, prefix_cache_test_config(), TelemetryLevel::Off);
        let mut forwarder = AsyncForwarder::new(&client, telemetry, 3).unwrap();
        let condition = WireCondition::new(0.0, None).unwrap();

        forwarder
            .send(
                message(WireMessageKind::VerifyWindow, 10),
                condition,
                BTreeMap::new(),
            )
            .unwrap();
        forwarder
            .send(
                message(WireMessageKind::VerifyWindow, 14),
                condition,
                BTreeMap::new(),
            )
            .unwrap();
        forwarder
            .send_tracked(
                message(WireMessageKind::RetireVerifyWindow, 10),
                condition,
                BTreeMap::new(),
            )
            .unwrap()
            .finish()
            .unwrap();

        let first = read_stage_message(&mut server, 1).unwrap();
        let second = read_stage_message(&mut server, 1).unwrap();
        let retire = read_stage_message(&mut server, 1).unwrap();
        assert_eq!(first.kind, WireMessageKind::VerifyWindow);
        assert_eq!(first.pos_start, 10);
        assert_eq!(second.kind, WireMessageKind::VerifyWindow);
        assert_eq!(second.pos_start, 14);
        assert_eq!(retire.kind, WireMessageKind::RetireVerifyWindow);
        assert_eq!(retire.pos_start, 10);
    }

    #[test]
    fn forward_receipt_has_a_terminal_wait_bound() {
        let (_sender, receiver) = mpsc::channel();
        let receipt = AsyncForwardReceipt { receiver };

        let error = receipt
            .finish_with_timeout(Duration::from_millis(1))
            .unwrap_err();

        assert!(error.to_string().contains("timed out"));
    }
}
