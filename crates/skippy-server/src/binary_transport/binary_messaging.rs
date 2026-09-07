use std::{
    collections::BTreeMap,
    future::Future,
    io::{self, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use super::stage_execution::{
    consume_optional_client_ready_hello, prepare_binary_stage_connection, take_ready_downstream,
    warm_downstream_preconnect_enabled,
};
use super::{
    direct_return::{PredictionReturnHub, PredictionReturnSinks},
    options::BinaryStageOptions,
    preconnect::DownstreamPreconnector,
};
use crate::{
    cli::ServeBinaryArgs,
    config::validate_config,
    frontend::{self, EmbeddedOpenAiArgs, iteration_scheduler::IterationScheduler},
    kv_integration::KvStageIntegration,
    runtime_state::{RuntimeLaunchOverrides, load_runtime_with_overrides, loaded_model_state_kind},
    telemetry::{Telemetry, lifecycle_attrs},
};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::json;
use skippy_protocol::binary::{WireMessageKind, read_stage_message_for_codec_policy, send_ready};
use skippy_runtime::ActivationBoundaryDesc;

pub(in crate::binary_transport) mod async_forwarder;
mod connection;
mod control_messages;
mod message_receive;
mod prefill_recording;
pub(in crate::binary_transport) mod reply;
mod session_lifecycle;
mod session_tracker;
mod summary;
mod telemetry;

use self::connection::handle_binary_connection;
use self::session_tracker::ConnectionSessionOwnership;

/// How often a waiting connection worker rechecks the shutdown flag, matching
/// the downstream DOWNSTREAM_SHUTDOWN_POLL cadence in stage_execution.
const WORKER_SHUTDOWN_POLL: Duration = Duration::from_millis(100);

/// Darwin `recv`/`peek` error for an expired `SO_RCVTIMEO` (os error 22).
const EINVAL: i32 = 22;

#[derive(Default)]
struct ConnectionWorkerControl {
    shutting_down: AtomicBool,
    sockets: Mutex<Vec<std::net::TcpStream>>,
}

impl ConnectionWorkerControl {
    fn track(&self, stream: &std::net::TcpStream) -> io::Result<()> {
        let tracked = stream.try_clone()?;
        let mut sockets = self
            .sockets
            .lock()
            .expect("connection sockets lock poisoned");
        if self.shutting_down.load(Ordering::Acquire) {
            let _ = tracked.shutdown(std::net::Shutdown::Both);
        }
        sockets.push(tracked);
        Ok(())
    }

    fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::Release);
        let sockets = self
            .sockets
            .lock()
            .expect("connection sockets lock poisoned");
        for socket in sockets.iter() {
            let _ = socket.shutdown(std::net::Shutdown::Both);
        }
    }

    fn clear(&self) {
        self.sockets
            .lock()
            .expect("connection sockets lock poisoned")
            .clear();
    }

    fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Acquire)
    }

    /// Block until `stream` has readable data (or EOF), returning false when
    /// shutdown is requested instead. Peeks under a short read timeout so no
    /// message bytes are consumed and the worker never sits in an
    /// uninterruptible blocking read: `TcpStream::shutdown` on a tracked
    /// clone does not unblock an in-flight `read` on Windows (#1538).
    ///
    /// Darwin quirk: when a blocking socket has `SO_RCVTIMEO` set, an expired
    /// timeout surfaces from `peek` as `EINVAL` (os error 22), not
    /// `WouldBlock`/`TimedOut`. Treating EINVAL as fatal here killed healthy
    /// stage connections on their first idle poll (two-node split smoke:
    /// "wait for the first binary stage message: Invalid argument (os error
    /// 22)"). We therefore tolerate EINVAL only while our own read timeout is
    /// provably in effect — `timeout_armed` is captured when we set it, so the
    /// guard cannot drift from the socket state — and a timeout-less EINVAL
    /// still fails the connection loudly.
    ///
    /// The tail `set_read_timeout(None)` is best-effort for the same reason:
    /// during teardown Darwin can reject the clear with EINVAL too, and that
    /// cleanup failure must not flip an already-decided poll outcome into a
    /// connection error. For the same teardown window, if even arming the
    /// timeout fails with EINVAL while shutdown is in progress, report
    /// `Ok(false)` — the worker is winding down either way.
    fn wait_for_readable(&self, stream: &TcpStream) -> io::Result<bool> {
        let timeout_armed = match stream.set_read_timeout(Some(WORKER_SHUTDOWN_POLL)) {
            Ok(()) => true,
            // Darwin rejects SO_RCVTIMEO operations with EINVAL (os error 22)
            // on a socket that has been shut down for teardown. If the worker
            // is shutting down, that is the expected outcome, not an error.
            Err(_) if self.is_shutting_down() => return Ok(false),
            Err(error) => return Err(error),
        };
        let mut probe = [0u8; 1];
        let ready = loop {
            if self.is_shutting_down() {
                break false;
            }
            let probe_started = Instant::now();
            match stream.peek(&mut probe) {
                Ok(_) => break true,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::Interrupted
                            | io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                    ) => {}
                Err(error)
                    if timeout_armed
                        && error.raw_os_error() == Some(EINVAL)
                        && probe_started.elapsed() >= WORKER_SHUTDOWN_POLL => {}
                Err(error) => {
                    let _ = stream.set_read_timeout(None);
                    return Err(error);
                }
            }
        };
        // Best-effort cleanup: on Darwin, clearing SO_RCVTIMEO during
        // connection teardown can itself fail with EINVAL once the peer (or
        // our own shutdown path) has closed the socket. The poll outcome is
        // already established at this point, so a failing timeout clear must
        // not flip a healthy idle shutdown into "binary stage connection
        // failed: Invalid argument (os error 22)".
        let _ = stream.set_read_timeout(None);
        Ok(ready)
    }
}

struct ConnectionWorker {
    control: Arc<ConnectionWorkerControl>,
    task: JoinHandle<()>,
}

#[derive(Default)]
struct ConnectionWorkers(Vec<ConnectionWorker>);

impl ConnectionWorkers {
    fn push(&mut self, worker: ConnectionWorker) {
        self.0.push(worker);
    }

    fn reap_finished(&mut self) -> usize {
        let mut panicked = 0;
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index].task.is_finished() {
                let worker = self.0.swap_remove(index);
                if worker.task.join().is_err() {
                    // A panicked worker only fails its own connection; the
                    // accept loop must keep serving other clients.
                    panicked += 1;
                }
            } else {
                index += 1;
            }
        }
        panicked
    }

    fn shutdown(mut self) -> Result<()> {
        for worker in &self.0 {
            worker.control.shutdown();
        }
        let mut panicked = false;
        for worker in self.0.drain(..) {
            panicked |= worker.task.join().is_err();
        }
        if panicked {
            bail!("binary stage connection worker panicked during shutdown");
        }
        Ok(())
    }
}

fn finish_connection_workers(
    accept_result: Result<()>,
    connection_workers: ConnectionWorkers,
) -> Result<()> {
    let shutdown_result = connection_workers.shutdown();
    match (accept_result, shutdown_result) {
        (Ok(()), result) => result,
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(shutdown_error)) => Err(error.context(format!(
            "connection worker shutdown also failed: {shutdown_error:#}"
        ))),
    }
}

pub async fn serve_binary(args: ServeBinaryArgs) -> Result<()> {
    serve_binary_stage(BinaryStageOptions::from_cli_args(args)?).await
}

pub async fn serve_binary_stage(options: BinaryStageOptions) -> Result<()> {
    serve_binary_stage_with_shutdown(options, std::future::pending::<()>()).await
}

pub async fn serve_binary_stage_with_shutdown(
    options: BinaryStageOptions,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    serve_binary_stage_with_shutdown_and_boundary_observer(options, shutdown, |_, _| {}).await
}

pub(crate) async fn serve_binary_stage_with_shutdown_and_boundary_observer(
    options: BinaryStageOptions,
    shutdown: impl Future<Output = ()> + Send + 'static,
    boundary_observer: impl FnOnce(Option<ActivationBoundaryDesc>, Option<ActivationBoundaryDesc>),
) -> Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_task = tokio::spawn({
        let stop = stop.clone();
        async move {
            shutdown.await;
            stop.store(true, Ordering::SeqCst);
        }
    });
    let result = run_binary_stage(options, stop, boundary_observer);
    stop_task.abort();
    result
}

fn run_binary_stage(
    options: BinaryStageOptions,
    shutdown: Arc<AtomicBool>,
    boundary_observer: impl FnOnce(Option<ActivationBoundaryDesc>, Option<ActivationBoundaryDesc>),
) -> Result<()> {
    let mtp_source = options.resolved_mtp_source();
    let BinaryStageOptions {
        config,
        topology,
        bind_addr,
        metrics_otlp_grpc,
        telemetry_queue_capacity,
        telemetry_level,
        max_inflight,
        reply_credit_limit,
        async_prefill_forward,
        downstream_wire_condition,
        downstream_connect_timeout_secs,
        native_mtp_enabled,
        continuous_batching,
        openai,
    } = options;
    let native_mtp_enabled = native_mtp_enabled && config.native_mtp_enabled;
    validate_config(&config, topology.as_ref())?;
    let max_inflight = max_inflight.min(config.lane_count as usize);
    let telemetry = Telemetry::new(
        metrics_otlp_grpc,
        telemetry_queue_capacity,
        config.clone(),
        telemetry_level,
    );
    telemetry.emit("stage.binary_server_start", lifecycle_attrs(&config));
    let warm_downstream = Arc::new(Mutex::new(None));
    let runtime = load_runtime_with_overrides(
        &config,
        &RuntimeLaunchOverrides {
            mtp_source,
            ..RuntimeLaunchOverrides::default()
        },
    )?
    .context("binary stage server requires model_path")?;
    let (input_boundary, output_boundary) = {
        let runtime = runtime
            .lock()
            .map_err(|_| anyhow!("runtime lock poisoned"))?;
        (
            runtime.input_activation_boundary(),
            runtime.output_activation_boundary(),
        )
    };
    let input_activation_width =
        activation_width_from_graph("input", input_boundary, config.layer_start > 0)?;
    let output_activation_width =
        activation_width_from_graph("output", output_boundary, config.downstream.is_some())?;
    if max_inflight > 0 {
        let timer = Instant::now();
        let sessions = runtime
            .lock()
            .map_err(|_| anyhow!("runtime lock poisoned"))?
            .prewarm_idle_sessions(max_inflight)
            .context("prewarm binary stage runtime sessions")?;
        let mut attrs = lifecycle_attrs(&config);
        attrs.insert("llama_stage.max_inflight".to_string(), json!(max_inflight));
        attrs.insert(
            "llama_stage.lane_count".to_string(),
            json!(sessions.lane_count),
        );
        attrs.insert(
            "llama_stage.runtime_sessions_active".to_string(),
            json!(sessions.active_sessions),
        );
        attrs.insert(
            "llama_stage.runtime_sessions_idle".to_string(),
            json!(sessions.idle_sessions),
        );
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed().as_secs_f64() * 1000.0),
        );
        telemetry.emit("stage.binary_runtime_prewarm", attrs);
    }
    let iteration_scheduler = IterationScheduler::new(
        runtime.clone(),
        &config,
        max_inflight.max(1),
        continuous_batching,
        telemetry.clone(),
    )
    .map_err(|error| anyhow!("create binary iteration scheduler: {error}"))?;
    let kv =
        KvStageIntegration::from_loaded_model(&config, loaded_model_state_kind(Some(&runtime)))?
            .map(Arc::new);
    let prediction_returns = Arc::new(PredictionReturnHub::default());
    let prediction_return_sinks = Arc::new(PredictionReturnSinks::default());
    let session_ownership = Arc::new(ConnectionSessionOwnership::default());
    let mut connection_workers = ConnectionWorkers::default();
    let listener = TcpListener::bind(bind_addr)?;
    listener.set_nonblocking(true)?;
    boundary_observer(input_boundary, output_boundary);
    if let Some(openai_options) = openai {
        if config.stage_index != 0 || config.layer_start != 0 {
            bail!("--openai-bind-addr is only supported on stage 0");
        }
        let openai_config = config.clone();
        let openai_runtime = runtime.clone();
        let openai_iteration_scheduler = iteration_scheduler.clone();
        let openai_telemetry = telemetry.clone();
        let openai_prediction_returns = prediction_returns.clone();
        tokio::spawn(async move {
            if let Err(error) =
                frontend::serve_embedded_openai_with_scheduler(
                    EmbeddedOpenAiArgs {
                        bind_addr: openai_options.bind_addr,
                        config: openai_config,
                        runtime: openai_runtime,
                        model_id: openai_options.model_id,
                        default_max_tokens: openai_options.default_max_tokens,
                        request_defaults: frontend::EmbeddedOpenAiRequestDefaults::default(),
                        generation_concurrency: openai_options.generation_concurrency,
                        continuous_batching,
                        adaptive_generation_min_concurrency: openai_options
                            .adaptive_generation_min_concurrency,
                        generation_queue_capacity: openai_options.generation_queue_capacity,
                        generation_admission_timeout_secs: openai_options
                            .generation_admission_timeout_secs,
                        prefill_chunk_size: openai_options.prefill_chunk_size,
                        prefill_chunk_policy: openai_options.prefill_chunk_policy,
                        prefill_chunk_schedule: openai_options.prefill_chunk_schedule,
                        prefill_adaptive_start: openai_options.prefill_adaptive_start,
                        prefill_adaptive_step: openai_options.prefill_adaptive_step,
                        prefill_adaptive_max: openai_options.prefill_adaptive_max,
                        prefill_adaptive_target_ms: openai_options.prefill_adaptive_target_ms,
                        draft_model_path: openai_options.draft_model_path,
                        speculative_window: openai_options.speculative_window,
                        adaptive_speculative_window: openai_options.adaptive_speculative_window,
                        draft_n_gpu_layers: openai_options.draft_n_gpu_layers,
                        speculative: openai_options.speculative.clone(),
                        native_mtp_enabled: native_mtp_enabled
                            && openai_options.speculative.native_mtp.enabled,
                        native_mtp_draft_model_path: openai_options.native_mtp_draft_model_path,
                        native_mtp_max_tokens: openai_options.native_mtp_max_tokens,
                        native_mtp_min_tokens: openai_options.native_mtp_min_tokens,
                        activation_width: output_activation_width,
                        reply_credit_limit,
                        downstream_connect_timeout_secs,
                        downstream_wire_condition,
                        prediction_returns: Some(openai_prediction_returns),
                        telemetry: openai_telemetry,
                        hook_policy: None,
                        generation_receipt: None,
                        linear_proposal_ingress: None,
                        openai_guardrails: Some(
                            frontend::OpenAiGuardrailsConfig::disabled_for_skippy(),
                        ),
                    },
                    openai_iteration_scheduler,
                )
                .await
            {
                eprintln!("embedded OpenAI server failed: {error:#}");
            }
        });
    }
    let _downstream_preconnector = warm_downstream_preconnect_enabled()
        .then(|| {
            DownstreamPreconnector::spawn(config.clone(), warm_downstream.clone(), shutdown.clone())
        })
        .transpose()
        .context("spawn downstream preconnector")?;
    println!(
        "skippy-server listening: binary={} stage_id={} layer_range={}..{} input_activation_width={} output_activation_width={}",
        bind_addr,
        config.stage_id,
        config.layer_start,
        config.layer_end,
        input_activation_width,
        output_activation_width,
    );

    let accept_result = (|| -> Result<()> {
        while !shutdown.load(Ordering::SeqCst) {
            let panicked_workers = connection_workers.reap_finished();
            if panicked_workers > 0 {
                telemetry.emit(
                    "stage.connection_worker_panic",
                    BTreeMap::from([
                        ("llama_stage.failure_contained".to_string(), json!(true)),
                        (
                            "llama_stage.panicked_workers".to_string(),
                            json!(panicked_workers),
                        ),
                    ]),
                );
            }
            let (mut upstream, _) = match listener.accept() {
                Ok(conn) => conn,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(50));
                    continue;
                }
                Err(error) => return Err(error).context("accept binary stage connection"),
            };
            prepare_binary_stage_connection(&upstream)?;
            let peer_addr = upstream.peer_addr().ok();
            eprintln!(
                "binary accepted connection: stage_id={} peer={peer_addr:?}",
                config.stage_id
            );
            let config = config.clone();
            let topology = topology.clone();
            let iteration_scheduler = iteration_scheduler.clone();
            let kv = kv.clone();
            let telemetry = telemetry.clone();
            let warm_downstream = warm_downstream.clone();
            let worker_shutdown = shutdown.clone();
            let prediction_returns = prediction_returns.clone();
            let prediction_return_sinks = prediction_return_sinks.clone();
            let session_ownership = session_ownership.clone();
            let worker_control = Arc::new(ConnectionWorkerControl::default());
            worker_control
                .track(&upstream)
                .context("track upstream binary stage connection")?;
            let task_control = worker_control.clone();
            let task = thread::spawn(move || {
                let connection_result = (|| -> Result<()> {
                    eprintln!(
                        "binary sending ready: stage_id={} peer={peer_addr:?}",
                        config.stage_id
                    );
                    consume_optional_client_ready_hello(&mut upstream)
                        .context("consume optional client ready hello")?;
                    send_ready(&mut upstream).context("failed to send binary ready")?;
                    upstream.flush().ok();
                    eprintln!(
                        "binary sent ready: stage_id={} peer={peer_addr:?}",
                        config.stage_id
                    );
                    if !task_control
                        .wait_for_readable(&upstream)
                        .context("wait for the first binary stage message")?
                    {
                        return Ok(());
                    }
                    let first_message = match read_stage_message_for_codec_policy(
                        &mut upstream,
                        input_activation_width,
                        config.activation_codec,
                        config.activation_codec_policy,
                    ) {
                        Ok(message) => message,
                        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                            return Ok(());
                        }
                        Err(error) => return Err(error.into()),
                    };
                    if first_message.kind == WireMessageKind::PredictionReturnOpen {
                        if config.stage_index == 0 {
                            return prediction_returns
                                .handle_return_connection(first_message, upstream);
                        }
                        return prediction_return_sinks.insert_opened_sink(first_message, upstream);
                    }
                    let downstream = take_ready_downstream(
                        &config,
                        &warm_downstream,
                        downstream_connect_timeout_secs,
                        &worker_shutdown,
                    )?;
                    if let Some(stream) = downstream.as_ref() {
                        task_control
                            .track(stream)
                            .context("track downstream binary stage connection")?;
                    }
                    handle_binary_connection(
                        &config,
                        topology.as_ref(),
                        &iteration_scheduler,
                        kv.as_ref(),
                        &telemetry,
                        &mut upstream,
                        downstream,
                        input_activation_width,
                        output_activation_width,
                        max_inflight,
                        reply_credit_limit,
                        async_prefill_forward,
                        downstream_wire_condition,
                        downstream_connect_timeout_secs,
                        native_mtp_enabled,
                        &prediction_return_sinks,
                        session_ownership,
                        &task_control,
                        first_message,
                    )
                })()
                .context("binary stage connection failed");
                if let Err(error) = connection_result {
                    let mut attrs = lifecycle_attrs(&config);
                    if let Some(peer_addr) = peer_addr {
                        attrs.insert("llama_stage.peer_addr".to_string(), json!(peer_addr));
                    }
                    attrs.insert("llama_stage.error".to_string(), json!(error.to_string()));
                    eprintln!("{error:#}");
                    telemetry.emit("stage.binary_connection_error", attrs);
                }
                task_control.clear();
            });
            connection_workers.push(ConnectionWorker {
                control: worker_control,
                task,
            });
        }
        Ok(())
    })();
    shutdown.store(true, Ordering::SeqCst);
    finish_connection_workers(accept_result, connection_workers)
}

fn activation_width_from_graph(
    edge: &str,
    descriptor: Option<ActivationBoundaryDesc>,
    required: bool,
) -> Result<i32> {
    let Some(descriptor) = descriptor else {
        if required {
            bail!("stage graph did not expose its {edge} activation boundary");
        }
        return Ok(0);
    };
    descriptor.raw_f32_width(edge)
}

#[cfg(test)]
mod shutdown_tests {
    use super::{
        ConnectionWorker, ConnectionWorkerControl, ConnectionWorkers, activation_width_from_graph,
        finish_connection_workers,
    };
    use anyhow::anyhow;
    use skippy_runtime::ActivationBoundaryDesc;
    use std::{
        io::{Read, Write},
        net::{TcpListener, TcpStream},
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant},
    };

    fn f32_boundary(elements_per_token: u64) -> ActivationBoundaryDesc {
        ActivationBoundaryDesc {
            version: 1,
            ggml_type: 0,
            layout: 1,
            elements_per_token,
            bytes_per_token: elements_per_token * std::mem::size_of::<f32>() as u64,
            required_frame_flags: 0,
            required_sidebands: 0,
        }
    }

    #[test]
    fn graph_boundary_is_the_only_activation_width_authority() {
        assert_eq!(
            activation_width_from_graph("output", Some(f32_boundary(1024)), true)
                .expect("valid graph boundary"),
            1024
        );
    }

    #[test]
    fn required_graph_boundary_cannot_be_omitted() {
        let error = activation_width_from_graph("input", None, true)
            .expect_err("required graph boundary must be present");
        assert!(error.to_string().contains("did not expose"));
    }

    #[test]
    fn absent_unused_graph_boundary_has_no_wire_width() {
        assert_eq!(
            activation_width_from_graph("input", None, false)
                .expect("unused edge may omit a boundary"),
            0
        );
    }

    #[test]
    fn unsupported_graph_boundary_fails_closed() {
        let mut boundary = f32_boundary(1024);
        boundary.ggml_type = 1;
        let error = activation_width_from_graph("output", Some(boundary), true)
            .expect_err("non-F32 graph boundary must not use the F32 codec");
        assert!(error.to_string().contains("requires graph-observed F32"));

        let mut boundary = f32_boundary(1024);
        boundary.bytes_per_token -= 1;
        let error = activation_width_from_graph("output", Some(boundary), true)
            .expect_err("inconsistent graph boundary must fail");
        assert!(error.to_string().contains("reports 4095 bytes"));

        let error = activation_width_from_graph("output", Some(f32_boundary(0)), true)
            .expect_err("empty graph boundary must fail");
        assert!(error.to_string().contains("zero elements"));
    }

    #[test]
    fn idle_poll_survives_a_full_read_timeout_expiry() {
        // Regression: on Darwin a SO_RCVTIMEO expiry inside `peek` surfaces
        // as EINVAL (os error 22). The worker must keep polling on an idle
        // connection instead of failing it, and must still observe data
        // arriving after several expired polls.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let control = Arc::new(ConnectionWorkerControl::default());
        control.track(&server).unwrap();
        let task_control = control.clone();
        let task = thread::spawn(move || {
            let started = Instant::now();
            let mut ready = false;
            while !ready {
                match task_control.wait_for_readable(&server) {
                    Ok(became_ready) => ready = became_ready,
                    Err(error) => panic!("idle poll must not fail: {error}"),
                }
            }
            assert!(ready, "connection must become readable after data arrives");
            assert!(
                started.elapsed() >= Duration::from_millis(200),
                "at least two timeout polls must have expired before data arrived"
            );
            let mut byte = [0u8; 1];
            server.read_exact(&mut byte).unwrap();
            assert_eq!(byte[0], b'x');
            task_control.clear();
        });
        thread::sleep(Duration::from_millis(250));
        client.write_all(b"x").unwrap();
        let (done_tx, done_rx) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let _ = done_tx.send(task.join().is_ok());
        });
        assert!(
            done_rx
                .recv_timeout(Duration::from_secs(5))
                .expect("worker must finish")
        );
    }

    #[test]
    fn shutdown_poll_on_a_shut_down_socket_stays_ok() {
        // Regression: during teardown on Darwin, the tail
        // `set_read_timeout(None)` fails with EINVAL once the socket has been
        // shut down ("Invalid argument (os error 22)" at shutdown). The
        // cleanup is best-effort — the poll's already-decided outcome must be
        // reported as Ok instead of surfacing a connection error.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let _client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let control = Arc::new(ConnectionWorkerControl::default());
        control.track(&server).unwrap();
        control.shutdown();
        let result = control.wait_for_readable(&server);
        assert!(
            result.is_ok(),
            "poll after shutdown must not fail: {:?}",
            result.err()
        );
    }

    #[test]
    fn shutdown_closes_and_joins_an_active_connection_worker() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let control = Arc::new(ConnectionWorkerControl::default());
        control.track(&server).unwrap();
        let task_control = control.clone();
        let task = thread::spawn(move || {
            if let Ok(true) = task_control.wait_for_readable(&server) {
                let mut byte = [0u8; 1];
                let _ = server.read(&mut byte);
            }
            task_control.clear();
        });
        let mut workers = ConnectionWorkers::default();
        workers.push(ConnectionWorker { control, task });

        let (cleanup_tx, cleanup_rx) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let _ = cleanup_tx.send(workers.shutdown());
        });
        cleanup_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("active worker cleanup must complete within one second")
            .unwrap();
        drop(client);
    }

    #[test]
    fn accept_error_still_closes_and_joins_active_connection_worker() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let control = Arc::new(ConnectionWorkerControl::default());
        control.track(&server).unwrap();
        let task_control = control.clone();
        let finished = Arc::new(AtomicBool::new(false));
        let task_finished = finished.clone();
        let task = thread::spawn(move || {
            if let Ok(true) = task_control.wait_for_readable(&server) {
                let mut byte = [0u8; 1];
                let _ = server.read(&mut byte);
            }
            task_control.clear();
            task_finished.store(true, Ordering::Release);
        });
        let mut workers = ConnectionWorkers::default();
        workers.push(ConnectionWorker { control, task });

        let (cleanup_tx, cleanup_rx) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let result = finish_connection_workers(Err(anyhow!("accept failed")), workers);
            let _ = cleanup_tx.send(result);
        });

        let result = cleanup_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("active worker cleanup must complete within one second");
        assert!(finished.load(Ordering::Acquire));
        let error = result.expect_err("accept failure must be returned after worker cleanup");
        assert!(format!("{error:#}").contains("accept failed"));
        drop(client);
    }

    #[test]
    fn reap_finished_contains_a_panicked_worker_and_keeps_reaping() {
        let control = Arc::new(ConnectionWorkerControl::default());
        let task = thread::spawn(|| panic!("connection worker exploded"));
        let deadline = Instant::now() + Duration::from_secs(1);
        while !task.is_finished() {
            assert!(
                Instant::now() < deadline,
                "panicking worker thread must finish within one second"
            );
            thread::sleep(Duration::from_millis(5));
        }
        let mut workers = ConnectionWorkers::default();
        workers.push(ConnectionWorker { control, task });

        assert_eq!(
            workers.reap_finished(),
            1,
            "the panicked worker must be counted, not turned into an error"
        );
        assert!(
            workers.0.is_empty(),
            "the panicked worker must still be reaped"
        );
        assert_eq!(workers.reap_finished(), 0);
    }

    #[test]
    fn reap_finished_leaves_running_workers_alone() {
        let (stop_tx, stop_rx) = mpsc::sync_channel::<()>(1);
        let control = Arc::new(ConnectionWorkerControl::default());
        let task = thread::spawn(move || {
            let _ = stop_rx.recv();
        });
        let mut workers = ConnectionWorkers::default();
        workers.push(ConnectionWorker { control, task });

        assert_eq!(workers.reap_finished(), 0);
        assert_eq!(workers.0.len(), 1, "a running worker must not be reaped");
        stop_tx.send(()).unwrap();
        workers.shutdown().unwrap();
    }
}
