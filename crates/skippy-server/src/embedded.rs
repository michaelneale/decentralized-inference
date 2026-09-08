use std::{
    net::SocketAddr,
    sync::atomic::{AtomicBool, Ordering},
    sync::{Arc, Mutex, OnceLock, TryLockError},
};

use anyhow::{Context, Result};
use axum::Router;
use openai_frontend::{OpenAiBackend, OpenAiFrontendConfig, OpenAiLifecycleObserver};
use skippy_protocol::{StageConfig, StageTopology};
use skippy_runtime::{ActivationBoundaryDesc, MtpSource};
use tokio::{sync::oneshot, task::JoinHandle};

use crate::{
    binary_transport::BinaryStageOptions,
    config::validate_config,
    frontend::{EmbeddedOpenAiArgs, serve_embedded_openai_with_shutdown},
    http::{StageHttpOptions, serve_stage_http_with_shutdown},
    runtime_state::{
        RuntimeLaunchOverrides, RuntimeSessionStats, RuntimeState, load_runtime_with_overrides,
        load_runtime_with_overrides_and_open_events,
    },
    telemetry::{Telemetry, TelemetryLevel, TelemetryStats, lifecycle_attrs, now_unix_nanos},
    tokenizer::{TokenizerCapability, TokenizerCapabilityError, tokenizer_http_router},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedState {
    Starting,
    Ready,
    Stopping,
    Stopped,
    Failed,
}

#[derive(Clone, Debug)]
pub struct EmbeddedRuntimeStatus {
    pub state: EmbeddedState,
    pub run_id: String,
    pub topology_id: String,
    pub model_id: String,
    pub stage_id: String,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub runtime_loaded: bool,
    pub started_at_unix_nanos: i64,
    pub stopped_at_unix_nanos: Option<i64>,
    pub last_error: Option<String>,
    /// Session stats, possibly a cached snapshot rather than a live read.
    ///
    /// `lane_count` is authoritative (it comes from `StageConfig`); everything
    /// else may be frozen. Display only — never gate a decision on it.
    pub sessions: RuntimeSessionStats,
    /// When [`Self::sessions`] was actually read, which may be arbitrarily
    /// earlier than this status. Derive any freshness signal from this rather
    /// than from the current time.
    pub sessions_captured_at_unix_nanos: i64,
    pub telemetry: TelemetryStats,
}

#[derive(Clone, Debug)]
pub struct EmbeddedServerStatus {
    pub name: &'static str,
    pub bind_addr: SocketAddr,
    pub state: EmbeddedState,
    pub started_at_unix_nanos: i64,
    pub stopped_at_unix_nanos: Option<i64>,
    pub last_error: Option<String>,
    pub input_activation_boundary: Option<ActivationBoundaryDesc>,
    pub output_activation_boundary: Option<ActivationBoundaryDesc>,
}

#[derive(Clone)]
pub struct EmbeddedRuntimeOptions {
    pub config: StageConfig,
    pub topology: Option<StageTopology>,
    pub n_threads: Option<usize>,
    pub n_threads_batch: Option<usize>,
    pub mtp_source: MtpSource,
    pub metrics_otlp_grpc: Option<String>,
    pub telemetry_queue_capacity: usize,
    pub telemetry_level: TelemetryLevel,
}

pub struct SkippyRuntimeHandle {
    config: Arc<StageConfig>,
    topology: Option<Arc<StageTopology>>,
    runtime: Arc<Mutex<RuntimeState>>,
    telemetry: Telemetry,
    status: Arc<Mutex<RuntimeHandleState>>,
    tokenizer_active: Arc<AtomicBool>,
    tokenizer_capability: OnceLock<Result<TokenizerCapability, TokenizerCapabilityError>>,
    /// Last session stats read out of [`Self::runtime`], and when.
    ///
    /// A native call (long prefill, decode batch) holds the runtime lock while
    /// it runs, so status reads take it opportunistically and fall back to
    /// this cache rather than queueing behind inference. The capture time
    /// travels with the value so a stalled runtime is distinguishable from an
    /// idle one.
    last_session_stats: Arc<Mutex<Captured<RuntimeSessionStats>>>,
}

/// A value together with when it was actually read from its source.
#[derive(Clone, Debug)]
struct Captured<V> {
    value: V,
    captured_at_unix_nanos: i64,
}

#[derive(Debug)]
struct RuntimeHandleState {
    state: EmbeddedState,
    started_at_unix_nanos: i64,
    stopped_at_unix_nanos: Option<i64>,
    last_error: Option<String>,
}

impl SkippyRuntimeHandle {
    pub fn input_activation_boundary(&self) -> Option<ActivationBoundaryDesc> {
        self.runtime
            .lock()
            .expect("runtime lock poisoned")
            .input_activation_boundary()
    }

    pub fn output_activation_boundary(&self) -> Option<ActivationBoundaryDesc> {
        self.runtime
            .lock()
            .expect("runtime lock poisoned")
            .output_activation_boundary()
    }

    /// Assemble a ready handle around an already-loaded runtime.
    ///
    /// Shared by both loaders so the stats cache is primed exactly once, in one
    /// place: priming has to happen while `runtime` is still unshared and
    /// uncontended, so that a status read which loses the race to the very
    /// first generation reports real lanes instead of zeros.
    fn ready(
        config: StageConfig,
        topology: Option<StageTopology>,
        runtime: Arc<Mutex<RuntimeState>>,
        telemetry: Telemetry,
    ) -> Self {
        let initial_session_stats = Captured {
            value: runtime
                .lock()
                .expect("runtime lock poisoned")
                .session_stats(),
            captured_at_unix_nanos: now_unix_nanos(),
        };
        Self {
            config: Arc::new(config),
            topology: topology.map(Arc::new),
            runtime,
            telemetry,
            status: Arc::new(Mutex::new(RuntimeHandleState {
                state: EmbeddedState::Ready,
                started_at_unix_nanos: now_unix_nanos(),
                stopped_at_unix_nanos: None,
                last_error: None,
            })),
            tokenizer_active: Arc::new(AtomicBool::new(true)),
            tokenizer_capability: OnceLock::new(),
            last_session_stats: Arc::new(Mutex::new(initial_session_stats)),
        }
    }

    pub fn load(options: EmbeddedRuntimeOptions) -> Result<Self> {
        validate_config(&options.config, options.topology.as_ref())?;
        let telemetry = Telemetry::new(
            options.metrics_otlp_grpc,
            options.telemetry_queue_capacity,
            options.config.clone(),
            options.telemetry_level,
        );
        telemetry.emit(
            "stage.embedded_runtime_load_start",
            lifecycle_attrs(&options.config),
        );
        let runtime = load_runtime_with_overrides(
            &options.config,
            &RuntimeLaunchOverrides {
                n_threads: options.n_threads,
                n_threads_batch: options.n_threads_batch,
                mtp_source: options.mtp_source,
            },
        )?
        .with_context(|| format!("stage {} requires model_path", options.config.stage_id))?;
        telemetry.emit(
            "stage.embedded_runtime_ready",
            lifecycle_attrs(&options.config),
        );
        Ok(Self::ready(
            options.config,
            options.topology,
            runtime,
            telemetry,
        ))
    }

    pub fn load_with_open_events(
        options: EmbeddedRuntimeOptions,
        mut model_open_event_reporter: Option<Box<dyn FnMut(skippy_runtime::RuntimeEvent) + Send>>,
    ) -> Result<Self> {
        validate_config(&options.config, options.topology.as_ref())?;
        let telemetry = Telemetry::new(
            options.metrics_otlp_grpc,
            options.telemetry_queue_capacity,
            options.config.clone(),
            options.telemetry_level,
        );
        telemetry.emit(
            "stage.embedded_runtime_load_start",
            lifecycle_attrs(&options.config),
        );
        let runtime = load_runtime_with_overrides_and_open_events(
            &options.config,
            &RuntimeLaunchOverrides {
                n_threads: options.n_threads,
                n_threads_batch: options.n_threads_batch,
                mtp_source: options.mtp_source,
            },
            model_open_event_reporter.as_mut().map(|reporter| {
                reporter.as_mut() as &mut (dyn FnMut(skippy_runtime::RuntimeEvent) + Send)
            }),
        )?
        .with_context(|| format!("stage {} requires model_path", options.config.stage_id))?;
        telemetry.emit(
            "stage.embedded_runtime_ready",
            lifecycle_attrs(&options.config),
        );
        Ok(Self::ready(
            options.config,
            options.topology,
            runtime,
            telemetry,
        ))
    }

    pub fn config(&self) -> &StageConfig {
        &self.config
    }

    pub fn topology(&self) -> Option<&StageTopology> {
        self.topology.as_deref()
    }

    pub fn runtime(&self) -> Arc<Mutex<RuntimeState>> {
        self.runtime.clone()
    }

    pub fn telemetry(&self) -> Telemetry {
        self.telemetry.clone()
    }

    /// Session stats without ever waiting on the inference lock, and when they
    /// were read. A cached value carries the earlier read's time, not now.
    fn session_stats_non_blocking(&self) -> Captured<RuntimeSessionStats> {
        read_without_blocking(&self.runtime, &self.last_session_stats, |runtime| {
            runtime.session_stats()
        })
    }

    /// Returns the stateless tokenizer capability backed by this already-loaded
    /// stage-zero runtime. This never opens a second model.
    pub fn tokenizer_capability(&self) -> Result<TokenizerCapability, TokenizerCapabilityError> {
        self.tokenizer_capability
            .get_or_init(|| {
                TokenizerCapability::from_stage_zero_with_lifecycle(
                    &self.config,
                    self.runtime.clone(),
                    self.tokenizer_active.clone(),
                )
            })
            .clone()
    }

    pub fn status(&self) -> EmbeddedRuntimeStatus {
        let handle = self.status.lock().expect("runtime status lock poisoned");
        let Captured {
            value: sessions,
            captured_at_unix_nanos: sessions_captured_at_unix_nanos,
        } = self.session_stats_non_blocking();
        EmbeddedRuntimeStatus {
            state: handle.state,
            run_id: self.config.run_id.clone(),
            topology_id: self.config.topology_id.clone(),
            model_id: self.config.model_id.clone(),
            stage_id: self.config.stage_id.clone(),
            stage_index: self.config.stage_index,
            layer_start: self.config.layer_start,
            layer_end: self.config.layer_end,
            runtime_loaded: matches!(handle.state, EmbeddedState::Ready | EmbeddedState::Stopping),
            started_at_unix_nanos: handle.started_at_unix_nanos,
            stopped_at_unix_nanos: handle.stopped_at_unix_nanos,
            last_error: handle.last_error.clone(),
            sessions,
            sessions_captured_at_unix_nanos,
            telemetry: self.telemetry.stats(),
        }
    }

    pub fn shutdown(&self) {
        self.tokenizer_active.store(false, Ordering::Release);
        let runtime = self.runtime.lock().expect("runtime lock poisoned");
        drop(runtime);
        let mut status = self.status.lock().expect("runtime status lock poisoned");
        if status.state == EmbeddedState::Stopped {
            return;
        }
        status.state = EmbeddedState::Stopped;
        status.stopped_at_unix_nanos = Some(now_unix_nanos());
        self.telemetry.emit(
            "stage.embedded_runtime_stopped",
            lifecycle_attrs(&self.config),
        );
    }
}

impl Drop for SkippyRuntimeHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub struct EmbeddedServerHandle {
    status: Arc<Mutex<ServerHandleState>>,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<Result<()>>>,
}

#[derive(Debug)]
struct ServerHandleState {
    name: &'static str,
    bind_addr: SocketAddr,
    state: EmbeddedState,
    started_at_unix_nanos: i64,
    stopped_at_unix_nanos: Option<i64>,
    last_error: Option<String>,
    input_activation_boundary: Option<ActivationBoundaryDesc>,
    output_activation_boundary: Option<ActivationBoundaryDesc>,
}

impl EmbeddedServerHandle {
    pub fn status(&self) -> EmbeddedServerStatus {
        let status = self.status.lock().expect("server status lock poisoned");
        EmbeddedServerStatus {
            name: status.name,
            bind_addr: status.bind_addr,
            state: status.state,
            started_at_unix_nanos: status.started_at_unix_nanos,
            stopped_at_unix_nanos: status.stopped_at_unix_nanos,
            last_error: status.last_error.clone(),
            input_activation_boundary: status.input_activation_boundary,
            output_activation_boundary: status.output_activation_boundary,
        }
    }

    pub async fn shutdown(mut self) -> Result<()> {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        let task = self.task.take().expect("server task already taken");
        task.await?
    }

    pub fn abort(mut self) {
        self.shutdown.take();
        if let Some(task) = self.task.take() {
            task.abort();
        }
        let mut status = self.status.lock().expect("server status lock poisoned");
        status.state = EmbeddedState::Stopped;
        status.stopped_at_unix_nanos = Some(now_unix_nanos());
    }
}

impl Drop for EmbeddedServerHandle {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

pub fn start_stage_http(options: StageHttpOptions) -> EmbeddedServerHandle {
    let bind_addr = options.bind_addr;
    spawn_async_server("stage-http", bind_addr, |shutdown| async move {
        serve_stage_http_with_shutdown(options, async move {
            let _ = shutdown.await;
        })
        .await
    })
}

pub fn start_embedded_openai(args: EmbeddedOpenAiArgs) -> EmbeddedServerHandle {
    let bind_addr = args.bind_addr;
    spawn_async_server("openai", bind_addr, |shutdown| async move {
        serve_embedded_openai_with_shutdown(args, async move {
            let _ = shutdown.await;
        })
        .await
    })
}

pub fn start_openai_backend(
    bind_addr: SocketAddr,
    backend: Arc<dyn OpenAiBackend>,
) -> EmbeddedServerHandle {
    spawn_openai_backend(bind_addr, openai_frontend::router_for(backend))
}

pub fn start_openai_backend_with_tokenizer(
    bind_addr: SocketAddr,
    backend: Arc<dyn OpenAiBackend>,
    tokenizer: TokenizerCapability,
) -> EmbeddedServerHandle {
    spawn_openai_backend(bind_addr, openai_backend_router(backend, tokenizer))
}

fn spawn_openai_backend(bind_addr: SocketAddr, router: Router) -> EmbeddedServerHandle {
    let status = Arc::new(Mutex::new(ServerHandleState {
        name: "openai-backend",
        bind_addr,
        state: EmbeddedState::Starting,
        started_at_unix_nanos: now_unix_nanos(),
        stopped_at_unix_nanos: None,
        last_error: None,
        input_activation_boundary: None,
        output_activation_boundary: None,
    }));
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task_status = status.clone();
    let task = tokio::spawn(async move {
        let result = async {
            let listener = tokio::net::TcpListener::bind(bind_addr).await?;
            {
                let mut status = task_status.lock().expect("server status lock poisoned");
                status.state = EmbeddedState::Ready;
            }
            axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await?;
            Ok(())
        }
        .await;
        finish_server_status(&task_status, &result);
        result
    });
    EmbeddedServerHandle {
        status,
        shutdown: Some(shutdown_tx),
        task: Some(task),
    }
}

pub(crate) fn openai_backend_router(
    backend: Arc<dyn OpenAiBackend>,
    tokenizer: TokenizerCapability,
) -> Router {
    openai_frontend::router_for(backend).merge(tokenizer_http_router(tokenizer))
}

/// Start an OpenAI backend with an optional metadata-only lifecycle observer.
///
/// The existing [`start_openai_backend`] entry point retains the no-observer
/// behavior for embedders that do not own a logging runtime.
pub fn start_openai_backend_with_lifecycle_observer(
    bind_addr: SocketAddr,
    backend: Arc<dyn OpenAiBackend>,
    lifecycle_observer: Option<Arc<dyn OpenAiLifecycleObserver>>,
) -> EmbeddedServerHandle {
    spawn_openai_backend(
        bind_addr,
        openai_backend_router_with_lifecycle_observer(backend, lifecycle_observer),
    )
}

/// Start a tokenizer-aware OpenAI backend with an optional metadata-only
/// lifecycle observer.
pub fn start_openai_backend_with_tokenizer_and_lifecycle_observer(
    bind_addr: SocketAddr,
    backend: Arc<dyn OpenAiBackend>,
    tokenizer: TokenizerCapability,
    lifecycle_observer: Option<Arc<dyn OpenAiLifecycleObserver>>,
) -> EmbeddedServerHandle {
    let router = openai_backend_router_with_lifecycle_observer(backend, lifecycle_observer)
        .merge(tokenizer_http_router(tokenizer));
    spawn_openai_backend(bind_addr, router)
}

fn openai_backend_router_with_lifecycle_observer(
    backend: Arc<dyn OpenAiBackend>,
    lifecycle_observer: Option<Arc<dyn OpenAiLifecycleObserver>>,
) -> Router {
    let config = lifecycle_observer.map_or_else(OpenAiFrontendConfig::default, |observer| {
        OpenAiFrontendConfig::default().with_lifecycle_observer(observer)
    });
    openai_frontend::router_for_with_config(backend, config)
}

pub fn start_binary_stage(options: BinaryStageOptions) -> EmbeddedServerHandle {
    let bind_addr = options.bind_addr;
    let status = Arc::new(Mutex::new(ServerHandleState {
        name: "binary-stage",
        bind_addr,
        state: EmbeddedState::Starting,
        started_at_unix_nanos: now_unix_nanos(),
        stopped_at_unix_nanos: None,
        last_error: None,
        input_activation_boundary: None,
        output_activation_boundary: None,
    }));
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task_status = status.clone();
    let runtime = tokio::runtime::Handle::current();
    let task = tokio::task::spawn_blocking(move || {
        let boundary_status = task_status.clone();
        let result = runtime.block_on(
            crate::binary_transport::serve_binary_stage_with_shutdown_and_boundary_observer(
                options,
                async move {
                    let _ = shutdown_rx.await;
                },
                move |input, output| {
                    publish_binary_stage_boundaries(&boundary_status, input, output);
                },
            ),
        );
        finish_server_status(&task_status, &result);
        result
    });
    EmbeddedServerHandle {
        status,
        shutdown: Some(shutdown_tx),
        task: Some(task),
    }
}

fn publish_binary_stage_boundaries(
    status: &Arc<Mutex<ServerHandleState>>,
    input: Option<ActivationBoundaryDesc>,
    output: Option<ActivationBoundaryDesc>,
) {
    let mut status = status.lock().expect("server status lock poisoned");
    status.input_activation_boundary = input;
    status.output_activation_boundary = output;
    status.state = EmbeddedState::Ready;
}

fn spawn_async_server<F, Fut>(
    name: &'static str,
    bind_addr: SocketAddr,
    serve: F,
) -> EmbeddedServerHandle
where
    F: FnOnce(oneshot::Receiver<()>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<()>> + Send + 'static,
{
    let status = Arc::new(Mutex::new(ServerHandleState {
        name,
        bind_addr,
        state: EmbeddedState::Starting,
        started_at_unix_nanos: now_unix_nanos(),
        stopped_at_unix_nanos: None,
        last_error: None,
        input_activation_boundary: None,
        output_activation_boundary: None,
    }));
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let task_status = status.clone();
    let task = tokio::spawn(async move {
        {
            let mut status = task_status.lock().expect("server status lock poisoned");
            status.state = EmbeddedState::Ready;
        }
        let result = serve(shutdown_rx).await;
        finish_server_status(&task_status, &result);
        result
    });
    EmbeddedServerHandle {
        status,
        shutdown: Some(shutdown_tx),
        task: Some(task),
    }
}

fn finish_server_status(status: &Arc<Mutex<ServerHandleState>>, result: &Result<()>) {
    let mut status = status.lock().expect("server status lock poisoned");
    status.stopped_at_unix_nanos = Some(now_unix_nanos());
    match result {
        Ok(()) => {
            status.state = EmbeddedState::Stopped;
        }
        Err(error) => {
            status.state = EmbeddedState::Failed;
            status.last_error = Some(error.to_string());
        }
    }
}

/// Read a value derived from `source` without waiting on its lock, reporting
/// when it was actually read.
///
/// `source` is the inference runtime, whose mutex a native call holds for as
/// long as it runs. Blocking on it from an observability path makes status
/// reads queue behind that work, and since those reads happen on async
/// executor threads it also parks a worker. So: take the lock only when free,
/// refreshing `cache`; otherwise serve the last published value.
///
/// Staleness is unbounded — under sustained load every read can lose the race.
/// That is only safe because the capture time makes it visible, so callers
/// must propagate it rather than stamp the value as freshly observed. Nothing
/// may gate admission, routing, eviction or shutdown on these values.
///
/// A poisoned `source` still panics, as the previous blocking `lock().expect`
/// did: cached stats over a panicked runtime would hide a real failure.
fn read_without_blocking<S, V>(
    source: &Mutex<S>,
    cache: &Mutex<Captured<V>>,
    read: impl FnOnce(&S) -> V,
) -> Captured<V>
where
    V: Clone,
{
    match source.try_lock() {
        Ok(guard) => {
            let value = read(&guard);
            drop(guard);
            let captured = Captured {
                value,
                captured_at_unix_nanos: now_unix_nanos(),
            };
            *cache.lock().expect("stats cache lock poisoned") = captured.clone();
            captured
        }
        // Inference holds the runtime: serve the last published snapshot,
        // carrying the time it was taken, instead of blocking.
        Err(TryLockError::WouldBlock) => cache.lock().expect("stats cache lock poisoned").clone(),
        Err(TryLockError::Poisoned(error)) => panic!("runtime lock poisoned: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use skippy_protocol::LoadMode;

    use super::*;

    #[test]
    fn binary_stage_cannot_be_ready_before_boundaries_are_published() {
        let status = Arc::new(Mutex::new(ServerHandleState {
            name: "binary-stage",
            bind_addr: "127.0.0.1:0".parse().expect("bind address"),
            state: EmbeddedState::Starting,
            started_at_unix_nanos: 1,
            stopped_at_unix_nanos: None,
            last_error: None,
            input_activation_boundary: None,
            output_activation_boundary: None,
        }));
        let boundary = ActivationBoundaryDesc {
            version: 1,
            ggml_type: skippy_runtime::GGML_TYPE_F32,
            layout: 1,
            elements_per_token: 1024,
            bytes_per_token: 4096,
            required_frame_flags: 0,
            required_sidebands: 0,
        };

        {
            let status = status.lock().expect("server status lock poisoned");
            assert_eq!(status.state, EmbeddedState::Starting);
            assert!(status.input_activation_boundary.is_none());
            assert!(status.output_activation_boundary.is_none());
        }

        publish_binary_stage_boundaries(&status, Some(boundary), Some(boundary));

        let status = status.lock().expect("server status lock poisoned");
        assert_eq!(status.state, EmbeddedState::Ready);
        assert_eq!(status.input_activation_boundary, Some(boundary));
        assert_eq!(status.output_activation_boundary, Some(boundary));
    }

    /// A ready handle over a modelless runtime.
    ///
    /// `status()` only reads lane bookkeeping and handle state, so it is
    /// exercisable without loading a GGUF. Built through
    /// [`SkippyRuntimeHandle::ready`] rather than by hand, so these tests cover
    /// the real cache-priming path both loaders use.
    fn test_handle(lane_count: u32) -> SkippyRuntimeHandle {
        let config = StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: "org/model:Q4_K_M".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 512,
            lane_count,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: Default::default(),
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: false,
            resident_tensor_names: Vec::new(),
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: false,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
            ..StageConfig::default()
        };
        let telemetry = Telemetry::new(None, 1, config.clone(), TelemetryLevel::Off);
        let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(lane_count)));
        SkippyRuntimeHandle::ready(config, None, runtime, telemetry)
    }

    #[test]
    fn status_does_not_block_while_inference_holds_the_runtime() {
        // The actual regression, pinned at the call site rather than on the
        // helper: `status()` must stay responsive while a decode loop owns the
        // runtime mutex for the whole turn. Asserting on the helper alone does
        // not catch `status()` being rewired back to a blocking `lock()`.
        let handle = Arc::new(test_handle(3));

        // Inference takes the runtime for the length of the turn.
        let held = handle.runtime.lock().expect("lock runtime");

        // Probe from a detached thread. It must not be joined: if `status()`
        // regresses to a blocking acquire the probe never returns, and joining
        // it would hang the suite instead of reporting a failure. The timeout
        // below is the assertion.
        let (tx, rx) = mpsc::channel();
        thread::spawn({
            let handle = Arc::clone(&handle);
            move || {
                let _ = tx.send(handle.status());
            }
        });
        let probe = rx.recv_timeout(Duration::from_secs(5));

        let status = probe.expect("status() must return while the runtime lock is held");
        assert_eq!(
            status.sessions.lane_count, 3,
            "a contended read must serve the primed snapshot, not zeros"
        );
        assert_eq!(status.state, EmbeddedState::Ready);

        drop(held);
    }

    #[test]
    fn status_reports_primed_lanes_before_any_generation() {
        // Regression for the zeros-on-the-first-turn defect: the cache is
        // primed during construction, so even a read that never wins the
        // runtime lock reports real lane counts.
        let handle = test_handle(2);
        let held = handle.runtime.lock().expect("lock runtime");

        let cached = handle.session_stats_non_blocking();

        assert_eq!(
            cached.value.lane_count, 2,
            "priming must publish real lanes"
        );
        assert_eq!(cached.value.lanes.len(), 2);
        assert!(
            cached.captured_at_unix_nanos > 0,
            "priming must record when it captured"
        );
        drop(held);
    }

    #[test]
    fn status_does_not_claim_a_fresh_capture_while_the_runtime_is_busy() {
        // The false-freshness defect: a contended read must report the time of
        // the earlier successful read, so a runtime wedged inside a native
        // call shows a capture time that stops advancing rather than looking
        // freshly observed on every tick.
        let handle = test_handle(2);

        let live = handle.status();
        assert!(live.sessions_captured_at_unix_nanos > 0);

        let held = handle.runtime.lock().expect("lock runtime");
        let while_busy = handle.status();
        drop(held);

        assert_eq!(
            while_busy.sessions_captured_at_unix_nanos, live.sessions_captured_at_unix_nanos,
            "a cached read must not advance the capture time"
        );
    }

    #[test]
    fn status_reads_live_stats_when_the_runtime_is_idle() {
        let handle = test_handle(4);

        let status = handle.status();

        assert_eq!(status.sessions.lane_count, 4);
        assert_eq!(status.sessions.active_sessions, 0);
        assert!(status.runtime_loaded);
    }

    #[test]
    fn shutdown_waits_for_runtime_lock_after_invalidating_tokenizer() {
        let handle = Arc::new(test_handle(1));
        let held = handle.runtime.lock().expect("runtime lock");
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let shutdown_handle = Arc::clone(&handle);
        thread::spawn(move || {
            shutdown_handle.shutdown();
            shutdown_tx.send(()).expect("send shutdown result");
        });

        assert!(
            shutdown_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "shutdown should synchronize with an in-flight runtime operation"
        );
        drop(held);
        shutdown_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("shutdown should complete after the runtime lock is released");
        assert_eq!(handle.status().state, EmbeddedState::Stopped);
    }

    fn empty_cache(value: u32) -> Mutex<Captured<u32>> {
        Mutex::new(Captured {
            value,
            captured_at_unix_nanos: 0,
        })
    }

    #[test]
    fn read_without_blocking_refreshes_cache_when_lock_is_free() {
        let source = Mutex::new(7u32);
        let cache = empty_cache(0);

        let read = read_without_blocking(&source, &cache, |v| *v);

        assert_eq!(read.value, 7);
        assert!(
            read.captured_at_unix_nanos > 0,
            "a live read must report when it happened"
        );
        assert_eq!(
            cache.lock().unwrap().value,
            7,
            "a free lock must refresh cache"
        );
    }

    #[test]
    fn read_without_blocking_serves_cache_instead_of_waiting_on_inference() {
        // The regression: a long native call holds the runtime for as long as
        // it runs. A status read must return the last published value
        // immediately rather than block behind it.
        let source = Mutex::new(7u32);
        let cache = empty_cache(0);

        // Prime the cache while the runtime is idle.
        let live = read_without_blocking(&source, &cache, |v| *v);
        assert_eq!(live.value, 7);

        // Now inference holds the runtime lock.
        let held = source.lock().expect("lock runtime");

        let observed = read_without_blocking(&source, &cache, |v| *v);
        assert_eq!(
            observed.value, 7,
            "a contended runtime must serve the cached snapshot, never block"
        );
        assert_eq!(
            observed.captured_at_unix_nanos, live.captured_at_unix_nanos,
            "a cached read must report the earlier capture time, not now"
        );

        drop(held);

        // Once the call ends, reads go live again.
        *source.lock().unwrap() = 9;
        let refreshed = read_without_blocking(&source, &cache, |v| *v);
        assert_eq!(refreshed.value, 9);
        assert!(
            refreshed.captured_at_unix_nanos >= live.captured_at_unix_nanos,
            "a fresh read must advance the capture time"
        );
        assert_eq!(cache.lock().unwrap().value, 9);
    }

    #[test]
    #[should_panic(expected = "runtime lock poisoned")]
    fn read_without_blocking_still_panics_on_a_poisoned_runtime() {
        // A poisoned runtime means inference panicked. Serving cached stats
        // over the top of that would report a healthy node, so this must keep
        // the pre-existing panic behaviour rather than fall back to the cache.
        let source = Mutex::new(7u32);
        let cache = empty_cache(0);
        let _ = std::panic::catch_unwind(|| {
            let _guard = source.lock().unwrap();
            panic!("inference exploded");
        });
        assert!(source.is_poisoned(), "precondition: source is poisoned");
        let _ = read_without_blocking(&source, &cache, |v| *v);
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use openai_frontend::{
        ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream, ModelObject,
        OpenAiFrontendRoute, OpenAiLifecycleEvent, OpenAiLifecycleObserver, OpenAiRequestContext,
        OpenAiResult,
    };
    use tower::ServiceExt;

    use super::*;

    struct ModelsBackend;

    #[async_trait]
    impl OpenAiBackend for ModelsBackend {
        async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
            Ok(vec![ModelObject::new("embedded-model")])
        }

        async fn chat_completion(
            &self,
            _request: ChatCompletionRequest,
        ) -> OpenAiResult<ChatCompletionResponse> {
            Err(openai_frontend::OpenAiError::unsupported(
                "not used by this test",
            ))
        }

        async fn chat_completion_stream(
            &self,
            _request: ChatCompletionRequest,
            _context: OpenAiRequestContext,
        ) -> OpenAiResult<ChatCompletionStream> {
            Err(openai_frontend::OpenAiError::unsupported(
                "not used by this test",
            ))
        }
    }

    #[derive(Default)]
    struct RecordingObserver(Mutex<Vec<OpenAiLifecycleEvent>>);

    impl OpenAiLifecycleObserver for RecordingObserver {
        fn observe(&self, event: &OpenAiLifecycleEvent) {
            self.0
                .lock()
                .expect("recording observer lock poisoned")
                .push(event.clone());
        }
    }

    #[tokio::test]
    async fn optional_observer_reaches_frontend_router_and_legacy_path_stays_available() {
        let observer = Arc::new(RecordingObserver::default());
        let observed_response = openai_backend_router_with_lifecycle_observer(
            Arc::new(ModelsBackend),
            Some(Arc::clone(&observer) as Arc<dyn OpenAiLifecycleObserver>),
        )
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("router response");
        assert_eq!(observed_response.status(), StatusCode::OK);
        assert!(
            observer
                .0
                .lock()
                .expect("recording observer lock poisoned")
                .iter()
                .any(|event| matches!(
                    event,
                    OpenAiLifecycleEvent::Admitted {
                        context: openai_frontend::OpenAiLifecycleContext {
                            route: OpenAiFrontendRoute::Models,
                            ..
                        }
                    }
                ))
        );

        let legacy_response =
            openai_backend_router_with_lifecycle_observer(Arc::new(ModelsBackend), None)
                .oneshot(
                    Request::builder()
                        .uri("/v1/models")
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("router response");
        assert_eq!(legacy_response.status(), StatusCode::OK);
    }
}
