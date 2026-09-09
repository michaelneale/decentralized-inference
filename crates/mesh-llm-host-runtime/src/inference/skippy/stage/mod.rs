use std::{
    collections::HashMap,
    net::SocketAddr,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow};
use skippy_coordinator::{ClaimDecision, ClaimFence, LoadClaimRef};
use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig, StageConfig};
use skippy_server::{EmbeddedServerHandle, binary_transport::BinaryStageOptions};
use tokio::{
    sync::{Mutex, mpsc, oneshot},
    task::JoinHandle,
};

mod inventory;
#[cfg(test)]
mod tests;
mod types;

use inventory::{resolve_inventory_source, run_stage_prepare_task};
pub(crate) use types::*;

struct RunningStage {
    load: StageLoadRequest,
    server: EmbeddedServerHandle,
    package: Option<super::materialization::ResolvedStagePackage>,
}

#[derive(Default)]
struct StageControlState {
    stages: HashMap<String, RunningStage>,
    coordinator_claims: ClaimFence,
    preparations: Arc<Mutex<HashMap<String, StagePreparationStatus>>>,
    preparation_tasks: HashMap<String, StagePreparationTask>,
    readiness_probe: Option<StageReadinessProbe>,
    package_prefetcher: Option<Arc<dyn StagePackagePrefetcher>>,
    telemetry: super::SkippyTelemetryOptions,
}

struct StagePreparationTask {
    cancelled: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

struct StageReadinessProbe {
    cancelled: Arc<AtomicBool>,
    handle: JoinHandle<Result<()>>,
}

pub(crate) struct StageControlHandle {
    sender: mpsc::UnboundedSender<StageControlCommand>,
    shutdown: Option<oneshot::Sender<()>>,
    task: JoinHandle<Result<()>>,
}

impl StageControlHandle {
    pub(crate) fn sender(&self) -> mpsc::UnboundedSender<StageControlCommand> {
        self.sender.clone()
    }

    pub(crate) async fn shutdown(mut self) -> Result<()> {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        self.task.await.context("join stage control loop")?
    }
}

#[async_trait::async_trait]
pub(crate) trait StagePackagePrefetcher: Send + Sync {
    async fn prefetch_stage_package(&self, request: &StagePrepareRequest) -> Result<()>;
}

pub(crate) fn spawn_stage_control_loop(
    package_prefetcher: Option<Arc<dyn StagePackagePrefetcher>>,
    telemetry: super::SkippyTelemetryOptions,
) -> StageControlHandle {
    spawn_stage_control_loop_with_state(StageControlState {
        package_prefetcher,
        telemetry,
        ..Default::default()
    })
}

fn spawn_stage_control_loop_with_state(mut state: StageControlState) -> StageControlHandle {
    let (tx, mut rx) = mpsc::unbounded_channel::<StageControlCommand>();
    let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = &mut shutdown_rx => break,
                command = rx.recv() => {
                    let Some(command) = command else { break };
                    tokio::select! {
                        biased;
                        _ = &mut shutdown_rx => break,
                        result = state.handle(command.request) => {
                            let _ = command.resp.send(result);
                        }
                    }
                }
            }
        }
        state.shutdown().await
    });
    StageControlHandle {
        sender: tx,
        shutdown: Some(shutdown_tx),
        task,
    }
}

impl StageControlState {
    async fn shutdown(&mut self) -> Result<()> {
        for (_, task) in self.preparation_tasks.drain() {
            task.cancelled.store(true, Ordering::Release);
            task.handle.abort();
            let _ = task.handle.await;
        }
        if let Some(mut probe) = self.readiness_probe.take() {
            probe.cancelled.store(true, Ordering::Release);
            let _ = (&mut probe.handle).await;
        }
        let mut first_error = None;
        for (_, stage) in self.stages.drain() {
            if let Err(error) = stage.server.shutdown().await
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(())
    }

    async fn handle(&mut self, request: StageControlRequest) -> Result<StageControlResponse> {
        match request {
            StageControlRequest::Claim(claim) => self
                .claim(claim)
                .await
                .map(StageControlResponse::ClaimAccepted),
            StageControlRequest::Load(load) => {
                self.load(load).await.map(StageControlResponse::Ready)
            }
            StageControlRequest::LoadLocal(mut load) => {
                load.local_source_required = true;
                self.load(load).await.map(StageControlResponse::Ready)
            }
            StageControlRequest::Stop(stop) => {
                self.stop(stop).await.map(StageControlResponse::Ready)
            }
            StageControlRequest::Status(filter) => {
                Ok(StageControlResponse::Status(self.statuses(&filter)))
            }
            StageControlRequest::Inventory(request) => Ok(StageControlResponse::Inventory(
                self.inventory(request).await,
            )),
            StageControlRequest::Prepare(request) => Ok(StageControlResponse::PrepareAccepted(
                self.prepare(request).await?,
            )),
            StageControlRequest::CancelPrepare(cancel) => Ok(
                StageControlResponse::PreparationStatus(self.cancel_prepare(cancel).await),
            ),
            StageControlRequest::StatusUpdate(_status) => Ok(StageControlResponse::StatusAck(
                self.apply_status_update(_status).await,
            )),
        }
    }

    async fn claim(&mut self, claim: StageCoordinatorClaim) -> Result<StageCoordinatorClaimAck> {
        let attempted_claim = claim.clone();
        match self
            .coordinator_claims
            .accept_claim(claim, current_time_unix_ms())
        {
            ClaimDecision::Accepted {
                supersedes_term: Some(_),
                claim,
            } => {
                self.fence_stale_runtime_for_claim(&claim).await?;
                Ok(StageCoordinatorClaimAck {
                    accepted: true,
                    claim,
                    error: None,
                })
            }
            ClaimDecision::Accepted { claim, .. } => Ok(StageCoordinatorClaimAck {
                accepted: true,
                claim,
                error: None,
            }),
            ClaimDecision::Rejected { reason, .. } => Ok(StageCoordinatorClaimAck {
                accepted: false,
                claim: attempted_claim,
                error: Some(reason.to_string()),
            }),
        }
    }

    async fn inventory(&self, request: StageInventoryRequest) -> StageLayerInventory {
        let preparing_ranges = self
            .preparations
            .lock()
            .await
            .values()
            .filter(|status| {
                status.model_id == request.model_id
                    && status.package_ref == request.package_ref
                    && status.manifest_sha256 == request.manifest_sha256
            })
            .cloned()
            .collect::<Vec<_>>();
        let source_request = request.clone();
        let source =
            match tokio::task::spawn_blocking(move || resolve_inventory_source(&source_request))
                .await
            {
                Ok(source) => source,
                Err(error) => {
                    tracing::warn!(
                        model_id = request.model_id,
                        package_ref = request.package_ref,
                        error = %error,
                        "GGUF inventory verification task failed"
                    );
                    None
                }
            };
        let layer_count = source
            .as_ref()
            .map(|source| source.layer_count)
            .unwrap_or(0);
        let available_ranges = if source.is_some() && layer_count > 0 {
            vec![LayerRange {
                layer_start: 0,
                layer_end: layer_count,
            }]
        } else {
            Vec::new()
        };
        let ready_ranges = self
            .stages
            .values()
            .filter(|stage| {
                stage.load.model_id == request.model_id
                    && stage.load.package_ref == request.package_ref
                    && stage.load.manifest_sha256 == request.manifest_sha256
            })
            .map(|stage| LayerRange {
                layer_start: stage.load.layer_start,
                layer_end: stage.load.layer_end,
            })
            .collect::<Vec<_>>();
        let missing_ranges = if source.is_none() && layer_count > 0 {
            vec![LayerRange {
                layer_start: 0,
                layer_end: layer_count,
            }]
        } else {
            Vec::new()
        };
        let content_addressed_ref =
            crate::inference::skippy::is_content_addressed_gguf_ref(&request.package_ref);
        let local_source_required = crate::inference::skippy::effective_local_source_required(
            &request.model_id,
            request.runtime_profile.as_deref(),
            request.local_source_required || content_addressed_ref,
        );
        let content_addressed_local_source =
            local_source_required.then_some(content_addressed_ref && source.is_some());
        StageLayerInventory {
            model_id: request.model_id,
            package_ref: request.package_ref,
            manifest_sha256: request.manifest_sha256,
            layer_count,
            ready_ranges,
            available_ranges,
            missing_ranges,
            preparing_ranges,
            source_model_path: (!local_source_required && !content_addressed_ref)
                .then(|| {
                    source
                        .as_ref()
                        .map(|source| source.path.to_string_lossy().to_string())
                })
                .flatten(),
            source_model_bytes: source.as_ref().and_then(|source| source.bytes),
            source_model_sha256: source.as_ref().and_then(|source| source.sha256.clone()),
            content_addressed_local_source,
            source_model_kind: source
                .as_ref()
                .map(|source| source.kind)
                .unwrap_or(SourceModelKind::Unknown),
        }
    }

    async fn prepare(
        &mut self,
        mut request: StagePrepareRequest,
    ) -> Result<StagePrepareAcceptedResponse> {
        if let Some(error) = self.validate_load_claim(&request.load) {
            return Ok(StagePrepareAcceptedResponse {
                accepted: false,
                status: preparation_status_from_load(
                    &request.load,
                    StagePreparationState::Failed,
                    Some(error.clone()),
                ),
                error: Some(error),
            });
        }
        let mut load = request.load.clone();
        let verified_load = tokio::task::spawn_blocking(move || {
            crate::inference::skippy::apply_verified_local_source(&mut load).map(|_| load)
        })
        .await
        .context("join verify local-required stage prepare source task")?;
        match verified_load {
            Ok(load) => request.load = load,
            Err(error) => {
                let error = format!("{error:#}");
                return Ok(StagePrepareAcceptedResponse {
                    accepted: false,
                    status: preparation_status_from_load(
                        &request.load,
                        StagePreparationState::Failed,
                        Some(error.clone()),
                    ),
                    error: Some(error),
                });
            }
        }
        let key = stage_key(
            &request.load.topology_id,
            &request.load.run_id,
            &request.load.stage_id,
        );
        let status =
            preparation_status_from_load(&request.load, StagePreparationState::Assigned, None);
        {
            let mut preparations = self.preparations.lock().await;
            if let Some(existing) = preparations.get(&key)
                && existing.state == StagePreparationState::Cancelled
                && existing.shutdown_generation >= request.load.shutdown_generation
            {
                let mut status = existing.clone();
                status.error = Some("stale shutdown generation".to_string());
                return Ok(StagePrepareAcceptedResponse {
                    accepted: false,
                    status,
                    error: Some("stale shutdown generation".to_string()),
                });
            }
            preparations.insert(key.clone(), status.clone());
        }
        if let Some(task) = self.preparation_tasks.remove(&key) {
            task.cancelled.store(true, Ordering::Release);
            task.handle.abort();
        }
        let preparations = Arc::clone(&self.preparations);
        let package_prefetcher = self.package_prefetcher.clone();
        let cancelled = Arc::new(AtomicBool::new(false));
        let task_cancelled = Arc::clone(&cancelled);
        let task_key = key.clone();
        let handle = tokio::spawn(async move {
            run_stage_prepare_task(
                preparations,
                task_key,
                request,
                package_prefetcher,
                task_cancelled,
            )
            .await;
        });
        self.preparation_tasks
            .insert(key.clone(), StagePreparationTask { cancelled, handle });
        Ok(StagePrepareAcceptedResponse {
            accepted: true,
            status,
            error: None,
        })
    }

    async fn cancel_prepare(
        &mut self,
        cancel: StageCancelPrepareRequest,
    ) -> StagePreparationStatus {
        let key = stage_key(&cancel.topology_id, &cancel.run_id, &cancel.stage_id);
        let mut preparations = self.preparations.lock().await;
        if let Some(existing) = preparations.get(&key)
            && cancel.shutdown_generation < existing.shutdown_generation
        {
            let mut status = existing.clone();
            status.error = Some("stale shutdown generation".to_string());
            return status;
        }

        if let Some(task) = self.preparation_tasks.remove(&key) {
            task.cancelled.store(true, Ordering::Release);
            task.handle.abort();
        }

        let status = preparations
            .get(&key)
            .cloned()
            .map(|mut status| {
                status.state = StagePreparationState::Cancelled;
                status.shutdown_generation = cancel.shutdown_generation;
                status.error = None;
                status
            })
            .unwrap_or_else(|| preparation_status_from_cancel(cancel));
        preparations.insert(key, status.clone());
        status
    }

    async fn apply_status_update(&mut self, status: StagePreparationStatus) -> StageStatusAck {
        if status.topology_id.is_empty() || status.run_id.is_empty() || status.stage_id.is_empty() {
            return StageStatusAck {
                accepted: false,
                error: Some(
                    "stage status update requires topology_id, run_id, and stage_id".into(),
                ),
            };
        }
        let key = stage_key(&status.topology_id, &status.run_id, &status.stage_id);
        let mut preparations = self.preparations.lock().await;
        if preparations.get(&key).is_some_and(|existing| {
            status.shutdown_generation < existing.shutdown_generation
                || (matches!(existing.state, StagePreparationState::Cancelled)
                    && status.shutdown_generation <= existing.shutdown_generation)
        }) {
            return StageStatusAck {
                accepted: false,
                error: Some("stale shutdown generation".to_string()),
            };
        }
        preparations.insert(key, status);
        StageStatusAck {
            accepted: true,
            error: None,
        }
    }

    async fn load(&mut self, mut load: StageLoadRequest) -> Result<StageReadyResponse> {
        anyhow::ensure!(
            load.backend == "skippy",
            "unsupported stage backend '{}'",
            load.backend
        );
        if let Some(error) = self.validate_load_claim(&load) {
            return Ok(StageReadyResponse {
                accepted: false,
                status: failed_status_from_load(&load, error.clone()),
                error: Some(error),
            });
        }
        load = tokio::task::spawn_blocking(move || {
            crate::inference::skippy::apply_verified_local_source(&mut load).map(|_| load)
        })
        .await
        .context("join verify local-required stage load source task")??;
        let key = stage_key(&load.topology_id, &load.run_id, &load.stage_id);
        if let Some(existing) = self.stages.remove(&key) {
            existing.server.shutdown().await?;
        }

        let bind_addr = materialize_stage_bind_addr(parse_bind_addr(&load.bind_addr)?)?;
        let mut effective_load = load;
        effective_load.bind_addr = bind_addr.to_string();
        super::configure_materialized_stage_cache();
        let package_request = effective_load.clone();
        let mut resolved_package = None;
        if let Some(package) = tokio::task::spawn_blocking(move || {
            super::materialization::resolve_stage_load_package(&package_request)
        })
        .await
        .context("join resolve stage load package task")??
        {
            effective_load.model_path = Some(package.local_ref.clone());
            effective_load.source_model_bytes = package.source_model_bytes;
            resolved_package = Some(package);
        }
        let config = stage_config(&effective_load, resolved_package.as_ref())?;
        let server = skippy_server::start_binary_stage(BinaryStageOptions {
            config,
            topology: None,
            bind_addr,
            metrics_otlp_grpc: self.telemetry.metrics_otlp_grpc.clone(),
            telemetry_queue_capacity: self.telemetry.queue_capacity,
            telemetry_level: self.telemetry.level,
            max_inflight: effective_load.lane_count as usize,
            reply_credit_limit: None,
            async_prefill_forward: true,
            downstream_wire_condition: super::benchmark_downstream_wire_condition()?,
            downstream_connect_timeout_secs: 30,
            native_mtp_enabled: effective_load.native_mtp_enabled,
            continuous_batching: effective_load.continuous_batching,
            openai: None,
        });
        self.stages.insert(
            key.clone(),
            RunningStage {
                load: effective_load.clone(),
                server,
                package: resolved_package,
            },
        );
        self.readiness_probe = Some(start_binary_stage_ready_probe(
            bind_addr,
            stage_load_timeout(&effective_load),
        ));
        let readiness_result = {
            let probe = self
                .readiness_probe
                .as_mut()
                .expect("binary stage readiness probe must remain registered while pending");
            (&mut probe.handle)
                .await
                .context("join binary stage readiness probe")
        };
        self.readiness_probe.take();
        if let Err(error) = readiness_result.and_then(|result| result) {
            let stage = self
                .stages
                .remove(&key)
                .expect("newly started stage must remain registered while readiness is pending");
            let last_error = stage.server.status().last_error;
            let context = stage_load_failure_context(
                &effective_load,
                "binary stage did not become ready",
                last_error.as_deref(),
            );
            let _ = stage.server.shutdown().await;
            return Err(error.context(context));
        }

        let status = self
            .statuses(&StageStatusFilter {
                topology_id: Some(effective_load.topology_id.clone()),
                run_id: Some(effective_load.run_id.clone()),
                stage_id: Some(effective_load.stage_id.clone()),
            })
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("stage status missing after load"))?;
        Ok(StageReadyResponse {
            accepted: true,
            status,
            error: None,
        })
    }

    async fn stop(&mut self, stop: StageStopRequest) -> Result<StageReadyResponse> {
        let key = stage_key(&stop.topology_id, &stop.run_id, &stop.stage_id);
        let Some(existing) = self.stages.remove(&key) else {
            let status = stopped_status(&stop);
            return Ok(StageReadyResponse {
                accepted: true,
                status,
                error: None,
            });
        };
        if stop.coordinator_term < existing.load.coordinator_term {
            let current_term = existing.load.coordinator_term;
            let status = status_from_running(&existing);
            self.stages.insert(key, existing);
            return Ok(StageReadyResponse {
                accepted: false,
                status,
                error: Some(format!(
                    "stale coordinator term {} < {}",
                    stop.coordinator_term, current_term
                )),
            });
        }
        if stop.shutdown_generation < existing.load.shutdown_generation {
            let status = status_from_running(&existing);
            self.stages.insert(key, existing);
            return Ok(StageReadyResponse {
                accepted: false,
                status,
                error: Some("stale shutdown generation".to_string()),
            });
        }
        let mut status = status_from_running(&existing);
        status.state = StageRuntimeState::Stopping;
        existing.server.shutdown().await?;
        status.state = StageRuntimeState::Stopped;
        status.shutdown_generation = stop.shutdown_generation;
        Ok(StageReadyResponse {
            accepted: true,
            status,
            error: None,
        })
    }

    fn statuses(&self, filter: &StageStatusFilter) -> Vec<StageStatusSnapshot> {
        self.stages
            .values()
            .filter(|stage| filter.matches(&stage.load))
            .map(status_from_running)
            .collect()
    }

    fn validate_load_claim(&self, load: &StageLoadRequest) -> Option<String> {
        if load.coordinator_term == 0 && load.coordinator_id.is_none() {
            return None;
        }
        self.coordinator_claims
            .validate_load(&load_claim_ref(load), current_time_unix_ms())
            .err()
            .map(|error| error.to_string())
    }

    async fn fence_stale_runtime_for_claim(&mut self, claim: &StageCoordinatorClaim) -> Result<()> {
        let stale_keys = self
            .stages
            .iter()
            .filter_map(|(key, stage)| {
                (stage.load.model_id == claim.model_id
                    && stage.load.package_ref == claim.package_ref
                    && stage.load.manifest_sha256 == claim.manifest_sha256
                    && stage.load.coordinator_term < claim.coordinator_term)
                    .then_some(key.clone())
            })
            .collect::<Vec<_>>();
        for key in stale_keys {
            if let Some(stage) = self.stages.remove(&key) {
                stage.server.shutdown().await?;
            }
        }

        let mut preparations = self.preparations.lock().await;
        let stale_preparations = preparations
            .iter()
            .filter_map(|(key, status)| {
                (status.model_id == claim.model_id
                    && status.package_ref == claim.package_ref
                    && status.manifest_sha256 == claim.manifest_sha256
                    && status.coordinator_term < claim.coordinator_term)
                    .then_some(key.clone())
            })
            .collect::<Vec<_>>();
        for key in stale_preparations {
            if let Some(task) = self.preparation_tasks.remove(&key) {
                task.cancelled.store(true, Ordering::Release);
                task.handle.abort();
            }
            if let Some(status) = preparations.get_mut(&key) {
                status.state = StagePreparationState::Cancelled;
                status.error = Some("superseded by newer coordinator term".to_string());
            }
        }

        Ok(())
    }
}

impl StageStatusFilter {
    fn matches(&self, load: &StageLoadRequest) -> bool {
        self.topology_id
            .as_ref()
            .is_none_or(|value| value == &load.topology_id)
            && self
                .run_id
                .as_ref()
                .is_none_or(|value| value == &load.run_id)
            && self
                .stage_id
                .as_ref()
                .is_none_or(|value| value == &load.stage_id)
    }
}

fn stage_key(topology_id: &str, run_id: &str, stage_id: &str) -> String {
    format!("{topology_id}\n{run_id}\n{stage_id}")
}

fn current_time_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn load_claim_ref(load: &StageLoadRequest) -> LoadClaimRef {
    LoadClaimRef {
        model_id: load.model_id.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        coordinator_id: load.coordinator_id.map(|id| id.to_string()),
        coordinator_term: load.coordinator_term,
        participant_set_hash: load.participant_set_hash.clone(),
        topology_hash: load.topology_hash.clone(),
    }
}

fn parse_bind_addr(bind_addr: &str) -> Result<SocketAddr> {
    bind_addr
        .parse()
        .with_context(|| format!("parse stage bind_addr {bind_addr:?}"))
}

fn materialize_stage_bind_addr(bind_addr: SocketAddr) -> Result<SocketAddr> {
    if bind_addr.port() != 0 {
        return Ok(bind_addr);
    }
    let listener = std::net::TcpListener::bind(bind_addr)
        .with_context(|| format!("reserve ephemeral stage bind address for {bind_addr}"))?;
    listener
        .local_addr()
        .context("read reserved ephemeral stage bind address")
}

fn start_binary_stage_ready_probe(bind_addr: SocketAddr, timeout: Duration) -> StageReadinessProbe {
    let cancelled = Arc::new(AtomicBool::new(false));
    let probe_cancelled = Arc::clone(&cancelled);
    let handle = tokio::task::spawn_blocking(move || {
        probe_binary_stage_ready(bind_addr, timeout, &probe_cancelled)
    });
    StageReadinessProbe { cancelled, handle }
}

pub(crate) fn stage_load_timeout(load: &StageLoadRequest) -> Duration {
    const MIN_STAGE_LOAD_TIMEOUT_SECS: u64 = 900;
    const MAX_STAGE_LOAD_TIMEOUT_SECS: u64 = 4 * 60 * 60;
    const STAGE_LOAD_BYTES_PER_SEC: u64 = 128 * 1024 * 1024;

    let scaled_secs = load
        .source_model_bytes
        .map(|bytes| {
            bytes.saturating_add(STAGE_LOAD_BYTES_PER_SEC.saturating_sub(1))
                / STAGE_LOAD_BYTES_PER_SEC
        })
        .unwrap_or(MIN_STAGE_LOAD_TIMEOUT_SECS);
    Duration::from_secs(
        MIN_STAGE_LOAD_TIMEOUT_SECS
            .max(scaled_secs)
            .min(MAX_STAGE_LOAD_TIMEOUT_SECS),
    )
}

fn stage_load_failure_context(
    load: &StageLoadRequest,
    error: &str,
    last_error: Option<&str>,
) -> String {
    let source_bytes = load
        .source_model_bytes
        .map(|bytes| bytes.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let device = load
        .selected_device
        .as_ref()
        .map(|device| device.backend_device.as_str())
        .unwrap_or("auto");
    format!(
        "split stage load failed: model={} topology={} run={} stage={} index={} layers={}..{} mode={:?} bind={} ctx={} lanes={} source_bytes={} device={} error={} last_error={}",
        load.model_id,
        load.topology_id,
        load.run_id,
        load.stage_id,
        load.stage_index,
        load.layer_start,
        load.layer_end,
        load.load_mode,
        load.bind_addr,
        load.ctx_size,
        load.lane_count,
        source_bytes,
        device,
        error,
        last_error.unwrap_or("none"),
    )
}

fn probe_binary_stage_ready(
    bind_addr: SocketAddr,
    timeout: Duration,
    cancelled: &AtomicBool,
) -> Result<()> {
    const PROBE_IO_TIMEOUT: Duration = Duration::from_secs(2);
    let deadline = std::time::Instant::now() + timeout;
    let mut last_error = None;
    while std::time::Instant::now() < deadline {
        if cancelled.load(Ordering::Acquire) {
            return Err(anyhow!("binary stage readiness probe cancelled"));
        }
        match std::net::TcpStream::connect_timeout(&bind_addr, PROBE_IO_TIMEOUT) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                stream.set_read_timeout(Some(PROBE_IO_TIMEOUT)).ok();
                stream.set_write_timeout(Some(PROBE_IO_TIMEOUT)).ok();
                match skippy_protocol::binary::recv_ready(&mut stream) {
                    Ok(()) => return Ok(()),
                    Err(error) => {
                        last_error =
                            Some(anyhow!(error).context("binary stage ready handshake failed"));
                    }
                }
            }
            Err(error) => {
                last_error = Some(anyhow!(error).context("connect binary stage listener"));
            }
        }
        for _ in 0..25 {
            if cancelled.load(Ordering::Acquire) {
                return Err(anyhow!("binary stage readiness probe cancelled"));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("timed out waiting for binary stage ready at {bind_addr}"))
        .context(format!(
            "binary stage did not become ready at {bind_addr} before timeout"
        )))
}

fn stage_config(
    load: &StageLoadRequest,
    package: Option<&super::materialization::ResolvedStagePackage>,
) -> Result<StageConfig> {
    anyhow::ensure!(!load.topology_id.is_empty(), "topology_id is required");
    anyhow::ensure!(!load.run_id.is_empty(), "run_id is required");
    anyhow::ensure!(!load.model_id.is_empty(), "model_id is required");
    anyhow::ensure!(!load.stage_id.is_empty(), "stage_id is required");
    anyhow::ensure!(
        load.layer_start < load.layer_end,
        "invalid stage layer range"
    );
    anyhow::ensure!(load.ctx_size > 0, "ctx_size must be greater than zero");
    anyhow::ensure!(load.lane_count > 0, "lane_count must be greater than zero");
    if let Some(device) = load.selected_device.as_ref() {
        anyhow::ensure!(
            !device.backend_device.is_empty(),
            "selected backend device must not be empty"
        );
    }
    let resident_tensor_names = admitted_resident_tensor_names(load, package)?;
    let mut config = StageConfig {
        run_id: load.run_id.clone(),
        topology_id: load.topology_id.clone(),
        model_id: load.model_id.clone(),
        package_ref: Some(load.package_ref.clone()),
        manifest_sha256: Some(load.manifest_sha256.clone()),
        source_model_path: package
            .map(|package| package.source_model_path.clone())
            .or_else(|| load.model_path.clone()),
        source_model_sha256: package
            .map(|package| package.source_model_sha256.clone())
            .or_else(|| load.source_model_sha256.clone()),
        source_model_bytes: package
            .and_then(|package| package.source_model_bytes)
            .or(load.source_model_bytes),
        materialized_path: None,
        materialized_pinned: false,
        model_path: load.model_path.clone(),
        model_part_paths: package
            .map(|package| {
                package
                    .model_part_paths
                    .iter()
                    .map(|path| path.to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default(),
        projector_path: load.projector_path.clone().or_else(|| {
            package
                .and_then(|package| package.projector_path.as_ref())
                .map(|path| path.to_string_lossy().into_owned())
        }),
        projector_use_gpu: load.projector_use_gpu,
        media_marker: load.media_marker.clone(),
        image_min_tokens: load.image_min_tokens,
        image_max_tokens: load.image_max_tokens,
        batch_max_tokens: load.batch_max_tokens,
        glm_dsa_policy: load.glm_dsa_policy,
        generation_signal_window: load.generation_signal_window,
        activation_codec: load.activation_codec,
        activation_codec_policy: load.activation_codec_policy,
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        ctx_size: load.ctx_size,
        lane_count: load.lane_count,
        n_batch: load.n_batch,
        n_ubatch: load.n_ubatch,
        n_gpu_layers: load.n_gpu_layers,
        mmap: load.mmap,
        mlock: load.mlock,
        repack: load.runtime_settings.repack,
        op_offload: load.runtime_settings.op_offload,
        no_host_buffer: load.runtime_settings.no_host_buffer,
        check_tensors: load.runtime_settings.check_tensors,
        direct_io: load.runtime_settings.direct_io,
        main_gpu: load.runtime_settings.main_gpu,
        split_mode: load.runtime_settings.split_mode,
        cache_type_k: empty_to_default(&load.cache_type_k, "f16"),
        cache_type_v: empty_to_default(&load.cache_type_v, "f16"),
        flash_attn_type: load.flash_attn_type,
        kv_offload: load.runtime_settings.kv_offload,
        kv_unified: load.runtime_settings.kv_unified,
        swa_full: load.runtime_settings.swa_full,
        cache_idle_slots: load.runtime_settings.cache_idle_slots,
        filter_tensors_on_load: matches!(
            load.load_mode,
            LoadMode::RuntimeSlice | LoadMode::LayerPackage
        ),
        resident_tensor_names,
        checkpoint_quantization: None,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        selected_device: load.selected_device.clone(),
        kv_cache: None,
        native_mtp_enabled: load.native_mtp_enabled,
        load_mode: load.load_mode.clone(),
        bind_addr: load.bind_addr.clone(),
        upstream: load.upstream.as_ref().map(peer_config),
        downstream: load.downstream.as_ref().map(peer_config),
    };
    let family_policy = super::family_policy_for_stage_config(&config);
    config.kv_cache = package.map_or_else(
        || family_policy.stage_kv_cache_config_for_stage(&config),
        |package| {
            family_policy.stage_kv_cache_config_for_package(&config, Path::new(&package.local_ref))
        },
    );
    Ok(config)
}

pub(crate) fn admitted_resident_tensor_names(
    load: &StageLoadRequest,
    package: Option<&super::materialization::ResolvedStagePackage>,
) -> Result<Vec<String>> {
    let package_ref = package
        .map(|package| package.local_ref.as_str())
        .unwrap_or(&load.package_ref);
    let manifest: skippy_package_format::PackageManifest = if super::is_package_v2_ref(package_ref)
    {
        let manifest_path = Path::new(package_ref).join("model-package.json");
        let manifest =
            serde_json::from_slice(&std::fs::read(&manifest_path).with_context(|| {
                format!("read package-v2 manifest {}", manifest_path.display())
            })?)
            .with_context(|| format!("parse package-v2 manifest {}", manifest_path.display()))?;
        skippy_model::package_carrier::resolve_package_carrier_from_dir(manifest, package_ref)
            .context("resolve package-v2 metadata carrier")?
    } else if super::is_content_addressed_gguf_ref(&load.package_ref) {
        let model_path = load
            .model_path
            .as_deref()
            .context("direct-GGUF stage load requires a verified local model path")?;
        let identity = super::verify_registered_content_source(
            &load.model_id,
            &load.package_ref,
            &load.manifest_sha256,
            load.source_model_sha256
                .as_deref()
                .context("direct-GGUF stage load requires a source SHA-256")?,
        )?;
        anyhow::ensure!(
            identity.source_model_path == Path::new(model_path),
            "verified direct-GGUF planning path differs from the stage load path"
        );
        super::direct_gguf_planning_manifest_from_identity(&load.model_id, &identity)?.0
    } else if load.package_ref.starts_with("gguf://") {
        let model_path = load
            .model_path
            .as_deref()
            .context("direct-GGUF stage load requires a local model path")?;
        let identity = super::synthetic_direct_gguf_package(&load.model_id, Path::new(model_path))?;
        super::direct_gguf_planning_manifest_from_identity(&load.model_id, &identity)?.0
    } else {
        return Ok(Vec::new());
    };
    let package_id = manifest
        .computed_package_id()
        .context("compute planning identity for resident tensor binding")?;
    anyhow::ensure!(
        package_id == load.admission.package_id,
        "stage admission package identity differs from the local planning manifest"
    );
    let mut names = manifest
        .materialization_tensors(&load.admission.resident_tensor_ids)
        .map_err(|error| anyhow!(error.to_string()))?
        .into_iter()
        .map(|tensor| tensor.native_name.to_string())
        .collect::<Vec<_>>();
    names.sort();
    anyhow::ensure!(
        !names.is_empty() && names.windows(2).all(|window| window[0] < window[1]),
        "admitted resident tensor names must be non-empty, strictly sorted, and unique"
    );
    Ok(names)
}

fn peer_config(peer: &StagePeerDescriptor) -> PeerConfig {
    PeerConfig {
        stage_id: peer.stage_id.clone(),
        stage_index: peer.stage_index,
        endpoint: peer.endpoint.clone(),
    }
}

fn empty_to_default(value: &str, default: &str) -> String {
    if value.is_empty() {
        default.to_string()
    } else {
        value.to_string()
    }
}

fn status_from_running(stage: &RunningStage) -> StageStatusSnapshot {
    let server = stage.server.status();
    let state = match server.state {
        skippy_server::EmbeddedState::Starting => StageRuntimeState::Starting,
        skippy_server::EmbeddedState::Ready => StageRuntimeState::Ready,
        skippy_server::EmbeddedState::Stopping => StageRuntimeState::Stopping,
        skippy_server::EmbeddedState::Stopped => StageRuntimeState::Stopped,
        skippy_server::EmbeddedState::Failed => StageRuntimeState::Failed,
    };
    let content_addressed_ref =
        crate::inference::skippy::is_content_addressed_gguf_ref(&stage.load.package_ref);
    StageStatusSnapshot {
        topology_id: stage.load.topology_id.clone(),
        run_id: stage.load.run_id.clone(),
        model_id: stage.load.model_id.clone(),
        backend: stage.load.backend.clone(),
        package_ref: Some(stage.load.package_ref.clone()),
        manifest_sha256: Some(stage.load.manifest_sha256.clone()),
        source_model_path: (!content_addressed_ref)
            .then(|| {
                stage
                    .package
                    .as_ref()
                    .map(|package| package.source_model_path.clone())
                    .or_else(|| stage.load.model_path.clone())
            })
            .flatten(),
        source_model_sha256: stage
            .package
            .as_ref()
            .map(|package| package.source_model_sha256.clone())
            .or_else(|| stage.load.source_model_sha256.clone()),
        source_model_bytes: stage
            .package
            .as_ref()
            .and_then(|package| package.source_model_bytes)
            .or(stage.load.source_model_bytes),
        materialized_path: None,
        materialized_pinned: false,
        projector_path: (!stage.load.local_source_required && !content_addressed_ref)
            .then(|| stage.load.projector_path.clone())
            .flatten(),
        stage_id: stage.load.stage_id.clone(),
        stage_index: stage.load.stage_index,
        layer_start: stage.load.layer_start,
        layer_end: stage.load.layer_end,
        admission: Some(stage.load.admission.clone()),
        activation_codec: stage.load.activation_codec,
        activation_codec_policy: stage.load.activation_codec_policy,
        state,
        bind_addr: server.bind_addr.to_string(),
        input_activation_boundary: server.input_activation_boundary,
        output_activation_boundary: server.output_activation_boundary,
        selected_device: stage.load.selected_device.clone(),
        ctx_size: stage.load.ctx_size,
        lane_count: stage.load.lane_count,
        n_batch: stage.load.n_batch,
        n_ubatch: stage.load.n_ubatch,
        flash_attn_type: stage.load.flash_attn_type,
        error: server.last_error.clone(),
        shutdown_generation: stage.load.shutdown_generation,
        coordinator_term: stage.load.coordinator_term,
        coordinator_id: stage.load.coordinator_id,
        lease_until_unix_ms: stage.load.lease_until_unix_ms,
    }
}

fn stopped_status(stop: &StageStopRequest) -> StageStatusSnapshot {
    StageStatusSnapshot {
        topology_id: stop.topology_id.clone(),
        run_id: stop.run_id.clone(),
        model_id: String::new(),
        backend: "skippy".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        projector_path: None,
        stage_id: stop.stage_id.clone(),
        stage_index: 0,
        layer_start: 0,
        layer_end: 0,
        admission: None,
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        activation_codec_policy: skippy_protocol::StageActivationCodecPolicy::default(),
        state: StageRuntimeState::Stopped,
        bind_addr: String::new(),
        input_activation_boundary: None,
        output_activation_boundary: None,
        selected_device: None,
        ctx_size: 0,
        lane_count: 0,
        n_batch: None,
        n_ubatch: None,
        flash_attn_type: FlashAttentionType::Auto,
        error: None,
        shutdown_generation: stop.shutdown_generation,
        coordinator_term: stop.coordinator_term,
        coordinator_id: None,
        lease_until_unix_ms: 0,
    }
}

fn failed_status_from_load(load: &StageLoadRequest, error: String) -> StageStatusSnapshot {
    let content_addressed_ref =
        crate::inference::skippy::is_content_addressed_gguf_ref(&load.package_ref);
    StageStatusSnapshot {
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        model_id: load.model_id.clone(),
        backend: load.backend.clone(),
        package_ref: Some(load.package_ref.clone()),
        manifest_sha256: Some(load.manifest_sha256.clone()),
        source_model_path: (!content_addressed_ref)
            .then(|| load.model_path.clone())
            .flatten(),
        source_model_sha256: load.source_model_sha256.clone(),
        source_model_bytes: load.source_model_bytes,
        materialized_path: None,
        materialized_pinned: false,
        projector_path: (!load.local_source_required && !content_addressed_ref)
            .then(|| load.projector_path.clone())
            .flatten(),
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        admission: Some(load.admission.clone()),
        activation_codec: load.activation_codec,
        activation_codec_policy: load.activation_codec_policy,
        state: StageRuntimeState::Failed,
        bind_addr: load.bind_addr.clone(),
        input_activation_boundary: None,
        output_activation_boundary: None,
        selected_device: load.selected_device.clone(),
        ctx_size: load.ctx_size,
        lane_count: load.lane_count,
        n_batch: load.n_batch,
        n_ubatch: load.n_ubatch,
        flash_attn_type: load.flash_attn_type,
        error: Some(error),
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load.coordinator_id,
        lease_until_unix_ms: load.lease_until_unix_ms,
    }
}

fn preparation_status_from_load(
    load: &StageLoadRequest,
    state: StagePreparationState,
    error: Option<String>,
) -> StagePreparationStatus {
    StagePreparationStatus {
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        model_id: load.model_id.clone(),
        backend: load.backend.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        admission: Some(load.admission.clone()),
        activation_codec: load.activation_codec,
        activation_codec_policy: load.activation_codec_policy,
        state,
        bytes_done: None,
        bytes_total: None,
        bind_addr: None,
        error,
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load.coordinator_id,
        lease_until_unix_ms: load.lease_until_unix_ms,
    }
}

fn preparation_status_from_cancel(cancel: StageCancelPrepareRequest) -> StagePreparationStatus {
    StagePreparationStatus {
        topology_id: cancel.topology_id,
        run_id: cancel.run_id,
        model_id: String::new(),
        backend: "skippy".to_string(),
        package_ref: String::new(),
        manifest_sha256: String::new(),
        stage_id: cancel.stage_id,
        stage_index: 0,
        layer_start: 0,
        layer_end: 0,
        admission: None,
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        activation_codec_policy: skippy_protocol::StageActivationCodecPolicy::default(),
        state: StagePreparationState::Cancelled,
        bytes_done: None,
        bytes_total: None,
        bind_addr: None,
        error: None,
        shutdown_generation: cancel.shutdown_generation,
        coordinator_term: 0,
        coordinator_id: None,
        lease_until_unix_ms: 0,
    }
}
