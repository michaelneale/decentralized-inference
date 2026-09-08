use super::loading::stage_health_ticks;
use super::recovery::{
    SplitLossRecoveryDecision, SplitWithdrawGraceAction,
    split_active_stage_nodes_pending_eligibility, split_connected_node_ids,
    split_locked_loss_recovery_decision, split_loss_recovery_decision,
    split_missing_active_stage_nodes, split_participants_meet_minimum, split_stages_meet_minimum,
    split_unavailable_active_stage_nodes, split_withdraw_grace_action,
};
use super::{
    OpenAiGuardrailPolicyHandle, RuntimeSliceStagePlan, SPLIT_PARTICIPANT_STABLE_FOR,
    SplitCoordinatorAck, SplitCoordinatorEvent, SplitCoordinatorLocalFallbackEvent,
    SplitCoordinatorReplaceEvent, SplitCoordinatorWithdrawEvent, SplitGenerationLoadSpec,
    SplitParticipant, SplitParticipantExclusion, SplitTopologyResourceInputs,
    collect_split_participants, load_split_runtime_generation, model_fits_runtime_capacity,
    now_unix_nanos, plan_runtime_slice_topology_with_resources,
    split_coordinator_lease_until_unix_ms, split_participant_exclusion_labels,
    split_participant_labels, split_participants_for_stages, split_stage_plan_labels,
};
use crate::inference::{election, skippy};
use crate::mesh;
use crate::plugin;
use crate::runtime::local_package::{
    SPLIT_DEFAULT_MIN_PARTICIPANTS, SplitParticipantSnapshot, split_node_labels,
};
use crate::runtime::survey;
use anyhow::Result;
use skippy_protocol::FlashAttentionType;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

/// Grace period before withdrawing a split whose only remaining recovery path is
/// full teardown. Holding the topology through transient peer loss avoids
/// forcing a manual restart while still withdrawing a genuinely dead split.
const SPLIT_STAGE_LOSS_WITHDRAW_GRACE: Duration = Duration::from_secs(75);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyGeneration {
    pub(super) topology_id: String,
    pub(super) run_id: String,
    pub(super) generation: u64,
    pub(super) coordinator_term: u64,
    pub(super) lease_until_unix_ms: u64,
    pub(super) participants: Vec<SplitParticipant>,
    pub(super) stages: Vec<RuntimeSliceStagePlan>,
    pub(super) admissions: BTreeMap<String, skippy_protocol::StageAdmissionDescriptor>,
    pub(super) activation_codec: skippy_protocol::StageActivationCodec,
    pub(super) activation_codec_policy: skippy_protocol::StageActivationCodecPolicy,
}

impl SplitTopologyGeneration {
    pub(super) fn new(
        topology_id: String,
        run_id: String,
        generation: u64,
        participants: Vec<SplitParticipant>,
        stages: Vec<RuntimeSliceStagePlan>,
    ) -> Self {
        Self {
            topology_id,
            run_id,
            generation,
            coordinator_term: now_unix_nanos().max(1) as u64,
            lease_until_unix_ms: split_coordinator_lease_until_unix_ms(),
            participants,
            stages,
            admissions: BTreeMap::new(),
            activation_codec: skippy_protocol::StageActivationCodec::default(),
            // New split generations negotiate a lossless codec per realized
            // activation frame. RawF32 remains the mandatory fallback, so
            // frames only compact when their complete payload (including
            // sidebands) round-trips exactly through BF16 or F16.
            activation_codec_policy: skippy_protocol::StageActivationCodecPolicy::AutoLosslessV1,
        }
    }

    pub(super) fn with_admissions(
        mut self,
        admissions: Vec<skippy_protocol::StageAdmissionDescriptor>,
    ) -> Result<Self> {
        anyhow::ensure!(
            admissions.len() == self.stages.len(),
            "stage admission count differs from topology stage count"
        );
        for (stage, admission) in self.stages.iter().zip(admissions) {
            anyhow::ensure!(
                admission.layer_start == stage.layer_start
                    && admission.layer_end == stage.layer_end,
                "stage {} admission range differs from its topology placement",
                stage.stage_id
            );
            let proto: skippy_protocol::proto::stage::StageAdmissionDescriptor =
                admission.clone().into();
            skippy_protocol::validate_stage_admission_descriptor(&proto)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            anyhow::ensure!(
                self.admissions
                    .insert(stage.stage_id.clone(), admission)
                    .is_none(),
                "topology contains duplicate stage id {}",
                stage.stage_id
            );
        }
        Ok(self)
    }
}

pub(super) struct SplitTopologyCoordinator {
    pub(super) node: mesh::Node,
    pub(super) mesh_config: plugin::MeshConfig,
    pub(super) model_name: String,
    pub(super) model_path: PathBuf,
    pub(super) model_ref: String,
    pub(super) config_model_id: Option<String>,
    pub(super) runtime_profile: String,
    pub(super) package: skippy::SkippyPackageIdentity,
    pub(super) compact_meta: crate::models::gguf::GgufCompactMeta,
    pub(super) active: SplitTopologyGeneration,
    pub(super) projector_path: Option<String>,
    pub(super) ctx_size: u32,
    pub(super) topology_resources: SplitTopologyResourceInputs,
    pub(super) cache_type_k_override: Option<String>,
    pub(super) cache_type_v_override: Option<String>,
    pub(super) n_batch_override: Option<u32>,
    pub(super) n_ubatch_override: Option<u32>,
    pub(super) flash_attention_override: FlashAttentionType,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    pub(super) capacity_budget_bytes: Option<u64>,
    pub(super) pinned_gpu: Option<crate::runtime::StartupPinnedGpuTarget>,
    pub(super) device_override: Option<String>,
    pub(super) slots: usize,
    pub(super) skippy_telemetry: skippy::SkippyTelemetryOptions,
    pub(super) survey_telemetry: survey::SurveyTelemetry,
    pub(super) event_tx: tokio::sync::mpsc::Sender<SplitCoordinatorEvent>,
    /// First observation of a withdraw-only stage loss. Cleared whenever the
    /// split is healthy or has a viable replacement or fallback path.
    pub(super) stage_loss_first_seen: Option<Instant>,
    /// The `unavailable_stage_nodes` set observed on the previous
    /// `evaluate_replan` tick (runtime-events flap/recovery classification
    /// only, `topology_events::classify_connection_changes`; carries no
    /// routing/election authority of its own).
    pub(super) previously_unavailable_stage_nodes: Vec<iroh::EndpointId>,
    /// A locked topology may be withdrawn after stage loss, but never replaced
    /// or collapsed to a local fallback.
    pub(super) topology_locked: bool,
    pub(super) local_source_required: bool,
    pub(super) health_interval: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SplitReplanDecision {
    Keep,
    Candidate,
}

pub(super) fn spawn_split_topology_coordinator(
    coordinator: SplitTopologyCoordinator,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(Box::pin(coordinator.run()))
}

impl SplitTopologyCoordinator {
    async fn run(mut self) {
        let mut peer_rx = self.node.peer_change_rx.clone();
        let mut health_tick = stage_health_ticks(self.health_interval);
        health_tick.tick().await;
        tracing::info!(
            model_ref = self.model_ref,
            topology_id = self.active.topology_id,
            generation = self.active.generation,
            stages = ?split_stage_plan_labels(&self.active.stages),
            participants = ?split_participant_labels(&self.active.participants),
            "split topology coordinator active"
        );

        loop {
            tokio::select! {
                changed = peer_rx.changed() => {
                    if !self.handle_peer_change(&mut peer_rx, changed).await {
                        break;
                    }
                }
                _ = health_tick.tick() => {
                    if !self.evaluate_replan("periodic_check").await {
                        break;
                    }
                }
            }
        }
    }

    async fn handle_peer_change<T>(
        &mut self,
        peer_rx: &mut tokio::sync::watch::Receiver<T>,
        changed: Result<(), tokio::sync::watch::error::RecvError>,
    ) -> bool {
        if changed.is_err() {
            tracing::debug!(
                model_ref = self.model_ref,
                "split topology coordinator peer watch closed"
            );
            return false;
        }
        tokio::time::sleep(SPLIT_PARTICIPANT_STABLE_FOR).await;
        drain_split_peer_changes(peer_rx);
        self.evaluate_replan("membership_changed").await
    }

    async fn evaluate_replan(&mut self, reason: &'static str) -> bool {
        let local_capacity_override = split_replan_local_capacity_override(
            self.capacity_budget_bytes,
            self.pinned_gpu
                .as_ref()
                .map(|gpu| gpu.allocatable_vram_bytes()),
        );
        let snapshot = collect_split_participants(
            &self.node,
            &self.model_name,
            &self.model_ref,
            &self.runtime_profile,
            &self.package,
            local_capacity_override,
            self.local_source_required,
        )
        .await;
        let connected_node_ids = split_connected_node_ids(&self.node).await;
        self.node
            .refresh_stage_runtime_statuses(Duration::from_secs(2))
            .await;
        let runtime_statuses = self.node.stage_runtime_statuses().await;
        let missing_stage_nodes =
            split_missing_active_stage_nodes(&self.active, &connected_node_ids);
        let unavailable_stage_nodes = split_unavailable_active_stage_nodes(
            &self.active,
            &connected_node_ids,
            &runtime_statuses,
        );
        let pending_eligibility_nodes = split_active_stage_nodes_pending_eligibility(
            &self.active,
            &connected_node_ids,
            &snapshot.participants,
            &unavailable_stage_nodes,
        );
        let candidate = if self.topology_locked {
            None
        } else if pending_eligibility_nodes.is_empty() {
            self.replan_candidate(reason, &snapshot, &unavailable_stage_nodes)
        } else {
            tracing::debug!(
                model_ref = self.model_ref,
                reason,
                stage_nodes = ?split_node_labels(&pending_eligibility_nodes),
                "split topology replan deferred while active stage peer eligibility settles"
            );
            None
        };

        if let Some(should_continue) = self
            .handle_loss_recovery(
                reason,
                &connected_node_ids,
                &missing_stage_nodes,
                &unavailable_stage_nodes,
                candidate.as_ref(),
            )
            .await
        {
            return should_continue;
        }

        if self.topology_locked {
            return true;
        }
        self.apply_replan_candidate(reason, snapshot.participants.len(), candidate)
            .await
    }

    fn replan_candidate(
        &self,
        reason: &'static str,
        snapshot: &SplitParticipantSnapshot,
        unavailable_stage_nodes: &[iroh::EndpointId],
    ) -> Option<SplitTopologyGeneration> {
        let planned_participants =
            split_recovery_candidate_participants(&snapshot.participants, unavailable_stage_nodes);
        if !split_participants_meet_minimum(&planned_participants) {
            log_split_replan_quorum_not_met(
                &self.model_ref,
                reason,
                &snapshot.participants,
                &snapshot.excluded,
            );
            return None;
        }
        self.try_build_local_replan_candidate(reason, &planned_participants, &snapshot.excluded)
    }

    fn try_build_local_replan_candidate(
        &self,
        reason: &'static str,
        planned_participants: &[SplitParticipant],
        excluded: &[SplitParticipantExclusion],
    ) -> Option<SplitTopologyGeneration> {
        match self.plan_replan_candidate(planned_participants) {
            Ok(candidate) if split_candidate_stage0_is_local(self.node.id(), &candidate) => {
                Some(candidate)
            }
            Ok(candidate) => {
                tracing::debug!(
                    model_ref = self.model_ref,
                    reason,
                    candidate_stages = ?split_stage_plan_labels(&candidate.stages),
                    "split topology replan skipped; stage 0 would move to another node"
                );
                None
            }
            Err(err) => {
                tracing::warn!(
                    model_ref = self.model_ref,
                    reason,
                    error = %err,
                    participants = ?split_participant_labels(planned_participants),
                    excluded = ?split_participant_exclusion_labels(excluded),
                    "split topology replan candidate failed"
                );
                None
            }
        }
    }

    async fn handle_loss_recovery(
        &mut self,
        reason: &'static str,
        connected_node_ids: &[iroh::EndpointId],
        missing_stage_nodes: &[iroh::EndpointId],
        unavailable_stage_nodes: &[iroh::EndpointId],
        candidate: Option<&SplitTopologyGeneration>,
    ) -> Option<bool> {
        // Computed unconditionally (cheap VRAM-vs-model-size arithmetic, no
        // I/O): the local-fallback-capacity boolean is the existing
        // §8.14 `resource_pressure_changed` input regardless of whether the
        // locked/unlocked branch below happens to consume it for a
        // decision this tick.
        let local_capacity_available = self.local_model_fits();
        let decision = if self.topology_locked {
            split_locked_loss_recovery_decision(
                &self.active,
                connected_node_ids,
                unavailable_stage_nodes,
            )
        } else {
            split_loss_recovery_decision(
                &self.active,
                connected_node_ids,
                unavailable_stage_nodes,
                candidate,
                local_capacity_available,
            )
        };
        if !matches!(decision, SplitLossRecoveryDecision::Withdraw) {
            self.stage_loss_first_seen = None;
        }
        super::topology_events::report_recovery_decision(
            &self.active.topology_id,
            matches!(decision, SplitLossRecoveryDecision::NoActiveStageLoss),
            matches!(decision, SplitLossRecoveryDecision::Withdraw),
            &self.previously_unavailable_stage_nodes,
            unavailable_stage_nodes,
            local_capacity_available,
        );
        self.previously_unavailable_stage_nodes = unavailable_stage_nodes.to_vec();
        match decision {
            SplitLossRecoveryDecision::NoActiveStageLoss => None,
            SplitLossRecoveryDecision::ReplacementSplit => {
                let candidate = candidate.expect("replacement split decision requires a candidate");
                Some(
                    self.handle_replacement_split_loss(
                        reason,
                        candidate,
                        missing_stage_nodes,
                        unavailable_stage_nodes,
                    )
                    .await,
                )
            }
            SplitLossRecoveryDecision::LocalFallback => Some(
                self.handle_local_fallback_loss(
                    reason,
                    missing_stage_nodes,
                    unavailable_stage_nodes,
                )
                .await,
            ),
            SplitLossRecoveryDecision::Withdraw => {
                let now = Instant::now();
                if self.stage_loss_first_seen.is_none() {
                    self.stage_loss_first_seen = Some(now);
                }
                match split_withdraw_grace_action(
                    self.stage_loss_first_seen,
                    now,
                    SPLIT_STAGE_LOSS_WITHDRAW_GRACE,
                ) {
                    SplitWithdrawGraceAction::Defer => {
                        tracing::warn!(
                            model_ref = self.model_ref,
                            reason,
                            topology_id = self.active.topology_id,
                            generation = self.active.generation,
                            missing_stage_nodes = ?split_node_labels(missing_stage_nodes),
                            unavailable_stage_nodes = ?split_node_labels(unavailable_stage_nodes),
                            grace_secs = SPLIT_STAGE_LOSS_WITHDRAW_GRACE.as_secs(),
                            "split topology lost an active stage peer; holding topology through grace period before withdrawing"
                        );
                        Some(true)
                    }
                    SplitWithdrawGraceAction::Withdraw => {
                        self.stage_loss_first_seen = None;
                        Some(
                            self.handle_withdraw_loss(
                                reason,
                                missing_stage_nodes,
                                unavailable_stage_nodes,
                            )
                            .await,
                        )
                    }
                }
            }
        }
    }

    async fn handle_replacement_split_loss(
        &mut self,
        reason: &'static str,
        candidate: &SplitTopologyGeneration,
        missing_stage_nodes: &[iroh::EndpointId],
        unavailable_stage_nodes: &[iroh::EndpointId],
    ) -> bool {
        tracing::info!(
            model_ref = self.model_ref,
            reason,
            active_topology_id = self.active.topology_id,
            active_generation = self.active.generation,
            candidate_topology_id = candidate.topology_id,
            candidate_generation = candidate.generation,
            missing_stage_nodes = ?split_node_labels(missing_stage_nodes),
            unavailable_stage_nodes = ?split_node_labels(unavailable_stage_nodes),
            active_stages = ?split_stage_plan_labels(&self.active.stages),
            candidate_stages = ?split_stage_plan_labels(&candidate.stages),
            participants = ?split_participant_labels(&candidate.participants),
            "split topology lost an active stage peer; loading replacement split generation"
        );
        match self
            .load_and_publish_candidate(reason, candidate.clone())
            .await
        {
            Ok(()) => true,
            Err(err) => {
                tracing::warn!(
                    model_ref = self.model_ref,
                    reason,
                    error = %err,
                    "split topology replacement failed during load-and-cutover"
                );
                self.publish_loss_fallback(reason, unavailable_stage_nodes.to_vec())
                    .await
            }
        }
    }

    async fn handle_local_fallback_loss(
        &mut self,
        reason: &'static str,
        missing_stage_nodes: &[iroh::EndpointId],
        unavailable_stage_nodes: &[iroh::EndpointId],
    ) -> bool {
        tracing::warn!(
            model_ref = self.model_ref,
            reason,
            topology_id = self.active.topology_id,
            generation = self.active.generation,
            missing_stage_nodes = ?split_node_labels(missing_stage_nodes),
            unavailable_stage_nodes = ?split_node_labels(unavailable_stage_nodes),
            "split topology lost an active stage peer; requesting local runtime fallback"
        );
        self.publish_local_fallback(reason, unavailable_stage_nodes.to_vec())
            .await
    }

    async fn handle_withdraw_loss(
        &mut self,
        reason: &'static str,
        missing_stage_nodes: &[iroh::EndpointId],
        unavailable_stage_nodes: &[iroh::EndpointId],
    ) -> bool {
        tracing::warn!(
            model_ref = self.model_ref,
            reason,
            topology_id = self.active.topology_id,
            generation = self.active.generation,
            missing_stage_nodes = ?split_node_labels(missing_stage_nodes),
            unavailable_stage_nodes = ?split_node_labels(unavailable_stage_nodes),
            "split topology lost an active stage peer and no replacement path is available; withdrawing active generation"
        );
        self.publish_withdrawal(reason, unavailable_stage_nodes.to_vec())
            .await
    }

    async fn apply_replan_candidate(
        &mut self,
        reason: &'static str,
        participant_count: usize,
        candidate: Option<SplitTopologyGeneration>,
    ) -> bool {
        let Some(candidate) = split_candidate_for_replan(participant_count, candidate) else {
            return true;
        };

        let (replan_decision, replan_decision_reason) =
            split_replan_decision_with_reason(&self.active, &candidate);
        match replan_decision {
            SplitReplanDecision::Keep => {
                self.log_replan_keep(reason, &candidate, replan_decision_reason);
            }
            SplitReplanDecision::Candidate => {
                self.apply_selected_replan_candidate(reason, candidate, replan_decision_reason)
                    .await;
            }
        }
        true
    }

    fn log_replan_keep(
        &self,
        reason: &'static str,
        candidate: &SplitTopologyGeneration,
        decision_reason: &'static str,
    ) {
        tracing::debug!(
            model_ref = self.model_ref,
            reason,
            decision_reason,
            active_generation = self.active.generation,
            active_stages = self.active.stages.len(),
            candidate_stages = candidate.stages.len(),
            active_participants = self.active.participants.len(),
            candidate_participants = candidate.participants.len(),
            "split topology replan skipped; candidate is not materially better"
        );
    }

    async fn apply_selected_replan_candidate(
        &mut self,
        reason: &'static str,
        candidate: SplitTopologyGeneration,
        decision_reason: &'static str,
    ) {
        tracing::info!(
            model_ref = self.model_ref,
            reason,
            decision_reason,
            active_topology_id = self.active.topology_id,
            active_generation = self.active.generation,
            candidate_topology_id = candidate.topology_id,
            candidate_generation = candidate.generation,
            active_stages = ?split_stage_plan_labels(&self.active.stages),
            candidate_stages = ?split_stage_plan_labels(&candidate.stages),
            participants = ?split_participant_labels(&candidate.participants),
            "split topology replan candidate accepted; loading candidate generation"
        );
        if let Err(err) = self.load_and_publish_candidate(reason, candidate).await {
            tracing::warn!(
                model_ref = self.model_ref,
                reason,
                error = %err,
                "split topology replan candidate failed during load-and-cutover"
            );
        }
    }

    async fn publish_loss_fallback(
        &mut self,
        reason: &'static str,
        unavailable_stage_nodes: Vec<iroh::EndpointId>,
    ) -> bool {
        if self.local_model_fits() {
            return self
                .publish_local_fallback(reason, unavailable_stage_nodes.clone())
                .await;
        }
        self.publish_withdrawal(reason, unavailable_stage_nodes)
            .await
    }

    async fn publish_local_fallback(
        &mut self,
        reason: &'static str,
        unavailable_stage_nodes: Vec<iroh::EndpointId>,
    ) -> bool {
        match self
            .request_local_fallback(reason, unavailable_stage_nodes)
            .await
        {
            Err(err) => {
                tracing::warn!(
                    model_ref = self.model_ref,
                    reason,
                    error = %err,
                    "failed to publish split topology local fallback request"
                );
                true
            }
            _ => false,
        }
    }

    async fn publish_withdrawal(
        &mut self,
        reason: &'static str,
        unavailable_stage_nodes: Vec<iroh::EndpointId>,
    ) -> bool {
        let topology_op =
            super::topology_events::TopologyWithdrawalOperation::begin(&self.active.topology_id);
        match self
            .withdraw_active_generation(reason, unavailable_stage_nodes)
            .await
        {
            Err(err) => {
                tracing::warn!(
                    model_ref = self.model_ref,
                    reason,
                    error = %err,
                    "failed to publish split topology withdrawal"
                );
                true
            }
            _ => {
                topology_op.withdrawn();
                false
            }
        }
    }

    fn plan_replan_candidate(
        &self,
        planned_participants: &[SplitParticipant],
    ) -> Result<SplitTopologyGeneration> {
        let generation = self.active.generation.saturating_add(1);
        let run_id = format!("mesh-split-{}-g{}", now_unix_nanos(), generation);
        let topology_id = format!("topology-{run_id}");
        let resources = SplitTopologyResourceInputs {
            ctx_size_override: Some(self.ctx_size),
            parallel_override: Some(self.slots),
            ..self.topology_resources.clone()
        };
        let planned = plan_runtime_slice_topology_with_resources(
            &topology_id,
            &self.model_ref,
            &self.package,
            planned_participants,
            &[],
            resources,
        )?;
        let admissions = super::realize_split_stage_admissions(
            &self.model_path,
            &self.model_ref,
            &self.package,
            &planned,
            &self.runtime_profile,
        )?;
        let stages = planned.stages;
        let participants = split_participants_for_stages(planned_participants, &stages);
        anyhow::ensure!(
            split_stages_meet_minimum(&stages),
            "split runtime needs at least two stage participants"
        );
        SplitTopologyGeneration::new(topology_id, run_id, generation, participants, stages)
            .with_admissions(admissions)
    }

    fn local_model_fits(&self) -> bool {
        let local_capacity = split_replan_local_capacity_override(
            self.capacity_budget_bytes,
            self.pinned_gpu
                .as_ref()
                .map(|gpu| gpu.allocatable_vram_bytes()),
        )
        .unwrap_or_else(|| self.node.vram_bytes());
        // Use the package's source model bytes when available — layer-package
        // refs use `hf://` pseudo-paths that `total_model_bytes` cannot stat.
        let model_bytes = if self.package.source_model_bytes > 0 {
            self.package.source_model_bytes
        } else {
            election::total_model_bytes(&self.model_path)
        };
        model_fits_runtime_capacity(model_bytes, local_capacity)
    }

    async fn load_and_publish_candidate(
        &mut self,
        reason: &'static str,
        candidate: SplitTopologyGeneration,
    ) -> Result<()> {
        let previous = self.active.clone();
        let loaded = load_split_runtime_generation(SplitGenerationLoadSpec {
            node: &self.node,
            mesh_config: &self.mesh_config,
            model_ref: &self.model_ref,
            config_model_id: self.config_model_id.as_deref(),
            runtime_profile: &self.runtime_profile,
            model_path: &self.model_path,
            package: &self.package,
            generation: &candidate,
            projector_path: self.projector_path.clone(),
            ctx_size: self.ctx_size,
            compact_meta: &self.compact_meta,
            capacity_budget_bytes: self.capacity_budget_bytes,
            cache_type_k_override: self.cache_type_k_override.as_deref(),
            cache_type_v_override: self.cache_type_v_override.as_deref(),
            n_batch_override: self.n_batch_override,
            n_ubatch_override: self.n_ubatch_override,
            flash_attention_override: self.flash_attention_override,
            openai_guardrail_policy: self.openai_guardrail_policy.clone(),
            pinned_gpu: self.pinned_gpu.as_ref(),
            device_override: self.device_override.as_deref(),
            slots: self.slots,
            skippy_telemetry: self.skippy_telemetry.clone(),
            survey_telemetry: self.survey_telemetry.clone(),
            serving_hooks_factory: None,
            local_source_required: self.local_source_required,
        })
        .await?;
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let event = SplitCoordinatorEvent::Replace(Box::new(SplitCoordinatorReplaceEvent {
            reason,
            generation: candidate.generation,
            loaded,
            ack: ack_tx,
        }));
        if let Err(err) = self.event_tx.send(event).await {
            let SplitCoordinatorEvent::Replace(event) = err.0 else {
                unreachable!("replace event send returned a non-replace event")
            };
            let event = *event;
            event.loaded.handle.shutdown().await;
            stop_split_generation(&self.node, &candidate, candidate.generation).await;
            anyhow::bail!("publish split topology candidate to runtime loop: receiver closed");
        }
        match ack_rx.await {
            Ok(SplitCoordinatorAck::Accepted) => {
                self.active = candidate;
                stop_split_generation(&self.node, &previous, self.active.generation).await;
                tracing::info!(
                    model_ref = self.model_ref,
                    topology_id = self.active.topology_id,
                    generation = self.active.generation,
                    stages = ?split_stage_plan_labels(&self.active.stages),
                    "split topology replan cutover complete"
                );
                Ok(())
            }
            Err(_) => {
                stop_split_generation(&self.node, &candidate, candidate.generation).await;
                anyhow::bail!("runtime loop dropped split topology candidate ack");
            }
        }
    }

    async fn request_local_fallback(
        &mut self,
        reason: &'static str,
        unavailable_stage_nodes: Vec<iroh::EndpointId>,
    ) -> Result<()> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let event = SplitCoordinatorEvent::LocalFallback(SplitCoordinatorLocalFallbackEvent {
            reason,
            generation: self.active.generation,
            topology_id: self.active.topology_id.clone(),
            run_id: self.active.run_id.clone(),
            unavailable_stage_nodes,
            ack: ack_tx,
        });
        if self.event_tx.send(event).await.is_err() {
            anyhow::bail!("publish split topology local fallback to runtime loop: receiver closed");
        }
        match ack_rx.await {
            Ok(SplitCoordinatorAck::Accepted) => Ok(()),
            Err(_) => anyhow::bail!("runtime loop dropped split topology local fallback ack"),
        }
    }

    async fn withdraw_active_generation(
        &mut self,
        reason: &'static str,
        unavailable_stage_nodes: Vec<iroh::EndpointId>,
    ) -> Result<()> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let event = SplitCoordinatorEvent::Withdraw(SplitCoordinatorWithdrawEvent {
            reason,
            generation: self.active.generation,
            topology_id: self.active.topology_id.clone(),
            run_id: self.active.run_id.clone(),
            unavailable_stage_nodes,
            ack: ack_tx,
        });
        if self.event_tx.send(event).await.is_err() {
            anyhow::bail!("publish split topology withdrawal to runtime loop: receiver closed");
        }
        match ack_rx.await {
            Ok(SplitCoordinatorAck::Accepted) => Ok(()),
            Err(_) => anyhow::bail!("runtime loop dropped split topology withdrawal ack"),
        }
    }
}

pub(super) fn split_replan_local_capacity_override(
    capacity_budget_bytes: Option<u64>,
    pinned_gpu_capacity_bytes: Option<u64>,
) -> Option<u64> {
    capacity_budget_bytes
        .filter(|bytes| *bytes > 0)
        .or(pinned_gpu_capacity_bytes)
}

fn split_candidate_stage0_is_local(
    local_node_id: iroh::EndpointId,
    candidate: &SplitTopologyGeneration,
) -> bool {
    candidate
        .stages
        .first()
        .is_some_and(|stage0| stage0.node_id == local_node_id)
}

#[cfg(test)]
pub(super) fn split_replan_decision(
    active: &SplitTopologyGeneration,
    candidate: &SplitTopologyGeneration,
) -> SplitReplanDecision {
    split_replan_decision_with_reason(active, candidate).0
}

pub(super) fn split_replan_decision_with_reason(
    active: &SplitTopologyGeneration,
    candidate: &SplitTopologyGeneration,
) -> (SplitReplanDecision, &'static str) {
    if split_active_stage_node_missing_from_participants(active, &candidate.participants) {
        return (
            SplitReplanDecision::Candidate,
            "active_stage_participant_missing",
        );
    }
    if candidate.stages.len() > active.stages.len() {
        return (SplitReplanDecision::Candidate, "candidate_has_more_stages");
    }
    if candidate.participants.len() > active.participants.len()
        && candidate.stages.len() == active.stages.len()
    {
        return (
            SplitReplanDecision::Candidate,
            "candidate_has_more_participants",
        );
    }
    if split_stage_node_signature(&candidate.stages) != split_stage_node_signature(&active.stages)
        && split_stage_balance_score(&candidate.stages) < split_stage_balance_score(&active.stages)
    {
        return (SplitReplanDecision::Candidate, "candidate_improves_balance");
    }
    (SplitReplanDecision::Keep, "candidate_not_materially_better")
}

pub(super) fn split_recovery_candidate_participants(
    participants: &[SplitParticipant],
    unavailable_stage_nodes: &[iroh::EndpointId],
) -> Vec<SplitParticipant> {
    if unavailable_stage_nodes.is_empty() {
        return participants.to_vec();
    }
    participants
        .iter()
        .copied()
        .filter(|participant| !unavailable_stage_nodes.contains(&participant.node_id))
        .collect()
}

pub(super) fn split_candidate_for_replan(
    participant_count: usize,
    candidate: Option<SplitTopologyGeneration>,
) -> Option<SplitTopologyGeneration> {
    if participant_count < SPLIT_DEFAULT_MIN_PARTICIPANTS {
        return None;
    }
    candidate
}

pub(super) fn log_split_replan_quorum_not_met(
    model_ref: &str,
    reason: &'static str,
    participants: &[SplitParticipant],
    excluded: &[SplitParticipantExclusion],
) {
    tracing::debug!(
        model_ref,
        reason,
        participants = ?split_participant_labels(participants),
        excluded = ?split_participant_exclusion_labels(excluded),
        "split topology replan skipped; quorum not met"
    );
}

pub(super) fn split_active_stage_node_missing_from_participants(
    active: &SplitTopologyGeneration,
    participants: &[SplitParticipant],
) -> bool {
    active.stages.iter().any(|stage| {
        participants
            .iter()
            .all(|participant| participant.node_id != stage.node_id)
    })
}

pub(super) async fn stop_split_generation(
    node: &mesh::Node,
    generation: &SplitTopologyGeneration,
    shutdown_generation: u64,
) {
    if let Some(stage0) = generation.stages.first()
        && stage0.node_id == node.id()
    {
        node.unregister_stage_transport_alias(
            &generation.topology_id,
            &generation.run_id,
            &stage0.stage_id,
        )
        .await;
    }
    for stage in generation.stages.iter().skip(1) {
        let stop = skippy::StageStopRequest {
            topology_id: generation.topology_id.clone(),
            run_id: generation.run_id.clone(),
            stage_id: stage.stage_id.clone(),
            shutdown_generation,
            coordinator_term: generation.coordinator_term,
        };
        let result = if stage.node_id == node.id() {
            node.send_local_stage_control(skippy::StageControlRequest::Stop(stop))
                .await
        } else {
            node.send_stage_control(stage.node_id, skippy::StageControlRequest::Stop(stop))
                .await
        };
        if let Err(err) = result {
            tracing::warn!(
                topology_id = %generation.topology_id,
                run_id = %generation.run_id,
                stage_id = %stage.stage_id,
                node = %stage.node_id.fmt_short(),
                error = %err,
                "failed to stop split stage generation"
            );
        }
        if stage.node_id != node.id() {
            node.stop_stage_transport_bridge(
                &generation.topology_id,
                &generation.run_id,
                &stage.stage_id,
            )
            .await;
        }
    }
}

pub(super) fn split_stage_node_signature(
    stages: &[RuntimeSliceStagePlan],
) -> Vec<iroh::EndpointId> {
    stages.iter().map(|stage| stage.node_id).collect()
}

pub(super) fn split_stage_balance_score(stages: &[RuntimeSliceStagePlan]) -> u32 {
    let Some(min) = stages
        .iter()
        .map(|stage| stage.layer_end.saturating_sub(stage.layer_start))
        .min()
    else {
        return 0;
    };
    let max = stages
        .iter()
        .map(|stage| stage.layer_end.saturating_sub(stage.layer_start))
        .max()
        .unwrap_or(min);
    max.saturating_sub(min)
}

fn drain_split_peer_changes<T>(peer_rx: &mut tokio::sync::watch::Receiver<T>) {
    while peer_rx.has_changed().unwrap_or(false) {
        let _ = peer_rx.borrow_and_update();
    }
}
