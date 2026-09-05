use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};

use crate::{
    DeliveryClass, DiagnosticEventKind, DiagnosticFact, EventSystemHealthEventKind,
    EventSystemHealthFact, GenerationEventKind, GenerationFact, KvRuntimeStateEventKind,
    KvRuntimeStateFact, ModelAvailabilityEventKind, ModelAvailabilityFact, ModelLoadingEventKind,
    ModelLoadingFact, ModelPreparationEventKind, ModelPreparationFact, ModelUnloadingEventKind,
    ModelUnloadingFact, NativeRuntimeEventKind, NativeRuntimeFact, NodeAvailabilityEventKind,
    NodeAvailabilityFact, PrefillEventKind, PrefillFact, RequestEventKind, RequestFact,
    ResourceHealthEventKind, ResourceHealthFact, RuntimeEventIngress, RuntimeFact,
    SessionEventKind, SessionFact, StageTopologyEventKind, StageTopologyFact, SubmitOutcome,
};

pub struct SaturatingIngress {
    sender: SyncSender<RuntimeFact>,
}

impl SaturatingIngress {
    pub fn connected(capacity: usize) -> (Self, Receiver<RuntimeFact>) {
        let (sender, receiver) = sync_channel(capacity);
        (Self { sender }, receiver)
    }
}

impl RuntimeEventIngress for SaturatingIngress {
    fn try_submit(&self, fact: RuntimeFact) -> SubmitOutcome {
        let class = fact.delivery_class();
        match self.sender.try_send(fact) {
            Ok(()) => SubmitOutcome::Accepted,
            Err(TrySendError::Disconnected(_)) => SubmitOutcome::RejectedShuttingDown,
            Err(TrySendError::Full(_)) => match class {
                DeliveryClass::Terminal => SubmitOutcome::TerminalDeliveryFailed,
                DeliveryClass::StateTransition => SubmitOutcome::RejectedCapacity,
                DeliveryClass::Progress => SubmitOutcome::DroppedProgress,
                DeliveryClass::Diagnostic => SubmitOutcome::DroppedDiagnostic,
            },
        }
    }
}

pub fn request_fact(kind: RequestEventKind) -> RuntimeFact {
    RuntimeFact::Request(RequestFact::new(kind))
}

pub fn generation_fact(kind: GenerationEventKind) -> RuntimeFact {
    RuntimeFact::Generation(GenerationFact::new(kind))
}

pub fn diagnostic_fact(kind: DiagnosticEventKind) -> RuntimeFact {
    RuntimeFact::Diagnostic(DiagnosticFact::new(kind))
}

pub fn one_fact_per_family() -> [RuntimeFact; 15] {
    [
        RuntimeFact::NativeRuntime(NativeRuntimeFact::new(
            NativeRuntimeEventKind::RuntimeResolutionStarted,
        )),
        RuntimeFact::ModelPreparation(ModelPreparationFact::new(
            ModelPreparationEventKind::ModelQueued,
        )),
        RuntimeFact::ModelLoading(ModelLoadingFact::new(
            ModelLoadingEventKind::ModelLoadRequested,
        )),
        RuntimeFact::ModelAvailability(ModelAvailabilityFact::new(
            ModelAvailabilityEventKind::NativeModelLoaded,
        )),
        RuntimeFact::ModelUnloading(ModelUnloadingFact::new(
            ModelUnloadingEventKind::UnloadRequested,
        )),
        RuntimeFact::StageTopology(StageTopologyFact::new(
            StageTopologyEventKind::StageStarting,
        )),
        RuntimeFact::Session(SessionFact::new(SessionEventKind::SessionRequested)),
        request_fact(RequestEventKind::RequestReceived),
        RuntimeFact::Prefill(PrefillFact::new(PrefillEventKind::PromptProcessingStarted)),
        generation_fact(GenerationEventKind::GenerationStarted),
        RuntimeFact::KvRuntimeState(KvRuntimeStateFact::new(
            KvRuntimeStateEventKind::KvCacheInitializationStarted,
        )),
        RuntimeFact::ResourceHealth(ResourceHealthFact::new(
            ResourceHealthEventKind::BackendInitializationStarted,
        )),
        diagnostic_fact(DiagnosticEventKind::WarningRaised),
        RuntimeFact::NodeAvailability(NodeAvailabilityFact::new(
            NodeAvailabilityEventKind::NodeStarting,
        )),
        RuntimeFact::EventSystemHealth(EventSystemHealthFact::new(
            EventSystemHealthEventKind::IngressQueuePressure,
        )),
    ]
}

pub fn all_facts() -> Vec<RuntimeFact> {
    let mut facts = Vec::with_capacity(184);
    macro_rules! append {
        ($variant:ident, $fact:ident, $kind:ty) => {
            facts.extend(
                <$kind>::ALL
                    .iter()
                    .copied()
                    .map(|kind| RuntimeFact::$variant($fact::new(kind))),
            );
        };
    }
    append!(NativeRuntime, NativeRuntimeFact, NativeRuntimeEventKind);
    append!(
        ModelPreparation,
        ModelPreparationFact,
        ModelPreparationEventKind
    );
    append!(ModelLoading, ModelLoadingFact, ModelLoadingEventKind);
    append!(
        ModelAvailability,
        ModelAvailabilityFact,
        ModelAvailabilityEventKind
    );
    append!(ModelUnloading, ModelUnloadingFact, ModelUnloadingEventKind);
    append!(StageTopology, StageTopologyFact, StageTopologyEventKind);
    append!(Session, SessionFact, SessionEventKind);
    append!(Request, RequestFact, RequestEventKind);
    append!(Prefill, PrefillFact, PrefillEventKind);
    append!(Generation, GenerationFact, GenerationEventKind);
    append!(KvRuntimeState, KvRuntimeStateFact, KvRuntimeStateEventKind);
    append!(ResourceHealth, ResourceHealthFact, ResourceHealthEventKind);
    append!(Diagnostic, DiagnosticFact, DiagnosticEventKind);
    append!(
        NodeAvailability,
        NodeAvailabilityFact,
        NodeAvailabilityEventKind
    );
    append!(
        EventSystemHealth,
        EventSystemHealthFact,
        EventSystemHealthEventKind
    );
    facts
}
