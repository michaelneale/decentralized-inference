use crate::{
    DiagnosticEventKind, EventSystemHealthEventKind, FactData, FactMetadata, GenerationEventKind,
    KvRuntimeStateEventKind, ModelAvailabilityEventKind, ModelLoadingEventKind,
    ModelPreparationEventKind, ModelUnloadingEventKind, NativeRuntimeEventKind,
    NodeAvailabilityEventKind, PrefillEventKind, RequestEventKind, ResourceHealthEventKind,
    ScopeIdentities, SessionEventKind, StageTopologyEventKind,
};

#[derive(Debug, Clone, PartialEq)]
pub struct FamilyFact<K> {
    kind: K,
    data: FactData,
    metadata: Option<FactMetadata>,
}

impl<K> FamilyFact<K> {
    #[must_use]
    pub fn new(kind: K) -> Self {
        Self {
            kind,
            data: FactData::default(),
            metadata: None,
        }
    }

    #[must_use]
    pub const fn with_data(kind: K, data: FactData) -> Self {
        Self {
            kind,
            data,
            metadata: None,
        }
    }

    /// Construct a fact with producer metadata. Existing constructors remain
    /// source-compatible; callers that need a native or reconciled source use
    /// this additive constructor.
    #[must_use]
    pub fn with_metadata(kind: K, data: FactData, metadata: FactMetadata) -> Self {
        Self {
            kind,
            data,
            metadata: Some(metadata),
        }
    }

    #[must_use]
    pub const fn kind(&self) -> &K {
        &self.kind
    }

    #[must_use]
    pub const fn data(&self) -> &FactData {
        &self.data
    }

    #[must_use]
    pub const fn metadata(&self) -> Option<&FactMetadata> {
        self.metadata.as_ref()
    }

    /// Fill only missing scope fields from a known operation scope.
    ///
    /// Synthetic facts may have some explicit identities already. Those values
    /// remain authoritative; this helper only supplies fields omitted by the
    /// synthetic producer and never changes producer metadata.
    #[must_use]
    pub fn with_scope(mut self, fallback: &ScopeIdentities) -> Self {
        if self.data.scope.model_id.is_none() {
            self.data.scope.model_id = fallback.model_id.clone();
        }
        if self.data.scope.topology_id.is_none() {
            self.data.scope.topology_id = fallback.topology_id.clone();
        }
        if self.data.scope.stage.is_none() {
            self.data.scope.stage = fallback.stage.clone();
        }
        if self.data.scope.session_id.is_none() {
            self.data.scope.session_id = fallback.session_id.clone();
        }
        if self.data.scope.request_id.is_none() {
            self.data.scope.request_id = fallback.request_id.clone();
        }
        if self.data.scope.device_id.is_none() {
            self.data.scope.device_id = fallback.device_id.clone();
        }
        self
    }
}

pub type NativeRuntimeFact = FamilyFact<NativeRuntimeEventKind>;
pub type ModelPreparationFact = FamilyFact<ModelPreparationEventKind>;
pub type ModelLoadingFact = FamilyFact<ModelLoadingEventKind>;
pub type ModelAvailabilityFact = FamilyFact<ModelAvailabilityEventKind>;
pub type ModelUnloadingFact = FamilyFact<ModelUnloadingEventKind>;
pub type StageTopologyFact = FamilyFact<StageTopologyEventKind>;
pub type SessionFact = FamilyFact<SessionEventKind>;
pub type RequestFact = FamilyFact<RequestEventKind>;
pub type PrefillFact = FamilyFact<PrefillEventKind>;
pub type GenerationFact = FamilyFact<GenerationEventKind>;
pub type KvRuntimeStateFact = FamilyFact<KvRuntimeStateEventKind>;
pub type ResourceHealthFact = FamilyFact<ResourceHealthEventKind>;
pub type DiagnosticFact = FamilyFact<DiagnosticEventKind>;
pub type NodeAvailabilityFact = FamilyFact<NodeAvailabilityEventKind>;
pub type EventSystemHealthFact = FamilyFact<EventSystemHealthEventKind>;

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeFact {
    NativeRuntime(NativeRuntimeFact),
    ModelPreparation(ModelPreparationFact),
    ModelLoading(ModelLoadingFact),
    ModelAvailability(ModelAvailabilityFact),
    ModelUnloading(ModelUnloadingFact),
    StageTopology(StageTopologyFact),
    Session(SessionFact),
    Request(RequestFact),
    Prefill(PrefillFact),
    Generation(GenerationFact),
    KvRuntimeState(KvRuntimeStateFact),
    ResourceHealth(ResourceHealthFact),
    Diagnostic(DiagnosticFact),
    NodeAvailability(NodeAvailabilityFact),
    EventSystemHealth(EventSystemHealthFact),
}

impl RuntimeFact {
    #[must_use]
    pub const fn data(&self) -> &FactData {
        match self {
            Self::NativeRuntime(fact) => fact.data(),
            Self::ModelPreparation(fact) => fact.data(),
            Self::ModelLoading(fact) => fact.data(),
            Self::ModelAvailability(fact) => fact.data(),
            Self::ModelUnloading(fact) => fact.data(),
            Self::StageTopology(fact) => fact.data(),
            Self::Session(fact) => fact.data(),
            Self::Request(fact) => fact.data(),
            Self::Prefill(fact) => fact.data(),
            Self::Generation(fact) => fact.data(),
            Self::KvRuntimeState(fact) => fact.data(),
            Self::ResourceHealth(fact) => fact.data(),
            Self::Diagnostic(fact) => fact.data(),
            Self::NodeAvailability(fact) => fact.data(),
            Self::EventSystemHealth(fact) => fact.data(),
        }
    }

    #[must_use]
    pub fn metadata(&self) -> Option<&FactMetadata> {
        match self {
            Self::NativeRuntime(fact) => fact.metadata(),
            Self::ModelPreparation(fact) => fact.metadata(),
            Self::ModelLoading(fact) => fact.metadata(),
            Self::ModelAvailability(fact) => fact.metadata(),
            Self::ModelUnloading(fact) => fact.metadata(),
            Self::StageTopology(fact) => fact.metadata(),
            Self::Session(fact) => fact.metadata(),
            Self::Request(fact) => fact.metadata(),
            Self::Prefill(fact) => fact.metadata(),
            Self::Generation(fact) => fact.metadata(),
            Self::KvRuntimeState(fact) => fact.metadata(),
            Self::ResourceHealth(fact) => fact.metadata(),
            Self::Diagnostic(fact) => fact.metadata(),
            Self::NodeAvailability(fact) => fact.metadata(),
            Self::EventSystemHealth(fact) => fact.metadata(),
        }
    }

    /// Attach producer metadata while preserving the existing family kind
    /// and fact data. This keeps metadata additive to `RuntimeFact`.
    #[must_use]
    pub fn with_metadata(self, metadata: FactMetadata) -> Self {
        match self {
            Self::NativeRuntime(fact) => {
                Self::NativeRuntime(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::ModelPreparation(fact) => {
                Self::ModelPreparation(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::ModelLoading(fact) => {
                Self::ModelLoading(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::ModelAvailability(fact) => {
                Self::ModelAvailability(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::ModelUnloading(fact) => {
                Self::ModelUnloading(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::StageTopology(fact) => {
                Self::StageTopology(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::Session(fact) => {
                Self::Session(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::Request(fact) => {
                Self::Request(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::Prefill(fact) => {
                Self::Prefill(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::Generation(fact) => {
                Self::Generation(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::KvRuntimeState(fact) => {
                Self::KvRuntimeState(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::ResourceHealth(fact) => {
                Self::ResourceHealth(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::Diagnostic(fact) => {
                Self::Diagnostic(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::NodeAvailability(fact) => {
                Self::NodeAvailability(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
            Self::EventSystemHealth(fact) => {
                Self::EventSystemHealth(FamilyFact::with_metadata(fact.kind, fact.data, metadata))
            }
        }
    }

    /// Fill missing scope fields on a synthetic fact without changing its
    /// explicit identities or producer metadata.
    #[must_use]
    pub fn with_scope(self, fallback: &ScopeIdentities) -> Self {
        match self {
            Self::NativeRuntime(fact) => Self::NativeRuntime(fact.with_scope(fallback)),
            Self::ModelPreparation(fact) => Self::ModelPreparation(fact.with_scope(fallback)),
            Self::ModelLoading(fact) => Self::ModelLoading(fact.with_scope(fallback)),
            Self::ModelAvailability(fact) => Self::ModelAvailability(fact.with_scope(fallback)),
            Self::ModelUnloading(fact) => Self::ModelUnloading(fact.with_scope(fallback)),
            Self::StageTopology(fact) => Self::StageTopology(fact.with_scope(fallback)),
            Self::Session(fact) => Self::Session(fact.with_scope(fallback)),
            Self::Request(fact) => Self::Request(fact.with_scope(fallback)),
            Self::Prefill(fact) => Self::Prefill(fact.with_scope(fallback)),
            Self::Generation(fact) => Self::Generation(fact.with_scope(fallback)),
            Self::KvRuntimeState(fact) => Self::KvRuntimeState(fact.with_scope(fallback)),
            Self::ResourceHealth(fact) => Self::ResourceHealth(fact.with_scope(fallback)),
            Self::Diagnostic(fact) => Self::Diagnostic(fact.with_scope(fallback)),
            Self::NodeAvailability(fact) => Self::NodeAvailability(fact.with_scope(fallback)),
            Self::EventSystemHealth(fact) => Self::EventSystemHealth(fact.with_scope(fallback)),
        }
    }

    #[must_use]
    pub const fn kind_id(&self) -> &'static str {
        match self {
            Self::NativeRuntime(fact) => fact.kind().as_str(),
            Self::ModelPreparation(fact) => fact.kind().as_str(),
            Self::ModelLoading(fact) => fact.kind().as_str(),
            Self::ModelAvailability(fact) => fact.kind().as_str(),
            Self::ModelUnloading(fact) => fact.kind().as_str(),
            Self::StageTopology(fact) => fact.kind().as_str(),
            Self::Session(fact) => fact.kind().as_str(),
            Self::Request(fact) => fact.kind().as_str(),
            Self::Prefill(fact) => fact.kind().as_str(),
            Self::Generation(fact) => fact.kind().as_str(),
            Self::KvRuntimeState(fact) => fact.kind().as_str(),
            Self::ResourceHealth(fact) => fact.kind().as_str(),
            Self::Diagnostic(fact) => fact.kind().as_str(),
            Self::NodeAvailability(fact) => fact.kind().as_str(),
            Self::EventSystemHealth(fact) => fact.kind().as_str(),
        }
    }
}
