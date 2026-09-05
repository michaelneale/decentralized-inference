mod execution;
mod lifecycle;

use crate::RuntimeFact;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryClass {
    Terminal,
    StateTransition,
    Progress,
    Diagnostic,
}

impl RuntimeFact {
    #[must_use]
    pub const fn delivery_class(&self) -> DeliveryClass {
        match self {
            Self::NativeRuntime(fact) => lifecycle::native_runtime(*fact.kind()),
            Self::ModelPreparation(fact) => lifecycle::model_preparation(*fact.kind()),
            Self::ModelLoading(fact) => lifecycle::model_loading(*fact.kind()),
            Self::ModelAvailability(fact) => lifecycle::model_availability(*fact.kind()),
            Self::ModelUnloading(fact) => lifecycle::model_unloading(*fact.kind()),
            Self::StageTopology(fact) => lifecycle::stage_topology(*fact.kind()),
            Self::Session(fact) => lifecycle::session(*fact.kind()),
            Self::Request(fact) => execution::request(*fact.kind()),
            Self::Prefill(fact) => execution::prefill(*fact.kind()),
            Self::Generation(fact) => execution::generation(*fact.kind()),
            Self::KvRuntimeState(fact) => execution::kv_runtime_state(*fact.kind()),
            Self::ResourceHealth(fact) => execution::resource_health(*fact.kind()),
            Self::Diagnostic(fact) => execution::diagnostic(*fact.kind()),
            Self::NodeAvailability(fact) => execution::node_availability(*fact.kind()),
            Self::EventSystemHealth(fact) => execution::event_system_health(*fact.kind()),
        }
    }
}
