use std::{fmt, sync::Arc};

use anyhow::Result;

use crate::{
    frontend::{
        GenerationLifecycleConfig, GenerationLifecycleIngress, GenerationReceiptConfig,
        LinearProposalIngressConfig,
    },
    kv_integration::KvLifecycleObserver,
    tokenizer::TokenizerCapability,
};

/// Constructs product-neutral serving hooks after Skippy has loaded the model
/// and can expose its authoritative tokenizer capability. `extra_generation_sink`
/// is an optional host-owned generation lifecycle sink (for example a runtime-event
/// adapter) that the implementation must fan generation observations out to
/// alongside its own exact-receipt sink. Event-only integrations can instead
/// use the independent `generation_lifecycle` slot, which supports split serving.
pub trait ModelServingHooksFactory: Send + Sync {
    fn create(
        &self,
        tokenizer: TokenizerCapability,
        extra_generation_sink: Option<Arc<dyn GenerationLifecycleIngress>>,
    ) -> Result<ModelServingHooks>;
}

impl fmt::Debug for dyn ModelServingHooksFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ModelServingHooksFactory")
    }
}

pub type SharedModelServingHooksFactory = Arc<dyn ModelServingHooksFactory>;

/// Optional generation observers and proposal source for serving.
/// Exact generation receipts remain restricted to local single-stage execution;
/// lifecycle-only observations also support split execution.
#[derive(Clone, Default)]
pub struct ModelServingHooks {
    generation_receipt: Option<GenerationReceiptConfig>,
    generation_lifecycle: Option<GenerationLifecycleConfig>,
    linear_proposal_ingress: Option<LinearProposalIngressConfig>,
    kv_lifecycle_observer: Option<Arc<dyn KvLifecycleObserver>>,
}

impl ModelServingHooks {
    #[must_use]
    pub fn with_generation_receipt(mut self, config: GenerationReceiptConfig) -> Self {
        self.generation_receipt = Some(config);
        self
    }

    #[must_use]
    pub fn with_generation_lifecycle(mut self, config: GenerationLifecycleConfig) -> Self {
        self.generation_lifecycle = Some(config);
        self
    }

    #[must_use]
    pub fn with_linear_proposal_ingress(mut self, config: LinearProposalIngressConfig) -> Self {
        self.linear_proposal_ingress = Some(config);
        self
    }

    #[must_use]
    pub fn with_kv_lifecycle_observer(mut self, observer: Arc<dyn KvLifecycleObserver>) -> Self {
        self.kv_lifecycle_observer = Some(observer);
        self
    }

    #[must_use]
    pub fn new(
        generation_receipt: GenerationReceiptConfig,
        linear_proposal_ingress: LinearProposalIngressConfig,
    ) -> Self {
        Self::default()
            .with_generation_receipt(generation_receipt)
            .with_linear_proposal_ingress(linear_proposal_ingress)
    }

    pub fn generation_receipt(&self) -> Option<GenerationReceiptConfig> {
        self.generation_receipt.clone()
    }

    pub fn generation_lifecycle(&self) -> Option<GenerationLifecycleConfig> {
        self.generation_lifecycle.clone()
    }

    pub fn linear_proposal_ingress(&self) -> Option<LinearProposalIngressConfig> {
        self.linear_proposal_ingress.clone()
    }

    pub fn kv_lifecycle_observer(&self) -> Option<Arc<dyn KvLifecycleObserver>> {
        self.kv_lifecycle_observer.clone()
    }
}

impl fmt::Debug for ModelServingHooks {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelServingHooks")
            .field(
                "generation_receipt",
                &self.generation_receipt.as_ref().map(|_| "configured"),
            )
            .field(
                "generation_lifecycle",
                &self.generation_lifecycle.as_ref().map(|_| "configured"),
            )
            .field(
                "linear_proposal_ingress",
                &self.linear_proposal_ingress.as_ref().map(|_| "configured"),
            )
            .field(
                "kv_lifecycle_observer",
                &self.kv_lifecycle_observer.as_ref().map(|_| "configured"),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use anyhow::Result;

    use crate::frontend::{
        GenerationAbort, GenerationCommit, GenerationReceipt, GenerationReceiptSink,
        GenerationStart, LinearProposalIngress, LinearProposalQuery, LinearProposalReceipt,
        LinearProposalSourceResponse,
    };

    use super::*;

    struct ReceiptSink;

    impl GenerationReceiptSink for ReceiptSink {
        fn begin(&self, _start: &GenerationStart) -> Result<()> {
            Ok(())
        }
        fn committed(&self, _commit: &GenerationCommit) -> Result<()> {
            Ok(())
        }
        fn abort(&self, _abort: &GenerationAbort) -> Result<()> {
            Ok(())
        }
        fn record(&self, _receipt: &GenerationReceipt) -> Result<()> {
            Ok(())
        }
    }

    struct ProposalIngress;

    impl LinearProposalIngress for ProposalIngress {
        fn propose(&self, _query: LinearProposalQuery) -> Result<LinearProposalSourceResponse> {
            Ok(LinearProposalSourceResponse::new(None))
        }

        fn report(&self, _receipt: &LinearProposalReceipt) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn neutral_hooks_are_absent_by_default_and_preserved_when_injected() {
        let empty = ModelServingHooks::default();
        assert!(empty.generation_receipt().is_none());
        assert!(empty.generation_lifecycle().is_none());
        assert!(empty.linear_proposal_ingress().is_none());

        let configured = ModelServingHooks::new(
            GenerationReceiptConfig::new(Arc::new(ReceiptSink)),
            LinearProposalIngressConfig::new(
                Arc::new(ProposalIngress),
                Duration::from_millis(4),
                32,
            )
            .unwrap(),
        );
        assert!(configured.generation_receipt().is_some());
        assert!(configured.linear_proposal_ingress().is_some());
        assert!(configured.clone().generation_receipt().is_some());
    }

    #[test]
    fn hooks_can_be_configured_independently() {
        let receipt = ModelServingHooks::default()
            .with_generation_receipt(GenerationReceiptConfig::new(Arc::new(ReceiptSink)));
        assert!(receipt.generation_receipt().is_some());
        assert!(receipt.generation_lifecycle().is_none());
        assert!(receipt.linear_proposal_ingress().is_none());

        let lifecycle = ModelServingHooks::default().with_generation_lifecycle(
            GenerationLifecycleConfig::from_ingress(Arc::new(
                crate::frontend::CompositeGenerationLifecycleIngress::new(Vec::new()),
            )),
        );
        assert!(lifecycle.generation_receipt().is_none());
        assert!(lifecycle.generation_lifecycle().is_some());

        let proposal = ModelServingHooks::default().with_linear_proposal_ingress(
            LinearProposalIngressConfig::new(
                Arc::new(ProposalIngress),
                Duration::from_millis(4),
                32,
            )
            .unwrap(),
        );
        assert!(proposal.generation_receipt().is_none());
        assert!(proposal.linear_proposal_ingress().is_some());
    }
}
