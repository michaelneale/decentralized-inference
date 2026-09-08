#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SetupPlan {
    pub runtime: SetupRuntimePlan,
    pub endpoints: SetupEndpointPlan,
    pub service: SetupServicePlan,
    pub github_star: SetupGitHubStarPlan,
    pub core_steps: Vec<SetupStep>,
}

impl SetupPlan {
    pub fn new(
        runtime: SetupRuntimePlan,
        endpoints: SetupEndpointPlan,
        service: SetupServicePlan,
        github_star: SetupGitHubStarPlan,
    ) -> Self {
        let core_steps = build_core_steps(runtime, endpoints, service);
        Self {
            runtime,
            endpoints,
            service,
            github_star,
            core_steps,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupRuntimePlan {
    InstallAndPrune,
    Skip,
}

/// Whether setup probes loopback for OpenAI-compatible servers already running
/// on this machine. Probing only reports; it never writes config.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupEndpointPlan {
    Probe,
    Skip,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupServicePlan {
    Install,
    Skip,
    PrintGuidance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupGitHubStarPlan {
    PromptIfEligible,
    Skip(SetupGitHubStarSkipReason),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupGitHubStarSkipReason {
    AutomaticYes,
    HiddenPrompt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupStep {
    InstallRuntime,
    PruneInactiveRuntimes,
    DiscoverLocalEndpoints,
    InstallService,
    PrintServiceGuidance,
}

fn build_core_steps(
    runtime: SetupRuntimePlan,
    endpoints: SetupEndpointPlan,
    service: SetupServicePlan,
) -> Vec<SetupStep> {
    let mut steps = Vec::new();
    if matches!(runtime, SetupRuntimePlan::InstallAndPrune) {
        steps.push(SetupStep::InstallRuntime);
        steps.push(SetupStep::PruneInactiveRuntimes);
    }
    if matches!(endpoints, SetupEndpointPlan::Probe) {
        steps.push(SetupStep::DiscoverLocalEndpoints);
    }
    match service {
        SetupServicePlan::Install => steps.push(SetupStep::InstallService),
        SetupServicePlan::PrintGuidance => steps.push(SetupStep::PrintServiceGuidance),
        SetupServicePlan::Skip => {}
    }
    steps
}
