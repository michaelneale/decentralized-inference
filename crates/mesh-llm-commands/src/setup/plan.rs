#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SetupPlan {
    pub runtime: SetupRuntimePlan,
    pub service: SetupServicePlan,
    pub github_star: SetupGitHubStarPlan,
    pub core_steps: Vec<SetupStep>,
}

impl SetupPlan {
    pub fn new(
        runtime: SetupRuntimePlan,
        service: SetupServicePlan,
        github_star: SetupGitHubStarPlan,
    ) -> Self {
        let core_steps = build_core_steps(runtime, service);
        Self {
            runtime,
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
    InstallService,
    PrintServiceGuidance,
}

fn build_core_steps(runtime: SetupRuntimePlan, service: SetupServicePlan) -> Vec<SetupStep> {
    let mut steps = Vec::new();
    if matches!(runtime, SetupRuntimePlan::InstallAndPrune) {
        steps.push(SetupStep::InstallRuntime);
        steps.push(SetupStep::PruneInactiveRuntimes);
    }
    match service {
        SetupServicePlan::Install => steps.push(SetupStep::InstallService),
        SetupServicePlan::PrintGuidance => steps.push(SetupStep::PrintServiceGuidance),
        SetupServicePlan::Skip => {}
    }
    steps
}
