use super::{
    SetupConfirmPrompt, SetupEnvironment, SetupGitHubStarPlan, SetupGitHubStarSkipReason,
    SetupOptions, SetupPlanError, SetupPlatform, SetupPrompter, SetupRuntimePlan, SetupServicePlan,
    SetupStep, plan_setup,
};
use std::collections::VecDeque;

#[derive(Default)]
struct FakePrompter {
    replies: VecDeque<Option<bool>>,
    prompts: Vec<SetupConfirmPrompt>,
}

impl FakePrompter {
    fn with_replies(replies: impl IntoIterator<Item = Option<bool>>) -> Self {
        Self {
            replies: replies.into_iter().collect(),
            prompts: Vec::new(),
        }
    }
}

impl SetupPrompter for FakePrompter {
    fn confirm(&mut self, prompt: SetupConfirmPrompt) -> Option<bool> {
        self.prompts.push(prompt);
        self.replies.pop_front().unwrap_or(None)
    }
}

#[test]
fn interactive_unix_enter_accepts_default_yes_service_prompt() {
    let options = SetupOptions::default();
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: true,
    };
    let mut prompter = FakePrompter::with_replies([None]);

    let plan = plan_setup(options, environment, &mut prompter).expect("plan should succeed");

    assert_eq!(plan.runtime, SetupRuntimePlan::InstallAndPrune);
    assert_eq!(plan.service, SetupServicePlan::Install);
    assert_eq!(plan.github_star, SetupGitHubStarPlan::PromptIfEligible,);
    assert_eq!(
        plan.core_steps,
        vec![
            SetupStep::InstallRuntime,
            SetupStep::PruneInactiveRuntimes,
            SetupStep::InstallService,
        ]
    );
    assert_eq!(prompter.prompts.len(), 1);
}

#[test]
fn interactive_unix_explicit_no_skips_service_prompt() {
    let options = SetupOptions::default();
    let environment = SetupEnvironment {
        platform: SetupPlatform::MacOs,
        interactive: true,
    };
    let mut prompter = FakePrompter::with_replies([Some(false)]);

    let plan = plan_setup(options, environment, &mut prompter).expect("plan should succeed");

    assert_eq!(plan.service, SetupServicePlan::Skip);
    assert_eq!(
        plan.core_steps,
        vec![SetupStep::InstallRuntime, SetupStep::PruneInactiveRuntimes]
    );
    assert_eq!(prompter.prompts.len(), 1);
}

#[test]
fn non_interactive_unix_never_prompts_and_prints_service_guidance() {
    let options = SetupOptions {
        no_interactive: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: false,
    };
    let mut prompter = FakePrompter::default();

    let plan = plan_setup(options, environment, &mut prompter).expect("plan should succeed");

    assert_eq!(plan.service, SetupServicePlan::PrintGuidance);
    assert_eq!(
        plan.github_star,
        SetupGitHubStarPlan::Skip(SetupGitHubStarSkipReason::HiddenPrompt),
    );
    assert_eq!(
        plan.core_steps,
        vec![
            SetupStep::InstallRuntime,
            SetupStep::PruneInactiveRuntimes,
            SetupStep::PrintServiceGuidance,
        ]
    );
    assert!(prompter.prompts.is_empty());
}

#[test]
fn service_flag_installs_service_without_prompt() {
    let options = SetupOptions {
        service: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: false,
    };
    let mut prompter = FakePrompter::default();

    let plan = plan_setup(options, environment, &mut prompter).expect("plan should succeed");

    assert_eq!(plan.service, SetupServicePlan::Install);
    assert!(prompter.prompts.is_empty());
}

#[test]
fn no_service_flag_skips_service_without_prompt() {
    let options = SetupOptions {
        no_service: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: true,
    };
    let mut prompter = FakePrompter::default();

    let plan = plan_setup(options, environment, &mut prompter).expect("plan should succeed");

    assert_eq!(plan.service, SetupServicePlan::Skip);
    assert!(prompter.prompts.is_empty());
}

#[test]
fn windows_service_flag_is_an_unsupported_error() {
    let options = SetupOptions {
        service: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Windows,
        interactive: true,
    };
    let mut prompter = FakePrompter::default();

    let error =
        plan_setup(options, environment, &mut prompter).expect_err("windows service must fail");

    assert_eq!(
        error,
        SetupPlanError::UnsupportedService {
            platform: SetupPlatform::Windows,
        }
    );
    assert!(prompter.prompts.is_empty());
}

#[test]
fn skip_runtime_omits_install_and_prune_steps() {
    let options = SetupOptions {
        skip_runtime: true,
        service: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: true,
    };
    let mut prompter = FakePrompter::default();

    let plan = plan_setup(options, environment, &mut prompter).expect("plan should succeed");

    assert_eq!(plan.runtime, SetupRuntimePlan::Skip);
    assert_eq!(plan.core_steps, vec![SetupStep::InstallService]);
}

#[test]
fn yes_skips_core_prompts_and_github_star_prompt() {
    let options = SetupOptions {
        yes: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: true,
    };
    let mut prompter = FakePrompter::default();

    let plan = plan_setup(options, environment, &mut prompter).expect("plan should succeed");

    assert_eq!(plan.service, SetupServicePlan::Install);
    assert_eq!(
        plan.github_star,
        SetupGitHubStarPlan::Skip(SetupGitHubStarSkipReason::AutomaticYes),
    );
    assert!(prompter.prompts.is_empty());
}

#[test]
fn yes_and_no_service_keeps_explicit_service_skip_without_prompt() {
    let options = SetupOptions {
        yes: true,
        no_service: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: true,
    };
    let mut prompter = FakePrompter::default();

    let plan = plan_setup(options, environment, &mut prompter).expect("plan should succeed");

    assert_eq!(plan.service, SetupServicePlan::Skip);
    assert_eq!(
        plan.github_star,
        SetupGitHubStarPlan::Skip(SetupGitHubStarSkipReason::AutomaticYes),
    );
    assert_eq!(
        plan.core_steps,
        vec![SetupStep::InstallRuntime, SetupStep::PruneInactiveRuntimes]
    );
    assert!(prompter.prompts.is_empty());
}
