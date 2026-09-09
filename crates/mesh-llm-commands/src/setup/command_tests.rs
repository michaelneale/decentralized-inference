use super::service_paths::ServiceInstallContext;
use super::service_runner::{ServiceCommand, ServiceCommandRunner};
use super::{
    SetupActions, SetupConfirmPrompt, SetupEnvironment, SetupGitHubStarPlan,
    SetupGitHubStarSkipReason, SetupOptions, SetupPlatform, SetupPrompter, SetupServicePlan,
    SetupStep, run_setup,
};
use crate::setup::command::CliSetupActions;
use crate::setup::github::{SetupGitHubOutcome, github_summary};
use crate::setup::github_runner::{GhCommand, GhCommandError, GhCommandOutput, GhCommandRunner};
use crate::setup::summary::service_summary;
use std::collections::VecDeque;
use std::fs;
use std::future::{Ready, ready};

#[derive(Default)]
struct FakePrompter {
    replies: VecDeque<Option<bool>>,
}

impl FakePrompter {
    fn with_replies(replies: impl IntoIterator<Item = Option<bool>>) -> Self {
        Self {
            replies: replies.into_iter().collect(),
        }
    }
}

impl SetupPrompter for FakePrompter {
    fn confirm(&mut self, _prompt: SetupConfirmPrompt) -> Option<bool> {
        self.replies.pop_front().unwrap_or(None)
    }
}

#[derive(Default)]
struct FakeActions {
    steps: Vec<SetupStep>,
    github: Vec<SetupGitHubStarPlan>,
}

impl SetupActions for FakeActions {
    type Error = anyhow::Error;
    type GitHubStarFuture<'a>
        = Ready<Result<(), Self::Error>>
    where
        Self: 'a;
    type StepFuture<'a>
        = Ready<Result<(), Self::Error>>
    where
        Self: 'a;

    fn run_step(&mut self, step: SetupStep) -> Self::StepFuture<'_> {
        self.steps.push(step);
        ready(Ok(()))
    }

    fn handle_github_star<'a>(
        &'a mut self,
        plan: SetupGitHubStarPlan,
        _prompter: &'a mut dyn super::SetupPrompter,
    ) -> Self::GitHubStarFuture<'a> {
        self.github.push(plan);
        ready(Ok(()))
    }
}

#[derive(Default)]
struct FakeServiceRunner {
    commands: Vec<ServiceCommand>,
}

impl ServiceCommandRunner for FakeServiceRunner {
    fn run(&mut self, command: &ServiceCommand) -> anyhow::Result<()> {
        self.commands.push(command.clone());
        Ok(())
    }
}

#[derive(Default)]
struct NoopGhRunner;

impl GhCommandRunner for NoopGhRunner {
    fn run(&mut self, _command: GhCommand) -> Result<GhCommandOutput, GhCommandError> {
        Err(GhCommandError::WaitFailed(
            "github runner should not be used in this test".to_string(),
        ))
    }
}

#[tokio::test]
async fn run_setup_executes_planned_steps_and_github_star_action() {
    let options = SetupOptions::default();
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: true,
    };
    let mut prompter = FakePrompter::with_replies([Some(false)]);
    let mut actions = FakeActions::default();

    let plan = run_setup(options, environment, &mut prompter, &mut actions)
        .await
        .expect("setup should execute");

    assert_eq!(plan.service, SetupServicePlan::Skip);
    assert_eq!(
        actions.steps,
        vec![SetupStep::InstallRuntime, SetupStep::PruneInactiveRuntimes]
    );
    assert_eq!(actions.github, vec![SetupGitHubStarPlan::PromptIfEligible]);
}

#[tokio::test]
async fn run_setup_stops_before_actions_on_plan_error() {
    let options = SetupOptions {
        service: true,
        no_service: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::Linux,
        interactive: true,
    };
    let mut prompter = FakePrompter::default();
    let mut actions = FakeActions::default();

    let error = run_setup(options, environment, &mut prompter, &mut actions)
        .await
        .expect_err("conflicting flags should fail planning");

    assert!(
        error
            .to_string()
            .contains("setup received both --service and --no-service"),
        "unexpected error: {error:#}"
    );
    assert!(actions.steps.is_empty());
    assert!(actions.github.is_empty());
}

#[tokio::test]
async fn run_setup_suppresses_github_star_for_yes_mode() {
    let options = SetupOptions {
        yes: true,
        ..SetupOptions::default()
    };
    let environment = SetupEnvironment {
        platform: SetupPlatform::MacOs,
        interactive: true,
    };
    let mut prompter = FakePrompter::default();
    let mut actions = FakeActions::default();

    let _plan = run_setup(options, environment, &mut prompter, &mut actions)
        .await
        .expect("setup should execute");

    assert_eq!(
        actions.github,
        vec![SetupGitHubStarPlan::Skip(
            SetupGitHubStarSkipReason::AutomaticYes
        )]
    );
}

#[test]
fn github_summary_reports_authenticated_star_success() {
    let plan = super::SetupPlan::new(
        super::SetupRuntimePlan::Skip,
        super::SetupServicePlan::Skip,
        SetupGitHubStarPlan::PromptIfEligible,
    );

    assert_eq!(
        github_summary(&plan, &SetupGitHubOutcome::Starred),
        "starred Mesh-LLM/mesh-llm with the authenticated GitHub CLI account"
    );
}

#[tokio::test]
async fn service_summary_reports_real_service_installation() {
    let plan = super::SetupPlan::new(
        super::SetupRuntimePlan::Skip,
        super::SetupServicePlan::Install,
        SetupGitHubStarPlan::Skip(SetupGitHubStarSkipReason::AutomaticYes),
    );
    let temp = tempfile::tempdir().expect("tempdir should exist");
    let binary_path = temp.path().join("bin/mesh-llm");
    fs::create_dir_all(binary_path.parent().expect("binary parent should exist"))
        .expect("binary dir should exist");
    fs::write(&binary_path, "binary").expect("binary should write");
    let context = ServiceInstallContext {
        platform: SetupPlatform::MacOs,
        home_dir: temp.path().join("home"),
        config_root: temp.path().join("config"),
        binary_path,
        user_id: "501".to_string(),
        start_service: false,
    };
    let mut actions = CliSetupActions::with_service_support(
        SetupEnvironment {
            platform: SetupPlatform::MacOs,
            interactive: false,
        },
        crate::runtime_native::NativeRuntimeConfigSelection::default(),
        context,
        Box::new(FakeServiceRunner::default()),
        Box::new(NoopGhRunner),
    );
    actions
        .run_step(SetupStep::InstallService)
        .await
        .expect("service install should execute");

    assert_eq!(
        service_summary(&plan, &actions),
        "installed; automatic start needs manual follow-up"
    );
}

#[test]
fn github_summary_reports_nonfatal_star_request_failure() {
    let plan = super::SetupPlan::new(
        super::SetupRuntimePlan::Skip,
        super::SetupServicePlan::Skip,
        SetupGitHubStarPlan::PromptIfEligible,
    );

    assert_eq!(
        github_summary(
            &plan,
            &SetupGitHubOutcome::StarRequestFailed(GhCommandError::TimedOut("gh api").to_string())
        ),
        "not starred; GitHub star request failed: timed out running `gh api`"
    );
}
