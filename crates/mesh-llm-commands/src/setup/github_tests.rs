use super::github::{SetupGitHubOutcome, execute_github_star_plan, github_summary};
use super::github_runner::{GhCommand, GhCommandError, GhCommandOutput, GhCommandRunner};
use super::{
    SetupConfirmPrompt, SetupGitHubStarPlan, SetupGitHubStarSkipReason, SetupPlan,
    SetupPromptDefault, SetupPromptKind, SetupPrompter,
};
use std::collections::VecDeque;

#[derive(Default)]
struct FakePrompter {
    prompts: Vec<SetupConfirmPrompt>,
    replies: VecDeque<Option<bool>>,
}

impl FakePrompter {
    fn with_replies(replies: impl IntoIterator<Item = Option<bool>>) -> Self {
        Self {
            prompts: Vec::new(),
            replies: replies.into_iter().collect(),
        }
    }
}

impl SetupPrompter for FakePrompter {
    fn confirm(&mut self, prompt: SetupConfirmPrompt) -> Option<bool> {
        self.prompts.push(prompt);
        self.replies.pop_front().unwrap_or(None)
    }
}

#[derive(Default)]
struct FakeGhCommandRunner {
    commands: Vec<GhCommand>,
    responses: VecDeque<Result<GhCommandOutput, GhCommandError>>,
}

impl FakeGhCommandRunner {
    fn with_responses(
        responses: impl IntoIterator<Item = Result<GhCommandOutput, GhCommandError>>,
    ) -> Self {
        Self {
            commands: Vec::new(),
            responses: responses.into_iter().collect(),
        }
    }
}

impl GhCommandRunner for FakeGhCommandRunner {
    fn run(&mut self, command: GhCommand) -> Result<GhCommandOutput, GhCommandError> {
        self.commands.push(command);
        self.responses
            .pop_front()
            .unwrap_or_else(|| Err(GhCommandError::WaitFailed("missing fake response".into())))
    }
}

fn prompt_plan() -> SetupGitHubStarPlan {
    SetupGitHubStarPlan::PromptIfEligible
}

fn skip_plan(reason: SetupGitHubStarSkipReason) -> SetupGitHubStarPlan {
    SetupGitHubStarPlan::Skip(reason)
}

#[test]
fn hidden_prompt_skip_never_runs_gh_or_prompts() {
    let mut runner = FakeGhCommandRunner::default();
    let mut prompter = FakePrompter::default();

    let outcome = execute_github_star_plan(
        skip_plan(SetupGitHubStarSkipReason::HiddenPrompt),
        &mut runner,
        &mut prompter,
    );

    assert_eq!(outcome, SetupGitHubOutcome::HiddenPrompt);
    assert!(runner.commands.is_empty());
    assert!(prompter.prompts.is_empty());
}

#[test]
fn automatic_yes_skip_never_runs_gh_or_prompts() {
    let mut runner = FakeGhCommandRunner::default();
    let mut prompter = FakePrompter::default();

    let outcome = execute_github_star_plan(
        skip_plan(SetupGitHubStarSkipReason::AutomaticYes),
        &mut runner,
        &mut prompter,
    );

    assert_eq!(outcome, SetupGitHubOutcome::AutomaticYes);
    assert!(runner.commands.is_empty());
    assert!(prompter.prompts.is_empty());
}

#[test]
fn unavailable_gh_skips_without_prompt() {
    let mut runner = FakeGhCommandRunner::with_responses([Err(GhCommandError::NotOnPath)]);
    let mut prompter = FakePrompter::default();

    let outcome = execute_github_star_plan(prompt_plan(), &mut runner, &mut prompter);

    assert_eq!(outcome, SetupGitHubOutcome::CliUnavailable);
    assert_eq!(runner.commands, vec![GhCommand::CheckAvailability]);
    assert!(prompter.prompts.is_empty());
}

#[test]
fn unauthenticated_gh_skips_without_prompt() {
    let mut runner = FakeGhCommandRunner::with_responses([
        Ok(GhCommandOutput {
            success: true,
            stdout: "gh version 2.85.0".to_string(),
            stderr: String::new(),
        }),
        Ok(GhCommandOutput {
            success: false,
            stdout: String::new(),
            stderr: "not logged in".to_string(),
        }),
    ]);
    let mut prompter = FakePrompter::default();

    let outcome = execute_github_star_plan(prompt_plan(), &mut runner, &mut prompter);

    assert_eq!(outcome, SetupGitHubOutcome::NotAuthenticated);
    assert_eq!(
        runner.commands,
        vec![GhCommand::CheckAvailability, GhCommand::CheckAuthentication]
    );
    assert!(prompter.prompts.is_empty());
}

#[test]
fn already_starred_skips_without_prompt() {
    let mut runner = FakeGhCommandRunner::with_responses([
        Ok(GhCommandOutput {
            success: true,
            stdout: "gh version 2.85.0".to_string(),
            stderr: String::new(),
        }),
        Ok(GhCommandOutput {
            success: true,
            stdout: "authenticated".to_string(),
            stderr: String::new(),
        }),
        Ok(GhCommandOutput {
            success: true,
            stdout: "true".to_string(),
            stderr: String::new(),
        }),
    ]);
    let mut prompter = FakePrompter::default();

    let outcome = execute_github_star_plan(prompt_plan(), &mut runner, &mut prompter);

    assert_eq!(outcome, SetupGitHubOutcome::AlreadyStarred);
    assert!(prompter.prompts.is_empty());
}

#[test]
fn eligible_default_yes_stars_with_api_fallback_and_exact_prompt() {
    let mut runner = FakeGhCommandRunner::with_responses([
        Ok(GhCommandOutput {
            success: true,
            stdout: "gh version 2.85.0".to_string(),
            stderr: String::new(),
        }),
        Ok(GhCommandOutput {
            success: true,
            stdout: "authenticated".to_string(),
            stderr: String::new(),
        }),
        Ok(GhCommandOutput {
            success: true,
            stdout: "false".to_string(),
            stderr: String::new(),
        }),
        Ok(GhCommandOutput {
            success: true,
            stdout: String::new(),
            stderr: String::new(),
        }),
    ]);
    let mut prompter = FakePrompter::with_replies([None]);

    let outcome = execute_github_star_plan(prompt_plan(), &mut runner, &mut prompter);

    assert_eq!(outcome, SetupGitHubOutcome::Starred);
    assert_eq!(prompter.prompts.len(), 1);
    assert_eq!(prompter.prompts[0].kind, SetupPromptKind::GitHubStar);
    assert_eq!(prompter.prompts[0].default, SetupPromptDefault::Yes);
    assert_eq!(
        prompter.prompts[0].message,
        "Star Mesh-LLM/mesh-llm on GitHub?"
    );
    assert_eq!(
        runner.commands,
        vec![
            GhCommand::CheckAvailability,
            GhCommand::CheckAuthentication,
            GhCommand::CheckViewerHasStarred,
            GhCommand::StarRepository,
        ]
    );
}

#[test]
fn eligible_explicit_no_skips_star_command() {
    let mut runner = FakeGhCommandRunner::with_responses([
        Ok(GhCommandOutput {
            success: true,
            stdout: "gh version 2.85.0".to_string(),
            stderr: String::new(),
        }),
        Ok(GhCommandOutput {
            success: true,
            stdout: "authenticated".to_string(),
            stderr: String::new(),
        }),
        Ok(GhCommandOutput {
            success: true,
            stdout: "false".to_string(),
            stderr: String::new(),
        }),
    ]);
    let mut prompter = FakePrompter::with_replies([Some(false)]);

    let outcome = execute_github_star_plan(prompt_plan(), &mut runner, &mut prompter);

    assert_eq!(outcome, SetupGitHubOutcome::DeclinedAtPrompt);
    assert_eq!(
        runner.commands,
        vec![
            GhCommand::CheckAvailability,
            GhCommand::CheckAuthentication,
            GhCommand::CheckViewerHasStarred,
        ]
    );
}

#[test]
fn summary_reports_nonfatal_eligibility_failures_honestly() {
    let plan = SetupPlan::new(
        super::SetupRuntimePlan::Skip,
        super::SetupServicePlan::Skip,
        prompt_plan(),
    );

    let summary = github_summary(
        &plan,
        &SetupGitHubOutcome::EligibilityCheckFailed("timed out running `gh --version`".into()),
    );

    assert_eq!(
        summary,
        "skipped; GitHub eligibility check failed: timed out running `gh --version`"
    );
}
