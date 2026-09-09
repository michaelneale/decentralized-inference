mod actions;
mod command;
mod environment;
mod github;
mod github_runner;
mod plan;
mod planner;
mod prompt;
mod service;
pub(crate) mod service_files;
pub(crate) mod service_paths;
pub(crate) mod service_runner;
mod service_templates;
pub(crate) mod summary;

pub use actions::SetupActions;
pub use command::{SetupCommandArgs, run_setup, run_setup_command};
pub use environment::{SetupEnvironment, SetupOptions, SetupPlatform};
pub use plan::{
    SetupGitHubStarPlan, SetupGitHubStarSkipReason, SetupPlan, SetupRuntimePlan, SetupServicePlan,
    SetupStep,
};
pub use planner::{SetupPlanError, plan_setup};
pub use prompt::{SetupConfirmPrompt, SetupPromptDefault, SetupPromptKind, SetupPrompter};

#[cfg(test)]
mod cli_actions_tests;

#[cfg(test)]
mod command_tests;

#[cfg(test)]
mod github_tests;

#[cfg(test)]
mod orchestration_tests;

#[cfg(test)]
mod service_tests;

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod tests;
