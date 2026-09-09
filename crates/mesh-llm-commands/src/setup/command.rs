use super::prompt::confirm_yes_no;
use super::service_paths::ServiceInstallContext;
use super::service_runner::{CliServiceCommandRunner, ServiceCommandRunner};
use super::summary::{
    print_runtime_install_result, print_service_install_result, print_setup_summary,
};
use super::{
    SetupActions, SetupConfirmPrompt, SetupEnvironment, SetupGitHubStarPlan, SetupOptions,
    SetupPlan, SetupPrompter, SetupStep,
    github::{SetupGitHubOutcome, execute_github_star_plan},
    github_runner::{GhCommandRunner, ProcessGhCommandRunner},
    plan_setup,
    service::{ServiceInstallReport, install_service},
};
use crate::runtime_native::{
    NativeRuntimeConfigSelection, SetupNativeRuntimeOptions, SetupNativeRuntimeOutcome,
    SetupNativeRuntimePruneResult, install_and_prune_native_runtime_for_setup,
};
use anyhow::{Result, anyhow};
use std::future::Future;
use std::pin::Pin;

#[derive(Clone, Copy, Debug)]
pub struct SetupCommandArgs<'a> {
    pub options: SetupOptions,
    pub environment: SetupEnvironment,
    pub configured: NativeRuntimeConfigSelection<'a>,
}

pub async fn run_setup<P, A>(
    options: SetupOptions,
    environment: SetupEnvironment,
    prompter: &mut P,
    actions: &mut A,
) -> Result<SetupPlan>
where
    P: SetupPrompter,
    A: SetupActions,
    A::Error: Into<anyhow::Error>,
{
    let plan = plan_setup(options, environment, prompter).map_err(anyhow::Error::new)?;
    for step in plan.core_steps.iter().copied() {
        actions.run_step(step).await.map_err(Into::into)?;
    }
    actions
        .handle_github_star(plan.github_star, prompter)
        .await
        .map_err(Into::into)?;
    Ok(plan)
}

pub async fn run_setup_command(args: SetupCommandArgs<'_>) -> Result<()> {
    let mut prompter = CliSetupPrompter;
    let mut actions = CliSetupActions::new(args.environment, args.configured, args.options.verbose);
    let plan = run_setup(args.options, args.environment, &mut prompter, &mut actions).await?;
    print_setup_summary(&plan, &actions, args.options.verbose);
    Ok(())
}

struct CliSetupPrompter;

impl SetupPrompter for CliSetupPrompter {
    fn confirm(&mut self, prompt: SetupConfirmPrompt) -> Option<bool> {
        confirm_yes_no(prompt.message)
    }
}

pub(crate) struct CliSetupActions<'a> {
    environment: SetupEnvironment,
    configured: NativeRuntimeConfigSelection<'a>,
    pub(crate) runtime_outcome: Option<SetupNativeRuntimeOutcome>,
    service_context: Option<ServiceInstallContext>,
    service_runner: Box<dyn ServiceCommandRunner>,
    github_runner: Box<dyn GhCommandRunner>,
    pub(crate) service_outcome: SetupServiceOutcome,
    pub(crate) github_outcome: SetupGitHubOutcome,
    verbose: bool,
}

impl<'a> CliSetupActions<'a> {
    pub(crate) fn new(
        environment: SetupEnvironment,
        configured: NativeRuntimeConfigSelection<'a>,
        verbose: bool,
    ) -> Self {
        Self::with_support(
            environment,
            configured,
            None,
            Box::new(CliServiceCommandRunner),
            Box::new(ProcessGhCommandRunner::default()),
            verbose,
        )
    }

    fn with_support(
        environment: SetupEnvironment,
        configured: NativeRuntimeConfigSelection<'a>,
        service_context: Option<ServiceInstallContext>,
        service_runner: Box<dyn ServiceCommandRunner>,
        github_runner: Box<dyn GhCommandRunner>,
        verbose: bool,
    ) -> Self {
        Self {
            environment,
            configured,
            runtime_outcome: None,
            service_context,
            service_runner,
            github_runner,
            service_outcome: SetupServiceOutcome::NotRequested,
            github_outcome: SetupGitHubOutcome::NotEvaluated,
            verbose,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_service_support(
        environment: SetupEnvironment,
        configured: NativeRuntimeConfigSelection<'a>,
        service_context: ServiceInstallContext,
        service_runner: Box<dyn ServiceCommandRunner>,
        github_runner: Box<dyn GhCommandRunner>,
    ) -> Self {
        Self::with_support(
            environment,
            configured,
            Some(service_context),
            service_runner,
            github_runner,
            false,
        )
    }

    async fn install_runtime(&mut self) -> Result<()> {
        let outcome = install_and_prune_native_runtime_for_setup(SetupNativeRuntimeOptions {
            skip_runtime: false,
            requested_runtime: None,
            manifest_path: None,
            bundle_dirs: &[],
            cache_dir: None,
            configured: self.configured,
            progress: None,
        })
        .await?;
        if self.verbose {
            print_runtime_install_result(&outcome);
        }
        self.runtime_outcome = Some(outcome);
        Ok(())
    }

    fn report_runtime_prune(&self) -> Result<()> {
        let outcome = self
            .runtime_outcome
            .as_ref()
            .ok_or_else(|| anyhow!("setup runtime prune step ran before runtime install"))?;
        match &outcome.prune {
            SetupNativeRuntimePruneResult::Skipped => {}
            SetupNativeRuntimePruneResult::Pruned(plan) => {
                if self.verbose {
                    if plan.remove_dirs.is_empty() {
                        eprintln!("Native runtime cache is already clean");
                    } else {
                        eprintln!(
                            "Pruned {} inactive native runtime cache entr{}",
                            plan.remove_dirs.len(),
                            if plan.remove_dirs.len() == 1 {
                                "y"
                            } else {
                                "ies"
                            }
                        );
                    }
                }
            }
            SetupNativeRuntimePruneResult::Warning(warning) => {
                eprintln!("warning: native runtime installed, but cache pruning failed: {warning}");
            }
        }
        Ok(())
    }

    fn install_service(&mut self) -> Result<()> {
        let context = match self.service_context.clone() {
            Some(context) => context,
            None => ServiceInstallContext::detect(self.environment.platform, true)?,
        };
        let report = install_service(&context, self.service_runner.as_mut())?;
        print_service_install_result(&report, self.verbose);
        self.service_outcome = SetupServiceOutcome::Installed(report);
        Ok(())
    }

    fn print_service_guidance(&self) {
        eprintln!("Service not installed. Run `mesh-llm setup --service` to enable it later.");
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SetupServiceOutcome {
    NotRequested,
    Installed(ServiceInstallReport),
    PrintedGuidance,
}

impl SetupActions for CliSetupActions<'_> {
    type Error = anyhow::Error;
    type GitHubStarFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), Self::Error>> + 'a>>
    where
        Self: 'a;
    type StepFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), Self::Error>> + 'a>>
    where
        Self: 'a;

    fn run_step(&mut self, step: SetupStep) -> Self::StepFuture<'_> {
        Box::pin(async move {
            match step {
                SetupStep::InstallRuntime => self.install_runtime().await,
                SetupStep::PruneInactiveRuntimes => self.report_runtime_prune(),
                SetupStep::InstallService => self.install_service(),
                SetupStep::PrintServiceGuidance => {
                    self.service_outcome = SetupServiceOutcome::PrintedGuidance;
                    self.print_service_guidance();
                    Ok(())
                }
            }
        })
    }

    fn handle_github_star<'a>(
        &'a mut self,
        plan: SetupGitHubStarPlan,
        prompter: &'a mut dyn super::SetupPrompter,
    ) -> Self::GitHubStarFuture<'a> {
        Box::pin(async move {
            self.github_outcome =
                execute_github_star_plan(plan, &mut *self.github_runner, prompter);
            Ok(())
        })
    }
}
