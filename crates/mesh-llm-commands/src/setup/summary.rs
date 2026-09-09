use super::SetupPlan;
use super::command::{CliSetupActions, SetupServiceOutcome};
use super::service::ServiceInstallStatus;
use crate::runtime_native::{
    SetupNativeRuntimeOutcome, SetupNativeRuntimePruneResult, SetupNativeRuntimeStatus,
};
use crate::terminal::{style_muted, style_ok, style_warn};
use mesh_llm_runtime_install::NativeRuntimeInstallStatus;

pub(crate) fn print_runtime_install_result(outcome: &SetupNativeRuntimeOutcome) {
    match &outcome.status {
        SetupNativeRuntimeStatus::Skipped => {}
        SetupNativeRuntimeStatus::Installed(installed) => match installed.status {
            NativeRuntimeInstallStatus::Installed => eprintln!(
                "{} Installed native runtime {} for mesh version {}",
                style_ok("✓"),
                installed.runtime.native_runtime_id,
                installed.runtime.mesh_version
            ),
            NativeRuntimeInstallStatus::AlreadyInstalled => eprintln!(
                "{} Native runtime {} is already installed for mesh version {}",
                style_ok("✓"),
                installed.runtime.native_runtime_id,
                installed.runtime.mesh_version
            ),
        },
    }
}

pub(crate) fn print_service_install_result(
    report: &crate::setup::service::ServiceInstallReport,
    verbose: bool,
) {
    if verbose {
        for line in &report.messages {
            eprintln!("{line}");
        }
    }
}

pub(crate) fn print_setup_summary(plan: &SetupPlan, actions: &CliSetupActions<'_>, verbose: bool) {
    eprintln!();
    if verbose {
        eprintln!("Setup summary");
        eprintln!("- Runtime: {}", runtime_summary(plan, actions));
        eprintln!("- Service: {}", service_summary(plan, actions));
        eprintln!(
            "- GitHub star: {}",
            super::github::github_summary(plan, &actions.github_outcome)
        );
        return;
    }

    eprintln!("{} Mesh setup complete", style_ok("✓"));
    eprintln!("  Runtime  {}", runtime_brief(plan, actions));
    eprintln!("  Service  {}", service_brief(plan, actions));
    if let Some(github) = github_brief(actions) {
        eprintln!("  GitHub star  {github}");
    }
}

fn runtime_summary(plan: &SetupPlan, actions: &CliSetupActions<'_>) -> String {
    match plan.runtime {
        super::SetupRuntimePlan::Skip => "skipped by --skip-runtime".to_string(),
        super::SetupRuntimePlan::InstallAndPrune => match actions.runtime_outcome.as_ref() {
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Installed(installed),
                prune: SetupNativeRuntimePruneResult::Pruned(plan),
            }) => {
                let install_status = match installed.status {
                    NativeRuntimeInstallStatus::Installed => "installed",
                    NativeRuntimeInstallStatus::AlreadyInstalled => "already installed",
                };
                if plan.remove_dirs.is_empty() {
                    format!("{install_status}; cache already clean")
                } else {
                    format!(
                        "{install_status}; pruned {} inactive cache entr{}",
                        plan.remove_dirs.len(),
                        if plan.remove_dirs.len() == 1 {
                            "y"
                        } else {
                            "ies"
                        }
                    )
                }
            }
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Installed(installed),
                prune: SetupNativeRuntimePruneResult::Warning(_),
            }) => match installed.status {
                NativeRuntimeInstallStatus::Installed => {
                    "installed; cache prune warning reported above".to_string()
                }
                NativeRuntimeInstallStatus::AlreadyInstalled => {
                    "already installed; cache prune warning reported above".to_string()
                }
            },
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Installed(installed),
                prune: SetupNativeRuntimePruneResult::Skipped,
            }) => match installed.status {
                NativeRuntimeInstallStatus::Installed => "installed".to_string(),
                NativeRuntimeInstallStatus::AlreadyInstalled => "already installed".to_string(),
            },
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Skipped,
                ..
            }) => "skipped".to_string(),
            None => "not recorded".to_string(),
        },
    }
}

fn runtime_brief(plan: &SetupPlan, actions: &CliSetupActions<'_>) -> String {
    match plan.runtime {
        super::SetupRuntimePlan::Skip => style_muted("skipped (--skip-runtime)"),
        super::SetupRuntimePlan::InstallAndPrune => match actions.runtime_outcome.as_ref() {
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Installed(installed),
                prune: SetupNativeRuntimePruneResult::Warning(_),
            }) => match installed.status {
                NativeRuntimeInstallStatus::Installed => style_warn("installed; prune warning"),
                NativeRuntimeInstallStatus::AlreadyInstalled => {
                    style_warn("already installed; prune warning")
                }
            },
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Installed(installed),
                ..
            }) => match installed.status {
                NativeRuntimeInstallStatus::Installed => style_ok("ready"),
                NativeRuntimeInstallStatus::AlreadyInstalled => style_ok("already ready"),
            },
            Some(SetupNativeRuntimeOutcome {
                status: SetupNativeRuntimeStatus::Skipped,
                ..
            }) => style_muted("skipped"),
            None => style_muted("not recorded"),
        },
    }
}

fn service_brief(plan: &SetupPlan, actions: &CliSetupActions<'_>) -> String {
    match plan.service {
        super::SetupServicePlan::Skip => style_muted("not installed"),
        super::SetupServicePlan::Install => match actions.service_outcome {
            SetupServiceOutcome::Installed(ref report) => match report.status {
                ServiceInstallStatus::Started => style_ok("running"),
                ServiceInstallStatus::NeedsManualStart => style_warn("installed; start manually"),
            },
            SetupServiceOutcome::NotRequested | SetupServiceOutcome::PrintedGuidance => {
                style_muted("not recorded")
            }
        },
        super::SetupServicePlan::PrintGuidance => match actions.service_outcome {
            SetupServiceOutcome::PrintedGuidance => style_muted("not installed"),
            SetupServiceOutcome::NotRequested | SetupServiceOutcome::Installed(_) => {
                style_muted("not recorded")
            }
        },
    }
}

fn github_brief(actions: &CliSetupActions<'_>) -> Option<String> {
    match actions.github_outcome {
        super::github::SetupGitHubOutcome::Starred => Some(style_ok("starred")),
        super::github::SetupGitHubOutcome::AlreadyStarred => Some(style_ok("already starred")),
        super::github::SetupGitHubOutcome::StarRequestFailed(_)
        | super::github::SetupGitHubOutcome::EligibilityCheckFailed(_) => {
            Some(style_warn("not starred"))
        }
        super::github::SetupGitHubOutcome::CliUnavailable => {
            Some(style_muted("skipped; gh unavailable"))
        }
        super::github::SetupGitHubOutcome::NotAuthenticated => {
            Some(style_muted("skipped; gh not authenticated"))
        }
        super::github::SetupGitHubOutcome::NotEvaluated => Some(style_muted("not recorded")),
        _ => None,
    }
}

pub(crate) fn service_summary(plan: &SetupPlan, actions: &CliSetupActions<'_>) -> String {
    match plan.service {
        super::SetupServicePlan::Skip => "not requested".to_string(),
        super::SetupServicePlan::Install => match actions.service_outcome {
            SetupServiceOutcome::Installed(ref report) => report.summary.clone(),
            SetupServiceOutcome::NotRequested | SetupServiceOutcome::PrintedGuidance => {
                "not recorded".to_string()
            }
        },
        super::SetupServicePlan::PrintGuidance => match actions.service_outcome {
            SetupServiceOutcome::PrintedGuidance => {
                "not installed; printed follow-up guidance".to_string()
            }
            SetupServiceOutcome::NotRequested | SetupServiceOutcome::Installed(_) => {
                "not recorded".to_string()
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_native::NativeRuntimeConfigSelection;
    use crate::setup::SetupEnvironment;
    use crate::setup::SetupPlatform;
    use crate::setup::github::SetupGitHubOutcome;
    use crate::terminal::strip_ansi_styles;

    fn actions_with_github_outcome(github_outcome: SetupGitHubOutcome) -> CliSetupActions<'static> {
        let mut actions = CliSetupActions::new(
            SetupEnvironment {
                platform: SetupPlatform::Linux,
                interactive: false,
            },
            NativeRuntimeConfigSelection::default(),
            false,
        );
        actions.github_outcome = github_outcome;
        actions
    }

    #[test]
    fn github_brief_reports_unavailable_cli() {
        let actions = actions_with_github_outcome(SetupGitHubOutcome::CliUnavailable);

        assert_eq!(
            github_brief(&actions).map(|brief| strip_ansi_styles(&brief)),
            Some("skipped; gh unavailable".to_string())
        );
    }

    #[test]
    fn github_brief_reports_unauthenticated_cli() {
        let actions = actions_with_github_outcome(SetupGitHubOutcome::NotAuthenticated);

        assert_eq!(
            github_brief(&actions).map(|brief| strip_ansi_styles(&brief)),
            Some("skipped; gh not authenticated".to_string())
        );
    }
}
