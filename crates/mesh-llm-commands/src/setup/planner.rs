use super::{
    SetupConfirmPrompt, SetupEnvironment, SetupGitHubStarPlan, SetupGitHubStarSkipReason,
    SetupOptions, SetupPlan, SetupPlatform, SetupPromptDefault, SetupPromptKind, SetupPrompter,
    SetupRuntimePlan, SetupServicePlan,
};
use std::error::Error;
use std::fmt::{self, Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SetupPlanError {
    ConflictingServiceFlags,
    UnsupportedService { platform: SetupPlatform },
}

impl Display for SetupPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConflictingServiceFlags => {
                f.write_str("setup received both --service and --no-service")
            }
            Self::UnsupportedService { platform } => write!(
                f,
                "setup cannot install the background service on {}",
                platform.display_name()
            ),
        }
    }
}

impl Error for SetupPlanError {}

pub fn plan_setup<P: SetupPrompter>(
    options: SetupOptions,
    environment: SetupEnvironment,
    prompter: &mut P,
) -> Result<SetupPlan, SetupPlanError> {
    if options.service && options.no_service {
        return Err(SetupPlanError::ConflictingServiceFlags);
    }

    if options.service && matches!(environment.platform, SetupPlatform::Windows) {
        return Err(SetupPlanError::UnsupportedService {
            platform: environment.platform,
        });
    }

    let runtime = if options.skip_runtime {
        SetupRuntimePlan::Skip
    } else {
        SetupRuntimePlan::InstallAndPrune
    };
    let service = plan_service(options, environment, prompter);
    let github_star = plan_github_star(options, environment);
    Ok(SetupPlan::new(runtime, service, github_star))
}

fn plan_service<P: SetupPrompter>(
    options: SetupOptions,
    environment: SetupEnvironment,
    prompter: &mut P,
) -> SetupServicePlan {
    if !environment.platform.supports_service() {
        return SetupServicePlan::Skip;
    }

    if options.no_service {
        return SetupServicePlan::Skip;
    }

    if options.service || options.yes {
        return SetupServicePlan::Install;
    }

    if !environment.prompts_visible(options) {
        return SetupServicePlan::PrintGuidance;
    }

    let prompt = SetupConfirmPrompt {
        kind: SetupPromptKind::InstallService,
        message: "Install the background service?",
        default: SetupPromptDefault::Yes,
    };
    let accepted = prompt.default.resolve(prompter.confirm(prompt));
    if accepted {
        SetupServicePlan::Install
    } else {
        SetupServicePlan::Skip
    }
}

fn plan_github_star(options: SetupOptions, environment: SetupEnvironment) -> SetupGitHubStarPlan {
    if options.yes {
        return SetupGitHubStarPlan::Skip(SetupGitHubStarSkipReason::AutomaticYes);
    }

    if environment.prompts_visible(options) {
        SetupGitHubStarPlan::PromptIfEligible
    } else {
        SetupGitHubStarPlan::Skip(SetupGitHubStarSkipReason::HiddenPrompt)
    }
}
