#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SetupOptions {
    pub yes: bool,
    pub no_interactive: bool,
    pub service: bool,
    pub no_service: bool,
    pub skip_runtime: bool,
    pub verbose: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SetupEnvironment {
    pub platform: SetupPlatform,
    pub interactive: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetupPlatform {
    Linux,
    MacOs,
    Windows,
}

impl SetupEnvironment {
    pub const fn prompts_visible(self, options: SetupOptions) -> bool {
        self.interactive && !options.no_interactive && !options.yes
    }
}

impl SetupPlatform {
    pub const fn supports_service(self) -> bool {
        matches!(self, Self::Linux | Self::MacOs)
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Linux => "linux",
            Self::MacOs => "macos",
            Self::Windows => "windows",
        }
    }
}
