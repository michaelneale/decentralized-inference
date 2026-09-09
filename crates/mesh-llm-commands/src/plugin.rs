use std::io::Write;

use anyhow::{Result, bail};
use mesh_llm_plugin_manager::install::install_plugin_archive;
use mesh_llm_plugin_manager::{
    PluginCatalog, PluginInstallOptions, PluginProgressEvent, PluginProgressReporter, PluginStore,
    default_store_root, install_plugin, update_plugin,
};
use reqwest::Client;

use mesh_llm_cli::PluginCommand;
use mesh_llm_tui::terminal_progress::{
    SpinnerHandle, clear_stderr_line, ratio_complete_u64, render_inline_gauge, start_spinner,
};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PluginListRows {
    pub externals: Vec<RuntimePluginRow>,
    pub inactive: Vec<InactivePluginRow>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimePluginRow {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InactivePluginRow {
    pub name: String,
    pub kind: String,
    pub status: String,
    pub error: Option<String>,
}

pub async fn run_plugin_command(
    command: &PluginCommand,
    runtime_rows: Option<&PluginListRows>,
) -> Result<bool> {
    match command {
        PluginCommand::Install {
            reference,
            archive,
            name,
            version,
        } => {
            install(
                reference.as_deref(),
                archive.as_deref(),
                name.as_deref(),
                version.as_deref().unwrap_or("dev"),
            )
            .await?
        }
        PluginCommand::Update { name } => update(name).await?,
        PluginCommand::Enable { name } => set_enabled(name, true)?,
        PluginCommand::Disable { name } => set_enabled(name, false)?,
        PluginCommand::Delete { name } => delete(name)?,
        PluginCommand::Info { name } => return info(name, runtime_rows),
        PluginCommand::Search { query } => search(query.as_deref()).await?,
        PluginCommand::List => {
            let Some(runtime_rows) = runtime_rows else {
                return Ok(false);
            };
            list(runtime_rows)?;
        }
    }
    Ok(true)
}

async fn install(
    reference: Option<&str>,
    archive: Option<&std::path::Path>,
    name: Option<&str>,
    version: &str,
) -> Result<()> {
    let options = PluginInstallOptions::from_env()?;
    let mut progress = CliPluginProgress::default();
    let outcome = match (reference, archive, name) {
        (Some(reference), None, None) => install_plugin(reference, &options, &mut progress).await?,
        (None, Some(archive), Some(name)) => {
            install_plugin_archive(name, version, archive, &options, &mut progress)?
        }
        _ => bail!("provide either a plugin reference or --archive with --name"),
    };
    progress.finish();
    if outcome.changed {
        eprintln!(
            "✅ Installed {} {}",
            outcome.metadata.name, outcome.metadata.installed_version
        );
    }
    Ok(())
}

async fn update(name: &str) -> Result<()> {
    let options = PluginInstallOptions::from_env()?;
    let mut progress = CliPluginProgress::default();
    let outcome = update_plugin(name, &options, &mut progress).await?;
    progress.finish();
    if outcome.changed {
        eprintln!(
            "✅ Updated {} to {}",
            outcome.metadata.name, outcome.metadata.installed_version
        );
    }
    Ok(())
}

fn set_enabled(name: &str, enabled: bool) -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    let metadata = store.set_enabled(name, enabled)?;
    if metadata.enabled {
        eprintln!("✅ Enabled {}", metadata.name);
    } else {
        eprintln!("⏸️  Disabled {}", metadata.name);
    }
    Ok(())
}

fn delete(name: &str) -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    store.delete(name)?;
    eprintln!("🗑️  Deleted {name}");
    Ok(())
}

fn info(name: &str, runtime_rows: Option<&PluginListRows>) -> Result<bool> {
    let store = PluginStore::new(default_store_root()?);
    if let Some(metadata) = store.load_optional(name)? {
        println!("name\t{}", metadata.name);
        println!("version\t{}", metadata.installed_version);
        println!("enabled\t{}", metadata.enabled);
        println!("source\t{}", metadata.source_repository);
        println!("target\t{}", metadata.target_triple);
        println!("asset\t{}", metadata.downloaded_asset_name);
        println!("path\t{}", metadata.install_path.display());
        if let Some(protocol) = metadata.last_protocol_version {
            println!("protocol\t{protocol}");
        }
        if let Some(status) = metadata.last_status {
            println!("status\t{status}");
        }
        if let Some(error) = metadata.last_error {
            println!("error\t{error}");
        }
        return Ok(true);
    }
    let Some(runtime_rows) = runtime_rows else {
        return Ok(false);
    };
    if let Some(row) = runtime_rows.externals.iter().find(|row| row.name == name) {
        for line in runtime_plugin_info_lines(row) {
            println!("{line}");
        }
        return Ok(true);
    }
    if let Some(row) = runtime_rows.inactive.iter().find(|row| row.name == name) {
        for line in inactive_plugin_info_lines(row) {
            println!("{line}");
        }
        return Ok(true);
    }
    bail!("plugin '{name}' is not installed")
}

fn runtime_plugin_info_lines(row: &RuntimePluginRow) -> Vec<String> {
    vec![
        format!("name\t{}", row.name),
        "kind\truntime".to_string(),
        format!("command\t{}", row.command),
        format!("args\t{}", row.args.join(" ")),
        "source\tbuilt-in/runtime".to_string(),
    ]
}

fn inactive_plugin_info_lines(row: &InactivePluginRow) -> Vec<String> {
    vec![
        format!("name\t{}", row.name),
        format!("kind\t{}", row.kind),
        format!("status\t{}", row.status),
        format!("error\t{}", row.error.clone().unwrap_or_default()),
    ]
}

async fn search(query: Option<&str>) -> Result<()> {
    let options = PluginInstallOptions::from_env()?;
    let mut spinner = start_spinner("Searching plugin catalog");
    let catalog = PluginCatalog::fetch(&Client::new(), &options.catalog_url).await;
    spinner.finish();
    let catalog = catalog?;
    let hits = catalog.search(query);
    if hits.is_empty() {
        eprintln!("🔎 No plugins found");
        return Ok(());
    }
    for entry in hits {
        println!(
            "{}\t{}\t{}\t{} <{}>",
            entry.name, entry.description, entry.github_url, entry.author_name, entry.author_email
        );
    }
    Ok(())
}

fn list(runtime_rows: &PluginListRows) -> Result<()> {
    let store = PluginStore::new(default_store_root()?);
    for metadata in store.list()? {
        let state = if metadata.enabled {
            "enabled"
        } else {
            "disabled"
        };
        println!(
            "{}\tversion={}\tstate={}\tsource={}",
            metadata.name, metadata.installed_version, state, metadata.source_repository
        );
    }

    for spec in &runtime_rows.externals {
        println!(
            "{}\tkind=runtime\tcommand={}\targs={}",
            spec.name,
            spec.command,
            spec.args.join(" ")
        );
    }
    for summary in &runtime_rows.inactive {
        println!(
            "{}\tkind={}\tstate={}\terror={}",
            summary.name,
            summary.kind,
            summary.status,
            summary.error.clone().unwrap_or_default()
        );
    }
    Ok(())
}

#[derive(Default)]
struct CliPluginProgress {
    spinner: Option<SpinnerHandle>,
    active_download: Option<String>,
    last_percent: Option<u64>,
}

impl CliPluginProgress {
    fn finish(&mut self) {
        if let Some(mut spinner) = self.spinner.take() {
            spinner.finish();
        }
        if self.active_download.take().is_some() {
            let _ = clear_stderr_line();
        }
    }

    fn spinner(&mut self, message: String) {
        self.finish();
        self.spinner = Some(start_spinner(&message));
    }

    fn started_download(&mut self, asset: String, total_bytes: Option<u64>) {
        self.finish();
        self.active_download = Some(asset.clone());
        self.last_percent = None;
        eprintln!("⬇️  Downloading {asset}");
        if let Some(total) = total_bytes {
            eprintln!("   size: {}", format_bytes(total));
        }
    }

    fn download_progress(&mut self, downloaded: u64, total: Option<u64>) {
        let Some(asset) = self.active_download.as_deref() else {
            return;
        };
        if let Some(total) = total.filter(|total| *total > 0) {
            let percent = downloaded.saturating_mul(100) / total;
            if self.last_percent == Some(percent) {
                return;
            }
            self.last_percent = Some(percent);
            let gauge = render_inline_gauge(
                ratio_complete_u64(downloaded, total),
                &format!(
                    "⬇️  {} {} / {} ({}%)",
                    asset,
                    format_bytes(downloaded),
                    format_bytes(total),
                    percent
                ),
            );
            eprint!("\r\x1b[2K{gauge}");
            let _ = std::io::stderr().flush();
        }
    }
}

impl PluginProgressReporter for CliPluginProgress {
    fn report(&mut self, event: PluginProgressEvent) {
        match event {
            PluginProgressEvent::ResolvingCatalog { name } => {
                self.spinner(format!("Looking up {name} in the plugin catalog"));
            }
            PluginProgressEvent::ResolvingGitHub { repo } => {
                self.spinner(format!("Checking GitHub releases for {repo}"));
            }
            PluginProgressEvent::SelectingAsset { target } => {
                self.spinner(format!("Finding compatible plugin asset for {target}"));
            }
            PluginProgressEvent::DownloadStarted { asset, total_bytes } => {
                self.started_download(asset, total_bytes);
            }
            PluginProgressEvent::DownloadProgress {
                downloaded_bytes,
                total_bytes,
            } => self.download_progress(downloaded_bytes, total_bytes),
            PluginProgressEvent::DownloadFinished { asset } => {
                self.finish();
                eprintln!("✅ Downloaded {asset}");
            }
            PluginProgressEvent::Extracting { asset } => {
                self.spinner(format!("Installing {asset}"));
            }
            PluginProgressEvent::Installed { name, version } => {
                self.finish();
                eprintln!("📦 Installed {name} {version}");
            }
            PluginProgressEvent::Updated { name, from, to } => {
                self.finish();
                eprintln!("⬆️  Updated {name} {from} -> {to}");
            }
            PluginProgressEvent::AlreadyCurrent { name, version } => {
                self.finish();
                eprintln!("✅ {name} is up to date ({version})");
            }
        }
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = UNITS[0];
    for candidate in &UNITS[1..] {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = candidate;
    }
    if unit == "B" {
        format!("{bytes} {unit}")
    } else {
        format!("{value:.1} {unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_plugin_info_lines_describe_builtin_runtime_plugin() {
        let row = RuntimePluginRow {
            name: "blobstore".to_string(),
            command: "/tmp/mesh-llm".to_string(),
            args: vec![
                "--log-format".to_string(),
                "json".to_string(),
                "--plugin".to_string(),
                "blobstore".to_string(),
            ],
        };

        assert_eq!(
            runtime_plugin_info_lines(&row),
            vec![
                "name\tblobstore".to_string(),
                "kind\truntime".to_string(),
                "command\t/tmp/mesh-llm".to_string(),
                "args\t--log-format json --plugin blobstore".to_string(),
                "source\tbuilt-in/runtime".to_string(),
            ]
        );
    }

    #[test]
    fn inactive_plugin_info_lines_describe_startup_failure() {
        let row = InactivePluginRow {
            name: "image-tools".to_string(),
            kind: "external".to_string(),
            status: "inactive".to_string(),
            error: Some("command not found".to_string()),
        };

        assert_eq!(
            inactive_plugin_info_lines(&row),
            vec![
                "name\timage-tools".to_string(),
                "kind\texternal".to_string(),
                "status\tinactive".to_string(),
                "error\tcommand not found".to_string(),
            ]
        );
    }
}
