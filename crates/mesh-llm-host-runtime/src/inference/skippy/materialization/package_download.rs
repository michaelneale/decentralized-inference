use std::{
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};
use sha2::{Digest, Sha256};
use skippy_package_format::{
    Artifact as PackageV2Artifact, PackageManifest as PackageManifestV2,
    stage_admission::StageAdmissionDescriptor as PackageV2StageAdmissionDescriptor,
};
use skippy_runtime::package::{self, PackageIntegrityOptions, PackageStageRequest};

use mesh_llm_events::terminal_progress::{
    SpinnerHandle, ratio_complete_u64, render_inline_gauge_with_reserved_width, start_spinner,
};
use mesh_llm_events::{ModelProgressStatus, OutputEvent, emit_event, interactive_tui_active};

#[path = "cache_resolution.rs"]
mod cache_resolution;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StagePackageRef {
    LocalPackage(PathBuf),
    HuggingFacePackage {
        repo: String,
        revision: Option<String>,
    },
    SyntheticDirectGguf(PathBuf),
}

impl StagePackageRef {
    pub fn parse(value: &str) -> Result<Self> {
        if let Some(rest) = value.strip_prefix("hf://") {
            let (repo, revision) = if let Some((repo, revision)) = rest.split_once('@') {
                (repo, Some(revision.to_string()))
            } else if let Some(index) = rest.rfind(':') {
                (&rest[..index], Some(rest[index + 1..].to_string()))
            } else {
                (rest, None)
            };
            if repo.split('/').count() != 2 || repo.contains(':') || repo.contains('@') {
                bail!("HF package repo id must look like namespace/repo");
            }
            return Ok(Self::HuggingFacePackage {
                repo: repo.to_string(),
                revision,
            });
        }

        let path = PathBuf::from(value);
        if path.join("model-package.json").is_file() {
            return Ok(Self::LocalPackage(path));
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
            return Ok(Self::SyntheticDirectGguf(path));
        }

        bail!("not a skippy package ref: {value}");
    }

    pub fn is_distributable_package(&self) -> bool {
        matches!(
            self,
            Self::LocalPackage(_) | Self::HuggingFacePackage { .. }
        )
    }

    pub fn as_package_ref(&self) -> Option<String> {
        match self {
            Self::LocalPackage(path) => Some(path.to_string_lossy().to_string()),
            Self::HuggingFacePackage { repo, revision } => Some(match revision {
                Some(revision) => format!("hf://{repo}@{revision}"),
                None => format!("hf://{repo}"),
            }),
            Self::SyntheticDirectGguf(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
struct LayerPackageDownloadProgressState {
    downloaded: u64,
    total: u64,
    bytes_per_sec: Option<f64>,
    last_draw: Option<Instant>,
    showed_progress: bool,
}

struct LayerPackageDownloadProgress {
    label: String,
    file: String,
    package_scope: Option<Arc<LayerPackageDownloadScope>>,
    completed_before: usize,
    preflight_spinner: Mutex<Option<SpinnerHandle>>,
    state: Mutex<LayerPackageDownloadProgressState>,
}

struct LayerPackageDownloadScope {
    package: String,
    total_files: usize,
    state: Mutex<LayerPackageDownloadScopeState>,
}

#[derive(Debug)]
struct LayerPackageDownloadScopeState {
    announced: bool,
    drawn_line: bool,
}

impl LayerPackageDownloadScope {
    fn new(label: &str, total_files: usize) -> Self {
        Self {
            package: layer_package_progress_package(label).to_string(),
            total_files,
            state: Mutex::new(LayerPackageDownloadScopeState {
                announced: false,
                drawn_line: false,
            }),
        }
    }

    fn has_drawn(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.announced || state.drawn_line)
            .unwrap_or(false)
    }

    fn complete_count(&self, completed: usize) -> usize {
        completed.min(self.total_files)
    }

    fn draw(
        &self,
        file: &str,
        completed_files: usize,
        downloaded: u64,
        total: u64,
        bytes_per_sec: Option<f64>,
        force: bool,
    ) {
        let Ok(mut scope_state) = self.state.lock() else {
            return;
        };
        if !scope_state.announced {
            eprintln!(
                "\r\x1b[K📦 Downloading layer package {} ({} file(s))",
                self.package, self.total_files
            );
            scope_state.announced = true;
        }
        let percent = if total == 0 {
            0
        } else {
            ((downloaded as f64 / total as f64) * 1000.0).round() as usize
        };
        let percent_major = (percent.min(1000)) / 10;
        let percent_minor = (percent.min(1000)) % 10;
        let speed_suffix = bytes_per_sec
            .filter(|bytes_per_sec| *bytes_per_sec > 0.0)
            .map(|bytes_per_sec| {
                format!(
                    " at {}/s",
                    format_layer_package_download_bytes(bytes_per_sec as u64)
                )
            })
            .unwrap_or_default();
        let (ratio, total_display) = match total {
            0 => (0.0, "?".to_string()),
            total => (
                ratio_complete_u64(downloaded, total),
                format_layer_package_download_bytes(total),
            ),
        };
        let gauge = render_inline_gauge_with_reserved_width(
            ratio,
            &format!(
                "⏬ {} {:>3}.{:01}% ({}/{}){}   files {}/{} complete",
                layer_package_artifact_display_for_package(&self.package, file),
                percent_major,
                percent_minor,
                format_layer_package_download_bytes(downloaded),
                total_display,
                speed_suffix,
                self.complete_count(completed_files),
                self.total_files,
            ),
            3,
        );
        eprint!("\r\x1b[K   {gauge}");
        let _ = std::io::stderr().flush();
        scope_state.drawn_line = true;
        if force {
            eprintln!();
            scope_state.drawn_line = false;
        }
    }
}

impl LayerPackageDownloadProgress {
    fn new(
        label: String,
        file: String,
        total_bytes: Option<u64>,
        package_scope: Option<Arc<LayerPackageDownloadScope>>,
        completed_before: usize,
    ) -> Self {
        let preflight_spinner = if interactive_tui_active()
            || package_scope
                .as_ref()
                .is_some_and(|scope| scope.has_drawn())
        {
            None
        } else {
            Some(start_spinner(&format!("Preparing download {file}")))
        };
        Self {
            label,
            file,
            package_scope,
            completed_before,
            preflight_spinner: Mutex::new(preflight_spinner),
            state: Mutex::new(LayerPackageDownloadProgressState {
                downloaded: 0,
                total: total_bytes.unwrap_or(0),
                bytes_per_sec: None,
                last_draw: None,
                showed_progress: false,
            }),
        }
    }

    fn emit(
        &self,
        downloaded_bytes: Option<u64>,
        total_bytes: Option<u64>,
        status: ModelProgressStatus,
    ) {
        let _ = emit_event(OutputEvent::ModelDownloadProgress {
            label: self.label.clone(),
            file: Some(self.file.clone()),
            downloaded_bytes,
            total_bytes,
            status,
        });
    }

    fn emit_ensuring(&self) {
        if !interactive_tui_active() {
            return;
        }
        let total = self
            .state
            .lock()
            .ok()
            .and_then(|state| (state.total > 0).then_some(state.total));
        self.emit(None, total, ModelProgressStatus::Ensuring);
    }

    fn emit_ready(&self, path: &Path) {
        let total = fs::metadata(path)
            .ok()
            .map(|metadata| metadata.len())
            .or_else(|| {
                self.state
                    .lock()
                    .ok()
                    .and_then(|state| (state.total > 0).then_some(state.total))
            });
        if interactive_tui_active() {
            self.emit(total, total, ModelProgressStatus::Ready);
            return;
        }
        if let Ok(mut spinner) = self.preflight_spinner.lock() {
            spinner.take();
        }
        let showed_progress = self
            .state
            .lock()
            .map(|state| state.showed_progress)
            .unwrap_or(false);
        if let Some(scope) = &self.package_scope {
            if !showed_progress {
                let total = total.unwrap_or(0);
                scope.draw(
                    &self.file,
                    self.completed_before + 1,
                    total,
                    total,
                    None,
                    true,
                );
            }
            return;
        }
        if !showed_progress {
            let file = layer_package_artifact_display(&self.label, &self.file);
            match total {
                Some(total) if total > 0 => eprintln!(
                    "   ✅ Ready {} ({})",
                    file,
                    format_layer_package_download_bytes(total)
                ),
                _ => eprintln!("   ✅ Ready {}", file),
            }
        }
    }

    fn draw(&self, state: &mut LayerPackageDownloadProgressState, force: bool) {
        if !force && state.downloaded == 0 && state.total == 0 {
            return;
        }
        let now = Instant::now();
        if !force
            && state
                .last_draw
                .is_some_and(|last| now.duration_since(last) < Duration::from_millis(150))
        {
            return;
        }
        state.last_draw = Some(now);
        state.showed_progress = true;
        if interactive_tui_active() {
            self.emit(
                (state.downloaded > 0).then_some(state.downloaded),
                (state.total > 0).then_some(state.total),
                ModelProgressStatus::Downloading,
            );
            return;
        }
        if let Ok(mut spinner) = self.preflight_spinner.lock() {
            spinner.take();
        }
        if let Some(scope) = &self.package_scope {
            let completed = if force {
                self.completed_before + 1
            } else {
                self.completed_before
            };
            scope.draw(
                &self.file,
                completed,
                state.downloaded,
                state.total,
                state.bytes_per_sec,
                force,
            );
        } else {
            draw_layer_package_file_progress(
                &layer_package_artifact_display(&self.label, &self.file),
                state.downloaded,
                state.total,
                state.bytes_per_sec,
                force,
            );
        }
    }
}

impl Drop for LayerPackageDownloadProgress {
    fn drop(&mut self) {
        if let Ok(mut spinner) = self.preflight_spinner.lock() {
            spinner.take();
        }
    }
}

impl ProgressHandler for LayerPackageDownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                if *total_bytes > 0 {
                    state.total = state.total.max(*total_bytes);
                }
            }
            DownloadEvent::Progress { files } => {
                if !files.is_empty() {
                    let downloaded: u64 = files.iter().map(|file| file.bytes_completed).sum();
                    state.downloaded = state.downloaded.max(downloaded);
                    let total: u64 = files.iter().map(|file| file.total_bytes).sum();
                    if total > 0 {
                        state.total = state.total.max(total);
                    }
                }
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            } => {
                state.downloaded = state.downloaded.max(*bytes_completed);
                if *total_bytes > 0 {
                    state.total = state.total.max(*total_bytes);
                }
                state.bytes_per_sec = *bytes_per_sec;
            }
            DownloadEvent::Complete => {
                if state.total > 0 {
                    state.downloaded = state.total;
                }
                state.bytes_per_sec = None;
            }
        }
        let should_show_progress = state.downloaded > 0 || state.total > 0;
        let force = matches!(event, DownloadEvent::Complete) && should_show_progress;
        if should_show_progress {
            self.draw(&mut state, force);
        } else if matches!(event, DownloadEvent::Complete)
            && let Ok(mut spinner) = self.preflight_spinner.lock()
        {
            spinner.take();
        }
    }
}

fn format_layer_package_download_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.0}KB", bytes as f64 / 1e3)
    } else {
        format!("{bytes}B")
    }
}

fn layer_package_progress_package(label: &str) -> &str {
    label.strip_prefix("layer package ").unwrap_or(label)
}

fn layer_package_progress_repo(package: &str) -> &str {
    package
        .split_once('@')
        .map(|(repo, _)| repo)
        .unwrap_or(package)
}

fn layer_package_artifact_display(label: &str, file: &str) -> String {
    layer_package_artifact_display_for_package(layer_package_progress_package(label), file)
}

fn layer_package_artifact_display_for_package(package: &str, file: &str) -> String {
    let repo = layer_package_progress_repo(package);
    if file.starts_with(repo) || file.starts_with('/') {
        file.to_string()
    } else {
        format!("{repo}/{file}")
    }
}

fn draw_layer_package_file_progress(
    file: &str,
    downloaded: u64,
    total: u64,
    bytes_per_sec: Option<f64>,
    force: bool,
) {
    let percent = if total == 0 {
        0
    } else {
        ((downloaded as f64 / total as f64) * 1000.0).round() as usize
    };
    let percent_major = (percent.min(1000)) / 10;
    let percent_minor = (percent.min(1000)) % 10;
    let speed_suffix = bytes_per_sec
        .filter(|bytes_per_sec| *bytes_per_sec > 0.0)
        .map(|bytes_per_sec| {
            format!(
                " at {}/s",
                format_layer_package_download_bytes(bytes_per_sec as u64)
            )
        })
        .unwrap_or_default();
    let (ratio, total_display) = match total {
        0 => (0.0, "?".to_string()),
        total => (
            ratio_complete_u64(downloaded, total),
            format_layer_package_download_bytes(total),
        ),
    };
    let gauge = render_inline_gauge_with_reserved_width(
        ratio,
        &format!(
            "⏬ {} {:>3}.{:01}% ({}/{}){}",
            file,
            percent_major,
            percent_minor,
            format_layer_package_download_bytes(downloaded),
            total_display,
            speed_suffix,
        ),
        3,
    );
    eprint!("\r\x1b[K   {gauge}");
    let _ = std::io::stderr().flush();
    if force {
        eprintln!();
    }
}

pub fn is_layer_package_ref(value: &str) -> bool {
    StagePackageRef::parse(value).is_ok_and(|package_ref| package_ref.is_distributable_package())
}

/// Resolve an `hf://` package ref to a local directory, downloading the manifest,
/// shared components (metadata, embeddings, output head), and assigned layer files
/// using the `hf_hub` Rust library.
///
/// Returns the local directory path containing the package files.
/// If `package_ref` is already a local package path, validates its manifest paths
/// and returns it.
/// Resolve a layer package from the local HF cache without touching the HF SDK.
/// Verifies that needed files exist locally; returns the snapshot dir path.
fn resolve_local_package_files(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<String> {
    let manifest_path = package_dir.join("model-package.json");
    let manifest_contents = fs::read(&manifest_path).context("read local package manifest")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_contents).context("parse local package manifest")?;

    if manifest
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        == Some(u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION))
    {
        anyhow::ensure!(
            is_metadata_only_package_inspection(
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            ),
            "package-v2 artifact selection requires an exact stage admission descriptor"
        );
        verify_package_v2_metadata(package_dir, &manifest_contents)?;
        return Ok(package_dir.to_string_lossy().to_string());
    }

    // Verify shared/metadata.gguf exists
    let metadata_path = manifest
        .pointer("/shared/metadata/path")
        .and_then(|v| v.as_str())
        .context("manifest missing /shared/metadata/path")?;
    let metadata_path = safe_manifest_file_path(metadata_path)?;
    anyhow::ensure!(
        package_dir.join(&metadata_path).is_file(),
        "missing shared metadata: {}",
        metadata_path.display()
    );
    if include_embeddings
        && let Some(path) = manifest
            .pointer("/shared/embeddings/path")
            .and_then(|v| v.as_str())
    {
        let path = safe_manifest_file_path(path)?;
        anyhow::ensure!(
            package_dir.join(&path).is_file(),
            "missing shared embeddings: {}",
            path.display()
        );
    }
    if include_output
        && let Some(path) = manifest
            .pointer("/shared/output/path")
            .and_then(|v| v.as_str())
    {
        let path = safe_manifest_file_path(path)?;
        anyhow::ensure!(
            package_dir.join(&path).is_file(),
            "missing shared output: {}",
            path.display()
        );
    }
    // Verify needed layer files exist
    if let Some(layers) = manifest.get("layers").and_then(|l| l.as_array()) {
        for (i, layer) in layers.iter().enumerate() {
            let idx = layer
                .get("layer_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(i as u64) as u32;
            if idx >= layer_start
                && idx < layer_end
                && let Some(path) = layer.get("path").and_then(|a| a.as_str())
            {
                let path = safe_manifest_file_path(path)?;
                anyhow::ensure!(
                    package_dir.join(&path).is_file(),
                    "missing layer file: {}",
                    path.display()
                );
            }
        }
    }
    Ok(package_dir.to_string_lossy().to_string())
}

fn package_integrity_cache_dir() -> PathBuf {
    crate::models::mesh_llm_cache_dir().join("skippy-package-integrity")
}

fn is_metadata_only_package_inspection(
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> bool {
    layer_start == layer_end && !include_embeddings && !include_output
}

fn verify_resolved_hf_package_files(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<String> {
    let manifest_contents = fs::read(package_dir.join("model-package.json"))
        .context("read resolved package manifest")?;
    let schema_version = serde_json::from_slice::<serde_json::Value>(&manifest_contents)
        .context("parse resolved package manifest")?
        .get("schema_version")
        .and_then(serde_json::Value::as_u64);
    if schema_version == Some(u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION)) {
        anyhow::ensure!(
            is_metadata_only_package_inspection(
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            ),
            "package-v2 artifact selection requires an exact stage admission descriptor"
        );
        verify_package_v2_metadata(package_dir, &manifest_contents)?;
        return Ok(package_dir.to_string_lossy().to_string());
    }
    let local_ref = resolve_local_package_files(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    )?;
    let metadata_only = is_metadata_only_package_inspection(
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    );
    let options = if metadata_only {
        // Metadata-only probes hash only the small shared metadata artifact.
        // Avoid the cross-run integrity cache here so a same-size metadata
        // rewrite cannot be hidden by coarse filesystem timestamp resolution.
        PackageIntegrityOptions::verify_without_cache()
    } else {
        PackageIntegrityOptions::verify_with_cache(package_integrity_cache_dir())
    };
    let report = if metadata_only {
        package::verify_layer_package_metadata_integrity(&local_ref, &options)
    } else {
        let request = PackageStageRequest {
            model_id: "hf-layer-package".to_string(),
            topology_id: "hf-layer-package-resolver".to_string(),
            package_ref: local_ref.clone(),
            stage_id: format!("layers-{layer_start}-{layer_end}"),
            layer_start,
            layer_end,
            include_embeddings,
            include_output,
        };
        package::verify_layer_package_integrity(&request, &options)
    }
    .map_err(|error| anyhow::anyhow!("verify resolved HF layer package artifacts: {error:#}"))?;
    tracing::debug!(
        artifacts = report.artifacts,
        verified_artifacts = report.verified_artifacts,
        cached_artifacts = report.cached_artifacts,
        manifest_sha256 = %report.manifest_sha256,
        metadata_only,
        "verified resolved HF layer package artifacts"
    );
    Ok(local_ref)
}

fn missing_cached_package_artifact(error: &anyhow::Error) -> bool {
    let message = error.to_string();
    message.starts_with("missing shared metadata:")
        || message.starts_with("missing shared embeddings:")
        || message.starts_with("missing shared output:")
        || message.starts_with("missing layer file:")
}

fn verify_cached_hf_package_files(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<Option<String>> {
    match verify_resolved_hf_package_files(
        package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    ) {
        Ok(local_ref) => Ok(Some(local_ref)),
        Err(error) if missing_cached_package_artifact(&error) => {
            tracing::debug!(
                package_dir = %package_dir.display(),
                error = %error,
                "cached HF layer package snapshot is incomplete; downloading missing artifacts"
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn manifest_artifact_bytes(artifact: &serde_json::Value) -> Option<u64> {
    artifact
        .get("artifact_bytes")
        .and_then(|value| value.as_u64())
}

fn layer_package_progress_label(repo: &str, revision: &str) -> String {
    if revision == "main" {
        format!("layer package {repo}")
    } else {
        format!("layer package {repo}@{revision}")
    }
}

fn download_layer_package_file(
    model_api: &hf_hub::HFRepositorySync<hf_hub::RepoTypeModel>,
    revision: &str,
    label: &str,
    file_name: &str,
    total_bytes: Option<u64>,
    package_scope: Option<Arc<LayerPackageDownloadScope>>,
    completed_before: usize,
) -> Result<PathBuf> {
    let progress = Arc::new(LayerPackageDownloadProgress::new(
        label.to_string(),
        file_name.to_string(),
        total_bytes,
        package_scope,
        completed_before,
    ));
    progress.emit_ensuring();
    let progress_handler: Option<Progress> = Some(progress.clone().into());
    let path = model_api
        .download_file()
        .filename(file_name.to_string())
        .revision(revision.to_string())
        .maybe_progress(progress_handler)
        .send()
        .with_context(|| format!("download layer package file: {file_name}"))?;
    progress.emit_ready(&path);
    Ok(path)
}

pub fn resolve_hf_package_to_local(
    package_ref: &str,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<String> {
    let parsed = StagePackageRef::parse(package_ref)?;
    let (repo, revision) = match &parsed {
        StagePackageRef::HuggingFacePackage { repo, revision } => (
            repo.clone(),
            revision.clone().unwrap_or_else(|| "main".to_string()),
        ),
        StagePackageRef::LocalPackage(path) => {
            return resolve_local_package_files(
                path,
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            );
        }
        _ => return Ok(package_ref.to_string()),
    };

    // Try to resolve from the local HF cache first — avoids the HF SDK entirely,
    // which is critical on NFS (where flock fails) and inside async runtimes
    // (where the sync SDK wrapper panics with "Cannot start a runtime").
    let cache_dir = crate::models::huggingface_hub_cache_dir();
    let repo_folder = format!("models--{}", repo.replace('/', "--"));
    let revision_cache_path = safe_manifest_file_path(&revision)
        .with_context(|| format!("invalid HF revision for local cache lookup: {revision}"))?;
    let ref_path = cache_dir
        .join(&repo_folder)
        .join("refs")
        .join(&revision_cache_path);
    let direct_snapshot_dir = cache_dir
        .join(&repo_folder)
        .join("snapshots")
        .join(&revision_cache_path);
    if direct_snapshot_dir.join("model-package.json").is_file()
        && let Some(local_ref) = cache_resolution::resolve_cached_hf_package_snapshot(
            &direct_snapshot_dir,
            layer_start,
            layer_end,
            include_embeddings,
            include_output,
        )?
    {
        return Ok(local_ref);
    }
    if let Ok(commit_hash) = fs::read_to_string(&ref_path) {
        let commit_hash = commit_hash.trim();
        let commit_hash_path = safe_manifest_file_path(commit_hash).with_context(|| {
            format!("invalid HF cache commit hash for local cache lookup: {commit_hash}")
        })?;
        let snapshot_dir = cache_dir
            .join(&repo_folder)
            .join("snapshots")
            .join(commit_hash_path);
        if snapshot_dir.join("model-package.json").is_file()
            && let Some(local_ref) = cache_resolution::resolve_cached_hf_package_snapshot(
                &snapshot_dir,
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            )?
        {
            return Ok(local_ref);
        }
    }
    let downloaded = crate::models::run_hf_sync(move || {
        download_hf_package_to_local_sync(
            &repo,
            &revision,
            layer_start,
            layer_end,
            include_embeddings,
            include_output,
        )
    })?;

    // Metadata-only probes (layer_start == layer_end == 0) download the
    // manifest and shared metadata but no layer files.  The downloaded
    // snapshot may be a skeleton whose hash must not propagate through
    // topology configs and stage loads.  Re-scan the local cache for a
    // snapshot that has at least one real layer artifact.
    //
    // Real stage loads (layer_start < layer_end) always download the
    // requested layer range, so the downloaded snapshot is guaranteed to
    // have the needed files — no fallback scan needed.
    let is_metadata_only = layer_start == 0 && layer_end == 0;
    if is_metadata_only {
        let downloaded_dir = std::path::Path::new(&downloaded);
        if downloaded_dir.join("model-package.json").is_file()
            && cache_resolution::resolve_cached_hf_package_snapshot(
                downloaded_dir,
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            )?
            .is_none()
        {
            // Downloaded snapshot is a skeleton — find one with real layers.
            let cache_dir = crate::models::huggingface_hub_cache_dir();
            for snapshot_dir in
                cache_resolution::cached_package_snapshots(&cache_dir, &repo_folder)?
            {
                if snapshot_dir.as_path() == downloaded_dir {
                    continue;
                }
                if let Ok(Some(better)) = cache_resolution::resolve_cached_hf_package_snapshot(
                    &snapshot_dir,
                    layer_start,
                    layer_end,
                    include_embeddings,
                    include_output,
                ) {
                    tracing::debug!(
                        downloaded = %downloaded,
                        better = %better,
                        "post-download: preferring cached snapshot with layer artifacts over skeleton"
                    );
                    return Ok(better);
                }
            }
        }
    }

    Ok(downloaded)
}

fn download_hf_package_to_local_sync(
    repo: &str,
    revision: &str,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<String> {
    let api = crate::models::build_hf_api(false)?;
    let (owner, name) = repo.split_once('/').context("invalid HF repo format")?;
    let model_api = api.model(owner, name);
    let progress_label = layer_package_progress_label(repo, revision);

    // Download manifest first
    let manifest_path = download_layer_package_file(
        &model_api,
        revision,
        &progress_label,
        "model-package.json",
        None,
        None,
        0,
    )
    .context("download layer package manifest")?;

    let package_dir = manifest_path
        .parent()
        .context("manifest has no parent directory")?
        .to_path_buf();

    // Read manifest to determine which files we need
    let manifest_contents = fs::read(&manifest_path).context("read package manifest")?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_contents).context("parse package manifest")?;

    let package_v2 = manifest
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        == Some(u64::from(skippy_package_format::PACKAGE_SCHEMA_VERSION));
    if package_v2 {
        anyhow::ensure!(
            is_metadata_only_package_inspection(
                layer_start,
                layer_end,
                include_embeddings,
                include_output,
            ),
            "package-v2 artifact selection requires an exact stage admission descriptor"
        );
    }

    // Collect the files we need to download
    let mut needed_files: Vec<(PathBuf, Option<u64>)> = Vec::new();

    if package_v2 {
        let manifest_v2: PackageManifestV2 =
            serde_json::from_slice(&manifest_contents).context("parse package-v2 manifest")?;
        manifest_v2
            .validate()
            .context("validate package-v2 manifest")?;
        let metadata_artifact = manifest_v2
            .artifact_catalog
            .entries
            .iter()
            .find(|artifact| artifact.id == manifest_v2.source_model.metadata_artifact_id)
            .context("package-v2 metadata artifact is absent")?;
        needed_files.push((
            safe_manifest_file_path(&metadata_artifact.path)?,
            Some(metadata_artifact.byte_size),
        ));
    } else {
        // Legacy packages always need shared/metadata.gguf.
        let metadata_artifact = manifest
            .pointer("/shared/metadata")
            .context("manifest missing required /shared/metadata")?;
        let metadata_path = metadata_artifact
            .get("path")
            .and_then(|v| v.as_str())
            .context("manifest missing required /shared/metadata/path")?;
        needed_files.push((
            safe_manifest_file_path(metadata_path)?,
            manifest_artifact_bytes(metadata_artifact),
        ));
    }
    if !package_v2
        && include_embeddings
        && let Some(artifact) = manifest.pointer("/shared/embeddings")
        && let Some(path) = artifact.get("path").and_then(|v| v.as_str())
    {
        needed_files.push((
            safe_manifest_file_path(path)?,
            manifest_artifact_bytes(artifact),
        ));
    }
    if !package_v2
        && include_output
        && let Some(artifact) = manifest.pointer("/shared/output")
        && let Some(path) = artifact.get("path").and_then(|v| v.as_str())
    {
        needed_files.push((
            safe_manifest_file_path(path)?,
            manifest_artifact_bytes(artifact),
        ));
    }

    // Layer files for assigned range — use explicit layer_index if present,
    // fall back to array position.
    if !package_v2 && let Some(layers) = manifest.get("layers").and_then(|l| l.as_array()) {
        for (i, layer) in layers.iter().enumerate() {
            let idx = layer
                .get("layer_index")
                .and_then(|v| v.as_u64())
                .unwrap_or(i as u64) as u32;
            if idx >= layer_start
                && idx < layer_end
                && let Some(path) = layer.get("path").and_then(|a| a.as_str())
            {
                needed_files.push((
                    safe_manifest_file_path(path)?,
                    manifest_artifact_bytes(layer),
                ));
            }
        }
    }
    if !package_v2
        && layer_start == 0
        && let Some(projectors) = manifest.get("projectors").and_then(|p| p.as_array())
    {
        for projector in projectors {
            if let Some(path) = projector.get("path").and_then(|value| value.as_str()) {
                needed_files.push((
                    safe_manifest_file_path(path)?,
                    manifest_artifact_bytes(projector),
                ));
            }
        }
    }

    let missing_files: Vec<_> = needed_files
        .iter()
        .filter(|(file, _)| !package_dir.join(file).is_file())
        .collect();
    let package_scope = Arc::new(LayerPackageDownloadScope::new(
        &progress_label,
        missing_files.len() + 1,
    ));

    // Download each needed file
    for (index, (file, total_bytes)) in missing_files.into_iter().enumerate() {
        let file_name = file.to_string_lossy().to_string();
        download_layer_package_file(
            &model_api,
            revision,
            &progress_label,
            &file_name,
            *total_bytes,
            Some(Arc::clone(&package_scope)),
            index + 1,
        )?;
    }

    verify_resolved_hf_package_files(
        &package_dir,
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    )
}

fn verify_package_v2_metadata(package_dir: &Path, manifest_bytes: &[u8]) -> Result<()> {
    let manifest: PackageManifestV2 =
        serde_json::from_slice(manifest_bytes).context("parse package-v2 manifest")?;
    manifest
        .validate()
        .context("validate package-v2 manifest")?;
    let computed = manifest
        .computed_package_id()
        .context("compute package-v2 identity")?;
    anyhow::ensure!(
        manifest.package_id == computed,
        "package-v2 manifest package_id does not match its content"
    );
    let artifact = manifest
        .artifact_catalog
        .entries
        .iter()
        .find(|artifact| artifact.id == manifest.source_model.metadata_artifact_id)
        .context("package-v2 metadata artifact is absent")?;
    verify_package_v2_artifact(package_dir, artifact)
}

fn verify_package_v2_artifact(package_dir: &Path, artifact: &PackageV2Artifact) -> Result<()> {
    let relative = safe_manifest_file_path(&artifact.path)?;
    let path = package_dir.join(&relative);
    let metadata = fs::metadata(&path)
        .with_context(|| format!("stat package-v2 artifact {}", relative.display()))?;
    anyhow::ensure!(
        metadata.is_file(),
        "package-v2 artifact is not a file: {}",
        relative.display()
    );
    anyhow::ensure!(
        metadata.len() == artifact.byte_size,
        "package-v2 artifact {} size differs from manifest",
        relative.display()
    );
    let mut file = fs::File::open(&path)
        .with_context(|| format!("open package-v2 artifact {}", relative.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("hash package-v2 artifact {}", relative.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = hex::encode(hasher.finalize());
    anyhow::ensure!(
        actual.eq_ignore_ascii_case(&artifact.sha256),
        "package-v2 artifact {} SHA-256 differs from manifest",
        relative.display()
    );
    Ok(())
}

pub fn resolve_package_v2_stage_to_local(
    package_ref: &str,
    admission: &PackageV2StageAdmissionDescriptor,
) -> Result<(String, Vec<PathBuf>, Option<PathBuf>)> {
    let local_ref = resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    let package_dir = PathBuf::from(&local_ref);
    let manifest_path = package_dir.join("model-package.json");
    let manifest_bytes = fs::read(&manifest_path)
        .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?;
    let manifest: PackageManifestV2 =
        serde_json::from_slice(&manifest_bytes).context("parse package-v2 manifest")?;
    let resolved = manifest
        .resolve_stage_admission(admission)
        .context("resolve exact package-v2 stage admission")?;
    let required = resolved
        .required_artifacts
        .iter()
        .map(|artifact| (*artifact).clone())
        .collect::<Vec<_>>();

    if let StagePackageRef::HuggingFacePackage { repo, revision } =
        StagePackageRef::parse(package_ref)?
    {
        let revision = revision.unwrap_or_else(|| "main".to_string());
        let missing = required
            .iter()
            .filter(|artifact| verify_package_v2_artifact(&package_dir, artifact).is_err())
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            let package_dir_for_download = package_dir.clone();
            let required_for_download = required.clone();
            crate::models::run_hf_sync(move || {
                let api = crate::models::build_hf_api(false)?;
                let (owner, name) = repo.split_once('/').context("invalid HF repo format")?;
                let model_api = api.model(owner, name);
                let label = layer_package_progress_label(&repo, &revision);
                let scope = Arc::new(LayerPackageDownloadScope::new(&label, missing.len()));
                for (index, artifact) in missing.iter().enumerate() {
                    download_layer_package_file(
                        &model_api,
                        &revision,
                        &label,
                        &artifact.path,
                        Some(artifact.byte_size),
                        Some(Arc::clone(&scope)),
                        index,
                    )?;
                }
                for artifact in &required_for_download {
                    verify_package_v2_artifact(&package_dir_for_download, artifact)?;
                }
                Ok(())
            })?;
        }
    }
    for artifact in &required {
        verify_package_v2_artifact(&package_dir, artifact)?;
    }

    let sidecar_ids = resolved
        .sidecars
        .iter()
        .map(|sidecar| sidecar.artifact.id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut model_artifacts = required
        .iter()
        .filter(|artifact| !sidecar_ids.contains(artifact.id.as_str()))
        .collect::<Vec<_>>();
    model_artifacts.sort_by(|left, right| {
        let left_primary = left.id == manifest.source_model.metadata_artifact_id;
        let right_primary = right.id == manifest.source_model.metadata_artifact_id;
        right_primary
            .cmp(&left_primary)
            .then_with(|| left.id.cmp(&right.id))
    });
    let model_parts = model_artifacts
        .into_iter()
        .map(|artifact| package_dir.join(&artifact.path))
        .collect::<Vec<_>>();
    let projector = resolved
        .sidecars
        .iter()
        .find(|sidecar| sidecar.kind == skippy_package_format::SidecarKind::Mmproj)
        .map(|sidecar| package_dir.join(&sidecar.artifact.path));
    Ok((local_ref, model_parts, projector))
}

fn safe_manifest_file_path(path: &str) -> Result<PathBuf> {
    anyhow::ensure!(!path.is_empty(), "manifest file path is empty");
    let path = Path::new(path);
    let mut components = path.components();
    let Some(first) = components.next() else {
        bail!("manifest file path is empty");
    };
    anyhow::ensure!(
        matches!(first, Component::Normal(_))
            && components.all(|component| matches!(component, Component::Normal(_))),
        "manifest file path must be a safe relative path: {}",
        path.display()
    );
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    use serial_test::serial;
    use sha2::{Digest, Sha256};

    fn restore_env(key: &str, previous: Option<OsString>) {
        if let Some(value) = previous {
            // SAFETY: the enclosing test contract is `#[serial]`, so this process
            // environment mutation cannot race another test.
            unsafe { std::env::set_var(key, value) };
        } else {
            // SAFETY: the enclosing test contract is `#[serial]`, so this process
            // environment mutation cannot race another test.
            unsafe { std::env::remove_var(key) };
        }
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    struct EnvRestore {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvRestore {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                previous: std::env::var_os(key),
            }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            restore_env(self.key, self.previous.take());
        }
    }

    fn write_cached_package_snapshot(snapshot: &Path, layer_sha: String) {
        fs::create_dir_all(snapshot.join("shared")).unwrap();
        fs::create_dir_all(snapshot.join("layers")).unwrap();
        fs::write(snapshot.join("shared/metadata.gguf"), b"metadata").unwrap();
        fs::write(snapshot.join("layers/layer-000.gguf"), b"layer").unwrap();
        fs::write(
            snapshot.join("model-package.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version": 1,
                "model_id": "model-a",
                "source_model": {
                    "path": "model-a.gguf",
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "files": [
                        {
                            "path": "model-a.gguf",
                            "size_bytes": 123,
                            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        }
                    ]
                },
                "format": "layer-package",
                "layer_count": 1,
                "activation_width": 4096,
                "shared": {
                    "metadata": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    },
                    "embeddings": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    },
                    "output": {
                        "path": "shared/metadata.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 8,
                        "sha256": sha256_hex(b"metadata")
                    }
                },
                "layers": [
                    {
                        "layer_index": 0,
                        "path": "layers/layer-000.gguf",
                        "tensor_count": 1,
                        "tensor_bytes": 1,
                        "artifact_bytes": 5,
                        "sha256": layer_sha
                    }
                ],
                "skippy_abi_version": "0.1.0",
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn layer_package_ref_detects_local_manifest_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-package.json"), "{}").unwrap();

        assert!(is_layer_package_ref(&dir.path().to_string_lossy()));
        assert!(!is_layer_package_ref("/tmp/not-a-package"));
        assert!(is_layer_package_ref("hf://Mesh-LLM/demo-package"));
    }

    #[test]
    fn package_ref_distinguishes_direct_gguf_from_distributable_packages() {
        let direct = StagePackageRef::parse("/models/model.gguf").unwrap();
        assert_eq!(
            direct,
            StagePackageRef::SyntheticDirectGguf(PathBuf::from("/models/model.gguf"))
        );
        assert!(!direct.is_distributable_package());
        assert!(direct.as_package_ref().is_none());

        let hf = StagePackageRef::parse("hf://Mesh-LLM/demo-package@abc123").unwrap();
        assert!(hf.is_distributable_package());
        assert_eq!(
            hf.as_package_ref().as_deref(),
            Some("hf://Mesh-LLM/demo-package@abc123")
        );
    }

    #[test]
    fn layer_package_artifact_display_names_repo_and_file_without_revision() {
        assert_eq!(
            layer_package_artifact_display(
                "layer package meshllm/demo-package@abc123",
                "layers/layer-005.gguf"
            ),
            "meshllm/demo-package/layers/layer-005.gguf"
        );
        assert_eq!(
            layer_package_artifact_display(
                "layer package meshllm/demo-package",
                "model-package.json"
            ),
            "meshllm/demo-package/model-package.json"
        );
    }

    #[test]
    fn safe_manifest_file_path_rejects_escaping_paths() {
        assert_eq!(
            safe_manifest_file_path("shared/metadata.gguf").unwrap(),
            PathBuf::from("shared/metadata.gguf")
        );

        for path in [
            "",
            "/tmp/metadata.gguf",
            "../metadata.gguf",
            "shared/../metadata.gguf",
        ] {
            let error = safe_manifest_file_path(path).unwrap_err().to_string();
            assert!(
                error.contains("manifest file path"),
                "unexpected error for {path:?}: {error}"
            );
        }
    }

    #[test]
    #[serial]
    fn env_restore_preserves_previous_value_after_unwind() {
        const KEY: &str = "MESH_LLM_TEST_ENV_RESTORE_PANIC";
        let _restore_outer = EnvRestore::capture(KEY);
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var(KEY, "before") };

        let panic_result = std::panic::catch_unwind(|| {
            let _restore_inner = EnvRestore::capture(KEY);
            // SAFETY: the enclosing test contract is `#[serial]`, so this process
            // environment mutation cannot race another test.
            unsafe { std::env::set_var(KEY, "during") };
            panic!("force unwind after env mutation");
        });

        assert!(panic_result.is_err());
        assert_eq!(std::env::var(KEY).unwrap(), "before");
    }

    #[test]
    fn local_package_resolution_rejects_manifest_traversal() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::json!({
                "shared": {
                    "metadata": { "path": "../metadata.gguf" }
                },
                "layers": []
            })
            .to_string(),
        )
        .unwrap();

        let error = resolve_local_package_files(dir.path(), 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("safe relative path"), "{error}");
    }

    #[test]
    fn local_package_ref_resolution_rejects_manifest_traversal() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::json!({
                "shared": {
                    "metadata": { "path": "../metadata.gguf" }
                },
                "layers": []
            })
            .to_string(),
        )
        .unwrap();

        let error = resolve_hf_package_to_local(&dir.path().to_string_lossy(), 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("safe relative path"), "{error}");
    }

    #[test]
    fn cached_hf_package_verification_treats_missing_artifacts_as_incomplete_cache() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("shared")).unwrap();
        fs::write(dir.path().join("shared/metadata.gguf"), b"metadata").unwrap();
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::json!({
                "shared": {
                    "metadata": { "path": "shared/metadata.gguf" },
                    "embeddings": { "path": "shared/embeddings.gguf" },
                    "output": { "path": "shared/output.gguf" }
                },
                "layers": []
            })
            .to_string(),
        )
        .unwrap();

        let resolved = verify_cached_hf_package_files(dir.path(), 0, 0, true, false).unwrap();

        assert_eq!(resolved, None);
    }

    #[test]
    #[serial]
    fn hf_package_resolution_rejects_revision_cache_traversal() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path()) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let error = resolve_hf_package_to_local("hf://owner/repo@../../evil", 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("invalid HF revision") || error.contains("safe relative path"),
            "{error}"
        );
    }

    #[test]
    #[serial]
    fn hf_package_resolution_rejects_ref_target_cache_traversal() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path()) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let refs_dir = temp
            .path()
            .join("hub")
            .join("models--owner--repo")
            .join("refs");
        fs::create_dir_all(&refs_dir).unwrap();
        fs::write(refs_dir.join("main"), "../../evil").unwrap();

        let error = resolve_hf_package_to_local("hf://owner/repo", 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("invalid HF cache commit hash") || error.contains("safe relative path"),
            "{error}"
        );
    }

    #[test]
    #[serial]
    fn hf_package_resolution_uses_direct_snapshot_revision_cache() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path()) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let snapshot = temp
            .path()
            .join("hub")
            .join("models--owner--repo")
            .join("snapshots")
            .join("abc123");
        write_cached_package_snapshot(&snapshot, sha256_hex(b"layer"));

        let resolved =
            resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 1, false, false).unwrap();

        assert_eq!(PathBuf::from(resolved), snapshot);
    }

    #[test]
    #[serial]
    /// With an explicit pinned revision that has all requested layers, the
    /// cache lookup returns it directly without downloading or scanning other
    /// snapshots.  A stale snapshot with different content must NOT be picked.
    fn pinned_revision_resolves_directly_from_cache() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");
        let _xdg_cache = EnvRestore::capture("XDG_CACHE_HOME");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path().join("hf")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("XDG_CACHE_HOME", temp.path().join("mesh-cache")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let repo_cache = temp
            .path()
            .join("hf")
            .join("hub")
            .join("models--owner--repo");

        // Create a complete snapshot at the pinned revision.
        let pinned_snapshot = repo_cache.join("snapshots").join("abc123");
        write_cached_package_snapshot(&pinned_snapshot, sha256_hex(b"layer"));

        // Create a stale snapshot that also has layers — must NOT be used.
        let stale_snapshot = repo_cache.join("snapshots").join("old-stale");
        write_cached_package_snapshot(&stale_snapshot, sha256_hex(b"layer"));

        let resolved =
            resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 0, false, false).unwrap();

        assert_eq!(PathBuf::from(resolved), pinned_snapshot);
    }

    #[test]
    #[serial]
    fn hf_package_metadata_only_cache_resolution_uses_metadata_integrity_scope() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");
        let _xdg_cache = EnvRestore::capture("XDG_CACHE_HOME");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path().join("hf")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("XDG_CACHE_HOME", temp.path().join("mesh-cache")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let snapshot = temp
            .path()
            .join("hf")
            .join("hub")
            .join("models--owner--repo")
            .join("snapshots")
            .join("abc123");
        write_cached_package_snapshot(
            &snapshot,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        );

        let resolved =
            resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 0, false, false).unwrap();
        assert_eq!(PathBuf::from(resolved), snapshot);

        let info = super::super::inspect_stage_package("hf://owner/repo@abc123").unwrap();
        assert_eq!(info.model_id, "model-a");
        assert_eq!(info.layer_count, 1);

        fs::write(snapshot.join("shared/metadata.gguf"), b"metadota").unwrap();
        let error = resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 0, false, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("checksum mismatch"), "{error}");
        assert!(error.contains("shared/metadata.gguf"), "{error}");
    }

    #[test]
    #[serial]
    fn hf_package_resolution_verifies_cached_snapshot_artifact_checksums() {
        let _hf_home = EnvRestore::capture("HF_HOME");
        let _hf_cache = EnvRestore::capture("HF_HUB_CACHE");
        let _huggingface_cache = EnvRestore::capture("HUGGINGFACE_HUB_CACHE");
        let _xdg_cache = EnvRestore::capture("XDG_CACHE_HOME");

        let temp = tempfile::tempdir().unwrap();
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("HF_HOME", temp.path().join("hf")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::set_var("XDG_CACHE_HOME", temp.path().join("mesh-cache")) };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HF_HUB_CACHE") };
        // SAFETY: the enclosing test contract is `#[serial]`, so this process
        // environment mutation cannot race another test.
        unsafe { std::env::remove_var("HUGGINGFACE_HUB_CACHE") };

        let snapshot = temp
            .path()
            .join("hf")
            .join("hub")
            .join("models--owner--repo")
            .join("snapshots")
            .join("abc123");
        write_cached_package_snapshot(
            &snapshot,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        );

        let error = resolve_hf_package_to_local("hf://owner/repo@abc123", 0, 1, false, false)
            .unwrap_err()
            .to_string();

        assert!(error.contains("checksum mismatch"), "{error}");
    }

    /// Integration test: resolves package metadata without downloading layer files from HF.
    /// Run with: cargo test -p mesh-llm resolve_hf_downloads_metadata_only -- --ignored
    #[test]
    #[ignore]
    fn resolve_hf_downloads_metadata_only() {
        let package_ref = "hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers";
        // Request 0 layers — should download manifest/shared metadata, but no layer files.
        let local_path = resolve_hf_package_to_local(package_ref, 0, 0, false, false).unwrap();
        let manifest = std::path::Path::new(&local_path).join("model-package.json");
        assert!(
            manifest.is_file(),
            "manifest should exist at {}",
            manifest.display()
        );

        // Verify manifest is valid JSON with expected fields
        let contents = std::fs::read_to_string(&manifest).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(parsed["schema_version"], 1);
        assert!(parsed["layers"].as_array().unwrap().len() > 50);

        // Verify the function didn't request any layer downloads
        // (we can't check the cache dir because previous test runs may have cached files)
    }

    /// Integration test: downloads manifest + a single layer file.
    /// Run with: cargo test -p mesh-llm resolve_hf_downloads_single_layer -- --ignored
    #[test]
    #[ignore]
    fn resolve_hf_downloads_single_layer() {
        let package_ref = "hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers";
        // Request just layer 0
        let local_path = resolve_hf_package_to_local(package_ref, 0, 1, false, false).unwrap();
        let manifest = std::path::Path::new(&local_path).join("model-package.json");
        assert!(manifest.is_file());

        // Read manifest to find layer 0's artifact path
        let contents = std::fs::read_to_string(&manifest).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        let layer0_artifact = parsed["layers"][0]["path"].as_str().unwrap();

        // Verify that specific layer file was downloaded
        let layer0_path = std::path::Path::new(&local_path).join(layer0_artifact);
        assert!(
            layer0_path.is_file(),
            "layer 0 should be downloaded at {}",
            layer0_path.display()
        );
        // Should be non-trivial size (layer files are typically > 1 MB)
        let size = std::fs::metadata(&layer0_path).unwrap().len();
        assert!(size > 1_000_000, "layer file should be > 1MB, got {size}");
    }
}
