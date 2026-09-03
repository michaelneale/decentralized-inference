use anyhow::{Error, Result};
use mesh_llm_native_runtime::{
    CachePrunePlan, CandidateRejection, HostRuntimeProfile, InstalledNativeRuntime,
};
use mesh_llm_runtime_install::{NativeRuntimeInstallOutcome, NativeRuntimeInstallStatus};
use serde::Serialize;
use serde_json::json;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize)]
pub(crate) struct AvailableRuntimeRow {
    pub(crate) id: String,
    pub(crate) mesh_version: Option<String>,
    pub(crate) skippy_abi: String,
    pub(crate) backend: String,
    pub(crate) os: String,
    pub(crate) arch: String,
    pub(crate) supported: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) rejection_reasons: Vec<CandidateRejection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) url: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct NativeRuntimeDoctorReport {
    pub(crate) healthy: bool,
    pub(crate) status: String,
    pub(crate) blockers: Vec<String>,
    pub(crate) recommendations: Vec<String>,
    pub(crate) running_mesh_version: String,
    pub(crate) selected_mesh_version: String,
    pub(crate) configured_skippy_abi: Option<String>,
    pub(crate) configured_selection: Option<String>,
    pub(crate) host: HostRuntimeProfile,
    pub(crate) cache_path: PathBuf,
    pub(crate) selected_runtime_id: Option<String>,
    pub(crate) selected_runtime_flavor: Option<String>,
    pub(crate) selected_runtime_path: Option<PathBuf>,
    pub(crate) installed_count: usize,
    pub(crate) selected_version_installed_count: usize,
}

pub(crate) trait RuntimeNativeFormatter {
    fn render_available(&self, rows: &[AvailableRuntimeRow]) -> Result<()>;
    fn render_installed(
        &self,
        installed: &[InstalledNativeRuntime],
        cache_root: &Path,
    ) -> Result<()>;
    fn render_install(&self, outcome: &NativeRuntimeInstallOutcome) -> Result<()>;
    fn render_install_error(&self, error: &Error) -> Result<()>;
    fn render_remove(
        &self,
        native_runtime_id: &str,
        mesh_version: &str,
        removed: bool,
    ) -> Result<()>;
    fn render_prune(&self, plan: &CachePrunePlan) -> Result<()>;
    fn render_doctor(&self, report: &NativeRuntimeDoctorReport) -> Result<()>;
}

pub(crate) struct HumanFormatter;
pub(crate) struct JsonFormatter;

pub(crate) fn runtime_native_formatter(json_output: bool) -> Box<dyn RuntimeNativeFormatter> {
    if json_output {
        Box::new(JsonFormatter)
    } else {
        Box::new(HumanFormatter)
    }
}

impl RuntimeNativeFormatter for HumanFormatter {
    fn render_available(&self, rows: &[AvailableRuntimeRow]) -> Result<()> {
        print_available_human(rows);
        Ok(())
    }

    fn render_installed(
        &self,
        installed: &[InstalledNativeRuntime],
        cache_root: &Path,
    ) -> Result<()> {
        print_installed_human(installed, cache_root);
        Ok(())
    }

    fn render_install(&self, outcome: &NativeRuntimeInstallOutcome) -> Result<()> {
        print_install_human(outcome);
        Ok(())
    }

    fn render_install_error(&self, error: &Error) -> Result<()> {
        eprintln!("❌ Native runtime install failed");
        // Resolution failures carry a multi-line explanation (catalogs
        // consulted, rejected candidates); keep every line aligned.
        let mut lines = error.to_string();
        lines.truncate(lines.trim_end().len());
        let mut lines = lines.lines();
        if let Some(first) = lines.next() {
            eprintln!("   Reason: {first}");
        }
        for line in lines {
            eprintln!("           {line}");
        }
        eprintln!("   Try: mesh-llm runtime list --available");
        Ok(())
    }

    fn render_remove(
        &self,
        native_runtime_id: &str,
        mesh_version: &str,
        removed: bool,
    ) -> Result<()> {
        if removed {
            eprintln!("✅ Removed native runtime {native_runtime_id} for MeshLLM {mesh_version}");
        } else {
            eprintln!(
                "🔎 Native runtime {native_runtime_id} for MeshLLM {mesh_version} was not installed"
            );
        }
        Ok(())
    }

    fn render_prune(&self, plan: &CachePrunePlan) -> Result<()> {
        if plan.remove_dirs.is_empty() {
            eprintln!("✅ Native runtime cache already pruned");
        } else {
            eprintln!(
                "✅ Pruned {} native runtime cache version(s)",
                plan.remove_dirs.len()
            );
            for dir in &plan.remove_dirs {
                eprintln!("   removed: {}", dir.display());
            }
        }
        Ok(())
    }

    fn render_doctor(&self, report: &NativeRuntimeDoctorReport) -> Result<()> {
        print_doctor_human(report);
        Ok(())
    }
}

impl RuntimeNativeFormatter for JsonFormatter {
    fn render_available(&self, rows: &[AvailableRuntimeRow]) -> Result<()> {
        print_json(rows)
    }

    fn render_installed(
        &self,
        installed: &[InstalledNativeRuntime],
        _cache_root: &Path,
    ) -> Result<()> {
        print_json(installed)
    }

    fn render_install(&self, outcome: &NativeRuntimeInstallOutcome) -> Result<()> {
        print_json(&json!({
            "status": install_status_label(outcome.status.clone()),
            "runtime": outcome.runtime,
            "resolution": outcome.resolution,
            "catalogs": outcome.sources,
        }))
    }

    fn render_install_error(&self, error: &Error) -> Result<()> {
        print_json(&json!({
            "status": "error",
            "error": {
                "type": "native_runtime_install_failed",
                "message": error.to_string(),
                "context": error.chain().skip(1).map(ToString::to_string).collect::<Vec<_>>(),
            },
        }))
    }

    fn render_remove(
        &self,
        native_runtime_id: &str,
        mesh_version: &str,
        removed: bool,
    ) -> Result<()> {
        print_json(&json!({
            "mesh_version": mesh_version,
            "native_runtime_id": native_runtime_id,
            "removed": removed,
        }))
    }

    fn render_prune(&self, plan: &CachePrunePlan) -> Result<()> {
        print_json(plan)
    }

    fn render_doctor(&self, report: &NativeRuntimeDoctorReport) -> Result<()> {
        print_json(report)
    }
}

fn print_json(value: &(impl Serialize + ?Sized)) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn install_status_label(status: NativeRuntimeInstallStatus) -> &'static str {
    match status {
        NativeRuntimeInstallStatus::AlreadyInstalled => "already_installed",
        NativeRuntimeInstallStatus::Installed => "installed",
    }
}

fn print_available_human(rows: &[AvailableRuntimeRow]) {
    if rows.is_empty() {
        println!("📦 No native runtime manifest entries found");
        println!("   Pass --manifest or --bundle-dir to inspect available runtimes.");
        return;
    }
    println!("📦 Available native runtimes");
    for row in rows {
        let marker = if row.supported { "✅" } else { "⚠️" };
        let status = if row.supported {
            "compatible"
        } else {
            "not compatible"
        };
        println!(
            "  - {marker} {} {status} ({}, {}/{})",
            row.id, row.backend, row.os, row.arch
        );
        if let Some(mesh_version) = row.mesh_version.as_deref() {
            println!(
                "    MeshLLM: {mesh_version}; Skippy ABI: {}",
                row.skippy_abi
            );
        } else {
            println!("    MeshLLM: unspecified; Skippy ABI: {}", row.skippy_abi);
        }
        for reason in &row.rejection_reasons {
            println!("    reason: {}", format_rejection(reason));
        }
    }
}

fn print_installed_human(installed: &[InstalledNativeRuntime], cache_root: &Path) {
    if installed.is_empty() {
        println!("📦 No local native runtimes found");
        println!("   cache: {}", cache_root.display());
        return;
    }
    println!("📦 Local native runtimes");
    println!("   cache: {}", cache_root.display());
    for runtime in installed {
        println!(
            "  - ✅ {} {} ({})",
            runtime.native_runtime_id, runtime.mesh_version, runtime.flavor
        );
        println!("    path: {}", runtime.path.display());
    }
}

fn print_install_human(outcome: &NativeRuntimeInstallOutcome) {
    match outcome.status {
        NativeRuntimeInstallStatus::AlreadyInstalled => {
            eprintln!(
                "✅ Native runtime already installed: {}",
                outcome.runtime.native_runtime_id
            );
            eprintln!("   version: {}", outcome.runtime.mesh_version);
            eprintln!("   flavor: {}", outcome.runtime.flavor);
            eprintln!("   path: {}", outcome.runtime.path.display());
        }
        NativeRuntimeInstallStatus::Installed => {
            eprintln!("✅ Installed {}", outcome.runtime.native_runtime_id);
            eprintln!("   version: {}", outcome.runtime.mesh_version);
            eprintln!("   flavor: {}", outcome.runtime.flavor);
            eprintln!("   path: {}", outcome.runtime.path.display());
        }
    }
    for line in outcome.sources.describe() {
        eprintln!("   catalog: {line}");
    }
}

fn print_doctor_human(report: &NativeRuntimeDoctorReport) {
    println!("🩺 MeshLLM doctor");
    println!();
    println!("Native runtime:");
    println!("  status: {}", report.status);
    println!("  running MeshLLM version: {}", report.running_mesh_version);
    println!(
        "  selected runtime version: {}",
        report.selected_mesh_version
    );
    if report.selected_mesh_version != report.running_mesh_version {
        println!("  version pin: native runtime version is pinned by config");
    }
    if let Some(skippy_abi) = &report.configured_skippy_abi {
        println!("  configured Skippy ABI: {skippy_abi}");
    }
    if let Some(selection) = &report.configured_selection {
        println!("  configured selection: {selection}");
    }
    println!("  cache: {}", report.cache_path.display());
    println!("  host: {}/{}", report.host.os, report.host.arch);
    let flavors = report
        .host
        .available_flavors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    println!("  detected flavors: {flavors}");
    match &report.selected_runtime_id {
        Some(id) => {
            println!("  selected: {id}");
            if let Some(flavor) = &report.selected_runtime_flavor {
                println!("  flavor: {flavor}");
            }
            if let Some(path) = &report.selected_runtime_path {
                println!("  path: {}", path.display());
            }
        }
        None => {
            println!("  selected: none");
        }
    }
    println!("  installed: {}", report.installed_count);
    println!(
        "  installed for selected version: {}",
        report.selected_version_installed_count
    );
    if !report.blockers.is_empty() {
        println!();
        println!("Blockers:");
        for blocker in &report.blockers {
            println!("  - {blocker}");
        }
    }
    if !report.recommendations.is_empty() {
        println!();
        println!("Recommended next steps:");
        for recommendation in &report.recommendations {
            println!("  - {recommendation}");
        }
    }
}

fn format_rejection(reason: &CandidateRejection) -> String {
    // The wording lives on `CandidateRejection` itself so the install
    // diagnostics and this listing describe a rejection the same way.
    reason.to_string()
}
