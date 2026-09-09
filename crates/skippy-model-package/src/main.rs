use anyhow::{Context, Result};
use clap::Parser;

mod cli;
mod generation_manifest;
mod glm_dsa_contract;
mod glm_dsa_generation_policy;
mod hash;
mod inspect;
mod package;
mod package_v2;
mod plan;
mod preflight;
mod progress;
mod source_inventory;
mod tensor_payload;
#[cfg(test)]
mod test_gguf;
#[cfg(test)]
mod tests;
mod validate;
mod verify_v2;
mod write;

use cli::{Args, Command};
use package::{ArtifactHook, ExplicitSourceIdentity};

fn prepare_model_download_directories() {
    let prepared = match model_hf::prepare_download_directories() {
        Ok(prepared) => prepared,
        Err(error) => {
            eprintln!(
                "⚠ Unable to prepare model download directories: {error:#}. \
                 Model downloads may fail; set MESH_LLM_DATA_DIR to a writable directory."
            );
            return;
        }
    };
    for fallback in &prepared.fallbacks {
        eprintln!("⚠ {fallback}");
    }
    // SAFETY: runs before any Tokio runtime, process is single-threaded.
    unsafe { prepared.apply_to_process_environment() };
}

// ponytail: main runs on a child thread because the Windows main thread has a
// 1 MB stack. sha256 over a multi-GB GGUF plus FFI slice writing blows that
// stack in debug builds. 8 MB matches the mesh-llm runtime default. If a real
// recursion sink appears, raise this or fix the recursion — don't go lower.
const MAIN_STACK_SIZE: usize = 8 * 1024 * 1024;

fn main() -> Result<()> {
    let args = Args::parse();
    if !matches!(args.command, Command::VerifyPackageV2 { .. }) {
        prepare_model_download_directories();
    }

    let handle = std::thread::Builder::new()
        .stack_size(MAIN_STACK_SIZE)
        .spawn(move || run(args))
        .context("spawn skippy-model-package worker thread")?;
    handle.join().unwrap_or_else(|panic| {
        std::panic::resume_unwind(panic);
    })
}

fn run(args: Args) -> Result<()> {
    match args.command {
        Command::Inspect { model } => inspect::inspect(model),
        Command::Plan { model, stages } => plan::build_plan(&model, stages).and_then(|output| {
            println!("{}", serde_json::to_string_pretty(&output)?);
            Ok(())
        }),
        Command::Write {
            model,
            layers,
            out,
            stage_index,
            include_embeddings,
            include_output,
            manifest,
        } => write::write_one(
            model,
            layers,
            out,
            stage_index,
            include_embeddings,
            include_output,
            manifest,
        ),
        Command::WriteStages {
            model,
            stages,
            out_dir,
        } => write::write_stages(model, stages, out_dir),
        Command::WritePackage {
            model,
            out_dir,
            projectors,
            after_artifact_command,
            transform_artifact_command,
            model_id,
            source_repo,
            source_revision,
            source_file,
            resume_existing_artifacts,
        } => package_v2::write_package(
            model,
            out_dir,
            projectors,
            ArtifactHook {
                command: after_artifact_command,
            },
            ArtifactHook {
                command: transform_artifact_command,
            },
            ExplicitSourceIdentity {
                model_id,
                source_repo,
                source_revision,
                source_file,
            },
            resume_existing_artifacts,
        ),
        Command::VerifyPackageV2 {
            package,
            source,
            source_file,
            source_projectors,
        } => {
            let report = verify_v2::verify_package(
                &package,
                &source,
                source_file.as_deref(),
                &source_projectors,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::Validate { full, slices } => validate::validate(full, slices),
        Command::ValidatePackage { full, package } => validate::validate_package(full, package),
        Command::Preflight {
            package,
            stages,
            verify_sha256,
        } => validate::run_preflight(package, stages, verify_sha256),
        Command::ValidateGlmDsaContract {
            package,
            require_generation_policy,
        } => {
            let report = glm_dsa_contract::validate_path_with_options(
                &package,
                glm_dsa_contract::GlmDsaContractOptions {
                    require_generation_policy,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            anyhow::ensure!(
                report.valid,
                "GLM-DSA contract validation failed for {}",
                package.display()
            );
            Ok(())
        }
        Command::RepairGlmDsaGenerationPolicy { package, in_place } => {
            glm_dsa_generation_policy::repair_package(&package, in_place)
        }
    }
}
