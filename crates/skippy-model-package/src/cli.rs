use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "skippy-model-package")]
#[command(about = "Inspect, plan, write, and validate skippy model packages")]
pub(crate) struct Args {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    Inspect {
        model: PathBuf,
    },
    /// Offline schema-v1 planning tool; its output is not admitted for serving.
    Plan {
        model: PathBuf,
        #[arg(long)]
        stages: usize,
    },
    /// Offline schema-v1 slice writer; its output is not admitted for serving.
    Write {
        model: PathBuf,
        #[arg(long)]
        layers: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        stage_index: Option<u32>,
        #[arg(long)]
        include_embeddings: bool,
        #[arg(long)]
        include_output: bool,
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    /// Offline schema-v1 stage writer; its output is not admitted for serving.
    WriteStages {
        model: PathBuf,
        #[arg(long)]
        stages: usize,
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Emit the source-complete v2 package used by graph-admitted serving.
    WritePackage {
        model: String,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long = "projector")]
        projectors: Vec<PathBuf>,
        #[arg(long)]
        after_artifact_command: Option<PathBuf>,
        #[arg(long)]
        transform_artifact_command: Option<PathBuf>,
        #[arg(long)]
        model_id: Option<String>,
        #[arg(long)]
        source_repo: Option<String>,
        #[arg(long)]
        source_revision: Option<String>,
        #[arg(long)]
        source_file: Option<String>,
        #[arg(long)]
        resume_existing_artifacts: bool,
    },
    /// Verify byte-preserving v2 packages against independent local source files.
    VerifyPackageV2 {
        package: PathBuf,
        #[arg(long)]
        source: PathBuf,
        /// Logical source primary filename, if different from the local filename.
        #[arg(long)]
        source_file: Option<String>,
        /// Independent originals for all declared projector sidecars.
        #[arg(long = "source-projector")]
        source_projectors: Vec<PathBuf>,
    },
    Validate {
        full: PathBuf,
        slices: Vec<PathBuf>,
    },
    ValidatePackage {
        full: PathBuf,
        package: PathBuf,
    },
    Preflight {
        package: PathBuf,
        #[arg(long)]
        stages: Option<usize>,
        #[arg(long)]
        verify_sha256: bool,
    },
    ValidateGlmDsaContract {
        package: PathBuf,
        #[arg(long)]
        require_generation_policy: bool,
    },
    RepairGlmDsaGenerationPolicy {
        package: PathBuf,
        #[arg(long)]
        in_place: bool,
    },
}
