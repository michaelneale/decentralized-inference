use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

use anyhow::{Context, Result, anyhow, bail};
use model_ref::split_gguf_shard_info;
use serde::Deserialize;
use serde_json::Value;
use skippy_runtime::{
    ActivationFrame, GGML_TYPE_F16, IterationBatchPhase, IterationBatchRequest, MtpSource,
    RuntimeConfig, RuntimeKvPage, RuntimeLoadMode, StageModel, StageSession,
    package::{PackageStageRequest, inspect_layer_package, materialize_layer_package_details},
    redirect_native_logs_to_file,
};

use crate::FamilySpec;

const MANIFEST_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../docs/skippy/llama-parity-candidates.json"
));

const PROMPT: &str = "Hello from the Skippy llama parity harness.";
const CTX_SIZE: u32 = 128;
const CACHE_SEQ_ID: i32 = 17;

pub(crate) fn p0_p1_manifest_rows() -> BTreeSet<(String, String, String)> {
    let manifest = manifest();
    manifest
        .rows_by_priority(["p0", "p1"])
        .into_iter()
        .filter(|row| is_parity_artifact_status(&row.status))
        .map(|row| {
            (
                manifest.priority_for(row).to_string(),
                row.llama_model.clone(),
                row.family.clone(),
            )
        })
        .collect()
}

pub(crate) fn assert_manifest_row_complete(spec: FamilySpec) -> Result<()> {
    let row = manifest_row(spec)?;
    if !is_parity_artifact_status(&row.status) {
        bail!(
            "{} / {} is {}, but P0/P1 parity modules require a certified artifact row",
            row.llama_model,
            row.family,
            row.status
        );
    }
    if row.repo.trim().is_empty() {
        bail!("{} / {} is missing repo", row.llama_model, row.family);
    }
    if row.include()?.files().is_empty() {
        bail!(
            "{} / {} is missing include files",
            row.llama_model,
            row.family
        );
    }
    Ok(())
}

pub(crate) fn activation_handoff_matches_full_model(spec: FamilySpec) -> Result<()> {
    prepare_native_logs()?;
    let Some(case) = resolve_case_for_ignored_test(spec)? else {
        return Ok(());
    };
    let layout = case.layout()?;
    if case.row.is_package_only() {
        return run_lightweight_activation_smoke(&layout, spec);
    }
    let splits = split_layers_for(case.row, layout.layer_count)?;
    run_correctness_chain(&layout, spec, splits)
}

pub(crate) fn graph_boundary_contract_matches_stage_roles(spec: FamilySpec) -> Result<()> {
    prepare_native_logs()?;
    let case = ResolvedCase::resolve_boundary_contract(spec)?;
    let layout = case.layout()?;
    if case.row.is_package_only() {
        bail!(
            "{} / {} requires a full-model artifact for graph boundary role coverage",
            spec.llama_model,
            spec.family
        );
    }
    let (first_cut, second_cut) = split_layers_for(case.row, layout.layer_count)?;
    if first_cut >= second_cut {
        bail!("graph boundary role coverage requires two ordered split layers");
    }
    let n_gpu_layers = case_n_gpu_layers(case.row);
    let first_shape = StageShape {
        stage_index: 0,
        layer_start: 0,
        layer_end: first_cut,
        include_embeddings: true,
        include_output: false,
    };
    let middle_shape = StageShape {
        stage_index: 1,
        layer_start: first_cut,
        layer_end: second_cut,
        include_embeddings: false,
        include_output: false,
    };
    let final_shape = StageShape {
        stage_index: 2,
        layer_start: second_cut,
        layer_end: layout.layer_count,
        // Layer packages include token embeddings on the final stage for tied
        // output weights. A downstream consumer must still expose its input
        // boundary when those tensors happen to be present.
        include_embeddings: true,
        include_output: true,
    };
    let first = open_stage_model(
        &stage_path(&layout, spec, first_shape)?,
        first_shape,
        n_gpu_layers,
    )?;
    let middle = open_stage_model(
        &stage_path(&layout, spec, middle_shape)?,
        middle_shape,
        n_gpu_layers,
    )?;
    let final_stage = open_stage_model(
        &stage_path(&layout, spec, final_shape)?,
        final_shape,
        n_gpu_layers,
    )?;

    if first.input_activation_boundary().is_some() {
        bail!("first stage unexpectedly exposed an input activation boundary");
    }
    if final_stage.output_activation_boundary().is_some() {
        bail!("final stage unexpectedly exposed an output activation boundary");
    }
    let first_output =
        required_graph_boundary(first.output_activation_boundary(), "first stage output")?;
    let middle_input =
        required_graph_boundary(middle.input_activation_boundary(), "middle stage input")?;
    let middle_output =
        required_graph_boundary(middle.output_activation_boundary(), "middle stage output")?;
    let final_input =
        required_graph_boundary(final_stage.input_activation_boundary(), "final stage input")?;

    for (edge, boundary) in [
        ("first stage output", first_output),
        ("middle stage input", middle_input),
        ("middle stage output", middle_output),
        ("final stage input", final_input),
    ] {
        boundary.raw_f32_width(edge)?;
        if boundary.bytes_per_token
            != boundary.elements_per_token * std::mem::size_of::<f32>() as u64
        {
            bail!("{edge} did not report its exact native F32 bytes per token");
        }
    }
    if first_output != middle_input {
        bail!(
            "first-to-middle graph boundary contracts do not match: producer {first_output:?}, consumer {middle_input:?}"
        );
    }
    if middle_output != final_input {
        bail!(
            "middle-to-final graph boundary contracts do not match: producer {middle_output:?}, consumer {final_input:?}"
        );
    }

    let (expected_required_frame_flags, expected_required_sidebands) = match spec.family {
        "gemma3n" => (
            skippy_runtime::ACTIVATION_FLAG_GEMMA3N_ALTUP,
            skippy_runtime::ACTIVATION_SIDEBAND_TOKEN_IDS,
        ),
        "qwen4exp" => (0, skippy_runtime::ACTIVATION_SIDEBAND_TOKEN_IDS),
        _ => (0, 0),
    };
    for (edge, boundary) in [
        ("first stage output", first_output),
        ("middle stage input", middle_input),
        ("middle stage output", middle_output),
        ("final stage input", final_input),
    ] {
        if boundary.required_frame_flags != expected_required_frame_flags {
            bail!(
                "{edge} required frame flags {:#x}, expected {expected_required_frame_flags:#x}",
                boundary.required_frame_flags
            );
        }
        if boundary.required_sidebands != expected_required_sidebands {
            bail!(
                "{edge} required sidebands {:#x}, expected {expected_required_sidebands:#x}",
                boundary.required_sidebands
            );
        }
    }

    let tokens = first.tokenize(case_prompt(case.row), true)?;
    let mut first_session = first.create_session()?;
    let first_frame = first_session.prefill_chunk_frame(&tokens, None, 0)?;
    assert_frame_matches_graph_boundary(
        "first stage output",
        &first_frame,
        first_output,
        tokens.len(),
    )?;
    if first.output_activation_boundary() != Some(first_output) {
        bail!("first stage graph boundary changed after prefill execution");
    }
    let mut middle_session = middle.create_session()?;
    let middle_frame = middle_session.prefill_chunk_frame(&tokens, Some(&first_frame), 0)?;
    assert_frame_matches_graph_boundary(
        "middle stage output",
        &middle_frame,
        middle_output,
        tokens.len(),
    )?;
    let mut final_session = final_stage.create_session()?;
    let final_frame = final_session.prefill_chunk_frame(&tokens, Some(&middle_frame), 0)?;
    if !final_frame.payload.is_empty() || final_frame.desc.payload_bytes != 0 {
        bail!("final stage unexpectedly emitted an activation frame");
    }
    Ok(())
}

fn assert_frame_matches_graph_boundary(
    edge: &str,
    frame: &ActivationFrame,
    boundary: skippy_runtime::ActivationBoundaryDesc,
    expected_token_count: usize,
) -> Result<()> {
    if frame.desc.token_count as usize != expected_token_count {
        bail!(
            "{edge} emitted {} tokens, expected {expected_token_count}",
            frame.desc.token_count
        );
    }
    let expected_payload_bytes = boundary.payload_bytes(edge, frame.desc.token_count)?;
    if frame.desc.payload_bytes != expected_payload_bytes
        || frame.payload.len() as u64 != expected_payload_bytes
    {
        bail!(
            "{edge} emitted {} descriptor bytes and {} payload bytes, expected {expected_payload_bytes}",
            frame.desc.payload_bytes,
            frame.payload.len()
        );
    }
    if frame.desc.flags != boundary.required_frame_flags {
        bail!(
            "{edge} frame flags {:#x} do not match boundary flags {:#x}",
            frame.desc.flags,
            boundary.required_frame_flags
        );
    }
    Ok(())
}

fn required_graph_boundary(
    boundary: Option<skippy_runtime::ActivationBoundaryDesc>,
    edge: &str,
) -> Result<skippy_runtime::ActivationBoundaryDesc> {
    boundary.with_context(|| format!("{edge} did not expose a graph boundary descriptor"))
}

pub(crate) fn cache_state_restore_matches_recompute(spec: FamilySpec) -> Result<()> {
    prepare_native_logs()?;
    let Some(case) = resolve_case_for_ignored_test(spec)? else {
        return Ok(());
    };
    let layout = case.layout()?;
    let (stage_start, stage_end) = cache_stage_range(case.row, &layout)?;
    let state_kind = if case.row.is_recurrent() {
        StateKind::KvRecurrent
    } else {
        StateKind::ResidentKv
    };
    run_stage_state_restore(&layout, spec, stage_start, stage_end, state_kind)
}

pub(crate) fn mixed_iteration_matches_serial(spec: FamilySpec) -> Result<()> {
    prepare_native_logs()?;
    let Some(case) = resolve_case_for_ignored_test(spec)? else {
        return Ok(());
    };
    let layout = case.layout()?;
    if case.row.is_package_only() {
        eprintln!(
            "skipping mixed-iteration parity for package-only {} / {}",
            spec.llama_model, spec.family
        );
        return Ok(());
    }
    let (split_layer, _) = split_layers_for(case.row, layout.layer_count)?;
    run_mixed_iteration_split(&layout, spec, split_layer)
}

fn run_mixed_iteration_split(
    layout: &TestLayout,
    spec: FamilySpec,
    split_layer: u32,
) -> Result<()> {
    let n_gpu_layers = case_n_gpu_layers(manifest_row(spec)?);
    let stage0_shape = StageShape {
        stage_index: 0,
        layer_start: 0,
        layer_end: split_layer,
        include_embeddings: true,
        include_output: false,
    };
    let stage1_shape = StageShape {
        stage_index: 1,
        layer_start: split_layer,
        layer_end: layout.layer_count,
        include_embeddings: false,
        include_output: true,
    };
    let stage0 = open_stage_model(
        &stage_path(layout, spec, stage0_shape)?,
        stage0_shape,
        n_gpu_layers,
    )?;
    let stage1 = open_stage_model(
        &stage_path(layout, spec, stage1_shape)?,
        stage1_shape,
        n_gpu_layers,
    )?;
    let tokens = stage0.tokenize(case_prompt(manifest_row(spec)?), true)?;
    if tokens.len() < 3 {
        bail!(
            "{} / {} mixed-iteration prompt produced fewer than three tokens",
            spec.llama_model,
            spec.family
        );
    }
    let prefix = &tokens[..2];
    let long_prefill = &tokens[..tokens.len().min(6)];
    let short_prefill = &tokens[..3];

    let mut mixed_stage0 = [
        stage0.create_session()?,
        stage0.create_session()?,
        stage0.create_session()?,
    ];
    let mut mixed_stage1 = [
        stage1.create_session()?,
        stage1.create_session()?,
        stage1.create_session()?,
    ];
    let mut serial_stage0 = [
        stage0.create_session()?,
        stage0.create_session()?,
        stage0.create_session()?,
    ];
    let mut serial_stage1 = [
        stage1.create_session()?,
        stage1.create_session()?,
        stage1.create_session()?,
    ];

    let mixed_prefix_frame = mixed_stage0[0].prefill_chunk_frame(prefix, None, 0)?;
    let (mixed_decode_token, _) =
        mixed_stage1[0].prefill_chunk_frame_sampled(prefix, None, Some(&mixed_prefix_frame), 0)?;
    let serial_prefix_frame = serial_stage0[0].prefill_chunk_frame(prefix, None, 0)?;
    let (serial_decode_token, _) = serial_stage1[0].prefill_chunk_frame_sampled(
        prefix,
        None,
        Some(&serial_prefix_frame),
        0,
    )?;
    if mixed_decode_token != serial_decode_token {
        bail!(
            "{} / {} mixed-iteration setup token {mixed_decode_token} did not match serial {serial_decode_token}",
            spec.llama_model,
            spec.family
        );
    }

    let decode_tokens = [mixed_decode_token];
    let mixed_stage0_output = {
        let [decode, long, short] = &mut mixed_stage0;
        let mut requests = [
            IterationBatchRequest {
                session: decode,
                token_ids: &decode_tokens,
                positions: &[],
                sampling: None,
                input: None,
                sample_last: true,
                phase: IterationBatchPhase::Decode,
            },
            IterationBatchRequest {
                session: long,
                token_ids: long_prefill,
                positions: &[],
                sampling: None,
                input: None,
                sample_last: false,
                phase: IterationBatchPhase::Prefill,
            },
            IterationBatchRequest {
                session: short,
                token_ids: short_prefill,
                positions: &[],
                sampling: None,
                input: None,
                sample_last: true,
                phase: IterationBatchPhase::Prefill,
            },
        ];
        StageSession::iteration_batch_sampled(&mut requests)?
    };
    if !mixed_stage0_output.samples.is_empty() {
        bail!(
            "{} / {} intermediate stage unexpectedly sampled tokens",
            spec.llama_model,
            spec.family
        );
    }

    let (_, serial_decode_frame) =
        serial_stage0[0].decode_step_frame(serial_decode_token, None, 0)?;
    let serial_long_frame = serial_stage0[1].prefill_chunk_frame(long_prefill, None, 0)?;
    let serial_short_frame = serial_stage0[2].prefill_chunk_frame(short_prefill, None, 0)?;
    let serial_frames = [serial_decode_frame, serial_long_frame, serial_short_frame];
    for (request_index, (mixed, serial)) in mixed_stage0_output
        .request_outputs
        .iter()
        .zip(&serial_frames)
        .enumerate()
    {
        if !activation_frames_match(mixed, serial) {
            bail!(
                "{} / {} mixed intermediate activation for request {request_index} did not match serial",
                spec.llama_model,
                spec.family
            );
        }
    }

    let mixed_stage1_output = {
        let [decode, long, short] = &mut mixed_stage1;
        let mut requests = [
            IterationBatchRequest {
                session: decode,
                token_ids: &decode_tokens,
                positions: &[],
                sampling: None,
                input: Some(&mixed_stage0_output.request_outputs[0]),
                sample_last: true,
                phase: IterationBatchPhase::Decode,
            },
            IterationBatchRequest {
                session: long,
                token_ids: long_prefill,
                positions: &[],
                sampling: None,
                input: Some(&mixed_stage0_output.request_outputs[1]),
                sample_last: false,
                phase: IterationBatchPhase::Prefill,
            },
            IterationBatchRequest {
                session: short,
                token_ids: short_prefill,
                positions: &[],
                sampling: None,
                input: Some(&mixed_stage0_output.request_outputs[2]),
                sample_last: true,
                phase: IterationBatchPhase::Prefill,
            },
        ];
        StageSession::iteration_batch_sampled(&mut requests)?
    };
    let (serial_decode_prediction, _) = serial_stage1[0].decode_step_frame_sampled(
        serial_decode_token,
        None,
        Some(&serial_frames[0]),
        0,
    )?;
    serial_stage1[1].prefill_chunk_frame(long_prefill, Some(&serial_frames[1]), 0)?;
    let (serial_short_prediction, _) = serial_stage1[2].prefill_chunk_frame_sampled(
        short_prefill,
        None,
        Some(&serial_frames[2]),
        0,
    )?;
    let mixed_samples = mixed_stage1_output
        .samples
        .iter()
        .map(|sample| (sample.request_index, sample.predicted_token))
        .collect::<Vec<_>>();
    let expected_samples = vec![(0, serial_decode_prediction), (2, serial_short_prediction)];
    if mixed_samples != expected_samples {
        bail!(
            "{} / {} mixed samples {mixed_samples:?} did not match serial {expected_samples:?}",
            spec.llama_model,
            spec.family
        );
    }
    for index in 0..3 {
        if mixed_stage0[index].token_count() != serial_stage0[index].token_count()
            || mixed_stage1[index].token_count() != serial_stage1[index].token_count()
        {
            bail!(
                "{} / {} mixed request {index} advanced session state differently from serial",
                spec.llama_model,
                spec.family
            );
        }
    }
    Ok(())
}

fn resolve_case_for_ignored_test(spec: FamilySpec) -> Result<Option<ResolvedCase>> {
    match ResolvedCase::resolve(spec) {
        Ok(case) => Ok(Some(case)),
        Err(error)
            if manifest_row(spec)?.is_package_only()
                && !env_flag("SKIPPY_PARITY_REQUIRE_PACKAGE_ONLY", false) =>
        {
            eprintln!(
                "skipping package-only parity artifact for {} / {}: {error:#}",
                spec.llama_model, spec.family
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn prepare_native_logs() -> Result<()> {
    static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

    if env_flag("SKIPPY_PARITY_NATIVE_LOGS", false) {
        return Ok(());
    }

    let path = LOG_PATH.get_or_init(|| {
        std::env::temp_dir().join(format!("skippy-parity-native-{}.log", std::process::id()))
    });
    redirect_native_logs_to_file(path)
}

#[derive(Debug, Deserialize)]
struct Manifest {
    support_priority: SupportPriority,
    candidates: Vec<CandidateRow>,
}

#[derive(Debug, Deserialize)]
struct SupportPriority {
    p0: PriorityRows,
    p1: PriorityRows,
}

#[derive(Debug, Deserialize)]
struct PriorityRows {
    llama_models: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CandidateRow {
    llama_model: String,
    family: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    repo: String,
    #[serde(default)]
    include: Option<IncludeSpec>,
    #[serde(default)]
    recurrent: Option<String>,
    #[serde(default)]
    recurrent_ranges: Option<Vec<String>>,
    #[serde(default)]
    splits: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    n_gpu_layers: Option<i32>,
}

impl CandidateRow {
    fn include(&self) -> Result<&IncludeSpec> {
        self.include
            .as_ref()
            .with_context(|| format!("{} / {} is missing include", self.llama_model, self.family))
    }

    fn is_package_only(&self) -> bool {
        self.status == "certified_package_only"
    }

    fn is_recurrent(&self) -> bool {
        self.recurrent.is_some()
            || self
                .recurrent_ranges
                .as_ref()
                .is_some_and(|ranges| !ranges.is_empty())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum IncludeSpec {
    One(String),
    Many(Vec<String>),
}

impl IncludeSpec {
    fn files(&self) -> Vec<&str> {
        match self {
            Self::One(file) => vec![file.as_str()],
            Self::Many(files) => files.iter().map(String::as_str).collect(),
        }
    }

    fn is_package_manifest_only(&self) -> bool {
        matches!(self, Self::One(value) if value == "model-package.json")
    }
}

impl Manifest {
    fn priority_for(&self, row: &CandidateRow) -> &'static str {
        if self
            .support_priority
            .p0
            .llama_models
            .iter()
            .any(|model| model == &row.llama_model)
        {
            "p0"
        } else if self
            .support_priority
            .p1
            .llama_models
            .iter()
            .any(|model| model == &row.llama_model)
        {
            "p1"
        } else {
            "p2"
        }
    }

    fn rows_by_priority<const N: usize>(&self, priorities: [&str; N]) -> Vec<&CandidateRow> {
        self.candidates
            .iter()
            .filter(|row| priorities.contains(&self.priority_for(row)))
            .collect()
    }
}

static MANIFEST: OnceLock<Manifest> = OnceLock::new();

fn manifest() -> &'static Manifest {
    MANIFEST.get_or_init(|| {
        serde_json::from_str(MANIFEST_JSON).expect("parity candidate manifest must parse")
    })
}

fn manifest_row(spec: FamilySpec) -> Result<&'static CandidateRow> {
    let row = manifest()
        .candidates
        .iter()
        .find(|row| row.llama_model == spec.llama_model && row.family == spec.family)
        .with_context(|| {
            format!(
                "missing manifest row for {} / {}",
                spec.llama_model, spec.family
            )
        })?;
    let actual_priority = manifest().priority_for(row);
    if actual_priority != spec.priority {
        bail!(
            "{} / {} declared as {}, manifest says {}",
            spec.llama_model,
            spec.family,
            spec.priority,
            actual_priority
        );
    }
    Ok(row)
}

struct ResolvedCase {
    row: &'static CandidateRow,
    artifact: ResolvedArtifact,
}

enum ResolvedArtifact {
    Gguf(PathBuf),
    LayerPackage(PathBuf),
}

struct TestLayout {
    layer_count: u32,
    full_model: StagePath,
    package_dir: Option<PathBuf>,
}

#[derive(Clone)]
struct StagePath {
    path: PathBuf,
    load_mode: RuntimeLoadMode,
    filter_tensors_on_load: bool,
}

impl ResolvedCase {
    fn resolve(spec: FamilySpec) -> Result<Self> {
        assert_manifest_row_complete(spec)?;
        Self::resolve_artifact(spec)
    }

    fn resolve_boundary_contract(spec: FamilySpec) -> Result<Self> {
        let row = manifest_row(spec)?;
        if row.repo.trim().is_empty() {
            bail!("{} / {} is missing repo", row.llama_model, row.family);
        }
        if row.include()?.files().is_empty() {
            bail!(
                "{} / {} is missing include files",
                row.llama_model,
                row.family
            );
        }
        Self::resolve_artifact(spec)
    }

    fn resolve_artifact(spec: FamilySpec) -> Result<Self> {
        let row = manifest_row(spec)?;
        download_if_requested(row)?;
        let artifact = if row.include()?.is_package_manifest_only() {
            ResolvedArtifact::LayerPackage(resolve_package_dir(row)?)
        } else {
            ResolvedArtifact::Gguf(resolve_primary_gguf(row)?)
        };
        Ok(Self { row, artifact })
    }

    fn layout(&self) -> Result<TestLayout> {
        match &self.artifact {
            ResolvedArtifact::Gguf(path) => {
                let layer_count = layer_count_for_gguf(path)?;
                Ok(TestLayout {
                    layer_count,
                    full_model: StagePath {
                        path: path.clone(),
                        load_mode: RuntimeLoadMode::RuntimeSlice,
                        filter_tensors_on_load: false,
                        resident_tensor_names: Vec::new(),
                    },
                    package_dir: None,
                })
            }
            ResolvedArtifact::LayerPackage(package_dir) => {
                let package_ref = package_dir.to_string_lossy();
                let info = inspect_layer_package(&package_ref)
                    .with_context(|| format!("inspect layer package {}", package_dir.display()))?;
                Ok(TestLayout {
                    layer_count: info.layer_count,
                    full_model: StagePath {
                        path: package_dir.clone(),
                        load_mode: RuntimeLoadMode::LayerPackage,
                        filter_tensors_on_load: true,
                        resident_tensor_names: Vec::new(),
                    },
                    package_dir: Some(package_dir.clone()),
                })
            }
        }
    }
}

fn download_if_requested(row: &CandidateRow) -> Result<()> {
    if env_flag("SKIPPY_PARITY_DOWNLOAD", true) {
        if row.is_package_only()
            && !row.include()?.is_package_manifest_only()
            && !env_flag("SKIPPY_PARITY_DOWNLOAD_PACKAGE_ONLY", false)
        {
            return Ok(());
        }
        let mut command = Command::new("hf");
        command.args(["download", &row.repo]);
        if !row.include()?.is_package_manifest_only() {
            for include in row.include()?.files() {
                command.args(["--include", include]);
            }
        }
        let status = command
            .status()
            .with_context(|| format!("run hf download for {}", row.repo))?;
        if !status.success() {
            bail!("hf download failed for {} with status {status}", row.repo);
        }
    }
    Ok(())
}

fn resolve_primary_gguf(row: &CandidateRow) -> Result<PathBuf> {
    let candidates = repo_snapshot_files(&row.repo)?
        .into_iter()
        .filter(|path| {
            let rel = repo_relative_path(&row.repo, path)
                .map(|path| path.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            row.include()
                .map(IncludeSpec::files)
                .unwrap_or_default()
                .iter()
                .any(|pattern| wildcard_match(pattern, &rel))
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        let lower = name.to_ascii_lowercase();
                        lower.ends_with(".gguf") && !lower.starts_with("mmproj")
                    })
        })
        .collect::<Vec<_>>();
    choose_primary_gguf(row, candidates)
}

fn resolve_package_dir(row: &CandidateRow) -> Result<PathBuf> {
    repo_snapshot_dirs(&row.repo)?
        .into_iter()
        .find(|dir| dir.join("model-package.json").is_file())
        .with_context(|| {
            format!(
                "no downloaded model-package.json for {}; run SKIPPY_PARITY_DOWNLOAD=1 cargo test -p skippy-correctness --test parity_models -- --ignored",
                row.repo
            )
        })
}

fn choose_primary_gguf(row: &CandidateRow, mut candidates: Vec<PathBuf>) -> Result<PathBuf> {
    candidates.sort_by_key(|path| {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        (
            !name.contains("-00001-of-"),
            name.contains("-000") && !name.contains("-00001-of-"),
            name,
        )
    });
    candidates.into_iter().next().with_context(|| {
        format!(
            "no downloaded GGUF for {} include {:?}; run SKIPPY_PARITY_DOWNLOAD=1 cargo test -p skippy-correctness --test parity_models -- --ignored",
            row.repo,
            row.include().map(IncludeSpec::files).unwrap_or_default()
        )
    })
}

fn repo_snapshot_files(repo: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for dir in repo_snapshot_dirs(repo)? {
        collect_files(&dir, &mut files)?;
    }
    Ok(files)
}

fn repo_snapshot_dirs(repo: &str) -> Result<Vec<PathBuf>> {
    let cache_dir = model_hf::huggingface_hub_cache_dir();
    let repo_dir = cache_dir.join(format!("models--{}", repo.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    if !snapshots.is_dir() {
        bail!(
            "Hugging Face cache has no snapshots for {} at {}; run with SKIPPY_PARITY_DOWNLOAD=1",
            repo,
            snapshots.display()
        );
    }
    let mut dirs = fs::read_dir(&snapshots)
        .with_context(|| format!("read {}", snapshots.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    dirs.sort();
    dirs.reverse();
    Ok(dirs)
}

fn repo_relative_path(repo: &str, path: &Path) -> Option<PathBuf> {
    let cache_dir = model_hf::huggingface_hub_cache_dir();
    let snapshots = cache_dir
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    for snapshot in fs::read_dir(snapshots).ok()? {
        let snapshot = snapshot.ok()?.path();
        if let Ok(relative) = path.strip_prefix(snapshot) {
            return Some(relative.to_path_buf());
        }
    }
    None
}

fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn wildcard_match(pattern: &str, value: &str) -> bool {
    let pattern = pattern.replace('\\', "/");
    if pattern == value {
        return true;
    }
    let parts = pattern.split('*').collect::<Vec<_>>();
    if parts.len() == 1 {
        return pattern == value;
    }
    let mut remainder = value;
    if let Some(first) = parts.first().filter(|part| !part.is_empty()) {
        let Some(stripped) = remainder.strip_prefix(first) else {
            return false;
        };
        remainder = stripped;
    }
    for part in parts.iter().skip(1).take(parts.len().saturating_sub(2)) {
        if part.is_empty() {
            continue;
        }
        let Some(index) = remainder.find(part) else {
            return false;
        };
        remainder = &remainder[index + part.len()..];
    }
    if let Some(last) = parts.last().filter(|part| !part.is_empty()) {
        return remainder.ends_with(last);
    }
    true
}

fn layer_count_for_gguf(path: &Path) -> Result<u32> {
    let max_layer = tensors_for_gguf(path)?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .with_context(|| format!("{} has no layer-indexed tensors", path.display()))?;
    Ok(max_layer.saturating_add(1))
}

fn tensors_for_gguf(path: &Path) -> Result<Vec<skippy_runtime::TensorInfo>> {
    let info = skippy_runtime::ModelInfo::open(path)
        .with_context(|| format!("inspect {}", path.display()))?;
    let tensors = info.tensors()?;
    if tensors.iter().any(|tensor| tensor.layer_index.is_some()) {
        return Ok(tensors);
    }

    let shard_paths = split_gguf_sibling_paths(path)?;
    if shard_paths.len() <= 1 {
        return Ok(tensors);
    }

    let mut all_tensors = tensors;
    for shard in shard_paths.into_iter().filter(|shard| shard != path) {
        let info = skippy_runtime::ModelInfo::open(&shard)
            .with_context(|| format!("inspect split GGUF shard {}", shard.display()))?;
        all_tensors.extend(info.tensors()?);
    }
    Ok(all_tensors)
}

fn split_gguf_sibling_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("split GGUF path has no UTF-8 file name")?;
    let Some(shard) = split_gguf_shard_info(file_name) else {
        return Ok(vec![path.to_path_buf()]);
    };
    let total = shard
        .total
        .parse::<u32>()
        .context("parse split GGUF shard total")?;
    let dir = path
        .parent()
        .ok_or_else(|| anyhow!("split GGUF path has no parent directory"))?;
    let mut paths = Vec::new();
    for part in 1..=total {
        let shard_name = format!("{}-{part:05}-of-{}.gguf", shard.prefix, shard.total);
        paths.push(dir.join(shard_name));
    }
    Ok(paths)
}

fn split_layers_for(row: &CandidateRow, layer_count: u32) -> Result<(u32, u32)> {
    if let Some(splits) = row.splits.as_deref() {
        return parse_reviewed_splits(row, splits, layer_count);
    }
    split_layers(layer_count)
}

fn parse_reviewed_splits(row: &CandidateRow, splits: &str, layer_count: u32) -> Result<(u32, u32)> {
    let values = splits
        .split(',')
        .map(|value| {
            value.trim().parse::<u32>().with_context(|| {
                format!(
                    "parse reviewed splits for {} / {}",
                    row.llama_model, row.family
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    match values.as_slice() {
        [split_1, split_2] if *split_1 > 0 && split_1 < split_2 && *split_2 < layer_count => {
            Ok((*split_1, *split_2))
        }
        _ => bail!(
            "{} / {} reviewed splits {splits:?} must contain two increasing boundaries inside 0..{layer_count}",
            row.llama_model,
            row.family
        ),
    }
}

/// Cut-selection lane for the representative lattice fixture (see
/// `docs/skippy/LLAMA_PARITY.md`, "representative cut lattice").
///
/// For `layer_count <= FULL_LATTICE_MAX_LAYERS` every ordered pair is returned.
/// Above that the set is the deterministic, deduplicated union of: all adjacent
/// pairs `(s, s+1)`, all first-edge pairs `(1, s2)`, all final-edge pairs
/// `(s1, layer_count - 1)`, plus the reviewed manifest pair when it exists.
/// Boundary-adjacent cuts are where the hand-copied per-model split snippets
/// in the native builders diverge, which is how a single reviewed cut can hide
/// a Granite-class null-weight failure.
pub(crate) const FULL_LATTICE_MAX_LAYERS: u32 = 12;

pub(crate) fn representative_cut_lattice(
    layer_count: u32,
    reviewed: Option<(u32, u32)>,
) -> Result<Vec<(u32, u32)>> {
    if layer_count < 3 {
        bail!("cut lattice requires at least three layers, got {layer_count}");
    }
    let last = layer_count - 1;
    let mut cuts: BTreeSet<(u32, u32)> = BTreeSet::new();
    if layer_count <= FULL_LATTICE_MAX_LAYERS {
        for s1 in 1..last {
            for s2 in (s1 + 1)..layer_count {
                cuts.insert((s1, s2));
            }
        }
    } else {
        for s in 1..last {
            cuts.insert((s, s + 1));
            cuts.insert((1, s + 1));
            cuts.insert((s, last));
        }
    }
    if let Some(pair) = reviewed.filter(|p| p.0 > 0 && p.0 < p.1 && p.1 < layer_count) {
        cuts.insert(pair);
    }
    if cuts.is_empty() {
        bail!("cut lattice for {layer_count} layers is empty");
    }
    Ok(cuts.into_iter().collect())
}

/// Runs the representative cut lattice for one family. Coverage mode and pair
/// count are printed so a green run states exactly which cuts were exercised.
pub(crate) fn representative_cut_lattice_matches_full_model(spec: FamilySpec) -> Result<()> {
    prepare_native_logs()?;
    let Some(case) = resolve_case_for_ignored_test(spec)? else {
        return Ok(());
    };
    let layout = case.layout()?;
    if case.row.is_package_only() {
        bail!(
            "{} / {} cut-lattice coverage requires a full-model artifact",
            spec.llama_model,
            spec.family
        );
    }
    let reviewed = case
        .row
        .splits
        .as_deref()
        .map(|splits| parse_reviewed_splits(case.row, splits, layout.layer_count))
        .transpose()?;
    let cuts = representative_cut_lattice(layout.layer_count, reviewed)?;
    let coverage_mode = if layout.layer_count <= FULL_LATTICE_MAX_LAYERS {
        "full-lattice"
    } else {
        "representative-union"
    };
    println!(
        "{} / {}: coverage mode {coverage_mode}, {} cuts over {} layers",
        spec.llama_model,
        spec.family,
        cuts.len(),
        layout.layer_count
    );
    for (index, (split_1, split_2)) in cuts.iter().enumerate() {
        run_correctness_chain(&layout, spec, (*split_1, *split_2)).with_context(|| {
            format!(
                "{} / {} cut {}/{} failed: splits=({},{}), layer_count={}, stages=0..{s1} / {s1}..{s2} / {s2}..{L}",
                spec.llama_model,
                spec.family,
                index + 1,
                cuts.len(),
                split_1,
                split_2,
                layout.layer_count,
                s1 = split_1,
                s2 = split_2,
                L = layout.layer_count,
            )
        })?;
    }
    Ok(())
}

fn split_layers(layer_count: u32) -> Result<(u32, u32)> {
    if layer_count < 3 {
        bail!("parity activation handoff requires at least three layers, got {layer_count}");
    }
    let split_1 = (layer_count / 3).max(1);
    let split_2 = ((layer_count * 2) / 3)
        .max(split_1 + 1)
        .min(layer_count - 1);
    Ok((split_1, split_2))
}

fn cache_stage_range(row: &CandidateRow, layout: &TestLayout) -> Result<(u32, u32)> {
    if row.is_package_only() {
        if layout.layer_count == 0 {
            bail!("package-only cache smoke requires at least one layer");
        }
        return Ok((0, 1));
    }
    let (split_1, split_2) = split_layers_for(row, layout.layer_count)?;
    Ok((split_1, split_2))
}

fn run_lightweight_activation_smoke(layout: &TestLayout, spec: FamilySpec) -> Result<()> {
    let n_gpu_layers = case_n_gpu_layers(manifest_row(spec)?);
    if layout.layer_count < 2 {
        bail!("package activation smoke requires at least two layers");
    }
    let stage0_shape = StageShape {
        stage_index: 0,
        layer_start: 0,
        layer_end: 1,
        include_embeddings: true,
        include_output: false,
    };
    let stage1_shape = StageShape {
        stage_index: 1,
        layer_start: 1,
        layer_end: 2,
        include_embeddings: false,
        include_output: false,
    };
    let stage0 = open_stage_model(
        &stage_path(layout, spec, stage0_shape)?,
        stage0_shape,
        n_gpu_layers,
    )?;
    let stage1 = open_stage_model(
        &stage_path(layout, spec, stage1_shape)?,
        stage1_shape,
        n_gpu_layers,
    )?;
    let token = first_prompt_token(&stage0)?;
    let mut session0 = stage0.create_session()?;
    let (_, activation0) = session0.decode_step_frame(token, None, 0)?;
    ensure_activation_payload("package stage 0", &activation0)?;
    let mut session1 = stage1.create_session()?;
    let (_, activation1) = session1.decode_step_frame(token, Some(&activation0), 0)?;
    ensure_activation_payload("package stage 1", &activation1)?;
    Ok(())
}

fn run_correctness_chain(layout: &TestLayout, spec: FamilySpec, splits: (u32, u32)) -> Result<()> {
    let binary = std::env::var("CARGO_BIN_EXE_skippy-correctness")
        .unwrap_or_else(|_| "skippy-correctness".to_string());
    let stage_server_bin = std::env::var_os("SKIPPY_STAGE_SERVER_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/skippy-server")
        });
    if !stage_server_bin.is_file() {
        bail!(
            "skippy-server binary is required for activation chain tests; build it with `cargo build -p skippy-server` or set SKIPPY_STAGE_SERVER_BIN"
        );
    }
    let model_id = format!(
        "meshllm/{}",
        sanitize_model_id_part(&format!("{}-{}", spec.llama_model, spec.family))
    );
    let output = Command::new(&binary)
        .args([
            "chain",
            "--model",
            &layout.full_model.path.to_string_lossy(),
            "--model-id",
            &model_id,
            "--splits",
            &format!("{},{}", splits.0, splits.1),
            "--layer-end",
            &layout.layer_count.to_string(),
            "--n-gpu-layers",
            &case_n_gpu_layers(manifest_row(spec)?).to_string(),
            "--stage-server-bin",
            &stage_server_bin.to_string_lossy(),
            "--prompt",
            case_prompt(manifest_row(spec)?),
        ])
        .output()
        .with_context(|| {
            format!(
                "run {binary} chain for {} / {}",
                spec.llama_model, spec.family
            )
        })?;
    if !output.status.success() {
        bail!(
            "{} / {} correctness chain failed with status {}\nstdout:\n{}\nstderr:\n{}",
            spec.llama_model,
            spec.family,
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let report: Value = serde_json::from_slice(&output.stdout).with_context(|| {
        format!(
            "parse correctness chain report for {} / {}",
            spec.llama_model, spec.family
        )
    })?;
    if report.get("matches").and_then(Value::as_bool) != Some(true) {
        bail!(
            "{} / {} correctness chain did not match baseline: {}",
            spec.llama_model,
            spec.family,
            report
        );
    }
    Ok(())
}

fn case_prompt(row: &CandidateRow) -> &str {
    row.prompt.as_deref().unwrap_or("Hello")
}

fn case_n_gpu_layers(row: &CandidateRow) -> i32 {
    row.n_gpu_layers.unwrap_or_else(parity_n_gpu_layers)
}

fn sanitize_model_id_part(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
struct StageShape {
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
}

fn stage_path(layout: &TestLayout, spec: FamilySpec, shape: StageShape) -> Result<StagePath> {
    if matches!(layout.full_model.load_mode, RuntimeLoadMode::RuntimeSlice) {
        return Ok(StagePath {
            path: layout.full_model.path.clone(),
            load_mode: RuntimeLoadMode::RuntimeSlice,
            filter_tensors_on_load: true,
            resident_tensor_names: Vec::new(),
        });
    }
    let package_dir = layout
        .package_dir
        .as_deref()
        .ok_or_else(|| anyhow!("layer-package stage requested without a package dir"))?;
    materialize_stage(
        package_dir,
        manifest_row(spec)?,
        shape.layer_start,
        shape.layer_end,
        shape.include_embeddings,
        shape.include_output,
    )
}

fn materialize_stage(
    package_dir: &Path,
    row: &CandidateRow,
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<StagePath> {
    let materialized = materialize_layer_package_details(&PackageStageRequest {
        model_id: format!("{}:{}", row.repo, row.family),
        topology_id: "parity-model-tests".to_string(),
        package_ref: package_dir.to_string_lossy().into_owned(),
        stage_id: format!("stage-{layer_start}-{layer_end}"),
        layer_start,
        layer_end,
        include_embeddings,
        include_output,
    })?;
    Ok(StagePath {
        path: materialized.output_path,
        load_mode: RuntimeLoadMode::LayerPackage,
        filter_tensors_on_load: true,
        resident_tensor_names: Vec::new(),
    })
}

fn run_stage_state_restore(
    layout: &TestLayout,
    spec: FamilySpec,
    stage_start: u32,
    stage_end: u32,
    state_kind: StateKind,
) -> Result<()> {
    let n_gpu_layers = case_n_gpu_layers(manifest_row(spec)?);
    let input_shape = StageShape {
        stage_index: 0,
        layer_start: 0,
        layer_end: stage_start,
        include_embeddings: true,
        include_output: false,
    };
    let target_shape = StageShape {
        stage_index: 1,
        layer_start: stage_start,
        layer_end: stage_end,
        include_embeddings: stage_start == 0,
        include_output: stage_end == layout.layer_count,
    };
    let target = open_stage_model(
        &stage_path(layout, spec, target_shape)?,
        target_shape,
        n_gpu_layers,
    )?;
    let token_source = if stage_start == 0 {
        &target
    } else {
        &open_stage_model(
            &stage_path(layout, spec, input_shape)?,
            input_shape,
            n_gpu_layers,
        )?
    };
    let mut tokens = token_source.tokenize(PROMPT, true)?;
    if tokens.len() < 2 {
        bail!("prompt produced fewer than two tokens");
    }
    tokens.truncate(2);
    let prefix = vec![tokens[0]];
    let continuation = tokens[1];
    let (prefill_input, decode_input) = if stage_start == 0 {
        (None, None)
    } else {
        let input = open_stage_model(
            &stage_path(layout, spec, input_shape)?,
            input_shape,
            n_gpu_layers,
        )?;
        let mut input_session = input.create_session()?;
        let prefill = input_session.prefill_chunk_frame(&prefix, None, 0)?;
        let (_, decode) = input_session.decode_step_frame(continuation, None, 0)?;
        (Some(prefill), Some(decode))
    };

    let mut source = target.create_session()?;
    source.prefill_chunk_frame(&prefix, prefill_input.as_ref(), 0)?;
    let payload = export_state_payload(&mut source, state_kind, stage_start, stage_end, 1)?;
    let (source_predicted, source_frame) =
        source.decode_step_frame(continuation, decode_input.as_ref(), 0)?;

    let mut restored = target.create_session()?;
    import_state_payload(
        &mut restored,
        &payload,
        state_kind,
        stage_start,
        stage_end,
        1,
        &prefix,
    )?;
    let (restored_predicted, restored_frame) =
        restored.decode_step_frame(continuation, decode_input.as_ref(), 0)?;
    if source_predicted != restored_predicted {
        bail!(
            "{} / {} restored token {restored_predicted} did not match source {source_predicted}",
            spec.llama_model,
            spec.family
        );
    }
    if !activation_frames_match(&source_frame, &restored_frame) {
        bail!(
            "{} / {} restored activation payload did not match recompute",
            spec.llama_model,
            spec.family
        );
    }
    let suffix_matches = verify_suffix_prefill_after_restore(
        &target,
        &payload,
        state_kind,
        stage_start,
        stage_end,
        &prefix,
        continuation,
        prefill_input.as_ref(),
        decode_input.as_ref(),
        target_shape.include_output,
    )?;
    if !suffix_matches {
        bail!(
            "{} / {} suffix prefill after cache restore did not match recompute",
            spec.llama_model,
            spec.family
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum StateKind {
    ResidentKv,
    KvRecurrent,
}

#[derive(Clone)]
enum StatePayload {
    ResidentKv,
    KvRecurrent {
        kv: Option<RuntimeKvPage>,
        recurrent: Vec<u8>,
    },
}

#[allow(clippy::too_many_arguments)]
fn verify_suffix_prefill_after_restore(
    target: &StageModel,
    payload: &StatePayload,
    state_kind: StateKind,
    stage_start: u32,
    stage_end: u32,
    prefix: &[i32],
    continuation: i32,
    prefill_input: Option<&ActivationFrame>,
    decode_input: Option<&ActivationFrame>,
    include_output: bool,
) -> Result<bool> {
    let mut source = target.create_session()?;
    source.prefill_chunk_frame(prefix, prefill_input, 0)?;
    let (source_predicted, source_frame) = if include_output {
        let (predicted, _, frame) = source.verify_tokens_frame(&[continuation], decode_input, 0)?;
        (Some(predicted), frame)
    } else {
        (
            None,
            source.prefill_chunk_frame(&[continuation], decode_input, 0)?,
        )
    };

    let mut restored = target.create_session()?;
    import_state_payload(
        &mut restored,
        payload,
        state_kind,
        stage_start,
        stage_end,
        1,
        prefix,
    )?;
    let (restored_predicted, restored_frame) = if include_output {
        let (predicted, _, frame) =
            restored.verify_tokens_frame(&[continuation], decode_input, 0)?;
        (Some(predicted), frame)
    } else {
        (
            None,
            restored.prefill_chunk_frame(&[continuation], decode_input, 0)?,
        )
    };

    Ok(source_predicted == restored_predicted
        && activation_frames_match(&source_frame, &restored_frame))
}

fn activation_frames_match(source: &ActivationFrame, restored: &ActivationFrame) -> bool {
    if source.desc.token_count != restored.desc.token_count
        || source.desc.flags != restored.desc.flags
        || source.payload.len() != restored.payload.len()
    {
        return false;
    }
    if source.payload == restored.payload {
        return true;
    }
    if !source.payload.len().is_multiple_of(4) {
        return false;
    }

    source
        .payload
        .as_chunks::<4>()
        .0
        .iter()
        .zip(restored.payload.as_chunks::<4>().0.iter())
        .all(|(lhs, rhs)| {
            let lhs = f32::from_le_bytes(*lhs);
            let rhs = f32::from_le_bytes(*rhs);
            if lhs.to_bits() == rhs.to_bits() {
                return true;
            }
            if !lhs.is_finite() || !rhs.is_finite() {
                return false;
            }
            let diff = (lhs - rhs).abs();
            let scale = lhs.abs().max(rhs.abs()).max(1.0);
            diff <= 1.0e-3 || diff / scale <= 1.0e-3
        })
}

fn export_state_payload(
    session: &mut StageSession,
    state_kind: StateKind,
    layer_start: u32,
    layer_end: u32,
    token_count: u64,
) -> Result<StatePayload> {
    match state_kind {
        StateKind::ResidentKv => {
            session.save_prefix(CACHE_SEQ_ID, token_count)?;
            Ok(StatePayload::ResidentKv)
        }
        StateKind::KvRecurrent => {
            let kv = match session.export_kv_page(
                layer_start as i32,
                layer_end as i32,
                0,
                token_count,
            ) {
                Ok(page) => Some(page),
                Err(error) if native_kv_unavailable(&error) => None,
                Err(error) => return Err(error),
            };
            let recurrent = session.export_recurrent_state()?;
            if recurrent.is_empty() {
                bail!("KvRecurrent family exported empty recurrent state");
            }
            Ok(StatePayload::KvRecurrent { kv, recurrent })
        }
    }
}

fn import_state_payload(
    session: &mut StageSession,
    payload: &StatePayload,
    state_kind: StateKind,
    layer_start: u32,
    layer_end: u32,
    token_count: u64,
    token_ids: &[i32],
) -> Result<()> {
    match (state_kind, payload) {
        (StateKind::ResidentKv, StatePayload::ResidentKv) => {
            session.restore_prefix(CACHE_SEQ_ID, token_ids)
        }
        (StateKind::KvRecurrent, StatePayload::KvRecurrent { kv, recurrent }) => {
            if let Some(kv) = kv {
                session.import_kv_page(&kv.desc, &kv.payload)?;
            }
            session.import_recurrent_state_for_token_count(recurrent, token_count)
        }
        _ => bail!("state payload kind mismatch"),
    }
    .with_context(|| {
        format!(
            "import state payload for layers {layer_start}..{layer_end} and {token_count} tokens"
        )
    })
}

fn native_kv_unavailable(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.contains("runtime memory type is not supported for native KV pages")
            || message.contains("runtime has no attention KV cache")
            || message.contains("no KV cache layers selected by layer range")
    })
}

fn open_stage_model(path: &StagePath, shape: StageShape, n_gpu_layers: i32) -> Result<StageModel> {
    StageModel::open(
        &path.path,
        &RuntimeConfig {
            stage_index: shape.stage_index,
            layer_start: shape.layer_start,
            layer_end: shape.layer_end,
            ctx_size: CTX_SIZE,
            lane_count: 4,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_runtime::SplitMode::Auto,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
            load_mode: path.load_mode,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_runtime::GlmDsaPolicy::Auto,
            include_embeddings: shape.include_embeddings,
            include_output: shape.include_output,
            mtp_source: MtpSource::Disabled,
            filter_tensors_on_load: path.filter_tensors_on_load,
            resident_tensor_names: Vec::new(),
            checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
            checkpoint_imatrix: None,
            checkpoint_imatrix_sha256: None,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
        },
    )
    .with_context(|| {
        format!(
            "open {} layers {}..{}",
            path.path.display(),
            shape.layer_start,
            shape.layer_end
        )
    })
}

fn first_prompt_token(model: &StageModel) -> Result<i32> {
    model
        .tokenize(PROMPT, true)?
        .into_iter()
        .next()
        .context("prompt produced no tokens")
}

fn ensure_activation_payload(label: &str, frame: &ActivationFrame) -> Result<()> {
    if frame.payload.is_empty() {
        bail!("{label} produced empty activation payload");
    }
    Ok(())
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(default)
}

fn env_i32(name: &str) -> Option<i32> {
    std::env::var(name).ok()?.parse().ok()
}

fn parity_n_gpu_layers() -> i32 {
    env_i32("SKIPPY_PARITY_N_GPU_LAYERS").unwrap_or(999)
}

fn is_parity_artifact_status(status: &str) -> bool {
    matches!(
        status,
        "certified" | "certified_package_only" | "needs_boundary_registration"
    )
}
