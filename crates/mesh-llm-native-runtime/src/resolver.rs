use crate::{
    HostCudaProfile, HostRuntimeProfile, NativeRuntimeArtifact, NativeRuntimeBackendKind,
    NativeRuntimeCache, NativeRuntimeReleaseManifest,
};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::BTreeSet, path::PathBuf};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeSelection {
    Recommended,
    Backend {
        kind: NativeRuntimeBackendKind,
        cuda_toolkit_major: Option<u32>,
    },
    Id(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateRejection {
    MeshVersionMismatch { expected: String, actual: String },
    SkippyAbiMismatch { expected: String, actual: String },
    OsMismatch { expected: String, actual: String },
    ArchMismatch { expected: String, actual: String },
    TargetTripleMismatch { expected: String, actual: String },
    BackendNotSupported { backend: NativeRuntimeBackendKind },
    CudaProfileMissing,
    CudaToolkitMajorMismatch { required: u32, installed: Vec<u32> },
    CudaToolkitMajorAboveDriver { required: u32, driver_max: u32 },
    CudaToolkitNotDetected { required: u32 },
    CudaGpuArchUnsupported { supported: Vec<String> },
    RocmProfileMissing,
    RocmGpuArchUnsupported { supported: Vec<String> },
    VulkanProfileMissing,
    SelectionMismatch { selection: String },
}

impl std::fmt::Display for CandidateRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MeshVersionMismatch { expected, actual } => {
                write!(
                    f,
                    "MeshLLM version mismatch: expected {expected}, found {actual}"
                )
            }
            Self::SkippyAbiMismatch { expected, actual } => {
                write!(
                    f,
                    "Skippy ABI mismatch: expected {expected}, found {actual}"
                )
            }
            Self::OsMismatch { expected, actual } => {
                write!(
                    f,
                    "OS mismatch: expected {expected}, artifact is for {actual}"
                )
            }
            Self::ArchMismatch { expected, actual } => write!(
                f,
                "CPU architecture mismatch: expected {expected}, artifact is for {actual}"
            ),
            Self::TargetTripleMismatch { expected, actual } => {
                write!(
                    f,
                    "target triple mismatch: expected {expected}, host is {actual}"
                )
            }
            Self::BackendNotSupported { backend } => {
                write!(f, "backend {backend} is not supported on this host")
            }
            Self::CudaProfileMissing => {
                write!(
                    f,
                    "CUDA runtime requires CUDA, but no CUDA profile was detected"
                )
            }
            Self::CudaToolkitMajorMismatch {
                required,
                installed,
            } => {
                let installed = installed
                    .iter()
                    .map(|major| major.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "CUDA toolkit mismatch: runtime requires CUDA {required}, host has CUDA {installed} installed"
                )
            }
            Self::CudaToolkitMajorAboveDriver {
                required,
                driver_max,
            } => write!(
                f,
                "CUDA driver too old: runtime requires CUDA {required}, driver supports up to CUDA {driver_max}"
            ),
            Self::CudaToolkitNotDetected { required } => write!(
                f,
                "no CUDA toolkit detected: runtime requires CUDA {required} libraries \
                 (libcudart, libcublas, libcublasLt) on the loader path; \
                 set MESH_LLM_CUDA_TOOLKIT_MAJORS if the toolkit is installed elsewhere"
            ),
            Self::CudaGpuArchUnsupported { supported } => write!(
                f,
                "CUDA GPU architecture unsupported: runtime supports {}",
                supported.join(", ")
            ),
            Self::RocmProfileMissing => {
                write!(
                    f,
                    "ROCm runtime requires ROCm, but no ROCm profile was detected"
                )
            }
            Self::RocmGpuArchUnsupported { supported } => write!(
                f,
                "ROCm GPU architecture unsupported: runtime supports {}",
                supported.join(", ")
            ),
            Self::VulkanProfileMissing => {
                write!(
                    f,
                    "Vulkan runtime requires Vulkan, but no Vulkan profile was detected"
                )
            }
            Self::SelectionMismatch { selection } => {
                write!(f, "selection mismatch: requested {selection}")
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub artifact: NativeRuntimeArtifact,
    pub compatible: bool,
    pub rank: i64,
    #[serde(default)]
    pub rejection_reasons: Vec<CandidateRejection>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRuntimeSource {
    Installed { path: PathBuf },
    Bundle { path: PathBuf },
    Download { url: String },
    Missing,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeResolution {
    pub selected: NativeRuntimeArtifact,
    pub source: NativeRuntimeSource,
    #[serde(default)]
    pub evaluated: Vec<CandidateEvaluation>,
}

pub struct NativeRuntimeResolver {
    mesh_version: String,
    skippy_abi: Option<String>,
    profile: HostRuntimeProfile,
    release_manifest: NativeRuntimeReleaseManifest,
    cache: NativeRuntimeCache,
    bundle_dirs: Vec<PathBuf>,
}

impl RuntimeSelection {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        let Some(value) = value else {
            return Ok(Self::Recommended);
        };
        let value = value.trim();
        if value.is_empty() || value.eq_ignore_ascii_case("recommended") {
            return Ok(Self::Recommended);
        }
        if let Some(id) = value.strip_prefix("exact:") {
            return Ok(Self::Id(id.to_string()));
        }
        if value.starts_with("meshllm-") || value.starts_with("mesh-llm-") {
            return Ok(Self::Id(value.to_string()));
        }
        let lower = value.to_ascii_lowercase();
        if let Some(major) = lower.strip_prefix("cuda").and_then(parse_cuda_major) {
            return Ok(Self::Backend {
                kind: NativeRuntimeBackendKind::Cuda,
                cuda_toolkit_major: Some(major),
            });
        }
        Ok(Self::Backend {
            kind: lower.parse()?,
            cuda_toolkit_major: None,
        })
    }
}

impl NativeRuntimeResolver {
    pub fn new(
        mesh_version: impl Into<String>,
        profile: HostRuntimeProfile,
        release_manifest: NativeRuntimeReleaseManifest,
        cache: NativeRuntimeCache,
    ) -> Self {
        Self {
            mesh_version: mesh_version.into(),
            skippy_abi: None,
            profile,
            release_manifest,
            cache,
            bundle_dirs: Vec::new(),
        }
    }

    pub fn with_bundle_dirs(mut self, bundle_dirs: Vec<PathBuf>) -> Self {
        self.bundle_dirs = bundle_dirs;
        self
    }

    pub fn with_skippy_abi_version(mut self, skippy_abi_version: impl Into<String>) -> Self {
        self.skippy_abi = Some(skippy_abi_version.into());
        self
    }

    pub fn resolve(&self, selection: &RuntimeSelection) -> Result<NativeRuntimeResolution> {
        let evaluated = self.evaluate(selection)?;
        let expected_abi = self.expected_skippy_abi();
        let Some(selected) = best_candidate(&evaluated) else {
            bail!(
                "no compatible native runtime found for Skippy ABI {} on {}/{}",
                expected_abi,
                self.profile.os,
                self.profile.arch
            );
        };
        Ok(NativeRuntimeResolution {
            source: self.source_for_artifact(&selected.artifact)?,
            selected: selected.artifact.clone(),
            evaluated,
        })
    }

    pub fn evaluate(&self, selection: &RuntimeSelection) -> Result<Vec<CandidateEvaluation>> {
        let artifacts = self.candidate_artifacts()?;
        Ok(evaluate_candidates(
            &artifacts,
            &self.profile,
            &self.mesh_version,
            Some(self.expected_skippy_abi()),
            selection,
        ))
    }

    fn expected_skippy_abi(&self) -> &str {
        self.skippy_abi
            .as_deref()
            .unwrap_or(self.release_manifest.skippy_abi.as_str())
    }

    fn candidate_artifacts(&self) -> Result<Vec<NativeRuntimeArtifact>> {
        let mut seen = BTreeSet::new();
        let mut artifacts = Vec::new();
        for artifact in &self.release_manifest.artifacts {
            let artifact = artifact_with_manifest_mesh_version(
                artifact,
                self.release_manifest.mesh_version.as_str(),
            );
            seen.insert(artifact_key(&artifact));
            artifacts.push(artifact);
        }
        for dir in &self.bundle_dirs {
            let manifest = crate::NativeRuntimeManifest::read_from_dir(dir)?;
            let artifact = manifest.runtime;
            if seen.insert(artifact_key(&artifact)) {
                artifacts.push(artifact);
            }
        }
        // Older cache versions can contain pre-checksum manifests (#1162), and
        // they cannot satisfy this resolver's exact MeshLLM version anyway.
        for installed in self.cache.installed_for_version(&self.mesh_version)? {
            let artifact = installed.manifest.runtime;
            if seen.insert(artifact_key(&artifact)) {
                artifacts.push(artifact);
            }
        }
        Ok(artifacts)
    }

    fn source_for_artifact(&self, artifact: &NativeRuntimeArtifact) -> Result<NativeRuntimeSource> {
        for dir in &self.bundle_dirs {
            let Ok(manifest) = crate::NativeRuntimeManifest::read_from_dir(dir) else {
                continue;
            };
            if artifact_identity_matches(&manifest.runtime, artifact) {
                return Ok(NativeRuntimeSource::Bundle { path: dir.clone() });
            }
        }
        let installed = self.cache.find_installed(
            artifact.mesh_version_or(&self.mesh_version),
            artifact.native_runtime_id(),
        )?;
        if let Some(installed) = installed {
            return Ok(NativeRuntimeSource::Installed {
                path: installed.path,
            });
        }
        Ok(artifact
            .url
            .as_ref()
            .map(|url| NativeRuntimeSource::Download { url: url.clone() })
            .unwrap_or(NativeRuntimeSource::Missing))
    }
}

fn parse_cuda_major(value: &str) -> Option<u32> {
    (!value.is_empty()).then_some(value)?.parse().ok()
}

fn artifact_key(artifact: &NativeRuntimeArtifact) -> String {
    format!(
        "{}\0{}\0{}",
        artifact.id,
        artifact.mesh_version.as_deref().unwrap_or_default(),
        artifact.skippy_abi
    )
}

pub fn select_native_runtime(
    release_manifest: &NativeRuntimeReleaseManifest,
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    selection: &RuntimeSelection,
) -> Option<CandidateEvaluation> {
    select_native_runtime_for_skippy_abi(
        release_manifest,
        profile,
        mesh_version,
        &release_manifest.skippy_abi,
        selection,
    )
}

pub fn select_native_runtime_for_skippy_abi(
    release_manifest: &NativeRuntimeReleaseManifest,
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    skippy_abi: &str,
    selection: &RuntimeSelection,
) -> Option<CandidateEvaluation> {
    let artifacts = release_manifest
        .artifacts
        .iter()
        .map(|artifact| {
            artifact_with_manifest_mesh_version(artifact, release_manifest.mesh_version.as_str())
        })
        .collect::<Vec<_>>();
    let evaluated = evaluate_candidates(
        &artifacts,
        profile,
        mesh_version,
        Some(skippy_abi),
        selection,
    );
    best_candidate(&evaluated).cloned()
}

pub fn select_native_runtime_from_artifacts(
    artifacts: &[NativeRuntimeArtifact],
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    skippy_abi: Option<&str>,
    selection: &RuntimeSelection,
) -> Option<CandidateEvaluation> {
    let evaluated = evaluate_candidates(artifacts, profile, mesh_version, skippy_abi, selection);
    best_candidate(&evaluated).cloned()
}

fn evaluate_candidates(
    artifacts: &[NativeRuntimeArtifact],
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    skippy_abi: Option<&str>,
    selection: &RuntimeSelection,
) -> Vec<CandidateEvaluation> {
    artifacts
        .iter()
        .map(|artifact| evaluate_artifact(artifact, profile, mesh_version, skippy_abi, selection))
        .collect()
}

fn artifact_with_manifest_mesh_version(
    artifact: &NativeRuntimeArtifact,
    manifest_mesh_version: &str,
) -> NativeRuntimeArtifact {
    let mut artifact = artifact.clone();
    if artifact.mesh_version.is_none() {
        artifact.mesh_version = Some(manifest_mesh_version.to_string());
    }
    artifact
}

fn evaluate_artifact(
    artifact: &NativeRuntimeArtifact,
    profile: &HostRuntimeProfile,
    mesh_version: &str,
    skippy_abi: Option<&str>,
    selection: &RuntimeSelection,
) -> CandidateEvaluation {
    let mut reasons = Vec::new();
    if artifact.mesh_version.as_deref() != Some(mesh_version) {
        let actual = artifact
            .mesh_version
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        reasons.push(CandidateRejection::MeshVersionMismatch {
            expected: mesh_version.to_string(),
            actual,
        });
    }
    if let Some(skippy_abi) = skippy_abi
        && artifact.skippy_abi != skippy_abi
    {
        reasons.push(CandidateRejection::SkippyAbiMismatch {
            expected: skippy_abi.to_string(),
            actual: artifact.skippy_abi.clone(),
        });
    }
    if artifact.platform.os != profile.os {
        reasons.push(CandidateRejection::OsMismatch {
            expected: profile.os.clone(),
            actual: artifact.platform.os.clone(),
        });
    }
    if artifact.platform.arch != profile.arch {
        reasons.push(CandidateRejection::ArchMismatch {
            expected: profile.arch.clone(),
            actual: artifact.platform.arch.clone(),
        });
    }
    match (&artifact.platform.target, &profile.target_triple) {
        (Some(expected), Some(actual)) if expected != actual => {
            reasons.push(CandidateRejection::TargetTripleMismatch {
                expected: expected.clone(),
                actual: actual.clone(),
            });
        }
        _ => {}
    }
    if !profile.supports_flavor(&artifact.backend.kind) {
        reasons.push(CandidateRejection::BackendNotSupported {
            backend: artifact.backend.kind.clone(),
        });
    }
    evaluate_backend_requirements(artifact, profile, &mut reasons);
    if let Some(reason) = selection_mismatch(selection, artifact) {
        reasons.push(reason);
    }
    CandidateEvaluation {
        artifact: artifact.clone(),
        compatible: reasons.is_empty(),
        rank: artifact.rank + artifact.backend.kind.default_rank(),
        rejection_reasons: reasons,
    }
}

fn artifact_identity_matches(
    candidate: &NativeRuntimeArtifact,
    selected: &NativeRuntimeArtifact,
) -> bool {
    candidate.id == selected.id
        && candidate.mesh_version.as_deref() == selected.mesh_version.as_deref()
        && candidate.skippy_abi == selected.skippy_abi
}

fn evaluate_backend_requirements(
    artifact: &NativeRuntimeArtifact,
    profile: &HostRuntimeProfile,
    reasons: &mut Vec<CandidateRejection>,
) {
    match artifact.backend.kind {
        NativeRuntimeBackendKind::Cuda => evaluate_cuda_requirements(artifact, profile, reasons),
        NativeRuntimeBackendKind::Rocm => evaluate_rocm_requirements(artifact, profile, reasons),
        NativeRuntimeBackendKind::Vulkan if profile.vulkan.is_none() => {
            reasons.push(CandidateRejection::VulkanProfileMissing);
        }
        _ => {}
    }
}

fn evaluate_cuda_requirements(
    artifact: &NativeRuntimeArtifact,
    profile: &HostRuntimeProfile,
    reasons: &mut Vec<CandidateRejection>,
) {
    let Some(requirements) = &artifact.backend.cuda else {
        return;
    };
    let Some(cuda) = &profile.cuda else {
        reasons.push(CandidateRejection::CudaProfileMissing);
        return;
    };
    evaluate_cuda_toolkit_major(artifact, requirements.toolkit_major, cuda, reasons);
    if !requirements.gpu_arches.is_empty()
        && requirements
            .gpu_arches
            .iter()
            .all(|arch| !cuda.gpu_arches.contains(arch))
    {
        reasons.push(CandidateRejection::CudaGpuArchUnsupported {
            supported: requirements.gpu_arches.clone(),
        });
    }
}

/// A CUDA runtime can only load if the CUDA runtime libraries it links against
/// are present. Runtimes that ship their own copies (Windows) are self
/// contained; runtimes that do not (Linux) need a matching toolkit major
/// installed on the host.
///
/// `nvidia-smi` reports the highest CUDA major the *driver* supports, which is
/// routinely newer than the installed toolkit. Treat it as an upper bound only,
/// so a CUDA 13 driver with a CUDA 12 toolkit still selects the cuda12 runtime.
fn evaluate_cuda_toolkit_major(
    artifact: &NativeRuntimeArtifact,
    required: u32,
    cuda: &HostCudaProfile,
    reasons: &mut Vec<CandidateRejection>,
) {
    if let Some(driver_max) = cuda.driver_max_major
        && required > driver_max
    {
        reasons.push(CandidateRejection::CudaToolkitMajorAboveDriver {
            required,
            driver_max,
        });
        return;
    }

    if artifact_bundles_cuda_runtime(artifact) {
        return;
    }

    // Host-linked runtime with no detected toolkit: reject rather than guess.
    //
    // Accepting here would let a driver-only profile pick a runtime whose
    // libraries cannot be resolved, which is exactly the failure this check
    // exists to prevent. A wrong choice surfaces as a bare
    // `libcudart.so.<major>: cannot open shared object file` at load; refusing
    // instead keeps the reason visible and actionable. Hosts with a toolkit
    // installed somewhere the loader cannot see it can set
    // `MESH_LLM_CUDA_TOOLKIT_MAJORS`.
    if cuda.toolkit_majors.is_empty() {
        reasons.push(CandidateRejection::CudaToolkitNotDetected { required });
        return;
    }

    if !cuda.toolkit_majors.contains(&required) {
        reasons.push(CandidateRejection::CudaToolkitMajorMismatch {
            required,
            installed: cuda.toolkit_majors.iter().copied().collect(),
        });
    }
}

/// True when the artifact ships the CUDA runtime libraries it links against and
/// therefore does not depend on a host toolkit.
///
/// Requires the complete set the runtime loads (`cudart`, `cublas`,
/// `cublasLt`). A single bundled library does not make an artifact self
/// contained: shipping `cudart` alone still leaves `cublas` to the host.
fn artifact_bundles_cuda_runtime(artifact: &NativeRuntimeArtifact) -> bool {
    let mut cudart = false;
    let mut cublas = false;
    let mut cublas_lt = false;
    for library in &artifact.libraries {
        let name = library.rsplit('/').next().unwrap_or(library).to_lowercase();
        let name = name.strip_prefix("lib").unwrap_or(&name);
        // `cublaslt` is checked first so it is not shadowed by `cublas`.
        if name.starts_with("cublaslt") {
            cublas_lt = true;
        } else if name.starts_with("cublas") {
            cublas = true;
        } else if name.starts_with("cudart") {
            cudart = true;
        }
    }
    cudart && cublas && cublas_lt
}

fn evaluate_rocm_requirements(
    artifact: &NativeRuntimeArtifact,
    profile: &HostRuntimeProfile,
    reasons: &mut Vec<CandidateRejection>,
) {
    let Some(requirements) = &artifact.backend.rocm else {
        return;
    };
    let Some(rocm) = &profile.rocm else {
        reasons.push(CandidateRejection::RocmProfileMissing);
        return;
    };
    // HIP release artifacts contain architecture-specific code objects and do
    // not have a portable kernel fallback. Accepting a partial match can leave
    // an enumerated GPU with no launchable kernels, which faults at inference.
    if rocm
        .gpu_arches
        .iter()
        .any(|arch| !requirements.gpu_arches.contains(arch))
    {
        reasons.push(CandidateRejection::RocmGpuArchUnsupported {
            supported: requirements.gpu_arches.clone(),
        });
    }
}

fn selection_mismatch(
    selection: &RuntimeSelection,
    artifact: &NativeRuntimeArtifact,
) -> Option<CandidateRejection> {
    match selection {
        RuntimeSelection::Recommended => None,
        RuntimeSelection::Id(id) if id == &artifact.id => None,
        RuntimeSelection::Id(id) => Some(CandidateRejection::SelectionMismatch {
            selection: id.clone(),
        }),
        RuntimeSelection::Backend {
            kind,
            cuda_toolkit_major,
        } if kind == &artifact.backend.kind => match (kind, cuda_toolkit_major) {
            (NativeRuntimeBackendKind::Cuda, Some(major))
                if artifact
                    .backend
                    .cuda
                    .as_ref()
                    .is_some_and(|cuda| cuda.toolkit_major != *major) =>
            {
                Some(CandidateRejection::SelectionMismatch {
                    selection: format!("cuda{major}"),
                })
            }
            _ => None,
        },
        RuntimeSelection::Backend { kind, .. } => Some(CandidateRejection::SelectionMismatch {
            selection: kind.to_string(),
        }),
    }
}

fn best_candidate(evaluated: &[CandidateEvaluation]) -> Option<&CandidateEvaluation> {
    evaluated
        .iter()
        .filter(|candidate| candidate.compatible)
        .max_by(compare_candidates)
}

fn compare_candidates(left: &&CandidateEvaluation, right: &&CandidateEvaluation) -> Ordering {
    left.rank
        .cmp(&right.rank)
        .then_with(|| right.artifact.id.cmp(&left.artifact.id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CudaRuntimeRequirements, HostCudaProfile, HostRocmProfile, HostRuntimeProfile,
        NativeRuntimeBackend, NativeRuntimeManifest, NativeRuntimePlatform,
    };
    use std::path::Path;

    fn artifact(id: &str, backend: NativeRuntimeBackend) -> NativeRuntimeArtifact {
        NativeRuntimeArtifact {
            id: id.to_string(),
            mesh_version: Some("0.68.0".to_string()),
            skippy_abi: "0.1.25".to_string(),
            platform: NativeRuntimePlatform {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                target: None,
            },
            backend,
            rank: 0,
            libraries: vec!["lib/libllama.so".to_string()],
            files: Default::default(),
            tools: Default::default(),
            url: None,
            sha256: None,
            signature: None,
        }
    }

    fn write_bundle_runtime(path: &Path, runtime: NativeRuntimeArtifact) {
        std::fs::create_dir_all(path.join("lib")).unwrap();
        std::fs::write(path.join("lib/libllama.so"), b"native runtime").unwrap();
        NativeRuntimeManifest { runtime }
            .write_to_dir(path)
            .unwrap();
    }

    fn profile() -> HostRuntimeProfile {
        HostRuntimeProfile {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            target_triple: None,
            available_flavors: BTreeSet::from([
                NativeRuntimeBackendKind::Cpu,
                NativeRuntimeBackendKind::Cuda,
            ]),
            gpus: Vec::new(),
            cuda: Some(HostCudaProfile {
                toolkit_majors: BTreeSet::from([12]),
                driver_max_major: Some(12),
                driver_version: None,
                gpu_arches: BTreeSet::from(["sm_90".to_string()]),
            }),
            rocm: None,
            vulkan: None,
        }
    }

    fn cuda_runtime(id: &str, toolkit_major: u32, arches: &[&str]) -> NativeRuntimeArtifact {
        artifact(
            id,
            NativeRuntimeBackend {
                kind: NativeRuntimeBackendKind::Cuda,
                cuda: Some(CudaRuntimeRequirements {
                    toolkit_major,
                    min_driver: None,
                    gpu_arches: arches.iter().map(|value| value.to_string()).collect(),
                }),
                rocm: None,
                vulkan: None,
            },
        )
    }

    fn rocm_profile(arches: &[&str]) -> HostRuntimeProfile {
        HostRuntimeProfile {
            available_flavors: BTreeSet::from([
                NativeRuntimeBackendKind::Cpu,
                NativeRuntimeBackendKind::Rocm,
            ]),
            cuda: None,
            rocm: Some(HostRocmProfile {
                version: Some("7.0".to_string()),
                gpu_arches: arches.iter().map(|value| value.to_string()).collect(),
            }),
            ..profile()
        }
    }

    fn rocm_runtime(id: &str, arches: &[&str]) -> NativeRuntimeArtifact {
        artifact(
            id,
            NativeRuntimeBackend::rocm(arches.iter().map(|value| value.to_string()).collect()),
        )
    }

    #[test]
    fn recommended_prefers_compatible_cuda_over_cpu() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![
                artifact(
                    "meshllm-runtime-linux-x86_64-cpu",
                    NativeRuntimeBackend::cpu(),
                ),
                cuda_runtime("meshllm-runtime-linux-x86_64-cuda12", 12, &["sm_90"]),
            ],
        };
        let selected = select_native_runtime(
            &manifest,
            &profile(),
            "0.68.0",
            &RuntimeSelection::Recommended,
        )
        .unwrap();

        assert_eq!(selected.artifact.id, "meshllm-runtime-linux-x86_64-cuda12");
    }

    #[test]
    fn cuda13_runtime_is_rejected_on_cuda12_host() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![cuda_runtime(
                "meshllm-runtime-linux-x86_64-cuda13",
                13,
                &["sm_90"],
            )],
        };

        assert!(
            select_native_runtime(
                &manifest,
                &profile(),
                "0.68.0",
                &RuntimeSelection::Recommended
            )
            .is_none()
        );
    }

    /// A CUDA 13 driver with only a CUDA 12 toolkit installed must still select
    /// the cuda12 runtime. Linux runtimes do not bundle `libcudart`, so
    /// selecting cuda13 here fails at load with
    /// `libcudart.so.13: cannot open shared object file`.
    #[test]
    fn newer_driver_with_older_toolkit_selects_installed_toolkit_runtime() {
        let mut host = profile();
        let cuda = host.cuda.as_mut().unwrap();
        cuda.toolkit_majors = BTreeSet::from([12]);
        cuda.driver_max_major = Some(13);

        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![
                cuda_runtime("meshllm-runtime-linux-x86_64-cuda12", 12, &["sm_90"]),
                cuda_runtime("meshllm-runtime-linux-x86_64-cuda13", 13, &["sm_90"]),
            ],
        };

        let selected =
            select_native_runtime(&manifest, &host, "0.68.0", &RuntimeSelection::Recommended)
                .unwrap();

        assert_eq!(selected.artifact.id, "meshllm-runtime-linux-x86_64-cuda12");
    }

    /// Runtimes that ship their own CUDA libraries (Windows) do not depend on a
    /// host toolkit, so a newer driver must not reject them. Regression test for
    /// the silent Vulkan fallback in #1127.
    #[test]
    fn self_contained_cuda_runtime_is_accepted_on_newer_driver() {
        let mut host = profile();
        host.os = "windows".to_string();
        let cuda = host.cuda.as_mut().unwrap();
        cuda.toolkit_majors = BTreeSet::new();
        cuda.driver_max_major = Some(13);

        let mut bundled = cuda_runtime("meshllm-runtime-windows-x86_64-cuda12", 12, &["sm_90"]);
        bundled.platform.os = "windows".to_string();
        bundled.libraries.extend([
            "lib/cudart64_12.dll".to_string(),
            "lib/cublas64_12.dll".to_string(),
            "lib/cublasLt64_12.dll".to_string(),
        ]);

        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![bundled],
        };

        let selected =
            select_native_runtime(&manifest, &host, "0.68.0", &RuntimeSelection::Recommended)
                .unwrap();

        assert_eq!(
            selected.artifact.id,
            "meshllm-runtime-windows-x86_64-cuda12"
        );
    }

    /// Bundling only part of the CUDA library set does not make an artifact
    /// self contained: `cublas` would still have to come from the host.
    #[test]
    fn partially_bundled_cuda_runtime_is_not_treated_as_self_contained() {
        let mut host = profile();
        host.os = "windows".to_string();
        let cuda = host.cuda.as_mut().unwrap();
        cuda.toolkit_majors = BTreeSet::new();
        cuda.driver_max_major = Some(13);

        let mut partial = cuda_runtime("meshllm-runtime-windows-x86_64-cuda12", 12, &["sm_90"]);
        partial.platform.os = "windows".to_string();
        partial.libraries.push("lib/cudart64_12.dll".to_string());

        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![partial],
        };

        assert!(
            select_native_runtime(&manifest, &host, "0.68.0", &RuntimeSelection::Recommended)
                .is_none()
        );
    }

    /// A host-linked runtime must not be selected when no CUDA toolkit was
    /// detected. Accepting on driver evidence alone reproduces the
    /// `libcudart.so.<major>: cannot open shared object file` failure.
    #[test]
    fn host_linked_runtime_is_rejected_without_detected_toolkit() {
        let mut host = profile();
        let cuda = host.cuda.as_mut().unwrap();
        cuda.toolkit_majors = BTreeSet::new();
        cuda.driver_max_major = Some(13);

        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![cuda_runtime(
                "meshllm-runtime-linux-x86_64-cuda12",
                12,
                &["sm_90"],
            )],
        };

        assert!(
            select_native_runtime(&manifest, &host, "0.68.0", &RuntimeSelection::Recommended)
                .is_none()
        );
    }

    /// The same host still falls back to CPU rather than being left with no
    /// runtime at all.
    #[test]
    fn undetected_toolkit_falls_back_to_cpu_runtime() {
        let mut host = profile();
        let cuda = host.cuda.as_mut().unwrap();
        cuda.toolkit_majors = BTreeSet::new();
        cuda.driver_max_major = Some(13);

        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![
                artifact(
                    "meshllm-runtime-linux-x86_64-cpu",
                    NativeRuntimeBackend::cpu(),
                ),
                cuda_runtime("meshllm-runtime-linux-x86_64-cuda12", 12, &["sm_90"]),
            ],
        };

        let selected =
            select_native_runtime(&manifest, &host, "0.68.0", &RuntimeSelection::Recommended)
                .unwrap();

        assert_eq!(selected.artifact.id, "meshllm-runtime-linux-x86_64-cpu");
    }

    #[test]
    fn unsupported_cuda_gpu_arch_is_rejected() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![cuda_runtime(
                "meshllm-runtime-linux-x86_64-cuda12-sm120",
                12,
                &["sm_120"],
            )],
        };

        assert!(
            select_native_runtime(
                &manifest,
                &profile(),
                "0.68.0",
                &RuntimeSelection::Recommended
            )
            .is_none()
        );
    }

    #[test]
    fn cuda12_runtime_with_pascal_arch_is_selected_for_61() {
        let mut host = profile();
        host.cuda.as_mut().unwrap().gpu_arches = BTreeSet::from(["61".to_string()]);
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![cuda_runtime(
                "meshllm-runtime-linux-x86_64-cuda12",
                12,
                &["61", "75", "80", "86", "87", "89", "90"],
            )],
        };

        let selected =
            select_native_runtime(&manifest, &host, "0.68.0", &RuntimeSelection::Recommended)
                .expect("CUDA 12 runtime with Pascal kernels should be compatible");

        assert_eq!(selected.artifact.id, "meshllm-runtime-linux-x86_64-cuda12");
    }

    #[test]
    fn unsupported_rocm_apu_arch_falls_back_to_cpu() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![
                artifact(
                    "meshllm-runtime-linux-x86_64-cpu",
                    NativeRuntimeBackend::cpu(),
                ),
                rocm_runtime(
                    "meshllm-runtime-linux-x86_64-rocm",
                    &["gfx1100", "gfx1101", "gfx1102"],
                ),
            ],
        };

        let selected = select_native_runtime(
            &manifest,
            &rocm_profile(&["gfx1103"]),
            "0.68.0",
            &RuntimeSelection::Recommended,
        )
        .expect("CPU runtime should remain a safe fallback");

        assert_eq!(
            selected.artifact.backend.kind,
            NativeRuntimeBackendKind::Cpu
        );
    }

    #[test]
    fn rocm_runtime_without_packaged_arches_falls_back_to_cpu() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![
                artifact(
                    "meshllm-runtime-linux-x86_64-cpu",
                    NativeRuntimeBackend::cpu(),
                ),
                rocm_runtime("meshllm-runtime-linux-x86_64-rocm", &[]),
            ],
        };

        let selected = select_native_runtime(
            &manifest,
            &rocm_profile(&["gfx1103"]),
            "0.68.0",
            &RuntimeSelection::Recommended,
        )
        .expect("CPU runtime should remain a safe fallback");

        assert_eq!(
            selected.artifact.backend.kind,
            NativeRuntimeBackendKind::Cpu
        );
    }

    #[test]
    fn rocm_runtime_requires_complete_detected_arch_coverage() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![rocm_runtime(
                "meshllm-runtime-linux-x86_64-rocm",
                &["gfx1100"],
            )],
        };

        assert!(
            select_native_runtime(
                &manifest,
                &rocm_profile(&["gfx1100", "gfx1151"]),
                "0.68.0",
                &RuntimeSelection::Recommended,
            )
            .is_none(),
            "a partial match could expose the unsupported GPU to a kernel launch"
        );
    }

    #[test]
    fn rocm_runtime_accepts_rdna_apu_arches_when_both_are_packaged() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.68.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![rocm_runtime(
                "meshllm-runtime-linux-x86_64-rocm",
                &["gfx1100", "gfx1103", "gfx1151"],
            )],
        };

        let selected = select_native_runtime(
            &manifest,
            &rocm_profile(&["gfx1103", "gfx1151"]),
            "0.68.0",
            &RuntimeSelection::Recommended,
        )
        .expect("packaged RDNA APU kernels should make ROCm safe to select");

        assert_eq!(
            selected.artifact.backend.kind,
            NativeRuntimeBackendKind::Rocm
        );
    }

    #[test]
    fn mesh_version_mismatch_rejects_matching_skippy_abi_candidate() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.67.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![NativeRuntimeArtifact {
                mesh_version: Some("0.67.0".to_string()),
                ..cuda_runtime("meshllm-runtime-linux-x86_64-cuda12", 12, &["sm_90"])
            }],
        };
        assert!(
            select_native_runtime_for_skippy_abi(
                &manifest,
                &profile(),
                "0.68.0",
                "0.1.25",
                &RuntimeSelection::Recommended,
            )
            .is_none()
        );
    }

    #[test]
    fn explicit_mesh_version_and_skippy_abi_select_matching_candidate() {
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: "0.67.0".to_string(),
            skippy_abi: "0.1.25".to_string(),
            artifacts: vec![NativeRuntimeArtifact {
                mesh_version: Some("0.67.0".to_string()),
                ..cuda_runtime("meshllm-runtime-linux-x86_64-cuda12", 12, &["sm_90"])
            }],
        };
        let selected = select_native_runtime_for_skippy_abi(
            &manifest,
            &profile(),
            "0.67.0",
            "0.1.25",
            &RuntimeSelection::Recommended,
        )
        .unwrap();

        assert_eq!(selected.artifact.id, "meshllm-runtime-linux-x86_64-cuda12");
    }

    #[test]
    fn resolve_can_select_bundle_runtime_without_release_manifest_entry() {
        let bundle = tempfile::tempdir().unwrap();
        let cache_root = tempfile::tempdir().unwrap();
        let bundled_artifact = artifact(
            "meshllm-runtime-linux-x86_64-cpu",
            NativeRuntimeBackend::cpu(),
        );
        write_bundle_runtime(bundle.path(), bundled_artifact.clone());

        let resolution = NativeRuntimeResolver::new(
            "0.68.0",
            HostRuntimeProfile {
                available_flavors: BTreeSet::from([NativeRuntimeBackendKind::Cpu]),
                cuda: None,
                ..profile()
            },
            NativeRuntimeReleaseManifest {
                mesh_version: "0.68.0".to_string(),
                skippy_abi: "0.1.25".to_string(),
                artifacts: Vec::new(),
            },
            NativeRuntimeCache::new(cache_root.path()),
        )
        .with_bundle_dirs(vec![bundle.path().to_path_buf()])
        .with_skippy_abi_version("0.1.25")
        .resolve(&RuntimeSelection::Recommended)
        .unwrap();

        assert_eq!(resolution.selected.id, bundled_artifact.id);
        assert!(matches!(
            resolution.source,
            NativeRuntimeSource::Bundle { .. }
        ));
    }

    #[test]
    fn resolve_prefers_bundle_over_identical_cache_entry() {
        let bundle = tempfile::tempdir().unwrap();
        let cache_root = tempfile::tempdir().unwrap();
        let bundled_artifact = artifact(
            "meshllm-runtime-linux-x86_64-cpu",
            NativeRuntimeBackend::cpu(),
        );
        write_bundle_runtime(bundle.path(), bundled_artifact);
        let cache = NativeRuntimeCache::new(cache_root.path());
        cache.install_from_dir(bundle.path()).unwrap();

        let resolution = NativeRuntimeResolver::new(
            "0.68.0",
            HostRuntimeProfile {
                available_flavors: BTreeSet::from([NativeRuntimeBackendKind::Cpu]),
                cuda: None,
                ..profile()
            },
            NativeRuntimeReleaseManifest {
                mesh_version: "0.68.0".to_string(),
                skippy_abi: "0.1.25".to_string(),
                artifacts: Vec::new(),
            },
            cache,
        )
        .with_bundle_dirs(vec![bundle.path().to_path_buf()])
        .with_skippy_abi_version("0.1.25")
        .resolve(&RuntimeSelection::Recommended)
        .unwrap();

        assert_eq!(
            resolution.source,
            NativeRuntimeSource::Bundle {
                path: bundle.path().to_path_buf()
            }
        );
    }

    #[test]
    fn stale_bundle_with_same_id_does_not_satisfy_selected_artifact() {
        let bundle = tempfile::tempdir().unwrap();
        let cache_root = tempfile::tempdir().unwrap();
        let runtime_id = "meshllm-runtime-linux-x86_64-cpu";
        let stale_bundle_artifact = NativeRuntimeArtifact {
            mesh_version: Some("0.67.0".to_string()),
            ..artifact(runtime_id, NativeRuntimeBackend::cpu())
        };
        write_bundle_runtime(bundle.path(), stale_bundle_artifact);

        let resolution = NativeRuntimeResolver::new(
            "0.68.0",
            HostRuntimeProfile {
                available_flavors: BTreeSet::from([NativeRuntimeBackendKind::Cpu]),
                cuda: None,
                ..profile()
            },
            NativeRuntimeReleaseManifest {
                mesh_version: "0.68.0".to_string(),
                skippy_abi: "0.1.25".to_string(),
                artifacts: vec![artifact(runtime_id, NativeRuntimeBackend::cpu())],
            },
            NativeRuntimeCache::new(cache_root.path()),
        )
        .with_bundle_dirs(vec![bundle.path().to_path_buf()])
        .with_skippy_abi_version("0.1.25")
        .resolve(&RuntimeSelection::Recommended)
        .unwrap();

        assert!(matches!(resolution.source, NativeRuntimeSource::Missing));
    }

    #[test]
    fn legacy_cached_manifest_does_not_block_current_resolution() {
        let temp = tempfile::tempdir().unwrap();
        let cache = NativeRuntimeCache::new(temp.path().join("cache"));
        let legacy = cache.runtime_dir("0.67.0", "meshllm-runtime-linux-x86_64-cpu");
        std::fs::create_dir_all(legacy.join("lib")).unwrap();
        std::fs::write(legacy.join("lib/libllama.so"), b"legacy runtime").unwrap();
        std::fs::write(
            legacy.join(crate::NATIVE_RUNTIME_MANIFEST_FILE),
            r#"{
  "runtime": {
    "id": "meshllm-runtime-linux-x86_64-cpu",
    "mesh_version": "0.67.0",
    "skippy_abi": "0.1.25",
    "platform": {"os": "linux", "arch": "x86_64"},
    "backend": {"kind": "cpu"},
    "libraries": ["lib/libllama.so"]
  }
}"#,
        )
        .unwrap();

        let resolution = NativeRuntimeResolver::new(
            "0.68.0",
            HostRuntimeProfile {
                available_flavors: BTreeSet::from([NativeRuntimeBackendKind::Cpu]),
                cuda: None,
                ..profile()
            },
            NativeRuntimeReleaseManifest {
                mesh_version: "0.68.0".to_string(),
                skippy_abi: "0.1.25".to_string(),
                artifacts: vec![artifact(
                    "meshllm-runtime-linux-x86_64-cpu",
                    NativeRuntimeBackend::cpu(),
                )],
            },
            cache,
        )
        .with_skippy_abi_version("0.1.25")
        .resolve(&RuntimeSelection::Recommended)
        .unwrap();

        assert_eq!(resolution.selected.id, "meshllm-runtime-linux-x86_64-cpu");
        assert!(matches!(resolution.source, NativeRuntimeSource::Missing));
    }
}
