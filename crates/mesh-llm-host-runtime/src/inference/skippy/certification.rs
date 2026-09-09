use std::{fs, path::PathBuf, time::Duration};

use anyhow::{Context, Result, bail};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use skippy_package_format::{
    PackageManifest as PackageManifestV2, TensorStorage, stage_admission::StageAdmissionDescriptor,
};
use skippy_runtime::package::{self, PackageIntegrityOptions, PackageStageRequest};

use super::materialization::{
    StagePackageInfo, StagePackageRef, inspect_stage_package, resolve_hf_package_to_local,
    resolve_package_v2_stage_to_local,
};

const RUNTIME_SMOKE_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone, Debug)]
pub struct SkippyCertificationRequest {
    pub model_ref: String,
    pub package_only: bool,
    pub api_base: Option<String>,
    pub prompt: String,
    pub max_tokens: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CertificationGateStatus {
    Passed,
    Failed,
    Incomplete,
    NotRequired,
}

#[derive(Debug, Serialize)]
pub struct SkippyCertificationReport {
    pub schema_version: u32,
    pub status: CertificationGateStatus,
    pub input: String,
    pub resolved_package_ref: String,
    pub local_package_dir: String,
    pub model_id: String,
    pub manifest_sha256: String,
    pub source_model_path: String,
    pub source_model_sha256: String,
    pub source_model_bytes: Option<u64>,
    pub layer_count: u32,
    pub package_gate: CertificationGate,
    pub materialized_stages: Vec<CertifiedStage>,
    pub runtime_gates: Vec<CertificationGate>,
}

#[derive(Debug, Serialize)]
pub struct CertificationGate {
    pub name: String,
    pub status: CertificationGateStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CertifiedStage {
    pub stage_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub include_embeddings: bool,
    pub include_output: bool,
    pub selected_part_count: usize,
    pub verified_artifacts: usize,
    pub cached_artifacts: usize,
    pub materialized_path: String,
    pub materialized_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CertificationStageRange {
    layer_start: u32,
    layer_end: u32,
    include_embeddings: bool,
    include_output: bool,
}

pub async fn certify_layer_package(
    request: SkippyCertificationRequest,
) -> Result<SkippyCertificationReport> {
    let resolved_package_ref = resolve_certification_package_ref(&request.model_ref)?;
    let info = inspect_stage_package(&resolved_package_ref)?;
    let ranges = certification_stage_ranges(info.layer_count)?;
    let materialized_stages = tokio::task::spawn_blocking({
        let package_ref = resolved_package_ref.clone();
        let model_id = info.model_id.clone();
        move || materialize_certification_stages(&package_ref, &model_id, &ranges)
    })
    .await
    .map_err(anyhow::Error::from)??;
    let runtime_gates = runtime_smoke_gates(&request, &info).await;
    let package_gate = CertificationGate {
        name: "package_materialization".to_string(),
        status: CertificationGateStatus::Passed,
        details: Some("manifest, selected artifacts, and two local stage ranges verified".into()),
    };
    let status = aggregate_certification_status(
        std::iter::once(package_gate.status).chain(runtime_gates.iter().map(|gate| gate.status)),
    );

    Ok(SkippyCertificationReport {
        schema_version: 1,
        status,
        input: request.model_ref,
        resolved_package_ref,
        local_package_dir: info.package_dir.display().to_string(),
        model_id: info.model_id,
        manifest_sha256: info.manifest_sha256,
        source_model_path: info.source_model_path,
        source_model_sha256: info.source_model_sha256,
        source_model_bytes: info.source_model_bytes,
        layer_count: info.layer_count,
        package_gate,
        materialized_stages,
        runtime_gates,
    })
}

pub fn resolve_certification_package_ref(input: &str) -> Result<String> {
    if let Ok(package_ref) = StagePackageRef::parse(input) {
        if let Some(package_ref) = package_ref.as_package_ref() {
            return Ok(package_ref);
        }
        bail!("direct GGUF inputs are not layer-package certification targets");
    }
    crate::models::remote_catalog::find_layer_package(input)
        .with_context(|| format!("no layer package found for {input:?}"))
}

fn materialize_certification_stages(
    package_ref: &str,
    model_id: &str,
    ranges: &[CertificationStageRange],
) -> Result<Vec<CertifiedStage>> {
    let metadata_ref = resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    if super::package::is_package_v2_ref(&metadata_ref) {
        return materialize_package_v2_certification_stages(package_ref, ranges);
    }
    ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            let local_ref = resolve_hf_package_to_local(
                package_ref,
                range.layer_start,
                range.layer_end,
                range.include_embeddings,
                range.include_output,
            )?;
            let stage_id = format!("cert-stage-{index}");
            let request = PackageStageRequest {
                model_id: model_id.to_string(),
                topology_id: "skippy-certification".to_string(),
                package_ref: local_ref,
                stage_id: stage_id.clone(),
                layer_start: range.layer_start,
                layer_end: range.layer_end,
                include_embeddings: range.include_embeddings,
                include_output: range.include_output,
            };
            let integrity_options =
                PackageIntegrityOptions::verify_with_cache(package_integrity_cache_dir());
            let selected =
                package::select_layer_package_parts_with_integrity(&request, &integrity_options)?;
            let materialized = package::materialize_layer_package_details(&request)?;
            let materialized_bytes = fs::metadata(&materialized.output_path)
                .with_context(|| {
                    format!(
                        "read materialized certification stage {}",
                        materialized.output_path.display()
                    )
                })?
                .len();
            Ok(CertifiedStage {
                stage_id,
                layer_start: range.layer_start,
                layer_end: range.layer_end,
                include_embeddings: range.include_embeddings,
                include_output: range.include_output,
                selected_part_count: materialized.selected_parts.len(),
                verified_artifacts: selected.integrity.verified_artifacts,
                cached_artifacts: selected.integrity.cached_artifacts,
                materialized_path: materialized.output_path.display().to_string(),
                materialized_bytes,
            })
        })
        .collect()
}

fn materialize_package_v2_certification_stages(
    package_ref: &str,
    ranges: &[CertificationStageRange],
) -> Result<Vec<CertifiedStage>> {
    let local_ref = resolve_hf_package_to_local(package_ref, 0, 0, false, false)?;
    let package_dir = PathBuf::from(&local_ref);
    let manifest_path = package_dir.join("model-package.json");
    let manifest: PackageManifestV2 = serde_json::from_slice(
        &fs::read(&manifest_path)
            .with_context(|| format!("read package-v2 manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse package-v2 manifest {}", manifest_path.display()))?;
    let manifest =
        skippy_model::package_carrier::resolve_package_carrier_from_dir(manifest, &package_dir)
            .context("resolve package-v2 metadata carrier")?;

    ranges
        .iter()
        .enumerate()
        .map(|(index, range)| {
            let resident_tensor_ids = manifest
                .tensor_catalog
                .entries
                .iter()
                .map(|tensor| -> Result<Option<String>> {
                    let TensorStorage::Owned { artifact_id, .. } = &tensor.storage else {
                        return Ok(None);
                    };
                    let selected = match tensor.layer_ordinal {
                        Some(layer) => layer >= range.layer_start && layer < range.layer_end,
                        None => {
                            let artifact = manifest
                                .artifact_catalog
                                .entries
                                .iter()
                                .find(|artifact| artifact.id == *artifact_id)
                                .with_context(|| {
                                    format!(
                                        "package-v2 tensor {:?} references missing artifact {:?}",
                                        tensor.id, artifact_id
                                    )
                                })?;
                            match artifact.path.as_str() {
                                "shared/common.gguf" => true,
                                "shared/embeddings.gguf" => range.include_embeddings,
                                "shared/output.gguf" => range.include_output,
                                path => bail!(
                                    "package-v2 non-layer tensor {:?} references unsupported artifact path {:?}",
                                    tensor.id,
                                    path
                                ),
                            }
                        }
                    };
                    Ok(selected.then(|| tensor.id.clone()))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();
            let descriptor = StageAdmissionDescriptor {
                package_id: manifest.package_id.clone(),
                resident_tensor_ids,
                sidecars: if range.include_embeddings {
                    manifest.sidecars.clone()
                } else {
                    Vec::new()
                },
            };
            let (_stage_ref, model_parts, projector_path) =
                resolve_package_v2_stage_to_local(package_ref, &descriptor)?;
            let materialized_path = model_parts
                .first()
                .context("package-v2 certification stage has no primary model artifact")?
                .display()
                .to_string();
            let selected_part_count = model_parts.len() + usize::from(projector_path.is_some());
            let materialized_bytes = model_parts.iter().chain(projector_path.iter()).try_fold(
                0_u64,
                |total, path| {
                    let bytes = fs::metadata(path)
                        .with_context(|| format!("stat package-v2 artifact {}", path.display()))?
                        .len();
                    total
                        .checked_add(bytes)
                        .context("package-v2 certification byte count overflow")
                },
            )?;
            Ok(CertifiedStage {
                stage_id: format!("cert-stage-{index}"),
                layer_start: range.layer_start,
                layer_end: range.layer_end,
                include_embeddings: range.include_embeddings,
                include_output: range.include_output,
                selected_part_count,
                verified_artifacts: selected_part_count,
                cached_artifacts: 0,
                materialized_path,
                materialized_bytes,
            })
        })
        .collect()
}

fn certification_stage_ranges(layer_count: u32) -> Result<Vec<CertificationStageRange>> {
    if layer_count < 2 {
        bail!("layer package certification requires at least two transformer layers");
    }
    let split = layer_count / 2;
    Ok(vec![
        CertificationStageRange {
            layer_start: 0,
            layer_end: split,
            include_embeddings: true,
            include_output: false,
        },
        CertificationStageRange {
            layer_start: split,
            layer_end: layer_count,
            include_embeddings: false,
            include_output: true,
        },
    ])
}

async fn runtime_smoke_gates(
    request: &SkippyCertificationRequest,
    package: &StagePackageInfo,
) -> Vec<CertificationGate> {
    if request.package_only {
        return required_runtime_gate_names()
            .iter()
            .map(|name| CertificationGate {
                name: (*name).to_string(),
                status: CertificationGateStatus::NotRequired,
                details: Some("package-only certification requested".to_string()),
            })
            .collect();
    }

    let Some(api_base) = request.api_base.as_deref() else {
        return required_runtime_gate_names()
            .iter()
            .map(|name| CertificationGate {
                name: (*name).to_string(),
                status: CertificationGateStatus::Incomplete,
                details: Some("pass --api-base to run runtime OpenAI smoke gates".to_string()),
            })
            .collect();
    };

    let client = match reqwest::Client::builder()
        .timeout(RUNTIME_SMOKE_TIMEOUT)
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return required_runtime_gate_names()
                .iter()
                .map(|name| failed_gate(name, &error))
                .collect();
        }
    };
    vec![
        smoke_v1_models(&client, api_base, &package.model_id).await,
        smoke_chat_completions(&client, api_base, package, request).await,
        smoke_responses(&client, api_base, package, request).await,
    ]
}

async fn smoke_v1_models(
    client: &reqwest::Client,
    api_base: &str,
    model_id: &str,
) -> CertificationGate {
    let url = format!("{}/v1/models", api_base.trim_end_matches('/'));
    match client.get(url).send().await {
        Ok(response) if response.status() == StatusCode::OK => {
            match response.json::<serde_json::Value>().await {
                Ok(value) if models_response_contains(&value, model_id) => CertificationGate {
                    name: "v1_models".to_string(),
                    status: CertificationGateStatus::Passed,
                    details: None,
                },
                Ok(_) => CertificationGate {
                    name: "v1_models".to_string(),
                    status: CertificationGateStatus::Failed,
                    details: Some(format!("model {model_id:?} was not present in /v1/models")),
                },
                Err(error) => failed_gate("v1_models", error),
            }
        }
        Ok(response) => failed_gate_message("v1_models", format!("HTTP {}", response.status())),
        Err(error) => failed_gate("v1_models", error),
    }
}

async fn smoke_chat_completions(
    client: &reqwest::Client,
    api_base: &str,
    package: &StagePackageInfo,
    request: &SkippyCertificationRequest,
) -> CertificationGate {
    let url = format!("{}/v1/chat/completions", api_base.trim_end_matches('/'));
    let body = json!({
        "model": package.model_id,
        "messages": [{ "role": "user", "content": request.prompt }],
        "max_tokens": request.max_tokens,
        "stream": false
    });
    smoke_post_json(
        client,
        &url,
        body,
        "v1_chat_completions",
        response_has_chat_choice_content,
        "chat completion choice content",
    )
    .await
}

async fn smoke_responses(
    client: &reqwest::Client,
    api_base: &str,
    package: &StagePackageInfo,
    request: &SkippyCertificationRequest,
) -> CertificationGate {
    let url = format!("{}/v1/responses", api_base.trim_end_matches('/'));
    let body = json!({
        "model": package.model_id,
        "input": request.prompt,
        "max_output_tokens": request.max_tokens
    });
    smoke_post_json(
        client,
        &url,
        body,
        "v1_responses",
        response_has_responses_output,
        "Responses output",
    )
    .await
}

async fn smoke_post_json(
    client: &reqwest::Client,
    url: &str,
    body: serde_json::Value,
    name: &str,
    valid_response: fn(&serde_json::Value) -> bool,
    expected: &'static str,
) -> CertificationGate {
    match client.post(url).json(&body).send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<serde_json::Value>().await {
                Ok(value) if valid_response(&value) => CertificationGate {
                    name: name.to_string(),
                    status: CertificationGateStatus::Passed,
                    details: None,
                },
                Ok(_) => failed_gate_message(name, format!("response missing {expected}")),
                Err(error) => failed_gate(name, error),
            }
        }
        Ok(response) => failed_gate_message(name, format!("HTTP {}", response.status())),
        Err(error) => failed_gate(name, error),
    }
}

fn response_has_chat_choice_content(value: &serde_json::Value) -> bool {
    value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .is_some_and(|choices| {
            choices.iter().any(|choice| {
                choice
                    .pointer("/message/content")
                    .is_some_and(response_content_has_text)
            })
        })
}

fn response_has_responses_output(value: &serde_json::Value) -> bool {
    value
        .get("output_text")
        .and_then(|output_text| output_text.as_str())
        .is_some_and(|output_text| !output_text.trim().is_empty())
        || value
            .get("output")
            .and_then(|output| output.as_array())
            .is_some_and(|items| {
                items.iter().any(|item| {
                    item.get("content")
                        .and_then(|content| content.as_array())
                        .is_some_and(|content| {
                            content.iter().any(|part| {
                                part.get("type").and_then(|kind| kind.as_str())
                                    == Some("output_text")
                                    && part
                                        .get("text")
                                        .and_then(|text| text.as_str())
                                        .is_some_and(|text| !text.trim().is_empty())
                            })
                        })
                })
            })
}

fn response_content_has_text(value: &serde_json::Value) -> bool {
    value.as_str().is_some_and(|text| !text.trim().is_empty())
        || value.as_array().is_some_and(|parts| {
            parts.iter().any(|part| {
                part.as_str().is_some_and(|text| !text.trim().is_empty())
                    || part
                        .get("text")
                        .and_then(|text| text.as_str())
                        .is_some_and(|text| !text.trim().is_empty())
            })
        })
}

fn models_response_contains(value: &serde_json::Value, model_id: &str) -> bool {
    value
        .get("data")
        .and_then(|data| data.as_array())
        .is_some_and(|models| {
            models
                .iter()
                .any(|model| model.get("id").and_then(|id| id.as_str()) == Some(model_id))
        })
}

fn aggregate_certification_status(
    statuses: impl IntoIterator<Item = CertificationGateStatus>,
) -> CertificationGateStatus {
    let mut saw_incomplete = false;
    for status in statuses {
        match status {
            CertificationGateStatus::Failed => return CertificationGateStatus::Failed,
            CertificationGateStatus::Incomplete => saw_incomplete = true,
            CertificationGateStatus::Passed | CertificationGateStatus::NotRequired => {}
        }
    }
    if saw_incomplete {
        CertificationGateStatus::Incomplete
    } else {
        CertificationGateStatus::Passed
    }
}

fn required_runtime_gate_names() -> &'static [&'static str] {
    &["v1_models", "v1_chat_completions", "v1_responses"]
}

fn package_integrity_cache_dir() -> PathBuf {
    crate::models::mesh_llm_cache_dir().join("skippy-package-integrity")
}

fn failed_gate(name: &str, error: impl std::fmt::Display) -> CertificationGate {
    failed_gate_message(name, error.to_string())
}

fn failed_gate_message(name: &str, details: String) -> CertificationGate {
    CertificationGate {
        name: name.to_string(),
        status: CertificationGateStatus::Failed,
        details: Some(details),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CertificationGateStatus, aggregate_certification_status, certification_stage_ranges,
        materialize_package_v2_certification_stages, models_response_contains,
        response_has_chat_choice_content, response_has_responses_output, smoke_chat_completions,
        smoke_responses,
    };
    use crate::inference::skippy::materialization::{StagePackageInfo, StagePackageLayerInfo};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn certification_ranges_split_two_stage_package() {
        let ranges = certification_stage_ranges(5).unwrap();

        assert_eq!(ranges[0].layer_start, 0);
        assert_eq!(ranges[0].layer_end, 2);
        assert!(ranges[0].include_embeddings);
        assert!(!ranges[0].include_output);
        assert_eq!(ranges[1].layer_start, 2);
        assert_eq!(ranges[1].layer_end, 5);
        assert!(!ranges[1].include_embeddings);
        assert!(ranges[1].include_output);
    }

    #[test]
    fn certification_ranges_reject_single_layer_package() {
        let error = certification_stage_ranges(1).unwrap_err().to_string();

        assert!(error.contains("at least two transformer layers"), "{error}");
    }

    #[test]
    fn package_v2_certification_selects_two_stage_artifact_closures() {
        let root = tempfile::tempdir().unwrap();
        crate::inference::skippy::write_test_package_v2_fixture(
            root.path(),
            "fixture/llama-1b",
            &[
                (
                    "layer-00000",
                    "layers/layer-00000.gguf",
                    "blk.0.attn.weight",
                ),
                (
                    "layer-00001",
                    "layers/layer-00001.gguf",
                    "blk.1.attn.weight",
                ),
            ],
        )
        .unwrap();
        let ranges = certification_stage_ranges(2).unwrap();

        let stages =
            materialize_package_v2_certification_stages(&root.path().to_string_lossy(), &ranges)
                .unwrap();

        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].selected_part_count, 2);
        assert_eq!(stages[1].selected_part_count, 2);
        assert!(stages.iter().all(|stage| stage.verified_artifacts == 2));
        assert!(stages.iter().all(|stage| {
            let path = std::path::Path::new(&stage.materialized_path);
            path.is_file() && path.ends_with("shared/metadata.gguf")
        }));
    }

    #[test]
    fn package_v2_certification_rejects_unclassified_non_layer_artifact() {
        let root = tempfile::tempdir().unwrap();
        crate::inference::skippy::write_test_package_v2_fixture(
            root.path(),
            "fixture/llama-1b",
            &[
                (
                    "renamed-embeddings",
                    "shared/renamed.gguf",
                    "token_embd.weight",
                ),
                (
                    "layer-00000",
                    "layers/layer-00000.gguf",
                    "blk.0.attn.weight",
                ),
                (
                    "layer-00001",
                    "layers/layer-00001.gguf",
                    "blk.1.attn.weight",
                ),
            ],
        )
        .unwrap();
        let ranges = certification_stage_ranges(2).unwrap();

        let error =
            materialize_package_v2_certification_stages(&root.path().to_string_lossy(), &ranges)
                .unwrap_err()
                .to_string();

        assert!(error.contains("unsupported artifact path"), "{error}");
    }

    #[test]
    fn aggregate_status_prefers_failed_over_incomplete() {
        let status = aggregate_certification_status([
            CertificationGateStatus::Passed,
            CertificationGateStatus::Incomplete,
            CertificationGateStatus::Failed,
        ]);

        assert_eq!(status, CertificationGateStatus::Failed);
    }

    #[test]
    fn aggregate_status_allows_not_required_runtime_gates() {
        let status = aggregate_certification_status([
            CertificationGateStatus::Passed,
            CertificationGateStatus::NotRequired,
        ]);

        assert_eq!(status, CertificationGateStatus::Passed);
    }

    #[test]
    fn models_response_requires_matching_model_id() {
        let response = json!({
            "object": "list",
            "data": [
                { "id": "other" },
                { "id": "org/repo:Q4_K_M" }
            ]
        });

        assert!(models_response_contains(&response, "org/repo:Q4_K_M"));
        assert!(!models_response_contains(&response, "missing"));
    }

    #[test]
    fn chat_response_validator_accepts_string_and_structured_text_content() {
        let string_content = json!({
            "choices": [
                { "message": { "content": "ok" } }
            ]
        });
        let structured_content = json!({
            "choices": [
                {
                    "message": {
                        "content": [
                            { "type": "text", "text": "ok" }
                        ]
                    }
                }
            ]
        });

        assert!(response_has_chat_choice_content(&string_content));
        assert!(response_has_chat_choice_content(&structured_content));
    }

    #[test]
    fn responses_response_validator_accepts_output_text_and_output_parts() {
        let output_text = json!({
            "output_text": "ok"
        });
        let output_parts = json!({
            "output": [
                {
                    "content": [
                        { "type": "output_text", "text": "ok" }
                    ]
                }
            ]
        });

        assert!(response_has_responses_output(&output_text));
        assert!(response_has_responses_output(&output_parts));
    }

    #[tokio::test]
    async fn chat_smoke_rejects_success_status_without_choice_content() {
        let api_base = spawn_single_response_server(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 14\r\n\r\n{\"choices\":[]}",
        )
        .await;
        let package = fake_package_info();
        let request = fake_certification_request();

        let gate =
            smoke_chat_completions(&reqwest::Client::new(), &api_base, &package, &request).await;

        assert_eq!(gate.status, CertificationGateStatus::Failed);
        assert!(
            gate.details
                .as_deref()
                .is_some_and(|details| details.contains("choice content")),
            "{gate:?}"
        );
    }

    #[tokio::test]
    async fn responses_smoke_rejects_success_status_without_output() {
        let api_base = spawn_single_response_server(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
        )
        .await;
        let package = fake_package_info();
        let request = fake_certification_request();

        let gate = smoke_responses(&reqwest::Client::new(), &api_base, &package, &request).await;

        assert_eq!(gate.status, CertificationGateStatus::Failed);
        assert!(
            gate.details
                .as_deref()
                .is_some_and(|details| details.contains("Responses output")),
            "{gate:?}"
        );
    }

    fn fake_certification_request() -> super::SkippyCertificationRequest {
        super::SkippyCertificationRequest {
            model_ref: "hf://meshllm/demo@abc123".to_string(),
            package_only: false,
            api_base: None,
            prompt: "Say ok.".to_string(),
            max_tokens: 2,
        }
    }

    fn fake_package_info() -> StagePackageInfo {
        StagePackageInfo {
            package_ref: "hf://meshllm/demo@abc123".to_string(),
            package_dir: std::path::PathBuf::from("/tmp/demo-package"),
            manifest_sha256: "a".repeat(64),
            model_id: "meshllm/demo".to_string(),
            source_model_path: "model.gguf".to_string(),
            source_model_sha256: "b".repeat(64),
            source_model_bytes: Some(42),
            layer_count: 2,
            activation_width: 4096,
            generation: None,
            projector_path: None,
            layers: vec![StagePackageLayerInfo {
                layer_index: 0,
                tensor_count: 1,
                tensor_bytes: 1,
                artifact_bytes: 1,
            }],
        }
    }

    async fn spawn_single_response_server(response: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 2048];
            let _ = stream.read(&mut buf).await.unwrap();
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        format!("http://{addr}")
    }
}
