use super::*;
use skippy_model::gguf_catalog::read_gguf_catalog;
use skippy_package_format::TensorStorage;
use skippy_package_format::stage_admission::StageAdmissionDescriptor;
use std::collections::BTreeMap;

use crate::test_gguf::{FixtureTensor, explicit, fixture, tensor};

fn tensor_info(name: &str) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        layer_index: None,
        role: skippy_ffi::TensorRole::Unknown,
        ggml_type: 0,
        byte_size: 4,
        element_count: 1,
    }
}

#[test]
fn metadata_descriptor_match_rejects_duplicate_names_with_an_omission() {
    let expected = BTreeMap::from([
        ("first".to_string(), (0, 1)),
        ("second".to_string(), (0, 1)),
    ]);
    let repeated = vec![tensor_info("first"), tensor_info("first")];

    assert!(!metadata_descriptors_match(&repeated, &expected));
}

fn write(source: &Path, out: &Path, resume: bool) -> Result<()> {
    write_package(
        source.display().to_string(),
        out.to_path_buf(),
        Vec::new(),
        ArtifactHook { command: None },
        ArtifactHook { command: None },
        explicit(source),
        resume,
    )
}

fn read_manifest(out: &Path) -> PackageManifest {
    let root = serde_json::from_slice(&fs::read(out.join("model-package.json")).unwrap()).unwrap();
    resolve_package_carrier(root, out.join("shared/metadata.gguf")).unwrap()
}

#[test]
fn writer_repacks_all_source_tensors_by_native_role() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("model.gguf");
    fixture(
        &source,
        &[tensor("unknown-global", 0), tensor("blk.1.arbitrary", 32)],
        None,
    );
    let out = temp.path().join("package");
    write(&source, &out, false).unwrap();
    let manifest = read_manifest(&out);
    manifest.validate().unwrap();
    assert_eq!(manifest.schema_version, 2);
    assert_eq!(manifest.tensor_catalog.entries.len(), 2);
    let directory = read_gguf_catalog(&source).unwrap();
    assert_eq!(manifest.model_metadata, directory.metadata);
    assert_eq!(manifest.source_model.metadata_artifact_id, "metadata");
    assert_eq!(manifest.artifact_catalog.entries.len(), 3);
    for tensor in &manifest.tensor_catalog.entries {
        assert_eq!(
            tensor.layer_ordinal,
            (tensor.name == "blk.1.arbitrary").then_some(1)
        );
        match tensor.storage {
            TensorStorage::Owned {
                ref artifact_id,
                data_offset,
                stored_length,
                alignment,
                ..
            } => {
                assert_eq!(
                    artifact_id,
                    if tensor.name == "blk.1.arbitrary" {
                        "layer-00001"
                    } else {
                        "common"
                    }
                );
                assert!(data_offset >= alignment);
                assert_eq!(stored_length, 16, "not the 32-byte offset gap/padding");
                assert_eq!(alignment, 32);
            }
            _ => panic!("independent allocations are not aliases"),
        }
    }
    let json = serde_json::to_value(&manifest).unwrap();
    assert!(json.get("model_metadata").is_none());
    assert!(json.get("tensor_catalog").is_none());
    assert_eq!(manifest.package_id, manifest.computed_package_id().unwrap());
}

#[test]
fn renamed_complete_shards_use_primary_count_and_bind_every_tensor() {
    let temp = tempfile::tempdir().unwrap();
    let first = temp.path().join("model-00001-of-00002.gguf");
    let second = temp.path().join("model-00002-of-00002.gguf");
    fixture(&first, &[tensor("first", 0)], Some((0, 2, 2)));
    fixture(&second, &[tensor("second", 0)], Some((1, 2, 1)));
    let out = temp.path().join("package");
    write(&second, &out, false).unwrap();
    let manifest = read_manifest(&out);
    manifest.validate().unwrap();
    assert_eq!(manifest.source_model.files.len(), 2);
    assert_eq!(manifest.artifact_catalog.entries.len(), 2);
    assert_eq!(manifest.tensor_catalog.entries.len(), 2);
    assert_eq!(manifest.source_model.sha256, file_sha256(&second).unwrap());
    assert_eq!(manifest.source_model.metadata_artifact_id, "metadata");
    assert_eq!(
        manifest
            .artifact_catalog
            .entries
            .iter()
            .map(|artifact| artifact.path.as_str())
            .collect::<Vec<_>>(),
        ["shared/metadata.gguf", "shared/common.gguf"]
    );
    let metadata_artifact = manifest
        .artifact_catalog
        .entries
        .iter()
        .find(|artifact| artifact.id == manifest.source_model.metadata_artifact_id)
        .unwrap();
    assert!(
        metadata_artifact.byte_size
            < manifest
                .source_model
                .files
                .iter()
                .map(|file| file.byte_size)
                .sum()
    );

    let (_, first_catalog) = crate::source_inventory::inspect(&first, "source-00000").unwrap();
    let (_, second_catalog) = crate::source_inventory::inspect(&second, "source-00001").unwrap();
    let expected = first_catalog
        .entries
        .iter()
        .chain(&second_catalog.entries)
        .map(|tensor| (tensor.id.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let resident_tensor_ids = manifest
        .tensor_catalog
        .entries
        .iter()
        .map(|tensor| tensor.id.clone())
        .collect::<Vec<_>>();
    let resolved = manifest
        .resolve_stage_admission(&StageAdmissionDescriptor {
            package_id: manifest.package_id.clone(),
            resident_tensor_ids,
            sidecars: Vec::new(),
        })
        .unwrap();
    assert_eq!(resolved.tensor_bindings.len(), expected.len());
    for binding in resolved.tensor_bindings {
        let tensor = expected.get(binding.tensor_id).unwrap();
        assert_eq!(binding.native_name, tensor.name);
        assert_eq!(binding.ggml_type, tensor.ggml_type);
        assert_eq!(binding.dimensions, tensor.dimensions);
        assert_eq!(binding.artifact.id, "common");
        assert!(binding.stored_length > 0);
    }
}

#[test]
fn incomplete_renamed_shard_cannot_claim_complete_source() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("renamed.gguf");
    fixture(&source, &[tensor("first", 0)], Some((0, 2, 2)));
    let out = temp.path().join("package");
    assert!(
        write(&source, &out, false)
            .unwrap_err()
            .to_string()
            .contains("incomplete source shard")
    );
    assert!(!out.join("model-package.json").exists());
}

#[test]
fn incorrect_declared_total_and_duplicate_source_names_fail() {
    let temp = tempfile::tempdir().unwrap();
    let first = temp.path().join("model-00001-of-00002.gguf");
    let second = temp.path().join("model-00002-of-00002.gguf");
    fixture(&first, &[tensor("first", 0)], Some((0, 2, 3)));
    fixture(&second, &[tensor("second", 0)], Some((1, 2, 3)));
    let out = temp.path().join("package");
    assert!(
        write(&first, &out, false)
            .unwrap_err()
            .to_string()
            .contains("incomplete source tensor")
    );
    fixture(&second, &[tensor("first", 0)], Some((1, 2, 2)));
    assert!(
        write(&first, &out, false)
            .unwrap_err()
            .to_string()
            .contains("duplicate source tensor")
    );
    assert!(!out.join("model-package.json").exists());
}

#[test]
fn resumed_artifacts_cannot_self_certify_missing_or_substituted_tensors() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source.gguf");
    fixture(&source, &[tensor("first", 0), tensor("second", 32)], None);
    let out = temp.path().join("package");
    write(&source, &out, false).unwrap();
    fs::remove_file(out.join("model-package.json")).unwrap();
    let artifact = out.join("shared/common.gguf");
    for tensors in [
        vec![tensor("first", 0)],
        vec![tensor("first", 0), tensor("replacement", 32)],
        vec![
            tensor("first", 0),
            FixtureTensor {
                dimensions: vec![4],
                ..tensor("second", 32)
            },
        ],
        vec![
            tensor("first", 0),
            FixtureTensor {
                dtype: 1,
                ..tensor("second", 32)
            },
        ],
    ] {
        fixture(&artifact, &tensors, None);
        assert!(write(&source, &out, true).is_err());
        assert!(!out.join("model-package.json").exists());
    }
}

#[test]
fn corruption_truncation_and_bad_offsets_fail_before_manifest() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source.gguf");
    fixture(&source, &[tensor("first", 0), tensor("second", 32)], None);
    let out = temp.path().join("package");
    write(&source, &out, false).unwrap();
    fs::remove_file(out.join("model-package.json")).unwrap();
    let artifact = out.join("shared/common.gguf");
    let original = fs::read(&artifact).unwrap();
    let mut corrupted = original.clone();
    let second_offset = read_gguf_catalog(&artifact)
        .unwrap()
        .tensors
        .into_iter()
        .find(|tensor| tensor.name == "second")
        .unwrap()
        .data_offset;
    corrupted[second_offset as usize] ^= 1;
    fs::write(&artifact, &corrupted).unwrap();
    assert!(
        write(&source, &out, true)
            .unwrap_err()
            .to_string()
            .contains("differs")
    );
    fs::write(&artifact, &original[..original.len() - 24]).unwrap();
    assert!(write(&source, &out, true).is_err());
    fixture(&artifact, &[tensor("first", 0), tensor("second", 1)], None);
    assert!(write(&source, &out, true).is_err());
    assert!(!out.join("model-package.json").exists());
}

#[test]
fn native_quantized_extent_is_not_guessed_from_padding() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("quantized.gguf");
    fixture(
        &source,
        &[FixtureTensor {
            name: "quant",
            dimensions: vec![32],
            dtype: 8,
            offset: 0,
        }],
        None,
    );
    let mut bytes = fs::read(&source).unwrap();
    bytes.extend_from_slice(&[0; 32]);
    fs::write(&source, bytes).unwrap();
    let out = temp.path().join("package");
    write(&source, &out, false).unwrap();
    let manifest = read_manifest(&out);
    assert!(matches!(
        manifest.tensor_catalog.entries[0].storage,
        TensorStorage::Owned {
            stored_length: 34,
            ..
        }
    ));
}

#[test]
fn shared_offsets_fail_closed_until_native_inspection_supports_aliases() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("aliases.gguf");
    fixture(&source, &[tensor("first", 0), tensor("alias", 0)], None);
    let out = temp.path().join("package");
    assert!(write(&source, &out, false).is_err());
    assert!(!out.join("model-package.json").exists());
}

#[test]
fn missing_independent_source_prevents_resume_certification() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source.gguf");
    fixture(&source, &[tensor("first", 0)], None);
    let input = resolve_package_input(source.display().to_string(), explicit(&source)).unwrap();
    fs::remove_file(&source).unwrap();
    assert!(SourceInventory::read(&input).is_err());
}

#[test]
fn resumed_hard_linked_source_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source.gguf");
    fixture(&source, &[tensor("first", 0)], None);
    let out = temp.path().join("package");
    let artifact = out.join("shared/common.gguf");
    fs::create_dir_all(artifact.parent().unwrap()).unwrap();
    fs::hard_link(&source, &artifact).unwrap();

    let error = write(&source, &out, true).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("package artifact must not be the independent source file")
    );
    assert!(!out.join("model-package.json").exists());
}

#[test]
fn refuses_transform_hooks_and_existing_completion_marker() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source.gguf");
    fixture(&source, &[tensor("first", 0)], None);
    let out = temp.path().join("package");
    let result = write_package(
        source.display().to_string(),
        out.clone(),
        Vec::new(),
        ArtifactHook { command: None },
        ArtifactHook {
            command: Some("must-not-run".into()),
        },
        explicit(&source),
        false,
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("transform the independent source")
    );
    write(&source, &out, false).unwrap();
    let old = fs::read(out.join("model-package.json")).unwrap();
    assert!(write(&source, &out, true).is_err());
    assert_eq!(old, fs::read(out.join("model-package.json")).unwrap());
}

#[test]
fn verified_resume_and_projector_sidecar_round_trip() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source.gguf");
    fixture(&source, &[tensor("first", 0)], None);
    let projector = temp.path().join("projector.gguf");
    fixture(&projector, &[tensor("vision", 0)], None);
    let out = temp.path().join("package");
    write(&source, &out, false).unwrap();
    fs::remove_file(out.join("model-package.json")).unwrap();
    write_package(
        source.display().to_string(),
        out.clone(),
        vec![projector],
        ArtifactHook { command: None },
        ArtifactHook { command: None },
        explicit(&source),
        true,
    )
    .unwrap();
    let manifest = read_manifest(&out);
    manifest.validate().unwrap();
    assert_eq!(manifest.tensor_catalog.entries.len(), 1);
    assert_eq!(manifest.sidecars.len(), 1);
    assert_eq!(
        manifest.sidecars[0].kind,
        skippy_package_format::SidecarKind::Mmproj
    );
    assert_eq!(
        manifest.sidecars[0].name.as_deref(),
        Some("projector-00000")
    );
    assert_eq!(manifest.artifact_catalog.entries.len(), 3);
}

#[cfg(unix)]
#[test]
fn upload_hook_can_delete_verified_copies_without_losing_inventory() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source.gguf");
    fixture(&source, &[tensor("first", 0), tensor("second", 32)], None);
    let hook = temp.path().join("upload.sh");
    fs::write(
        &hook,
        "#!/bin/sh\nset -eu\nrm -- \"$SKIPPY_PACKAGE_ARTIFACT_PATH\"\n",
    )
    .unwrap();
    fs::set_permissions(&hook, fs::Permissions::from_mode(0o755)).unwrap();
    let out = temp.path().join("package");
    write_package(
        source.display().to_string(),
        out.clone(),
        Vec::new(),
        ArtifactHook {
            command: Some(hook),
        },
        ArtifactHook { command: None },
        explicit(&source),
        false,
    )
    .unwrap();
    let manifest: PackageManifest =
        serde_json::from_slice(&fs::read(out.join("model-package.json")).unwrap()).unwrap();
    manifest.validate_root().unwrap();
    assert_eq!(manifest.artifact_catalog.entries.len(), 2);
    assert!(
        manifest
            .artifact_catalog
            .entries
            .iter()
            .all(|artifact| !out.join(&artifact.path).exists())
    );
    assert!(source.exists());
}
