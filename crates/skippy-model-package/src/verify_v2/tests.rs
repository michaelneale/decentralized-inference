use super::*;
use crate::cli::{Args, Command};
use crate::package::ArtifactHook;
use crate::package_v2::write_package;
use crate::test_gguf::{FixtureTensor, explicit, fixture, tensor};
use clap::Parser;

struct Case {
    _temp: tempfile::TempDir,
    source: PathBuf,
    package: PathBuf,
}

impl Case {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source.gguf");
        fixture(
            &source,
            &[tensor("unknown", 0), tensor("blk.1.anything", 32)],
            None,
        );
        let package = temp.path().join("package with spaces");
        Self {
            _temp: temp,
            source,
            package,
        }
    }
    fn write(&self, projectors: Vec<PathBuf>) {
        write_package(
            self.source.display().to_string(),
            self.package.clone(),
            projectors,
            ArtifactHook { command: None },
            ArtifactHook { command: None },
            explicit(&self.source),
            false,
        )
        .unwrap();
    }
    fn manifest(&self) -> PackageManifest {
        serde_json::from_slice(&fs::read(self.package.join("model-package.json")).unwrap()).unwrap()
    }
    fn save(&self, mut manifest: PackageManifest) {
        manifest.package_id = manifest.computed_package_id().unwrap();
        fs::write(
            self.package.join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }
    fn verify(&self) -> Result<VerificationReport> {
        verify_package(&self.package, &self.source, None, &[])
    }
    fn artifact(&self) -> PathBuf {
        self.package.join("artifacts/source-00000.gguf")
    }
}

#[test]
fn verifies_whole_shard_package_read_only_and_accepts_catalog_reordering() {
    let case = Case::new();
    case.write(vec![]);
    let before = fs::read(case.package.join("model-package.json")).unwrap();
    let report = case.verify().unwrap();
    assert!(report.source_completeness_verified);
    assert_eq!(
        (
            report.checked_source_files,
            report.checked_artifacts,
            report.checked_tensors
        ),
        (1, 1, 2)
    );
    assert_eq!(
        before,
        fs::read(case.package.join("model-package.json")).unwrap()
    );
    let mut manifest = case.manifest();
    manifest.tensor_catalog.entries.reverse();
    manifest.artifact_catalog.entries[0].id = "renamed-container".into();
    for t in &mut manifest.tensor_catalog.entries {
        if let TensorStorage::Owned { artifact_id, .. } = &mut t.storage {
            *artifact_id = "renamed-container".into();
        }
    }
    case.save(manifest);
    case.verify().unwrap();
}

#[test]
fn verifies_complete_multishard_source_and_rejects_missing_or_renamed_partial_source() {
    let mut case = Case::new();
    case.source = case._temp.path().join("model-00002-of-00002.gguf");
    let first = case._temp.path().join("model-00001-of-00002.gguf");
    fixture(&first, &[tensor("first", 0)], Some((0, 2, 2)));
    fixture(&case.source, &[tensor("second", 0)], Some((1, 2, 2)));
    case.write(vec![]);
    let mut manifest = case.manifest();
    manifest.source_model.files.reverse();
    manifest.artifact_catalog.entries.reverse();
    case.save(manifest);
    assert_eq!(case.verify().unwrap().checked_source_files, 2);
    fs::remove_file(first).unwrap();
    assert!(case.verify().is_err());
    let renamed = case._temp.path().join("renamed.gguf");
    fs::copy(&case.source, &renamed).unwrap();
    assert!(verify_package(&case.package, &renamed, None, &[]).is_err());
}

#[test]
fn rejects_self_consistent_but_incomplete_or_substituted_manifest_catalogs() {
    let case = Case::new();
    case.write(vec![]);
    let original = case.manifest();
    for mutation in 0..9 {
        let mut m = original.clone();
        match mutation {
            0 => {
                m.tensor_catalog.entries.pop();
            }
            1 => {
                m.tensor_catalog.entries[0].name = "substituted".into();
            }
            2 => {
                m.tensor_catalog.entries[0].id = "substituted".into();
            }
            3 => {
                m.tensor_catalog.entries[0].ggml_type = 1;
            }
            4 => {
                m.tensor_catalog.entries[0].dimensions = vec![4];
            }
            5 => {
                m.tensor_catalog.entries[0].layer_ordinal = Some(0);
            }
            6 => {
                if let TensorStorage::Owned { stored_length, .. } =
                    &mut m.tensor_catalog.entries[0].storage
                {
                    *stored_length = 8;
                }
            }
            7 => {
                if let TensorStorage::Owned { data_offset, .. } =
                    &mut m.tensor_catalog.entries[0].storage
                {
                    *data_offset += 32;
                }
            }
            _ => {
                m.tensor_catalog
                    .entries
                    .push(m.tensor_catalog.entries[0].clone());
            }
        }
        case.save(m);
        assert!(case.verify().is_err(), "mutation {mutation}");
    }
}

#[test]
fn rejects_forged_source_identity_metadata_and_primary_digest() {
    let case = Case::new();
    case.write(vec![]);
    let original = case.manifest();
    for mutation in 0..6 {
        let mut m = original.clone();
        match mutation {
            0 => m.source_model.files[0].sha256 = "0".repeat(64),
            1 => m.source_model.files[0].byte_size += 1,
            2 => m.source_model.sha256 = "0".repeat(64),
            3 => m.layer_count += 1,
            4 => {
                m.model_metadata
                    .insert("llama.block_count".into(), 99.into());
            }
            _ => m.source_model.files.push(m.source_model.files[0].clone()),
        }
        case.save(m);
        assert!(case.verify().is_err(), "mutation {mutation}");
    }
}

#[test]
fn rejects_corrupt_truncated_missing_and_rehashed_substituted_artifacts() {
    let case = Case::new();
    case.write(vec![]);
    let original = fs::read(case.artifact()).unwrap();
    let mut corrupted = original.clone();
    *corrupted.last_mut().unwrap() ^= 1;
    fs::write(case.artifact(), corrupted).unwrap();
    assert!(case.verify().unwrap_err().to_string().contains("SHA-256"));
    fs::write(case.artifact(), &original[..original.len() - 1]).unwrap();
    assert!(
        case.verify()
            .unwrap_err()
            .to_string()
            .contains("size mismatch")
    );
    fixture(&case.artifact(), &[tensor("omission", 0)], None);
    let mut manifest = case.manifest();
    manifest.artifact_catalog.entries[0].sha256 = file_sha256(&case.artifact()).unwrap();
    manifest.artifact_catalog.entries[0].byte_size = fs::metadata(case.artifact()).unwrap().len();
    manifest.tensor_catalog.entries = inspect(&case.artifact(), "source-00000").unwrap().1.entries;
    case.save(manifest);
    assert!(
        case.verify().is_err(),
        "output cannot redefine source expectation"
    );
    fs::remove_file(case.artifact()).unwrap();
    assert!(case.verify().is_err());
}

#[test]
fn missing_source_and_package_as_its_own_source_reject() {
    let case = Case::new();
    case.write(vec![]);
    assert!(verify_package(&case.package, &case.artifact(), Some("source.gguf"), &[]).is_err());
    fs::remove_file(&case.source).unwrap();
    assert!(case.verify().is_err());
}

#[test]
fn hard_linked_source_is_not_independent() {
    let case = Case::new();
    case.write(vec![]);
    fs::remove_file(case.artifact()).unwrap();
    fs::hard_link(&case.source, case.artifact()).unwrap();
    assert!(
        case.verify()
            .unwrap_err()
            .to_string()
            .contains("own independent source")
    );
}

#[test]
fn rejects_v1_unknown_fields_wrong_package_id_and_duplicate_artifacts() {
    let case = Case::new();
    case.write(vec![]);
    let original = case.manifest();
    let mut m = original.clone();
    m.schema_version = 1;
    case.save(m);
    assert!(case.verify().is_err());
    let mut m = original.clone();
    m.artifact_catalog
        .entries
        .push(m.artifact_catalog.entries[0].clone());
    case.save(m);
    assert!(case.verify().is_err());
    let mut m = original;
    m.package_id = format!("sha256:{}", "0".repeat(64));
    fs::write(
        case.package.join("model-package.json"),
        serde_json::to_vec(&m).unwrap(),
    )
    .unwrap();
    assert!(case.verify().is_err());
    case.save(m);
    let mut value = serde_json::to_value(case.manifest()).unwrap();
    value["unknown"] = true.into();
    fs::write(
        case.package.join("model-package.json"),
        serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    assert!(case.verify().is_err());
}

#[test]
fn rejects_unproven_alias_claims_and_native_shared_offsets() {
    let case = Case::new();
    case.write(vec![]);
    let mut m = case.manifest();
    m.tensor_catalog.entries[1].storage = TensorStorage::Alias {
        target_tensor_id: m.tensor_catalog.entries[0].id.clone(),
    };
    case.save(m);
    assert!(
        case.verify()
            .unwrap_err()
            .to_string()
            .contains("alias claims")
    );
    fixture(
        &case.source,
        &[tensor("first", 0), tensor("alias", 0)],
        None,
    );
    assert!(SourceInventory::read_local(&case.source, "source.gguf").is_err());
}

#[test]
fn projectors_require_matching_independent_sources_and_integrity() {
    let case = Case::new();
    let projector = case._temp.path().join("projector.gguf");
    fixture(&projector, &[tensor("vision", 0)], None);
    case.write(vec![projector.clone()]);
    assert!(case.verify().is_err());
    let report = verify_package(
        &case.package,
        &case.source,
        None,
        std::slice::from_ref(&projector),
    )
    .unwrap();
    assert_eq!(report.checked_projectors, 1);
    let mut m = case.manifest();
    m.sidecars.push(m.sidecars[0].clone());
    case.save(m);
    assert!(
        verify_package(
            &case.package,
            &case.source,
            None,
            &[projector.clone(), projector.clone()]
        )
        .is_err()
    );
    let mut m = case.manifest();
    m.sidecars.pop();
    case.save(m);
    fixture(&projector, &[tensor("substituted", 0)], None);
    assert!(verify_package(&case.package, &case.source, None, &[projector]).is_err());
}

#[cfg(unix)]
#[test]
fn rejects_symlink_escapes_and_symlinked_source_evidence() {
    use std::os::unix::fs::symlink;
    let case = Case::new();
    case.write(vec![]);
    fs::remove_file(case.artifact()).unwrap();
    symlink(&case.source, case.artifact()).unwrap();
    assert!(
        case.verify()
            .unwrap_err()
            .to_string()
            .contains("escapes package")
    );
    fs::remove_file(case.artifact()).unwrap();
    fs::copy(&case.source, case.artifact()).unwrap();
    let alias = case._temp.path().join("outside.gguf");
    symlink(case.artifact(), &alias).unwrap();
    assert!(verify_package(&case.package, &alias, Some("source.gguf"), &[]).is_err());
}

#[test]
fn quantized_stored_size_is_verified_against_native_inventory() {
    let case = Case::new();
    fixture(
        &case.source,
        &[FixtureTensor {
            name: "quant",
            dimensions: vec![32],
            dtype: 8,
            offset: 0,
        }],
        None,
    );
    let mut bytes = fs::read(&case.source).unwrap();
    bytes.extend_from_slice(&[0; 32]);
    fs::write(&case.source, bytes).unwrap();
    case.write(vec![]);
    case.verify().unwrap();
    let mut m = case.manifest();
    if let TensorStorage::Owned { stored_length, .. } = &mut m.tensor_catalog.entries[0].storage {
        *stored_length = 64;
    }
    case.save(m);
    assert!(case.verify().is_err());
}

#[test]
fn verifier_cli_requires_source_and_dispatches_success_and_failure() {
    assert!(
        Args::try_parse_from(["skippy-model-package", "verify-package-v2", "package"]).is_err()
    );
    let case = Case::new();
    case.write(vec![]);
    let args = Args::try_parse_from([
        "skippy-model-package",
        "verify-package-v2",
        case.package.to_str().unwrap(),
        "--source",
        case.source.to_str().unwrap(),
    ])
    .unwrap();
    assert!(matches!(args.command, Command::VerifyPackageV2 { .. }));
    crate::run(args).unwrap();
    fs::remove_file(&case.source).unwrap();
    assert!(
        crate::run(Args {
            command: Command::VerifyPackageV2 {
                package: case.package,
                source: case.source,
                source_file: None,
                source_projectors: vec![]
            }
        })
        .is_err()
    );
}

#[test]
fn explicit_logical_source_filename_is_required_for_relocated_source() {
    let case = Case::new();
    case.write(vec![]);
    let renamed = case._temp.path().join("renamed.gguf");
    fs::rename(&case.source, &renamed).unwrap();
    assert!(verify_package(&case.package, &renamed, None, &[]).is_err());
    verify_package(&case.package, &renamed, Some("source.gguf"), &[]).unwrap();
}

#[test]
fn rejects_unaccounted_artifacts_and_artifact_path_traversal() {
    let case = Case::new();
    case.write(vec![]);
    let original = case.manifest();
    let extra = case.package.join("extra.gguf");
    fixture(&extra, &[tensor("extra", 0)], None);
    let mut m = original.clone();
    m.artifact_catalog.entries.push(Artifact {
        id: "unaccounted".into(),
        path: "extra.gguf".into(),
        byte_size: fs::metadata(&extra).unwrap().len(),
        sha256: file_sha256(&extra).unwrap(),
    });
    case.save(m);
    assert!(
        case.verify()
            .unwrap_err()
            .to_string()
            .contains("unaccounted")
    );
    let mut m = original;
    m.artifact_catalog.entries[0].path = "../source.gguf".into();
    case.save(m);
    assert!(case.verify().is_err());
}

#[test]
fn verifies_projectors_by_content_and_rejects_corrupt_projector_copy() {
    let case = Case::new();
    let first = case._temp.path().join("vision.gguf");
    let second = case._temp.path().join("audio.gguf");
    fixture(&first, &[tensor("vision", 0)], None);
    fixture(&second, &[tensor("audio", 0)], None);
    case.write(vec![first.clone(), second.clone()]);
    let originals = vec![second, first];
    assert_eq!(
        verify_package(&case.package, &case.source, None, &originals)
            .unwrap()
            .checked_projectors,
        2
    );
    let path = case.package.join("projectors/projector-00000.gguf");
    let mut bytes = fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(path, bytes).unwrap();
    assert!(
        verify_package(&case.package, &case.source, None, &originals)
            .unwrap_err()
            .to_string()
            .contains("SHA-256")
    );
}
