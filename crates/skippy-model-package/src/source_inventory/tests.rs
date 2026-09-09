use super::*;
use skippy_ffi::TensorRole;
use skippy_model::gguf_catalog::GgufTensor;

fn inventory() -> (GgufCatalog, Vec<TensorInfo>) {
    let tensor = GgufTensor {
        name: "base".to_string(),
        dimensions: vec![8, 2],
        ggml_type: 0,
        data_offset: 64,
    };
    let native = TensorInfo {
        name: tensor.name.clone(),
        layer_index: None,
        role: TensorRole::Unknown,
        ggml_type: 0,
        byte_size: 64,
        element_count: 16,
    };
    (
        GgufCatalog {
            version: 3,
            artifact_bytes: 256,
            alignment: 32,
            data_start: 64,
            metadata: BTreeMap::new(),
            tensors: vec![tensor],
        },
        vec![native],
    )
}

#[test]
fn explicit_shared_storage_normalizes_to_one_owned_base() {
    // Test the catalog normalization contract independently: native GGUF opening
    // currently rejects aliases, so this is not evidence of end-to-end ingestion.
    let (mut directory, mut native) = inventory();
    directory.tensors.push(GgufTensor {
        name: "view".to_string(),
        ..directory.tensors[0].clone()
    });
    native.push(TensorInfo {
        name: "view".to_string(),
        ..native[0].clone()
    });
    let catalog = catalog_from_inspection(&directory, &native, "artifact").unwrap();
    assert!(matches!(
        catalog.entries[0].storage,
        TensorStorage::Owned { .. }
    ));
    assert_eq!(
        catalog.entries[1].storage,
        TensorStorage::Alias {
            target_tensor_id: "base".to_string()
        }
    );
    directory.tensors.reverse();
    native.reverse();
    assert_eq!(
        catalog,
        catalog_from_inspection(&directory, &native, "artifact").unwrap()
    );
}

#[test]
fn rejects_partial_overlaps_mismatched_aliases_and_out_of_bounds_storage() {
    let (mut directory, mut native) = inventory();
    directory.tensors.push(GgufTensor {
        name: "view".to_string(),
        data_offset: 96,
        ..directory.tensors[0].clone()
    });
    native.push(TensorInfo {
        name: "view".to_string(),
        ..native[0].clone()
    });
    assert!(catalog_from_inspection(&directory, &native, "artifact").is_err());
    directory.tensors[1].data_offset = 64;
    directory.tensors[1].dimensions = vec![16];
    assert!(catalog_from_inspection(&directory, &native, "artifact").is_err());
    directory.tensors.truncate(1);
    native.truncate(1);
    directory.artifact_bytes = 127;
    assert!(catalog_from_inspection(&directory, &native, "artifact").is_err());
}
