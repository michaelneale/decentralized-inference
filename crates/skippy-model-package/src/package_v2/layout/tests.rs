use super::*;
use skippy_runtime::TensorInfo;

fn info(name: &str, role: TensorRole, layer: i32) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        layer_index: u32::try_from(layer).ok(),
        role,
        ggml_type: 0,
        byte_size: 16,
        element_count: 4,
    }
}

#[test]
fn assigns_every_role_to_its_canonical_artifact() {
    let planned = plan_artifacts(&[
        info("tokenizer.ggml.tokens", TensorRole::Tokenizer, -1),
        info("unknown.thing", TensorRole::Unknown, -1),
        info("caption.notes", TensorRole::Metadata, -1),
        info("token_embd.weight", TensorRole::Embedding, -1),
        info("output_norm.weight", TensorRole::FinalNorm, -1),
        info("output.weight", TensorRole::Output, -1),
        info("blk.0.attn_q.weight", TensorRole::Layer, 0),
        info("blk.1.attn_q.weight", TensorRole::Layer, 1),
        info("blk.11.ffn_down.weight", TensorRole::Layer, 11),
    ])
    .unwrap();
    assert_eq!(
        planned
            .iter()
            .map(|artifact| artifact.path.as_str())
            .collect::<Vec<_>>(),
        [
            "shared/common.gguf",
            "shared/embeddings.gguf",
            "shared/output.gguf",
            "layers/layer-00000.gguf",
            "layers/layer-00001.gguf",
            "layers/layer-00011.gguf",
        ]
    );
    let by_id = planned
        .iter()
        .map(|artifact| (artifact.id.as_str(), artifact))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        by_id["common"].tensor_names,
        ["caption.notes", "tokenizer.ggml.tokens", "unknown.thing"]
    );
    assert_eq!(by_id["embeddings"].tensor_names, ["token_embd.weight"]);
    assert_eq!(
        by_id["output"].tensor_names,
        ["output.weight", "output_norm.weight"]
    );
    assert_eq!(by_id["layer-00000"].tensor_names, ["blk.0.attn_q.weight"]);
    assert_eq!(
        by_id["layer-00011"].tensor_names,
        ["blk.11.ffn_down.weight"]
    );
}

#[test]
fn omits_empty_shared_artifacts_but_never_a_layer() {
    let planned = plan_artifacts(&[info("blk.0.attn_q.weight", TensorRole::Layer, 0)]).unwrap();
    assert_eq!(planned.len(), 1);
    assert_eq!(planned[0].path, "layers/layer-00000.gguf");
    assert!(matches!(
        planned[0].kind,
        PlannedArtifactKind::Layer { ordinal: 0 }
    ));
}

#[test]
fn layer_tensor_without_index_fails_closed() {
    assert!(plan_artifacts(&[info("blk.0.attn_q.weight", TensorRole::Layer, -1)]).is_err());
}

#[test]
fn every_tensor_is_bound_exactly_once() {
    let tensors = vec![
        info("tokenizer.ggml.tokens", TensorRole::Tokenizer, -1),
        info("token_embd.weight", TensorRole::Embedding, -1),
        info("output_norm.weight", TensorRole::FinalNorm, -1),
        info("blk.0.attn_q.weight", TensorRole::Layer, 0),
        info("blk.1.attn_q.weight", TensorRole::Layer, 1),
    ];
    let planned = plan_artifacts(&tensors).unwrap();
    let mut bound = planned
        .iter()
        .flat_map(|artifact| &artifact.tensor_names)
        .cloned()
        .collect::<Vec<_>>();
    bound.sort();
    let mut expected = tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(bound, expected);
}
