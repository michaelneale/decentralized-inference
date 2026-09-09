use std::collections::BTreeSet;

#[path = "parity_models/mod.rs"]
mod harness;

use harness::{
    FULL_LATTICE_MAX_LAYERS, activation_handoff_matches_full_model, assert_manifest_row_complete,
    cache_state_restore_matches_recompute, graph_boundary_contract_matches_stage_roles,
    mixed_iteration_matches_serial, p0_p1_manifest_rows, representative_cut_lattice,
    representative_cut_lattice_matches_full_model,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct FamilySpec {
    priority: &'static str,
    llama_model: &'static str,
    family: &'static str,
}

macro_rules! family_module {
    ($module:ident, $priority:literal, $llama_model:literal, $family:literal) => {
        mod $module {
            const SPEC: super::FamilySpec = super::FamilySpec {
                priority: $priority,
                llama_model: $llama_model,
                family: $family,
            };

            #[test]
            fn manifest_row_is_complete() {
                super::assert_manifest_row_complete(SPEC).unwrap();
            }

            #[test]
            #[ignore = "downloads and loads model-family GGUF/package artifacts"]
            fn activation_handoff_matches_full_model() {
                super::activation_handoff_matches_full_model(SPEC).unwrap();
            }

            #[test]
            #[ignore = "downloads and loads model-family GGUF/package artifacts"]
            fn cache_state_restore_matches_recompute() {
                super::cache_state_restore_matches_recompute(SPEC).unwrap();
            }

            #[test]
            #[ignore = "downloads and loads model-family GGUF/package artifacts"]
            fn mixed_iteration_matches_serial() {
                super::mixed_iteration_matches_serial(SPEC).unwrap();
            }
        }
    };
}

family_module!(p0_llama_llama, "p0", "llama", "llama");

#[test]
fn representative_cut_lattice_covers_both_coverage_modes() {
    let reviewed = representative_cut_lattice(16, Some((5, 11))).unwrap();
    assert_eq!(reviewed.first(), Some(&(1, 2)));
    assert_eq!(reviewed.last(), Some(&(14, 15)));
    assert!(
        reviewed.contains(&(5, 11)),
        "reviewed pair must be included"
    );
    assert_eq!(
        reviewed.len(),
        reviewed.iter().collect::<BTreeSet<_>>().len()
    );
    let expected: BTreeSet<(u32, u32)> = (1..15)
        .flat_map(|s| [(s, s + 1), (1, s + 1), (s, 15)])
        .chain(std::iter::once((5, 11)))
        .collect();
    assert_eq!(reviewed, expected.into_iter().collect::<Vec<_>>());

    let full = representative_cut_lattice(FULL_LATTICE_MAX_LAYERS, None).unwrap();
    let full_expected: BTreeSet<(u32, u32)> = (1..FULL_LATTICE_MAX_LAYERS - 1)
        .flat_map(|s1| ((s1 + 1)..FULL_LATTICE_MAX_LAYERS).map(move |s2| (s1, s2)))
        .collect();
    assert_eq!(full.len(), full_expected.len());
    assert_eq!(full, full_expected.into_iter().collect::<Vec<_>>());
    assert_eq!(full.len(), 55); // (12-1)(12-2)/2
    assert_eq!(full.first(), Some(&(1, 2)));
    assert_eq!(full.last(), Some(&(10, 11)));

    assert!(representative_cut_lattice(2, None).is_err());
    assert!(representative_cut_lattice(0, None).is_err());
    let dropped = representative_cut_lattice(6, Some((3, 9))).unwrap();
    assert!(!dropped.contains(&(3, 9)));
}

#[test]
#[ignore = "downloads and loads model-family GGUF artifacts"]
fn p0_llama_representative_cut_lattice_matches_full_model() {
    representative_cut_lattice_matches_full_model(FamilySpec {
        priority: "p0",
        llama_model: "llama",
        family: "llama",
    })
    .unwrap();
}

#[test]
fn manifest_row_is_complete_for_p0_llama_fixture() {
    assert_manifest_row_complete(FamilySpec {
        priority: "p0",
        llama_model: "llama",
        family: "llama",
    })
    .unwrap();
}

family_module!(p0_qwen2_qwen2, "p0", "qwen2", "qwen2");
family_module!(p0_qwen3_qwen3_dense, "p0", "qwen3", "qwen3_dense");

#[test]
#[ignore = "downloads and loads the pinned Qwen3 GGUF artifact"]
fn qwen3_graph_boundary_contract_matches_first_middle_and_final_stage_roles() {
    graph_boundary_contract_matches_stage_roles(FamilySpec {
        priority: "p0",
        llama_model: "qwen3",
        family: "qwen3_dense",
    })
    .unwrap();
}

#[test]
#[ignore = "downloads and loads the pinned Gemma3n GGUF artifact"]
fn gemma3n_graph_boundary_contract_matches_first_middle_and_final_stage_roles() {
    graph_boundary_contract_matches_stage_roles(FamilySpec {
        priority: "p0",
        llama_model: "gemma3n",
        family: "gemma3n",
    })
    .unwrap();
}

#[test]
#[ignore = "loads the remote-only Qwen4Exp GGUF artifact"]
fn qwen4exp_graph_boundary_contract_matches_first_middle_and_final_stage_roles() {
    graph_boundary_contract_matches_stage_roles(FamilySpec {
        priority: "p2",
        llama_model: "qwen4exp",
        family: "qwen4exp",
    })
    .unwrap();
}
family_module!(p0_qwen3next_qwen3next, "p0", "qwen3next", "qwen3next");
family_module!(p0_qwen35moe_qwen35moe, "p0", "qwen35moe", "qwen35moe");
family_module!(p0_qwen2vl_qwen2vl, "p0", "qwen2vl", "qwen2vl");
family_module!(p0_qwen3vl_qwen3vl, "p0", "qwen3vl", "qwen3vl");
family_module!(p0_qwen2moe_qwen2moe, "p0", "qwen2moe", "qwen2moe");
family_module!(p0_qwen3moe_qwen3moe, "p0", "qwen3moe", "qwen3moe");
family_module!(p0_mistral3_mistral, "p0", "mistral3", "mistral");
family_module!(p0_mistral4_mistral4, "p0", "mistral4", "mistral4");
family_module!(p0_phi3_phi, "p0", "phi3", "phi");
family_module!(p0_phi2_phi2, "p0", "phi2", "phi2");
family_module!(p0_phimoe_phimoe, "p0", "phimoe", "phimoe");
family_module!(p0_gemma_gemma, "p0", "gemma", "gemma");
family_module!(p0_gemma2_gemma2, "p0", "gemma2", "gemma2");
family_module!(p0_gemma3_gemma3, "p0", "gemma3", "gemma3");
family_module!(p0_gemma4_gemma4_a4b, "p0", "gemma4", "gemma4_a4b");
family_module!(p0_gemma4_gemma4_e4b, "p0", "gemma4", "gemma4_e4b");
family_module!(p0_gemma3n_gemma3n, "p0", "gemma3n", "gemma3n");
family_module!(p0_deepseek_deepseek, "p0", "deepseek", "deepseek");
family_module!(p0_deepseek2_deepseek2, "p0", "deepseek2", "deepseek2");
family_module!(p0_deepseek2_deepseek3, "p0", "deepseek2", "deepseek3");
family_module!(p0_glm4_glm4, "p0", "glm4", "glm4");
family_module!(p0_glm4_glm47_flash, "p0", "glm4", "glm47_flash");
family_module!(p0_glm4_moe_glm4_moe, "p0", "glm4-moe", "glm4_moe");
family_module!(p0_granite_granite, "p0", "granite", "granite");
family_module!(
    p0_granite_hybrid_granite_hybrid,
    "p0",
    "granite-hybrid",
    "granite_hybrid"
);
family_module!(p0_command_r_command_r, "p0", "command-r", "command_r");
family_module!(p0_cohere2_cohere2, "p0", "cohere2", "cohere2");
family_module!(p0_minimax_m2_minimax_m27, "p0", "minimax-m2", "minimax_m27");
family_module!(p0_lfm2_lfm2, "p0", "lfm2", "lfm2");
family_module!(p0_lfm2_lfm2_vl, "p0", "lfm2", "lfm2_vl");
family_module!(
    p0_hunyuan_dense_hunyuan_dense,
    "p0",
    "hunyuan-dense",
    "hunyuan_dense"
);
family_module!(
    p0_hunyuan_moe_hunyuan_moe,
    "p0",
    "hunyuan-moe",
    "hunyuan_moe"
);
family_module!(p0_hunyuan_vl_hunyuan_vl, "p0", "hunyuan-vl", "hunyuan_vl");
family_module!(p0_llama4_llama4, "p0", "llama4", "llama4");
family_module!(p0_lfm2moe_lfm2moe, "p0", "lfm2moe", "lfm2moe");
family_module!(p0_openai_moe_openai_moe, "p0", "openai-moe", "openai_moe");
family_module!(p0_qwen_qwen, "p0", "qwen", "qwen");
family_module!(p0_qwen35_qwen35, "p0", "qwen35", "qwen35");
family_module!(p0_qwen3vlmoe_qwen3vlmoe, "p0", "qwen3vlmoe", "qwen3vlmoe");

family_module!(p1_gptneox_gptneox, "p1", "gptneox", "gptneox");
family_module!(p1_bloom_bloom, "p1", "bloom", "bloom");
family_module!(p1_baichuan_baichuan, "p1", "baichuan", "baichuan");
family_module!(p1_jamba_jamba, "p1", "jamba", "jamba");
family_module!(p1_mamba_mamba, "p1", "mamba", "mamba");
family_module!(p1_mamba2_mamba2, "p1", "mamba2", "mamba2");
family_module!(p1_rwkv6_rwkv6, "p1", "rwkv6", "rwkv6");
family_module!(p1_rwkv7_rwkv7, "p1", "rwkv7", "rwkv7");
family_module!(p1_arwkv7_rwkv7, "p1", "arwkv7", "rwkv7");
family_module!(p1_falcon_h1_falcon_h1, "p1", "falcon-h1", "falcon_h1");
family_module!(p1_falcon_falcon, "p1", "falcon", "falcon");
family_module!(p1_olmo_olmo, "p1", "olmo", "olmo");
family_module!(p1_olmo2_olmo2, "p1", "olmo2", "olmo2");
family_module!(p1_olmoe_olmoe, "p1", "olmoe", "olmoe");
family_module!(p1_internlm2_internlm2, "p1", "internlm2", "internlm2");
family_module!(p1_exaone_exaone, "p1", "exaone", "exaone");
family_module!(p1_exaone4_exaone4, "p1", "exaone4", "exaone4");
family_module!(p1_starcoder2_starcoder2, "p1", "starcoder2", "starcoder2");
family_module!(p1_stablelm_stablelm, "p1", "stablelm", "stablelm");
family_module!(p1_gpt2_gpt2, "p1", "gpt2", "gpt2");
family_module!(p1_mpt_mpt, "p1", "mpt", "mpt");
family_module!(p1_arcee_arcee, "p1", "arcee", "arcee");
family_module!(p1_bitnet_bitnet, "p1", "bitnet", "bitnet");
family_module!(p1_chatglm_chatglm, "p1", "chatglm", "chatglm");
family_module!(p1_codeshell_codeshell, "p1", "codeshell", "codeshell");
family_module!(
    p1_deepseek2ocr_deepseek2ocr,
    "p1",
    "deepseek2ocr",
    "deepseek2ocr"
);
family_module!(p1_ernie4_5_ernie4_5, "p1", "ernie4-5", "ernie4_5");
family_module!(
    p1_ernie4_5_moe_ernie4_5_moe,
    "p1",
    "ernie4-5-moe",
    "ernie4_5_moe"
);
family_module!(p1_exaone_moe_exaone_moe, "p1", "exaone-moe", "exaone_moe");
family_module!(
    p1_kimi_linear_kimi_linear,
    "p1",
    "kimi-linear",
    "kimi_linear"
);
family_module!(p1_minicpm_minicpm, "p1", "minicpm", "minicpm");
family_module!(p1_minicpm3_minicpm3, "p1", "minicpm3", "minicpm3");
family_module!(p1_nemotron_nemotron, "p1", "nemotron", "nemotron");
family_module!(p1_nemotron_h_nemotron_h, "p1", "nemotron-h", "nemotron_h");
family_module!(
    p1_nemotron_h_moe_nemotron_h_moe,
    "p1",
    "nemotron-h-moe",
    "nemotron_h_moe"
);
family_module!(p1_seed_oss_seed_oss, "p1", "seed-oss", "seed_oss");
family_module!(
    p1_smallthinker_smallthinker,
    "p1",
    "smallthinker",
    "smallthinker"
);
family_module!(p1_smollm3_smollm3, "p1", "smollm3", "smollm3");
family_module!(p1_starcoder_starcoder, "p1", "starcoder", "starcoder");

const FAMILY_SPECS: &[FamilySpec] = &[
    FamilySpec {
        priority: "p0",
        llama_model: "llama",
        family: "llama",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "qwen2",
        family: "qwen2",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "qwen3",
        family: "qwen3_dense",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "qwen3next",
        family: "qwen3next",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "qwen35moe",
        family: "qwen35moe",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "qwen2vl",
        family: "qwen2vl",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "qwen3vl",
        family: "qwen3vl",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "qwen2moe",
        family: "qwen2moe",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "qwen3moe",
        family: "qwen3moe",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "mistral3",
        family: "mistral",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "mistral4",
        family: "mistral4",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "phi3",
        family: "phi",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "phi2",
        family: "phi2",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "phimoe",
        family: "phimoe",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "gemma",
        family: "gemma",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "gemma2",
        family: "gemma2",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "gemma3",
        family: "gemma3",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "gemma4",
        family: "gemma4_a4b",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "gemma4",
        family: "gemma4_e4b",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "gemma3n",
        family: "gemma3n",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "deepseek",
        family: "deepseek",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "deepseek2",
        family: "deepseek2",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "deepseek2",
        family: "deepseek3",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "glm4",
        family: "glm4",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "glm4",
        family: "glm47_flash",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "glm4-moe",
        family: "glm4_moe",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "granite",
        family: "granite",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "granite-hybrid",
        family: "granite_hybrid",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "command-r",
        family: "command_r",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "cohere2",
        family: "cohere2",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "minimax-m2",
        family: "minimax_m27",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "lfm2",
        family: "lfm2",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "lfm2",
        family: "lfm2_vl",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "hunyuan-dense",
        family: "hunyuan_dense",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "hunyuan-moe",
        family: "hunyuan_moe",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "hunyuan-vl",
        family: "hunyuan_vl",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "llama4",
        family: "llama4",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "lfm2moe",
        family: "lfm2moe",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "openai-moe",
        family: "openai_moe",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "qwen",
        family: "qwen",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "qwen35",
        family: "qwen35",
    },
    FamilySpec {
        priority: "p0",
        llama_model: "qwen3vlmoe",
        family: "qwen3vlmoe",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "gptneox",
        family: "gptneox",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "bloom",
        family: "bloom",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "baichuan",
        family: "baichuan",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "jamba",
        family: "jamba",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "mamba",
        family: "mamba",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "mamba2",
        family: "mamba2",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "rwkv6",
        family: "rwkv6",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "rwkv7",
        family: "rwkv7",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "arwkv7",
        family: "rwkv7",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "falcon-h1",
        family: "falcon_h1",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "falcon",
        family: "falcon",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "olmo",
        family: "olmo",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "olmo2",
        family: "olmo2",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "olmoe",
        family: "olmoe",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "internlm2",
        family: "internlm2",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "exaone",
        family: "exaone",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "exaone4",
        family: "exaone4",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "starcoder2",
        family: "starcoder2",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "stablelm",
        family: "stablelm",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "gpt2",
        family: "gpt2",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "mpt",
        family: "mpt",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "arcee",
        family: "arcee",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "bitnet",
        family: "bitnet",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "chatglm",
        family: "chatglm",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "codeshell",
        family: "codeshell",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "deepseek2ocr",
        family: "deepseek2ocr",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "ernie4-5",
        family: "ernie4_5",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "ernie4-5-moe",
        family: "ernie4_5_moe",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "exaone-moe",
        family: "exaone_moe",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "kimi-linear",
        family: "kimi_linear",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "minicpm",
        family: "minicpm",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "minicpm3",
        family: "minicpm3",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "nemotron",
        family: "nemotron",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "nemotron-h",
        family: "nemotron_h",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "nemotron-h-moe",
        family: "nemotron_h_moe",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "seed-oss",
        family: "seed_oss",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "smallthinker",
        family: "smallthinker",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "smollm3",
        family: "smollm3",
    },
    FamilySpec {
        priority: "p1",
        llama_model: "starcoder",
        family: "starcoder",
    },
];

#[test]
fn certified_p0_p1_manifest_rows_all_have_family_modules() {
    let expected = p0_p1_manifest_rows();
    let declared = FAMILY_SPECS
        .iter()
        .map(|spec| {
            (
                spec.priority.to_string(),
                spec.llama_model.to_string(),
                spec.family.to_string(),
            )
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        expected, declared,
        "every certified P0/P1 manifest row must have a dedicated test module"
    );
}
