use serde::Deserialize;
use std::{env, fs, path::PathBuf};

#[derive(Debug, Deserialize)]
struct CatalogFile {
    model: Vec<CatalogEntry>,
}

#[derive(Debug, Deserialize)]
struct CatalogEntry {
    id: String,
    file: String,
    url: String,
    size: String,
    description: String,
    draft: Option<String>,
    moe: Option<MoeEntry>,
    #[serde(default)]
    extra_files: Vec<AssetEntry>,
    mmproj: Option<AssetEntry>,
}

#[derive(Debug, Deserialize)]
struct AssetEntry {
    file: String,
    url: String,
}

#[derive(Debug, Deserialize)]
struct MoeEntry {
    n_expert: u32,
    n_expert_used: u32,
    min_experts_per_node: u32,
    #[serde(default)]
    ranking: Vec<u32>,
}

fn main() {
    println!("cargo:rerun-if-changed=ui/dist");
    println!("cargo:rerun-if-changed=models/metadata.toml");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let ui_dist = manifest_dir.join("ui").join("dist");
    if !ui_dist.exists() {
        fs::create_dir_all(&ui_dist).expect("create ui/dist");
    }
    let metadata_path = manifest_dir.join("models").join("metadata.toml");
    let raw = fs::read_to_string(&metadata_path).expect("read models/metadata.toml");
    let parsed: CatalogFile = toml::from_str(&raw).expect("parse models/metadata.toml");
    let generated = render_catalog(&parsed.model);

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    fs::write(out_dir.join("model_catalog.rs"), generated).expect("write generated catalog");
}

fn render_catalog(models: &[CatalogEntry]) -> String {
    let mut out = String::new();
    out.push_str("pub static CURATED_MODELS: &[CuratedModel] = &[\n");
    for model in models {
        let (source_repo, source_revision, source_file) = parse_huggingface_source(&model.url);
        out.push_str("    CuratedModel {\n");
        out.push_str(&format!("        name: {:?},\n", model.id));
        out.push_str(&format!("        id: {:?},\n", model.id));
        out.push_str(&format!("        file: {:?},\n", model.file));
        out.push_str(&format!("        url: {:?},\n", model.url));
        out.push_str(&format!(
            "        source_repo: {},\n",
            render_option_str(source_repo.as_deref())
        ));
        out.push_str(&format!(
            "        source_file: {:?},\n",
            source_file.as_deref().unwrap_or(&model.file)
        ));
        out.push_str(&format!(
            "        source_revision: {},\n",
            render_option_str(source_revision.as_deref())
        ));
        out.push_str(&format!("        size: {:?},\n", model.size));
        out.push_str(&format!("        description: {:?},\n", model.description));
        out.push_str(&format!(
            "        draft: {},\n",
            render_option_str(model.draft.as_deref())
        ));
        out.push_str(&format!(
            "        moe: {},\n",
            render_moe(model.moe.as_ref())
        ));
        out.push_str(&format!(
            "        extra_files: {},\n",
            render_assets(&model.extra_files)
        ));
        out.push_str(&format!(
            "        mmproj: {},\n",
            render_optional_asset(model.mmproj.as_ref())
        ));
        out.push_str("    },\n");
    }
    out.push_str("];\n");
    out
}

fn render_option_str(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("Some({value:?})"),
        None => "None".to_string(),
    }
}

fn render_moe(value: Option<&MoeEntry>) -> String {
    match value {
        Some(value) => format!(
            "Some(MoeConfig {{ n_expert: {}, n_expert_used: {}, min_experts_per_node: {}, ranking: &[{}] }})",
            value.n_expert,
            value.n_expert_used,
            value.min_experts_per_node,
            value.ranking
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        None => "None".to_string(),
    }
}

fn render_assets(assets: &[AssetEntry]) -> String {
    if assets.is_empty() {
        return "&[]".to_string();
    }

    let mut rendered = String::from("&[");
    for asset in assets {
        rendered.push_str(&format!(
            "RemoteAsset {{ file: {:?}, url: {:?} }}, ",
            asset.file, asset.url
        ));
    }
    rendered.push(']');
    rendered
}

fn render_optional_asset(asset: Option<&AssetEntry>) -> String {
    match asset {
        Some(asset) => format!(
            "Some(RemoteAsset {{ file: {:?}, url: {:?} }})",
            asset.file, asset.url
        ),
        None => "None".to_string(),
    }
}

fn parse_huggingface_source(url: &str) -> (Option<String>, Option<String>, Option<String>) {
    let Some(rest) = url.strip_prefix("https://huggingface.co/") else {
        return (None, None, None);
    };
    let parts: Vec<&str> = rest.split('/').collect();
    if parts.len() < 5 || parts.get(2) != Some(&"resolve") {
        return (None, None, None);
    }

    let repo = format!("{}/{}", parts[0], parts[1]);
    let revision = parts.get(3).map(|value| value.to_string());
    let file = parts[4..].join("/");
    (Some(repo), revision, Some(file))
}
