use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use tokio::io::AsyncWriteExt;
use tokio_stream::StreamExt;

#[derive(Clone, Debug)]
pub struct CuratedModel {
    pub name: &'static str,
    pub id: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub source_repo: Option<&'static str>,
    pub source_file: &'static str,
    pub size: &'static str,
    pub description: &'static str,
    pub draft: Option<&'static str>,
    pub moe: Option<MoeConfig>,
    pub extra_files: &'static [RemoteAsset],
    pub mmproj: Option<RemoteAsset>,
}

#[derive(Clone, Debug)]
pub struct RemoteAsset {
    pub file: &'static str,
    pub url: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct MoeConfig {
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub min_experts_per_node: u32,
    pub ranking: &'static [u32],
}

include!(concat!(env!("OUT_DIR"), "/model_catalog.rs"));

#[derive(Clone, Debug)]
struct RuntimeConfig {
    model_dirs: Vec<PathBuf>,
    huggingface_token: Option<String>,
}

static RUNTIME_CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();
static WARNED_NO_HF_TOKEN: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub exact_ref: String,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub curated: Option<&'static CuratedModel>,
}

#[derive(Clone, Debug)]
pub struct ModelDetails {
    pub display_name: String,
    pub exact_ref: String,
    pub source: &'static str,
    pub download_url: String,
    pub size_label: Option<String>,
    pub description: Option<String>,
    pub draft: Option<String>,
    pub vision: bool,
    pub moe: Option<MoeConfig>,
}

#[derive(Clone, Debug)]
pub enum ExactModelRef {
    Curated(&'static CuratedModel),
    HuggingFace { repo: String, file: String },
    Url { url: String, filename: String },
}

#[derive(Debug, Deserialize)]
struct HuggingFaceRepoSummary {
    id: String,
    downloads: Option<u64>,
    likes: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceRepoDetail {
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "modelId")]
    model_id: Option<String>,
    #[serde(default)]
    siblings: Vec<HuggingFaceSibling>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceSibling {
    rfilename: String,
}

pub fn init_runtime(config: &crate::plugin::MeshConfig, cli_dirs: &[PathBuf]) {
    let runtime = RuntimeConfig {
        model_dirs: configured_model_dirs(config, cli_dirs),
        huggingface_token: config.huggingface_token(),
    };
    let _ = RUNTIME_CONFIG.set(runtime);
}

fn runtime_config() -> &'static RuntimeConfig {
    RUNTIME_CONFIG.get_or_init(|| RuntimeConfig {
        model_dirs: default_model_dirs(),
        huggingface_token: std::env::var("HF_TOKEN")
            .ok()
            .filter(|s| !s.trim().is_empty()),
    })
}

fn configured_model_dirs(config: &crate::plugin::MeshConfig, cli_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let raw_dirs = if !cli_dirs.is_empty() {
        cli_dirs.to_vec()
    } else {
        config.model_dirs()
    };
    if raw_dirs.is_empty() {
        return default_model_dirs();
    }

    let mut seen = HashSet::new();
    let mut dirs = Vec::new();
    for dir in raw_dirs {
        let dir = expand_path(&dir);
        if seen.insert(dir.clone()) {
            dirs.push(dir);
        }
    }
    dirs
}

fn default_model_dirs() -> Vec<PathBuf> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let mut dirs = vec![home.join(".models")];
    if let Some(data_dir) = dirs::data_dir() {
        let goose_dir = data_dir.join("Block.goose").join("models");
        if goose_dir.exists() {
            dirs.push(goose_dir);
        }
    }
    dirs
}

fn expand_path(path: &Path) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path.to_path_buf();
    };
    if text == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    }
    if let Some(rest) = text.strip_prefix("~/") {
        return dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(rest);
    }
    path.to_path_buf()
}

pub fn model_dirs() -> Vec<PathBuf> {
    runtime_config().model_dirs.clone()
}

pub fn primary_models_dir() -> PathBuf {
    runtime_config()
        .model_dirs
        .first()
        .cloned()
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn parse_size_gb(s: &str) -> f64 {
    let s = s.trim();
    if let Some(gb) = s.strip_suffix("GB") {
        gb.trim().parse().unwrap_or(0.0)
    } else if let Some(mb) = s.strip_suffix("MB") {
        mb.trim().parse::<f64>().unwrap_or(0.0) / 1000.0
    } else {
        0.0
    }
}

pub fn list_curated_models() {
    eprintln!("Curated models:");
    eprintln!();
    for model in CURATED_MODELS {
        let draft_info = model
            .draft
            .map(|draft| format!(" (draft: {draft})"))
            .unwrap_or_default();
        eprintln!(
            "  {:40} {:>6}  {}{}",
            model.id, model.size, model.description, draft_info
        );
    }
}

pub fn find_curated_model_exact(query: &str) -> Option<&'static CuratedModel> {
    let q = query.to_lowercase();
    CURATED_MODELS.iter().find(|model| {
        model.id.to_lowercase() == q
            || model.file.to_lowercase() == q
            || model.file.trim_end_matches(".gguf").to_lowercase() == q
    })
}

pub fn find_curated_model(query: &str) -> Option<&'static CuratedModel> {
    let q = query.to_lowercase();
    find_curated_model_exact(query).or_else(|| {
        CURATED_MODELS.iter().find(|model| {
            model.id.to_lowercase().contains(&q)
                || model.file.to_lowercase().contains(&q)
                || model.description.to_lowercase().contains(&q)
        })
    })
}

pub fn search_curated_models(query: &str) -> Vec<&'static CuratedModel> {
    let q = query.to_lowercase();
    CURATED_MODELS
        .iter()
        .filter(|model| {
            model.id.to_lowercase().contains(&q)
                || model.file.to_lowercase().contains(&q)
                || model.description.to_lowercase().contains(&q)
        })
        .collect()
}

pub fn metadata_for_model_name(name: &str) -> Option<&'static CuratedModel> {
    let q = name.to_lowercase();
    CURATED_MODELS.iter().find(|model| {
        model.id.to_lowercase() == q
            || model.file.to_lowercase() == q
            || model.file.trim_end_matches(".gguf").to_lowercase() == q
    })
}

pub fn scan_local_models() -> Vec<String> {
    let mut names = Vec::new();
    for models_dir in model_dirs() {
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
                    continue;
                }
                let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                    continue;
                };
                let size = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
                if size <= 500_000_000 {
                    continue;
                }
                let name = split_gguf_base_name(stem).unwrap_or(stem).to_string();
                if !names.contains(&name) {
                    names.push(name);
                }
            }
        }
    }
    names.sort();
    names
}

pub fn find_model_path(stem: &str) -> PathBuf {
    let filename = format!("{stem}.gguf");
    for dir in model_dirs() {
        let candidate = dir.join(&filename);
        if candidate.exists() {
            return candidate;
        }
        let split_prefix = format!("{stem}-00001-of-");
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with(&split_prefix) && name.ends_with(".gguf") {
                        return entry.path();
                    }
                }
            }
        }
    }
    primary_models_dir().join(filename)
}

pub(crate) fn split_gguf_base_name(stem: &str) -> Option<&str> {
    let suffix = stem.rfind("-of-")?;
    let part_num = &stem[suffix + 4..];
    if part_num.len() != 5 || !part_num.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let dash = stem[..suffix].rfind('-')?;
    let seq = &stem[dash + 1..suffix];
    if seq.len() != 5 || !seq.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(&stem[..dash])
}

pub async fn resolve_model_input(input: &Path) -> Result<PathBuf> {
    let text = input.to_string_lossy();

    if input.exists() {
        return Ok(input.to_path_buf());
    }

    if !text.contains('/') {
        for dir in model_dirs() {
            let candidate = dir.join(input);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
        if let Some(model) = find_curated_model(&text) {
            return download_curated_model(model).await;
        }
        bail!(
            "Model not found: {}\nNot a local file, not in configured model dirs, and not in curated metadata.\nUse a local path, a curated model id (run `mesh-llm models`), `mesh-llm search <query>`, or a Hugging Face URL/ref.",
            text
        );
    }

    if let Some((repo, file)) = parse_huggingface_ref(&text) {
        return download_huggingface_model(&repo, &file).await;
    }

    if text.starts_with("http://") || text.starts_with("https://") {
        let filename = remote_filename(&text)?;
        let dest = primary_models_dir().join(filename);
        if existing_download(&dest).await {
            return Ok(dest);
        }
        eprintln!("📥 Downloading {}...", dest.display());
        download_url(&text, &dest).await?;
        return Ok(dest);
    }

    bail!("Model not found: {text}");
}

pub async fn download_exact_ref(input: &str) -> Result<PathBuf> {
    match parse_exact_model_ref(input)? {
        ExactModelRef::Curated(model) => download_curated_model(model).await,
        ExactModelRef::HuggingFace { repo, file } => download_huggingface_model(&repo, &file).await,
        ExactModelRef::Url { url, filename } => {
            let dest = primary_models_dir().join(filename);
            if existing_download(&dest).await {
                return Ok(dest);
            }
            eprintln!("📥 Downloading {}...", dest.display());
            download_url(&url, &dest).await?;
            Ok(dest)
        }
    }
}

pub fn parse_exact_model_ref(input: &str) -> Result<ExactModelRef> {
    if let Some(model) = find_curated_model_exact(input) {
        return Ok(ExactModelRef::Curated(model));
    }
    if let Some((repo, file)) = parse_huggingface_ref(input) {
        return Ok(ExactModelRef::HuggingFace { repo, file });
    }
    if input.starts_with("http://") || input.starts_with("https://") {
        return Ok(ExactModelRef::Url {
            url: input.to_string(),
            filename: remote_filename(input)?,
        });
    }
    bail!(
        "Expected an exact model ref. Use a curated id from `mesh-llm models`, a Hugging Face ref like org/repo/file.gguf, or a direct URL."
    );
}

fn normalize_gguf_name(name: &str) -> String {
    name.trim_end_matches(".gguf").to_lowercase()
}

fn matching_curated_model_by_basename(repo_file: &str) -> Option<&'static CuratedModel> {
    let basename = Path::new(repo_file)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(repo_file)
        .to_lowercase();
    CURATED_MODELS.iter().find(|model| {
        model.file.to_lowercase() == basename
            || normalize_gguf_name(model.file) == normalize_gguf_name(&basename)
    })
}

pub fn matching_curated_model_for_huggingface(
    repo: &str,
    file: &str,
) -> Option<&'static CuratedModel> {
    let repo = repo.to_lowercase();
    let file = file.to_lowercase();
    CURATED_MODELS
        .iter()
        .find(|model| {
            model.source_repo.map(str::to_lowercase) == Some(repo.clone())
                && model.source_file.to_lowercase() == file
        })
        .or_else(|| matching_curated_model_by_basename(file.as_str()))
}

pub fn matching_curated_model_for_url(url: &str) -> Option<&'static CuratedModel> {
    CURATED_MODELS
        .iter()
        .find(|model| model.url.eq_ignore_ascii_case(url))
        .or_else(|| matching_curated_model_by_basename(url))
}

pub async fn show_exact_model(input: &str) -> Result<ModelDetails> {
    match parse_exact_model_ref(input)? {
        ExactModelRef::Curated(model) => Ok(ModelDetails {
            display_name: model.id.to_string(),
            exact_ref: model.id.to_string(),
            source: "curated",
            download_url: model.url.to_string(),
            size_label: Some(model.size.to_string()),
            description: Some(model.description.to_string()),
            draft: model.draft.map(str::to_string),
            vision: model.mmproj.is_some(),
            moe: model.moe.clone(),
        }),
        ExactModelRef::HuggingFace { repo, file } => {
            let exact_ref = format!("{repo}/{file}");
            let curated = matching_curated_model_for_huggingface(&repo, &file);
            Ok(ModelDetails {
                display_name: Path::new(&file)
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(&file)
                    .to_string(),
                exact_ref: exact_ref.clone(),
                source: "huggingface",
                download_url: huggingface_resolve_url(&repo, &file),
                size_label: curated.map(|model| model.size.to_string()),
                description: curated.map(|model| model.description.to_string()),
                draft: curated.and_then(|model| model.draft.map(str::to_string)),
                vision: curated.map(|model| model.mmproj.is_some()).unwrap_or(false),
                moe: curated.and_then(|model| model.moe.clone()),
            })
        }
        ExactModelRef::Url { url, filename } => {
            let curated = matching_curated_model_for_url(&url);
            Ok(ModelDetails {
                display_name: filename.clone(),
                exact_ref: url.clone(),
                source: "url",
                download_url: url,
                size_label: curated.map(|model| model.size.to_string()),
                description: curated.map(|model| model.description.to_string()),
                draft: curated.and_then(|model| model.draft.map(str::to_string)),
                vision: curated.map(|model| model.mmproj.is_some()).unwrap_or(false),
                moe: curated.and_then(|model| model.moe.clone()),
            })
        }
    }
}

pub async fn search_huggingface(query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    warn_if_missing_huggingface_token();
    let repo_limit = limit.clamp(1, 8);
    let client = http_client()?;
    let repos: Vec<HuggingFaceRepoSummary> = client
        .get("https://huggingface.co/api/models")
        .query(&[
            ("search", query),
            ("filter", "gguf"),
            ("limit", &repo_limit.to_string()),
        ])
        .send()
        .await
        .context("Search Hugging Face")?
        .error_for_status()
        .context("Hugging Face search failed")?
        .json()
        .await
        .context("Parse Hugging Face search response")?;

    let mut hits = Vec::new();
    for repo in repos {
        let detail: HuggingFaceRepoDetail = client
            .get(format!("https://huggingface.co/api/models/{}", repo.id))
            .send()
            .await
            .with_context(|| format!("Fetch Hugging Face repo {}", repo.id))?
            .error_for_status()
            .with_context(|| format!("Hugging Face repo {} returned an error", repo.id))?
            .json()
            .await
            .with_context(|| format!("Parse Hugging Face repo {}", repo.id))?;

        let repo_id = detail.id.or(detail.model_id).unwrap_or(repo.id.clone());
        let mut files: Vec<String> = detail
            .siblings
            .into_iter()
            .map(|sibling| sibling.rfilename)
            .filter(|file| file.ends_with(".gguf"))
            .collect();
        if files.is_empty() {
            continue;
        }
        files.sort_by(|left, right| {
            file_preference_score(left)
                .cmp(&file_preference_score(right))
                .then_with(|| left.cmp(right))
        });
        for file in files.into_iter().take(3) {
            let curated = matching_curated_model_for_huggingface(&repo_id, &file);
            hits.push(SearchHit {
                exact_ref: format!("{repo_id}/{file}"),
                downloads: repo.downloads,
                likes: repo.likes,
                curated,
            });
            if hits.len() >= limit {
                return Ok(hits);
            }
        }
    }
    Ok(hits)
}

pub async fn download_curated_model(model: &CuratedModel) -> Result<PathBuf> {
    let dir = primary_models_dir();
    tokio::fs::create_dir_all(&dir).await?;
    let dest = dir.join(model.file);

    let mut files: Vec<(&str, &str)> = vec![(model.file, model.url)];
    for asset in model.extra_files {
        files.push((asset.file, asset.url));
    }
    if let Some(asset) = &model.mmproj {
        files.push((asset.file, asset.url));
    }

    let mut all_present = true;
    let mut total_size = 0u64;
    for (file, _) in &files {
        let path = dir.join(file);
        let size = tokio::fs::metadata(&path)
            .await
            .map(|meta| meta.len())
            .unwrap_or(0);
        if size < 1_000_000 {
            all_present = false;
            break;
        }
        total_size += size;
    }

    if all_present {
        eprintln!(
            "✅ {} already exists ({:.1}GB, {} file{})",
            model.id,
            total_size as f64 / 1e9,
            files.len(),
            if files.len() > 1 { "s" } else { "" },
        );
        return Ok(dest);
    }

    eprintln!("📥 Downloading {} ({})...", model.id, model.size);
    let mut needed = Vec::new();
    for (file, url) in &files {
        let path = dir.join(file);
        if existing_download(&path).await {
            let size = tokio::fs::metadata(&path)
                .await
                .map(|meta| meta.len())
                .unwrap_or(0);
            eprintln!("  ✅ {file} already exists ({:.1}GB)", size as f64 / 1e9);
            continue;
        }
        needed.push((file.to_string(), url.to_string()));
    }

    if needed.len() > 1 {
        eprintln!("  ⚡ Downloading {} files in parallel...", needed.len());
        let total = needed.len();
        let completed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut handles = Vec::new();
        for (file, url) in needed {
            let path = dir.join(&file);
            let completed = completed.clone();
            handles.push(tokio::spawn(async move {
                download_url(&url, &path).await?;
                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                eprintln!("  ✅ {file} [{done}/{total}]");
                Ok::<(), anyhow::Error>(())
            }));
        }
        for handle in handles {
            handle.await??;
        }
    } else if let Some((file, url)) = needed.into_iter().next() {
        download_url(&url, &dir.join(file)).await?;
    }

    eprintln!("✅ Downloaded {} to {}", model.id, dir.display());
    Ok(dest)
}

pub async fn download_huggingface_model(repo: &str, file: &str) -> Result<PathBuf> {
    let assets = huggingface_download_assets(repo, file)?;
    download_remote_assets(&format!("{repo}/{file}"), assets).await
}

pub async fn download_url(url: &str, dest: &Path) -> Result<()> {
    download_with_resume(dest, url, |_, _| {}).await
}

pub fn huggingface_download_assets(repo: &str, file: &str) -> Result<Vec<(String, String)>> {
    remote_split_parts(file)?
        .unwrap_or_else(|| vec![file.to_string()])
        .into_iter()
        .map(|part| {
            let filename = remote_basename(&part)?;
            Ok((filename, huggingface_resolve_url(repo, &part)))
        })
        .collect()
}

pub async fn download_url_with_progress<F>(url: &str, dest: &Path, on_progress: F) -> Result<()>
where
    F: FnMut(u64, Option<u64>) + Send,
{
    download_with_resume(dest, url, on_progress).await
}

pub fn huggingface_resolve_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

fn parse_huggingface_ref(input: &str) -> Option<(String, String)> {
    if let Some(rest) = input.strip_prefix("https://huggingface.co/") {
        return parse_huggingface_url_tail(rest);
    }
    if let Some(rest) = input.strip_prefix("http://huggingface.co/") {
        return parse_huggingface_url_tail(rest);
    }
    if input.ends_with(".gguf") {
        let parts: Vec<&str> = input.splitn(3, '/').collect();
        if parts.len() == 3 {
            return Some((format!("{}/{}", parts[0], parts[1]), parts[2].to_string()));
        }
    }
    None
}

fn parse_huggingface_url_tail(tail: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = tail.split('/').collect();
    if parts.len() < 5 {
        return None;
    }
    if parts.get(2) != Some(&"resolve") {
        return None;
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    let file = parts[4..].join("/");
    if file.ends_with(".gguf") {
        Some((repo, file))
    } else {
        None
    }
}

fn remote_filename(input: &str) -> Result<String> {
    input
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow!("Cannot extract filename from URL: {input}"))
}

fn remote_basename(input: &str) -> Result<String> {
    Path::new(input)
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow!("Cannot extract filename from {input}"))
}

fn remote_split_parts(input: &str) -> Result<Option<Vec<String>>> {
    let re = regex_lite::Regex::new(r"-00001-of-(\d{5})\.gguf$").unwrap();
    let Some(caps) = re.captures(input) else {
        return Ok(None);
    };
    let part_count: u32 = caps[1]
        .parse()
        .with_context(|| format!("Parse split GGUF part count from {input}"))?;
    let parts = (1..=part_count)
        .map(|index| input.replacen("-00001-of-", &format!("-{index:05}-of-"), 1))
        .collect();
    Ok(Some(parts))
}

async fn download_remote_assets(label: &str, assets: Vec<(String, String)>) -> Result<PathBuf> {
    let dir = primary_models_dir();
    tokio::fs::create_dir_all(&dir).await?;

    let primary = assets
        .first()
        .map(|(file, _)| dir.join(file))
        .ok_or_else(|| anyhow!("No download assets for {label}"))?;

    let mut all_present = true;
    for (file, _) in &assets {
        if !existing_download(&dir.join(file)).await {
            all_present = false;
            break;
        }
    }
    if all_present {
        return Ok(primary);
    }

    eprintln!("📥 Downloading {label}...");
    let mut needed = Vec::new();
    for (file, url) in assets {
        let path = dir.join(&file);
        if existing_download(&path).await {
            let size = tokio::fs::metadata(&path)
                .await
                .map(|meta| meta.len())
                .unwrap_or(0);
            eprintln!("  ✅ {file} already exists ({:.1}GB)", size as f64 / 1e9);
            continue;
        }
        needed.push((file, url));
    }

    if needed.len() > 1 {
        eprintln!("  ⚡ Downloading {} files in parallel...", needed.len());
        let total = needed.len();
        let completed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut handles = Vec::new();
        for (file, url) in needed {
            let path = dir.join(&file);
            let completed = completed.clone();
            handles.push(tokio::spawn(async move {
                download_url(&url, &path).await?;
                let done = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                eprintln!("  ✅ {file} [{done}/{total}]");
                Ok::<(), anyhow::Error>(())
            }));
        }
        for handle in handles {
            handle.await??;
        }
    } else if let Some((file, url)) = needed.into_iter().next() {
        download_url(&url, &dir.join(file)).await?;
    }

    Ok(primary)
}

async fn existing_download(path: &Path) -> bool {
    tokio::fs::metadata(path)
        .await
        .map(|meta| meta.len() > 1_000_000)
        .unwrap_or(false)
}

fn file_preference_score(file: &str) -> usize {
    if file.contains("-00001-of-") {
        return 0;
    }
    const PREFERRED: &[&str] = &[
        "Q4_K_M", "Q4_K_S", "Q4_1", "Q5_K_M", "Q5_K_S", "Q8_0", "BF16",
    ];
    PREFERRED
        .iter()
        .position(|needle| file.contains(needle))
        .unwrap_or(PREFERRED.len() + 1)
}

fn warn_if_missing_huggingface_token() {
    if runtime_config().huggingface_token.is_some() {
        return;
    }
    if WARNED_NO_HF_TOKEN.swap(true, Ordering::Relaxed) {
        return;
    }
    eprintln!("Warning: no Hugging Face token configured.");
    eprintln!("Unauthenticated Hugging Face requests may be slower or throttled.");
    eprintln!("Add one to ~/.mesh-llm/config.toml:");
    eprintln!("  [huggingface]");
    eprintln!("  token = \"hf_...\"");
    eprintln!("Or set HF_TOKEN in your environment.");
    eprintln!("Create a token at: https://huggingface.co/settings/tokens");
}

fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .connect_timeout(std::time::Duration::from_secs(30))
        .user_agent(format!("mesh-llm/{}", crate::VERSION))
        .build()
        .context("build HTTP client")
}

async fn download_with_resume<F>(dest: &Path, url: &str, mut on_progress: F) -> Result<()>
where
    F: FnMut(u64, Option<u64>) + Send,
{
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let tmp = dest.with_extension("gguf.part");
    let client = http_client()?;

    let mut attempt: u64 = 0;
    loop {
        attempt += 1;
        let existing_bytes = if tmp.exists() {
            tokio::fs::metadata(&tmp).await?.len()
        } else {
            0
        };

        if attempt <= 3 || attempt % 10 == 0 {
            eprintln!(
                "  attempt {attempt}{}...",
                if existing_bytes > 0 {
                    format!(" (resuming from {:.1}MB)", existing_bytes as f64 / 1e6)
                } else {
                    String::new()
                }
            );
        }

        let mut request = client.get(url);
        if url.contains("huggingface.co/") {
            warn_if_missing_huggingface_token();
            if let Some(token) = &runtime_config().huggingface_token {
                request = request.bearer_auth(token);
            }
        }
        if existing_bytes > 0 {
            request = request.header("Range", format!("bytes={existing_bytes}-"));
        }

        let backoff_secs = std::cmp::min(3 * (1u64 << (attempt - 1).min(4)), 60);
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => {
                eprintln!("  connection failed: {error}");
                eprintln!("  retrying in {backoff_secs}s...");
                tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
                continue;
            }
        };

        let status = response.status();
        if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
            if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                let _ = tokio::fs::remove_file(&tmp).await;
                eprintln!("  server rejected resume, starting fresh...");
                continue;
            }
            eprintln!("  HTTP {status}, retrying in {backoff_secs}s...");
            tokio::time::sleep(std::time::Duration::from_secs(backoff_secs)).await;
            continue;
        }

        let total_bytes = if status == reqwest::StatusCode::PARTIAL_CONTENT {
            response
                .headers()
                .get("content-range")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.rsplit('/').next())
                .and_then(|value| value.parse::<u64>().ok())
        } else {
            response
                .content_length()
                .map(|value| value + existing_bytes)
        };
        on_progress(existing_bytes, total_bytes);

        if attempt == 1 || existing_bytes == 0 {
            if let Some(total) = total_bytes {
                let remaining = total.saturating_sub(existing_bytes);
                if let Some(free) = free_disk_space(dest) {
                    let needed = remaining + 1_000_000_000;
                    if free < needed {
                        bail!(
                            "Not enough disk space: need {:.1}GB but only {:.1}GB free on {}",
                            needed as f64 / 1e9,
                            free as f64 / 1e9,
                            dest.parent().unwrap_or(dest).display()
                        );
                    }
                }
            }
        }

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&tmp)
            .await
            .context("Failed to open temp file")?;

        let mut stream = response.bytes_stream();
        let mut downloaded = existing_bytes;
        let mut last_progress = std::time::Instant::now();
        let mut got_data = false;

        print_progress(downloaded, total_bytes);
        loop {
            match stream.next().await {
                Some(Ok(chunk)) => {
                    file.write_all(&chunk)
                        .await
                        .context("Failed to write chunk")?;
                    downloaded += chunk.len() as u64;
                    got_data = true;

                    if last_progress.elapsed() >= std::time::Duration::from_millis(500) {
                        on_progress(downloaded, total_bytes);
                        print_progress(downloaded, total_bytes);
                        last_progress = std::time::Instant::now();
                    }
                }
                Some(Err(error)) => {
                    file.flush().await.ok();
                    eprint!("\r");
                    eprintln!(
                        "  download interrupted at {:.1}MB: {error}",
                        downloaded as f64 / 1e6
                    );
                    if got_data {
                        attempt = 0;
                    }
                    let retry_secs = std::cmp::min(3 * (1u64 << attempt.min(4)), 60);
                    eprintln!("  retrying in {retry_secs}s (will resume)...");
                    tokio::time::sleep(std::time::Duration::from_secs(retry_secs)).await;
                    break;
                }
                None => {
                    file.flush().await?;
                    eprint!("\r");
                    on_progress(downloaded, total_bytes);
                    print_progress(downloaded, total_bytes);
                    eprintln!();
                    tokio::fs::rename(&tmp, dest)
                        .await
                        .context("Failed to move downloaded file")?;
                    return Ok(());
                }
            }
        }
    }
}

fn free_disk_space(path: &Path) -> Option<u64> {
    let mut check = path.to_path_buf();
    loop {
        if check.exists() {
            break;
        }
        if !check.pop() {
            return None;
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c_path = std::ffi::CString::new(check.as_os_str().as_bytes()).ok()?;
        let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
        let ret = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
        if ret == 0 {
            Some(stat.f_bavail as u64 * stat.f_frsize as u64)
        } else {
            None
        }
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn print_progress(downloaded: u64, total: Option<u64>) {
    if let Some(total) = total {
        let pct = (downloaded as f64 / total as f64) * 100.0;
        eprint!(
            "\r  {:.1}/{:.1}MB ({:.1}%)",
            downloaded as f64 / 1e6,
            total as f64 / 1e6,
            pct
        );
    } else {
        eprint!("\r  {:.1}MB", downloaded as f64 / 1e6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exact_hf_refs() {
        let parsed = parse_exact_model_ref(
            "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        )
        .unwrap();
        match parsed {
            ExactModelRef::HuggingFace { repo, file } => {
                assert_eq!(repo, "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF");
                assert_eq!(file, "qwen2.5-coder-32b-instruct-q4_k_m.gguf");
            }
            _ => panic!("expected Hugging Face ref"),
        }
    }

    #[test]
    fn expands_home_in_model_dirs() {
        let expanded = expand_path(Path::new("~/models"));
        assert!(expanded.ends_with("models"));
    }

    #[test]
    fn matches_split_huggingface_refs_using_repo_and_remote_path() {
        let matched = matching_curated_model_for_huggingface(
            "Qwen/Qwen3-Coder-Next-GGUF",
            "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
        )
        .unwrap();
        assert_eq!(matched.id, "Qwen3-Coder-Next-Q4_K_M");
    }

    #[test]
    fn matches_curated_url_exactly() {
        let matched = matching_curated_model_for_url(
            "https://registry.ollama.ai/v2/library/qwen3.5/blobs/sha256:d4b8b4f4c350f5d322dc8235175eeae02d32c6f3fd70bdb9ea481e3abb7d7fc4",
        )
        .unwrap();
        assert_eq!(matched.id, "Qwen3.5-27B-Q4_K_M");
    }

    #[test]
    fn expands_split_huggingface_download_assets() {
        let assets = huggingface_download_assets(
            "Qwen/Qwen3-Coder-Next-GGUF",
            "Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf",
        )
        .unwrap();
        assert_eq!(assets.len(), 4);
        assert_eq!(assets[0].0, "Qwen3-Coder-Next-Q4_K_M-00001-of-00004.gguf");
        assert_eq!(
            assets[3].1,
            "https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-Q4_K_M/Qwen3-Coder-Next-Q4_K_M-00004-of-00004.gguf"
        );
    }

    #[test]
    fn test_free_disk_space() {
        let path = std::env::temp_dir().join("test_file.gguf");
        let free = free_disk_space(&path);

        #[cfg(unix)]
        {
            assert!(
                free.is_some(),
                "should get free space for {}",
                path.display()
            );
            assert!(free.unwrap() > 1_000_000_000);
        }

        #[cfg(not(unix))]
        {
            assert!(free.is_none());
        }
    }
}
