use crate::{ConfigEditor, MeshConfig, validate_config};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use toml_edit::{ArrayOfTables, DocumentMut, Item, Table, value};

pub fn config_path(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.to_path_buf());
    }
    if let Ok(path) = std::env::var("MESH_LLM_CONFIG") {
        return Ok(PathBuf::from(path));
    }
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".mesh-llm").join("config.toml"))
}

pub fn load_config(override_path: Option<&Path>) -> Result<MeshConfig> {
    let path = config_path(override_path)?;
    if !path.exists() {
        return Ok(MeshConfig::default());
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config {}", path.display()))?;
    parse_config_toml(&raw).with_context(|| format!("Invalid config {}", path.display()))
}

pub fn parse_config_toml(raw: &str) -> Result<MeshConfig> {
    let config: MeshConfig = toml::from_str(raw).context("failed to parse config TOML")?;
    validate_config(&config)?;
    Ok(config)
}

pub fn config_to_toml(config: &MeshConfig) -> Result<String> {
    validate_config(config)?;
    toml::to_string(config).context("toml serialization failed")
}

#[derive(Clone, Debug)]
pub struct ConfigStore {
    path: PathBuf,
}

impl ConfigStore {
    pub fn open(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn default_path() -> Result<Self> {
        Ok(Self {
            path: config_path(None)?,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load(&self) -> Result<MeshConfig> {
        load_config(Some(&self.path))
    }

    pub fn save(&self, config: &MeshConfig) -> Result<()> {
        let toml_str = config_to_toml(config)?;
        atomic_write(&self.path, toml_str.as_bytes())
            .with_context(|| format!("failed to write config {}", self.path.display()))
    }

    pub fn update<F>(&self, edit: F) -> Result<MeshConfig>
    where
        F: FnOnce(&mut ConfigEditor) -> Result<()>,
    {
        let mut editor = ConfigEditor::new(self.load()?);
        edit(&mut editor)?;
        let config = editor.into_config();
        self.save(&config)?;
        Ok(config)
    }

    pub fn edit_preserving<F>(&self, edit: F) -> Result<MeshConfig>
    where
        F: FnOnce(&mut DocumentMut) -> Result<()>,
    {
        let mut doc = self.read_document()?;
        edit(&mut doc)?;
        let config = parse_config_toml(&doc.to_string())
            .with_context(|| format!("invalid edited config {}", self.path.display()))?;
        self.write_document(&doc)?;
        Ok(config)
    }

    pub fn model_refs(&self) -> Result<Vec<String>> {
        let doc = self.read_document()?;
        let Some(models) = doc.get("models").and_then(Item::as_array_of_tables) else {
            return Ok(Vec::new());
        };
        Ok(models.iter().filter_map(model_ref_from_table).collect())
    }

    pub fn add_model_ref(&self, model_ref: &str) -> Result<Vec<String>> {
        let model_ref = normalize_model_ref(model_ref)?;
        self.edit_preserving(|doc| {
            let models = ensure_models_array(doc)?;
            if !models
                .iter()
                .filter_map(model_ref_from_table)
                .any(|configured| configured == model_ref)
            {
                let mut table = Table::new();
                table["model"] = value(model_ref);
                models.push(table);
            }
            Ok(())
        })?;
        self.model_refs()
    }

    pub fn remove_model_ref(&self, model_ref: &str) -> Result<Vec<String>> {
        let model_ref = normalize_model_ref(model_ref)?;
        self.edit_preserving(|doc| {
            let Some(models) = doc.get("models").and_then(Item::as_array_of_tables) else {
                return Ok(());
            };
            let mut next = ArrayOfTables::new();
            for table in models.iter() {
                let keep = model_ref_from_table(table)
                    .map(|configured| configured != model_ref)
                    .unwrap_or(true);
                if keep {
                    next.push(table.clone());
                }
            }
            doc["models"] = Item::ArrayOfTables(next);
            Ok(())
        })?;
        self.model_refs()
    }

    fn read_document(&self) -> Result<DocumentMut> {
        if !self.path.exists() {
            return Ok(DocumentMut::new());
        }
        let raw = std::fs::read_to_string(&self.path)
            .with_context(|| format!("failed to read config {}", self.path.display()))?;
        raw.parse::<DocumentMut>()
            .with_context(|| format!("failed to parse config {}", self.path.display()))
    }

    fn write_document(&self, doc: &DocumentMut) -> Result<()> {
        atomic_write(&self.path, doc.to_string().as_bytes())
            .with_context(|| format!("failed to write config {}", self.path.display()))
    }
}

fn ensure_models_array(doc: &mut DocumentMut) -> Result<&mut ArrayOfTables> {
    if !doc.as_table().contains_key("models") {
        doc["models"] = Item::ArrayOfTables(ArrayOfTables::new());
    }
    doc["models"]
        .as_array_of_tables_mut()
        .ok_or_else(|| anyhow::anyhow!("config key `models` is not a TOML array of tables"))
}

fn model_ref_from_table(table: &Table) -> Option<String> {
    table
        .get("model")
        .and_then(Item::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn normalize_model_ref(model_ref: &str) -> Result<&str> {
    let model_ref = model_ref.trim();
    if model_ref.is_empty() {
        bail!("model ref cannot be empty");
    }
    Ok(model_ref)
}

fn atomic_write(target: &Path, contents: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file_name = target
        .file_name()
        .unwrap_or(target.as_os_str())
        .to_string_lossy();
    let parent = target.parent().unwrap_or(Path::new("."));
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let tmp = parent.join(format!(".{}.{}.{}.tmp", file_name, pid, nanos));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)?;
    file.write_all(contents)?;
    file.sync_all()?;
    drop(file);
    #[cfg(windows)]
    if target.exists() {
        std::fs::remove_file(target)?;
    }
    if let Err(e) = std::fs::rename(&tmp, target) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}
