//! Deterministic, serializable reflection of the public mesh-llm CLI.
//!
//! The inventory is intentionally built from [`Cli::command`].  Clap is the
//! source of truth for command names, aliases, argument metadata, defaults,
//! possible values, and the declaration order used by the help output.  The
//! only nodes that are not present in Clap's command tree are the documented
//! `serve`/`client` runtime surfaces (which are normalized into root flags)
//! and the external plugin catch-all.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use clap::{Arg, ArgAction, Command, CommandFactory};
use serde::{Deserialize, Serialize};

use crate::parser::{Cli, RuntimeSurface, runtime_surface_help};

/// Schema version for the generated CLI inventory document.
pub const CLI_INVENTORY_SCHEMA_VERSION: u32 = 1;

/// The three node kinds consumed by the website explorer.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InventoryNodeKind {
    Command,
    Option,
    Positional,
}

/// Root document written by `mesh-llm-cli-inventory`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CliInventory {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    pub root: InventoryNode,
}

/// A command, option, or positional argument in the reflected CLI tree.
///
/// Common fields are always serialized.  Command-only metadata is present for
/// command nodes (including `false` values so consumers can distinguish a
/// real command from a synthetic surface); leaf metadata is present for option
/// and positional nodes.  This keeps the JSON shape predictable without
/// placing command fields on every leaf.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InventoryNode {
    pub kind: InventoryNodeKind,
    pub name: String,
    pub path: String,
    pub description: String,
    pub hidden: bool,
    pub aliases: Vec<String>,
    pub children: Vec<Self>,

    /// Whether a command was synthesized rather than reflected directly from
    /// a Clap subcommand.  Omitted for option and positional nodes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synthetic: Option<bool>,
    /// Whether a command represents Clap's external-subcommand catch-all.
    /// Omitted for option and positional nodes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external: Option<bool>,

    /// Clap's stable argument identifier.  Omitted for command nodes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Short option spelling without the leading `-`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short: Option<String>,
    /// Long option spelling without the leading `--`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long: Option<String>,
    /// Value placeholders declared by Clap.  An empty array is retained for
    /// leaves that do not take a value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_names: Option<Vec<String>>,
    /// Values supplied by Clap as defaults, rendered lossily as UTF-8 for JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_values: Option<Vec<String>>,
    /// Accepted possible-value names and aliases in declaration order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub possible_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeatable: Option<bool>,
    /// Canonical spellings of arguments that conflict with this leaf.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conflicts: Option<Vec<String>>,
}

impl InventoryNode {
    fn command(metadata: CommandMetadata, children: Vec<Self>) -> Self {
        Self {
            kind: InventoryNodeKind::Command,
            name: metadata.name,
            path: metadata.path,
            description: metadata.description,
            hidden: metadata.hidden,
            aliases: metadata.aliases,
            children,
            synthetic: Some(metadata.synthetic),
            external: Some(metadata.external),
            id: None,
            short: None,
            long: None,
            value_names: None,
            default_values: None,
            possible_values: None,
            required: None,
            global: None,
            repeatable: None,
            conflicts: None,
        }
    }

    fn leaf(
        kind: InventoryNodeKind,
        name: String,
        path: String,
        description: String,
        hidden: bool,
        aliases: Vec<String>,
        metadata: LeafMetadata,
    ) -> Self {
        Self {
            kind,
            name,
            path,
            description,
            hidden,
            aliases,
            children: Vec::new(),
            synthetic: None,
            external: None,
            id: Some(metadata.id),
            short: metadata.short,
            long: metadata.long,
            value_names: Some(metadata.value_names),
            default_values: Some(metadata.default_values),
            possible_values: Some(metadata.possible_values),
            required: Some(metadata.required),
            global: Some(metadata.global),
            repeatable: Some(metadata.repeatable),
            conflicts: Some(metadata.conflicts),
        }
    }
}

#[derive(Debug)]
struct CommandMetadata {
    name: String,
    path: String,
    description: String,
    hidden: bool,
    aliases: Vec<String>,
    synthetic: bool,
    external: bool,
}

#[derive(Debug)]
struct LeafMetadata {
    id: String,
    short: Option<String>,
    long: Option<String>,
    value_names: Vec<String>,
    default_values: Vec<String>,
    possible_values: Vec<String>,
    required: bool,
    global: bool,
    repeatable: bool,
    conflicts: Vec<String>,
}

/// Build the complete CLI inventory from the current Clap reflection.
pub fn build_cli_inventory() -> CliInventory {
    let mut command = Cli::command();
    // `Command::build` injects a synthetic `help` subcommand at every level by
    // default.  Disable that injection on the reflected copy before building
    // so an explicitly declared command named `help` remains distinguishable
    // from Clap's generated help surface.
    disable_generated_help_subcommands(&mut command);
    command.build();

    let root_name = command.get_name().to_string();
    let root_path = root_name.clone();
    let root_global_ids = command
        .get_arguments()
        .filter(|arg| arg.is_global_set())
        .map(|arg| arg.get_id().to_string())
        .collect::<HashSet<_>>();

    let mut children = Vec::new();
    children.push(build_runtime_surface_node(
        &command,
        RuntimeSurface::Serve,
        &root_path,
    ));
    children.push(build_runtime_surface_node(
        &command,
        RuntimeSurface::Client,
        &root_path,
    ));

    children.push(build_runtime_surface_node(
        &command,
        RuntimeSurface::Share,
        &root_path,
    ));

    for subcommand in command.get_subcommands() {
        if matches!(subcommand.get_name(), "serve" | "client" | "share") {
            continue;
        }
        children.push(build_command_node(subcommand, &root_path, &root_global_ids));
    }

    if command.is_allow_external_subcommands_set() {
        children.push(build_external_plugin_node(&root_path));
    }

    children.extend(build_leaf_nodes(&command, &root_path, |arg| {
        !arg.is_positional()
    }));

    let description = command_description(&command);
    CliInventory {
        schema_version: CLI_INVENTORY_SCHEMA_VERSION,
        root: InventoryNode::command(
            CommandMetadata {
                name: root_name,
                path: root_path,
                description,
                hidden: command.is_hide_set(),
                aliases: command_aliases(&command),
                synthetic: false,
                external: false,
            },
            children,
        ),
    }
}

/// Serialize an inventory using the canonical pretty JSON representation.
pub fn serialize_cli_inventory(inventory: &CliInventory) -> Result<String, serde_json::Error> {
    let mut json = serde_json::to_string_pretty(inventory)?;
    json.push('\n');
    Ok(json)
}

/// Build and serialize the current inventory in one step.
pub fn cli_inventory_json() -> Result<String, serde_json::Error> {
    serialize_cli_inventory(&build_cli_inventory())
}

/// Atomically write a freshly generated inventory to `path`.
pub fn write_cli_inventory(path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    let json = cli_inventory_json().map_err(io::Error::other)?;
    atomic_write(path, json.as_bytes())
}

/// Verify that `path` already contains the canonical inventory bytes.
pub fn check_cli_inventory(path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    let expected = cli_inventory_json().map_err(io::Error::other)?;
    let mut actual = String::new();
    File::open(path)?.read_to_string(&mut actual)?;
    if actual == expected {
        return Ok(());
    }

    Err(io::Error::new(
        ErrorKind::InvalidData,
        format!(
            "CLI inventory at {} is stale; regenerate it from mesh-llm-cli",
            path.display()
        ),
    ))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary_path = temporary_path(path);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary_path)?;

    let write_result = (|| {
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary_path, path)
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    write_result
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut temporary_path = OsString::from(path.as_os_str());
    temporary_path.push(".tmp");
    PathBuf::from(temporary_path)
}

fn build_command_node(
    command: &Command,
    parent_path: &str,
    inherited_global_ids: &HashSet<String>,
) -> InventoryNode {
    let path = join_path(parent_path, command.get_name());
    let mut global_ids = inherited_global_ids.clone();
    command
        .get_arguments()
        .filter(|arg| arg.is_global_set())
        .for_each(|arg| {
            global_ids.insert(arg.get_id().to_string());
        });

    let mut children = Vec::new();
    for subcommand in command.get_subcommands() {
        children.push(build_command_node(subcommand, &path, &global_ids));
    }

    children.extend(build_leaf_nodes(command, &path, |arg| {
        !(arg.is_global_set() && inherited_global_ids.contains(arg.get_id().as_str()))
    }));

    InventoryNode::command(
        CommandMetadata {
            name: command.get_name().to_string(),
            path,
            description: command_description(command),
            hidden: command.is_hide_set(),
            aliases: command_aliases(command),
            synthetic: false,
            external: false,
        },
        children,
    )
}

fn build_runtime_surface_node(
    root: &Command,
    surface: RuntimeSurface,
    parent_path: &str,
) -> InventoryNode {
    let name = match surface {
        RuntimeSurface::Serve => "serve",
        RuntimeSurface::Client => "client",
        RuntimeSurface::Share => "share",
    };
    let path = join_path(parent_path, name);
    let help = runtime_surface_help(surface);
    let description = help.split_once("\n\n").map_or_else(
        || help.trim().to_string(),
        |(summary, _)| summary.trim().to_string(),
    );

    let mut children = Vec::new();
    if surface == RuntimeSurface::Share {
        let endpoint = Arg::new("endpoint")
            .required(true)
            .help("HTTP OpenAI-compatible upstream URL");
        children.push(build_leaf_node(root, &path, &endpoint));
    }
    children.extend(build_leaf_nodes(root, &path, |arg| {
        runtime_surface_allows_arg(surface, arg)
    }));

    InventoryNode::command(
        CommandMetadata {
            name: name.to_string(),
            path,
            description,
            hidden: false,
            aliases: Vec::new(),
            synthetic: true,
            external: false,
        },
        children,
    )
}

/// Disable Clap's generated help subtree recursively while retaining any
/// explicitly declared `help` command in the reflected tree.
fn disable_generated_help_subcommands(command: &mut Command) {
    *command = std::mem::take(command).disable_help_subcommand(true);
    for subcommand in command.get_subcommands_mut() {
        disable_generated_help_subcommands(subcommand);
    }
}

fn build_external_plugin_node(parent_path: &str) -> InventoryNode {
    let name = "PLUGIN_COMMAND ...";
    InventoryNode::command(
        CommandMetadata {
            name: name.to_string(),
            path: join_path(parent_path, name),
            description: "Run a CLI command contributed by a configured plugin.".to_string(),
            hidden: false,
            aliases: Vec::new(),
            synthetic: true,
            external: true,
        },
        Vec::new(),
    )
}

fn build_leaf_nodes<F>(command: &Command, parent_path: &str, include: F) -> Vec<InventoryNode>
where
    F: Fn(&Arg) -> bool,
{
    command
        .get_arguments()
        .filter(|arg| include(arg))
        .map(|arg| build_leaf_node(command, parent_path, arg))
        .collect()
}

fn build_leaf_node(command: &Command, parent_path: &str, arg: &Arg) -> InventoryNode {
    let kind = if arg.is_positional() {
        InventoryNodeKind::Positional
    } else {
        InventoryNodeKind::Option
    };
    let name = argument_name(arg, kind);
    let path = join_path(parent_path, &name);
    let metadata = LeafMetadata {
        id: arg.get_id().to_string(),
        short: arg.get_short().map(|short| short.to_string()),
        long: arg.get_long().map(str::to_string),
        value_names: argument_value_names(arg),
        default_values: arg
            .get_default_values()
            .iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect(),
        possible_values: possible_values(arg),
        required: arg.is_required_set(),
        global: arg.is_global_set(),
        repeatable: is_repeatable(arg),
        conflicts: command
            .get_arg_conflicts_with(arg)
            .into_iter()
            .map(argument_spelling)
            .collect(),
    };

    InventoryNode::leaf(
        kind,
        name,
        path,
        argument_description(arg),
        arg.is_hide_set(),
        argument_aliases(arg),
        metadata,
    )
}

fn runtime_surface_allows_arg(surface: RuntimeSurface, arg: &Arg) -> bool {
    let id = arg.get_id().as_str();
    !runtime_surface_excluded_arg(surface, id)
}

const RUNTIME_SURFACE_ALWAYS_EXCLUDED_ARGS: &[&str] = &["client", "plugin", "shared_endpoint"];

fn runtime_surface_excluded_arg(surface: RuntimeSurface, id: &str) -> bool {
    if RUNTIME_SURFACE_ALWAYS_EXCLUDED_ARGS.contains(&id) {
        return true;
    }

    // Client mode has no local model or serving pipeline.  Keep shared
    // networking, logging, ownership, and management flags by reflecting
    // the root command, while omitting serving-only controls.
    matches!(surface, RuntimeSurface::Client | RuntimeSurface::Share) && client_excluded_arg(id)
}

fn client_excluded_arg(id: &str) -> bool {
    matches!(
        id,
        "model"
            | "gguf"
            | "mmproj"
            | "checkpoint_quantization"
            | "checkpoint_imatrix"
            | "local_model_only"
            | "native_serving_plugin"
            | "native_serving_plugin_config"
            | "native_serving_plugin_state"
            | "native_serving_plugin_deadline_ms"
            | "draft"
            | "draft_max"
            | "no_draft"
            | "split"
            | "split_topology_lock"
            | "ctx_size"
    ) || id.starts_with("speculative_")
}

fn command_description(command: &Command) -> String {
    command
        .get_long_about()
        .or_else(|| command.get_about())
        .map_or_else(String::new, ToString::to_string)
        .trim()
        .to_string()
}

fn argument_description(arg: &Arg) -> String {
    arg.get_long_help()
        .or_else(|| arg.get_help())
        .map_or_else(String::new, ToString::to_string)
        .trim()
        .to_string()
}

fn command_aliases(command: &Command) -> Vec<String> {
    let mut aliases = command
        .get_all_aliases()
        .map(str::to_string)
        .collect::<Vec<_>>();
    aliases.dedup();
    aliases
}

fn argument_aliases(arg: &Arg) -> Vec<String> {
    let mut aliases = arg
        .get_all_aliases()
        .unwrap_or_default()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    aliases.extend(
        arg.get_all_short_aliases()
            .unwrap_or_default()
            .into_iter()
            .map(|alias| alias.to_string()),
    );
    aliases.dedup();
    aliases
}

fn argument_name(arg: &Arg, kind: InventoryNodeKind) -> String {
    if kind == InventoryNodeKind::Positional {
        let value_name = arg
            .get_value_names()
            .and_then(|names| names.first())
            .map(ToString::to_string);
        return value_name.unwrap_or_else(|| {
            let mut name = arg.get_id().to_string().to_ascii_uppercase();
            if is_repeatable(arg) {
                name.push_str("...");
            }
            name
        });
    }

    arg.get_long()
        .map(|long| format!("--{long}"))
        .or_else(|| arg.get_short().map(|short| format!("-{short}")))
        .unwrap_or_else(|| arg.get_id().to_string())
}

fn argument_value_names(arg: &Arg) -> Vec<String> {
    if !arg.get_action().takes_values() {
        return Vec::new();
    }
    arg.get_value_names()
        .map(|names| names.iter().map(ToString::to_string).collect())
        .unwrap_or_default()
}

fn argument_spelling(arg: &Arg) -> String {
    argument_name(
        arg,
        if arg.is_positional() {
            InventoryNodeKind::Positional
        } else {
            InventoryNodeKind::Option
        },
    )
}

fn possible_values(arg: &Arg) -> Vec<String> {
    let mut values = Vec::new();
    for value in arg.get_possible_values() {
        for name in value.get_name_and_aliases() {
            if !values.iter().any(|existing| existing == name) {
                values.push(name.to_string());
            }
        }
    }
    values
}

fn is_repeatable(arg: &Arg) -> bool {
    matches!(arg.get_action(), ArgAction::Append | ArgAction::Count)
        || arg
            .get_num_args()
            .is_some_and(|range| range.max_values() > 1)
}

fn join_path(parent: &str, child: &str) -> String {
    format!("{parent} {child}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_child<'a>(node: &'a InventoryNode, name: &str) -> &'a InventoryNode {
        node.children
            .iter()
            .find(|child| child.name == name)
            .unwrap_or_else(|| panic!("missing child {name:?} under {}", node.path))
    }

    fn flatten<'a>(node: &'a InventoryNode, output: &mut Vec<&'a InventoryNode>) {
        output.push(node);
        for child in &node.children {
            flatten(child, output);
        }
    }

    #[test]
    fn serialization_is_byte_for_byte_deterministic() {
        let first = cli_inventory_json().expect("serialize inventory");
        let second = cli_inventory_json().expect("serialize inventory");
        assert_eq!(first, second);
        assert!(first.ends_with('\n'));
        assert_eq!(
            first.matches("\n").count(),
            first.len() - first.replace('\n', "").len()
        );
    }

    #[test]
    fn hierarchy_order_suppresses_generated_help_and_places_external_last() {
        let inventory = build_cli_inventory();
        let children = &inventory.root.children;
        assert_eq!(children[0].name, "serve");
        assert_eq!(children[1].name, "client");
        assert!(children.iter().all(|child| child.name != "help"));
        assert_eq!(
            children.last().map(|child| child.name.as_str()),
            Some("--version")
        );

        let first_leaf = children
            .iter()
            .position(|child| child.kind != InventoryNodeKind::Command)
            .expect("root options");
        assert!(
            children[..first_leaf]
                .iter()
                .all(|child| child.kind == InventoryNodeKind::Command)
        );
        assert!(
            children[first_leaf..]
                .iter()
                .all(|child| child.kind != InventoryNodeKind::Command)
        );

        let external = children
            .iter()
            .find(|child| child.name == "PLUGIN_COMMAND ...")
            .expect("plugin catch-all");
        assert!(external.external == Some(true));
        assert!(external.synthetic == Some(true));
        assert_eq!(
            children
                .iter()
                .position(|child| child.name == "PLUGIN_COMMAND ..."),
            Some(first_leaf - 1)
        );
    }

    #[test]
    fn explicit_help_subcommand_is_retained() {
        let mut command = Command::new("mesh-llm")
            .subcommand(Command::new("help").about("A user-declared command named help."));
        disable_generated_help_subcommands(&mut command);
        command.build();

        let help_commands = command
            .get_subcommands()
            .filter(|subcommand| subcommand.get_name() == "help")
            .collect::<Vec<_>>();
        assert_eq!(help_commands.len(), 1);
        assert_eq!(
            help_commands[0].get_about().map(ToString::to_string),
            Some("A user-declared command named help.".to_string())
        );
    }

    #[test]
    fn reflected_leaf_metadata_preserves_clap_details() {
        let inventory = build_cli_inventory();
        let log_format = find_child(&inventory.root, "--log-format");
        assert_eq!(log_format.kind, InventoryNodeKind::Option);
        assert_eq!(log_format.id.as_deref(), Some("log_format"));
        assert_eq!(log_format.long.as_deref(), Some("log-format"));
        assert_eq!(log_format.default_values.as_deref().unwrap(), ["pretty"]);
        assert_eq!(
            log_format.possible_values.as_deref().unwrap(),
            ["pretty", "json"]
        );
        assert_eq!(log_format.repeatable, Some(false));

        let join = find_child(&inventory.root, "--join");
        assert_eq!(join.short.as_deref(), Some("j"));
        assert_eq!(join.repeatable, Some(true));

        let auth = find_child(&inventory.root, "auth");
        let init = find_child(auth, "init");
        let no_passphrase = find_child(init, "--no-passphrase");
        assert_eq!(no_passphrase.conflicts.as_deref().unwrap(), ["--keychain"]);

        let models = find_child(&inventory.root, "models");
        let search = find_child(models, "search");
        let query = find_child(search, "QUERY");
        assert_eq!(query.required, Some(true));
        assert_eq!(query.repeatable, Some(true));

        let gpus = find_child(&inventory.root, "gpus");
        assert_eq!(gpus.aliases, vec!["gpu"]);
        let run_benchmark = find_child(gpus, "run-benchmark");
        assert!(run_benchmark.hidden);
        assert!(!find_child(run_benchmark, "--backend").hidden);
    }

    #[test]
    fn pseudo_surfaces_reuse_root_leaves_with_surface_exclusions() {
        let inventory = build_cli_inventory();
        let serve = find_child(&inventory.root, "serve");
        let client = find_child(&inventory.root, "client");
        let share = find_child(&inventory.root, "share");
        assert_eq!(share.synthetic, Some(true));
        assert!(share.children.iter().all(|child| child.name != "--model"));
        assert!(share.children.iter().any(
            |child| child.kind == InventoryNodeKind::Positional && child.required == Some(true)
        ));

        assert_eq!(serve.synthetic, Some(true));
        assert_eq!(client.synthetic, Some(true));
        assert!(find_child(serve, "--model").global == Some(false));
        assert!(serve.children.iter().all(|child| child.name != "--plugin"));
        assert!(client.children.iter().all(|child| child.name != "--model"));
        assert!(
            client
                .children
                .iter()
                .all(|child| child.name != "--checkpoint-quantization")
        );
        assert!(client.children.iter().all(|child| child.name != "--client"));
        assert!(client.children.iter().all(|child| child.name != "--plugin"));
        assert!(find_child(client, "--auto").path == "mesh-llm client --auto");
        assert!(serve.children.iter().any(|child| child.name == "--help"));
        assert!(
            client
                .children
                .iter()
                .any(|child| child.name == "--version")
        );
    }

    #[test]
    fn inherited_global_arguments_are_not_duplicated_in_descendants() {
        let inventory = build_cli_inventory();
        let root_global = inventory
            .root
            .children
            .iter()
            .filter(|child| child.global == Some(true))
            .map(|child| child.id.clone().expect("leaf id"))
            .collect::<HashSet<_>>();
        assert!(!root_global.is_empty());

        let mut all_nodes = Vec::new();
        flatten(&inventory.root, &mut all_nodes);
        for node in all_nodes {
            if node.path == inventory.root.path
                || node.path.starts_with("mesh-llm serve")
                || node.path.starts_with("mesh-llm client")
                || node.path.starts_with("mesh-llm share")
            {
                continue;
            }
            if node.kind == InventoryNodeKind::Command {
                assert!(!node.children.iter().any(|child| {
                    child
                        .id
                        .as_ref()
                        .is_some_and(|id| root_global.contains(id) && child.global == Some(true))
                }));
            }
        }
    }

    #[test]
    fn inventory_contains_exactly_one_serve_and_client_surface_node() {
        let inventory = build_cli_inventory();
        let serve_count = inventory
            .root
            .children
            .iter()
            .filter(|child| child.name == "serve")
            .count();
        let client_count = inventory
            .root
            .children
            .iter()
            .filter(|child| child.name == "client")
            .count();

        assert_eq!(serve_count, 1, "serve should appear exactly once");
        assert_eq!(client_count, 1, "client should appear exactly once");
    }
}
