use crate::command::{DynResult, ensure_eq, ensure_set_eq, run_command, trimmed_stderr_or_stdout};
use crate::{publish_consistency, release_targets, workflow_checks};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn check_attestation_default_version(repo_root: &Path) -> DynResult<()> {
    let runtime_version = resolve_runtime_version(repo_root)?;
    let default_node_version = host_runtime_package_version(repo_root)?;
    ensure_eq(
        runtime_version.as_str(),
        default_node_version.as_str(),
        "xtask release-attestation default node version",
    )
}

pub(crate) fn default_node_version() -> DynResult<String> {
    host_runtime_package_version(&repo_root()?)
}

fn resolve_runtime_version(repo_root: &Path) -> DynResult<String> {
    let runtime_lib = repo_root
        .join("crates")
        .join("mesh-llm-host-runtime")
        .join("src")
        .join("lib.rs");
    let contents = fs::read_to_string(runtime_lib)?;
    extract_runtime_version(repo_root, &contents)
}

fn extract_runtime_version(repo_root: &Path, contents: &str) -> DynResult<String> {
    const LITERAL_PREFIX: &str = "pub const VERSION: &str = \"";
    const CARGO_PKG_VERSION: &str = "pub const VERSION: &str = env!(\"CARGO_PKG_VERSION\");";
    const RELEASE_VERSION_ALIAS: &str = "pub const VERSION: &str = RELEASE_VERSION;";
    for line in contents.lines().map(str::trim) {
        if line == CARGO_PKG_VERSION {
            return host_runtime_package_version(repo_root);
        }
        if line == RELEASE_VERSION_ALIAS {
            return host_runtime_package_version(repo_root);
        }
        if let Some(rest) = line.strip_prefix(LITERAL_PREFIX) {
            return Ok(rest
                .strip_suffix("\";")
                .ok_or("malformed mesh-llm-host-runtime VERSION constant")?
                .to_string());
        }
    }
    Err("missing mesh-llm-host-runtime VERSION constant".into())
}

pub(crate) fn host_runtime_package_version(repo_root: &Path) -> DynResult<String> {
    let runtime_manifest = repo_root
        .join("crates")
        .join("mesh-llm-host-runtime")
        .join("Cargo.toml");
    let runtime_contents = fs::read_to_string(runtime_manifest)?;
    if let Some(version) = extract_manifest_string(&runtime_contents, "version")? {
        return Ok(version);
    }
    if has_manifest_bool(&runtime_contents, "version.workspace", true) {
        let workspace_manifest = repo_root.join("Cargo.toml");
        let workspace_contents = fs::read_to_string(workspace_manifest)?;
        return extract_manifest_string(&workspace_contents, "version")?
            .ok_or_else(|| "missing workspace package version".into());
    }
    Err("missing mesh-llm-host-runtime package version".into())
}

fn extract_manifest_string(contents: &str, key: &str) -> DynResult<Option<String>> {
    let prefix = format!("{key} = \"");
    for line in contents.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix(&prefix) {
            return Ok(Some(
                rest.strip_suffix('"')
                    .ok_or_else(|| format!("malformed manifest {key} value"))?
                    .to_string(),
            ));
        }
    }
    Ok(None)
}

fn has_manifest_bool(contents: &str, key: &str, expected: bool) -> bool {
    let expected_value = if expected { "true" } else { "false" };
    let expected_line = format!("{key} = {expected_value}");
    contents
        .lines()
        .map(str::trim)
        .any(|line| line == expected_line)
}

pub(crate) fn host_supports_shell_parity_checks() -> bool {
    !cfg!(windows)
}

pub(crate) fn repo_root() -> DynResult<PathBuf> {
    // CARGO_MANIFEST_DIR is <repo>/tools/xtask; go up two levels to reach the repo root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "could not determine repo root from xtask manifest directory".into())
}

#[derive(Debug, Deserialize)]
pub(crate) struct CargoMetadata {
    pub(crate) packages: Vec<CargoPackage>,
    pub(crate) workspace_members: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CargoPackage {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) manifest_path: PathBuf,
    #[serde(default)]
    pub(crate) dependencies: Vec<CargoDependency>,
    pub(crate) description: Option<String>,
    pub(crate) license: Option<String>,
    pub(crate) license_file: Option<String>,
    pub(crate) repository: Option<String>,
    pub(crate) readme: Option<String>,
    pub(crate) publish: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CargoDependency {
    pub(crate) name: String,
    pub(crate) req: String,
    pub(crate) kind: Option<String>,
    pub(crate) path: Option<PathBuf>,
}

pub(crate) fn workspace_metadata(repo_root: &Path, context: &str) -> DynResult<CargoMetadata> {
    let mut cargo = Command::new("cargo");
    cargo
        .current_dir(repo_root)
        .arg("metadata")
        .arg("--format-version=1")
        .arg("--no-deps");
    let output = run_command(&mut cargo)?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed while checking {context}: {}",
            trimmed_stderr_or_stdout(&output)
        )
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

pub(crate) fn workspace_package_names(repo_root: &Path) -> DynResult<BTreeSet<String>> {
    let metadata = workspace_metadata(repo_root, "CI crate lists")?;
    let workspace_members = metadata
        .workspace_members
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut names = BTreeSet::new();
    for package in metadata.packages {
        if workspace_members.contains(&package.id) {
            names.insert(package.name);
        }
    }

    if names.is_empty() {
        return Err("cargo metadata returned no workspace package names".into());
    }

    Ok(names)
}

pub(crate) fn script_workspace_members(
    repo_root: &Path,
    relative_path: &str,
) -> DynResult<BTreeSet<String>> {
    let contents = fs::read_to_string(repo_root.join(relative_path))?;
    let mut in_array = false;
    let mut members = BTreeSet::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if !in_array {
            if trimmed == "WORKSPACE_MEMBERS=(" {
                in_array = true;
            }
            continue;
        }

        if trimmed == ")" {
            return Ok(members);
        }

        let Some(member) = trimmed
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
        else {
            return Err(format!(
                "{relative_path} WORKSPACE_MEMBERS: expected quoted crate name, got `{trimmed}`"
            )
            .into());
        };
        if !members.insert(member.to_string()) {
            return Err(format!(
                "{relative_path} WORKSPACE_MEMBERS: duplicate crate name `{member}`"
            )
            .into());
        }
    }

    Err(format!("{relative_path}: missing WORKSPACE_MEMBERS array").into())
}

fn check_subset(
    workspace: &BTreeSet<String>,
    candidates: &BTreeSet<String>,
    context: &str,
) -> DynResult<()> {
    let unknown = candidates
        .difference(workspace)
        .cloned()
        .collect::<Vec<_>>();
    if unknown.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{context}: found crate names not in workspace: {}",
        unknown.join(", ")
    )
    .into())
}

pub(crate) fn check_release_targets_command() -> DynResult<()> {
    release_targets::check_release_targets()
}

pub(crate) fn check_ci_crate_lists_command() -> DynResult<()> {
    let repo_root = repo_root()?;
    workflow_checks::check_ci_script_workspace_members(&repo_root)?;
    workflow_checks::check_ci_crate_test_coverage_files(&repo_root)?;
    check_attestation_default_version(&repo_root)?;
    check_runtime_event_abi_mirror(&repo_root)?;
    println!("repo consistency checks passed: ci-crate-lists");
    Ok(())
}

/// `mesh-llm-runtime-event-contracts` (a publishable dependency-leaf crate)
/// mirrors the raw ABI struct `skippy_ffi::abi::SkippyRuntimeEventV1` in its
/// own test fixture `RawNativeEventV1Fixture` rather than reading
/// `skippy-ffi`'s source directly, because a published crate tarball
/// contains only its own package root (see
/// `publish_consistency::check_publish_literal_includes`). This check keeps
/// that mirror honest by comparing both structs' field names, in
/// declaration order, from this dev-only cross-crate-aware binary.
fn check_runtime_event_abi_mirror(repo_root: &Path) -> DynResult<()> {
    let abi_relative = Path::new("crates/skippy-ffi/src/abi.rs");
    let abi_source = fs::read_to_string(repo_root.join(abi_relative))?;
    let abi_fields = struct_field_names(&abi_source, "pub struct SkippyRuntimeEventV1 {")
        .map_err(|error| format!("{}: {error}", abi_relative.display()))?;

    let mirror_relative = Path::new("crates/mesh-llm-runtime-event-contracts/src/tests/native.rs");
    let mirror_source = fs::read_to_string(repo_root.join(mirror_relative))?;
    let mirror_fields = struct_field_names(&mirror_source, "struct RawNativeEventV1Fixture {")
        .map_err(|error| format!("{}: {error}", mirror_relative.display()))?;

    ensure_eq(
        &abi_fields.join(", "),
        &mirror_fields.join(", "),
        "skippy-ffi SkippyRuntimeEventV1 fields mirrored by mesh-llm-runtime-event-contracts RawNativeEventV1Fixture",
    )
}

/// Extracts field names, in declaration order, from a `struct <Name> { ... }`
/// body located by `decl_marker` (e.g. `"struct Foo {"` or
/// `"pub struct Foo {"`). Matches loosely (optional `pub ` prefix, skips
/// blank/comment lines, splits on the first `:`) so it works for both
/// FFI-exported and plain internal structs without requiring the source to
/// actually parse as Rust.
fn struct_field_names(source: &str, decl_marker: &str) -> DynResult<Vec<String>> {
    let body = source
        .split_once(decl_marker)
        .ok_or_else(|| format!("missing `{decl_marker}` declaration"))?
        .1
        .split_once("\n}")
        .ok_or_else(|| format!("`{decl_marker}` declaration should close"))?
        .0;

    let fields = body
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let trimmed = trimmed.strip_prefix("pub ").unwrap_or(trimmed);
            if trimmed.is_empty() || trimmed.starts_with("//") {
                return None;
            }
            trimmed
                .split_once(':')
                .map(|(name, _)| name.trim().to_string())
        })
        .collect::<Vec<_>>();

    if fields.is_empty() {
        return Err(format!("no fields found for `{decl_marker}`").into());
    }
    Ok(fields)
}

pub(crate) fn check_publish_crates_command() -> DynResult<()> {
    let repo_root = repo_root()?;
    publish_consistency::check_publish_crates_consistency(&repo_root)?;
    println!("repo consistency checks passed: publish-crates");
    Ok(())
}

pub(crate) fn check_test_all_coverage_command() -> DynResult<()> {
    let repo_root = repo_root()?;
    let workspace_crates = workspace_package_names(&repo_root)?;
    let test_all_source = fs::read_to_string(repo_root.join("just/ci.just"))?;
    let (dynamic_targets, excluded_targets, isolated_targets) =
        test_all_test_targets(&test_all_source)?;
    check_subset(
        &workspace_crates,
        &dynamic_targets,
        "test-all dynamic test targets",
    )?;
    check_subset(
        &workspace_crates,
        &isolated_targets,
        "test-all isolated test targets",
    )?;
    check_subset(
        &workspace_crates,
        &excluded_targets,
        "test-all test excludes",
    )?;
    let expected_excluded = dynamic_targets
        .union(&isolated_targets)
        .cloned()
        .collect::<BTreeSet<_>>();
    ensure_set_eq(
        &expected_excluded,
        &excluded_targets,
        "test-all dynamic/exclude target parity",
    )?;

    println!("repo consistency checks passed: test-all-rust-crate-coverage");
    Ok(())
}

fn test_all_test_targets(
    contents: &str,
) -> DynResult<(BTreeSet<String>, BTreeSet<String>, BTreeSet<String>)> {
    let mut lines = contents
        .lines()
        .skip_while(|line| line.trim_end() != "test-all:");
    lines
        .next()
        .ok_or("just/ci.just: missing test-all recipe")?;
    let mut commands = Vec::new();
    let mut current = String::new();
    for line in lines
        .take_while(|line| {
            line.trim().is_empty() || line.chars().next().is_some_and(char::is_whitespace)
        })
        .map(str::trim)
    {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let continued = line.strip_suffix('\\');
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(continued.unwrap_or(line).trim());
        if continued.is_none() {
            commands.push(std::mem::take(&mut current));
        }
    }

    let dynamic = commands
        .iter()
        .find(|command| {
            command.contains("cargo test ")
                && (command.contains(" -p ") || command.contains(" -p="))
        })
        .ok_or("test-all: missing dynamic cargo test command")?;
    let workspace = commands
        .iter()
        .find(|command| command.contains("cargo test --workspace"))
        .ok_or("test-all: missing workspace cargo test command")?;
    let isolated = commands
        .iter()
        .filter(|command| {
            command.contains("cargo test ")
                && (command.contains(" --package ") || command.contains(" --package="))
        })
        .map(|command| command_flag_values(command, "--package"))
        .fold(BTreeSet::new(), |mut targets, values| {
            targets.extend(values);
            targets
        });
    Ok((
        command_flag_values(dynamic, "-p"),
        command_flag_values(workspace, "--exclude"),
        isolated,
    ))
}

fn command_flag_values(command: &str, flag: &str) -> BTreeSet<String> {
    let equals_prefix = format!("{flag}=");
    let mut values = BTreeSet::new();
    let mut arguments = command.split_whitespace();
    while let Some(argument) = arguments.next() {
        if argument == flag {
            if let Some(value) = arguments.next() {
                values.insert(value.to_string());
            }
        } else if let Some(value) = argument.strip_prefix(&equals_prefix) {
            values.insert(value.to_string());
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::{struct_field_names, test_all_test_targets};
    use std::collections::BTreeSet;

    #[test]
    fn struct_field_names_extracts_pub_fields_in_order() {
        let source =
            "#[repr(C)]\npub struct Foo {\n    pub a: u32,\n    // a comment\n    pub b: u64,\n}\n";
        let fields = struct_field_names(source, "pub struct Foo {").unwrap();
        assert_eq!(fields, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn struct_field_names_extracts_private_fields_in_order() {
        let source = "struct Bar {\n    a: u32,\n    b: Vec<u8>,\n}\n";
        let fields = struct_field_names(source, "struct Bar {").unwrap();
        assert_eq!(fields, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn struct_field_names_errors_on_missing_declaration() {
        let error =
            struct_field_names("struct Other {\n    a: u32,\n}\n", "struct Bar {").unwrap_err();
        assert!(error.to_string().contains("missing `struct Bar {`"));
    }

    #[test]
    fn test_all_targets_are_parsed_from_commands() {
        let justfile = r#"
test-all:
    cargo test \
        -p dynamic-a \
        -p dynamic-b
    cargo test --workspace \
        --exclude dynamic-a \
        --exclude dynamic-b

# Another recipe
"#;
        let expected = BTreeSet::from(["dynamic-a".to_string(), "dynamic-b".to_string()]);
        let (dynamic, excluded, isolated) = test_all_test_targets(justfile).unwrap();
        assert_eq!(dynamic, expected);
        assert_eq!(excluded, expected);
        assert!(isolated.is_empty());
    }

    #[test]
    fn test_all_targets_accept_equals_form_flags() {
        let justfile = r#"
test-all:
    cargo test -p=dynamic-a -p=dynamic-b
    cargo test --workspace --exclude=dynamic-a --exclude dynamic-b
"#;
        let expected = BTreeSet::from(["dynamic-a".to_string(), "dynamic-b".to_string()]);
        let (dynamic, excluded, isolated) = test_all_test_targets(justfile).unwrap();
        assert_eq!(dynamic, expected);
        assert_eq!(excluded, expected);
        assert!(isolated.is_empty());
    }

    #[test]
    fn test_all_targets_accepts_isolated_static_tests() {
        let justfile = r#"
test-all:
    cargo test \
        -p dynamic-a
    cargo test --workspace \
        --exclude dynamic-a \
        --exclude static-a
    cargo test --package static-a --no-default-features --lib
"#;
        let dynamic = BTreeSet::from(["dynamic-a".to_string()]);
        let excluded = BTreeSet::from(["dynamic-a".to_string(), "static-a".to_string()]);
        let isolated = BTreeSet::from(["static-a".to_string()]);
        let (actual_dynamic, actual_excluded, actual_isolated) =
            test_all_test_targets(justfile).unwrap();
        assert_eq!(actual_dynamic, dynamic);
        assert_eq!(actual_excluded, excluded);
        assert_eq!(actual_isolated, isolated);
    }
}
