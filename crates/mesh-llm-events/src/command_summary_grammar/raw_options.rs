use super::vocabulary::is_static_summary_token;

#[derive(Clone, Copy)]
struct CommandPath<'a> {
    tokens: [Option<&'a str>; 4],
    len: usize,
    valid: bool,
}

impl CommandPath<'_> {
    fn matches(self, expected: &[&str]) -> bool {
        self.valid
            && self.len == expected.len()
            && expected
                .iter()
                .enumerate()
                .all(|(index, token)| self.tokens[index] == Some(*token))
    }
}

pub(super) fn raw_option_is_allowed(tokens: &[&str], index: usize, option: &str) -> bool {
    let prefix = &tokens[..index];
    match option {
        "--backend" => matches_backend_prefix(prefix),
        "--mode" => matches_mode_prefix(prefix),
        "--port" => matches_port_prefix(prefix),
        _ => false,
    }
}

fn command_path<'a>(tokens: &'a [&'a str], end: usize) -> CommandPath<'a> {
    let mut path = CommandPath {
        tokens: [None; 4],
        len: 0,
        valid: true,
    };
    for token in &tokens[..end] {
        if !is_static_summary_token(token) {
            continue;
        }
        if path.len >= path.tokens.len() {
            path.valid = false;
            return path;
        }
        path.tokens[path.len] = Some(*token);
        path.len += 1;
    }
    path
}

fn matches_backend_prefix(prefix: &[&str]) -> bool {
    let path = command_path(prefix, prefix.len());
    path.matches(&["mesh-llm", "gpus", "run-benchmark"])
        && prefix == ["mesh-llm", "gpus", "run-benchmark"]
}

fn matches_mode_prefix(prefix: &[&str]) -> bool {
    prefix == ["mesh-llm", "runtime", "guardrails"]
}

fn matches_port_prefix(prefix: &[&str]) -> bool {
    matches!(
        prefix,
        ["mesh-llm", "status"]
            | ["mesh-llm", "load"]
            | ["mesh-llm", "unload"]
            | ["mesh-llm", "goose"]
            | ["mesh-llm", "claude"]
            | ["mesh-llm", "doctor", "split"]
            | ["mesh-llm", "doctor", "split", "--json"]
            | ["mesh-llm", "doctor", "split", "--json", "--json"]
            | ["mesh-llm", "runtime", "status"]
            | ["mesh-llm", "runtime", "load"]
            | ["mesh-llm", "runtime", "unload"]
            | ["mesh-llm", "runtime", "guardrails", "--mode", "disabled"]
            | ["mesh-llm", "runtime", "guardrails", "--mode", "metrics"]
            | ["mesh-llm", "runtime", "guardrails", "--mode", "enforce"]
            | [
                "mesh-llm",
                "runtime",
                "guardrails",
                "--mode",
                "disabled",
                "--json",
            ]
            | [
                "mesh-llm",
                "runtime",
                "guardrails",
                "--mode",
                "metrics",
                "--json",
            ]
            | [
                "mesh-llm",
                "runtime",
                "guardrails",
                "--mode",
                "enforce",
                "--json",
            ]
            | ["mesh-llm", "runtime", "bootstrap"]
            | ["mesh-llm", "runtime", "bootstrap", "--json"]
            | ["mesh-llm", "runtime", "remote"]
            | ["mesh-llm", "runtime", "remote", "--json"]
            | ["mesh-llm", "runtime", "remote-model"]
            | ["mesh-llm", "runtime", "remote-model", "--json"]
            | ["mesh-llm", "runtime", "apply-config"]
            | ["mesh-llm", "runtime", "apply-config", "--json"]
            | ["mesh-llm", "kv-cache", "status"]
            | ["mesh-llm", "kv-cache", "status", "--json"]
            | ["mesh-llm", "kv-cache", "prune"]
            | ["mesh-llm", "kv-cache", "prune", "--yes"]
            | ["mesh-llm", "kv-cache", "prune", "--json"]
            | ["mesh-llm", "kv-cache", "prune", "--yes", "--json"]
            | ["mesh-llm", "kv-cache", "clear"]
            | ["mesh-llm", "kv-cache", "clear", "--yes"]
            | ["mesh-llm", "kv-cache", "clear", "--json"]
            | ["mesh-llm", "kv-cache", "clear", "--yes", "--json"]
    )
}
