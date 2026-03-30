#!/usr/bin/env bash

set -euo pipefail

REPO="${MESH_LLM_INSTALL_REPO:-michaelneale/mesh-llm}"
REPO_REF="${MESH_LLM_INSTALL_REF:-main}"
INSTALL_DIR="${MESH_LLM_INSTALL_DIR:-$HOME/.local/bin}"
INSTALL_FLAVOR="${MESH_LLM_INSTALL_FLAVOR:-}"
INSTALL_SERVICE="${MESH_LLM_INSTALL_SERVICE:-0}"
INSTALL_SERVICE_ARGS="${MESH_LLM_INSTALL_SERVICE_ARGS:-}"
INSTALL_SERVICE_START="${MESH_LLM_INSTALL_SERVICE_START:-1}"

SERVICE_NAME="mesh-llm"
SERVICE_LABEL="com.mesh-llm.mesh-llm"
SERVICE_CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/mesh-llm"
SERVICE_ARGS_FILE="$SERVICE_CONFIG_DIR/service.args"
SERVICE_ENV_FILE="$SERVICE_CONFIG_DIR/service.env"
SERVICE_RUNNER="$SERVICE_CONFIG_DIR/run-service.sh"
SYSTEMD_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SYSTEMD_UNIT_PATH="$SYSTEMD_UNIT_DIR/$SERVICE_NAME.service"
LAUNCHD_AGENT_DIR="$HOME/Library/LaunchAgents"
LAUNCHD_PLIST_PATH="$LAUNCHD_AGENT_DIR/$SERVICE_LABEL.plist"
LAUNCHD_LOG_DIR="$HOME/Library/Logs/mesh-llm"
LAUNCHD_STDOUT_LOG="$LAUNCHD_LOG_DIR/stdout.log"
LAUNCHD_STDERR_LOG="$LAUNCHD_LOG_DIR/stderr.log"
SYSTEMD_ARGS_COMMENT_PREFIX="# mesh-llm-install-args: "
DIST_DIR="dist"
SYSTEMD_TEMPLATE_PATH="$DIST_DIR/$SERVICE_NAME.service"
LAUNCHD_TEMPLATE_PATH="$DIST_DIR/$SERVICE_LABEL.plist"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: required command not found: $1" >&2
        exit 1
    fi
}

path_contains_install_dir() {
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) return 0 ;;
        *) return 1 ;;
    esac
}

bool_is_true() {
    local value="${1:-}"
    value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    case "$value" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

usage() {
    cat <<EOF
Usage: install.sh [--service] [--service-args '<mesh-llm args>'] [--no-start-service]

Options:
  --service                  Install a per-user background service for this platform.
  --service-args '<args>'    Seed the installed service with these mesh-llm args.
                             The string is shell-split once during install.
  --no-start-service         Install the service files but do not start them yet.
  -h, --help                 Show this help text.

Environment overrides:
  MESH_LLM_INSTALL_DIR
  MESH_LLM_INSTALL_FLAVOR
  MESH_LLM_INSTALL_REF=main
  MESH_LLM_INSTALL_SERVICE=1
  MESH_LLM_INSTALL_SERVICE_ARGS='--auto'
  MESH_LLM_INSTALL_SERVICE_START=0
EOF
}

parse_args() {
    while (($# > 0)); do
        case "$1" in
            --service)
                INSTALL_SERVICE=1
                ;;
            --service-args)
                if (($# < 2)); then
                    echo "error: --service-args requires a value" >&2
                    exit 1
                fi
                INSTALL_SERVICE_ARGS="$2"
                shift
                ;;
            --no-start-service)
                INSTALL_SERVICE_START=0
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "error: unknown argument: $1" >&2
                echo >&2
                usage >&2
                exit 1
                ;;
        esac
        shift
    done
}

platform_id() {
    printf "%s/%s\n" "$(uname -s)" "$(uname -m)"
}

probe_nvidia() {
    command -v nvidia-smi >/dev/null 2>&1 ||
        command -v nvcc >/dev/null 2>&1 ||
        [[ -e /dev/nvidiactl ]] ||
        [[ -d /proc/driver/nvidia/gpus ]]
}

probe_rocm() {
    command -v rocm-smi >/dev/null 2>&1 ||
        command -v rocminfo >/dev/null 2>&1 ||
        command -v hipcc >/dev/null 2>&1 ||
        [[ -x /opt/rocm/bin/hipcc ]]
}

probe_vulkan() {
    if command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary >/dev/null 2>&1; then
        return 0
    fi
    if command -v glslc >/dev/null 2>&1; then
        if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists vulkan 2>/dev/null; then
            return 0
        fi
        if [[ -f /usr/include/vulkan/vulkan.h || -f /usr/local/include/vulkan/vulkan.h ]]; then
            return 0
        fi
        if [[ -n "${VULKAN_SDK:-}" ]]; then
            return 0
        fi
    fi
    return 1
}

supported_flavors() {
    case "$(platform_id)" in
        Darwin/arm64)
            echo "metal"
            ;;
        Linux/x86_64)
            echo "cpu cuda rocm vulkan"
            ;;
        *)
            echo "error: unsupported platform: $(platform_id)" >&2
            exit 1
            ;;
    esac
}

recommended_flavor() {
    case "$(platform_id)" in
        Darwin/arm64)
            echo "metal"
            ;;
        Linux/x86_64)
            if probe_nvidia; then
                echo "cuda"
            elif probe_rocm; then
                echo "rocm"
            elif probe_vulkan; then
                echo "vulkan"
            else
                echo "cpu"
            fi
            ;;
        *)
            echo "error: unsupported platform: $(platform_id)" >&2
            exit 1
            ;;
    esac
}

recommendation_reason() {
    case "$(recommended_flavor)" in
        metal)
            echo "Apple Silicon host detected."
            ;;
        cuda)
            echo "NVIDIA tooling or devices were detected."
            ;;
        rocm)
            echo "ROCm/HIP tooling was detected."
            ;;
        vulkan)
            echo "Vulkan tooling was detected."
            ;;
        cpu)
            echo "No supported GPU runtime was detected."
            ;;
    esac
}

validate_flavor() {
    local flavor="$1"
    local supported
    for supported in $(supported_flavors); do
        if [[ "$supported" == "$flavor" ]]; then
            return 0
        fi
    done
    echo "error: unsupported flavor '$flavor' for $(platform_id)" >&2
    exit 1
}

choose_flavor() {
    local recommended
    recommended="$(recommended_flavor)"

    if [[ -n "$INSTALL_FLAVOR" ]]; then
        validate_flavor "$INSTALL_FLAVOR"
        echo "$INSTALL_FLAVOR"
        return 0
    fi

    if [[ ! -t 0 || ! -t 1 ]]; then
        echo "$recommended"
        return 0
    fi

    local flavors
    flavors=($(supported_flavors))

    if [[ ${#flavors[@]} -eq 1 ]]; then
        echo "$recommended"
        return 0
    fi

    echo "Mesh LLM installer"
    echo "Platform: $(platform_id)"
    echo "Recommended flavor: $recommended"
    echo "Reason: $(recommendation_reason)"
    echo
    echo "Available flavors:"

    local index=1
    local flavor
    for flavor in "${flavors[@]}"; do
        if [[ "$flavor" == "$recommended" ]]; then
            echo "  $index. $flavor (recommended)"
        else
            echo "  $index. $flavor"
        fi
        index=$((index + 1))
    done

    echo
    local reply
    read -r -p "Install which flavor? [$recommended] " reply
    reply="${reply:-$recommended}"

    if [[ "$reply" =~ ^[0-9]+$ ]]; then
        local selection=$((reply - 1))
        if (( selection >= 0 && selection < ${#flavors[@]} )); then
            reply="${flavors[$selection]}"
        fi
    fi

    validate_flavor "$reply"
    echo "$reply"
}

asset_name() {
    local flavor="$1"
    case "$(platform_id)" in
        Darwin/arm64)
            echo "mesh-llm-aarch64-apple-darwin.tar.gz"
            ;;
        Linux/x86_64)
            case "$flavor" in
                cpu) echo "mesh-llm-x86_64-unknown-linux-gnu.tar.gz" ;;
                cuda) echo "mesh-llm-x86_64-unknown-linux-gnu-cuda.tar.gz" ;;
                rocm) echo "mesh-llm-x86_64-unknown-linux-gnu-rocm.tar.gz" ;;
                vulkan) echo "mesh-llm-x86_64-unknown-linux-gnu-vulkan.tar.gz" ;;
                *)
                    echo "error: unsupported Linux flavor '$flavor'" >&2
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "error: unsupported platform: $(platform_id)" >&2
            exit 1
            ;;
    esac
}

stale_binary_names() {
    cat <<'EOF'
mesh-llm
rpc-server
llama-server
llama-moe-split
rpc-server-cpu
llama-server-cpu
rpc-server-cuda
llama-server-cuda
rpc-server-rocm
llama-server-rocm
rpc-server-vulkan
llama-server-vulkan
rpc-server-metal
llama-server-metal
EOF
}

remove_stale_binaries() {
    mkdir -p "$INSTALL_DIR"
    local name
    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        rm -f "$INSTALL_DIR/$name"
    done < <(stale_binary_names)
}

install_bundle() {
    local bundle_dir="$1"
    remove_stale_binaries

    local file
    for file in "$bundle_dir"/*; do
        mv -f "$file" "$INSTALL_DIR/"
    done
}

write_service_args_file() {
    local path="$1"
    local raw_args="$2"

    mkdir -p "$(dirname "$path")"

    if ! eval "set -- $raw_args"; then
        echo "error: failed to parse service args: $raw_args" >&2
        exit 1
    fi

    {
        echo "# One mesh-llm CLI argument per line."
        echo "# Blank lines and lines beginning with # are ignored."
        echo "# Examples:"
        echo "#   --auto"
        echo "#   --model"
        echo "#   Qwen2.5-3B"
        local arg
        for arg in "$@"; do
            printf '%s\n' "$arg"
        done
    } > "$path"
}

serialize_shell_args() {
    local out=""
    local escaped
    local arg
    for arg in "$@"; do
        printf -v escaped '%q' "$arg"
        out+="${out:+ }$escaped"
    done
    printf '%s' "$out"
}

parse_service_args() {
    local raw_args="$1"

    if ! eval "set -- $raw_args"; then
        echo "error: failed to parse service args: $raw_args" >&2
        exit 1
    fi

    SERVICE_ARGS_VALUES=("$@")
    SERVICE_ARGS_SERIALIZED="$(serialize_shell_args "$@")"
}

read_existing_systemd_args() {
    if [[ ! -f "$SYSTEMD_UNIT_PATH" ]]; then
        return 1
    fi

    local existing
    existing="$(sed -n "s/^${SYSTEMD_ARGS_COMMENT_PREFIX}//p" "$SYSTEMD_UNIT_PATH" | head -n 1)"
    [[ -n "$existing" ]] || return 1
    printf '%s\n' "$existing"
}

resolve_systemd_service_args() {
    local existing

    if [[ -n "$INSTALL_SERVICE_ARGS" ]]; then
        parse_service_args "$INSTALL_SERVICE_ARGS"
        return
    fi

    if existing="$(read_existing_systemd_args)"; then
        parse_service_args "$existing"
        return
    fi

    parse_service_args "--auto"
}

systemd_escape_assignment_value() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//%/%%}"
    printf '%s' "$value"
}

systemd_quote_token() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    value="${value//$/$$}"
    value="${value//%/%%}"
    printf '"%s"' "$value"
}

local_template_path() {
    local rel_path="$1"
    local source_path="${BASH_SOURCE[0]-}"
    local script_dir

    if [[ -z "$source_path" || "$source_path" != */* ]]; then
        return 1
    fi

    script_dir="$(cd "$(dirname "$source_path")" && pwd)"
    [[ -f "$script_dir/$rel_path" ]] || return 1
    printf '%s\n' "$script_dir/$rel_path"
}

template_stream() {
    local rel_path="$1"
    local local_path

    if local_path="$(local_template_path "$rel_path")"; then
        cat "$local_path"
        return 0
    fi

    curl -fsSL "https://raw.githubusercontent.com/${REPO}/${REPO_REF}/${rel_path}"
}

render_template_to_file() {
    local template_path="$1"
    local output_path="$2"
    shift 2

    local -a replacements=("$@")
    local -a env_vars=()
    local pair
    local key
    local value
    for pair in "${replacements[@]}"; do
        key="${pair%%=*}"
        value="${pair#*=}"
        env_vars+=("TPL_${key}=${value}")
    done

    env "${env_vars[@]}" awk '
        BEGIN {
            split("ARGS_METADATA ENV_LINES", multiline_keys, " ");
            for (i in multiline_keys) {
                multiline[multiline_keys[i]] = 1;
            }
        }
        {
            line = $0;
            for (name in multiline) {
                marker = "@" name "@";
                if (line == marker) {
                    print ENVIRON["TPL_" name];
                    next;
                }
            }
            for (var in ENVIRON) {
                if (index(var, "TPL_") != 1) {
                    continue;
                }
                name = substr(var, 5);
                if (name in multiline) {
                    continue;
                }
                marker = "@" name "@";
                gsub(marker, ENVIRON[var], line);
            }
            print line;
        }
    ' < <(template_stream "$template_path") > "$output_path"
}

ensure_service_args_file() {
    if [[ -n "$INSTALL_SERVICE_ARGS" ]]; then
        write_service_args_file "$SERVICE_ARGS_FILE" "$INSTALL_SERVICE_ARGS"
        return
    fi

    if [[ -f "$SERVICE_ARGS_FILE" ]]; then
        return
    fi

    write_service_args_file "$SERVICE_ARGS_FILE" "--auto"
}

ensure_service_env_file() {
    if [[ -f "$SERVICE_ENV_FILE" ]]; then
        return
    fi

    mkdir -p "$(dirname "$SERVICE_ENV_FILE")"
    {
        echo "# Optional environment variables for mesh-llm."
        echo "# Use plain KEY=value lines."
        echo "# Example:"
        echo "# MESH_LLM_NO_SELF_UPDATE=1"
    } > "$SERVICE_ENV_FILE"
}

write_service_runner() {
    mkdir -p "$(dirname "$SERVICE_RUNNER")"

    cat > "$SERVICE_RUNNER" <<EOF
#!/usr/bin/env bash

set -euo pipefail

BIN="$INSTALL_DIR/mesh-llm"
ARGS_FILE="$SERVICE_ARGS_FILE"
ENV_FILE="$SERVICE_ENV_FILE"

if [[ ! -x "\$BIN" ]]; then
    echo "mesh-llm binary not found or not executable: \$BIN" >&2
    exit 1
fi

if [[ -f "\$ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    . "\$ENV_FILE"
    set +a
fi

args=()
if [[ -f "\$ARGS_FILE" ]]; then
    while IFS= read -r line || [[ -n "\$line" ]]; do
        line="\${line%\$'\\r'}"
        case "\$line" in
            ""|\#*) continue ;;
        esac
        args+=("\$line")
    done < "\$ARGS_FILE"
fi

exec "\$BIN" "\${args[@]}"
EOF

    chmod +x "$SERVICE_RUNNER"
}

ensure_launchd_service_files() {
    mkdir -p "$SERVICE_CONFIG_DIR"
    ensure_service_args_file
    ensure_service_env_file
    write_service_runner
}

install_systemd_service() {
    need_cmd systemctl
    mkdir -p "$SERVICE_CONFIG_DIR" "$SYSTEMD_UNIT_DIR"
    ensure_service_env_file
    resolve_systemd_service_args

    local env_lines=""
    local exec_line
    local i
    local value
    for ((i = 0; i < ${#SERVICE_ARGS_VALUES[@]}; i++)); do
        value="$(systemd_escape_assignment_value "${SERVICE_ARGS_VALUES[$i]}")"
        env_lines+="Environment=\"MESH_LLM_ARG_${i}=${value}\"\n"
    done

    exec_line="ExecStart=$(systemd_quote_token "$INSTALL_DIR/mesh-llm")"
    for ((i = 0; i < ${#SERVICE_ARGS_VALUES[@]}; i++)); do
        exec_line+=" \${MESH_LLM_ARG_${i}}"
    done

    render_template_to_file "$SYSTEMD_TEMPLATE_PATH" "$SYSTEMD_UNIT_PATH" \
        "ARGS_METADATA=$SYSTEMD_ARGS_COMMENT_PREFIX$SERVICE_ARGS_SERIALIZED" \
        "SERVICE_ENV_FILE=$SERVICE_ENV_FILE" \
        "ENV_LINES=$(printf '%b' "$env_lines")" \
        "EXEC_LINE=$exec_line"

    systemctl --user daemon-reload || true

    if bool_is_true "$INSTALL_SERVICE_START"; then
        if systemctl --user enable "$SERVICE_NAME.service" &&
            (systemctl --user restart "$SERVICE_NAME.service" ||
                systemctl --user start "$SERVICE_NAME.service"); then
            echo "Installed and started systemd user service: $SERVICE_NAME.service"
        else
            echo "Installed $SYSTEMD_UNIT_PATH" >&2
            echo "warning: could not start the systemd user service automatically." >&2
            echo "Start it with: systemctl --user enable --now $SERVICE_NAME.service" >&2
        fi
    else
        echo "Installed $SYSTEMD_UNIT_PATH"
        echo "Start it with: systemctl --user enable --now $SERVICE_NAME.service"
    fi

    echo "Command: $exec_line"
    echo "Optional env: $SERVICE_ENV_FILE"
    echo "Edit unit args: $SYSTEMD_UNIT_PATH"
    echo "Logs: journalctl --user -u $SERVICE_NAME.service -f"
    echo "Boot without login (optional): sudo loginctl enable-linger \$USER"
}

install_launchd_service() {
    need_cmd launchctl
    ensure_launchd_service_files
    mkdir -p "$LAUNCHD_AGENT_DIR" "$LAUNCHD_LOG_DIR"

    render_template_to_file "$LAUNCHD_TEMPLATE_PATH" "$LAUNCHD_PLIST_PATH" \
        "SERVICE_LABEL=$SERVICE_LABEL" \
        "SERVICE_RUNNER=$SERVICE_RUNNER" \
        "HOME_DIR=$HOME" \
        "STDOUT_LOG=$LAUNCHD_STDOUT_LOG" \
        "STDERR_LOG=$LAUNCHD_STDERR_LOG"

    local launch_domain="gui/$(id -u)"
    if bool_is_true "$INSTALL_SERVICE_START"; then
        launchctl bootout "$launch_domain" "$LAUNCHD_PLIST_PATH" >/dev/null 2>&1 || true
        if launchctl bootstrap "$launch_domain" "$LAUNCHD_PLIST_PATH"; then
            launchctl enable "$launch_domain/$SERVICE_LABEL" >/dev/null 2>&1 || true
            launchctl kickstart -k "$launch_domain/$SERVICE_LABEL" >/dev/null 2>&1 || true
            echo "Installed and started launchd agent: $SERVICE_LABEL"
        else
            echo "Installed $LAUNCHD_PLIST_PATH" >&2
            echo "warning: could not start the launchd agent automatically." >&2
            echo "Start it with: launchctl bootstrap $launch_domain $LAUNCHD_PLIST_PATH" >&2
        fi
    else
        echo "Installed $LAUNCHD_PLIST_PATH"
        echo "Start it with: launchctl bootstrap $launch_domain $LAUNCHD_PLIST_PATH"
    fi

    echo "Service args: $SERVICE_ARGS_FILE"
    echo "Optional env: $SERVICE_ENV_FILE"
    echo "Logs: $LAUNCHD_STDOUT_LOG and $LAUNCHD_STDERR_LOG"
}

install_service() {
    case "$(uname -s)" in
        Darwin)
            install_launchd_service
            ;;
        Linux)
            install_systemd_service
            ;;
        *)
            echo "error: service install is not supported on $(uname -s)" >&2
            exit 1
            ;;
    esac
}

main() {
    parse_args "$@"
    need_cmd curl
    need_cmd tar
    need_cmd mktemp

    local flavor
    flavor="$(choose_flavor)"
    local asset
    asset="$(asset_name "$flavor")"
    local url="https://github.com/${REPO}/releases/latest/download/${asset}"

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local tmp_dir_escaped
    printf -v tmp_dir_escaped '%q' "$tmp_dir"
    trap "rm -rf -- $tmp_dir_escaped" EXIT

    local archive="$tmp_dir/$asset"
    echo "Installing flavor: $flavor"
    echo "Downloading $url"
    curl -fsSL "$url" -o "$archive"

    tar -xzf "$archive" -C "$tmp_dir"

    if [[ ! -d "$tmp_dir/mesh-bundle" ]]; then
        echo "error: release archive did not contain mesh-bundle/" >&2
        exit 1
    fi

    install_bundle "$tmp_dir/mesh-bundle"

    echo "Installed $asset to $INSTALL_DIR"

    if bool_is_true "$INSTALL_SERVICE"; then
        echo
        install_service
    fi

    if ! path_contains_install_dir; then
        echo
        echo "$INSTALL_DIR is not on your PATH."
        echo "Add it with one of these commands:"
        echo
        echo "bash:"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc"
        echo "  source ~/.bashrc"
        echo
        echo "zsh:"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.zshrc"
        echo "  source ~/.zshrc"
    fi
}

if [[ "${BASH_SOURCE[0]-}" == "$0" || ( -z "${BASH_SOURCE[0]-}" && "$0" == "bash" ) ]]; then
    main "$@"
fi
