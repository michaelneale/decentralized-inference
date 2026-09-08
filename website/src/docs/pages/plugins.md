---
title: Plugins
---

# Plugins

Plugins add capabilities to mesh-llm without putting every integration into the core runtime. A plugin is a native process managed by mesh-llm over a local control connection. The host owns lifecycle, MCP and HTTP exposure, routing, and policy; the plugin owns its feature logic and local state.

Use this page to install a plugin. If you are writing one, start with [Developing Plugins](/docs/pages/developing-plugins/).

## Official plugins

The Mesh-LLM organization currently publishes five first-party plugin repositories. They are separate from the main `mesh-llm` repository and release native archives for supported platforms.

| Plugin | Use it for | Install |
| --- | --- | --- |
| [`blackboard`](https://github.com/Mesh-LLM/blackboard) | Share short-lived status, findings, questions, and tips across a mesh. The plugin also ships a Blackboard Agent Skill. | `mesh-llm plugins install blackboard` |
| [`openai-endpoint`](https://github.com/Mesh-LLM/openai-endpoint) | Add an already-running OpenAI-compatible server such as vLLM, TGI, Ollama, or Lemonade Server. | `mesh-llm plugins install openai-endpoint` |
| [`flash-moe`](https://github.com/Mesh-LLM/flash-moe) | Attach a Flash-MoE inference endpoint, or let the plugin supervise a local Flash-MoE process. | `mesh-llm plugins install flash-moe` |
| [`metrics`](https://github.com/Mesh-LLM/metrics) | Advertise metrics support for mesh-llm telemetry. Configure the OTLP destination in mesh-llm, not in the plugin. | `mesh-llm plugins install metrics` |
| [`agents`](https://github.com/Mesh-LLM/agents) | Run mesh-native A2A agents and expose their tools through the mesh MCP endpoint. | `mesh-llm plugins install agents` |

The catalog can also contain community plugins. Search it before installing an unfamiliar integration:

```bash
mesh-llm plugins search
mesh-llm plugins search database
```

Treat a catalog entry as a discovery aid: read the linked repository, check its release assets and permissions, and only then install it.

## Install a plugin

Install the latest compatible release for the current machine:

```bash
mesh-llm plugins install openai-endpoint
```

You can also install directly from a GitHub repository. This is useful when a plugin is not yet in the catalog or when you want to pin a release:

```bash
mesh-llm plugins install Mesh-LLM/openai-endpoint
mesh-llm plugins install Mesh-LLM/openai-endpoint@0.1.2
```

Use the catalog name, such as `agents`, for catalog installs. Use the fully qualified `owner/repository` form, such as `Mesh-LLM/agents`, only when installing directly from GitHub.

Plugin authors can install a local release archive through the same validation
boundary without publishing it first:

```bash
mesh-llm plugins install --archive ./my-plugin-0.1.0-local.tar.gz \
  --name my-plugin --version 0.1.0
```

`--archive` accepts `.tar.gz` or `.zip`, requires `--name`, and conflicts with
the positional catalog/GitHub reference. `--version` is optional and defaults
to `dev`. Rebuild and reinstall local archives; `plugins update` is for GitHub
release sources.

The installer selects a native release archive using the plugin name, version, operating system, and CPU architecture. It does not download GPU-specific plugin variants. If a plugin does not publish an archive for your target, build it from its repository and configure the binary with `command`.

After installing, inspect the recorded source, version, target, and path:

```bash
mesh-llm plugins list
mesh-llm plugins info openai-endpoint
```

## Configure a plugin

Add one `[[plugin]]` table to `~/.mesh-llm/config.toml`:

```toml
[[plugin]]
name = "openai-endpoint"
url = "http://127.0.0.1:8000/v1"
```

## Publish LLM servers already running on this machine

If you already run Ollama, LM Studio, a LiteLLM proxy, vLLM, or any other
OpenAI-compatible server locally, Mesh can find it and publish its models to
your mesh without you looking up ports or hand-editing TOML.

`mesh-llm setup` reports what it finds. To publish it:

```bash
mesh-llm plugins install openai-endpoint
mesh-llm plugins discover           # report only
mesh-llm plugins discover --apply   # write the config entry
```

```text
✓ Found Ollama on :11434 (2 models)
    llama3:8b
    qwen3:4b
✓ Publishing Ollama on :11434 (2 models) via openai-endpoint
```

`--apply` adds (or updates) a single `openai-endpoint` block in
`~/.mesh-llm/config.toml`. It edits the file in place rather than rewriting it,
so your other settings and comments are preserved. Restart Mesh and the
server's models appear in `/v1/models` under their own names and are gossiped
to your peers.

Details worth knowing:

- Probes go to **loopback addresses only**, never traverse a configured HTTP
  proxy, and use a sub-second timeout. They cover the well-known ports `11434`
  (Ollama), `1234` (LM Studio), `4000` (LiteLLM), `8000`, and `8080`. No
  network scanning or port sweeping happens.
- Discovery never overwrites your existing choices. If `openai-endpoint` is
  already configured for another URL, or you have disabled it, `discover`
  reports and stops. An explicit `startup.optional` value is preserved; it is
  only defaulted to `true` when you have not set it.
- Only **one** `openai-endpoint` entry is possible, because a second
  `[[plugin]]` block with the same name is rejected at config load. When
  several servers are running, `discover` selects the one serving the most
  models and names the others as unpublishable.
- A server that has no model loaded is reported but not published.
- Skip probing during setup with `mesh-llm setup --no-discover-endpoints`.

## External-endpoint-only workflow

Mesh can expose an existing OpenAI-compatible provider without loading any
local model or adding a placeholder model.

1. Install the endpoint plugin:

   ```bash
   mesh-llm plugins install openai-endpoint
   ```

2. Configure on-demand mode when configured local models must not load eagerly,
   then add the provider:

   ```toml
   [runtime]
   mode = "on_demand"

   [[plugin]]
   name = "openai-endpoint"
   url = "http://127.0.0.1:8000/v1"
   ```

3. Start the durable runtime:

   ```bash
   mesh-llm serve
   ```

4. Verify the provider's models through Mesh:

   ```bash
   curl -s http://localhost:9337/v1/models | jq '.data[].id'
   ```

5. Send a completion using one listed model:

   ```bash
   curl -s http://localhost:9337/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"<provider-model>","messages":[{"role":"user","content":"Say hello."}]}'
   ```

Plugin-process health and inference-endpoint health are separate. A connected,
healthy plugin process can report its configured provider as unavailable; in
that case inspect the provider URL and `/v1/models` response before reinstalling
the plugin.

See [Runtime Lifecycle](/docs/pages/runtime-lifecycle/) for zero-model and
on-demand behavior, and [OpenAI-Compatible API](/docs/pages/openai-compatible-api/)
for client semantics.

For plugins that do not need an endpoint, the name is usually enough:

```toml
[[plugin]]
name = "blackboard"

[[plugin]]
name = "metrics"

[telemetry]
enabled = true
endpoint = "https://otel.example.com"
```

For Flash-MoE, see the [Flash-MoE repository](https://github.com/Mesh-LLM/flash-moe) for both existing-endpoint and managed-process examples. For agents, see the [agents repository](https://github.com/Mesh-LLM/agents) for agent definitions and runtime configuration.

### Plugin fields

| Field | Meaning |
| --- | --- |
| `name` | Installed plugin identifier. Required. |
| `enabled` | Start the plugin with mesh-llm. Defaults to `true`. |
| `web_ui_enabled` | Show a declared plugin web UI in the console. Defaults to `true` when omitted; it does not start or stop the plugin process. |
| `command` | Explicit executable path or command. Useful for locally built plugins. |
| `args` | Arguments passed to the plugin process. |
| `url` | Optional endpoint passed to the plugin as `MESH_LLM_PLUGIN_URL`. |
| `settings` | Plugin-specific settings declared by the installed plugin. Use `[plugin.settings]` in TOML. |
| `[plugin.startup]` | Startup and failure behavior. |

Startup options include `connect_timeout_secs`, `init_timeout_secs`, `optional`, and `lazy_start`. Mark an integration `optional = true` when mesh-llm should continue starting if that plugin is unavailable. Use `lazy_start = true` for plugins that should start only when directly used.

Plugin settings are validated against the schema exposed by the installed plugin. For example:

```toml
[[plugin]]
name = "my-plugin"

[plugin.settings]
mode = "strict"
retention_days = 14
```

### Plugin web UI

Some plugins declare local, host-projected console pages or a configuration
section. Their process lifecycle and UI projection are independent. To hide a
declared UI while retaining the plugin's MCP, HTTP, and endpoint capabilities:

```toml
[[plugin]]
name = "example-plugin"
enabled = true
web_ui_enabled = false
```

You can also change this preference in the console's **Configuration →
Plugins** tab. The toggle persists only `web_ui_enabled`; it does not install,
start, stop, or disable the plugin. A ready plugin page is available at the
static console path `/plugins/<plugin-name>/<page-id>`. Invalid or missing UI
assets leave the plugin process and non-UI capabilities available.

## Plugin storage

By default, mesh-llm stores plugin metadata under `~/.mesh-llm/plugins/` and extracted plugin files under `~/.mesh-llm/plugins/installed/<name>/`. The metadata record at `~/.mesh-llm/plugins/<name>/plugin-install.json` records the source, version, target, install path, enabled state, and packaged settings schema.

Set `MESH_LLM_PLUGIN_DIR` to use a different store root. With that override, the corresponding paths are `<root>/<name>/plugin-install.json` and `<root>/installed/<name>/`. The extracted directory contains the plugin executable, documentation, and any bundled `skills/` directories. Use `mesh-llm plugins info <name>` to confirm the resolved install path before inspecting files manually.

## Use plugin features

The host aggregates plugin capabilities into the local console and MCP endpoint. Plugin-owned names are namespaced to prevent collisions:

- Blackboard exposes tools such as `blackboard.feed` and `blackboard.post`, plus the `blackboard://snapshot` resource.
- Agents exposes tools such as `agents.get_agents`, `agents.send_message`, and `agents.get_task`.
- Endpoint plugins register inference services that mesh-llm can route to when their endpoint is healthy.

Plugins may also ship Agent Skills. After installing a plugin that includes skills, run:

```bash
mesh-llm skills install
```

The supported agent launchers (`mesh-llm goose`, `mesh-llm pi`, `mesh-llm opencode`, and `mesh-llm claude`) install available plugin skills for that client before starting it.

## Manage installed plugins

```bash
mesh-llm plugins update openai-endpoint
mesh-llm plugins enable openai-endpoint
mesh-llm plugins disable openai-endpoint
mesh-llm plugins delete openai-endpoint
```

Disabling keeps the extracted files and metadata on disk but prevents startup. Deleting removes the extracted files and local metadata. Restart mesh-llm after changing a plugin's configuration unless the plugin's own documentation says that it can be reloaded dynamically.

## Troubleshoot startup

Start with the plugin state:

```bash
mesh-llm plugins info <name>
```

Then check the following:

- `command` points to an executable that exists and can run on this machine.
- `url` includes the correct `/v1` path for OpenAI-compatible endpoints.
- The plugin is enabled and has a compatible release for the local target.
- The endpoint is reachable from the mesh node, not only from your laptop.
- A plugin with required settings is installed before those settings are added to `config.toml`.
- An endpoint can be unhealthy while its plugin process is healthy; check the endpoint itself before reinstalling the plugin.

For the protocol and host/plugin boundary, read [Plugin Architecture](/docs/pages/plugin-architecture/). For the author API and release contract, read [Developing Plugins](/docs/pages/developing-plugins/).

## Further plugin documentation

| Page | What it covers |
| --- | --- |
| [Plugin Architecture](/docs/pages/plugin-architecture/) | Control sessions, side streams, host projections, and ownership boundaries |
| [Developing Plugins](/docs/pages/developing-plugins/) | Rust author API, manifests, packaging, skills, and testing |
| [Plugin Reference](/docs/pages/plugin-reference/) | MCP, HTTP, inference, capabilities, mesh channels, and control messages |
| [Plugin Web UI exemplar](https://github.com/Mesh-LLM/mesh-llm/tree/main/docs/plugins/exemplars/web-ui) | A maintained manifest, package, config, bundle, and lifecycle reference |

Useful next topics for this section are a plugin compatibility matrix by mesh-llm release, a configuration-schema guide with examples, a security and permissions guide for third-party binaries, and a release checklist for plugin authors.
