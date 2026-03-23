//! MCP server for blackboard operations.
//!
//! Run with: `mesh-llm blackboard --mcp`
//! Exposes tools over stdio so any MCP-compatible agent can post, search, and
//! read the mesh blackboard.

use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::io::stdio,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Base URL for the mesh-llm management API.
fn base_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("failed to build http client")
}

// ── Tool parameter types ────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct PostParams {
    /// The message to post to the blackboard.
    pub text: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct SearchParams {
    /// Free-text search query (terms are OR'd — any match counts).
    pub query: String,
    /// Max results to return (default: 20).
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct FeedParams {
    /// Only show items from the last N hours (default: 24).
    #[serde(default)]
    pub since_hours: Option<f64>,
    /// Filter by author name (substring match).
    #[serde(default)]
    pub from: Option<String>,
    /// Max items to return (default: 20).
    #[serde(default)]
    pub limit: Option<usize>,
}

// ── Server ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BlackboardServer {
    port: u16,
    tool_router: ToolRouter<Self>,
}

impl BlackboardServer {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl BlackboardServer {
    /// Post a message to the mesh blackboard. The message is broadcast to all
    /// connected peers. PII/secrets are automatically scrubbed before sending.
    #[tool(
        name = "blackboard_post",
        description = "Post a message to the mesh blackboard. The message is broadcast to all connected peers. Prefix with QUESTION:, STATUS:, FINDING:, TIP: etc. for easy searching."
    )]
    async fn post(&self, Parameters(params): Parameters<PostParams>) -> String {
        // Local PII scrub before sending
        let clean = crate::blackboard::pii_scrub(&params.text);
        let issues = crate::blackboard::pii_check(&clean);
        if !issues.is_empty() {
            return format!(
                "Blocked — PII/secret issues detected:\n{}",
                issues.iter().map(|i| format!("• {i}")).collect::<Vec<_>>().join("\n")
            );
        }

        let client = http_client();
        let body = serde_json::json!({ "text": clean });
        match client
            .post(format!("{}/api/blackboard/post", base_url(self.port)))
            .json(&body)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<serde_json::Value>().await {
                    Ok(item) => format!("Posted (id: {})", item["id"]),
                    Err(_) => "Posted.".into(),
                }
            }
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                format!("Error ({status}): {text}")
            }
            Err(e) => format!("Cannot reach mesh-llm on port {}: {e}", self.port),
        }
    }

    /// Search the mesh blackboard using free-text query. Terms are OR'd — any
    /// matching term counts as a hit. Results are ranked by relevance.
    #[tool(
        name = "blackboard_search",
        description = "Search the mesh blackboard. Terms are OR'd — any matching term is a hit. Results ranked by number of matching terms. Use this to find what other agents/humans have shared."
    )]
    async fn search(&self, Parameters(params): Parameters<SearchParams>) -> String {
        let client = http_client();
        let limit = params.limit.unwrap_or(20);
        match client
            .get(format!("{}/api/blackboard/search", base_url(self.port)))
            .query(&[
                ("q", params.query.as_str()),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<Vec<crate::blackboard::BlackboardItem>>().await {
                    Ok(items) if items.is_empty() => "No results.".into(),
                    Ok(items) => format_items(&items),
                    Err(e) => format!("Failed to parse response: {e}"),
                }
            }
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                format!("Error ({status}): {text}")
            }
            Err(e) => format!("Cannot reach mesh-llm on port {}: {e}", self.port),
        }
    }

    /// Get the recent blackboard feed. Shows what people and agents have been
    /// posting. Optionally filter by author name.
    #[tool(
        name = "blackboard_feed",
        description = "Get the recent blackboard feed — see what other agents and humans have posted. Optionally filter by author name and time range."
    )]
    async fn feed(&self, Parameters(params): Parameters<FeedParams>) -> String {
        let client = http_client();
        let limit = params.limit.unwrap_or(20);
        let hours = params.since_hours.unwrap_or(24.0);
        let since_secs = {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now.saturating_sub((hours * 3600.0) as u64)
        };

        let mut query_params: Vec<(&str, String)> = vec![
            ("limit", limit.to_string()),
            ("since", since_secs.to_string()),
        ];
        if let Some(ref from) = params.from {
            query_params.push(("from", from.clone()));
        }

        match client
            .get(format!("{}/api/blackboard/feed", base_url(self.port)))
            .query(&query_params)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<Vec<crate::blackboard::BlackboardItem>>().await {
                    Ok(items) if items.is_empty() => "Blackboard is empty.".into(),
                    Ok(items) => format_items(&items),
                    Err(e) => format!("Failed to parse response: {e}"),
                }
            }
            Ok(resp) if resp.status().as_u16() == 404 => {
                "Blackboard not enabled. Restart mesh-llm with --blackboard.".into()
            }
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                format!("Error ({status}): {text}")
            }
            Err(e) => format!("Cannot reach mesh-llm on port {}: {e}", self.port),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for BlackboardServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("mesh-blackboard", env!("CARGO_PKG_VERSION"))
                    .with_title("Mesh Blackboard")
                    .with_description(
                        "Shared blackboard across a GPU mesh — post findings, search what others have shared, and read the feed.",
                    ),
            )
            .with_instructions(
                "Use blackboard_feed to see recent messages from other agents and humans on the mesh. \
                 Use blackboard_search to find specific topics. \
                 Use blackboard_post to share findings, questions, or status updates. \
                 Prefix posts with QUESTION:, FINDING:, TIP:, STATUS: etc. for easy searching.",
            )
    }
}

/// Format blackboard items as readable text for the agent.
fn format_items(items: &[crate::blackboard::BlackboardItem]) -> String {
    let mut out = String::new();
    for item in items {
        let ago = {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let diff = now.saturating_sub(item.timestamp);
            if diff < 60 {
                format!("{diff}s ago")
            } else if diff < 3600 {
                format!("{}m ago", diff / 60)
            } else if diff < 86400 {
                format!("{}h ago", diff / 3600)
            } else {
                format!("{}d ago", diff / 86400)
            }
        };
        out.push_str(&format!("[{:x}] {} ({})\n", item.id, item.from, ago));
        out.push_str(&item.text);
        out.push_str("\n\n");
    }
    out.trim_end().to_string()
}

/// Run the MCP server over stdio.
pub async fn run_mcp_server(port: u16) -> anyhow::Result<()> {
    let server = BlackboardServer::new(port);
    let transport = stdio();
    server.serve(transport).await?.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackboard::BlackboardItem;

    #[test]
    fn test_format_items_empty() {
        let items: Vec<BlackboardItem> = vec![];
        assert_eq!(format_items(&items), "");
    }

    #[test]
    fn test_format_items_single() {
        let mut item = BlackboardItem::new("alice".into(), "abc".into(), "FINDING: CUDA OOM fix".into());
        // Pin timestamp to now so the "ago" string is predictable
        item.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let out = format_items(&[item.clone()]);
        assert!(out.contains("alice"));
        assert!(out.contains("FINDING: CUDA OOM fix"));
        assert!(out.contains(&format!("{:x}", item.id)));
    }

    #[test]
    fn test_format_items_multiple() {
        let a = BlackboardItem::new("alice".into(), "a".into(), "first".into());
        let b = BlackboardItem::new("bob".into(), "b".into(), "second".into());
        let out = format_items(&[a, b]);
        assert!(out.contains("alice"));
        assert!(out.contains("bob"));
        assert!(out.contains("first"));
        assert!(out.contains("second"));
    }

    #[test]
    fn test_format_items_ago_minutes() {
        let mut item = BlackboardItem::new("alice".into(), "a".into(), "old msg".into());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        item.timestamp = now - 300; // 5 minutes ago
        let out = format_items(&[item]);
        assert!(out.contains("5m ago"));
    }

    #[test]
    fn test_format_items_ago_hours() {
        let mut item = BlackboardItem::new("bob".into(), "b".into(), "older msg".into());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        item.timestamp = now - 7200; // 2 hours ago
        let out = format_items(&[item]);
        assert!(out.contains("2h ago"));
    }

    #[test]
    fn test_server_construction() {
        let server = BlackboardServer::new(3131);
        assert_eq!(server.port, 3131);
    }

    #[test]
    fn test_server_info() {
        let server = BlackboardServer::new(3131);
        let info = server.get_info();
        assert_eq!(info.server_info.name, "mesh-blackboard");
    }

    /// Posting with PII should be blocked before it hits the network.
    #[tokio::test]
    async fn test_post_blocks_pii() {
        let server = BlackboardServer::new(19999); // no server running here
        let params = PostParams {
            text: "my key is sk-abc123secret456token789".into(),
        };
        let result = server.post(Parameters(params)).await;
        assert!(result.contains("Blocked"), "Should block PII: {result}");
        assert!(result.contains("API key"));
    }

    /// Post to a port with nothing listening should get a connection error, not panic.
    #[tokio::test]
    async fn test_post_no_server() {
        let server = BlackboardServer::new(19999);
        let params = PostParams {
            text: "clean message no secrets here".into(),
        };
        let result = server.post(Parameters(params)).await;
        assert!(result.contains("Cannot reach mesh-llm"), "Should report connection error: {result}");
    }

    /// Search with nothing listening should get a connection error.
    #[tokio::test]
    async fn test_search_no_server() {
        let server = BlackboardServer::new(19999);
        let params = SearchParams {
            query: "CUDA".into(),
            limit: None,
        };
        let result = server.search(Parameters(params)).await;
        assert!(result.contains("Cannot reach mesh-llm"), "Should report connection error: {result}");
    }

    /// Feed with nothing listening should get a connection error.
    #[tokio::test]
    async fn test_feed_no_server() {
        let server = BlackboardServer::new(19999);
        let params = FeedParams {
            since_hours: None,
            from: None,
            limit: None,
        };
        let result = server.feed(Parameters(params)).await;
        assert!(result.contains("Cannot reach mesh-llm"), "Should report connection error: {result}");
    }
}
