//! Pipeline routing: multi-model collaboration before final dispatch.
//!
//! Instead of routing every request to a single model, pipeline routing
//! can involve multiple models in sequence:
//!
//! - **Pre-plan**: small fast model analyses the request, produces a plan,
//!   which gets injected as context for the strong model.
//! - **Chat augment**: fast model drafts a response, second model refines it.
//!
//! The pipeline runs at the HTTP level (not TCP tunnel) so it can read
//! and modify request/response bodies.

use reqwest::Client;
use serde_json::{json, Value};
use std::time::Instant;

/// A pipeline stage result.
#[derive(Debug)]
pub struct PlanResult {
    pub plan_text: String,
    pub model_used: String,
    pub elapsed_ms: u64,
}

/// Ask a small model to create a plan/analysis for a task.
///
/// The planner gets the user's message and produces a brief plan:
/// what files to look at, what approach to take, potential pitfalls.
/// This plan gets injected as a system message for the strong model.
pub async fn pre_plan(
    client: &Client,
    planner_url: &str,
    planner_model: &str,
    user_messages: &[Value],
) -> Result<PlanResult, String> {
    // Extract the last user message as the task
    let last_user = user_messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"));

    let task = last_user
        .and_then(|m| {
            let content = m.get("content")?;
            // Handle string content
            if let Some(s) = content.as_str() {
                return Some(s.to_string());
            }
            // Handle array content (multi-part: [{"type":"text","text":"..."}])
            if let Some(parts) = content.as_array() {
                let texts: Vec<&str> = parts
                    .iter()
                    .filter_map(|p| {
                        if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                            p.get("text").and_then(|t| t.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                if !texts.is_empty() {
                    return Some(texts.join("\n"));
                }
            }
            None
        })
        .unwrap_or_default();

    if task.is_empty() {
        return Err("no user message found".into());
    }

    let plan_request = json!({
        "model": planner_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a task planner for a coding assistant. Given a user's request, produce a brief plan (3-5 bullet points). Focus on:\n- What files or information to look at first\n- What approach to take\n- Potential pitfalls or edge cases\n- What order to do things in\n\nBe concise. No code. Just a plan."
            },
            {
                "role": "user",
                "content": task
            }
        ],
        "max_tokens": 256,
        "temperature": 0.3,
        "stream": false
    });

    let start = Instant::now();
    let resp = client
        .post(format!("{planner_url}/v1/chat/completions"))
        .json(&plan_request)
        .send()
        .await
        .map_err(|e| format!("planner request failed: {e}"))?;

    let body: Value = resp
        .json()
        .await
        .map_err(|e| format!("planner response parse failed: {e}"))?;

    let plan_text = body
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let model_used = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(planner_model)
        .to_string();

    let elapsed_ms = start.elapsed().as_millis() as u64;

    if plan_text.is_empty() {
        return Err("planner returned empty response".into());
    }

    Ok(PlanResult {
        plan_text,
        model_used,
        elapsed_ms,
    })
}

/// Inject a plan into a chat completion request body.
///
/// Adds the plan as a system message right before the last user message,
/// so the strong model sees it as context.
pub fn inject_plan(body: &mut Value, plan: &PlanResult) {
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(m) => m,
        None => return,
    };

    // Find the position of the last user message
    let last_user_idx = messages
        .iter()
        .rposition(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"));

    let plan_message = json!({
        "role": "system",
        "content": format!(
            "[Task Plan from {} — {}ms]\n{}",
            plan.model_used, plan.elapsed_ms, plan.plan_text
        )
    });

    match last_user_idx {
        Some(idx) => messages.insert(idx, plan_message),
        None => messages.push(plan_message),
    }
}

/// Decide if a request should go through pipeline routing.
///
/// Pipeline adds latency, so only use it when it's worth it:
/// - Agentic tasks (needs_tools + complex content)
/// - Deep complexity
/// - Skip for simple chat, quick lookups, etc.
pub fn should_pipeline(classification: &crate::router::Classification) -> bool {
    use crate::router::{Category, Complexity};

    // Only pipeline for complex tasks that need tools
    if !classification.needs_tools {
        return false;
    }

    match classification.complexity {
        Complexity::Deep => true,
        Complexity::Moderate => {
            // Moderate + tool use — pipeline if it's code or reasoning
            matches!(
                classification.category,
                Category::Code | Category::Reasoning | Category::ToolCall
            )
        }
        Complexity::Quick => false,
    }
}

/// Full pipeline: classify → optionally pre-plan → return augmented body + target model.
///
/// Returns (body, model_name, pipeline_used).
/// If pipeline is skipped, body is returned unmodified.
#[allow(dead_code)]
pub async fn route_with_pipeline(
    client: &Client,
    planner_url: &str,
    planner_model: &str,
    strong_model: &str,
    mut body: Value,
    classification: &crate::router::Classification,
) -> (Value, String, Option<PlanResult>) {
    if !should_pipeline(classification) {
        return (body, strong_model.to_string(), None);
    }

    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();

    match pre_plan(client, planner_url, planner_model, &messages).await {
        Ok(plan) => {
            tracing::info!(
                "pipeline: pre-plan by {} in {}ms ({} chars)",
                plan.model_used,
                plan.elapsed_ms,
                plan.plan_text.len()
            );
            inject_plan(&mut body, &plan);
            (body, strong_model.to_string(), Some(plan))
        }
        Err(e) => {
            tracing::warn!("pipeline: pre-plan failed: {e}, falling back to direct routing");
            (body, strong_model.to_string(), None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::Client;
    use serde_json::json;

    #[test]
    fn test_inject_plan_before_last_user_message() {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Read server.py and fix bugs"}
            ]
        });
        let plan = PlanResult {
            plan_text: "1. Read the file\n2. Find bugs\n3. Fix them".into(),
            model_used: "hermes-7b".into(),
            elapsed_ms: 150,
        };
        inject_plan(&mut body, &plan);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["role"], "system"); // plan injected before user msg
        assert!(msgs[1]["content"].as_str().unwrap().contains("Task Plan"));
        assert_eq!(msgs[2]["role"], "user"); // user msg moved to end
    }

    #[test]
    fn test_inject_plan_multi_turn() {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is a hash table?"},
                {"role": "assistant", "content": "A hash table is..."},
                {"role": "user", "content": "Now fix the bugs in server.py"}
            ]
        });
        let plan = PlanResult {
            plan_text: "1. Read server.py\n2. Fix the bugs".into(),
            model_used: "hermes-7b".into(),
            elapsed_ms: 100,
        };
        inject_plan(&mut body, &plan);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 5);
        // Plan should be before the LAST user message, not the first
        assert_eq!(msgs[3]["role"], "system");
        assert_eq!(msgs[4]["role"], "user");
        assert_eq!(msgs[4]["content"], "Now fix the bugs in server.py");
    }

    #[test]
    fn test_should_pipeline_complex_code() {
        use crate::router::{Category, Classification, Complexity};
        let cl = Classification {
            category: Category::Code,
            complexity: Complexity::Deep,
            needs_tools: true,
        };
        assert!(should_pipeline(&cl));
    }

    #[test]
    fn test_should_not_pipeline_simple_chat() {
        use crate::router::{Category, Classification, Complexity};
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Quick,
            needs_tools: false,
        };
        assert!(!should_pipeline(&cl));
    }

    #[test]
    fn test_should_not_pipeline_quick_tool_use() {
        use crate::router::{Category, Classification, Complexity};
        let cl = Classification {
            category: Category::ToolCall,
            complexity: Complexity::Quick,
            needs_tools: true,
        };
        assert!(!should_pipeline(&cl));
    }

    #[test]
    fn test_should_pipeline_moderate_code_with_tools() {
        use crate::router::{Category, Classification, Complexity};
        let cl = Classification {
            category: Category::Code,
            complexity: Complexity::Moderate,
            needs_tools: true,
        };
        assert!(should_pipeline(&cl));
    }

    #[test]
    fn test_should_not_pipeline_moderate_chat() {
        use crate::router::{Category, Classification, Complexity};
        let cl = Classification {
            category: Category::Chat,
            complexity: Complexity::Moderate,
            needs_tools: false,
        };
        assert!(!should_pipeline(&cl));
    }

    #[tokio::test]
    async fn test_route_with_pipeline_preplan_failure_falls_back_unmodified() {
        use crate::router::{Category, Classification, Complexity};

        let client = Client::new();
        let body = json!({
            "messages": [
                {"role": "user", "content": "Inspect the codebase and call tools to fix the bug"}
            ]
        });
        let classification = Classification {
            category: Category::Code,
            complexity: Complexity::Deep,
            needs_tools: true,
        };

        let (out_body, model, plan) = route_with_pipeline(
            &client,
            "http://127.0.0.1:9",
            "planner",
            "strong",
            body.clone(),
            &classification,
        )
        .await;

        assert_eq!(out_body, body);
        assert_eq!(model, "strong");
        assert!(plan.is_none());
    }
}
