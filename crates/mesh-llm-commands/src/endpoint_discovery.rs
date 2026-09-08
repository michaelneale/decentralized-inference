//! Turn OpenAI-compatible servers already running on this machine into a
//! published `openai-endpoint` entry.
//!
//! Detection itself lives in the standalone `endpoint-discovery` crate so the
//! SDK can offer it without depending on the CLI command surface.

mod apply;

pub use apply::{
    EndpointPublishPlan, EndpointPublishStatus, ExistingEndpointEntry, existing_endpoint_entry_at,
    plan_endpoint_publish, run_discover_endpoints,
};
pub use endpoint_discovery::{DiscoveredEndpoint, discover_local_endpoints};

/// Plugin that fronts an already-running OpenAI-compatible server.
pub const OPENAI_ENDPOINT_PLUGIN: &str = "openai-endpoint";
