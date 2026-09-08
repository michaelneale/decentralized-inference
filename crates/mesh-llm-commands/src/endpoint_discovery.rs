//! Detect OpenAI-compatible LLM servers already running on this machine and
//! offer to publish their models to the mesh.

mod apply;
mod probe;

pub use apply::{
    EndpointPublishPlan, EndpointPublishStatus, ExistingEndpointEntry, existing_endpoint_entry_at,
    plan_endpoint_publish, run_discover_endpoints,
};
pub use probe::{DiscoveredEndpoint, OPENAI_ENDPOINT_PLUGIN, discover_local_endpoints};
