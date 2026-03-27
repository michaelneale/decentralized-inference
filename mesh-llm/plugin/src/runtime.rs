use anyhow::{bail, Result};
use rmcp::model::{
    CallToolResult, CancelTaskParams, CancelTaskResult, CompleteRequestParams, CompleteResult,
    GetPromptRequestParams, GetPromptResult, GetTaskInfoParams, GetTaskPayloadResult,
    GetTaskResult, GetTaskResultParams, ListPromptsResult, ListResourceTemplatesResult,
    ListResourcesResult, ListTasksResult, ListToolsResult, PaginatedRequestParams,
    ReadResourceRequestParams, ReadResourceResult, ServerInfo, SetLevelRequestParams,
    SubscribeRequestParams, UnsubscribeRequestParams,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::{
    context::PluginContext,
    error::{PluginError, PluginResult, PluginRpcResult},
    helpers::{
        json_response, parse_get_prompt_request, parse_read_resource_request, parse_rpc_params,
        parse_tool_call_request, CompletionRouter, PromptRouter, ResourceRouter, TaskRouter,
        ToolCallRequest, ToolRouter,
    },
    io::{connect_from_env, read_envelope, write_envelope, LocalStream},
    proto, PROTOCOL_VERSION,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshVisibility {
    #[default]
    Private,
    Public,
}

impl MeshVisibility {
    fn from_proto(value: i32) -> Self {
        match proto::MeshVisibility::try_from(value).unwrap_or(proto::MeshVisibility::Unspecified) {
            proto::MeshVisibility::Public => Self::Public,
            _ => Self::Private,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginStartupPolicy {
    #[default]
    Any,
    PrivateMeshOnly,
    PublicMeshOnly,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PluginInitializeRequest {
    pub host_protocol_version: u32,
    pub host_version: String,
    pub host_context_json: String,
    pub mesh_visibility: MeshVisibility,
    pub plugin_config_json: String,
}

impl From<proto::InitializeRequest> for PluginInitializeRequest {
    fn from(value: proto::InitializeRequest) -> Self {
        Self {
            host_protocol_version: value.host_protocol_version,
            host_version: value.host_version,
            host_context_json: value.host_context_json,
            mesh_visibility: MeshVisibility::from_proto(value.mesh_visibility),
            plugin_config_json: value.plugin_config_json,
        }
    }
}

#[derive(Clone)]
pub struct PluginMetadata {
    plugin_id: String,
    plugin_version: String,
    server_info: ServerInfo,
    capabilities: Vec<String>,
    startup_policy: PluginStartupPolicy,
}

impl PluginMetadata {
    pub fn new(
        plugin_id: impl Into<String>,
        plugin_version: impl Into<String>,
        server_info: ServerInfo,
    ) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            plugin_version: plugin_version.into(),
            server_info,
            capabilities: Vec::new(),
            startup_policy: PluginStartupPolicy::Any,
        }
    }

    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn with_startup_policy(mut self, startup_policy: PluginStartupPolicy) -> Self {
        self.startup_policy = startup_policy;
        self
    }
}

type InitializeFuture<'a> = Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>>;
type InitFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
type HealthFuture<'a> = Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
type SubscribeFuture<'a> = Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>>;
type SetLogLevelFuture<'a> = Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>>;

type InitializeHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            PluginInitializeRequest,
            &'a mut PluginContext<'ctx>,
        ) -> InitializeFuture<'a>
        + Send
        + Sync,
>;
type InitHandler =
    Arc<dyn for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> InitFuture<'a> + Send + Sync>;
type HealthHandler =
    Arc<dyn for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> HealthFuture<'a> + Send + Sync>;
type SubscribeHandler = Arc<
    dyn for<'a, 'ctx> Fn(SubscribeRequestParams, &'a mut PluginContext<'ctx>) -> SubscribeFuture<'a>
        + Send
        + Sync,
>;
type UnsubscribeHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            UnsubscribeRequestParams,
            &'a mut PluginContext<'ctx>,
        ) -> SubscribeFuture<'a>
        + Send
        + Sync,
>;
type SetLogLevelHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            SetLevelRequestParams,
            &'a mut PluginContext<'ctx>,
        ) -> SetLogLevelFuture<'a>
        + Send
        + Sync,
>;
type ChannelHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::ChannelMessage, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
        + Send
        + Sync,
>;
type BulkHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::BulkTransferMessage, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
        + Send
        + Sync,
>;
type MeshEventHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::MeshEvent, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
        + Send
        + Sync,
>;

#[crate::async_trait]
pub trait Plugin: Send {
    fn plugin_id(&self) -> &str;
    fn plugin_version(&self) -> String;
    fn server_info(&self) -> ServerInfo;

    fn capabilities(&self) -> Vec<String> {
        Vec::new()
    }

    async fn initialize(
        &mut self,
        _request: PluginInitializeRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<()> {
        Ok(())
    }

    async fn on_initialized(&mut self, _context: &mut PluginContext<'_>) -> Result<()> {
        Ok(())
    }

    async fn health(&mut self, _context: &mut PluginContext<'_>) -> Result<String> {
        Ok("ok".into())
    }

    async fn list_tools(
        &mut self,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListToolsResult>> {
        Ok(None)
    }

    async fn call_tool(
        &mut self,
        _request: ToolCallRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CallToolResult>> {
        Ok(None)
    }

    async fn list_prompts(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListPromptsResult>> {
        Ok(None)
    }

    async fn get_prompt(
        &mut self,
        _request: GetPromptRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetPromptResult>> {
        Ok(None)
    }

    async fn list_resources(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourcesResult>> {
        Ok(None)
    }

    async fn read_resource(
        &mut self,
        _request: ReadResourceRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ReadResourceResult>> {
        Ok(None)
    }

    async fn list_resource_templates(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourceTemplatesResult>> {
        Ok(None)
    }

    async fn subscribe_resource(
        &mut self,
        _request: SubscribeRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn unsubscribe_resource(
        &mut self,
        _request: UnsubscribeRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn complete(
        &mut self,
        _request: CompleteRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CompleteResult>> {
        Ok(None)
    }

    async fn set_log_level(
        &mut self,
        _request: SetLevelRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn list_tasks(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListTasksResult>> {
        Ok(None)
    }

    async fn get_task_info(
        &mut self,
        _request: GetTaskInfoParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskResult>> {
        Ok(None)
    }

    async fn get_task_result(
        &mut self,
        _request: GetTaskResultParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskPayloadResult>> {
        Ok(None)
    }

    async fn cancel_task(
        &mut self,
        _request: CancelTaskParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CancelTaskResult>> {
        Ok(None)
    }

    async fn handle_rpc(
        &mut self,
        request: proto::RpcRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginRpcResult {
        match request.method.as_str() {
            "tools/list" => match self.list_tools(context).await? {
                Some(result) => json_response(&result),
                None => Err(PluginError::method_not_found(
                    "Unsupported MCP method 'tools/list'",
                )),
            },
            "tools/call" => {
                let tool_call = parse_tool_call_request(&request)?;
                match self.call_tool(tool_call, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tools/call'",
                    )),
                }
            }
            "prompts/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_prompts(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'prompts/list'",
                    )),
                }
            }
            "prompts/get" => {
                let params = parse_get_prompt_request(&request)?;
                match self.get_prompt(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'prompts/get'",
                    )),
                }
            }
            "resources/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_resources(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/list'",
                    )),
                }
            }
            "resources/read" => {
                let params = parse_read_resource_request(&request)?;
                match self.read_resource(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/read'",
                    )),
                }
            }
            "resources/templates/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_resource_templates(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/templates/list'",
                    )),
                }
            }
            "resources/subscribe" => {
                let params: SubscribeRequestParams = parse_rpc_params(&request)?;
                match self.subscribe_resource(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/subscribe'",
                    )),
                }
            }
            "resources/unsubscribe" => {
                let params: UnsubscribeRequestParams = parse_rpc_params(&request)?;
                match self.unsubscribe_resource(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/unsubscribe'",
                    )),
                }
            }
            "completion/complete" => {
                let params: CompleteRequestParams = parse_rpc_params(&request)?;
                match self.complete(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'completion/complete'",
                    )),
                }
            }
            "logging/setLevel" => {
                let params: SetLevelRequestParams = parse_rpc_params(&request)?;
                match self.set_log_level(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'logging/setLevel'",
                    )),
                }
            }
            "tasks/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_tasks(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/list'",
                    )),
                }
            }
            "tasks/get" => {
                let params: GetTaskInfoParams = parse_rpc_params(&request)?;
                match self.get_task_info(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/get'",
                    )),
                }
            }
            "tasks/result" => {
                let params: GetTaskResultParams = parse_rpc_params(&request)?;
                match self.get_task_result(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/result'",
                    )),
                }
            }
            "tasks/cancel" => {
                let params: CancelTaskParams = parse_rpc_params(&request)?;
                match self.cancel_task(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/cancel'",
                    )),
                }
            }
            _ => Err(PluginError::method_not_found(format!(
                "Unsupported MCP method '{}'",
                request.method
            ))),
        }
    }

    async fn on_rpc_notification(
        &mut self,
        _notification: proto::RpcNotification,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_channel_message(
        &mut self,
        _message: proto::ChannelMessage,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_bulk_transfer_message(
        &mut self,
        _message: proto::BulkTransferMessage,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_mesh_event(
        &mut self,
        _event: proto::MeshEvent,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_host_error(
        &mut self,
        error: proto::ErrorResponse,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        bail!("host error: {}", error.message)
    }
}

pub struct SimplePlugin {
    metadata: PluginMetadata,
    tool_router: Option<ToolRouter>,
    prompt_router: Option<PromptRouter>,
    resource_router: Option<ResourceRouter>,
    completion_router: Option<CompletionRouter>,
    task_router: Option<TaskRouter>,
    initialize_handler: Option<InitializeHandler>,
    on_initialized: Option<InitHandler>,
    health_handler: Option<HealthHandler>,
    subscribe_handler: Option<SubscribeHandler>,
    unsubscribe_handler: Option<UnsubscribeHandler>,
    set_log_level_handler: Option<SetLogLevelHandler>,
    channel_handler: Option<ChannelHandler>,
    bulk_handler: Option<BulkHandler>,
    mesh_event_handler: Option<MeshEventHandler>,
}

impl SimplePlugin {
    pub fn new(metadata: PluginMetadata) -> Self {
        Self {
            metadata,
            tool_router: None,
            prompt_router: None,
            resource_router: None,
            completion_router: None,
            task_router: None,
            initialize_handler: None,
            on_initialized: None,
            health_handler: None,
            subscribe_handler: None,
            unsubscribe_handler: None,
            set_log_level_handler: None,
            channel_handler: None,
            bulk_handler: None,
            mesh_event_handler: None,
        }
    }

    pub fn with_tool_router(mut self, router: ToolRouter) -> Self {
        self.tool_router = Some(router);
        self
    }

    pub fn with_prompt_router(mut self, router: PromptRouter) -> Self {
        self.prompt_router = Some(router);
        self
    }

    pub fn with_resource_router(mut self, router: ResourceRouter) -> Self {
        self.resource_router = Some(router);
        self
    }

    pub fn with_completion_router(mut self, router: CompletionRouter) -> Self {
        self.completion_router = Some(router);
        self
    }

    pub fn with_task_router(mut self, router: TaskRouter) -> Self {
        self.task_router = Some(router);
        self
    }

    pub fn on_initialize<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                PluginInitializeRequest,
                &'a mut PluginContext<'ctx>,
            ) -> InitializeFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.initialize_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_initialized<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> InitFuture<'a> + Send + Sync + 'static,
    {
        self.on_initialized = Some(Arc::new(handler));
        self
    }

    pub fn with_health<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> HealthFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.health_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_subscribe_resource<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                SubscribeRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> SubscribeFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.subscribe_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_unsubscribe_resource<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                UnsubscribeRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> SubscribeFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.unsubscribe_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_set_log_level<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                SetLevelRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> SetLogLevelFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.set_log_level_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_channel_message<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(proto::ChannelMessage, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.channel_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_bulk_transfer_message<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                proto::BulkTransferMessage,
                &'a mut PluginContext<'ctx>,
            ) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.bulk_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_mesh_event<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(proto::MeshEvent, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.mesh_event_handler = Some(Arc::new(handler));
        self
    }
}

#[crate::async_trait]
impl Plugin for SimplePlugin {
    fn plugin_id(&self) -> &str {
        &self.metadata.plugin_id
    }

    fn plugin_version(&self) -> String {
        self.metadata.plugin_version.clone()
    }

    fn server_info(&self) -> ServerInfo {
        self.metadata.server_info.clone()
    }

    fn capabilities(&self) -> Vec<String> {
        self.metadata.capabilities.clone()
    }

    async fn initialize(
        &mut self,
        request: PluginInitializeRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<()> {
        match self.metadata.startup_policy {
            PluginStartupPolicy::Any => {}
            PluginStartupPolicy::PrivateMeshOnly
                if request.mesh_visibility != MeshVisibility::Private =>
            {
                return Err(PluginError::startup_disabled(format!(
                    "Plugin '{}' requires a private mesh",
                    self.metadata.plugin_id
                )));
            }
            PluginStartupPolicy::PublicMeshOnly
                if request.mesh_visibility != MeshVisibility::Public =>
            {
                return Err(PluginError::startup_disabled(format!(
                    "Plugin '{}' requires a public mesh",
                    self.metadata.plugin_id
                )));
            }
            PluginStartupPolicy::PrivateMeshOnly | PluginStartupPolicy::PublicMeshOnly => {}
        }
        match &self.initialize_handler {
            Some(handler) => handler(request, context).await,
            None => Ok(()),
        }
    }

    async fn on_initialized(&mut self, context: &mut PluginContext<'_>) -> Result<()> {
        match &self.on_initialized {
            Some(handler) => handler(context).await,
            None => Ok(()),
        }
    }

    async fn health(&mut self, context: &mut PluginContext<'_>) -> Result<String> {
        match &self.health_handler {
            Some(handler) => handler(context).await,
            None => Ok("ok".into()),
        }
    }

    async fn list_tools(
        &mut self,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListToolsResult>> {
        Ok(self
            .tool_router
            .as_ref()
            .map(|router| router.list_tools_result()))
    }

    async fn call_tool(
        &mut self,
        request: ToolCallRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CallToolResult>> {
        match &self.tool_router {
            Some(router) => Ok(Some(router.call(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_prompts(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListPromptsResult>> {
        Ok(self
            .prompt_router
            .as_ref()
            .map(|router| router.list_prompts_result()))
    }

    async fn get_prompt(
        &mut self,
        request: GetPromptRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetPromptResult>> {
        match &self.prompt_router {
            Some(router) => Ok(Some(router.get(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_resources(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourcesResult>> {
        Ok(self
            .resource_router
            .as_ref()
            .map(|router| router.list_resources_result()))
    }

    async fn read_resource(
        &mut self,
        request: ReadResourceRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ReadResourceResult>> {
        match &self.resource_router {
            Some(router) => Ok(Some(router.read(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_resource_templates(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourceTemplatesResult>> {
        Ok(self
            .resource_router
            .as_ref()
            .map(|router| router.list_resource_templates_result()))
    }

    async fn subscribe_resource(
        &mut self,
        request: SubscribeRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        match &self.subscribe_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn unsubscribe_resource(
        &mut self,
        request: UnsubscribeRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        match &self.unsubscribe_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn complete(
        &mut self,
        request: CompleteRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CompleteResult>> {
        match &self.completion_router {
            Some(router) => Ok(Some(router.complete(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn set_log_level(
        &mut self,
        request: SetLevelRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        match &self.set_log_level_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_tasks(
        &mut self,
        request: Option<PaginatedRequestParams>,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListTasksResult>> {
        match &self.task_router {
            Some(router) => router.list_tasks(request, context).await,
            None => Ok(None),
        }
    }

    async fn get_task_info(
        &mut self,
        request: GetTaskInfoParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskResult>> {
        match &self.task_router {
            Some(router) => router.get_task_info(request, context).await,
            None => Ok(None),
        }
    }

    async fn get_task_result(
        &mut self,
        request: GetTaskResultParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskPayloadResult>> {
        match &self.task_router {
            Some(router) => router.get_task_result(request, context).await,
            None => Ok(None),
        }
    }

    async fn cancel_task(
        &mut self,
        request: CancelTaskParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CancelTaskResult>> {
        match &self.task_router {
            Some(router) => router.cancel_task(request, context).await,
            None => Ok(None),
        }
    }

    async fn on_channel_message(
        &mut self,
        message: proto::ChannelMessage,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.channel_handler {
            Some(handler) => handler(message, context).await,
            None => Ok(()),
        }
    }

    async fn on_bulk_transfer_message(
        &mut self,
        message: proto::BulkTransferMessage,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.bulk_handler {
            Some(handler) => handler(message, context).await,
            None => Ok(()),
        }
    }

    async fn on_mesh_event(
        &mut self,
        event: proto::MeshEvent,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.mesh_event_handler {
            Some(handler) => handler(event, context).await,
            None => Ok(()),
        }
    }
}

pub struct PluginRuntime;

impl PluginRuntime {
    pub async fn run<P: Plugin>(plugin: P) -> Result<()> {
        let stream = connect_from_env().await?;
        Self::run_with_stream(plugin, stream).await
    }

    pub async fn run_with_stream<P: Plugin>(mut plugin: P, mut stream: LocalStream) -> Result<()> {
        loop {
            let envelope = read_envelope(&mut stream).await?;
            let request_id = envelope.request_id;
            let plugin_id = plugin.plugin_id().to_string();

            match envelope.payload {
                Some(proto::envelope::Payload::InitializeRequest(request)) => {
                    let init_result = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        plugin
                            .initialize(PluginInitializeRequest::from(request), &mut context)
                            .await
                    };
                    if let Err(err) = init_result {
                        let response = proto::Envelope {
                            protocol_version: PROTOCOL_VERSION,
                            plugin_id: plugin_id.clone(),
                            request_id,
                            payload: Some(proto::envelope::Payload::ErrorResponse(
                                err.into_error_response(),
                            )),
                        };
                        write_envelope(&mut stream, &response).await?;
                        break;
                    }
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::InitializeResponse(
                            proto::InitializeResponse {
                                plugin_id: plugin_id.clone(),
                                plugin_protocol_version: PROTOCOL_VERSION,
                                plugin_version: plugin.plugin_version(),
                                server_info_json: serde_json::to_string(&plugin.server_info())?,
                                capabilities: plugin.capabilities(),
                            },
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_initialized(&mut context).await?;
                }
                Some(proto::envelope::Payload::HealthRequest(_)) => {
                    let detail = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        plugin.health(&mut context).await?
                    };
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::HealthResponse(
                            proto::HealthResponse {
                                status: proto::health_response::Status::Ok as i32,
                                detail,
                            },
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                }
                Some(proto::envelope::Payload::ShutdownRequest(_)) => {
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::ShutdownResponse(
                            proto::ShutdownResponse {},
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                    break;
                }
                Some(proto::envelope::Payload::RpcRequest(request)) => {
                    let payload = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        match plugin.handle_rpc(request, &mut context).await {
                            Ok(payload) => payload,
                            Err(err) => {
                                proto::envelope::Payload::ErrorResponse(err.into_error_response())
                            }
                        }
                    };
                    write_envelope(
                        &mut stream,
                        &proto::Envelope {
                            protocol_version: PROTOCOL_VERSION,
                            plugin_id: plugin_id.clone(),
                            request_id,
                            payload: Some(payload),
                        },
                    )
                    .await?;
                }
                Some(proto::envelope::Payload::RpcNotification(notification)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin
                        .on_rpc_notification(notification, &mut context)
                        .await?;
                }
                Some(proto::envelope::Payload::ChannelMessage(message)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_channel_message(message, &mut context).await?;
                }
                Some(proto::envelope::Payload::BulkTransferMessage(message)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin
                        .on_bulk_transfer_message(message, &mut context)
                        .await?;
                }
                Some(proto::envelope::Payload::MeshEvent(event)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_mesh_event(event, &mut context).await?;
                }
                Some(proto::envelope::Payload::ErrorResponse(error)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_host_error(error, &mut context).await?;
                }
                _ => {}
            }
        }

        Ok(())
    }
}
