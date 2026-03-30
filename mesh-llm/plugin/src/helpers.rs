use rmcp::model::{AnnotateAble, RawResource, RawResourceTemplate};
use rmcp::model::{
    CallToolResult, CancelTaskParams, CancelTaskResult, CompleteRequestParams, CompleteResult,
    CompletionInfo, Content, GetPromptRequestParams, GetPromptResult, GetTaskInfoParams,
    GetTaskPayloadResult, GetTaskResult, GetTaskResultParams, Implementation, ListPromptsResult,
    ListResourceTemplatesResult, ListResourcesResult, ListTasksResult, ListToolsResult,
    PaginatedRequestParams, Prompt, PromptArgument, ReadResourceRequestParams, ReadResourceResult,
    Resource, ResourceContents, ResourceTemplate, ServerCapabilities, ServerInfo, Task, TaskStatus,
    Tool,
};
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::{
    context::PluginContext,
    error::{PluginError, PluginResult, PluginRpcResult},
    proto,
};

fn default_arguments() -> serde_json::Value {
    json!({})
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallRequest {
    pub name: String,
    #[serde(default = "default_arguments")]
    pub arguments: serde_json::Value,
}

impl ToolCallRequest {
    pub fn arguments<T: DeserializeOwned>(&self) -> PluginResult<T> {
        serde_json::from_value(self.arguments.clone()).map_err(|err| {
            PluginError::invalid_params(format!(
                "Invalid arguments for tool '{}': {err}",
                self.name
            ))
        })
    }

    pub fn arguments_or_default<T>(&self) -> PluginResult<T>
    where
        T: DeserializeOwned + Default,
    {
        self.arguments()
    }
}

pub fn json_string<T: Serialize>(value: &T) -> PluginResult<String> {
    serde_json::to_string(value).map_err(|err| PluginError::internal(err.to_string()))
}

pub fn json_bytes<T: Serialize>(value: &T) -> PluginResult<Vec<u8>> {
    serde_json::to_vec(value).map_err(|err| PluginError::internal(err.to_string()))
}

pub fn structured_tool_result<T: Serialize>(value: T) -> PluginResult<CallToolResult> {
    let value =
        serde_json::to_value(value).map_err(|err| PluginError::internal(err.to_string()))?;
    Ok(CallToolResult::structured(value))
}

pub fn tool_error(message: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![Content::text(message.into())])
}

pub fn list_tools(tools: Vec<Tool>) -> ListToolsResult {
    ListToolsResult {
        tools,
        meta: None,
        next_cursor: None,
    }
}

pub fn list_prompts(prompts: Vec<Prompt>) -> ListPromptsResult {
    ListPromptsResult {
        prompts,
        meta: None,
        next_cursor: None,
    }
}

pub fn list_resources(resources: Vec<Resource>) -> ListResourcesResult {
    ListResourcesResult {
        resources,
        meta: None,
        next_cursor: None,
    }
}

pub fn list_resource_templates(
    resource_templates: Vec<ResourceTemplate>,
) -> ListResourceTemplatesResult {
    ListResourceTemplatesResult {
        resource_templates,
        meta: None,
        next_cursor: None,
    }
}

pub fn list_tasks(tasks: Vec<Task>) -> ListTasksResult {
    ListTasksResult::new(tasks)
}

pub fn read_resource_result(contents: Vec<ResourceContents>) -> ReadResourceResult {
    ReadResourceResult::new(contents)
}

pub fn get_prompt_result(messages: Vec<rmcp::model::PromptMessage>) -> GetPromptResult {
    GetPromptResult::new(messages)
}

pub fn complete_result(values: Vec<String>) -> PluginResult<CompleteResult> {
    let completion =
        CompletionInfo::with_all_values(values).map_err(PluginError::invalid_params)?;
    Ok(CompleteResult::new(completion))
}

pub fn prompt(
    name: impl Into<String>,
    description: impl Into<String>,
    arguments: Option<Vec<PromptArgument>>,
) -> Prompt {
    Prompt::new(name, Some(description.into()), arguments)
}

pub fn prompt_argument(
    name: impl Into<String>,
    description: impl Into<String>,
    required: bool,
) -> PromptArgument {
    PromptArgument::new(name)
        .with_description(description)
        .with_required(required)
}

pub fn text_resource(uri: impl Into<String>, name: impl Into<String>) -> Resource {
    RawResource::new(uri, name).no_annotation()
}

pub fn resource_template(
    uri_template: impl Into<String>,
    name: impl Into<String>,
) -> ResourceTemplate {
    RawResourceTemplate::new(uri_template, name).no_annotation()
}

pub fn task(
    task_id: impl Into<String>,
    status: TaskStatus,
    created_at: impl Into<String>,
    last_updated_at: impl Into<String>,
) -> Task {
    Task::new(
        task_id.into(),
        status,
        created_at.into(),
        last_updated_at.into(),
    )
}

pub fn get_task_result(task: Task) -> GetTaskResult {
    GetTaskResult { meta: None, task }
}

pub fn get_task_payload_result<T: Serialize>(value: T) -> PluginResult<GetTaskPayloadResult> {
    let value =
        serde_json::to_value(value).map_err(|err| PluginError::internal(err.to_string()))?;
    Ok(GetTaskPayloadResult::new(value))
}

pub fn cancel_task_result(task: Task) -> CancelTaskResult {
    CancelTaskResult { meta: None, task }
}

pub fn plugin_server_info_full(
    implementation_name: impl Into<String>,
    implementation_version: impl Into<String>,
    title: impl Into<String>,
    description: impl Into<String>,
    instructions: Option<impl Into<String>>,
) -> ServerInfo {
    let info = ServerInfo::new(
        ServerCapabilities::builder()
            .enable_tools()
            .enable_tool_list_changed()
            .enable_prompts()
            .enable_prompts_list_changed()
            .enable_resources()
            .enable_resources_list_changed()
            .enable_resources_subscribe()
            .enable_completions()
            .enable_logging()
            .enable_tasks()
            .build(),
    )
    .with_server_info(
        Implementation::new(implementation_name, implementation_version)
            .with_title(title)
            .with_description(description),
    );
    match instructions {
        Some(instructions) => info.with_instructions(instructions.into()),
        None => info,
    }
}

pub fn plugin_server_info(
    implementation_name: impl Into<String>,
    implementation_version: impl Into<String>,
    title: impl Into<String>,
    description: impl Into<String>,
    instructions: Option<impl Into<String>>,
) -> ServerInfo {
    let info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
        .with_server_info(
            Implementation::new(implementation_name, implementation_version)
                .with_title(title)
                .with_description(description),
        );
    match instructions {
        Some(instructions) => info.with_instructions(instructions.into()),
        None => info,
    }
}

pub fn empty_object_schema() -> serde_json::Map<String, serde_json::Value> {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false
    })
    .as_object()
    .cloned()
    .unwrap()
}

pub fn json_schema_for<T: JsonSchema>() -> serde_json::Map<String, serde_json::Value> {
    serde_json::to_value(schemars::schema_for!(T))
        .ok()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_else(|| {
            serde_json::json!({
                "type": "object",
                "additionalProperties": true
            })
            .as_object()
            .cloned()
            .unwrap()
        })
}

pub fn tool_with_schema(
    name: impl Into<String>,
    description: impl Into<String>,
    schema: serde_json::Map<String, serde_json::Value>,
) -> Tool {
    Tool::new(name.into(), description.into(), Arc::new(schema))
}

pub fn json_schema_tool<T: JsonSchema>(
    name: impl Into<String>,
    description: impl Into<String>,
) -> Tool {
    tool_with_schema(name, description, json_schema_for::<T>())
}

pub fn channel_message(
    channel: impl Into<String>,
    target_peer_id: impl Into<String>,
    content_type: impl Into<String>,
    body: Vec<u8>,
    message_kind: impl Into<String>,
) -> proto::ChannelMessage {
    proto::ChannelMessage {
        channel: channel.into(),
        source_peer_id: String::new(),
        target_peer_id: target_peer_id.into(),
        content_type: content_type.into(),
        body,
        message_kind: message_kind.into(),
        correlation_id: String::new(),
        metadata_json: String::new(),
    }
}

pub fn json_channel_message<T: Serialize>(
    channel: impl Into<String>,
    target_peer_id: impl Into<String>,
    message_kind: impl Into<String>,
    payload: &T,
) -> PluginResult<proto::ChannelMessage> {
    Ok(channel_message(
        channel,
        target_peer_id,
        "application/json",
        json_bytes(payload)?,
        message_kind,
    ))
}

pub fn json_reply_channel_message<T: Serialize>(
    message: &proto::ChannelMessage,
    message_kind: impl Into<String>,
    payload: &T,
) -> PluginResult<proto::ChannelMessage> {
    let mut reply = json_channel_message(
        message.channel.clone(),
        message.source_peer_id.clone(),
        message_kind,
        payload,
    )?;
    reply.correlation_id = message.correlation_id.clone();
    Ok(reply)
}

#[allow(clippy::too_many_arguments)]
pub fn bulk_transfer_message(
    kind: i32,
    channel: impl Into<String>,
    target_peer_id: impl Into<String>,
    content_type: impl Into<String>,
    total_bytes: u64,
    offset: u64,
    body: Vec<u8>,
    final_chunk: bool,
) -> proto::BulkTransferMessage {
    proto::BulkTransferMessage {
        kind,
        transfer_id: String::new(),
        channel: channel.into(),
        source_peer_id: String::new(),
        target_peer_id: target_peer_id.into(),
        content_type: content_type.into(),
        correlation_id: String::new(),
        metadata_json: String::new(),
        total_bytes,
        offset,
        body,
        final_chunk,
    }
}

pub fn accept_bulk_transfer_message(
    message: &proto::BulkTransferMessage,
) -> proto::BulkTransferMessage {
    let mut response = bulk_transfer_message(
        proto::bulk_transfer_message::Kind::Accept as i32,
        message.channel.clone(),
        message.source_peer_id.clone(),
        message.content_type.clone(),
        message.total_bytes,
        0,
        Vec::new(),
        false,
    );
    response.transfer_id = message.transfer_id.clone();
    response.correlation_id = message.correlation_id.clone();
    response
}

pub struct BulkTransferSequence {
    pub transfer_id: String,
    pub correlation_id: String,
    pub messages: Vec<proto::BulkTransferMessage>,
}

#[allow(clippy::too_many_arguments)]
pub fn bulk_transfer_sequence(
    channel: impl Into<String>,
    target_peer_id: impl Into<String>,
    content_type: impl Into<String>,
    bytes: Vec<u8>,
    chunk_size: usize,
    correlation_id: impl Into<String>,
    transfer_id: impl Into<String>,
    metadata_json: impl Into<String>,
) -> BulkTransferSequence {
    let channel = channel.into();
    let target_peer_id = target_peer_id.into();
    let content_type = content_type.into();
    let correlation_id = correlation_id.into();
    let transfer_id = transfer_id.into();
    let metadata_json = metadata_json.into();
    let total_bytes = bytes.len() as u64;
    let chunk_size = chunk_size.max(1);

    let mut messages = Vec::new();

    let mut offer = bulk_transfer_message(
        proto::bulk_transfer_message::Kind::Offer as i32,
        channel.clone(),
        target_peer_id.clone(),
        content_type.clone(),
        total_bytes,
        0,
        Vec::new(),
        false,
    );
    offer.transfer_id = transfer_id.clone();
    offer.correlation_id = correlation_id.clone();
    offer.metadata_json = metadata_json.clone();
    messages.push(offer);

    let mut offset = 0usize;
    for chunk in bytes.chunks(chunk_size) {
        let mut message = bulk_transfer_message(
            proto::bulk_transfer_message::Kind::Chunk as i32,
            channel.clone(),
            target_peer_id.clone(),
            content_type.clone(),
            total_bytes,
            offset as u64,
            chunk.to_vec(),
            false,
        );
        message.transfer_id = transfer_id.clone();
        message.correlation_id = correlation_id.clone();
        message.metadata_json = metadata_json.clone();
        messages.push(message);
        offset += chunk.len();
    }

    let mut complete = bulk_transfer_message(
        proto::bulk_transfer_message::Kind::Complete as i32,
        channel,
        target_peer_id,
        content_type,
        total_bytes,
        total_bytes,
        Vec::new(),
        true,
    );
    complete.transfer_id = transfer_id.clone();
    complete.correlation_id = correlation_id.clone();
    complete.metadata_json = metadata_json;
    messages.push(complete);

    BulkTransferSequence {
        transfer_id,
        correlation_id,
        messages,
    }
}

pub fn json_response<T: Serialize>(value: &T) -> PluginRpcResult {
    Ok(proto::envelope::Payload::RpcResponse(proto::RpcResponse {
        result_json: serde_json::to_string(value)
            .map_err(|err| PluginError::internal(err.to_string()))?,
    }))
}

pub fn parse_rpc_params<T: DeserializeOwned>(
    request: &proto::RpcRequest,
) -> Result<T, PluginError> {
    serde_json::from_str(&request.params_json).map_err(|err| {
        PluginError::invalid_params(format!("Invalid params for '{}': {err}", request.method))
    })
}

pub fn parse_tool_call_request(request: &proto::RpcRequest) -> PluginResult<ToolCallRequest> {
    parse_rpc_params(request)
}

pub fn parse_optional_json(raw: &str) -> Option<serde_json::Value> {
    if raw.trim().is_empty() {
        None
    } else {
        serde_json::from_str(raw).ok()
    }
}

pub fn parse_get_prompt_request(
    request: &proto::RpcRequest,
) -> PluginResult<GetPromptRequestParams> {
    parse_rpc_params(request)
}

pub fn parse_read_resource_request(
    request: &proto::RpcRequest,
) -> PluginResult<ReadResourceRequestParams> {
    parse_rpc_params(request)
}

pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = PluginResult<CallToolResult>> + Send + 'a>>;
pub type JsonToolFuture<'a, T> = Pin<Box<dyn Future<Output = PluginResult<T>> + Send + 'a>>;
pub type PromptFuture<'a> =
    Pin<Box<dyn Future<Output = PluginResult<GetPromptResult>> + Send + 'a>>;
pub type ResourceFuture<'a> =
    Pin<Box<dyn Future<Output = PluginResult<ReadResourceResult>> + Send + 'a>>;
pub type CompletionFuture<'a> =
    Pin<Box<dyn Future<Output = PluginResult<CompleteResult>> + Send + 'a>>;
pub type TaskListFuture<'a> =
    Pin<Box<dyn Future<Output = PluginResult<ListTasksResult>> + Send + 'a>>;
pub type TaskInfoFuture<'a> =
    Pin<Box<dyn Future<Output = PluginResult<GetTaskResult>> + Send + 'a>>;
pub type TaskResultFuture<'a> =
    Pin<Box<dyn Future<Output = PluginResult<GetTaskPayloadResult>> + Send + 'a>>;
pub type TaskCancelFuture<'a> =
    Pin<Box<dyn Future<Output = PluginResult<CancelTaskResult>> + Send + 'a>>;

type ToolHandler = Arc<
    dyn for<'a, 'ctx> Fn(ToolCallRequest, &'a mut PluginContext<'ctx>) -> ToolFuture<'a>
        + Send
        + Sync,
>;

pub struct ToolRouter {
    tools: Vec<Tool>,
    handlers: HashMap<String, ToolHandler>,
}

impl ToolRouter {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            handlers: HashMap::new(),
        }
    }

    pub fn add_raw<F>(&mut self, tool: Tool, handler: F)
    where
        F: for<'a, 'ctx> Fn(ToolCallRequest, &'a mut PluginContext<'ctx>) -> ToolFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        let name = tool.name.to_string();
        self.tools.push(tool);
        self.handlers.insert(name, Arc::new(handler));
    }

    pub fn add_json<TArgs, TResult, F>(&mut self, tool: Tool, handler: F)
    where
        TArgs: DeserializeOwned + Send + 'static,
        TResult: Serialize + Send + 'static,
        F: for<'a, 'ctx> Fn(TArgs, &'a mut PluginContext<'ctx>) -> JsonToolFuture<'a, TResult>
            + Send
            + Sync
            + 'static,
    {
        let handler = Arc::new(handler);
        self.add_raw(tool, move |request, context| {
            let handler = Arc::clone(&handler);
            Box::pin(async move {
                let args: TArgs = request.arguments()?;
                let value = handler(args, context).await?;
                structured_tool_result(value)
            })
        });
    }

    pub fn add_json_default<TArgs, TResult, F>(&mut self, tool: Tool, handler: F)
    where
        TArgs: DeserializeOwned + Default + Send + 'static,
        TResult: Serialize + Send + 'static,
        F: for<'a, 'ctx> Fn(TArgs, &'a mut PluginContext<'ctx>) -> JsonToolFuture<'a, TResult>
            + Send
            + Sync
            + 'static,
    {
        let handler = Arc::new(handler);
        self.add_raw(tool, move |request, context| {
            let handler = Arc::clone(&handler);
            Box::pin(async move {
                let args: TArgs = request.arguments_or_default()?;
                let value = handler(args, context).await?;
                structured_tool_result(value)
            })
        });
    }

    pub fn list_tools_result(&self) -> ListToolsResult {
        list_tools(self.tools.clone())
    }

    pub async fn call(
        &self,
        request: ToolCallRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<CallToolResult> {
        let Some(handler) = self.handlers.get(&request.name).cloned() else {
            return Err(PluginError::method_not_found(format!(
                "Unknown tool '{}'",
                request.name
            )));
        };
        handler(request, context).await
    }
}

impl Default for ToolRouter {
    fn default() -> Self {
        Self::new()
    }
}

type PromptHandler = Arc<
    dyn for<'a, 'ctx> Fn(GetPromptRequestParams, &'a mut PluginContext<'ctx>) -> PromptFuture<'a>
        + Send
        + Sync,
>;

pub struct PromptRouter {
    prompts: Vec<Prompt>,
    handlers: HashMap<String, PromptHandler>,
}

impl PromptRouter {
    pub fn new() -> Self {
        Self {
            prompts: Vec::new(),
            handlers: HashMap::new(),
        }
    }

    pub fn add<F>(&mut self, prompt: Prompt, handler: F)
    where
        F: for<'a, 'ctx> Fn(
                GetPromptRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> PromptFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        let name = prompt.name.to_string();
        self.prompts.push(prompt);
        self.handlers.insert(name, Arc::new(handler));
    }

    pub fn list_prompts_result(&self) -> ListPromptsResult {
        list_prompts(self.prompts.clone())
    }

    pub async fn get(
        &self,
        request: GetPromptRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<GetPromptResult> {
        let Some(handler) = self.handlers.get(&request.name).cloned() else {
            return Err(PluginError::invalid_params(format!(
                "Unknown prompt '{}'",
                request.name
            )));
        };
        handler(request, context).await
    }
}

impl Default for PromptRouter {
    fn default() -> Self {
        Self::new()
    }
}

enum ResourceReadMatcher {
    Exact(String),
    Prefix(String),
}

impl ResourceReadMatcher {
    fn matches(&self, uri: &str) -> bool {
        match self {
            Self::Exact(expected) => uri == expected,
            Self::Prefix(prefix) => uri.starts_with(prefix),
        }
    }
}

type ResourceHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            ReadResourceRequestParams,
            &'a mut PluginContext<'ctx>,
        ) -> ResourceFuture<'a>
        + Send
        + Sync,
>;

pub struct ResourceRouter {
    resources: Vec<Resource>,
    resource_templates: Vec<ResourceTemplate>,
    handlers: Vec<(ResourceReadMatcher, ResourceHandler)>,
}

impl ResourceRouter {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            resource_templates: Vec::new(),
            handlers: Vec::new(),
        }
    }

    pub fn add_exact<F>(&mut self, resource: Resource, handler: F)
    where
        F: for<'a, 'ctx> Fn(
                ReadResourceRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> ResourceFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        let uri = resource.raw.uri.to_string();
        self.resources.push(resource);
        self.handlers
            .push((ResourceReadMatcher::Exact(uri), Arc::new(handler)));
    }

    pub fn add_prefix_template<F>(
        &mut self,
        resource_template: ResourceTemplate,
        prefix: impl Into<String>,
        handler: F,
    ) where
        F: for<'a, 'ctx> Fn(
                ReadResourceRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> ResourceFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.resource_templates.push(resource_template);
        self.handlers.push((
            ResourceReadMatcher::Prefix(prefix.into()),
            Arc::new(handler),
        ));
    }

    pub fn list_resources_result(&self) -> ListResourcesResult {
        list_resources(self.resources.clone())
    }

    pub fn list_resource_templates_result(&self) -> ListResourceTemplatesResult {
        list_resource_templates(self.resource_templates.clone())
    }

    pub async fn read(
        &self,
        request: ReadResourceRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<ReadResourceResult> {
        let Some((_, handler)) = self
            .handlers
            .iter()
            .find(|(matcher, _)| matcher.matches(&request.uri))
        else {
            return Err(PluginError::invalid_params(format!(
                "Unknown resource '{}'",
                request.uri
            )));
        };
        handler(request, context).await
    }
}

impl Default for ResourceRouter {
    fn default() -> Self {
        Self::new()
    }
}

enum CompletionMatcher {
    PromptArgument {
        prompt_name: String,
        argument_name: Option<String>,
    },
    ResourceArgument {
        resource_uri: String,
        argument_name: Option<String>,
    },
}

impl CompletionMatcher {
    fn matches(&self, request: &CompleteRequestParams) -> bool {
        match self {
            Self::PromptArgument {
                prompt_name,
                argument_name,
            } => {
                request.r#ref.as_prompt_name() == Some(prompt_name.as_str())
                    && argument_name
                        .as_ref()
                        .map(|name| request.argument.name == *name)
                        .unwrap_or(true)
            }
            Self::ResourceArgument {
                resource_uri,
                argument_name,
            } => {
                request.r#ref.as_resource_uri() == Some(resource_uri.as_str())
                    && argument_name
                        .as_ref()
                        .map(|name| request.argument.name == *name)
                        .unwrap_or(true)
            }
        }
    }
}

type CompletionHandler = Arc<
    dyn for<'a, 'ctx> Fn(CompleteRequestParams, &'a mut PluginContext<'ctx>) -> CompletionFuture<'a>
        + Send
        + Sync,
>;

pub struct CompletionRouter {
    handlers: Vec<(CompletionMatcher, CompletionHandler)>,
}

impl CompletionRouter {
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    pub fn add_prompt_argument_values(
        &mut self,
        prompt_name: impl Into<String>,
        argument_name: impl Into<String>,
        values: Vec<String>,
    ) {
        let values = Arc::new(values);
        self.add_prompt_argument(prompt_name, argument_name, move |_request, _context| {
            let values = values.clone();
            Box::pin(async move { complete_result(values.as_ref().clone()) })
        });
    }

    pub fn add_resource_argument_values(
        &mut self,
        resource_uri: impl Into<String>,
        argument_name: impl Into<String>,
        values: Vec<String>,
    ) {
        let values = Arc::new(values);
        self.add_resource_argument(resource_uri, argument_name, move |_request, _context| {
            let values = values.clone();
            Box::pin(async move { complete_result(values.as_ref().clone()) })
        });
    }

    pub fn add_prompt_argument<F>(
        &mut self,
        prompt_name: impl Into<String>,
        argument_name: impl Into<String>,
        handler: F,
    ) where
        F: for<'a, 'ctx> Fn(
                CompleteRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> CompletionFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.handlers.push((
            CompletionMatcher::PromptArgument {
                prompt_name: prompt_name.into(),
                argument_name: Some(argument_name.into()),
            },
            Arc::new(handler),
        ));
    }

    pub fn add_prompt<F>(&mut self, prompt_name: impl Into<String>, handler: F)
    where
        F: for<'a, 'ctx> Fn(
                CompleteRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> CompletionFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.handlers.push((
            CompletionMatcher::PromptArgument {
                prompt_name: prompt_name.into(),
                argument_name: None,
            },
            Arc::new(handler),
        ));
    }

    pub fn add_resource_argument<F>(
        &mut self,
        resource_uri: impl Into<String>,
        argument_name: impl Into<String>,
        handler: F,
    ) where
        F: for<'a, 'ctx> Fn(
                CompleteRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> CompletionFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.handlers.push((
            CompletionMatcher::ResourceArgument {
                resource_uri: resource_uri.into(),
                argument_name: Some(argument_name.into()),
            },
            Arc::new(handler),
        ));
    }

    pub fn add_resource<F>(&mut self, resource_uri: impl Into<String>, handler: F)
    where
        F: for<'a, 'ctx> Fn(
                CompleteRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> CompletionFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.handlers.push((
            CompletionMatcher::ResourceArgument {
                resource_uri: resource_uri.into(),
                argument_name: None,
            },
            Arc::new(handler),
        ));
    }

    pub async fn complete(
        &self,
        request: CompleteRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<CompleteResult> {
        let Some((_, handler)) = self
            .handlers
            .iter()
            .find(|(matcher, _)| matcher.matches(&request))
        else {
            return complete_result(vec![request.argument.value]);
        };
        handler(request, context).await
    }
}

impl Default for CompletionRouter {
    fn default() -> Self {
        Self::new()
    }
}

type TaskListHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            Option<PaginatedRequestParams>,
            &'a mut PluginContext<'ctx>,
        ) -> TaskListFuture<'a>
        + Send
        + Sync,
>;
type TaskInfoHandler = Arc<
    dyn for<'a, 'ctx> Fn(GetTaskInfoParams, &'a mut PluginContext<'ctx>) -> TaskInfoFuture<'a>
        + Send
        + Sync,
>;
type TaskResultHandler = Arc<
    dyn for<'a, 'ctx> Fn(GetTaskResultParams, &'a mut PluginContext<'ctx>) -> TaskResultFuture<'a>
        + Send
        + Sync,
>;
type TaskCancelHandler = Arc<
    dyn for<'a, 'ctx> Fn(CancelTaskParams, &'a mut PluginContext<'ctx>) -> TaskCancelFuture<'a>
        + Send
        + Sync,
>;

pub struct TaskRouter {
    list_handler: Option<TaskListHandler>,
    info_handler: Option<TaskInfoHandler>,
    result_handler: Option<TaskResultHandler>,
    cancel_handler: Option<TaskCancelHandler>,
}

impl TaskRouter {
    pub fn new() -> Self {
        Self {
            list_handler: None,
            info_handler: None,
            result_handler: None,
            cancel_handler: None,
        }
    }

    pub fn with_list<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                Option<PaginatedRequestParams>,
                &'a mut PluginContext<'ctx>,
            ) -> TaskListFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.list_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_get_info<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(GetTaskInfoParams, &'a mut PluginContext<'ctx>) -> TaskInfoFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.info_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_get_result<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                GetTaskResultParams,
                &'a mut PluginContext<'ctx>,
            ) -> TaskResultFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.result_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_cancel<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(CancelTaskParams, &'a mut PluginContext<'ctx>) -> TaskCancelFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.cancel_handler = Some(Arc::new(handler));
        self
    }

    pub async fn list_tasks(
        &self,
        request: Option<PaginatedRequestParams>,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListTasksResult>> {
        match &self.list_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    pub async fn get_task_info(
        &self,
        request: GetTaskInfoParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskResult>> {
        match &self.info_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    pub async fn get_task_result(
        &self,
        request: GetTaskResultParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskPayloadResult>> {
        match &self.result_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    pub async fn cancel_task(
        &self,
        request: CancelTaskParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CancelTaskResult>> {
        match &self.cancel_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }
}

impl Default for TaskRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Default)]
pub struct SubscriptionSet {
    uris: BTreeSet<String>,
}

impl SubscriptionSet {
    pub fn subscribe(&mut self, uri: impl Into<String>) {
        self.uris.insert(uri.into());
    }

    pub fn unsubscribe(&mut self, uri: &str) {
        self.uris.remove(uri);
    }

    pub fn list(&self) -> Vec<String> {
        self.uris.iter().cloned().collect()
    }
}

#[derive(Clone, Debug)]
pub struct TaskRecord<T> {
    pub task: Task,
    pub payload: T,
}

#[derive(Clone, Debug, Default)]
pub struct TaskStore<T> {
    tasks: BTreeMap<String, TaskRecord<T>>,
}

impl<T> TaskStore<T> {
    pub fn insert(&mut self, task: Task, payload: T) {
        self.tasks
            .insert(task.task_id.clone(), TaskRecord { task, payload });
    }

    pub fn list(&self) -> Vec<Task> {
        self.tasks.values().map(|task| task.task.clone()).collect()
    }

    pub fn get(&self, task_id: &str) -> PluginResult<&TaskRecord<T>> {
        self.tasks
            .get(task_id)
            .ok_or_else(|| PluginError::invalid_params(format!("Unknown task '{task_id}'")))
    }

    pub fn get_mut(&mut self, task_id: &str) -> PluginResult<&mut TaskRecord<T>> {
        self.tasks
            .get_mut(task_id)
            .ok_or_else(|| PluginError::invalid_params(format!("Unknown task '{task_id}'")))
    }

    pub fn values(&self) -> impl Iterator<Item = &TaskRecord<T>> {
        self.tasks.values()
    }
}
