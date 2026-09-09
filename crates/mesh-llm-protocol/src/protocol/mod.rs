pub mod convert;
pub mod v0;
use anyhow::Result;
pub use convert::*;
use iroh::endpoint::{ConnectOptions, Connection};
use iroh::{Endpoint, EndpointAddr};
use prost::Message;
pub use v0::*;
pub const ALPN_CONTROL_V1: &[u8] = b"mesh-llm-control/1";
pub const ALPN_V1: &[u8] = b"mesh-llm/1";
pub const NODE_PROTOCOL_GENERATION: u32 = 1;
pub const MAX_CONTROL_FRAME_BYTES: usize = 8 * 1024 * 1024;

pub const STREAM_GOSSIP: u8 = 0x01;
pub const STREAM_TUNNEL: u8 = 0x02;
pub const STREAM_TUNNEL_MAP: u8 = 0x03;
pub const STREAM_TUNNEL_HTTP: u8 = 0x04;
pub const STREAM_ROUTE_REQUEST: u8 = 0x05;
pub const STREAM_PEER_DOWN: u8 = 0x06;
pub const STREAM_PEER_LEAVING: u8 = 0x07;
pub const STREAM_PLUGIN_CHANNEL: u8 = 0x08;
pub const STREAM_PLUGIN_BULK_TRANSFER: u8 = 0x09;
/// Reserved legacy mesh-plane config subscription stream ID.
///
/// Config and inventory control now live exclusively on `mesh-llm-control/1`;
/// keep 0x0b reserved so old wire values are not accidentally reused.
pub const STREAM_CONFIG_SUBSCRIBE: u8 = 0x0b;
/// Reserved legacy mesh-plane config push stream ID.
///
/// Config and inventory control now live exclusively on `mesh-llm-control/1`;
/// keep 0x0c reserved so old wire values are not accidentally reused.
pub const STREAM_CONFIG_PUSH: u8 = 0x0c;
pub const STREAM_SUBPROTOCOL: u8 = 0x0d;
pub const STREAM_DIRECT_PATH_REQUEST: u8 = 0x0e;
const _: () = {
    let _ = STREAM_CONFIG_SUBSCRIBE;
    let _ = STREAM_CONFIG_PUSH;
    let _ = STREAM_SUBPROTOCOL;
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ControlProtocol {
    ProtoV1,
    JsonV0,
}

#[derive(Debug, PartialEq)]
pub enum ControlFrameError {
    OversizeFrame { size: usize },
    BadGeneration { got: u32 },
    InvalidEndpointId { got: usize },
    InvalidSenderId { got: usize },
    MissingDirectPathAddress,
    MissingHttpPort,
    MissingOwnerId,
    MissingControlOwnerId,
    InvalidConfigHashLength { got: usize },
    InvalidSubprotocol,
    InvalidPublicKeyLength { got: usize },
    MissingSignature,
    InvalidSignatureLength { got: usize },
    MissingConfig,
    MissingControlEnvelope,
    MissingControlCommand,
    MissingControlResult,
    MissingControlOwnership,
    MissingRequestId,
    InvalidOwnerControlErrorCode { got: i32 },
    InvalidInventoryDisposition { got: i32 },
    MissingInventoryModelRef,
    MissingModelRef,
    InvalidModelRefCombination,
    InvalidInventoryOrder,
    InvalidKvCacheOperation { got: i32 },
    MissingKvCacheStatus,
    DecodeError(String),
    WrongStreamType { expected: u8, got: u8 },
    ForgedSender,
}

impl std::fmt::Display for ControlFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ControlFrameError::OversizeFrame { size } => write!(
                f,
                "control frame too large: {} bytes (max {})",
                size, MAX_CONTROL_FRAME_BYTES
            ),
            ControlFrameError::BadGeneration { got } => write!(
                f,
                "bad protocol generation: expected {}, got {}",
                NODE_PROTOCOL_GENERATION, got
            ),
            ControlFrameError::InvalidEndpointId { got } => {
                write!(f, "invalid endpoint_id length: expected 32, got {}", got)
            }
            ControlFrameError::InvalidSenderId { got } => {
                write!(f, "invalid sender_id length: expected 32, got {}", got)
            }
            ControlFrameError::MissingDirectPathAddress => {
                write!(f, "direct path request missing endpoint address")
            }
            ControlFrameError::MissingHttpPort => {
                write!(f, "HOST-role peer annotation missing http_port")
            }
            ControlFrameError::MissingOwnerId => write!(f, "config frame missing owner_id"),
            ControlFrameError::MissingControlOwnerId => {
                write!(f, "owner control handshake missing owner_id")
            }
            ControlFrameError::InvalidConfigHashLength { got } => {
                write!(f, "invalid config_hash length: expected 32, got {}", got)
            }
            ControlFrameError::InvalidSubprotocol => {
                write!(f, "subprotocol entries require a non-empty name and major")
            }
            ControlFrameError::InvalidPublicKeyLength { got } => {
                write!(f, "invalid public key length: expected 32, got {}", got)
            }
            ControlFrameError::MissingSignature => write!(f, "config push missing signature"),
            ControlFrameError::InvalidSignatureLength { got } => {
                write!(f, "invalid signature length: expected 64, got {got}")
            }
            ControlFrameError::MissingConfig => {
                write!(f, "config field is required but missing")
            }
            ControlFrameError::MissingControlEnvelope => {
                write!(f, "owner control envelope requires exactly one payload")
            }
            ControlFrameError::MissingControlCommand => {
                write!(
                    f,
                    "owner control request requires exactly one command variant"
                )
            }
            ControlFrameError::MissingControlResult => {
                write!(
                    f,
                    "owner control response requires exactly one result variant"
                )
            }
            ControlFrameError::MissingControlOwnership => {
                write!(f, "owner control handshake missing ownership attestation")
            }
            ControlFrameError::MissingRequestId => {
                write!(f, "owner control request_id must be non-zero")
            }
            ControlFrameError::InvalidOwnerControlErrorCode { got } => {
                write!(f, "invalid owner control error code: {got}")
            }
            ControlFrameError::InvalidInventoryDisposition { got } => {
                write!(f, "invalid inventory scan disposition: {got}")
            }
            ControlFrameError::MissingInventoryModelRef => {
                write!(f, "inventory entry requires a canonical model ref")
            }
            ControlFrameError::MissingModelRef => {
                write!(
                    f,
                    "model lifecycle command requires exactly one model reference"
                )
            }
            ControlFrameError::InvalidModelRefCombination => {
                write!(
                    f,
                    "model lifecycle command has an invalid model identifier combination"
                )
            }
            ControlFrameError::InvalidInventoryOrder => {
                write!(
                    f,
                    "inventory entries must be strictly sorted by canonical model ref"
                )
            }
            ControlFrameError::InvalidKvCacheOperation { got } => {
                write!(f, "invalid owner-control kv-cache operation: {got}")
            }
            ControlFrameError::MissingKvCacheStatus => {
                write!(f, "owner-control kv-cache response missing status payload")
            }
            ControlFrameError::DecodeError(msg) => write!(f, "protobuf decode error: {}", msg),
            ControlFrameError::WrongStreamType { expected, got } => write!(
                f,
                "wrong stream type: expected {:#04x}, got {:#04x}",
                expected, got
            ),
            ControlFrameError::ForgedSender => {
                write!(f, "frame peer_id does not match QUIC connection identity")
            }
        }
    }
}

impl std::error::Error for ControlFrameError {}

pub trait ValidateControlFrame: prost::Message + Default + Sized {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::GossipFrame {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if self.sender_id.len() != 32 {
            return Err(ControlFrameError::InvalidSenderId {
                got: self.sender_id.len(),
            });
        }
        for pa in &self.peers {
            validate_peer_announcement(pa)?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::TunnelMap {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.owner_peer_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.owner_peer_id.len(),
            });
        }
        for entry in &self.entries {
            if entry.target_peer_id.len() != 32 {
                return Err(ControlFrameError::InvalidEndpointId {
                    got: entry.target_peer_id.len(),
                });
            }
        }
        Ok(())
    }
}
impl ValidateControlFrame for crate::proto::node::RouteTableRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if !self.requester_id.is_empty() && self.requester_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.requester_id.len(),
            });
        }
        Ok(())
    }
}
impl ValidateControlFrame for crate::proto::node::RouteTable {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        for entry in &self.entries {
            if entry.endpoint_id.len() != 32 {
                return Err(ControlFrameError::InvalidEndpointId {
                    got: entry.endpoint_id.len(),
                });
            }
        }
        Ok(())
    }
}
impl ValidateControlFrame for crate::proto::node::PeerDown {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if self.peer_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.peer_id.len(),
            });
        }
        Ok(())
    }
}
impl ValidateControlFrame for crate::proto::node::PeerLeaving {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if self.peer_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.peer_id.len(),
            });
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::DirectPathRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if self.requester_id.len() != 32 {
            return Err(ControlFrameError::InvalidEndpointId {
                got: self.requester_id.len(),
            });
        }
        if self.serialized_addr.is_empty() {
            return Err(ControlFrameError::MissingDirectPathAddress);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlEnvelope {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        let payloads = [
            self.handshake.is_some(),
            self.request.is_some(),
            self.response.is_some(),
            self.error.is_some(),
        ];
        if payloads.into_iter().filter(|present| *present).count() != 1 {
            return Err(ControlFrameError::MissingControlEnvelope);
        }
        if let Some(handshake) = &self.handshake {
            handshake.validate_frame()?;
        }
        if let Some(request) = &self.request {
            request.validate_frame()?;
        }
        if let Some(response) = &self.response {
            response.validate_frame()?;
        }
        if let Some(error) = &self.error {
            error.validate_frame()?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlHandshake {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        let ownership = self
            .ownership
            .as_ref()
            .ok_or(ControlFrameError::MissingControlOwnership)?;
        if ownership.owner_id.trim().is_empty() {
            return Err(ControlFrameError::MissingControlOwnerId);
        }
        validate_public_key_length(ownership.owner_sign_public_key.len())?;
        validate_endpoint_id_length(ownership.node_endpoint_id.len())?;
        if ownership.signature.is_empty() {
            return Err(ControlFrameError::MissingSignature);
        }
        if ownership.signature.len() != 64 {
            return Err(ControlFrameError::InvalidSignatureLength {
                got: ownership.signature.len(),
            });
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.request_id == 0 {
            return Err(ControlFrameError::MissingRequestId);
        }
        let commands = [
            self.get_config.is_some(),
            self.watch_config.is_some(),
            self.apply_config.is_some(),
            self.refresh_inventory.is_some(),
            self.load_model.is_some(),
            self.unload_model.is_some(),
            self.ensure_model.is_some(),
            self.drain_model.is_some(),
            self.kv_cache.is_some(),
        ];
        if commands.into_iter().filter(|present| *present).count() != 1 {
            return Err(ControlFrameError::MissingControlCommand);
        }
        if let Some(request) = &self.get_config {
            request.validate_frame()?;
        }
        if let Some(request) = &self.watch_config {
            request.validate_frame()?;
        }
        if let Some(request) = &self.apply_config {
            request.validate_frame()?;
        }
        if let Some(request) = &self.refresh_inventory {
            request.validate_frame()?;
        }
        if let Some(request) = &self.load_model {
            request.validate_frame()?;
        }
        if let Some(request) = &self.unload_model {
            request.validate_frame()?;
        }
        if let Some(request) = &self.ensure_model {
            request.validate_frame()?;
        }
        if let Some(request) = &self.drain_model {
            request.validate_frame()?;
        }
        if let Some(request) = &self.kv_cache {
            request.validate_frame()?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.request_id == 0 {
            return Err(ControlFrameError::MissingRequestId);
        }
        let results = [
            self.get_config.is_some(),
            self.watch_config.is_some(),
            self.apply_config.is_some(),
            self.refresh_inventory.is_some(),
            self.load_model.is_some(),
            self.unload_model.is_some(),
            self.ensure_model.is_some(),
            self.drain_model.is_some(),
            self.kv_cache.is_some(),
        ];
        if results.into_iter().filter(|present| *present).count() != 1 {
            return Err(ControlFrameError::MissingControlResult);
        }
        if let Some(response) = &self.get_config {
            response.validate_frame()?;
        }
        if let Some(response) = &self.watch_config {
            response.validate_frame()?;
        }
        if let Some(response) = &self.apply_config {
            response.validate_frame()?;
        }
        if let Some(response) = &self.refresh_inventory {
            response.validate_frame()?;
        }
        if let Some(response) = &self.load_model {
            response.validate_frame()?;
        }
        if let Some(response) = &self.unload_model {
            response.validate_frame()?;
        }
        if let Some(response) = &self.ensure_model {
            response.validate_frame()?;
        }
        if let Some(response) = &self.drain_model {
            response.validate_frame()?;
        }
        if let Some(response) = &self.kv_cache {
            response.validate_frame()?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlError {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if matches!(
            crate::proto::node::OwnerControlErrorCode::try_from(self.code),
            Err(_) | Ok(crate::proto::node::OwnerControlErrorCode::Unspecified)
        ) {
            return Err(ControlFrameError::InvalidOwnerControlErrorCode { got: self.code });
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlGetConfigRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlGetConfigResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        self.snapshot
            .as_ref()
            .ok_or(ControlFrameError::MissingConfig)?
            .validate_frame()
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlWatchConfigRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlWatchConfigResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        let results = [
            self.accepted.is_some(),
            self.snapshot.is_some(),
            self.update.is_some(),
        ];
        if results.into_iter().filter(|present| *present).count() != 1 {
            return Err(ControlFrameError::MissingControlResult);
        }
        if let Some(accepted) = &self.accepted {
            accepted.validate_frame()?;
        }
        if let Some(snapshot) = &self.snapshot {
            snapshot.validate_frame()?;
        }
        if let Some(update) = &self.update {
            update.validate_frame()?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlWatchAccepted {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.target_node_id.len())?;
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlApplyConfigRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        if self.config.is_none() {
            return Err(ControlFrameError::MissingConfig);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlApplyConfigResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.success || !self.config_hash.is_empty() {
            validate_config_hash_length(self.config_hash.len())?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlRefreshInventoryRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlRefreshInventoryResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        self.snapshot
            .as_ref()
            .ok_or(ControlFrameError::MissingConfig)?
            .validate_frame()?;
        if let Some(inventory) = &self.inventory {
            inventory.validate_frame()?;
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlLoadModelRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        let model = self
            .model
            .as_ref()
            .ok_or(ControlFrameError::MissingModelRef)?;
        validate_owner_control_model_for_load_or_ensure(model)
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlUnloadModelRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        let model = self
            .model
            .as_ref()
            .ok_or(ControlFrameError::MissingModelRef)?;
        validate_owner_control_model_for_unload_or_drain(model)
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlEnsureModelRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        let model = self
            .model
            .as_ref()
            .ok_or(ControlFrameError::MissingModelRef)?;
        validate_owner_control_model_for_load_or_ensure(model)
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlDrainModelRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        let model = self
            .model
            .as_ref()
            .ok_or(ControlFrameError::MissingModelRef)?;
        validate_owner_control_model_for_unload_or_drain(model)
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlKvCacheRequest {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.requester_node_id.len())?;
        validate_endpoint_id_length(self.target_node_id.len())?;
        match crate::proto::node::OwnerControlKvCacheOperation::try_from(self.operation) {
            Ok(crate::proto::node::OwnerControlKvCacheOperation::Status)
            | Ok(crate::proto::node::OwnerControlKvCacheOperation::Prune)
            | Ok(crate::proto::node::OwnerControlKvCacheOperation::Clear) => Ok(()),
            Ok(crate::proto::node::OwnerControlKvCacheOperation::Unspecified) | Err(_) => {
                Err(ControlFrameError::InvalidKvCacheOperation {
                    got: self.operation,
                })
            }
        }
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlLoadModelResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_owner_control_model_for_load_or_ensure(
            self.target
                .as_ref()
                .ok_or(ControlFrameError::MissingModelRef)?,
        )
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlUnloadModelResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_owner_control_model_for_unload_or_drain(
            self.target
                .as_ref()
                .ok_or(ControlFrameError::MissingModelRef)?,
        )
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlEnsureModelResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_owner_control_model_for_load_or_ensure(
            self.target
                .as_ref()
                .ok_or(ControlFrameError::MissingModelRef)?,
        )
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlDrainModelResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_owner_control_model_for_unload_or_drain(
            self.target
                .as_ref()
                .ok_or(ControlFrameError::MissingModelRef)?,
        )
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlKvCacheResponse {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.status_json.is_empty() {
            return Err(ControlFrameError::MissingKvCacheStatus);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlRefreshInventory {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        use crate::proto::node::OwnerControlRefreshInventoryDisposition;

        if !matches!(
            OwnerControlRefreshInventoryDisposition::try_from(self.disposition),
            Ok(OwnerControlRefreshInventoryDisposition::Executed)
                | Ok(OwnerControlRefreshInventoryDisposition::Coalesced)
        ) {
            return Err(ControlFrameError::InvalidInventoryDisposition {
                got: self.disposition,
            });
        }
        let mut previous = None;
        for entry in &self.entries {
            let canonical = entry.canonical_model_ref.trim();
            if canonical.is_empty() {
                return Err(ControlFrameError::MissingInventoryModelRef);
            }
            if previous.is_some_and(|value| value >= canonical) {
                return Err(ControlFrameError::InvalidInventoryOrder);
            }
            previous = Some(canonical);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlConfigSnapshot {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.node_id.len())?;
        validate_config_hash_length(self.config_hash.len())?;
        if self.config.is_none() {
            return Err(ControlFrameError::MissingConfig);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::OwnerControlConfigUpdate {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        validate_endpoint_id_length(self.node_id.len())?;
        validate_config_hash_length(self.config_hash.len())?;
        if self.config.is_none() {
            return Err(ControlFrameError::MissingConfig);
        }
        Ok(())
    }
}

impl ValidateControlFrame for crate::proto::node::MeshSubprotocolOpen {
    fn validate_frame(&self) -> Result<(), ControlFrameError> {
        if self.r#gen != NODE_PROTOCOL_GENERATION {
            return Err(ControlFrameError::BadGeneration { got: self.r#gen });
        }
        if self.name.trim().is_empty() || self.major == 0 {
            return Err(ControlFrameError::InvalidSubprotocol);
        }
        Ok(())
    }
}

pub fn validate_peer_announcement(
    pa: &crate::proto::node::PeerAnnouncement,
) -> Result<(), ControlFrameError> {
    if pa.endpoint_id.len() != 32 {
        return Err(ControlFrameError::InvalidEndpointId {
            got: pa.endpoint_id.len(),
        });
    }
    if pa.role == crate::proto::node::NodeRole::Host as i32 && pa.http_port.is_none() {
        return Err(ControlFrameError::MissingHttpPort);
    }
    for subprotocol in &pa.subprotocols {
        if subprotocol.name.trim().is_empty() || subprotocol.major == 0 {
            return Err(ControlFrameError::InvalidSubprotocol);
        }
    }
    Ok(())
}

fn validate_endpoint_id_length(len: usize) -> Result<(), ControlFrameError> {
    if len != 32 {
        return Err(ControlFrameError::InvalidEndpointId { got: len });
    }
    Ok(())
}

fn validate_config_hash_length(len: usize) -> Result<(), ControlFrameError> {
    if len != 32 {
        return Err(ControlFrameError::InvalidConfigHashLength { got: len });
    }
    Ok(())
}

pub fn validate_owner_control_model_for_load_or_ensure(
    model: &crate::proto::node::OwnerControlModelRef,
) -> Result<(), ControlFrameError> {
    let canonical = !model.canonical_model_ref.trim().is_empty();
    let instance = model
        .instance_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    match (canonical, instance) {
        (true, false) => Ok(()),
        (false, false) => Err(ControlFrameError::MissingModelRef),
        _ => Err(ControlFrameError::InvalidModelRefCombination),
    }
}

pub fn validate_owner_control_model_for_unload_or_drain(
    model: &crate::proto::node::OwnerControlModelRef,
) -> Result<(), ControlFrameError> {
    let canonical = !model.canonical_model_ref.trim().is_empty();
    let instance = model
        .instance_id
        .as_deref()
        .is_some_and(|id| !id.trim().is_empty());
    if canonical ^ instance {
        Ok(())
    } else if canonical || instance {
        Err(ControlFrameError::InvalidModelRefCombination)
    } else {
        Err(ControlFrameError::MissingModelRef)
    }
}

fn validate_public_key_length(len: usize) -> Result<(), ControlFrameError> {
    if len != 32 {
        return Err(ControlFrameError::InvalidPublicKeyLength { got: len });
    }
    Ok(())
}

pub fn protocol_from_alpn(alpn: &[u8]) -> ControlProtocol {
    if alpn == ALPN_V0 {
        ControlProtocol::JsonV0
    } else {
        ControlProtocol::ProtoV1
    }
}

pub fn connection_protocol(conn: &Connection) -> ControlProtocol {
    protocol_from_alpn(conn.alpn())
}

pub async fn connect_mesh(endpoint: &Endpoint, addr: EndpointAddr) -> Result<Connection> {
    let opts = ConnectOptions::new().with_additional_alpns(vec![ALPN_V0.to_vec()]);
    let connecting = endpoint.connect_with_opts(addr, ALPN_V1, opts).await?;
    Ok(connecting.await?)
}

pub async fn write_len_prefixed(send: &mut iroh::endpoint::SendStream, body: &[u8]) -> Result<()> {
    ensure_control_frame_size(body)?;
    send.write_all(&(body.len() as u32).to_le_bytes()).await?;
    send.write_all(body).await?;
    Ok(())
}

pub fn ensure_control_frame_size(body: &[u8]) -> Result<(), ControlFrameError> {
    if body.len() > MAX_CONTROL_FRAME_BYTES {
        return Err(ControlFrameError::OversizeFrame { size: body.len() });
    }
    Ok(())
}

pub async fn read_len_prefixed(recv: &mut iroh::endpoint::RecvStream) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    recv.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_CONTROL_FRAME_BYTES {
        anyhow::bail!("control frame too large: {} bytes", len);
    }
    let mut buf = vec![0u8; len];
    recv.read_exact(&mut buf).await?;
    Ok(buf)
}

pub fn encode_control_frame(stream_type: u8, msg: &impl prost::Message) -> Vec<u8> {
    let proto_bytes = msg.encode_to_vec();
    let len = proto_bytes.len() as u32;
    let mut buf = Vec::with_capacity(1 + 4 + proto_bytes.len());
    buf.push(stream_type);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&proto_bytes);
    buf
}

pub fn decode_control_frame<T: ValidateControlFrame>(
    expected_stream_type: u8,
    data: &[u8],
) -> Result<T, ControlFrameError> {
    const HEADER_LEN: usize = 5;
    if data.len() < HEADER_LEN {
        return Err(ControlFrameError::DecodeError(format!(
            "frame too short: {} bytes (minimum {})",
            data.len(),
            HEADER_LEN
        )));
    }
    let actual_type = data[0];
    if actual_type != expected_stream_type {
        return Err(ControlFrameError::WrongStreamType {
            expected: expected_stream_type,
            got: actual_type,
        });
    }
    let len = u32::from_le_bytes(data[1..5].try_into().unwrap()) as usize;
    if len > MAX_CONTROL_FRAME_BYTES {
        return Err(ControlFrameError::OversizeFrame { size: len });
    }
    let proto_bytes = data.get(5..5 + len).ok_or_else(|| {
        ControlFrameError::DecodeError(format!(
            "frame truncated: header says {} bytes but only {} available",
            len,
            data.len().saturating_sub(5)
        ))
    })?;
    let msg = T::decode(proto_bytes).map_err(|e| ControlFrameError::DecodeError(e.to_string()))?;
    msg.validate_frame()?;
    Ok(msg)
}

pub fn encode_owner_control_envelope(msg: &crate::proto::node::OwnerControlEnvelope) -> Vec<u8> {
    msg.encode_to_vec()
}

pub fn decode_owner_control_envelope(
    data: &[u8],
) -> Result<crate::proto::node::OwnerControlEnvelope, ControlFrameError> {
    let msg = crate::proto::node::OwnerControlEnvelope::decode(data)
        .map_err(|e| ControlFrameError::DecodeError(e.to_string()))?;
    msg.validate_frame()?;
    Ok(msg)
}

pub fn owner_control_error_envelope(
    code: crate::proto::node::OwnerControlErrorCode,
    request_id: Option<u64>,
    message: impl Into<String>,
) -> crate::proto::node::OwnerControlEnvelope {
    crate::proto::node::OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: None,
        response: None,
        error: Some(crate::proto::node::OwnerControlError {
            code: code as i32,
            message: message.into(),
            request_id,
            current_revision: None,
        }),
    }
}

pub fn owner_control_rejection_envelope(
    data: &[u8],
    request_id: Option<u64>,
    err: &ControlFrameError,
) -> crate::proto::node::OwnerControlEnvelope {
    let code = if matches!(err, ControlFrameError::MissingControlCommand) {
        crate::proto::node::OwnerControlErrorCode::UnknownCommand
    } else if serde_json::from_slice::<serde_json::Value>(data).is_ok() {
        crate::proto::node::OwnerControlErrorCode::LegacyJsonUnsupported
    } else {
        crate::proto::node::OwnerControlErrorCode::BadRequest
    };
    owner_control_error_envelope(code, request_id, err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::node::{
        CompactModelMetadata, ConfigApplyMode, GossipFrame, InferenceAdmissionState,
        NodeConfigSnapshot, NodeGpuConfig, NodeModelEntry, NodeRole,
        OwnerControlApplyConfigRequest, OwnerControlApplyConfigResponse,
        OwnerControlConfigSnapshot, OwnerControlConfigUpdate, OwnerControlEnvelope,
        OwnerControlError, OwnerControlErrorCode, OwnerControlGetConfigRequest,
        OwnerControlGetConfigResponse, OwnerControlHandshake, OwnerControlInventoryEntry,
        OwnerControlKvCacheOperation, OwnerControlKvCacheRequest, OwnerControlKvCacheResponse,
        OwnerControlRefreshInventory, OwnerControlRefreshInventoryDisposition,
        OwnerControlRefreshInventoryRequest, OwnerControlRefreshInventoryResponse,
        OwnerControlRequest, OwnerControlResponse, OwnerControlWatchAccepted,
        OwnerControlWatchConfigResponse, PeerAnnouncement, SignedNodeOwnership,
    };
    use prost::Message;

    fn control_plane_test_config() -> NodeConfigSnapshot {
        NodeConfigSnapshot {
            version: 1,
            gpu: Some(NodeGpuConfig {
                assignment: crate::proto::node::GpuAssignment::Auto as i32,
            }),
            models: vec![NodeModelEntry {
                model: "Qwen3-8B".to_string(),
                mmproj: None,
                ctx_size: Some(8192),
                gpu_id: None,
                model_ref: None,
                mmproj_ref: None,
            }],
            plugins: vec![],
            config_toml: None,
            mesh_requirements: None,
        }
    }

    fn control_plane_test_snapshot() -> OwnerControlConfigSnapshot {
        OwnerControlConfigSnapshot {
            node_id: vec![0x55; 32],
            revision: 7,
            config_hash: vec![0xA5; 32],
            config: Some(control_plane_test_config()),
            hostname: Some("node-01".to_string()),
        }
    }

    fn control_plane_test_handshake() -> OwnerControlEnvelope {
        OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: Some(OwnerControlHandshake {
                ownership: Some(SignedNodeOwnership {
                    version: 1,
                    cert_id: "cert-1".to_string(),
                    owner_id: "owner-1".to_string(),
                    owner_sign_public_key: vec![0x11; 32],
                    node_endpoint_id: vec![0x22; 32],
                    issued_at_unix_ms: 1,
                    expires_at_unix_ms: 2,
                    node_label: Some("node-01".to_string()),
                    hostname_hint: Some("node-01".to_string()),
                    signature: vec![0x33; 64],
                }),
            }),
            request: None,
            response: None,
            error: None,
        }
    }

    #[test]
    fn control_plane_messages_constants_are_stable() {
        assert_eq!(ALPN_CONTROL_V1, b"mesh-llm-control/1");
        assert_eq!(ALPN_V1, b"mesh-llm/1");
        assert_eq!(ALPN_V0, b"mesh-llm/0");
        assert_eq!(STREAM_CONFIG_SUBSCRIBE, 0x0b);
        assert_eq!(STREAM_CONFIG_PUSH, 0x0c);
        assert_eq!(STREAM_SUBPROTOCOL, 0x0d);
        assert_eq!(STREAM_DIRECT_PATH_REQUEST, 0x0e);
    }

    #[test]
    fn control_plane_messages_roundtrip_commands_and_responses() {
        let handshake = control_plane_test_handshake();
        let decoded = decode_owner_control_envelope(&encode_owner_control_envelope(&handshake))
            .expect("handshake must decode");
        assert!(decoded.handshake.is_some());

        let get_request = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                request_id: 10,
                get_config: Some(OwnerControlGetConfigRequest {
                    requester_node_id: vec![0x10; 32],
                    target_node_id: vec![0x20; 32],
                }),
                watch_config: None,
                apply_config: None,
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            response: None,
            error: None,
        };
        let decoded = decode_owner_control_envelope(&encode_owner_control_envelope(&get_request))
            .expect("get-config request must decode");
        assert_eq!(decoded.request.unwrap().request_id, 10);

        let watch_response = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(OwnerControlResponse {
                request_id: 11,
                get_config: None,
                watch_config: Some(OwnerControlWatchConfigResponse {
                    accepted: Some(OwnerControlWatchAccepted {
                        target_node_id: vec![0x21; 32],
                    }),
                    snapshot: None,
                    update: None,
                }),
                apply_config: None,
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            error: None,
        };
        decode_owner_control_envelope(&encode_owner_control_envelope(&watch_response))
            .expect("watch-config response must decode");

        let apply_request = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                request_id: 12,
                get_config: None,
                watch_config: None,
                apply_config: Some(OwnerControlApplyConfigRequest {
                    requester_node_id: vec![0x30; 32],
                    target_node_id: vec![0x40; 32],
                    expected_revision: 7,
                    config: Some(control_plane_test_config()),
                }),
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            response: None,
            error: None,
        };
        decode_owner_control_envelope(&encode_owner_control_envelope(&apply_request))
            .expect("apply-config request must decode");

        let apply_response = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(OwnerControlResponse {
                request_id: 12,
                get_config: None,
                watch_config: None,
                apply_config: Some(OwnerControlApplyConfigResponse {
                    success: true,
                    current_revision: 8,
                    config_hash: vec![0x99; 32],
                    error: None,
                    apply_mode: ConfigApplyMode::Live as i32,
                    diagnostics: Vec::new(),
                }),
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            error: None,
        };
        decode_owner_control_envelope(&encode_owner_control_envelope(&apply_response))
            .expect("apply-config response must decode");

        let refresh_request = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                request_id: 13,
                get_config: None,
                watch_config: None,
                apply_config: None,
                refresh_inventory: Some(OwnerControlRefreshInventoryRequest {
                    requester_node_id: vec![0x50; 32],
                    target_node_id: vec![0x60; 32],
                }),
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            response: None,
            error: None,
        };
        decode_owner_control_envelope(&encode_owner_control_envelope(&refresh_request))
            .expect("refresh-inventory request must decode");

        let refresh_response = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(OwnerControlResponse {
                request_id: 13,
                get_config: None,
                watch_config: None,
                apply_config: None,
                refresh_inventory: Some(OwnerControlRefreshInventoryResponse {
                    snapshot: Some(control_plane_test_snapshot()),
                    inventory: None,
                }),
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            error: None,
        };
        decode_owner_control_envelope(&encode_owner_control_envelope(&refresh_response))
            .expect("refresh-inventory response must decode");

        let get_response = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(OwnerControlResponse {
                request_id: 14,
                get_config: Some(OwnerControlGetConfigResponse {
                    snapshot: Some(control_plane_test_snapshot()),
                }),
                watch_config: None,
                apply_config: None,
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            error: None,
        };
        decode_owner_control_envelope(&encode_owner_control_envelope(&get_response))
            .expect("get-config response must decode");

        let update_response = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(OwnerControlResponse {
                request_id: 15,
                get_config: None,
                watch_config: Some(OwnerControlWatchConfigResponse {
                    accepted: None,
                    snapshot: None,
                    update: Some(OwnerControlConfigUpdate {
                        node_id: vec![0x55; 32],
                        revision: 8,
                        config_hash: vec![0x77; 32],
                        config: Some(control_plane_test_config()),
                    }),
                }),
                apply_config: None,
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            error: None,
        };
        decode_owner_control_envelope(&encode_owner_control_envelope(&update_response))
            .expect("watch update response must decode");

        let cache_request = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                request_id: 16,
                kv_cache: Some(OwnerControlKvCacheRequest {
                    requester_node_id: vec![0x61; 32],
                    target_node_id: vec![0x62; 32],
                    operation: OwnerControlKvCacheOperation::Prune as i32,
                    target_bytes: Some(1024),
                    model_identity: Some("blake3:model".to_string()),
                }),
                ..Default::default()
            }),
            response: None,
            error: None,
        };
        let decoded = decode_owner_control_envelope(&encode_owner_control_envelope(&cache_request))
            .expect("kv-cache request must decode");
        assert_eq!(decoded.request.unwrap().request_id, 16);

        let cache_response = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(OwnerControlResponse {
                request_id: 16,
                kv_cache: Some(OwnerControlKvCacheResponse {
                    status_json: br#"{"version":1}"#.to_vec(),
                    freed_bytes: Some(512),
                }),
                ..Default::default()
            }),
            error: None,
        };
        decode_owner_control_envelope(&encode_owner_control_envelope(&cache_response))
            .expect("kv-cache response must decode");
    }

    #[test]
    fn control_plane_messages_unknown_command_rejects_with_structured_error() {
        let envelope = OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: Some(OwnerControlRequest {
                request_id: 42,
                get_config: None,
                watch_config: None,
                apply_config: None,
                refresh_inventory: None,
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            response: None,
            error: None,
        };
        let bytes = encode_owner_control_envelope(&envelope);
        let err = decode_owner_control_envelope(&bytes)
            .expect_err("missing command variant must be rejected");
        assert!(matches!(err, ControlFrameError::MissingControlCommand));

        let rejection = owner_control_rejection_envelope(&bytes, Some(42), &err);
        let error = rejection
            .error
            .expect("structured rejection must carry an error");
        assert_eq!(
            crate::proto::node::OwnerControlErrorCode::try_from(error.code).unwrap(),
            OwnerControlErrorCode::UnknownCommand
        );
        assert_eq!(error.request_id, Some(42));
    }

    #[test]
    fn owner_control_handshake_empty_owner_id_uses_handshake_error() {
        let mut envelope = control_plane_test_handshake();
        envelope
            .handshake
            .as_mut()
            .and_then(|handshake| handshake.ownership.as_mut())
            .expect("test handshake must include ownership")
            .owner_id = "   ".to_string();

        let err = decode_owner_control_envelope(&encode_owner_control_envelope(&envelope))
            .expect_err("handshake with blank owner_id must be rejected");
        assert!(matches!(err, ControlFrameError::MissingControlOwnerId));
        assert_eq!(err.to_string(), "owner control handshake missing owner_id");
    }

    #[test]
    fn owner_control_error_rejects_invalid_error_code() {
        for code in [OwnerControlErrorCode::Unspecified as i32, 9999] {
            let err = OwnerControlError {
                code,
                message: "invalid".to_string(),
                request_id: Some(1),
                current_revision: None,
            }
            .validate_frame()
            .expect_err("invalid owner-control error code must be rejected");
            assert!(matches!(
                err,
                ControlFrameError::InvalidOwnerControlErrorCode { got } if got == code
            ));
            assert_eq!(
                err.to_string(),
                format!("invalid owner control error code: {code}")
            );
        }
    }

    #[test]
    fn control_plane_messages_legacy_json_rejects_with_structured_error() {
        let legacy_json = br#"{"owner_id":"legacy","command":"GetConfig"}"#;
        let err = decode_owner_control_envelope(legacy_json)
            .expect_err("legacy json must not decode on protobuf-only control plane");
        let rejection = owner_control_rejection_envelope(legacy_json, Some(99), &err);
        let error = rejection
            .error
            .expect("structured rejection must carry an error");
        assert_eq!(
            crate::proto::node::OwnerControlErrorCode::try_from(error.code).unwrap(),
            OwnerControlErrorCode::LegacyJsonUnsupported
        );
        assert_eq!(error.request_id, Some(99));
    }

    #[test]
    fn outbound_control_frame_size_rejects_before_write() {
        let oversized = vec![0u8; MAX_CONTROL_FRAME_BYTES + 1];

        let err = ensure_control_frame_size(&oversized)
            .expect_err("oversize outbound frame must fail before length/body write");

        assert!(matches!(
            err,
            ControlFrameError::OversizeFrame { size } if size == MAX_CONTROL_FRAME_BYTES + 1
        ));
    }

    #[test]
    fn refresh_inventory_snapshot_only_frozen_bytes_remain_compatible() {
        let mut frozen = vec![0x0a, 0x48, 0x0a, 0x20];
        frozen.extend_from_slice(&[0x55; 32]);
        frozen.extend_from_slice(&[0x10, 0x07, 0x1a, 0x20]);
        frozen.extend_from_slice(&[0xa5; 32]);
        frozen.extend_from_slice(&[0x22, 0x00]);
        frozen.extend_from_slice(&[0x98, 0x06, 0x01]);

        let decoded = OwnerControlRefreshInventoryResponse::decode(frozen.as_slice())
            .expect("frozen snapshot-only response must decode");

        decoded
            .validate_frame()
            .expect("snapshot-only response from an old server must remain valid");
        assert!(decoded.inventory.is_none());
        assert_eq!(decoded.snapshot.expect("snapshot").revision, 7);
    }

    #[test]
    fn refresh_inventory_rich_response_roundtrips_in_sorted_order() {
        let response = OwnerControlRefreshInventoryResponse {
            snapshot: Some(control_plane_test_snapshot()),
            inventory: Some(OwnerControlRefreshInventory {
                entries: vec![
                    inventory_entry("hf://mesh/alpha-GGUF:Q4_K_M", "alpha", 1024),
                    inventory_entry("hf://mesh/beta-GGUF:Q8_0", "beta", 2048),
                ],
                disposition: OwnerControlRefreshInventoryDisposition::Executed as i32,
            }),
        };

        response
            .validate_frame()
            .expect("rich response must validate");
        let encoded = response.encode_to_vec();
        let decoded = OwnerControlRefreshInventoryResponse::decode(encoded.as_slice())
            .expect("rich response must decode");

        assert_eq!(decoded, response);
        assert_eq!(decoded.inventory.expect("inventory").entries.len(), 2);
    }

    #[test]
    fn peer_announcement_admission_round_trip() {
        let mut peer = PeerAnnouncement {
            endpoint_id: vec![0x42; 32],
            role: NodeRole::Worker as i32,
            inference_admission_state: Some(InferenceAdmissionState::Accepting as i32),
            ..Default::default()
        };

        let frame_with_state = GossipFrame {
            r#gen: NODE_PROTOCOL_GENERATION,
            sender_id: vec![0x11; 32],
            peers: vec![peer.clone()],
        };

        let encoded = frame_with_state.encode_to_vec();
        let decoded = GossipFrame::decode(encoded.as_slice())
            .expect("peer-announcement frame should decode after encode");
        let decoded_peer = &decoded.peers[0];

        assert_eq!(
            decoded_peer.inference_admission_state,
            peer.inference_admission_state
        );
        assert!(decoded.validate_frame().is_ok());

        peer.inference_admission_state = None;
        let frame_without_state = GossipFrame {
            peers: vec![peer],
            ..frame_with_state
        };
        let reencoded = frame_without_state.encode_to_vec();
        let redecoded = GossipFrame::decode(reencoded.as_slice())
            .expect("peer-announcement frame should decode without admission state");

        assert!(redecoded.peers[0].inference_admission_state.is_none());
    }

    #[test]
    fn refresh_inventory_rejects_unsorted_or_unspecified_details() {
        let mut inventory = OwnerControlRefreshInventory {
            entries: vec![
                inventory_entry("hf://mesh/beta", "beta", 2),
                inventory_entry("hf://mesh/alpha", "alpha", 1),
            ],
            disposition: OwnerControlRefreshInventoryDisposition::Executed as i32,
        };
        assert!(matches!(
            inventory.validate_frame(),
            Err(ControlFrameError::InvalidInventoryOrder)
        ));

        inventory
            .entries
            .sort_by(|left, right| left.canonical_model_ref.cmp(&right.canonical_model_ref));
        inventory.disposition = OwnerControlRefreshInventoryDisposition::Unspecified as i32;
        assert!(matches!(
            inventory.validate_frame(),
            Err(ControlFrameError::InvalidInventoryDisposition { got: 0 })
        ));
    }

    #[test]
    fn metadata_rich_inventory_has_deterministic_bounded_size() {
        let entries = (0..128)
            .map(|index| {
                let mut entry = inventory_entry(
                    &format!("hf://mesh/model-{index:03}"),
                    &format!("model-{index:03}"),
                    index,
                );
                entry.metadata.as_mut().expect("metadata").model_key = "x".repeat(70_000);
                entry
            })
            .collect();
        let response = OwnerControlRefreshInventoryResponse {
            snapshot: Some(control_plane_test_snapshot()),
            inventory: Some(OwnerControlRefreshInventory {
                entries,
                disposition: OwnerControlRefreshInventoryDisposition::Coalesced as i32,
            }),
        };

        response
            .validate_frame()
            .expect("synthetic response is valid");
        let encoded = response.encode_to_vec();
        assert_eq!(encoded.len(), response.encoded_len());
        assert_eq!(encoded.len(), response.clone().encoded_len());
        assert!(encoded.len() > MAX_CONTROL_FRAME_BYTES);
        assert!(matches!(
            ensure_control_frame_size(&encoded),
            Err(ControlFrameError::OversizeFrame { size }) if size == encoded.len()
        ));
    }

    fn inventory_entry(
        canonical_model_ref: &str,
        display_name: &str,
        total_size_bytes: u64,
    ) -> OwnerControlInventoryEntry {
        OwnerControlInventoryEntry {
            canonical_model_ref: canonical_model_ref.to_string(),
            display_name: Some(display_name.to_string()),
            total_size_bytes,
            metadata: Some(CompactModelMetadata {
                model_key: canonical_model_ref.to_string(),
                architecture: "llama".to_string(),
                quantization_type: "Q4_K_M".to_string(),
                ..Default::default()
            }),
        }
    }
}
