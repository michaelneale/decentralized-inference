// Protocol infrastructure — extracted from mesh.rs

#[cfg(test)]
use crate::mesh::AttestedLogHead;
#[cfg(test)]
use crate::mesh::NodeRole;
use crate::mesh::PeerAnnouncement;

pub(crate) mod config_diagnostic;
pub(crate) mod convert;
use anyhow::Result;
pub(crate) use convert::*;
use iroh::endpoint::Connection;
use iroh::{Endpoint, EndpointAddr, EndpointId};
pub(crate) use mesh_llm_protocol::protocol::{ControlFrameError, ValidateControlFrame};
use prost::Message;

#[cfg(test)]
mod config_tests;
pub const ALPN_CONTROL_V1: &[u8] = b"mesh-llm-control/1";
pub const ALPN_V1: &[u8] = b"mesh-llm/1";
#[cfg(test)]
pub const ALPN: &[u8] = ALPN_V1;
pub(crate) const NODE_PROTOCOL_GENERATION: u32 = 1;
pub(crate) const MAX_CONTROL_FRAME_BYTES: usize = 8 * 1024 * 1024; // 8 MiB

pub(crate) const STREAM_GOSSIP: u8 = 0x01;
pub(crate) const STREAM_TUNNEL: u8 = 0x02;
pub(crate) const STREAM_TUNNEL_MAP: u8 = 0x03;
pub const STREAM_TUNNEL_HTTP: u8 = 0x04;
pub(crate) const STREAM_ROUTE_REQUEST: u8 = 0x05;
pub(crate) const STREAM_PEER_DOWN: u8 = 0x06;
pub(crate) const STREAM_PEER_LEAVING: u8 = 0x07;
pub(crate) const STREAM_PLUGIN_CHANNEL: u8 = 0x08;
pub(crate) const STREAM_PLUGIN_BULK_TRANSFER: u8 = 0x09;
pub(crate) const STREAM_PLUGIN_MESH_STREAM: u8 = 0x0a;
// Stream IDs 0x0b and 0x0c remain reserved for the retired mesh-plane config
// subscription and push operations. Config and inventory control now live
// exclusively on `mesh-llm-control/1`.
#[cfg(test)]
pub(crate) const STREAM_CONFIG_SUBSCRIBE: u8 = 0x0b;
#[cfg(test)]
pub(crate) const STREAM_CONFIG_PUSH: u8 = 0x0c;
pub(crate) const STREAM_SUBPROTOCOL: u8 = 0x0d;
pub(crate) const STREAM_DIRECT_PATH_REQUEST: u8 = 0x0e;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ControlProtocol {
    ProtoV1,
}

pub(crate) fn protocol_from_alpn(alpn: &[u8]) -> ControlProtocol {
    let _ = alpn;
    ControlProtocol::ProtoV1
}

pub(crate) fn connection_protocol(conn: &Connection) -> ControlProtocol {
    protocol_from_alpn(conn.alpn())
}

pub(crate) async fn connect_mesh(endpoint: &Endpoint, addr: EndpointAddr) -> Result<Connection> {
    let connecting = endpoint.connect(addr, ALPN_V1).await?;
    Ok(connecting)
}

pub(crate) async fn write_len_prefixed(
    send: &mut iroh::endpoint::SendStream,
    body: &[u8],
) -> Result<()> {
    ensure_control_frame_size(body)?;
    send.write_all(&(body.len() as u32).to_le_bytes()).await?;
    send.write_all(body).await?;
    Ok(())
}

pub(crate) fn ensure_control_frame_size(body: &[u8]) -> Result<(), ControlFrameError> {
    if body.len() > MAX_CONTROL_FRAME_BYTES {
        return Err(ControlFrameError::OversizeFrame { size: body.len() });
    }
    Ok(())
}

pub(crate) async fn read_len_prefixed(recv: &mut iroh::endpoint::RecvStream) -> Result<Vec<u8>> {
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

pub(crate) async fn write_gossip_payload(
    send: &mut iroh::endpoint::SendStream,
    protocol: ControlProtocol,
    anns: &[PeerAnnouncement],
    sender_id: EndpointId,
) -> Result<()> {
    let _ = protocol;
    let frame = build_gossip_frame(anns, sender_id);
    write_len_prefixed(send, &frame.encode_to_vec()).await?;
    Ok(())
}

pub(crate) fn decode_gossip_payload(
    protocol: ControlProtocol,
    remote: EndpointId,
    buf: &[u8],
) -> Result<Vec<(EndpointAddr, PeerAnnouncement)>> {
    let _ = protocol;
    let frame = crate::proto::node::GossipFrame::decode(buf)
        .map_err(|e| anyhow::anyhow!("gossip decode from {}: {e}", remote.fmt_short()))?;
    frame
        .validate_frame()
        .map_err(|e| anyhow::anyhow!("invalid gossip frame from {}: {e}", remote.fmt_short()))?;
    if frame.sender_id.as_slice() != remote.as_bytes() {
        anyhow::bail!(
            "gossip sender_id mismatch from {}: connection identity does not match frame sender_id",
            remote.fmt_short()
        );
    }
    Ok(frame
        .peers
        .iter()
        .filter_map(proto_ann_to_local)
        .collect::<Vec<_>>())
}

#[cfg(test)]
pub(crate) fn encode_control_frame(stream_type: u8, msg: &impl prost::Message) -> Vec<u8> {
    let proto_bytes = msg.encode_to_vec();
    let len = proto_bytes.len() as u32;
    let mut buf = Vec::with_capacity(1 + 4 + proto_bytes.len());
    buf.push(stream_type);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&proto_bytes);
    buf
}

#[cfg(test)]
pub(crate) fn decode_control_frame<T: ValidateControlFrame>(
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

#[cfg(test)]
pub(crate) mod tests;
