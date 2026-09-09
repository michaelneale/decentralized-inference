use crate::api::routes::kv_cache::{KvCacheOperation, execute_operation};
use crate::mesh::owner_control_error_envelope;
use crate::proto::node::{
    OwnerControlEnvelope, OwnerControlErrorCode, OwnerControlKvCacheOperation,
    OwnerControlKvCacheRequest, OwnerControlKvCacheResponse, OwnerControlResponse,
};
use crate::protocol::NODE_PROTOCOL_GENERATION;

pub(crate) async fn execute(
    request_id: u64,
    request: OwnerControlKvCacheRequest,
) -> OwnerControlEnvelope {
    let operation = match decode_operation(&request) {
        Ok(operation) => operation,
        Err(message) => {
            return owner_control_error_envelope(
                OwnerControlErrorCode::BadRequest,
                Some(request_id),
                None,
                message,
            );
        }
    };

    match execute_operation(operation).await {
        Ok(result) => OwnerControlEnvelope {
            r#gen: NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(OwnerControlResponse {
                request_id,
                kv_cache: Some(OwnerControlKvCacheResponse {
                    status_json: result.status_json,
                    freed_bytes: result.freed_bytes,
                }),
                ..Default::default()
            }),
            error: None,
        },
        Err(error) => owner_control_error_envelope(
            OwnerControlErrorCode::ControlUnavailable,
            Some(request_id),
            None,
            error.to_string(),
        ),
    }
}

fn decode_operation(request: &OwnerControlKvCacheRequest) -> Result<KvCacheOperation, String> {
    let model_identity = request
        .model_identity
        .as_ref()
        .map(|identity| identity.trim().to_string())
        .transpose_nonempty("model_identity")?;
    match OwnerControlKvCacheOperation::try_from(request.operation).ok() {
        Some(OwnerControlKvCacheOperation::Status) => {
            if request.target_bytes.is_some() || model_identity.is_some() {
                return Err("kv-cache status does not accept mutation parameters".to_string());
            }
            Ok(KvCacheOperation::Status)
        }
        Some(OwnerControlKvCacheOperation::Prune) => Ok(KvCacheOperation::Prune {
            target_bytes: request.target_bytes,
            model_identity,
        }),
        Some(OwnerControlKvCacheOperation::Clear) => {
            if request.target_bytes.is_some() {
                return Err("kv-cache clear does not accept target_bytes".to_string());
            }
            Ok(KvCacheOperation::Clear { model_identity })
        }
        Some(OwnerControlKvCacheOperation::Unspecified) | None => {
            Err("kv-cache operation is unknown or unspecified".to_string())
        }
    }
}

trait NonemptyStringOption {
    fn transpose_nonempty(self, field: &str) -> Result<Option<String>, String>;
}

impl NonemptyStringOption for Option<String> {
    fn transpose_nonempty(self, field: &str) -> Result<Option<String>, String> {
        match self {
            Some(value) if value.is_empty() => Err(format!("{field} must not be empty")),
            value => Ok(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(operation: OwnerControlKvCacheOperation) -> OwnerControlKvCacheRequest {
        OwnerControlKvCacheRequest {
            requester_node_id: vec![1; 32],
            target_node_id: vec![2; 32],
            operation: operation as i32,
            target_bytes: None,
            model_identity: None,
        }
    }

    #[test]
    fn rejects_parameters_that_do_not_apply_to_operation() {
        let mut status = request(OwnerControlKvCacheOperation::Status);
        status.target_bytes = Some(1);
        assert!(decode_operation(&status).is_err());

        let mut clear = request(OwnerControlKvCacheOperation::Clear);
        clear.target_bytes = Some(1);
        assert!(decode_operation(&clear).is_err());
    }

    #[test]
    fn exact_model_identity_must_not_be_blank() {
        let mut prune = request(OwnerControlKvCacheOperation::Prune);
        prune.model_identity = Some("  ".to_string());
        assert!(decode_operation(&prune).is_err());
    }
}
