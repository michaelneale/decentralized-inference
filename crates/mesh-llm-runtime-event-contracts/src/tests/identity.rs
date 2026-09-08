use crate::{
    ChildOperationId, EventId, EventSequence, OperationId, OperationScope, ProcessInstanceId,
};

const UUID_BYTES: [u8; 16] = [
    0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0x4d, 0xef, 0x80, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde,
];

#[test]
fn request_root_bytes_match_the_logging_uuid_contract() {
    // Given
    let root = OperationId::from_bytes(UUID_BYTES);

    // When
    let bytes = root.into_bytes();

    // Then
    assert_eq!(bytes, UUID_BYTES);
    assert_eq!(root.to_string(), "12345678-9abc-4def-8012-3456789abcde");
}

#[test]
fn child_scope_retains_its_typed_root_relationship() {
    // Given
    let root = OperationId::from_bytes(UUID_BYTES);
    let child = ChildOperationId::from_bytes([0x55; 16]);

    // When
    let scope = OperationScope::with_child(root, child);

    // Then
    assert_eq!(scope.root(), root);
    assert_eq!(scope.child(), Some(child));
}

#[test]
fn event_identity_pairs_process_and_monotonic_sequence() {
    // Given
    let process = ProcessInstanceId::from_bytes([0x44; 16]);
    let sequence = EventSequence::new(17);

    // When
    let event_id = EventId::new(process, sequence);

    // Then
    assert_eq!(event_id.process_instance_id(), process);
    assert_eq!(event_id.sequence(), sequence);
}
