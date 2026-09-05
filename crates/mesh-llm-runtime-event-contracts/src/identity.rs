//! Process-local runtime event identities.
//!
//! Event sequence is the sole reducer order key. Native sequence is source data
//! only, and identities carry no cross-process ordering or correlation promise.
//! A request root `OperationId` stores the same UUID bytes as the logging
//! request ID, allowing correlation without a mapping table.

use std::fmt;

use uuid::Uuid;

macro_rules! uuid_identity {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(Uuid);

        impl $name {
            #[must_use]
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            #[must_use]
            pub const fn from_uuid(value: Uuid) -> Self {
                Self(value)
            }

            #[must_use]
            pub const fn from_bytes(value: [u8; 16]) -> Self {
                Self(Uuid::from_bytes(value))
            }

            #[must_use]
            pub const fn into_bytes(self) -> [u8; 16] {
                self.0.into_bytes()
            }

            #[must_use]
            pub const fn as_uuid(&self) -> &Uuid {
                &self.0
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }
    };
}

uuid_identity!(ProcessInstanceId);
uuid_identity!(OperationId);
uuid_identity!(ChildOperationId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EventSequence(u64);

impl EventSequence {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId {
    process_instance_id: ProcessInstanceId,
    sequence: EventSequence,
}

impl EventId {
    #[must_use]
    pub const fn new(process_instance_id: ProcessInstanceId, sequence: EventSequence) -> Self {
        Self {
            process_instance_id,
            sequence,
        }
    }

    #[must_use]
    pub const fn process_instance_id(self) -> ProcessInstanceId {
        self.process_instance_id
    }

    #[must_use]
    pub const fn sequence(self) -> EventSequence {
        self.sequence
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationScope {
    Root(OperationId),
    Child {
        root: OperationId,
        child: ChildOperationId,
    },
}

impl OperationScope {
    #[must_use]
    pub const fn root_only(root: OperationId) -> Self {
        Self::Root(root)
    }

    #[must_use]
    pub const fn with_child(root: OperationId, child: ChildOperationId) -> Self {
        Self::Child { root, child }
    }

    #[must_use]
    pub const fn root(self) -> OperationId {
        match self {
            Self::Root(root) | Self::Child { root, .. } => root,
        }
    }

    #[must_use]
    pub const fn child(self) -> Option<ChildOperationId> {
        match self {
            Self::Root(_) => None,
            Self::Child { child, .. } => Some(child),
        }
    }
}
