use std::ffi::c_char;

pub const STAGE_PLANNER_CONFIG_V1_ABI_VERSION: u32 = 1;
pub const STAGE_PLANNER_TENSOR_V1_ABI_VERSION: u32 = 1;
pub const STAGE_PLANNER_PROFILE_V1_ABI_VERSION: u32 = 1;
pub const STAGE_PLAN_DESC_V1_ABI_VERSION: u32 = 1;
pub const STAGE_PLAN_PROFILE_DESC_V1_ABI_VERSION: u32 = 1;
pub const STAGE_PLAN_VALUE_DESC_V1_ABI_VERSION: u32 = 1;
pub const STAGE_PLAN_STATE_DESC_V1_ABI_VERSION: u32 = 1;
pub const STAGE_PLAN_MAX_DIMS: usize = 4;

#[repr(C)]
pub struct StagePlanner {
    _private: [u8; 0],
}

#[repr(C)]
pub struct StagePlan {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagePlanStringRefV1 {
    pub offset: u64,
    pub length: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StagePlannerTensorV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub tensor_id: *const c_char,
    pub native_name: *const c_char,
    pub ggml_type: i32,
    pub dimension_count: u32,
    pub dimensions: [i64; STAGE_PLAN_MAX_DIMS],
    pub split_no: u32,
    pub reserved: u32,
    pub data_offset: u64,
    pub stored_length: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StagePlannerProfileV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub profile_id: *const c_char,
    pub n_tokens: u32,
    pub n_sequences: u32,
    pub n_outputs: u32,
    pub n_recurrent_rollback_sequences: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StagePlannerConfigV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub package_id: *const c_char,
    pub shard_paths: *const *const c_char,
    pub shard_count: usize,
    pub tensors: *const StagePlannerTensorV1,
    pub tensor_count: usize,
    pub profiles: *const StagePlannerProfileV1,
    pub profile_count: usize,
    pub graph_configuration_id: *const c_char,
    pub backend_id: *const c_char,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagePlanValueKind {
    ActivationImport = 0,
    ActivationExport = 1,
    RequestInput = 2,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagePlanStateKind {
    KvKey = 0,
    KvValue = 1,
    RecurrentConv = 2,
    RecurrentSsm = 3,
    DerivedPersistent = 4,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StagePlanStateAccess {
    Read = 0,
    Write = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagePlanDescV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub package_id: StagePlanStringRefV1,
    pub plan_id: StagePlanStringRefV1,
    pub layer_count: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub reserved: u32,
    pub profile_count: u64,
    pub resident_tensor_count: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagePlanProfileDescV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub profile_id: StagePlanStringRefV1,
    pub graph_identity: StagePlanStringRefV1,
    pub profile_identity: StagePlanStringRefV1,
    pub slice_identity: StagePlanStringRefV1,
    pub source_snapshot_identity: StagePlanStringRefV1,
    pub graph_configuration_id: StagePlanStringRefV1,
    pub backend_id: StagePlanStringRefV1,
    pub n_tokens: u32,
    pub n_sequences: u32,
    pub n_outputs: u32,
    pub n_recurrent_rollback_sequences: u32,
    pub activation_import_count: u64,
    pub activation_export_count: u64,
    pub request_input_count: u64,
    pub state_effect_count: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagePlanValueDescV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub identity: StagePlanStringRefV1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagePlanStateDescV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub identity: StagePlanStringRefV1,
    pub kind: i32,
    pub access: i32,
    pub layer: i32,
    pub reserved: i32,
    pub write_ordinal: i64,
}
