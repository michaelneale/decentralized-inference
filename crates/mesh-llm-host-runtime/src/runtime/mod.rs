pub(crate) mod activity_policy;
mod auto_join;
mod capacity;
pub(crate) mod config_state;
mod context_planning;
mod control_loop;
mod daemon_startup;
mod dashboard;
mod discovery;
pub mod instance;
mod instance_lifecycle;
mod interactive;
mod local;
mod local_model_only;
mod local_package;
mod local_split;
mod model_lifecycle;
pub(crate) mod model_reconciliation;
mod operational_logging;
mod options;
mod plugin_host_role;
mod proxy;
mod publication;
mod release_attestation;
mod run_auto;
mod runtime_registry;
mod serving_surface;
mod split_participant_settle;
mod split_planning;
mod split_topology_lock;
pub(crate) mod stage_admission;
mod startup_handles;
mod startup_identity;
mod startup_models;
mod startup_retry;
mod status;
pub(crate) mod survey;
#[cfg(test)]
mod tests;
mod tracing_writer;
pub(crate) mod wakeable;

pub(crate) use self::capacity::runtime_model_required_bytes;
use self::capacity::{
    RuntimeCapacityLedger, RuntimeCapacityReservation, model_fits_runtime_capacity,
};
use self::context_planning::RuntimeResourcePlanningProfile;
pub(crate) use self::dashboard::runtime_status_from_process_status;
use self::discovery::{lan_rediscovery, nostr_rediscovery, start_new_mesh};
pub(crate) use self::instance_lifecycle::{
    DrainCoordinator, DrainResult, InstanceAdmissionError, InstanceLifecycleRecord,
    InstanceLifecycleState, InstanceRequestGuard,
};
use self::interactive::InitialPromptMode;
use self::local::{
    LocalOpenAiModelStartSpec, LocalRuntimeModelHandle, LocalRuntimeModelStartSpec,
    ManagedModelController, OpenAiGuardrailPolicyHandle, RuntimeEvent, SplitCoordinatorAck,
    SplitCoordinatorEvent, SplitRuntimeReason, SplitRuntimeStart, StartupRuntimePlan,
    add_runtime_local_target, add_serving_assignment, advertise_model_ready, local_process_payload,
    openai_guardrail_policy_handle, remove_runtime_local_target, remove_serving_assignment,
    resolved_model_name, runtime_model_planning_bytes, set_advertised_model_context,
    set_openai_guardrail_policy_mode, set_runtime_verified_served_model_capabilities,
    start_local_openai_model, start_runtime_local_model, start_runtime_split_model,
    startup_runtime_plan, stop_split_generation_cleanup, withdraw_advertised_model,
};
use self::local_model_only::*;
pub(crate) use self::model_reconciliation::{
    DesiredRuntimeIntent, IntentPersistence, IntentSource, ModelIntent,
    ModelTargetReconciliationAction, ModelTargetReconciliationCandidate,
    ModelTargetReconciliationCapacityState, ModelTargetReconciliationInput,
    ModelTargetReconciliationPolicy, ModelTargetReconciliationState, current_time_secs,
    plan_model_target_reconciliation,
};
pub use self::options::{MeshGuardrailMode, RuntimeOptions, RuntimeSurface};
use self::proxy::{api_proxy, bootstrap_proxy};
#[cfg(test)]
pub(crate) use self::release_attestation::assert_release_attestation_reports_missing_for_unstamped_binary;
use mesh_llm_events::{ConsoleSessionMode, sort_dashboard_endpoint_rows};
pub(crate) use mesh_llm_node::serving::{UnloadOptions, UnloadTarget};

pub(crate) use self::activity_policy::{ActivityPolicyGuard, AdmissionResult, IngressType};
pub use self::auto_join::console_session_mode_for_runtime_surface;
use self::auto_join::*;
use self::control_loop::*;
use self::dashboard::*;
pub use self::discovery::nostr_relays;
use self::model_lifecycle::*;
pub(crate) use self::operational_logging::{
    NativeSkippyOperationalEvent, RuntimeOperationalEvent, record_native_skippy_operational_event,
    record_runtime_operational_event, record_runtime_operational_event_with_context,
};
use self::publication::*;
pub use self::run_auto::load_resolved_plugins;
use self::run_auto::*;
pub(crate) use self::run_auto::{
    EmbeddedRuntimeDiscoveryMode, EmbeddedRuntimeMode, EmbeddedRuntimeOptions, run, run_cli,
    run_embedded_runtime,
};
use self::runtime_registry::*;
use self::serving_surface::*;
use self::startup_handles::*;
pub(crate) use self::startup_models::StartupPinnedGpuTarget;
use self::startup_models::*;
use self::tracing_writer::*;

#[cfg(test)]
pub(crate) use self::serving_surface::{
    assert_active_serve_path_spawn_gate_behavior,
    assert_interactive_handler_spawns_once_across_startup_callbacks,
    assert_passive_path_immediate_spawn_behavior,
    assert_quitting_during_startup_cancels_without_late_ready_render,
    assert_startup_launch_plan_describes_planned_runtime_before_process_start,
};
#[cfg(test)]
pub(crate) use self::startup_models::{
    assert_mesh_requirements_cli_accepts_each_bound_independently,
    assert_mesh_requirements_cli_overrides_config_per_field_before_genesis,
    assert_mesh_requirements_config_rejects_min_greater_than_max_after_merge,
    assert_mesh_requirements_rejects_local_policy_mutation_on_existing_mesh,
};
