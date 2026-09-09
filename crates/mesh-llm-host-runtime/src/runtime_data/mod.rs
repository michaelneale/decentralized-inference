//! Runtime-data snapshot ownership and compatibility guardrails.
//!
//! Broad runtime reads should go through the collector so API payloads stay
//! stable while subsystem publishers mutate their own snapshots.

mod api_views;
mod collector;
mod inventory;
mod metrics;
#[cfg(test)]
mod plugin_tests;
mod plugins;
mod processes;
mod producers;
mod snapshots;
mod subscriptions;

pub(crate) use self::api_views::{collect_views, mesh_models, status_payload};
pub(crate) use self::collector::RuntimeDataCollector;
#[cfg(test)]
pub(crate) use self::inventory::InventoryScanError;
pub(crate) use self::inventory::{
    InventoryScanDisposition, InventoryScanOutcome, InventoryScanResult, sorted_inventory_entries,
};
#[cfg(test)]
pub(crate) use self::metrics::RuntimeLlamaMetricSample;
pub(crate) use self::metrics::{
    RuntimeLlamaEndpointStatus, RuntimeLlamaMetricItem, RuntimeLlamaMetricsSnapshot,
    RuntimeLlamaRuntimeItems, RuntimeLlamaRuntimeSnapshot, RuntimeLlamaSlotItem,
    RuntimeLlamaSlotSnapshot, RuntimeLlamaSlotsSnapshot,
};
pub(crate) use self::processes::{
    RuntimeProcessSnapshot, remove_runtime_process_snapshot, runtime_process_payloads,
    upsert_runtime_process_snapshot,
};
pub(crate) use self::producers::{RuntimeDataProducer, RuntimeDataSource};
pub(crate) use self::snapshots::{
    HardwareViewInput, ModelViewInput, PluginDataKey, PluginEndpointKey, StatusViewInput,
};
pub(crate) use self::subscriptions::RuntimeDataDirty;

#[cfg(test)]
pub(crate) mod tests {
    mod inventory;
    use super::api_views::{collect_views, mesh_models, status_payload};
    use super::processes::{RuntimeProcessSnapshot, runtime_process_payloads};
    use super::snapshots::{
        HardwareViewInput, ModelViewInput, PluginDataKey, PluginEndpointKey, StatusViewInput,
    };
    use super::subscriptions::{RuntimeDataDirty, RuntimeDataVersion};
    use super::{InventoryScanDisposition, InventoryScanError, InventoryScanResult};
    use super::{RuntimeDataCollector, RuntimeDataSource};
    use super::{RuntimeLlamaEndpointStatus, RuntimeLlamaSlotSnapshot, RuntimeLlamaSlotsSnapshot};
    use crate::api::RuntimeProcessPayload;
    use crate::api::status::{
        LocalInstance, NodeState, RuntimeStatusPayload, StatusPayload, build_gpus,
        build_ownership_payload,
    };
    use crate::inference::election;
    use crate::mesh::{MeshCatalogEntry, NodeRole, PeerInfo};
    use crate::models::LocalModelInventorySnapshot;
    use crate::network::openai::transport::{self, ResponseAdapter};
    use crate::plugin::{PluginEndpointSummary, PluginSummary, PluginWebUiState};
    use crate::runtime::instance::LocalInstanceSnapshot;
    use crate::{ReleaseAttestationStatus, ReleaseAttestationSummary};
    use iroh::{EndpointAddr, EndpointId, SecretKey};
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::{collections::HashMap, collections::HashSet};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    #[test]
    fn runtime_data_collector_shell_constructs_and_clones() {
        let collector = RuntimeDataCollector::new();
        let clone = collector.clone();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
        let plugin_data_key = PluginDataKey {
            plugin_name: "plugin-a".into(),
            data_key: "status".into(),
        };
        let plugin_endpoint_key = PluginEndpointKey {
            plugin_name: "plugin-b".into(),
            endpoint_id: "chat".into(),
        };
        let plugin_data_producer = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: Some(plugin_data_key.clone()),
            plugin_endpoint_key: None,
        });
        let plugin_endpoint_producer = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: None,
            plugin_endpoint_key: Some(plugin_endpoint_key.clone()),
        });

        assert_eq!(producer.source().scope, "runtime");
        assert!(producer.source().plugin_data_key.is_none());
        assert!(producer.source().plugin_endpoint_key.is_none());
        assert_eq!(
            plugin_data_producer.source().plugin_data_key.as_ref(),
            Some(&plugin_data_key)
        );
        assert_eq!(
            plugin_endpoint_producer
                .source()
                .plugin_endpoint_key
                .as_ref(),
            Some(&plugin_endpoint_key)
        );
        assert!(
            producer
                .snapshots()
                .runtime_status
                .local_processes
                .is_empty()
        );
        assert!(clone.snapshots().local_instances.instances.is_empty());
        assert!(
            producer
                .collector()
                .plugin_data_snapshot()
                .entries
                .is_empty()
        );
    }

    #[test]
    fn runtime_data_collector_exposes_initial_snapshots() {
        let collector = RuntimeDataCollector::new();
        let views = collect_views(&collector);

        assert!(
            collector
                .runtime_status_snapshot()
                .local_processes
                .is_empty()
        );
        assert!(collector.local_instances_snapshot().instances.is_empty());
        assert!(collector.plugin_data_snapshot().entries.is_empty());
        assert!(collector.plugin_endpoints_snapshot().entries.is_empty());
        assert!(views.runtime_status.primary_model.is_none());
        assert!(views.runtime_status.primary_backend.is_none());
        assert!(!views.runtime_status.is_host);
        assert!(!views.runtime_status.is_client);
        assert!(!views.runtime_status.llama_ready);
        assert!(views.runtime_status.llama_port.is_none());
        assert!(views.runtime_status.local_processes.is_empty());
        assert!(views.local_instances.instances.is_empty());
        assert!(views.plugin_data.entries.is_empty());
        assert!(views.plugin_endpoints.entries.is_empty());
    }

    #[test]
    fn runtime_data_version_advances_and_marks_dirty_bits() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });

        let initial = collector.subscription_state();
        assert_runtime_data_state_contains_dirty(&initial, 0, &[]);

        let status_state = producer.mark_status_dirty();
        assert_runtime_data_state_contains_dirty(&status_state, 1, &[RuntimeDataDirty::STATUS]);
        assert_eq!(status_state.dirty, RuntimeDataDirty::STATUS);

        let processes_changed = producer.publish_local_processes(|local_processes| {
            local_processes.push(RuntimeProcessSnapshot {
                model: "Qwen3-8B".into(),
                instance_id: None,
                profile: String::new(),
                backend: "metal".into(),
                pid: 4242,
                port: 9337,
                slots: 4,
                context_length: Some(8192),
                command: None,
                state: "ready".into(),
                start: None,
                health: Some("ready".into()),
            });
            true
        });
        assert!(processes_changed);

        let processes_state = collector.subscription_state();
        assert_runtime_data_state_contains_dirty(
            &processes_state,
            2,
            &[RuntimeDataDirty::STATUS, RuntimeDataDirty::PROCESSES],
        );

        let no_change = producer.publish_local_processes(|_| false);
        assert!(!no_change);
        assert_eq!(collector.subscription_state(), processes_state);

        let models_state = producer.mark_models_dirty();
        assert_runtime_data_state_contains_dirty(
            &models_state,
            3,
            &[
                RuntimeDataDirty::STATUS,
                RuntimeDataDirty::PROCESSES,
                RuntimeDataDirty::MODELS,
            ],
        );

        let routing_state = producer.mark_routing_dirty();
        assert_runtime_data_state_contains_dirty(&routing_state, 4, &[RuntimeDataDirty::ROUTING]);

        let processes_state = producer.mark_processes_dirty();
        assert_runtime_data_state_contains_dirty(
            &processes_state,
            5,
            &[RuntimeDataDirty::PROCESSES],
        );

        let inventory_state = producer.mark_inventory_dirty();
        assert_runtime_data_state_contains_dirty(
            &inventory_state,
            6,
            &[RuntimeDataDirty::INVENTORY],
        );

        let plugins_state = producer.mark_plugins_dirty();
        assert_runtime_data_state_contains_dirty(&plugins_state, 7, &[RuntimeDataDirty::PLUGINS]);

        let runtime_status_changed = producer.publish_runtime_status(|runtime_status| {
            runtime_status.primary_backend = Some("metal".into());
            true
        });
        assert!(runtime_status_changed);

        let final_state = collector.subscription_state();
        assert_runtime_data_state_contains_dirty(&final_state, 8, &[RuntimeDataDirty::STATUS]);
    }

    fn assert_runtime_data_state_contains_dirty(
        state: &super::subscriptions::RuntimeDataSubscriptionState,
        expected_version: u64,
        expected_dirty: &[RuntimeDataDirty],
    ) {
        if expected_version == 0 {
            assert_eq!(state.version, RuntimeDataVersion::default());
        } else {
            assert_eq!(state.version.get(), expected_version);
        }

        if expected_dirty.is_empty() {
            assert!(state.dirty.is_empty());
            return;
        }

        for dirty in expected_dirty {
            assert!(state.dirty.contains(*dirty));
        }
    }

    #[tokio::test]
    async fn runtime_data_subscribe_notifies_once_per_update() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
        let mut subscription = collector.subscribe();

        assert!(!subscription.has_changed().expect("watch channel open"));

        producer.mark_status_dirty();
        subscription
            .changed()
            .await
            .expect("status update delivered");
        let first = *subscription.borrow_and_update();

        assert_eq!(first.version.get(), 1);
        assert!(first.dirty.contains(RuntimeDataDirty::STATUS));
        assert!(!subscription.has_changed().expect("watch channel open"));

        producer.mark_models_dirty();
        producer.mark_routing_dirty();
        subscription
            .changed()
            .await
            .expect("coalesced updates delivered");
        let second = *subscription.borrow_and_update();

        assert_eq!(second.version.get(), 3);
        assert!(second.dirty.contains(RuntimeDataDirty::STATUS));
        assert!(second.dirty.contains(RuntimeDataDirty::MODELS));
        assert!(second.dirty.contains(RuntimeDataDirty::ROUTING));
        assert!(!subscription.has_changed().expect("watch channel open"));
    }

    #[test]
    fn runtime_data_process_snapshot_matches_existing_runtime_views() {
        let legacy_processes = vec![
            RuntimeProcessPayload {
                name: "Zulu".into(),
                instance_id: None,
                profile: String::new(),
                backend: "llama".into(),
                status: "ready".into(),
                port: 9444,
                pid: 11,
                slots: 4,
                context_length: None,
            },
            RuntimeProcessPayload {
                name: "Alpha".into(),
                instance_id: None,
                profile: String::new(),
                backend: "llama".into(),
                status: "starting".into(),
                port: 9337,
                pid: 10,
                slots: 4,
                context_length: None,
            },
        ];
        let collector_rows = legacy_processes
            .iter()
            .map(RuntimeProcessSnapshot::from_payload)
            .collect::<Vec<_>>();

        assert_eq!(collector_rows[0].model, "Zulu");
        assert_eq!(collector_rows[0].backend, "llama");
        assert_eq!(collector_rows[0].pid, 11);
        assert_eq!(collector_rows[0].port, 9444);
        assert_eq!(collector_rows[0].command, None);
        assert_eq!(collector_rows[0].state, "ready");
        assert_eq!(collector_rows[0].start, None);
        assert_eq!(collector_rows[0].health.as_deref(), Some("ready"));

        let round_trip = runtime_process_payloads(&collector_rows);
        assert_eq!(round_trip, legacy_processes);
    }

    #[test]
    fn runtime_data_status_snapshot_matches_api_payloads() {
        let collector = RuntimeDataCollector::new();
        collector.replace_local_instances_snapshot(vec![LocalInstanceSnapshot {
            pid: 111,
            api_port: Some(3131),
            version: Some("0.68.0".into()),
            started_at_unix: 456,
            runtime_dir: PathBuf::from("/tmp/runtime-1"),
            is_self: true,
        }]);
        let hardware = collector.build_hardware_view(HardwareViewInput {
            gpu_name: Some("RTX 4090".into()),
            gpu_vram: Some("25769803776".into()),
            gpu_reserved_bytes: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            my_hostname: Some("node.local".into()),
            my_is_soc: Some(false),
            my_vram_gb: 25.769803776,
            model_size_gb: 12.5,
            first_joined_mesh_ts: Some(123),
        });
        let snapshot = collector.build_status_view(StatusViewInput {
            version: "0.68.0".into(),
            latest_version: Some("0.68.0".into()),
            node_id: "node-1".into(),
            owner: crate::crypto::OwnershipSummary::default(),
            release_attestation: ReleaseAttestationSummary {
                status: ReleaseAttestationStatus::Valid,
                signer_key_id: Some("ed25519:test-signer".into()),
                verified: true,
                ..ReleaseAttestationSummary::default()
            },
            token: "invite-token".into(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            model_name: "Qwen-Test".into(),
            models: vec!["Qwen-Test".into()],
            available_models: vec!["Qwen-Test".into()],
            requested_models: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            inflight_requests: 2,
            mesh_id: Some("mesh-1".into()),
            mesh_name: Some("test-mesh".into()),
            mesh_discovery_mode: "nostr".into(),
            discovery_scope: "public".into(),
            discovery_source: "nostr-relay".into(),
            nostr_discovery: true,
            publication_state: "public".into(),
            local_processes: vec![],
            peers: vec![],
            wakeable_nodes: vec![],
            routing_affinity: crate::network::affinity::AffinityStatsSnapshot::default(),
            hardware,
        });

        let payload = status_payload(snapshot);
        let expected = StatusPayload {
            version: "0.68.0".into(),
            latest_version: Some("0.68.0".into()),
            node_id: "node-1".into(),
            owner: build_ownership_payload(&crate::crypto::OwnershipSummary::default()),
            release_attestation: ReleaseAttestationSummary {
                status: ReleaseAttestationStatus::Valid,
                signer_key_id: Some("ed25519:test-signer".into()),
                verified: true,
                ..ReleaseAttestationSummary::default()
            },
            token: "invite-token".into(),
            node_state: NodeState::Standby,
            node_status: NodeState::Standby.node_status_alias().into(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            runtime: RuntimeStatusPayload {
                backend: None,
                openai_guardrails: None,
                models: vec![],
                stages: vec![],
                daemon_state: None,
                capabilities: None,
                lifecycle_instances: vec![],
                intent_summary: None,
            },
            model_name: "Qwen-Test".into(),
            models: vec!["Qwen-Test".into()],
            available_models: vec!["Qwen-Test".into()],
            requested_models: vec![],
            wanted_model_refs: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 25.769803776,
            model_size_gb: 12.5,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![LocalInstance {
                pid: 111,
                api_port: Some(3131),
                version: Some("0.68.0".into()),
                started_at_unix: 456,
                runtime_dir: "/tmp/runtime-1".into(),
                is_self: true,
            }],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 2,
            mesh_id: Some("mesh-1".into()),
            mesh_name: Some("test-mesh".into()),
            mesh_discovery_mode: "nostr".into(),
            discovery_scope: "public".into(),
            discovery_source: "nostr-relay".into(),
            nostr_discovery: true,
            publication_state: "public".into(),
            my_hostname: Some("node.local".into()),
            my_is_soc: Some(false),
            gpus: build_gpus(
                Some("RTX 4090"),
                Some("25769803776"),
                None,
                None,
                None,
                None,
            ),
            routing_affinity: crate::network::affinity::AffinityStatsSnapshot::default(),
            routing_metrics: crate::network::metrics::RoutingMetricsStatusSnapshot::default(),
            first_joined_mesh_ts: Some(123),
            mesh_requirements: None,
            recent_mesh_rejections: vec![],
            logging: None,
        };

        assert_eq!(
            serde_json::to_value(&payload).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
    }

    pub(crate) fn assert_release_attestation_status_surfaces_in_api_and_runtime_data() {
        let collector = RuntimeDataCollector::new();
        let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x33; 32]).public());
        let peer = PeerInfo {
            id: peer_id,
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            mesh_id: None,
            mesh_policy_hash: None,
            genesis_policy: None,
            role: NodeRole::Worker,
            first_joined_mesh_ts: Some(456),
            models: vec!["Peer-Model".into()],
            vram_bytes: 32_000_000_000,
            rtt_ms: Some(7),
            model_source: None,
            admitted: true,
            serving_models: vec!["Peer-Model".into()],
            hosted_models: vec!["Peer-Model".into()],
            hosted_models_known: true,
            available_models: vec!["Peer-Model".into()],
            requested_models: vec![],
            explicit_model_interests: vec![],
            last_seen: std::time::Instant::now(),
            last_mentioned: std::time::Instant::now(),
            version: Some("0.66.0".into()),
            gpu_name: None,
            hostname: Some("peer.local".into()),
            is_soc: Some(false),
            gpu_vram: None,
            gpu_reserved_bytes: None,
            memory: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
            release_attestation_summary: ReleaseAttestationSummary {
                status: ReleaseAttestationStatus::Invalid,
                signer_key_id: Some("ed25519:peer-signer".into()),
                error: Some("release attestation signature verification failed".into()),
                ..ReleaseAttestationSummary::default()
            },
            artifact_transfer_supported: false,
            stage_protocol_generation_supported: false,
            stage_status_list_supported: false,
            local_gguf_content_id_supported: false,
            advertised_model_throughput: vec![],
            cache_affinity: None,
            display_rtt: None,
            selected_path: None,
            propagated_latency: None,
            owner_summary: crate::crypto::OwnershipSummary::default(),
            inference_admission_state: None,
        };
        let hardware = collector.build_hardware_view(HardwareViewInput {
            gpu_name: None,
            gpu_vram: None,
            gpu_reserved_bytes: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            my_hostname: Some("node.local".into()),
            my_is_soc: Some(false),
            my_vram_gb: 24.0,
            model_size_gb: 8.0,
            first_joined_mesh_ts: Some(123),
        });
        let snapshot = collector.build_status_view(StatusViewInput {
            version: "0.66.0".into(),
            latest_version: Some("0.66.0".into()),
            node_id: "node-1".into(),
            owner: crate::crypto::OwnershipSummary::default(),
            release_attestation: ReleaseAttestationSummary {
                status: ReleaseAttestationStatus::Valid,
                signer_key_id: Some("ed25519:self-signer".into()),
                node_version: Some("0.66.0".into()),
                verified: true,
                ..ReleaseAttestationSummary::default()
            },
            token: "invite-token".into(),
            is_host: true,
            is_client: false,
            llama_ready: true,
            model_name: "Self-Model".into(),
            models: vec!["Self-Model".into()],
            available_models: vec!["Self-Model".into()],
            requested_models: vec![],
            serving_models: vec!["Self-Model".into()],
            hosted_models: vec!["Self-Model".into()],
            draft_name: None,
            api_port: 3131,
            inflight_requests: 1,
            mesh_id: Some("mesh-1".into()),
            mesh_name: Some("test-mesh".into()),
            mesh_discovery_mode: "mdns".into(),
            discovery_scope: "lan".into(),
            discovery_source: "mdns-sd".into(),
            nostr_discovery: false,
            publication_state: "private".into(),
            local_processes: vec![],
            peers: vec![peer],
            wakeable_nodes: vec![],
            routing_affinity: crate::network::affinity::AffinityStatsSnapshot::default(),
            hardware,
        });

        assert_eq!(
            snapshot.release_attestation.status,
            ReleaseAttestationStatus::Valid
        );
        assert_eq!(
            snapshot.peers[0].release_attestation.status,
            ReleaseAttestationStatus::Invalid
        );
        assert_eq!(snapshot.peers[0].owner.status, "unsigned");

        let payload = status_payload(snapshot);
        assert_eq!(
            payload.release_attestation.status,
            ReleaseAttestationStatus::Valid
        );
        assert_eq!(payload.owner.status, "unsigned");
        assert_eq!(
            payload.peers[0].release_attestation.status,
            ReleaseAttestationStatus::Invalid
        );
        assert_eq!(
            payload.peers[0]
                .release_attestation
                .signer_key_id
                .as_deref(),
            Some("ed25519:peer-signer")
        );
        assert_eq!(payload.peers[0].owner.status, "unsigned");
    }

    #[test]
    fn status_payload_exposes_peer_advertised_model_throughput() {
        let collector = RuntimeDataCollector::new();
        let peer_id = EndpointId::from(SecretKey::from_bytes(&[0x44; 32]).public());
        let peer = PeerInfo {
            id: peer_id,
            addr: EndpointAddr {
                id: peer_id,
                addrs: Default::default(),
            },
            mesh_id: None,
            mesh_policy_hash: None,
            genesis_policy: None,
            role: NodeRole::Worker,
            first_joined_mesh_ts: Some(456),
            models: vec!["Qwen/Qwen3-Coder".into()],
            vram_bytes: 32_000_000_000,
            rtt_ms: Some(7),
            model_source: None,
            admitted: true,
            serving_models: vec!["Qwen/Qwen3-Coder".into()],
            hosted_models: vec!["Qwen/Qwen3-Coder".into()],
            hosted_models_known: true,
            available_models: vec!["Qwen/Qwen3-Coder".into()],
            requested_models: vec![],
            explicit_model_interests: vec![],
            last_seen: std::time::Instant::now(),
            last_mentioned: std::time::Instant::now(),
            version: Some("0.70.0".into()),
            gpu_name: None,
            hostname: Some("peer.local".into()),
            is_soc: Some(false),
            gpu_vram: None,
            gpu_reserved_bytes: None,
            memory: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            available_model_metadata: vec![],
            experts_summary: None,
            available_model_sizes: HashMap::new(),
            served_model_descriptors: vec![],
            served_model_runtime: vec![],
            owner_attestation: None,
            release_attestation_summary: ReleaseAttestationSummary::default(),
            artifact_transfer_supported: false,
            stage_protocol_generation_supported: false,
            stage_status_list_supported: false,
            local_gguf_content_id_supported: false,
            advertised_model_throughput: vec![crate::network::metrics::ModelThroughputHint {
                model_name: "Qwen/Qwen3-Coder".into(),
                avg_tokens_per_second_milli: 13_400,
                throughput_samples: 27,
            }],
            cache_affinity: None,
            display_rtt: None,
            selected_path: None,
            propagated_latency: None,
            owner_summary: crate::crypto::OwnershipSummary::default(),
            inference_admission_state: None,
        };
        let hardware = collector.build_hardware_view(HardwareViewInput {
            gpu_name: None,
            gpu_vram: None,
            gpu_reserved_bytes: None,
            gpu_mem_bandwidth_gbps: None,
            gpu_compute_tflops_fp32: None,
            gpu_compute_tflops_fp16: None,
            my_hostname: Some("node.local".into()),
            my_is_soc: Some(false),
            my_vram_gb: 24.0,
            model_size_gb: 8.0,
            first_joined_mesh_ts: Some(123),
        });
        let snapshot = collector.build_status_view(StatusViewInput {
            version: "0.70.0".into(),
            latest_version: Some("0.70.0".into()),
            node_id: "node-1".into(),
            owner: crate::crypto::OwnershipSummary::default(),
            release_attestation: ReleaseAttestationSummary::default(),
            token: "invite-token".into(),
            is_host: true,
            is_client: false,
            llama_ready: true,
            model_name: "Self-Model".into(),
            models: vec!["Self-Model".into()],
            available_models: vec!["Self-Model".into()],
            requested_models: vec![],
            serving_models: vec!["Self-Model".into()],
            hosted_models: vec!["Self-Model".into()],
            draft_name: None,
            api_port: 3131,
            inflight_requests: 1,
            mesh_id: Some("mesh-1".into()),
            mesh_name: Some("test-mesh".into()),
            mesh_discovery_mode: "nostr".into(),
            discovery_scope: "public".into(),
            discovery_source: "nostr-relay".into(),
            nostr_discovery: false,
            publication_state: "private".into(),
            local_processes: vec![],
            peers: vec![peer],
            wakeable_nodes: vec![],
            routing_affinity: crate::network::affinity::AffinityStatsSnapshot::default(),
            hardware,
        });

        assert_eq!(
            snapshot.peers[0].advertised_model_throughput[0].model_name,
            "Qwen/Qwen3-Coder"
        );

        let payload = status_payload(snapshot);
        assert_eq!(payload.peers[0].advertised_model_throughput.len(), 1);

        let json = serde_json::to_value(&payload).expect("serialize status payload");
        assert_eq!(
            json["peers"][0]["advertised_model_throughput"],
            json!([
                {
                    "model_name": "Qwen/Qwen3-Coder",
                    "avg_tokens_per_second_milli": 13400,
                    "throughput_samples": 27,
                }
            ])
        );
    }

    #[test]
    fn runtime_data_model_snapshot_matches_api_payloads() {
        let collector = RuntimeDataCollector::new();
        let local_inventory = LocalModelInventorySnapshot {
            model_names: HashSet::from(["Example-Model".to_string()]),
            size_by_name: HashMap::from([("Example-Model".to_string(), 8_000_000_000)]),
            metadata_by_name: HashMap::from([(
                "Example-Model".to_string(),
                crate::proto::node::CompactModelMetadata {
                    model_key: "Example-Model".to_string(),
                    context_length: 131_072,
                    embedding_size: 4096,
                    head_count: 32,
                    layer_count: 36,
                    tokenizer_model_name: "gpt2".to_string(),
                    quantization_type: "Q4_K_M".to_string(),
                    ..Default::default()
                },
            )]),
            display_name_by_name: HashMap::new(),
        };
        let snapshot = collector.build_model_view(ModelViewInput {
            peers: vec![],
            catalog: vec![MeshCatalogEntry {
                model_name: "Example-Model".into(),
                descriptor: None,
            }],
            served_models: vec![],
            active_demand: HashMap::new(),
            my_serving_models: vec![],
            my_hosted_models: vec![],
            local_inventory,
            node_hostname: Some("node.local".into()),
            my_vram_gb: 24.0,
            model_name: "Another-Model".into(),
            model_size_bytes: 0,
            now_unix_secs: 1_700_000_000,
        });

        let payload = mesh_models(snapshot);
        assert_eq!(payload.len(), 1);
        assert_eq!(payload[0].name, "Example-Model");
        assert_eq!(payload[0].status, "cold");
        assert_eq!(payload[0].size_gb, 8.0);
        assert_eq!(payload[0].context_length, Some(131_072));
        assert_eq!(payload[0].quantization, Some("Q4_K_M".to_string()));
        assert_eq!(payload[0].tokenizer, Some("gpt2".to_string()));
        assert_eq!(payload[0].layer_count, Some(36));
        assert_eq!(payload[0].head_count, Some(32));
        assert_eq!(payload[0].embedding_size, Some(4096));
        assert_eq!(
            payload[0].download_command,
            "mesh-llm models download Example-Model"
        );
        assert_eq!(
            payload[0].run_command,
            "mesh-llm serve --model Example-Model"
        );
        assert_eq!(
            payload[0].auto_command,
            "mesh-llm serve --auto --model Example-Model"
        );
        assert_eq!(payload[0].fit_label, "Likely comfortable");
    }

    #[test]
    fn runtime_data_model_snapshot_uses_local_display_name_for_synthetic_refs() {
        let collector = RuntimeDataCollector::new();
        let synthetic_ref = "local-gguf/sha256-66243256b95c5f7c".to_string();
        let local_inventory = LocalModelInventorySnapshot {
            model_names: HashSet::from([synthetic_ref.clone()]),
            size_by_name: HashMap::from([(synthetic_ref.clone(), 8_000_000_000)]),
            metadata_by_name: HashMap::from([(
                synthetic_ref.clone(),
                crate::proto::node::CompactModelMetadata {
                    model_key: synthetic_ref.clone(),
                    context_length: 32_768,
                    quantization_type: "Q4_K_M".to_string(),
                    ..Default::default()
                },
            )]),
            display_name_by_name: HashMap::from([(
                synthetic_ref.clone(),
                "MyModel-7B-Q4_K_M".to_string(),
            )]),
        };
        let snapshot = collector.build_model_view(ModelViewInput {
            peers: vec![],
            catalog: vec![MeshCatalogEntry {
                model_name: synthetic_ref.clone(),
                descriptor: None,
            }],
            served_models: vec![],
            active_demand: HashMap::new(),
            my_serving_models: vec![],
            my_hosted_models: vec![],
            local_inventory,
            node_hostname: Some("node.local".into()),
            my_vram_gb: 24.0,
            model_name: "Another-Model".into(),
            model_size_bytes: 0,
            now_unix_secs: 1_700_000_000,
        });

        let payload = mesh_models(snapshot);
        assert_eq!(payload.len(), 1);
        // The canonical name stays the synthetic ref.
        assert_eq!(payload[0].name, synthetic_ref);
        // The display name is the GGUF file stem, not the synthetic ref.
        assert_eq!(payload[0].display_name, "MyModel-7B-Q4_K_M");
        // Local metadata still flows through.
        assert_eq!(payload[0].size_gb, 8.0);
        assert_eq!(payload[0].context_length, Some(32_768));
        assert_eq!(payload[0].quantization, Some("Q4_K_M".to_string()));
    }

    #[test]
    fn runtime_data_model_snapshot_includes_routable_model_refs_without_catalog_entry() {
        let collector = RuntimeDataCollector::new();
        let model_ref = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL".to_string();
        let snapshot = collector.build_model_view(ModelViewInput {
            peers: vec![],
            catalog: vec![],
            served_models: vec![model_ref.clone()],
            active_demand: HashMap::new(),
            my_serving_models: vec![model_ref.clone()],
            my_hosted_models: vec![model_ref.clone()],
            local_inventory: LocalModelInventorySnapshot::default(),
            node_hostname: Some("white".into()),
            my_vram_gb: 28.0,
            model_name: model_ref.clone(),
            model_size_bytes: 22_000_000_000,
            now_unix_secs: 1_700_000_000,
        });

        let payload = mesh_models(snapshot);
        let model = payload
            .iter()
            .find(|model| model.name == model_ref)
            .expect("routable model ref should be exposed as a mesh model");
        assert_eq!(model.status, "warm");
        assert_eq!(model.node_count, 1);
        assert_eq!(model.active_nodes, vec!["white".to_string()]);
        assert_eq!(model.size_gb, 22.0);
    }

    #[test]
    fn runtime_data_model_snapshot_keeps_known_text_only_descriptor_authoritative() {
        let collector = RuntimeDataCollector::new();
        let model_name = "Qwen3VL-2B-Instruct-Q4_K_M".to_string();
        let descriptor = crate::mesh::ServedModelDescriptor {
            identity: crate::mesh::ServedModelIdentity {
                model_name: model_name.clone(),
                source_kind: crate::mesh::ModelSourceKind::LocalGguf,
                local_file_name: Some(format!("{model_name}.gguf")),
                ..Default::default()
            },
            capabilities_known: true,
            capabilities: crate::models::ModelCapabilities::default(),
            topology: None,
            metadata: None,
        };

        let snapshot = collector.build_model_view(ModelViewInput {
            peers: vec![],
            catalog: vec![MeshCatalogEntry {
                model_name: model_name.clone(),
                descriptor: Some(descriptor),
            }],
            served_models: vec![model_name.clone()],
            active_demand: HashMap::new(),
            my_serving_models: vec![model_name.clone()],
            my_hosted_models: vec![model_name.clone()],
            local_inventory: LocalModelInventorySnapshot {
                model_names: HashSet::from([model_name.clone()]),
                size_by_name: HashMap::new(),
                metadata_by_name: HashMap::new(),
                display_name_by_name: HashMap::new(),
            },
            node_hostname: Some("node.local".into()),
            my_vram_gb: 24.0,
            model_name: model_name.clone(),
            model_size_bytes: 0,
            now_unix_secs: 1_700_000_000,
        });

        let payload = mesh_models(snapshot);
        assert_eq!(payload.len(), 1);
        assert_eq!(payload[0].name, model_name);
        assert_eq!(payload[0].status, "warm");
        assert!(!payload[0].multimodal);
        assert_eq!(payload[0].multimodal_status, None);
        assert!(!payload[0].vision);
        assert_eq!(payload[0].vision_status, None);
    }

    #[test]
    fn runtime_data_model_snapshot_uses_static_media_for_unknown_descriptor() {
        let collector = RuntimeDataCollector::new();
        let model_name = "Qwen3VL-2B-Instruct-Q4_K_M".to_string();
        let descriptor = crate::mesh::ServedModelDescriptor {
            identity: crate::mesh::ServedModelIdentity {
                model_name: model_name.clone(),
                source_kind: crate::mesh::ModelSourceKind::LocalGguf,
                local_file_name: Some(format!("{model_name}.gguf")),
                ..Default::default()
            },
            capabilities_known: false,
            capabilities: crate::models::ModelCapabilities::default(),
            topology: None,
            metadata: None,
        };

        let snapshot = collector.build_model_view(ModelViewInput {
            peers: vec![],
            catalog: vec![MeshCatalogEntry {
                model_name: model_name.clone(),
                descriptor: Some(descriptor),
            }],
            served_models: vec![model_name.clone()],
            active_demand: HashMap::new(),
            my_serving_models: vec![model_name.clone()],
            my_hosted_models: vec![model_name.clone()],
            local_inventory: LocalModelInventorySnapshot {
                model_names: HashSet::from([model_name.clone()]),
                size_by_name: HashMap::new(),
                metadata_by_name: HashMap::new(),
                display_name_by_name: HashMap::new(),
            },
            node_hostname: Some("node.local".into()),
            my_vram_gb: 24.0,
            model_name: model_name.clone(),
            model_size_bytes: 0,
            now_unix_secs: 1_700_000_000,
        });

        let payload = mesh_models(snapshot);
        assert_eq!(payload.len(), 1);
        assert_eq!(payload[0].name, model_name);
        assert!(payload[0].multimodal);
        assert_eq!(payload[0].multimodal_status, Some("supported"));
        assert!(payload[0].vision);
        assert_eq!(payload[0].vision_status, Some("supported"));
    }

    #[test]
    fn runtime_data_model_snapshot_reports_known_verified_vision_descriptor() {
        let collector = RuntimeDataCollector::new();
        let model_name = "Qwen3VL-2B-Instruct-Q4_K_M".to_string();
        let descriptor = crate::mesh::ServedModelDescriptor {
            identity: crate::mesh::ServedModelIdentity {
                model_name: model_name.clone(),
                source_kind: crate::mesh::ModelSourceKind::LocalGguf,
                local_file_name: Some(format!("{model_name}.gguf")),
                ..Default::default()
            },
            capabilities_known: true,
            capabilities: crate::models::ModelCapabilities {
                multimodal: true,
                vision: crate::models::CapabilityLevel::Supported,
                ..Default::default()
            },
            topology: None,
            metadata: None,
        };

        let snapshot = collector.build_model_view(ModelViewInput {
            peers: vec![],
            catalog: vec![MeshCatalogEntry {
                model_name: model_name.clone(),
                descriptor: Some(descriptor),
            }],
            served_models: vec![model_name.clone()],
            active_demand: HashMap::new(),
            my_serving_models: vec![model_name.clone()],
            my_hosted_models: vec![model_name.clone()],
            local_inventory: LocalModelInventorySnapshot {
                model_names: HashSet::from([model_name.clone()]),
                size_by_name: HashMap::new(),
                metadata_by_name: HashMap::new(),
                display_name_by_name: HashMap::new(),
            },
            node_hostname: Some("node.local".into()),
            my_vram_gb: 24.0,
            model_name: model_name.clone(),
            model_size_bytes: 0,
            now_unix_secs: 1_700_000_000,
        });

        let payload = mesh_models(snapshot);
        assert_eq!(payload.len(), 1);
        assert_eq!(payload[0].name, model_name);
        assert!(payload[0].multimodal);
        assert_eq!(payload[0].multimodal_status, Some("supported"));
        assert!(payload[0].vision);
        assert_eq!(payload[0].vision_status, Some("supported"));
    }

    #[test]
    fn runtime_data_llama_items_preserve_slot_index_and_busy_state() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });

        producer.publish_llama_slots_snapshot(RuntimeLlamaSlotsSnapshot {
            status: RuntimeLlamaEndpointStatus::Ready,
            model: Some("Qwen3-8B".to_string()),
            instance_id: None,
            last_attempt_unix_ms: Some(1),
            last_success_unix_ms: Some(1),
            error: None,
            slots: vec![
                RuntimeLlamaSlotSnapshot {
                    id: Some(10),
                    is_processing: Some(false),
                    ..RuntimeLlamaSlotSnapshot::default()
                },
                RuntimeLlamaSlotSnapshot {
                    id: Some(20),
                    id_task: Some(42),
                    n_ctx: Some(8192),
                    is_processing: Some(true),
                    ..RuntimeLlamaSlotSnapshot::default()
                },
            ],
        });

        let snapshot = collector.runtime_llama_snapshot();
        assert_eq!(snapshot.items.slots_total, 2);
        assert_eq!(snapshot.items.slots_busy, 1);
        assert_eq!(snapshot.items.slots[0].index, 0);
        assert_eq!(snapshot.items.slots[0].id, Some(10));
        assert!(!snapshot.items.slots[0].is_processing);
        assert_eq!(snapshot.items.slots[1].index, 1);
        assert_eq!(snapshot.items.slots[1].id, Some(20));
        assert!(snapshot.items.slots[1].is_processing);
    }

    #[test]
    fn runtime_data_llama_slots_keep_per_model_snapshots() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });

        producer.publish_llama_slots_snapshot(RuntimeLlamaSlotsSnapshot {
            status: RuntimeLlamaEndpointStatus::Ready,
            model: Some("model-b".to_string()),
            instance_id: None,
            last_attempt_unix_ms: Some(1),
            last_success_unix_ms: Some(1),
            error: None,
            slots: vec![RuntimeLlamaSlotSnapshot {
                id: Some(0),
                is_processing: Some(false),
                ..RuntimeLlamaSlotSnapshot::default()
            }],
        });
        producer.publish_llama_slots_snapshot(RuntimeLlamaSlotsSnapshot {
            status: RuntimeLlamaEndpointStatus::Ready,
            model: Some("model-a".to_string()),
            instance_id: None,
            last_attempt_unix_ms: Some(2),
            last_success_unix_ms: Some(2),
            error: None,
            slots: vec![RuntimeLlamaSlotSnapshot {
                id: Some(0),
                is_processing: Some(true),
                ..RuntimeLlamaSlotSnapshot::default()
            }],
        });

        let by_model = collector.runtime_llama_snapshots_by_model();
        assert_eq!(by_model.len(), 2);
        assert_eq!(by_model["model-a"].items.slots_busy, 1);
        assert_eq!(by_model["model-b"].items.slots_busy, 0);
        assert_eq!(
            collector.runtime_llama_snapshot().slots.model.as_deref(),
            Some("model-a")
        );

        producer.publish_llama_slots_snapshot(RuntimeLlamaSlotsSnapshot {
            status: RuntimeLlamaEndpointStatus::Unavailable,
            model: Some("model-a".to_string()),
            instance_id: None,
            last_attempt_unix_ms: Some(3),
            last_success_unix_ms: None,
            error: None,
            slots: Vec::new(),
        });

        let by_model = collector.runtime_llama_snapshots_by_model();
        assert_eq!(
            by_model["model-a"].slots.status,
            RuntimeLlamaEndpointStatus::Unavailable
        );
        assert_eq!(
            by_model["model-b"].slots.status,
            RuntimeLlamaEndpointStatus::Ready
        );
        assert_eq!(
            collector.runtime_llama_snapshot().slots.model.as_deref(),
            Some("model-b")
        );
        assert_eq!(collector.runtime_llama_snapshot().items.slots_total, 1);
    }

    #[test]
    fn runtime_data_llama_slots_keep_per_instance_snapshots_for_same_model() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });

        producer.publish_llama_slots_snapshot(RuntimeLlamaSlotsSnapshot {
            status: RuntimeLlamaEndpointStatus::Ready,
            model: Some("model-a".to_string()),
            instance_id: Some("runtime-1".to_string()),
            last_attempt_unix_ms: Some(1),
            last_success_unix_ms: Some(1),
            error: None,
            slots: vec![RuntimeLlamaSlotSnapshot {
                id: Some(1),
                is_processing: Some(false),
                ..RuntimeLlamaSlotSnapshot::default()
            }],
        });
        producer.publish_llama_slots_snapshot(RuntimeLlamaSlotsSnapshot {
            status: RuntimeLlamaEndpointStatus::Ready,
            model: Some("model-a".to_string()),
            instance_id: Some("runtime-2".to_string()),
            last_attempt_unix_ms: Some(2),
            last_success_unix_ms: Some(2),
            error: None,
            slots: vec![RuntimeLlamaSlotSnapshot {
                id: Some(2),
                is_processing: Some(true),
                ..RuntimeLlamaSlotSnapshot::default()
            }],
        });

        let by_instance = collector.runtime_llama_snapshots_by_instance();
        assert_eq!(by_instance.len(), 2);
        assert_eq!(by_instance["runtime-1"].items.slots_busy, 0);
        assert_eq!(by_instance["runtime-2"].items.slots_busy, 1);

        producer.publish_llama_slots_snapshot(RuntimeLlamaSlotsSnapshot {
            status: RuntimeLlamaEndpointStatus::Unavailable,
            model: Some("model-a".to_string()),
            instance_id: Some("runtime-1".to_string()),
            last_attempt_unix_ms: Some(3),
            last_success_unix_ms: None,
            error: None,
            slots: Vec::new(),
        });

        let by_instance = collector.runtime_llama_snapshots_by_instance();
        assert_eq!(
            by_instance["runtime-1"].slots.status,
            RuntimeLlamaEndpointStatus::Unavailable
        );
        assert_eq!(
            by_instance["runtime-2"].slots.status,
            RuntimeLlamaEndpointStatus::Ready
        );
        assert_eq!(collector.runtime_llama_snapshot().items.slots_busy, 1);
        assert_eq!(
            collector
                .runtime_llama_snapshot()
                .slots
                .instance_id
                .as_deref(),
            Some("runtime-2")
        );
    }

    #[test]
    fn runtime_data_local_instance_snapshot_replaces_existing_scan_results() {
        let collector = RuntimeDataCollector::new();
        let producer = collector.producer(RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });

        let original = LocalInstanceSnapshot {
            pid: 100,
            api_port: Some(3131),
            version: Some("0.1.0".into()),
            started_at_unix: 1,
            runtime_dir: PathBuf::from("/tmp/runtime-a"),
            is_self: false,
        };
        let replacement = LocalInstanceSnapshot {
            pid: 200,
            api_port: Some(4141),
            version: Some("0.2.0".into()),
            started_at_unix: 2,
            runtime_dir: PathBuf::from("/tmp/runtime-b"),
            is_self: true,
        };

        assert!(
            crate::runtime::instance::publish_local_instance_scan_results(
                &producer,
                vec![original.clone()],
            )
        );
        assert_eq!(
            collector.local_instances_snapshot().instances,
            vec![original]
        );

        assert!(
            crate::runtime::instance::publish_local_instance_scan_results(
                &producer,
                vec![replacement.clone()],
            )
        );
        assert_eq!(
            collector.local_instances_snapshot().instances,
            vec![replacement]
        );
    }

    #[test]
    fn runtime_data_plugin_clear_removes_only_target_plugin_reports() {
        let collector = RuntimeDataCollector::new();
        let alpha = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: Some(PluginDataKey {
                plugin_name: "alpha".into(),
                data_key: "summary".into(),
            }),
            plugin_endpoint_key: None,
        });
        let alpha_endpoint = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: None,
            plugin_endpoint_key: Some(PluginEndpointKey {
                plugin_name: "alpha".into(),
                endpoint_id: "chat".into(),
            }),
        });
        let beta = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: Some(PluginDataKey {
                plugin_name: "beta".into(),
                data_key: "summary".into(),
            }),
            plugin_endpoint_key: None,
        });
        let beta_endpoint = collector.producer(RuntimeDataSource {
            scope: "plugin",
            plugin_data_key: None,
            plugin_endpoint_key: Some(PluginEndpointKey {
                plugin_name: "beta".into(),
                endpoint_id: "embed".into(),
            }),
        });

        alpha.publish_plugin_summary(PluginSummary {
            name: "alpha".into(),
            kind: "external".into(),
            enabled: true,
            status: "running".into(),
            pid: Some(1001),
            version: Some("1.0.0".into()),
            capabilities: Vec::new(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            web_ui: PluginWebUiState::default(),
            startup: None,
            error: None,
        });
        alpha.publish_plugin_payload("metrics", json!({"requests": 1}));
        alpha_endpoint.publish_plugin_endpoint(PluginEndpointSummary {
            plugin_name: "alpha".into(),
            plugin_status: "running".into(),
            endpoint_id: "chat".into(),
            state: "healthy".into(),
            available: true,
            kind: "mcp".into(),
            transport_kind: "http".into(),
            protocol: Some("http".into()),
            address: Some("http://127.0.0.1:9000/mcp".into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: true,
            managed_by_plugin: true,
            detail: None,
            models: Vec::new(),
        });
        beta.publish_plugin_summary(PluginSummary {
            name: "beta".into(),
            kind: "external".into(),
            enabled: true,
            status: "running".into(),
            pid: Some(1002),
            version: Some("2.0.0".into()),
            capabilities: Vec::new(),
            command: None,
            args: Vec::new(),
            tools: Vec::new(),
            manifest: None,
            web_ui: PluginWebUiState::default(),
            startup: None,
            error: None,
        });
        beta.publish_plugin_payload("metrics", json!({"requests": 7}));
        beta_endpoint.publish_plugin_endpoint(PluginEndpointSummary {
            plugin_name: "beta".into(),
            plugin_status: "running".into(),
            endpoint_id: "embed".into(),
            state: "healthy".into(),
            available: true,
            kind: "inference".into(),
            transport_kind: "tcp".into(),
            protocol: None,
            address: Some("127.0.0.1:9444".into()),
            args: Vec::new(),
            namespace: None,
            supports_streaming: false,
            managed_by_plugin: false,
            detail: None,
            models: vec!["beta-model".into()],
        });

        assert!(alpha.clear_plugin_reports("alpha"));

        let alpha_snapshot = collector.plugin_snapshot("alpha");
        assert!(alpha_snapshot.summary.is_none());
        assert!(alpha_snapshot.providers.is_empty());
        assert!(alpha_snapshot.payloads.is_empty());
        assert!(alpha_snapshot.endpoints.is_empty());
        assert!(
            collector
                .plugin_endpoint_snapshot("alpha", "chat")
                .is_none()
        );

        let beta_snapshot = collector.plugin_snapshot("beta");
        assert_eq!(
            beta_snapshot
                .summary
                .as_ref()
                .map(|summary| summary.name.as_str()),
            Some("beta")
        );
        assert_eq!(
            beta_snapshot.payloads.get("metrics"),
            Some(&json!({"requests": 7}))
        );
        assert_eq!(beta_snapshot.endpoints.len(), 1);
        assert_eq!(beta_snapshot.endpoints[0].endpoint_id, "embed");
        assert!(
            collector
                .plugin_endpoint_snapshot("beta", "embed")
                .is_some()
        );

        let all = collector.plugins_snapshot();
        assert_eq!(
            all.plugins
                .iter()
                .map(|plugin| plugin.name.as_str())
                .collect::<Vec<_>>(),
            vec!["beta"]
        );
        assert_eq!(
            all.endpoints
                .iter()
                .map(|endpoint| (endpoint.plugin_name.as_str(), endpoint.endpoint_id.as_str()))
                .collect::<Vec<_>>(),
            vec![("beta", "embed")]
        );
    }

    async fn start_local_http_server(response: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            if let Ok((mut conn, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = conn.read(&mut buf).await;
                let _ = conn.write_all(response.as_bytes()).await;
                let _ = conn.shutdown().await;
            }
        });
        port
    }

    async fn connected_proxy_stream() -> (TcpStream, tokio::task::JoinHandle<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).await.unwrap();
        let (server, _) = listener.accept().await.unwrap();
        let client_reader = tokio::spawn(async move {
            let mut client = client;
            let mut buf = Vec::new();
            client.read_to_end(&mut buf).await.unwrap();
            buf
        });
        (server, client_reader)
    }

    #[tokio::test]
    async fn runtime_data_routing_snapshot_reflects_proxy_attempts_and_inflight() {
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let collector = node.runtime_data_collector();
        let upstream_port = start_local_http_server(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 69\r\n\r\n{\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":7,\"total_tokens\":10}}",
        )
        .await;
        let (proxy_stream, client_reader) = connected_proxy_stream().await;

        let routed = transport::route_to_target(
            node.clone(),
            proxy_stream.into(),
            Some("glm"),
            election::InferenceTarget::Local(upstream_port),
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}",
            crate::network::openai::transport::RouteTargetContext {
                request_id: mesh_llm_events::logging::identifiers::RequestId::default(),
                response_adapter: ResponseAdapter::None,
                route_observer: crate::logging::OpenAiRouteObserver::default(),
            },
        )
        .await;

        assert!(routed.response_written());
        let response = String::from_utf8(client_reader.await.unwrap()).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"));

        let snapshot = collector.routing_snapshot();
        assert_eq!(snapshot.status.request_count, 1);
        assert_eq!(snapshot.status.successful_requests, 1);
        assert_eq!(snapshot.status.local_node.current_inflight_requests, 0);
        assert_eq!(snapshot.status.local_node.peak_inflight_requests, 1);
        assert_eq!(snapshot.status.local_node.local_attempt_count, 1);
        assert_eq!(snapshot.status.completion_tokens_observed, 7);
        assert_eq!(snapshot.status.pressure.fronted_request_count, 1);
        assert_eq!(snapshot.status.pressure.locally_served_request_count, 1);

        let model = snapshot
            .models
            .get("glm")
            .expect("glm model snapshot present");
        assert_eq!(model.request_count, 1);
        assert_eq!(model.successful_requests, 1);
        assert_eq!(model.completion_tokens_observed, 7);
        assert_eq!(model.targets.len(), 1);
        assert_eq!(model.targets[0].kind, "local");
        assert_eq!(model.targets[0].attempt_count, 1);
        assert_eq!(model.targets[0].success_count, 1);
    }

    #[tokio::test]
    async fn runtime_data_request_updates_stay_non_blocking() {
        let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let collector = node.runtime_data_collector();
        let mut subscription = collector.subscribe();

        let guard = node.begin_inflight_request();
        assert!(subscription.has_changed().expect("watch channel open"));
        let opened = *subscription.borrow_and_update();
        assert_eq!(opened.version.get(), 1);
        assert!(opened.dirty.contains(RuntimeDataDirty::ROUTING));
        assert_eq!(
            collector
                .routing_snapshot()
                .status
                .local_node
                .current_inflight_requests,
            1
        );

        node.record_inference_attempt(
            Some("glm"),
            &election::InferenceTarget::Local(9337),
            std::time::Duration::from_millis(3),
            std::time::Duration::from_millis(12),
            crate::network::metrics::AttemptOutcome::Success,
            Some(5),
        );
        assert!(subscription.has_changed().expect("watch channel open"));
        let attempted = *subscription.borrow_and_update();
        assert_eq!(attempted.version.get(), 2);
        assert!(attempted.dirty.contains(RuntimeDataDirty::ROUTING));

        node.record_routed_request(
            Some("glm"),
            1,
            crate::network::metrics::RequestOutcome::Success(
                crate::network::metrics::RequestService::Local,
            ),
        );
        assert!(subscription.has_changed().expect("watch channel open"));
        let requested = *subscription.borrow_and_update();
        assert_eq!(requested.version.get(), 3);
        assert!(requested.dirty.contains(RuntimeDataDirty::ROUTING));

        drop(guard);
        assert!(subscription.has_changed().expect("watch channel open"));
        let completed = *subscription.borrow_and_update();
        assert_eq!(completed.version.get(), 4);
        assert!(completed.dirty.contains(RuntimeDataDirty::ROUTING));

        let snapshot = collector.routing_snapshot();
        assert_eq!(snapshot.status.request_count, 1);
        assert_eq!(snapshot.status.successful_requests, 1);
        assert_eq!(snapshot.status.local_node.current_inflight_requests, 0);
        assert_eq!(snapshot.status.local_node.peak_inflight_requests, 1);
        assert_eq!(snapshot.status.local_node.local_attempt_count, 1);
        assert_eq!(snapshot.models["glm"].request_count, 1);
        assert_eq!(snapshot.models["glm"].targets[0].success_count, 1);
    }
}
