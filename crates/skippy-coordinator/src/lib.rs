//! Coordinator lease and fencing policy for skippy split topologies.
//!
//! This crate intentionally owns only the pure coordination rules. Mesh
//! transport, protobuf conversion, node identity types, and stage runtime
//! process management stay in the host runtime.

pub mod topology;

use std::collections::HashMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CoordinatorClaim {
    pub model_id: String,
    pub package_ref: String,
    pub manifest_sha256: String,
    pub topology_id: String,
    pub run_id: String,
    pub coordinator_id: String,
    pub coordinator_term: u64,
    pub participant_set_hash: String,
    pub topology_hash: String,
    pub lease_until_unix_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadClaimRef {
    pub model_id: String,
    pub package_ref: String,
    pub manifest_sha256: String,
    pub topology_id: String,
    pub run_id: String,
    pub coordinator_id: Option<String>,
    pub coordinator_term: u64,
    pub participant_set_hash: String,
    pub topology_hash: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClaimDecision {
    Accepted {
        supersedes_term: Option<u64>,
        claim: CoordinatorClaim,
    },
    Rejected {
        current: Option<CoordinatorClaim>,
        reason: ClaimRejection,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ClaimRejection {
    #[error("coordinator claim requires model_id")]
    MissingModelId,
    #[error("coordinator claim requires package_ref")]
    MissingPackageRef,
    #[error("coordinator claim requires manifest_sha256")]
    MissingManifestSha256,
    #[error("coordinator claim requires topology_id and run_id")]
    MissingTopologyRun,
    #[error("coordinator claim requires coordinator_id")]
    MissingCoordinatorId,
    #[error("coordinator claim requires non-zero term")]
    MissingTerm,
    #[error("coordinator claim requires participant and topology hashes")]
    MissingHashes,
    #[error("coordinator claim lease is expired")]
    ExpiredLease,
    #[error("stale coordinator term {claim_term} < {current_term}")]
    StaleTerm { claim_term: u64, current_term: u64 },
    #[error("conflicting coordinator claim for existing term")]
    ConflictingSameTerm,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum LoadRejection {
    #[error("missing coordinator term")]
    MissingTerm,
    #[error("missing coordinator id")]
    MissingCoordinatorId,
    #[error("missing coordinator claim")]
    MissingClaim,
    #[error("coordinator term mismatch: load={load_term} claim={claim_term}")]
    TermMismatch { load_term: u64, claim_term: u64 },
    #[error("coordinator id mismatch")]
    CoordinatorMismatch,
    #[error("coordinator claim does not match topology/run")]
    TopologyRunMismatch,
    #[error("coordinator claim participant set hash does not match load")]
    ParticipantSetMismatch,
    #[error("coordinator claim topology hash does not match load")]
    TopologyHashMismatch,
    #[error("coordinator lease expired")]
    ExpiredLease,
}

#[derive(Clone, Debug, Default)]
pub struct ClaimFence {
    claims: HashMap<ClaimKey, CoordinatorClaim>,
}

impl ClaimFence {
    pub fn accept_claim(&mut self, claim: CoordinatorClaim, now_unix_ms: u64) -> ClaimDecision {
        if let Some(reason) = validate_claim_shape(&claim, now_unix_ms) {
            return ClaimDecision::Rejected {
                current: None,
                reason,
            };
        }

        let key = ClaimKey::from_claim(&claim);
        if let Some(current) = self.claims.get(&key) {
            if claim.coordinator_term < current.coordinator_term {
                return ClaimDecision::Rejected {
                    current: Some(current.clone()),
                    reason: ClaimRejection::StaleTerm {
                        claim_term: claim.coordinator_term,
                        current_term: current.coordinator_term,
                    },
                };
            }
            if claim.coordinator_term == current.coordinator_term
                && !same_claim_epoch(&claim, current)
            {
                return ClaimDecision::Rejected {
                    current: Some(current.clone()),
                    reason: ClaimRejection::ConflictingSameTerm,
                };
            }
        }

        let previous = self.claims.insert(key, claim.clone());
        ClaimDecision::Accepted {
            supersedes_term: previous.as_ref().and_then(|current| {
                (claim.coordinator_term > current.coordinator_term)
                    .then_some(current.coordinator_term)
            }),
            claim,
        }
    }

    pub fn validate_load(
        &self,
        load: &LoadClaimRef,
        now_unix_ms: u64,
    ) -> Result<(), LoadRejection> {
        if load.coordinator_term == 0 {
            return Err(LoadRejection::MissingTerm);
        }
        let Some(coordinator_id) = load.coordinator_id.as_deref() else {
            return Err(LoadRejection::MissingCoordinatorId);
        };
        let key = ClaimKey::from_load(load);
        let Some(claim) = self.claims.get(&key) else {
            return Err(LoadRejection::MissingClaim);
        };
        if claim.coordinator_term != load.coordinator_term {
            return Err(LoadRejection::TermMismatch {
                load_term: load.coordinator_term,
                claim_term: claim.coordinator_term,
            });
        }
        if claim.coordinator_id != coordinator_id {
            return Err(LoadRejection::CoordinatorMismatch);
        }
        if claim.topology_id != load.topology_id || claim.run_id != load.run_id {
            return Err(LoadRejection::TopologyRunMismatch);
        }
        if claim.participant_set_hash != load.participant_set_hash {
            return Err(LoadRejection::ParticipantSetMismatch);
        }
        if claim.topology_hash != load.topology_hash {
            return Err(LoadRejection::TopologyHashMismatch);
        }
        if claim.lease_until_unix_ms < now_unix_ms {
            return Err(LoadRejection::ExpiredLease);
        }
        Ok(())
    }

    pub fn current_claim_for(
        &self,
        model_id: &str,
        package_ref: &str,
        manifest_sha256: &str,
    ) -> Option<&CoordinatorClaim> {
        self.claims.get(&ClaimKey {
            model_id: model_id.to_string(),
            package_ref: package_ref.to_string(),
            manifest_sha256: manifest_sha256.to_string(),
        })
    }
}

pub fn quorum_requirement(planned_stage_count: usize) -> usize {
    planned_stage_count / 2 + 1
}

pub fn same_claim_epoch(left: &CoordinatorClaim, right: &CoordinatorClaim) -> bool {
    left.model_id == right.model_id
        && left.package_ref == right.package_ref
        && left.manifest_sha256 == right.manifest_sha256
        && left.topology_id == right.topology_id
        && left.run_id == right.run_id
        && left.coordinator_id == right.coordinator_id
        && left.participant_set_hash == right.participant_set_hash
        && left.topology_hash == right.topology_hash
}

fn validate_claim_shape(claim: &CoordinatorClaim, now_unix_ms: u64) -> Option<ClaimRejection> {
    if claim.model_id.is_empty() {
        return Some(ClaimRejection::MissingModelId);
    }
    if claim.package_ref.is_empty() {
        return Some(ClaimRejection::MissingPackageRef);
    }
    if claim.manifest_sha256.is_empty() {
        return Some(ClaimRejection::MissingManifestSha256);
    }
    if claim.topology_id.is_empty() || claim.run_id.is_empty() {
        return Some(ClaimRejection::MissingTopologyRun);
    }
    if claim.coordinator_id.is_empty() {
        return Some(ClaimRejection::MissingCoordinatorId);
    }
    if claim.coordinator_term == 0 {
        return Some(ClaimRejection::MissingTerm);
    }
    if claim.participant_set_hash.is_empty() || claim.topology_hash.is_empty() {
        return Some(ClaimRejection::MissingHashes);
    }
    if claim.lease_until_unix_ms <= now_unix_ms {
        return Some(ClaimRejection::ExpiredLease);
    }
    None
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ClaimKey {
    model_id: String,
    package_ref: String,
    manifest_sha256: String,
}

impl ClaimKey {
    fn from_claim(claim: &CoordinatorClaim) -> Self {
        Self {
            model_id: claim.model_id.clone(),
            package_ref: claim.package_ref.clone(),
            manifest_sha256: claim.manifest_sha256.clone(),
        }
    }

    fn from_load(load: &LoadClaimRef) -> Self {
        Self {
            model_id: load.model_id.clone(),
            package_ref: load.package_ref.clone(),
            manifest_sha256: load.manifest_sha256.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim(term: u64) -> CoordinatorClaim {
        CoordinatorClaim {
            model_id: "model".to_string(),
            package_ref: "hf://pkg".to_string(),
            manifest_sha256: "manifest".to_string(),
            topology_id: format!("topology-{term}"),
            run_id: format!("run-{term}"),
            coordinator_id: "node-a".to_string(),
            coordinator_term: term,
            participant_set_hash: "participants".to_string(),
            topology_hash: format!("topology-hash-{term}"),
            lease_until_unix_ms: 10_000,
        }
    }

    fn load(term: u64) -> LoadClaimRef {
        LoadClaimRef {
            model_id: "model".to_string(),
            package_ref: "hf://pkg".to_string(),
            manifest_sha256: "manifest".to_string(),
            topology_id: format!("topology-{term}"),
            run_id: format!("run-{term}"),
            coordinator_id: Some("node-a".to_string()),
            coordinator_term: term,
            participant_set_hash: "participants".to_string(),
            topology_hash: format!("topology-hash-{term}"),
        }
    }

    #[test]
    fn accepts_first_valid_claim() {
        let mut fence = ClaimFence::default();
        assert!(matches!(
            fence.accept_claim(claim(1), 1_000),
            ClaimDecision::Accepted {
                supersedes_term: None,
                ..
            }
        ));
    }

    #[test]
    fn rejects_stale_claim_after_newer_term() {
        let mut fence = ClaimFence::default();
        fence.accept_claim(claim(2), 1_000);
        assert!(matches!(
            fence.accept_claim(claim(1), 1_000),
            ClaimDecision::Rejected {
                reason: ClaimRejection::StaleTerm {
                    claim_term: 1,
                    current_term: 2
                },
                ..
            }
        ));
    }

    #[test]
    fn rejects_conflicting_claim_for_same_term() {
        let mut fence = ClaimFence::default();
        fence.accept_claim(claim(1), 1_000);
        let mut conflicting = claim(1);
        conflicting.coordinator_id = "node-b".to_string();
        assert!(matches!(
            fence.accept_claim(conflicting, 1_000),
            ClaimDecision::Rejected {
                reason: ClaimRejection::ConflictingSameTerm,
                ..
            }
        ));
    }

    #[test]
    fn newer_claim_supersedes_old_term() {
        let mut fence = ClaimFence::default();
        fence.accept_claim(claim(1), 1_000);
        assert!(matches!(
            fence.accept_claim(claim(2), 1_000),
            ClaimDecision::Accepted {
                supersedes_term: Some(1),
                ..
            }
        ));
    }

    #[test]
    fn validates_load_against_current_claim() {
        let mut fence = ClaimFence::default();
        fence.accept_claim(claim(3), 1_000);
        assert_eq!(fence.validate_load(&load(3), 1_000), Ok(()));
    }

    #[test]
    fn rejects_load_when_claim_hashes_do_not_match() {
        let mut fence = ClaimFence::default();
        fence.accept_claim(claim(3), 1_000);

        let mut participant_mismatch = load(3);
        participant_mismatch.participant_set_hash = "other-participants".to_string();
        assert_eq!(
            fence.validate_load(&participant_mismatch, 1_000),
            Err(LoadRejection::ParticipantSetMismatch)
        );

        let mut topology_mismatch = load(3);
        topology_mismatch.topology_hash = "other-topology".to_string();
        assert_eq!(
            fence.validate_load(&topology_mismatch, 1_000),
            Err(LoadRejection::TopologyHashMismatch)
        );
    }

    #[test]
    fn rejects_load_without_matching_claim() {
        let fence = ClaimFence::default();
        assert_eq!(
            fence.validate_load(&load(1), 1_000),
            Err(LoadRejection::MissingClaim)
        );
    }

    #[test]
    fn rejects_load_for_stale_term() {
        let mut fence = ClaimFence::default();
        fence.accept_claim(claim(2), 1_000);
        assert_eq!(
            fence.validate_load(&load(1), 1_000),
            Err(LoadRejection::TermMismatch {
                load_term: 1,
                claim_term: 2,
            })
        );
    }

    #[test]
    fn rejects_expired_claim_and_expired_load() {
        let mut fence = ClaimFence::default();
        assert!(matches!(
            fence.accept_claim(claim(1), 10_001),
            ClaimDecision::Rejected {
                reason: ClaimRejection::ExpiredLease,
                ..
            }
        ));

        fence.accept_claim(claim(2), 1_000);
        assert_eq!(
            fence.validate_load(&load(2), 10_001),
            Err(LoadRejection::ExpiredLease)
        );
    }

    #[test]
    fn quorum_is_majority_of_planned_stages() {
        assert_eq!(quorum_requirement(1), 1);
        assert_eq!(quorum_requirement(2), 2);
        assert_eq!(quorum_requirement(3), 2);
        assert_eq!(quorum_requirement(4), 3);
    }
}
