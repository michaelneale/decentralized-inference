//! Deadline, ordering, and shutdown behavior of the plugin dispatch workers.

use std::{
    sync::{Arc, atomic::Ordering, mpsc::sync_channel},
    thread,
    time::{Duration, Instant},
};

use mesh_native_serving_plugin_api as abi;
use skippy_server::frontend::{
    GenerationAbort, GenerationCommit, GenerationLifecycleIngress, GenerationLifecycleObservation,
    GenerationStart, LinearProposalDiscardReason, LinearProposalIngress, LinearProposalQuery,
    LinearProposalReceipt, LinearProposalSourceOutcome, OpaqueProposalDecisionId,
};

use super::{
    PluginCommand, PluginCommandQueue, PluginCommandQueueError, PluginDriver,
    finish_after_passive_fence,
};
use crate::{
    NativeLifecycleIngress, NativeProposalIngress,
    test_support::{
        CallbackGate, fake_active, fake_active_with_candidate_and_proposal_gate,
        fake_active_with_candidate_and_report_and_commit_gate, fake_active_with_events,
        fake_active_with_failing_proposal, fake_active_with_late_candidate,
        fake_active_with_observations, fake_active_with_options, fake_active_with_timing,
        fake_observations, proposal_query, wait_for_event,
    },
};

#[test]
fn blocking_plugin_cannot_extend_the_decode_deadline() {
    let active = fake_active(Duration::from_millis(250));
    let observations = fake_observations(&active);
    let driver = PluginDriver::spawn(active).unwrap();
    let ingress = NativeLifecycleIngress {
        driver: Arc::new(driver),
    };
    ingress
        .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();
    let driver = Arc::clone(&ingress.driver);
    wait_for_event(observations.events.as_ref(), "begin");
    let started = Instant::now();
    let result = driver
        .propose(LinearProposalQuery::new(
            1,
            2,
            16,
            16,
            0,
            8_192,
            started + Duration::from_millis(5),
        ))
        .unwrap();
    let elapsed = started.elapsed();

    assert!(result.proposal.unwrap().is_none());
    assert_eq!(
        result.telemetry.outcome,
        LinearProposalSourceOutcome::HostDeadlineExceeded
    );
    assert!(
        elapsed < Duration::from_millis(150),
        "decode waited {elapsed:?} for a blocking plugin"
    );
    drop(ingress);
    drop(driver);
    assert_eq!(observations.cancel_count.load(Ordering::SeqCst), 1);
}

#[test]
fn slow_commit_abstains_before_plugin_dispatch_and_later_positions_recover() {
    let (active, events, _) = fake_active_with_timing(
        Duration::ZERO,
        Duration::from_millis(40),
        Duration::ZERO,
        false,
    );
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    ingress
        .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();
    ingress
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![4].into_boxed_slice(),
            },
        ))
        .unwrap();
    wait_for_event(events.as_ref(), "commit");

    let started = Instant::now();
    let missed = driver
        .propose(proposal_query(started + Duration::from_millis(5)))
        .unwrap();
    assert!(missed.proposal.unwrap().is_none());
    assert_eq!(
        missed.telemetry.outcome,
        LinearProposalSourceOutcome::HostDeadlineExceeded
    );
    assert!(
        started.elapsed() < Duration::from_millis(30),
        "proposal wait exceeded its deadline: {:?}",
        started.elapsed()
    );

    let recovered = driver
        .propose(LinearProposalQuery::new(
            1,
            2,
            1,
            2,
            1,
            8,
            Instant::now() + Duration::from_millis(100),
        ))
        .unwrap();
    assert!(recovered.proposal.unwrap().is_none());
    assert_eq!(
        recovered.telemetry.outcome,
        LinearProposalSourceOutcome::Abstained
    );
    assert_eq!(*events.lock().unwrap(), ["begin", "commit", "proposal"]);
}

#[test]
fn slow_passive_discard_cannot_run_the_next_proposal_after_its_deadline() {
    let (active, events, _) = fake_active_with_timing(
        Duration::ZERO,
        Duration::ZERO,
        Duration::from_millis(100),
        false,
    );
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    driver
        .enqueue(PluginCommand::Discard(
            vec![1],
            LinearProposalDiscardReason::PositionMismatch,
        ))
        .unwrap();
    wait_for_event(events.as_ref(), "discard");

    let started = Instant::now();
    let response = driver
        .propose(proposal_query(started + Duration::from_millis(20)))
        .unwrap();
    assert!(response.proposal.unwrap().is_none());
    assert_eq!(
        response.telemetry.outcome,
        LinearProposalSourceOutcome::HostDeadlineExceeded
    );
    assert!(started.elapsed() < Duration::from_millis(60));
    assert_eq!(*events.lock().unwrap(), ["discard"]);
}

#[test]
fn worker_reports_pre_dispatch_deadlines_without_running_the_callback() {
    let (active, events, _) = fake_active_with_timing(
        Duration::ZERO,
        Duration::from_millis(40),
        Duration::ZERO,
        false,
    );
    let driver = PluginDriver::spawn(active).unwrap();
    driver
        .enqueue(PluginCommand::Begin(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();
    driver
        .enqueue(PluginCommand::Committed(GenerationCommit {
            request_id: 1,
            session_id: 2,
            generated_token_count: 1,
            token_ids: vec![4].into_boxed_slice(),
        }))
        .unwrap();
    wait_for_event(events.as_ref(), "commit");
    let (reply, response) = sync_channel(0);
    driver
        .queue
        .try_enqueue(PluginCommand::Proposal(
            proposal_query(Instant::now() + Duration::from_millis(5)),
            reply,
        ))
        .unwrap();

    let response = response.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(response.proposal.unwrap().is_none());
    assert_eq!(
        response.telemetry.outcome,
        LinearProposalSourceOutcome::DeadlineExceededBeforeDispatch
    );
    assert_eq!(*events.lock().unwrap(), ["begin", "commit"]);
}

#[test]
fn late_candidate_is_reported_and_not_forwarded_to_the_decode() {
    let active = fake_active_with_late_candidate(Duration::from_millis(20));
    let observations = fake_observations(&active);
    let driver = PluginDriver::spawn(active).unwrap();
    driver
        .enqueue(PluginCommand::Begin(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();
    wait_for_event(observations.events.as_ref(), "begin");
    let (reply, response) = sync_channel(0);
    driver
        .queue
        .try_enqueue(PluginCommand::Proposal(
            proposal_query(Instant::now() + Duration::from_millis(5)),
            reply,
        ))
        .unwrap();

    let response = response.recv_timeout(Duration::from_millis(100)).unwrap();
    assert!(response.proposal.unwrap().is_none());
    assert_eq!(
        response.telemetry.outcome,
        LinearProposalSourceOutcome::CandidateReturnedTooLate
    );
}

#[test]
fn dropped_proposal_reply_discards_an_on_time_candidate() {
    let proposal_gate = CallbackGate::new();
    let active = fake_active_with_candidate_and_proposal_gate(Arc::clone(&proposal_gate));
    let observations = fake_observations(&active);
    let driver = PluginDriver::spawn(active).unwrap();
    driver
        .enqueue(PluginCommand::Begin(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();
    wait_for_event(observations.events.as_ref(), "begin");

    let (reply, response) = sync_channel(0);
    driver
        .queue
        .try_enqueue(PluginCommand::Proposal(
            proposal_query(Instant::now() + Duration::from_secs(1)),
            reply,
        ))
        .unwrap();
    proposal_gate.wait_until_entered();
    assert!(response.recv_timeout(Duration::from_millis(5)).is_err());
    drop(response);
    proposal_gate.release();

    wait_for_event(observations.events.as_ref(), "discard");
    assert_eq!(
        observations.discard_reasons.lock().unwrap().as_slice(),
        &[abi::ProposalDiscardReason::DEADLINE_EXCEEDED]
    );
}

#[test]
fn stopped_worker_rejects_lifecycle_delivery() {
    let queue = PluginCommandQueue::new();
    queue.close();

    assert!(matches!(
        queue.try_enqueue(PluginCommand::Abort(GenerationAbort {
            request_id: 1,
            session_id: 2,
        })),
        Err(PluginCommandQueueError::Stopped)
    ));
}

#[test]
fn lifecycle_ingress_shares_plugin_queue_order_with_proposals() {
    let (active, events) = fake_active_with_events(Duration::ZERO);
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    ingress
        .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();
    ingress
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![4].into_boxed_slice(),
            },
        ))
        .unwrap();
    driver
        .propose(LinearProposalQuery::new(
            1,
            2,
            1,
            2,
            1,
            8,
            Instant::now() + Duration::from_millis(100),
        ))
        .unwrap();

    assert_eq!(*events.lock().unwrap(), ["begin", "commit", "proposal"]);
}

#[test]
fn proposal_requires_lifecycle_commits_before_lookup() {
    let (active, events) = fake_active_with_events(Duration::ZERO);
    active
        .begin(&GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        })
        .unwrap();
    active
        .committed(&GenerationCommit {
            request_id: 1,
            session_id: 2,
            generated_token_count: 2,
            token_ids: vec![4, 5].into_boxed_slice(),
        })
        .unwrap();
    let result = active
        .propose(LinearProposalQuery::new(
            1,
            2,
            1,
            3,
            2,
            8,
            Instant::now() + Duration::from_millis(100),
        ))
        .unwrap();

    assert!(result.is_none());
    assert_eq!(*events.lock().unwrap(), ["begin", "commit", "proposal"]);
}

#[test]
fn blocking_commit_cannot_extend_the_proposal_deadline() {
    let (active, events, _) = fake_active_with_timing(
        Duration::ZERO,
        Duration::from_millis(250),
        Duration::ZERO,
        false,
    );
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    ingress
        .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();
    ingress
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![4].into_boxed_slice(),
            },
        ))
        .unwrap();
    wait_for_event(events.as_ref(), "commit");

    let started = Instant::now();
    let result = driver
        .propose(LinearProposalQuery::new(
            1,
            2,
            1,
            2,
            1,
            8,
            started + Duration::from_millis(5),
        ))
        .unwrap();

    assert!(result.proposal.unwrap().is_none());
    assert_eq!(
        result.telemetry.outcome,
        LinearProposalSourceOutcome::HostDeadlineExceeded
    );
    assert!(
        started.elapsed() < Duration::from_millis(150),
        "proposal waited {:?} for a blocking commit",
        started.elapsed()
    );
}

#[test]
fn accepted_proposal_commits_report_and_later_lifecycle_commands_are_ordered_by_the_driver() {
    let commit_gate = CallbackGate::new();
    let report_gate = CallbackGate::new();
    let (active, events) = fake_active_with_candidate_and_report_and_commit_gate(
        Arc::clone(&report_gate),
        Arc::clone(&commit_gate),
    );
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let _release_gate = report_gate.release_on_drop();
    let _release_commit_gate = commit_gate.release_on_drop();
    let lifecycle = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    lifecycle
        .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();
    wait_for_event(events.as_ref(), "begin");

    let first = driver
        .propose(proposal_query(Instant::now() + Duration::from_millis(100)))
        .unwrap();
    assert!(first.proposal.unwrap().is_some());

    lifecycle
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![42].into_boxed_slice(),
            },
        ))
        .unwrap();
    commit_gate.wait_until_entered();
    lifecycle
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 2,
                token_ids: vec![43].into_boxed_slice(),
            },
        ))
        .unwrap();
    let proposals = NativeProposalIngress {
        driver: Arc::clone(&driver),
    };
    proposals
        .report(
            &LinearProposalReceipt::test_fixture_with_generated_token_count(
                OpaqueProposalDecisionId::new(vec![7]).unwrap(),
                2,
            ),
        )
        .unwrap();

    assert_eq!(*events.lock().unwrap(), ["begin", "proposal", "commit"]);
    commit_gate.release();

    report_gate.wait_until_entered();
    assert_eq!(
        *events.lock().unwrap(),
        ["begin", "proposal", "commit", "commit", "report"]
    );
    lifecycle
        .try_submit(GenerationLifecycleObservation::Committed(
            GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 3,
                token_ids: vec![44].into_boxed_slice(),
            },
        ))
        .unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        ["begin", "proposal", "commit", "commit", "report"]
    );
    let next_driver = Arc::clone(&driver);
    let second = thread::spawn(move || {
        next_driver
            .propose(LinearProposalQuery::new(
                1,
                2,
                1,
                4,
                3,
                8,
                Instant::now() + Duration::from_secs(1),
            ))
            .unwrap()
    });
    assert_eq!(
        *events.lock().unwrap(),
        ["begin", "proposal", "commit", "commit", "report"]
    );
    report_gate.release();
    let second = second.join().unwrap();
    assert!(second.proposal.unwrap().is_some());
    assert_eq!(
        *events.lock().unwrap(),
        [
            "begin",
            "proposal",
            "commit",
            "commit",
            "report",
            "report_done",
            "report_complete",
            "commit",
            "proposal"
        ]
    );
}

#[test]
fn finish_waits_for_prior_passive_disposition() {
    let (active, events, _) = fake_active_with_timing(
        Duration::ZERO,
        Duration::ZERO,
        Duration::from_millis(20),
        false,
    );
    let driver = PluginDriver::spawn(active).unwrap();
    driver
        .enqueue_terminal(PluginCommand::Discard(
            vec![1],
            LinearProposalDiscardReason::PositionMismatch,
        ))
        .unwrap();
    let passive_queue = Arc::clone(&driver.passive_queue);
    let finish_events = Arc::clone(&events);
    let finish = thread::spawn(move || {
        finish_after_passive_fence(&passive_queue, || {
            finish_events.lock().unwrap().push("finish");
            Ok(())
        })
    });

    wait_for_event(events.as_ref(), "discard");
    assert_eq!(*events.lock().unwrap(), ["discard"]);
    finish.join().unwrap().unwrap();
    assert_eq!(*events.lock().unwrap(), ["discard", "finish"]);
}

#[test]
fn lifecycle_callback_failure_is_observed_without_poisoning_the_driver() {
    let (active, _, _) = fake_active_with_options(Duration::ZERO, true);
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    ingress
        .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
            request_id: 7,
            session_id: 9,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();

    driver
        .propose(LinearProposalQuery::new(
            7,
            9,
            1,
            1,
            0,
            8,
            Instant::now() + Duration::from_millis(100),
        ))
        .unwrap();

    assert_eq!(driver.lifecycle_delivery_failures(), 1);
    assert!(driver.ensure_healthy().is_ok());
}

#[test]
fn generation_abort_bypasses_unhealthy_driver_gate() {
    let (active, _, abort_count) = fake_active_with_observations(Duration::ZERO);
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    *driver.fatal_error.lock().unwrap() = Some("report proposal failed".to_string());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };

    ingress
        .try_submit(GenerationLifecycleObservation::Aborted(GenerationAbort {
            request_id: 7,
            session_id: 9,
        }))
        .unwrap();

    drop(ingress);
    drop(driver);
    assert_eq!(abort_count.load(Ordering::SeqCst), 1);
}

#[test]
fn receipt_unavailable_recovers_and_cleans_an_active_generation() {
    let (active, observations, abort_count) = fake_active_with_observations(Duration::ZERO);
    let driver = Arc::new(PluginDriver::spawn(active).unwrap());
    let ingress = NativeLifecycleIngress {
        driver: Arc::clone(&driver),
    };
    ingress
        .try_submit(GenerationLifecycleObservation::Started(GenerationStart {
            request_id: 7,
            session_id: 9,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();
    wait_for_event(observations.as_ref(), "begin");

    let response = driver
        .propose(LinearProposalQuery::new(
            7,
            9,
            1,
            1,
            0,
            8,
            Instant::now() + Duration::from_millis(100),
        ))
        .unwrap();
    assert!(
        response.proposal.is_ok(),
        "an active generation should accept a proposal query before recovery"
    );

    ingress.receipt_unavailable(&GenerationAbort {
        request_id: 7,
        session_id: 9,
    });

    let deadline = Instant::now() + Duration::from_secs(1);
    while abort_count.load(Ordering::SeqCst) == 0 {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the abort callback"
        );
        thread::yield_now();
    }

    let response = driver
        .propose(LinearProposalQuery::new(
            7,
            9,
            1,
            1,
            0,
            8,
            Instant::now() + Duration::from_millis(100),
        ))
        .unwrap();
    assert!(response.proposal.is_err());
    assert_eq!(abort_count.load(Ordering::SeqCst), 1);
}

#[test]
fn a_full_queue_can_still_be_closed_and_drained() {
    // Closing takes no queue capacity, so a saturated queue can never strand
    // its worker the way a queued shutdown command could.
    let queue = PluginCommandQueue::new();
    let mut queued = 0_usize;
    while queue
        .try_enqueue(PluginCommand::Abort(GenerationAbort {
            request_id: 1,
            session_id: 2,
        }))
        .is_ok()
    {
        queued += 1;
        assert!(queued < 10_000, "queue never reported full");
    }

    queue.close();
    for _ in 0..queued {
        assert!(queue.next().is_some(), "close must not discard queued work");
    }
    assert!(
        queue.next().is_none(),
        "a drained, closed queue must stop its worker"
    );
}

#[test]
fn dropping_the_driver_drains_its_backlog_and_shuts_the_plugin_down() {
    let (active, events, _) = fake_active_with_timing(
        Duration::ZERO,
        Duration::from_millis(2),
        Duration::ZERO,
        false,
    );
    let observations = fake_observations(&active);
    let driver = PluginDriver::spawn(active).unwrap();
    driver
        .enqueue(PluginCommand::Begin(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();
    for generated_token_count in 1..=20 {
        driver
            .enqueue(PluginCommand::Committed(GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count,
                token_ids: vec![4].into_boxed_slice(),
            }))
            .unwrap();
    }

    drop(driver);
    assert_eq!(
        events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| **event == "commit")
            .count(),
        20,
        "backlog must be delivered"
    );
    assert_eq!(observations.shutdown_count.load(Ordering::SeqCst), 1);
}

#[test]
fn a_full_passive_queue_still_accepts_the_terminal_discard() {
    // Reserved headroom keeps a deadline discard deliverable, so the plugin
    // always learns the fate of a decision the host withheld.
    let queue = PluginCommandQueue::new();
    while queue
        .try_enqueue(PluginCommand::Discard(
            vec![1],
            LinearProposalDiscardReason::PositionMismatch,
        ))
        .is_ok()
    {}

    assert!(matches!(
        queue.try_enqueue(PluginCommand::Discard(
            vec![2],
            LinearProposalDiscardReason::PositionMismatch,
        )),
        Err(PluginCommandQueueError::Full)
    ));
    queue
        .try_enqueue_terminal(PluginCommand::Discard(
            vec![3],
            LinearProposalDiscardReason::DeadlineExceeded,
        ))
        .expect("terminal discard must use the reserved headroom");
}

#[test]
fn a_full_terminal_queue_still_accepts_the_passive_fence() {
    let queue = PluginCommandQueue::new();
    while queue
        .try_enqueue(PluginCommand::Discard(
            vec![1],
            LinearProposalDiscardReason::PositionMismatch,
        ))
        .is_ok()
    {}
    while queue
        .try_enqueue_terminal(PluginCommand::Discard(
            vec![2],
            LinearProposalDiscardReason::DeadlineExceeded,
        ))
        .is_ok()
    {}

    let (ack, _reply) = sync_channel(1);
    queue
        .try_enqueue_terminal(PluginCommand::Fence(ack))
        .expect("the fence must retain one exclusive terminal slot");
}

#[test]
fn a_late_candidate_delivers_its_discard_to_the_plugin() {
    // The host withholds a late candidate, so `discard` is the only way the
    // plugin can learn that decision's fate.
    let active = fake_active_with_late_candidate(Duration::from_millis(20));
    let observations = fake_observations(&active);
    let driver = PluginDriver::spawn(active).unwrap();
    driver
        .enqueue(PluginCommand::Begin(GenerationStart {
            request_id: 1,
            session_id: 2,
            agent_session_id: None,
            prompt_token_ids: Arc::from([3]),
            frontend_request_id: None,
        }))
        .unwrap();
    wait_for_event(observations.events.as_ref(), "begin");
    let (reply, response) = sync_channel(1);
    driver
        .queue
        .try_enqueue(PluginCommand::Proposal(
            proposal_query(Instant::now() + Duration::from_millis(5)),
            reply,
        ))
        .unwrap();

    let response = response.recv_timeout(Duration::from_millis(500)).unwrap();
    assert!(response.proposal.unwrap().is_none());
    assert_eq!(
        response.telemetry.outcome,
        LinearProposalSourceOutcome::CandidateReturnedTooLate
    );
    wait_for_event(observations.events.as_ref(), "discard");
}

#[test]
fn a_failing_proposal_callback_reports_source_error_instead_of_abstaining() {
    // Fail open for decode, but keep the plugin's failure distinguishable
    // from a deliberate abstention.
    let driver = PluginDriver::spawn(fake_active_with_failing_proposal()).unwrap();
    let response = driver
        .propose(proposal_query(Instant::now() + Duration::from_millis(200)))
        .unwrap();

    assert!(response.proposal.is_err());
    assert_eq!(
        response.telemetry.outcome,
        LinearProposalSourceOutcome::SourceError
    );
}
