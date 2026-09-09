import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).with_name("agentic-replay.py")
SPEC = importlib.util.spec_from_file_location("agentic_replay", SCRIPT)
AGENTIC_REPLAY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = AGENTIC_REPLAY
SPEC.loader.exec_module(AGENTIC_REPLAY)


def activity(**changes):
    snapshot = {
        "fills": 0,
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "bytes_read": 0,
        "bytes_written": 0,
        "evictions": 0,
        "corrupt_entries": 0,
    }
    snapshot.update(changes)
    return snapshot


def request(request_id="s:0", ttft=2.0):
    return {
        "request_id": request_id,
        "session_id": "s",
        "source_dataset": "buzz",
        "assistant_turn": 0,
        "prompt_tokens": 19_000,
        "ttft_seconds": ttft,
        "content_sha256": "same",
    }


def passing_run():
    baseline = request()
    high_load = {
        "failed_requests": 0,
        "decode_inter_token_p99_seconds": 0.1,
        "content_sha256_by_request": {"s:0": "same"},
    }
    return {
        "phases": {
            "disk_off_cold": {"requests": [baseline]},
            "disk_on_empty": {
                "requests": [baseline],
                "activity_delta": activity(writes=1, bytes_written=100),
            },
            "multi_turn_growth": {
                "requests": [baseline],
                "activity_delta": activity(writes=1),
            },
            "same_process_l1": {
                "requests": [baseline],
                "activity_delta": activity(),
            },
            "restart_l3": {
                "requests": [request("s:restart", 0.5)],
                "activity_deltas": [activity(fills=1, bytes_read=100)],
            },
            "concurrent_fill": {
                "requests": [baseline],
                "activity_delta": activity(fills=1, bytes_read=100),
            },
            "concurrent_record": {
                "requests": [baseline],
                "activity_delta": activity(writes=1, bytes_written=100),
            },
            "low_space": {
                "requests": [baseline],
                "status_after": {"effective": {"state": "read_only_low_space"}},
                "activity_delta": activity(),
            },
            "lifecycle_under_traffic": {
                "requests": [baseline],
                "prune": {},
                "clear": {},
                "final_clear": {"status": {"usage": {"manifests": 0}}},
            },
            "high_load_off_c1": {
                "requests": [baseline],
                "summary": high_load,
            },
            "high_load_on_c1": {
                "requests": [baseline],
                "summary": dict(high_load, decode_inter_token_p99_seconds=0.104),
            },
        }
    }


ARGS = SimpleNamespace(
    prompt_token_range="18000:24000",
    max_l3_ttft_ratio=0.5,
    identical_repeats=100,
    max_payload_write_amplification=1.2,
    require_source_dataset=["buzz"],
    concurrency=[1],
    max_decode_p99_regression_pct=5.0,
)


class DiskL3LifecycleGateTests(unittest.TestCase):
    def test_complete_evidence_passes(self):
        gates = AGENTIC_REPLAY.evaluate_l3_lifecycle_gates(passing_run(), ARGS)
        self.assertTrue(gates["passed"], gates)

    def test_duplicate_physical_fill_fails_closed(self):
        run = copy.deepcopy(passing_run())
        run["phases"]["concurrent_fill"]["activity_delta"]["fills"] = 2
        gates = AGENTIC_REPLAY.evaluate_l3_lifecycle_gates(run, ARGS)
        self.assertFalse(gates["passed"])
        check = next(
            item for item in gates["checks"] if item["name"] == "single_physical_fill"
        )
        self.assertFalse(check["passed"])


if __name__ == "__main__":
    unittest.main()
