from __future__ import annotations

import importlib.util
import copy
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/generate-runtime-event-inventory.py"
INVENTORY = (
    ROOT / "crates/mesh-llm-runtime-event-contracts/inventory/runtime_events.toml"
)


def load_generator():
    spec = importlib.util.spec_from_file_location("runtime_event_inventory_generator", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RuntimeEventInventoryGeneratorTests(unittest.TestCase):
    def test_load_inventory_preserves_toml_quoted_values_comments_and_nesting(self) -> None:
        generator = load_generator()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.toml"
            path.write_text(
                """
title = "keep key = value [literal]" # this is a comment
single_quoted = 'keep # and ] inside'
nested = { value = "nested key = value [literal]" } # inline table

[section]
value = "bracket ] and # inside"

[[section.items]]
name = "first"

[[section.items]]
name = "second"
""",
                encoding="utf-8",
            )

            inventory = generator.load_inventory(path)

        self.assertEqual(inventory["title"], "keep key = value [literal]")
        self.assertEqual(inventory["single_quoted"], "keep # and ] inside")
        self.assertEqual(
            inventory["nested"],
            {"value": "nested key = value [literal]"},
        )
        self.assertEqual(
            inventory["section"],
            {
                "value": "bracket ] and # inside",
                "items": [{"name": "first"}, {"name": "second"}],
            },
        )

    def test_resolves_every_atomic_event_once(self) -> None:
        generator = load_generator()
        inventory = generator.load_inventory(INVENTORY)

        events = generator.resolved_events(inventory)

        self.assertEqual(len(events), 184)
        self.assertEqual(len({event["id"] for event in events}), 184)
        completed = next(event for event in events if event["id"] == "request_completed")
        self.assertEqual(completed["deliveryClass"], "terminal")
        self.assertIn("outcome", completed["projectedKeys"])
        self.assertNotIn("reason_code", completed["projectedKeys"])

    def test_resolves_event_specific_projection_keys(self) -> None:
        generator = load_generator()
        inventory = generator.load_inventory(INVENTORY)

        events = {event["id"]: event for event in generator.resolved_events(inventory)}

        self.assertNotEqual(
            events["request_received"]["projectedKeys"],
            events["request_failed"]["projectedKeys"],
        )
        self.assertIn("reason_code", events["request_failed"]["projectedKeys"])
        self.assertNotIn("reason_code", events["request_received"]["projectedKeys"])
        self.assertIn("progress", events["generation_progress"]["projectedKeys"])
        self.assertNotIn("progress", events["generation_completed"]["projectedKeys"])

    def test_rejects_projection_contract_mutations(self) -> None:
        generator = load_generator()
        original = generator.load_inventory(INVENTORY)
        mutations = (
            ("missing event-key entry", self._without_family_default(original)),
            ("unknown projected key", self._with_profile_key(original, "mystery")),
            ("forbidden projected key", self._with_profile_key(original, "prompt")),
            ("overbroad projection profile", self._with_overbroad_profile(original)),
            ("duplicate event projection override", self._with_duplicate_override(original)),
            ("unresolved event kind", self._with_unknown_override_event(original)),
            ("unresolved event family", self._without_event_family(original)),
        )
        for expected, inventory in mutations:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(generator.InventoryContractError, expected):
                    generator.resolved_events(inventory)

    @staticmethod
    def _without_family_default(inventory):
        mutated = copy.deepcopy(inventory)
        del mutated["families"][0]["default_projection_profile"]
        return mutated

    @staticmethod
    def _with_profile_key(inventory, key: str):
        mutated = copy.deepcopy(inventory)
        mutated["projection_profiles"][0]["keys"].append(key)
        return mutated

    @staticmethod
    def _with_overbroad_profile(inventory):
        mutated = copy.deepcopy(inventory)
        mutated["projection_profiles"][0]["keys"] = list(
            mutated["projected_event_keys"]
        )
        return mutated

    @staticmethod
    def _with_duplicate_override(inventory):
        mutated = copy.deepcopy(inventory)
        duplicated = mutated["event_projection_overrides"][0]["event_ids"][0]
        mutated["event_projection_overrides"][1]["event_ids"].append(duplicated)
        return mutated

    @staticmethod
    def _without_event_family(inventory):
        mutated = copy.deepcopy(inventory)
        mutated["families"] = mutated["families"][1:]
        return mutated

    @staticmethod
    def _with_unknown_override_event(inventory):
        mutated = copy.deepcopy(inventory)
        mutated["event_projection_overrides"][0]["event_ids"].append("unknown_event")
        return mutated

    def test_check_reports_stale_output(self) -> None:
        generator = load_generator()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stale.txt"
            output.write_text("stale\n", encoding="utf-8")

            result = generator.update_or_check({output: "fresh\n"}, True)

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
