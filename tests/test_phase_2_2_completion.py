from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from starter.agent import (
    HARD_CONSTRAINT_ATTRIBUTES,
    SLOT_DECAY_LAMBDA,
    SOFT_SLOT_MIN_WEIGHT,
    Agent,
    _active_retrieval_terms,
    _slot_weight,
)


class Phase22CompletionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        catalog_path = Path(self.temp_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "RED_SHOES",
                "title": "Red running shoes",
                "categories": ["Clothing, Shoes"],
                "details": {"Color": "Red", "Material": "Leather"},
            },
            {
                "parent_asin": "BLUE_SHOES",
                "title": "Blue running shoes",
                "categories": ["Clothing, Shoes"],
                "details": {"Color": "Blue", "Material": "Cotton"},
            },
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(catalog_path)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_soft_slot_uses_exponential_decay(self) -> None:
        self.assertNotIn("color", HARD_CONSTRAINT_ATTRIBUTES)
        self.assertAlmostEqual(_slot_weight("color", 1, 1), 1.0)
        self.assertAlmostEqual(
            _slot_weight("color", 1, 3),
            math.exp(-SLOT_DECAY_LAMBDA * 2),
        )
        self.assertLess(_slot_weight("color", 1, 4), SOFT_SLOT_MIN_WEIGHT)

    def test_old_soft_preference_stops_driving_retrieval(self) -> None:
        self.agent.reset("session-a", {})
        self.agent.respond("session-a", "I want blue shoes. Size 10.", 1, 2)
        state = self.agent._sessions["session-a"]

        fresh_terms = _active_retrieval_terms(state, 1)
        stale_terms = _active_retrieval_terms(state, 4)

        self.assertIn("blue", fresh_terms)
        self.assertNotIn("blue", stale_terms)
        self.assertIn("shoes", stale_terms)
        self.assertIn("10", stale_terms)
        self.assertIn("blue", state.active_constraints["color"])

    def test_hard_constraints_do_not_decay(self) -> None:
        self.agent.reset("session-a", {})
        self.agent.respond("session-a", "I want shoes. Size 10. Under $80.", 1, 2)
        state = self.agent._sessions["session-a"]

        late_terms = _active_retrieval_terms(state, 9)

        self.assertIn("shoes", late_terms)
        self.assertIn("10", late_terms)
        self.assertIn("80", late_terms)
        self.assertEqual(_slot_weight("size", 1, 9), 1.0)
        self.assertEqual(_slot_weight("budget", 1, 9), 1.0)

    def test_negative_color_is_enforced_by_retrieval(self) -> None:
        self.agent.reset("session-a", {})
        self.agent.respond("session-a", "I want shoes.", 1, 10)
        response = self.agent.respond("session-a", "Not red.", 2, 10)

        returned = [item["parent_asin"] for item in response["recommendations"]]
        self.assertIn("BLUE_SHOES", returned)
        self.assertNotIn("RED_SHOES", returned)
        self.assertEqual(
            self.agent._sessions["session-a"].negative_constraints["color"],
            ["red"],
        )

    def test_negative_material_is_enforced_by_retrieval(self) -> None:
        self.agent.reset("session-a", {})
        self.agent.respond("session-a", "I want shoes.", 1, 10)
        response = self.agent.respond("session-a", "No leather.", 2, 10)

        returned = [item["parent_asin"] for item in response["recommendations"]]
        self.assertIn("BLUE_SHOES", returned)
        self.assertNotIn("RED_SHOES", returned)


if __name__ == "__main__":
    unittest.main()
