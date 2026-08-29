from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent


class SessionStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        catalog_path = Path(self.temp_directory.name) / "catalog.jsonl"
        products = [
            {"parent_asin": "BLUE_SHOES", "title": "Blue running shoes"},
            {"parent_asin": "RED_SHOES", "title": "Red running shoes"},
            {"parent_asin": "BLUE_SHIRT", "title": "Blue cotton shirt"},
        ]
        catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(catalog_path)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_messages_are_recorded_in_order(self) -> None:
        self.agent.reset("session-a", {"summary": "test user"})

        self.agent.respond("session-a", "I want shoes", 1, 10)
        self.agent.respond("session-a", "Preferably blue", 2, 10)

        self.assertEqual(
            self.agent._sessions["session-a"].messages,
            ["I want shoes", "Preferably blue"],
        )

    def test_sessions_are_isolated(self) -> None:
        self.agent.reset("session-a", {"summary": "first user"})
        self.agent.reset("session-b", {"summary": "second user"})

        self.agent.respond("session-a", "I want shoes", 1, 10)

        self.assertEqual(
            self.agent._sessions["session-a"].messages,
            ["I want shoes"],
        )
        self.assertEqual(
            self.agent._sessions["session-b"].messages,
            [],
        )
        self.assertEqual(self.agent._sessions["session-b"].active_constraints, {})

    def test_reset_clears_existing_state(self) -> None:
        self.agent.reset("session-a", {"summary": "old profile"})
        self.agent.respond("session-a", "Old message", 1, 10)

        self.agent.reset("session-a", {"summary": "new profile"})

        self.assertEqual(
            self.agent._sessions["session-a"].user_profile,
            {"summary": "new profile"},
        )
        self.assertEqual(
            self.agent._sessions["session-a"].messages,
            [],
        )
        self.assertEqual(self.agent._sessions["session-a"].active_constraints, {})

    def test_constraints_accumulate_and_affect_retrieval(self) -> None:
        self.agent.reset("session-a", {})

        self.agent.respond("session-a", "I'm looking for shoes.", 1, 3)
        response = self.agent.respond("session-a", "Preferably blue.", 2, 3)

        self.assertEqual(
            self.agent._sessions["session-a"].active_constraints,
            {"category": ["I'm looking for shoes"], "color": ["Preferably blue"]},
        )
        self.assertEqual(response["recommendations"][0]["parent_asin"], "BLUE_SHOES")

    def test_intent_override_replaces_relevant_attribute_only(self) -> None:
        self.agent.reset("session-a", {})
        self.agent.respond("session-a", "I'm looking for shoes. Blue. Size 10.", 1, 3)
        self.agent.respond("session-a", "Actually, replace blue with red.", 2, 3)

        self.assertEqual(
            self.agent._sessions["session-a"].active_constraints,
            {
                "category": ["I'm looking for shoes"],
                "color": ["red"],
                "size": ["Size 10"],
            },
        )
        context = self.agent._sessions["session-a"].retrieval_context().lower()
        self.assertIn("shoes", context)
        self.assertIn("size 10", context)
        self.assertIn("red", context)
        self.assertNotIn("blue", context)

    def test_browsing_filler_is_removed_but_category_is_retained(self) -> None:
        self.agent.reset("session-a", {})

        self.agent.respond(
            "session-a",
            "I'm looking for dresses, but I'm still exploring.",
            1,
            3,
        )

        self.assertEqual(
            self.agent._sessions["session-a"].active_constraints,
            {"category": ["I'm looking for dresses"]},
        )
        context = self.agent._sessions["session-a"].retrieval_context()
        self.assertIn("dresses", context)
        self.assertNotIn("exploring", context)

    def test_respond_before_reset_raises_error(self) -> None:
        with self.assertRaises(RuntimeError):
            self.agent.respond("unknown-session", "Hello", 1, 10)


if __name__ == "__main__":
    unittest.main()
