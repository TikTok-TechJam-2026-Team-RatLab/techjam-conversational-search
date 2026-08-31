from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data_parser import CatalogItem
from src.proactive_guidance import (
    ALLOWED_ASK_ATTRIBUTES,
    candidate_attribute_values,
    choose_clarification,
)
from starter.agent import Agent


def item(
    parent_asin: str,
    *,
    color: str = "",
    material: str = "",
    feature: str = "same feature",
) -> CatalogItem:
    details = {}
    if color:
        details["Color"] = color
    if material:
        details["Material"] = material
    return CatalogItem(
        parent_asin=parent_asin,
        title=f"{color} {material} shirt".strip(),
        categories=["Clothing", "Shirts"],
        features=[feature] if feature else [],
        details=details,
        description=[],
        price=None,
        average_rating=0.0,
        rating_number=0,
        store="Same Store",
        dense_text="",
    )


class ExpectedInformationGainTest(unittest.TestCase):
    def test_highest_entropy_supported_attribute_is_selected(self) -> None:
        candidates = [
            item("A", color="Red", material="Cotton"),
            item("B", color="Blue", material="Cotton"),
            item("C", color="Green", material="Cotton"),
        ]

        decision = choose_clarification(candidates)

        self.assertEqual(decision.ask_attribute, "color")
        self.assertGreater(decision.information_gain, 0.0)
        self.assertIn("color", decision.message.lower())

    def test_active_or_previously_asked_attributes_are_not_repeated(self) -> None:
        candidates = [
            item("A", color="Red", material="Cotton", feature="soft"),
            item("B", color="Blue", material="Leather", feature="soft"),
        ]

        decision = choose_clarification(
            candidates,
            unavailable_attributes={"color", "material"},
        )

        self.assertNotIn(decision.ask_attribute, {"color", "material"})
        self.assertIn(decision.ask_attribute, ALLOWED_ASK_ATTRIBUTES)

    def test_unstructured_candidates_fall_back_to_other_once(self) -> None:
        candidates = [
            item("A", feature=""),
            item("B", feature=""),
        ]

        first = choose_clarification(candidates)
        repeated = choose_clarification(candidates, unavailable_attributes={"other"})

        self.assertEqual(first.ask_attribute, "other")
        self.assertIsNone(repeated.ask_attribute)

    def test_single_candidate_does_not_trigger_over_generality(self) -> None:
        decision = choose_clarification([item("A", color="Red")])

        self.assertIsNone(decision.ask_attribute)

    def test_catalog_evidence_maps_to_allowed_attributes(self) -> None:
        product = item(
            "A",
            color="Blue",
            material="Cotton",
            feature="strong arch support for running",
        )

        values = candidate_attribute_values(product)

        self.assertIn("blue", values["color"])
        self.assertIn("cotton", values["material"])
        self.assertTrue(values["use_case"])
        self.assertTrue(set(values).issubset(ALLOWED_ASK_ATTRIBUTES))


class AgentGuidanceIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temp_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "RED_SHIRT",
                "title": "Red shirt",
                "categories": ["Clothing", "Shirts"],
                "features": ["soft fabric"],
                "details": {"Color": "Red", "Material": "Cotton"},
            },
            {
                "parent_asin": "BLUE_SHIRT",
                "title": "Blue shirt",
                "categories": ["Clothing", "Shirts"],
                "features": ["soft fabric"],
                "details": {"Color": "Blue", "Material": "Cotton"},
            },
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(self.catalog_path)
        self.agent.reset("session", {})

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_agent_preserves_recommendations_and_asks_valid_question(self) -> None:
        response = self.agent.respond("session", "I want a shirt.", 1, 10)

        self.assertEqual(
            {item["parent_asin"] for item in response["recommendations"]},
            {"RED_SHIRT", "BLUE_SHIRT"},
        )
        self.assertEqual(response["ask_attribute"], "color")
        self.assertIn(response["ask_attribute"], ALLOWED_ASK_ATTRIBUTES)
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_declined_attribute_is_recorded_and_not_repeated(self) -> None:
        first = self.agent.respond("session", "I want a shirt.", 1, 10)
        second = self.agent.respond(
            "session",
            "I don't have an additional preference for color.",
            2,
            10,
        )

        state = self.agent._sessions["session"]
        self.assertEqual(first["ask_attribute"], "color")
        self.assertIn("color", state.declined_attributes)
        self.assertNotEqual(second["ask_attribute"], "color")

    def test_boundary_reply_marks_pending_question_as_declined(self) -> None:
        first = self.agent.respond("session", "I want a shirt.", 1, 10)
        self.agent.respond(
            "session",
            f"I don't have a preference for {first['ask_attribute']}; please use your judgment.",
            2,
            10,
        )

        state = self.agent._sessions["session"]
        self.assertIn(first["ask_attribute"], state.declined_attributes)
        self.assertIsNotNone(state.pending_ask_attribute)
        self.assertNotEqual(state.pending_ask_attribute, first["ask_attribute"])

    def test_reset_clears_question_history(self) -> None:
        self.agent.respond("session", "I want a shirt.", 1, 10)

        self.agent.reset("session", {})

        state = self.agent._sessions["session"]
        self.assertEqual(state.asked_attributes, set())
        self.assertEqual(state.declined_attributes, set())
        self.assertIsNone(state.pending_ask_attribute)


if __name__ == "__main__":
    unittest.main()
