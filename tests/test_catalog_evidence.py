from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.catalog_evidence import CatalogEvidenceIndex, extract_message_facts
from src.data_parser import load_catalog
from starter.agent import Agent


class CatalogEvidenceIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temp_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "POPULAR",
                "title": "Popular trail shirt",
                "categories": ["Clothing", "Shirts", "Trail Shirts"],
                "features": ["soft everyday fabric"],
                "rating_number": 1_000,
                "average_rating": 4.5,
            },
            {
                "parent_asin": "TARGET",
                "title": "Technical trail shirt",
                "categories": ["Clothing", "Shirts", "Trail Shirts"],
                "features": ["reinforced shoulder seams"],
                "rating_number": 10,
                "average_rating": 4.8,
            },
            {
                "parent_asin": "TIE_WINNER",
                "title": "Another trail shirt",
                "categories": ["Clothing", "Shirts", "Trail Shirts"],
                "features": ["soft everyday fabric"],
                "rating_number": 2_000,
                "average_rating": 4.4,
            },
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.catalog = load_catalog(self.catalog_path)
        self.index = CatalogEvidenceIndex(self.catalog)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_customer_fact_boundaries_are_preserved(self) -> None:
        facts = extract_message_facts(
            "For that, what matters is: 100% Cotton; Button closure."
        )

        self.assertEqual(facts, ("100 cotton", "button closure"))

    def test_exact_fact_recovers_product_outside_base_pool(self) -> None:
        ranking = self.index.rank(
            [("POPULAR", -1.0)],
            messages=["For that, what matters is: reinforced shoulder seams."],
            active_constraints={},
            negative_constraints={},
            limit=3,
        )

        self.assertEqual(ranking.results[0][0], "TARGET")
        self.assertIn("reinforced shoulder seams", ranking.matched_phrases)

    def test_popularity_breaks_an_exact_evidence_tie(self) -> None:
        ranking = self.index.rank(
            [],
            messages=["I'm looking for Trail Shirts."],
            active_constraints={},
            negative_constraints={},
            limit=3,
        )

        self.assertEqual(ranking.results[0][0], "TIE_WINNER")

    def test_raw_catalog_fact_prevents_noisy_slot_from_overriding_it(self) -> None:
        ranking = self.index.rank(
            [],
            messages=["For that, what matters is: reinforced shoulder seams."],
            active_constraints={"other": ["soft everyday fabric"]},
            negative_constraints={},
            limit=3,
        )

        self.assertEqual(ranking.results[0][0], "TARGET")

    def test_named_override_removes_stale_exact_fact(self) -> None:
        ranking = self.index.rank(
            [],
            messages=[
                "For that, what matters is: soft everyday fabric.",
                "Actually, replace soft everyday fabric with reinforced shoulder seams.",
            ],
            active_constraints={},
            negative_constraints={},
            limit=3,
        )

        self.assertEqual(ranking.results[0][0], "TARGET")
        self.assertNotIn("soft everyday fabric", ranking.matched_phrases)

    def test_no_exact_fact_preserves_base_retrieval_order(self) -> None:
        base = [("TARGET", -1.0), ("TIE_WINNER", -2.0), ("POPULAR", -3.0)]

        ranking = self.index.rank(
            base,
            messages=["I would like something unexpectedly whimsical."],
            active_constraints={},
            negative_constraints={},
            limit=3,
        )

        self.assertEqual(list(ranking.results), base)
        self.assertFalse(ranking.has_catalog_evidence)


class ProgressiveRecommendationTest(unittest.TestCase):
    def test_agent_asks_broadly_then_expands_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            products = [
                {
                    "parent_asin": f"ITEM_{index:02d}",
                    "title": f"Catalog shirt {index}",
                    "categories": ["Clothing", "Shirts"],
                    "rating_number": 100 - index,
                }
                for index in range(12)
            ]
            catalog_path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            agent = Agent(
                catalog_path,
                embeddings_path=Path(directory) / "missing.npy",
                manifest_path=Path(directory) / "missing.json",
            )
            agent.reset("session", {})

            first = agent.respond("session", "I'm looking for Shirts.", 1, 10)
            agent.respond(
                "session",
                "I don't have an additional preference for other.",
                2,
                10,
            )
            agent.respond(
                "session",
                "I don't have an additional preference for other.",
                3,
                10,
            )
            fourth = agent.respond(
                "session",
                "I don't have an additional preference for other.",
                4,
                10,
            )

        self.assertEqual(first["ask_attribute"], "other")
        self.assertEqual(len(first["recommendations"]), 1)
        self.assertEqual(len(fourth["recommendations"]), 10)


if __name__ == "__main__":
    unittest.main()
