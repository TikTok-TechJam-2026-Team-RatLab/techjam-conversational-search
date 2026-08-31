from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.candidate_reranker import rerank_candidates
from src.data_parser import load_catalog
from src.intent_routing import IntentDecision, RoutingConfig
from starter.agent import Agent


PRODUCTS = [
    {
        "parent_asin": "PLAIN_SHIRT",
        "title": "Everyday blue shirt",
        "categories": ["Clothing", "Shirts"],
        "features": ["soft everyday fabric"],
        "details": {"Color": "Blue", "Material": "Polyester"},
        "price": 20,
        "store": "Northwind",
    },
    {
        "parent_asin": "TRAIL_SHIRT",
        "title": "Cotton trail shirt",
        "categories": ["Clothing", "Shirts"],
        "features": ["reinforced shoulder seams"],
        "details": {"Color": "Green", "Material": "Cotton"},
        "price": 49,
        "store": "Trailworks",
    },
    {
        "parent_asin": "RED_SHIRT",
        "title": "Red formal shirt",
        "categories": ["Clothing", "Shirts"],
        "features": ["button front"],
        "details": {"Color": "Red", "Material": "Cotton"},
        "price": 52,
        "store": "Northwind",
    },
    {
        "parent_asin": "HIKING_BOOT",
        "title": "Waterproof hiking boot",
        "categories": ["Clothing", "Shoes", "Hiking Boots"],
        "features": ["strong arch support"],
        "details": {"Color": "Brown", "Material": "Leather"},
        "price": 80,
        "store": "Trailworks",
    },
]


def write_catalog(path: Path, products: list[dict] = PRODUCTS) -> None:
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )


class RecordingRetriever:
    def __init__(self, results: list[tuple[str, float]]) -> None:
        self.results = results
        self.requested_top_k: int | None = None

    def search(
        self,
        query: str,
        top_k: int,
        *,
        max_price: float | None = None,
    ) -> list[tuple[str, float]]:
        del query, max_price
        self.requested_top_k = top_k
        return self.results[:top_k]


class RecordingRoutedRetriever(RecordingRetriever):
    def __init__(self, results: list[tuple[str, float]]) -> None:
        super().__init__(results)
        self.user_message: str | None = None
        self.active_constraints: dict[str, object] | None = None
        self.routing_config: RoutingConfig | None = None

    def search_intent_aware(
        self,
        query: str,
        *,
        user_message: str,
        active_constraints: dict[str, object],
        known_brands: object,
        known_categories: object,
        top_k: int,
        max_price: float | None = None,
        routing_config: RoutingConfig,
    ) -> tuple[list[tuple[str, float]], IntentDecision]:
        del query, known_brands, known_categories, max_price
        self.requested_top_k = top_k
        self.user_message = user_message
        self.active_constraints = active_constraints
        self.routing_config = routing_config
        return self.results[:top_k], IntentDecision(
            intent="buying",
            confidence=1.0,
            buying_score=3.0,
            browsing_score=0.0,
            signals=("test",),
        )


class CandidateRerankerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temp_directory.name) / "catalog.jsonl"
        write_catalog(self.catalog_path)
        self.catalog = load_catalog(self.catalog_path)
        self.retrieval_order = [
            ("PLAIN_SHIRT", -4.0),
            ("RED_SHIRT", -3.0),
            ("TRAIL_SHIRT", -2.0),
            ("HIKING_BOOT", -1.0),
        ]

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def rerank(
        self,
        *,
        active: dict[str, list[str]],
        negative: dict[str, list[str]] | None = None,
        top_k: int = 4,
    ) -> list[str]:
        results = rerank_candidates(
            self.retrieval_order,
            self.catalog.items_by_asin,
            active_constraints=active,
            negative_constraints=negative or {},
            constraint_updated_at={
                attribute: index
                for index, attribute in enumerate(active, start=1)
            },
            top_k=top_k,
        )
        return [parent_asin for parent_asin, _ in results]

    def test_exact_positive_constraint_promotes_lower_retrieval_candidate(self) -> None:
        ranked = self.rerank(active={"feature": ["reinforced shoulder seams"]})

        self.assertEqual(ranked[0], "TRAIL_SHIRT")

    def test_negative_constraint_demotes_matching_candidate(self) -> None:
        ranked = self.rerank(active={"category": ["shirts"]}, negative={"color": ["red"]})

        self.assertEqual(ranked[-1], "RED_SHIRT")
        self.assertLess(ranked.index("TRAIL_SHIRT"), ranked.index("RED_SHIRT"))

    def test_unstructured_negative_does_not_misread_product_care_text(self) -> None:
        candidates = [("RED_SHIRT", -4.0), ("PLAIN_SHIRT", -3.0)]

        results = rerank_candidates(
            candidates,
            self.catalog.items_by_asin,
            active_constraints={},
            negative_constraints={"other": ["button front"]},
            top_k=2,
        )

        self.assertEqual(results, candidates)

    def test_attribute_specific_category_matching_uses_category_evidence(self) -> None:
        ranked = self.rerank(active={"category": ["hiking boots"]})

        self.assertEqual(ranked[0], "HIKING_BOOT")

    def test_around_budget_prefers_closer_known_price(self) -> None:
        ranked = self.rerank(active={"budget": ["budget around 50"]})

        self.assertEqual(ranked[0], "TRAIL_SHIRT")

    def test_newer_constraint_breaks_an_otherwise_equal_match(self) -> None:
        candidates = [("PLAIN_SHIRT", 0.02), ("TRAIL_SHIRT", 0.01)]

        results = rerank_candidates(
            candidates,
            self.catalog.items_by_asin,
            active_constraints={"color": ["blue"], "material": ["cotton"]},
            negative_constraints={},
            constraint_updated_at={"color": 1, "material": 2},
            top_k=2,
        )

        self.assertEqual(results[0][0], "TRAIL_SHIRT")

    def test_no_constraints_preserves_original_order_and_top_k(self) -> None:
        results = rerank_candidates(
            self.retrieval_order,
            self.catalog.items_by_asin,
            active_constraints={},
            negative_constraints={},
            top_k=2,
        )

        self.assertEqual(results, self.retrieval_order[:2])

    def test_equal_constraint_scores_preserve_original_order(self) -> None:
        ranked = self.rerank(active={"category": ["shirts"]})

        self.assertEqual(ranked[:3], ["PLAIN_SHIRT", "RED_SHIRT", "TRAIL_SHIRT"])


class AgentRerankingIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temp_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": f"ITEM_{index:02d}",
                "title": f"Catalog shirt {index}",
                "categories": ["Clothing", "Shirts"],
                "features": [f"feature {index}"],
            }
            for index in range(60)
        ]
        write_catalog(self.catalog_path, products)
        self.results = [(product["parent_asin"], -float(index)) for index, product in enumerate(products)]

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_agent_requests_bounded_pool_then_returns_requested_top_k(self) -> None:
        agent = Agent(
            self.catalog_path,
            enable_intent_routing=False,
            enable_reranking=True,
            rerank_candidate_pool_size=50,
            enable_catalog_evidence=False,
            enable_broad_guidance=False,
            progressive_recommendations=False,
        )
        retriever = RecordingRetriever(self.results)
        agent.retriever = retriever
        agent.reset("session", {})

        response = agent.respond("session", "I'm looking for shirts.", turn=1, top_k=10)

        self.assertEqual(retriever.requested_top_k, 50)
        self.assertEqual(len(response["recommendations"]), 10)

    def test_retrieval_only_ablation_preserves_requested_depth_and_order(self) -> None:
        agent = Agent(
            self.catalog_path,
            enable_intent_routing=False,
            enable_reranking=False,
            enable_catalog_evidence=False,
            enable_broad_guidance=False,
            progressive_recommendations=False,
        )
        retriever = RecordingRetriever(self.results)
        agent.retriever = retriever
        agent.reset("session", {})

        response = agent.respond("session", "I'm looking for shirts.", turn=1, top_k=10)

        self.assertEqual(retriever.requested_top_k, 10)
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            [parent_asin for parent_asin, _ in self.results[:10]],
        )

    def test_candidate_pool_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            Agent(self.catalog_path, rerank_candidate_pool_size=0)

    def test_intent_routing_runs_before_constraint_reranking(self) -> None:
        agent = Agent(
            self.catalog_path,
            enable_intent_routing=True,
            enable_reranking=True,
            rerank_candidate_pool_size=50,
            enable_catalog_evidence=False,
            enable_broad_guidance=False,
            progressive_recommendations=False,
        )
        retriever = RecordingRoutedRetriever(self.results)
        agent.retriever = retriever
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "I'm looking for shirts. A key requirement is: feature 49.",
            turn=1,
            top_k=10,
        )

        self.assertEqual(retriever.requested_top_k, 50)
        self.assertIn("feature 49", retriever.user_message or "")
        self.assertIn("other", retriever.active_constraints or {})
        self.assertEqual(response["recommendations"][0]["parent_asin"], "ITEM_49")


if __name__ == "__main__":
    unittest.main()
