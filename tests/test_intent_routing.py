from __future__ import annotations

import unittest

from src.intent_routing import (
    RankedResult,
    RoutingConfig,
    classify_intent,
    distribution_based_score_fusion,
    reciprocal_rank_fusion,
    route_and_fuse,
)


def result(parent_asin: str, score: float, rank: int) -> RankedResult:
    return RankedResult(parent_asin=parent_asin, score=score, rank=rank)


class IntentRouterTest(unittest.TestCase):
    def test_hard_constraints_route_to_buying(self) -> None:
        decision = classify_intent("Find me running shoes in size 10 under $100")

        self.assertEqual(decision.intent, "buying")
        self.assertIn("message:budget", decision.signals)
        self.assertIn("message:size", decision.signals)

    def test_generic_use_case_routes_to_browsing(self) -> None:
        decision = classify_intent("Any ideas for a summer wedding gift?")

        self.assertEqual(decision.intent, "browsing")
        self.assertIn("message:exploration_phrase", decision.signals)
        self.assertIn("message:generic_use_case", decision.signals)

    def test_explicit_exploration_overrides_generic_shopping_language(self) -> None:
        decision = classify_intent(
            "I'm looking for shoes, but I'm still exploring.",
            {"category": ["shoes"]},
            known_categories={"shoes"},
        )

        self.assertEqual(decision.intent, "browsing")
        self.assertIn("message:explicit_browsing", decision.signals)

    def test_explicit_override_to_concrete_requirement_routes_to_buying(self) -> None:
        decision = classify_intent(
            "Actually, ignore my earlier preference. What I need is: leather.",
            {"category": ["boots"], "material": ["leather"]},
        )

        self.assertEqual(decision.intent, "buying")
        self.assertIn("message:key_requirement", decision.signals)

    def test_accumulated_state_keeps_short_follow_up_in_buying(self) -> None:
        decision = classify_intent(
            "Blue, please",
            {"category": ["running shoes"], "size": ["10"], "budget": ["under 100"]},
        )

        self.assertEqual(decision.intent, "buying")
        self.assertIn("state:hard_constraints", decision.signals)

    def test_known_brand_is_a_buying_signal(self) -> None:
        decision = classify_intent(
            "Show me Acme trainers", known_brands={"acme"}
        )

        self.assertEqual(decision.intent, "buying")
        self.assertIn("message:known_brand", decision.signals)

    def test_unlabelled_number_is_not_treated_as_size_or_budget(self) -> None:
        decision = classify_intent("I need something for 2 people")

        self.assertNotIn("message:size", decision.signals)
        self.assertNotIn("message:budget", decision.signals)


class ReciprocalRankFusionTest(unittest.TestCase):
    def test_product_present_in_both_sources_rises_to_top(self) -> None:
        sparse = [result("A", 10.0, 1), result("B", 9.0, 2)]
        dense = [result("B", 0.9, 1), result("C", 0.8, 2)]

        fused = reciprocal_rank_fusion(sparse, dense)

        self.assertEqual(fused[0].parent_asin, "B")
        self.assertEqual({item.parent_asin for item in fused}, {"A", "B", "C"})

    def test_duplicate_in_one_source_is_counted_once(self) -> None:
        sparse = [result("A", 10.0, 1), result("A", 9.0, 2)]

        fused = reciprocal_rank_fusion(sparse, [], k=60)

        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].fused_score, 0.7 / 61)

    def test_empty_source_is_supported(self) -> None:
        fused = reciprocal_rank_fusion([], [result("A", 0.9, 1)])

        self.assertEqual([item.parent_asin for item in fused], ["A"])


class DistributionBasedFusionTest(unittest.TestCase):
    def test_independent_scales_are_normalized(self) -> None:
        sparse = [result("A", 100.0, 1), result("B", 10.0, 2)]
        dense = [result("B", 0.99, 1), result("A", 0.10, 2)]

        fused = distribution_based_score_fusion(
            sparse, dense, sparse_weight=0.25, dense_weight=0.75
        )

        self.assertEqual(fused[0].parent_asin, "B")

    def test_zero_variance_uses_source_rank_before_asin_for_ties(self) -> None:
        sparse = [result("B", 1.0, 2), result("A", 1.0, 1)]

        fused = distribution_based_score_fusion(sparse, [])

        self.assertEqual([item.parent_asin for item in fused], ["A", "B"])

    def test_tied_scores_preserve_better_source_rank_before_asin(self) -> None:
        sparse = [result("Z", 1.0, 1), result("A", 1.0, 2)]

        fused = distribution_based_score_fusion(sparse, [], sparse_weight=1.0)

        self.assertEqual([item.parent_asin for item in fused], ["Z", "A"])

    def test_distance_scores_can_be_marked_lower_is_better(self) -> None:
        dense = [result("CLOSE", 0.1, 1), result("FAR", 0.9, 2)]

        fused = distribution_based_score_fusion(
            [], dense, dense_higher_is_better=False
        )

        self.assertEqual(fused[0].parent_asin, "CLOSE")


class RouteAndFuseTest(unittest.TestCase):
    def test_buying_uses_rrf(self) -> None:
        sparse = [result("A", 100.0, 1), result("B", 10.0, 2)]
        dense = [result("B", 0.99, 1), result("C", 0.90, 2)]

        fused, decision = route_and_fuse(
            "Shoes in size 10 under $100", {}, sparse, dense
        )

        self.assertEqual(decision.intent, "buying")
        self.assertEqual(fused[0].parent_asin, "B")

    def test_browsing_uses_distribution_fusion_and_respects_limit(self) -> None:
        sparse = [result("A", 100.0, 1), result("B", 10.0, 2)]
        dense = [result("B", 0.99, 1), result("C", 0.90, 2)]

        fused, decision = route_and_fuse(
            "Any gift ideas for summer?", {}, sparse, dense, limit=1
        )

        self.assertEqual(decision.intent, "browsing")
        self.assertEqual(len(fused), 1)

    def test_sparse_only_fallback_preserves_validated_source_order_and_scores(self) -> None:
        sparse = [result("Z", -10.0, 1), result("A", -5.0, 2)]

        fused, decision = route_and_fuse(
            "I'm still exploring.", {}, sparse, [], limit=2
        )

        self.assertEqual(decision.intent, "browsing")
        self.assertEqual([item.parent_asin for item in fused], ["Z", "A"])
        self.assertEqual([item.fused_score for item in fused], [-10.0, -5.0])

    def test_routing_config_rejects_unusable_weight_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "buying fusion"):
            RoutingConfig(buying_sparse_weight=0.0, buying_dense_weight=0.0)


if __name__ == "__main__":
    unittest.main()
