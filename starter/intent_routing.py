"""Compatibility imports for the original teammate branch.

Runtime code lives in :mod:`src.intent_routing`; this module remains so existing
imports from the prototype branch continue to work.
"""

from src.intent_routing import (
    DEFAULT_ROUTING_CONFIG,
    FusedResult,
    IntentDecision,
    RankedResult,
    RoutingConfig,
    as_ranked_results,
    classify_intent,
    distribution_based_score_fusion,
    reciprocal_rank_fusion,
    route_and_fuse,
)

__all__ = [
    "DEFAULT_ROUTING_CONFIG",
    "FusedResult",
    "IntentDecision",
    "RankedResult",
    "RoutingConfig",
    "as_ranked_results",
    "classify_intent",
    "distribution_based_score_fusion",
    "reciprocal_rank_fusion",
    "route_and_fuse",
]
