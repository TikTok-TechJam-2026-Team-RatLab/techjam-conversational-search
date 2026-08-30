from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


BUDGET_RE = re.compile(
    r"(?:[$£€]\s*\d+(?:\.\d{1,2})?|\b(?:under|below|less than|up to|budget)\s+[$£€]?\s*\d+)",
    re.IGNORECASE,
)
SIZE_RE = re.compile(
    r"\b(?:size\s*[:#-]?\s*(?:\d+(?:\.\d+)?|xxs|xs|s|m|l|xl|xxl)|"
    r"\d+(?:\.\d+)?\s*(?:cm|mm|inches?|inch|in|ft))\b",
    re.IGNORECASE,
)
PURCHASE_RE = re.compile(
    r"\b(?:buy|purchase|order|find me|i need|i want|looking for|must have|show me)\b",
    re.IGNORECASE,
)
BROWSING_RE = re.compile(
    r"\b(?:ideas?|inspiration|recommend|suggest|something for|still exploring|not sure|"
    r"what would|any good|trendy|popular)\b",
    re.IGNORECASE,
)
USE_CASE_RE = re.compile(
    r"\b(?:gift|wedding|summer|winter|travel|vacation|party|office|school|birthday|holiday)\b",
    re.IGNORECASE,
)
COLOR_WORDS = {
    "black", "blue", "brown", "gold", "gray", "green", "grey", "orange",
    "pink", "purple", "red", "silver", "white", "yellow",
}
MATERIAL_WORDS = {
    "cotton", "denim", "leather", "linen", "metal", "polyester", "silk",
    "stainless steel", "wood", "wool",
}
HARD_CONSTRAINTS = {"budget", "size", "brand"}
SOFT_CONSTRAINTS = {"category", "color", "material", "feature", "style"}


@dataclass(frozen=True)
class RankedResult:
    """One retriever result, with rank starting at one."""

    parent_asin: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not self.parent_asin:
            raise ValueError("parent_asin must not be empty")
        if self.rank < 1:
            raise ValueError("rank must start at one")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")


@dataclass(frozen=True)
class IntentDecision:
    intent: str
    confidence: float
    buying_score: float
    browsing_score: float
    signals: tuple[str, ...]

    @property
    def alpha(self) -> float:
        """Buying weight for future soft routing experiments."""

        return self.confidence if self.intent == "buying" else 1.0 - self.confidence


@dataclass(frozen=True)
class FusedResult:
    parent_asin: str
    fused_score: float
    sparse_rank: int | None
    dense_rank: int | None


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase.lower())}(?!\w)", text))


def _matches_vocabulary(text: str, vocabulary: Iterable[str]) -> bool:
    return any(value and _contains_phrase(text, value) for value in vocabulary)


def classify_intent(
    user_message: str,
    active_constraints: Mapping[str, object] | None = None,
    *,
    known_brands: Iterable[str] = (),
    known_categories: Iterable[str] = (),
    buying_threshold: float = 2.0,
) -> IntentDecision:
    """Classify a turn as concrete buying intent or exploratory browsing.

    The current message and accumulated dialogue constraints are both used so a
    short follow-up such as "blue" does not erase an already concrete intent.
    """

    text = user_message.casefold()
    constraints = active_constraints or {}
    active_keys = {key.casefold() for key, value in constraints.items() if value}
    signals: list[str] = []
    buying_score = 0.0
    browsing_score = 0.0

    def add_buying(name: str, weight: float) -> None:
        nonlocal buying_score
        buying_score += weight
        signals.append(name)

    def add_browsing(name: str, weight: float) -> None:
        nonlocal browsing_score
        browsing_score += weight
        signals.append(name)

    if BUDGET_RE.search(text):
        add_buying("message:budget", 2.0)
    if SIZE_RE.search(text):
        add_buying("message:size", 2.0)
    if _matches_vocabulary(text, known_brands):
        add_buying("message:known_brand", 1.5)
    if _matches_vocabulary(text, known_categories):
        add_buying("message:known_category", 1.0)
    if any(_contains_phrase(text, color) for color in COLOR_WORDS):
        add_buying("message:color", 0.75)
    if any(_contains_phrase(text, material) for material in MATERIAL_WORDS):
        add_buying("message:material", 0.75)
    if PURCHASE_RE.search(text):
        add_buying("message:purchase_phrase", 1.5)

    hard_count = len(active_keys & HARD_CONSTRAINTS)
    soft_count = len(active_keys & SOFT_CONSTRAINTS)
    if hard_count:
        add_buying("state:hard_constraints", min(3.0, hard_count * 1.5))
    if soft_count:
        add_buying("state:soft_constraints", min(1.5, soft_count * 0.5))

    if BROWSING_RE.search(text):
        add_browsing("message:exploration_phrase", 1.5)
    if USE_CASE_RE.search(text) and not (BUDGET_RE.search(text) or SIZE_RE.search(text)):
        add_browsing("message:generic_use_case", 1.0)
    if not text.strip():
        add_browsing("message:empty", 1.0)

    intent = (
        "buying"
        if buying_score >= buying_threshold and buying_score >= browsing_score
        else "browsing"
    )
    total = buying_score + browsing_score
    if total == 0:
        confidence = 0.5
    else:
        confidence = 0.5 + 0.5 * abs(buying_score - browsing_score) / total

    return IntentDecision(
        intent=intent,
        confidence=confidence,
        buying_score=buying_score,
        browsing_score=browsing_score,
        signals=tuple(signals),
    )


def _best_by_asin(results: Sequence[RankedResult]) -> dict[str, RankedResult]:
    """Deduplicate a source list without double-counting repeated products."""

    best: dict[str, RankedResult] = {}
    for result in results:
        previous = best.get(result.parent_asin)
        if previous is None or result.rank < previous.rank:
            best[result.parent_asin] = result
    return best


def reciprocal_rank_fusion(
    sparse_results: Sequence[RankedResult],
    dense_results: Sequence[RankedResult],
    *,
    k: float = 60.0,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
    limit: int | None = None,
) -> list[FusedResult]:
    """Merge retriever ranks using weighted reciprocal rank fusion."""

    if k <= 0:
        raise ValueError("k must be positive")
    if sparse_weight < 0 or dense_weight < 0:
        raise ValueError("fusion weights must be non-negative")

    sparse = _best_by_asin(sparse_results)
    dense = _best_by_asin(dense_results)
    asins = set(sparse) | set(dense)
    fused = [
        FusedResult(
            parent_asin=asin,
            fused_score=(
                sparse_weight / (k + sparse[asin].rank) if asin in sparse else 0.0
            ) + (
                dense_weight / (k + dense[asin].rank) if asin in dense else 0.0
            ),
            sparse_rank=sparse[asin].rank if asin in sparse else None,
            dense_rank=dense[asin].rank if asin in dense else None,
        )
        for asin in asins
    ]
    fused.sort(key=lambda item: (-item.fused_score, item.parent_asin))
    return fused[:limit] if limit is not None else fused


def _distribution_scores(
    results: Mapping[str, RankedResult],
    *,
    higher_is_better: bool,
) -> dict[str, float]:
    """Normalize one source to [0, 1] using its mean +/- three sigma."""

    if not results:
        return {}
    oriented = {
        asin: result.score if higher_is_better else -result.score
        for asin, result in results.items()
    }
    values = list(oriented.values())
    mean = statistics.fmean(values)
    stddev = statistics.pstdev(values)
    if stddev < 1e-12:
        return {asin: 0.5 for asin in oriented}
    lower = mean - 3.0 * stddev
    width = 6.0 * stddev
    return {
        asin: min(1.0, max(0.0, (score - lower) / width))
        for asin, score in oriented.items()
    }


def distribution_based_score_fusion(
    sparse_results: Sequence[RankedResult],
    dense_results: Sequence[RankedResult],
    *,
    sparse_weight: float = 0.35,
    dense_weight: float = 0.65,
    sparse_higher_is_better: bool = True,
    dense_higher_is_better: bool = True,
    limit: int | None = None,
) -> list[FusedResult]:
    """Normalize each score distribution independently, then combine it."""

    if sparse_weight < 0 or dense_weight < 0:
        raise ValueError("fusion weights must be non-negative")
    weight_total = sparse_weight + dense_weight
    if weight_total <= 0:
        raise ValueError("at least one fusion weight must be positive")
    sparse_weight /= weight_total
    dense_weight /= weight_total

    sparse = _best_by_asin(sparse_results)
    dense = _best_by_asin(dense_results)
    sparse_scores = _distribution_scores(
        sparse, higher_is_better=sparse_higher_is_better
    )
    dense_scores = _distribution_scores(dense, higher_is_better=dense_higher_is_better)
    asins = set(sparse) | set(dense)
    fused = [
        FusedResult(
            parent_asin=asin,
            fused_score=(
                sparse_weight * sparse_scores.get(asin, 0.0)
                + dense_weight * dense_scores.get(asin, 0.0)
            ),
            sparse_rank=sparse[asin].rank if asin in sparse else None,
            dense_rank=dense[asin].rank if asin in dense else None,
        )
        for asin in asins
    ]
    fused.sort(key=lambda item: (-item.fused_score, item.parent_asin))
    return fused[:limit] if limit is not None else fused


def route_and_fuse(
    user_message: str,
    active_constraints: Mapping[str, object] | None,
    sparse_results: Sequence[RankedResult],
    dense_results: Sequence[RankedResult],
    *,
    known_brands: Iterable[str] = (),
    known_categories: Iterable[str] = (),
    limit: int = 20,
    rrf_k: float = 60.0,
    buying_sparse_weight: float = 1.0,
    buying_dense_weight: float = 1.0,
    browsing_sparse_weight: float = 0.35,
    browsing_dense_weight: float = 0.65,
    sparse_higher_is_better: bool = True,
    dense_higher_is_better: bool = True,
) -> tuple[list[FusedResult], IntentDecision]:
    """Route buying intent to RRF and browsing intent to DBSF."""

    if limit < 0:
        raise ValueError("limit must not be negative")
    decision = classify_intent(
        user_message,
        active_constraints,
        known_brands=known_brands,
        known_categories=known_categories,
    )
    if decision.intent == "buying":
        results = reciprocal_rank_fusion(
            sparse_results,
            dense_results,
            k=rrf_k,
            sparse_weight=buying_sparse_weight,
            dense_weight=buying_dense_weight,
            limit=limit,
        )
    else:
        results = distribution_based_score_fusion(
            sparse_results,
            dense_results,
            sparse_weight=browsing_sparse_weight,
            dense_weight=browsing_dense_weight,
            sparse_higher_is_better=sparse_higher_is_better,
            dense_higher_is_better=dense_higher_is_better,
            limit=limit,
        )
    return results, decision
