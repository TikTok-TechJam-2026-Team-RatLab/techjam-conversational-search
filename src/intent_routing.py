from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)
BUDGET_RE = re.compile(
    r"(?:[$£€]\s*\d+(?:\.\d{1,2})?|\b(?:under|below|less than|up to|budget)"
    r"\s+[$£€]?\s*\d+)",
    re.IGNORECASE,
)
SIZE_RE = re.compile(
    r"\b(?:size\s*[:#-]?\s*(?:\d+(?:\.\d+)?|xxs|xs|s|m|l|xl|xxl)|"
    r"\d+(?:\.\d+)?\s*(?:cm|mm|inches?|inch|in|ft))\b",
    re.IGNORECASE,
)
COMMITMENT_RE = re.compile(
    r"\b(?:buy|purchase|order|must have|ready to buy|ready to order)\b",
    re.IGNORECASE,
)
SHOPPING_REQUEST_RE = re.compile(
    r"\b(?:find me|i need|i want|looking for|show me)\b",
    re.IGNORECASE,
)
KEY_REQUIREMENT_RE = re.compile(
    r"\b(?:a key requirement is|what i need is|must be|has to be|need it to)\b",
    re.IGNORECASE,
)
EXPLICIT_BROWSING_RE = re.compile(
    r"\b(?:still exploring|just browsing|not sure|open to ideas?|looking for ideas?|"
    r"need inspiration)\b",
    re.IGNORECASE,
)
BROWSING_RE = re.compile(
    r"\b(?:ideas?|inspiration|recommend|suggest|something for|what would|any good|"
    r"trendy|popular)\b",
    re.IGNORECASE,
)
USE_CASE_RE = re.compile(
    r"\b(?:gift|wedding|summer|winter|travel|vacation|party|office|school|birthday|"
    r"holiday)\b",
    re.IGNORECASE,
)
COLOR_WORDS = frozenset({
    "black", "blue", "brown", "gold", "gray", "green", "grey", "orange",
    "pink", "purple", "red", "silver", "white", "yellow",
})
MATERIAL_WORDS = frozenset({
    "cotton", "denim", "leather", "linen", "metal", "nylon", "polyester",
    "rayon", "silk", "spandex", "stainless steel", "wood", "wool",
})
HARD_CONSTRAINTS = frozenset({"budget", "size", "brand"})
SOFT_CONSTRAINTS = frozenset({
    "category", "color", "material", "feature", "style", "use_case", "other",
})


@dataclass(frozen=True)
class RoutingConfig:
    """Tunable routing and fusion parameters kept in one reproducible object."""

    buying_threshold: float = 2.0
    rrf_k: float = 60.0
    buying_sparse_weight: float = 0.70
    buying_dense_weight: float = 0.30
    browsing_sparse_weight: float = 0.30
    browsing_dense_weight: float = 0.70

    def __post_init__(self) -> None:
        values = (
            self.buying_threshold,
            self.rrf_k,
            self.buying_sparse_weight,
            self.buying_dense_weight,
            self.browsing_sparse_weight,
            self.browsing_dense_weight,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("routing parameters must be finite")
        if self.buying_threshold < 0:
            raise ValueError("buying_threshold must not be negative")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if min(
            self.buying_sparse_weight,
            self.buying_dense_weight,
            self.browsing_sparse_weight,
            self.browsing_dense_weight,
        ) < 0:
            raise ValueError("fusion weights must not be negative")
        if self.buying_sparse_weight + self.buying_dense_weight <= 0:
            raise ValueError("buying fusion needs at least one positive weight")
        if self.browsing_sparse_weight + self.browsing_dense_weight <= 0:
            raise ValueError("browsing fusion needs at least one positive weight")


DEFAULT_ROUTING_CONFIG = RoutingConfig()


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
        """A bounded buying weight for later soft-routing experiments."""

        total = self.buying_score + self.browsing_score
        return 0.5 if total <= 0 else self.buying_score / total


@dataclass(frozen=True)
class FusedResult:
    parent_asin: str
    fused_score: float
    sparse_rank: int | None
    dense_rank: int | None

    @property
    def first_rank(self) -> int:
        ranks = [rank for rank in (self.sparse_rank, self.dense_rank) if rank is not None]
        return min(ranks) if ranks else 2**31 - 1


def as_ranked_results(results: Sequence[tuple[str, float]]) -> list[RankedResult]:
    return [
        RankedResult(parent_asin=str(parent_asin), score=float(score), rank=rank)
        for rank, (parent_asin, score) in enumerate(results, start=1)
    ]


def _normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def _message_phrases(text: str, max_width: int = 6) -> set[str]:
    words = _normalize(text).split()
    return {
        " ".join(words[start:start + width])
        for width in range(1, min(max_width, len(words)) + 1)
        for start in range(len(words) - width + 1)
    }


def _matches_vocabulary(phrases: set[str], vocabulary: Iterable[str]) -> bool:
    values = vocabulary if isinstance(vocabulary, Set) else set(vocabulary)
    return not phrases.isdisjoint(values)


def classify_intent(
    user_message: str,
    active_constraints: Mapping[str, object] | None = None,
    *,
    known_brands: Iterable[str] = (),
    known_categories: Iterable[str] = (),
    buying_threshold: float = DEFAULT_ROUTING_CONFIG.buying_threshold,
) -> IntentDecision:
    """Classify concrete buying intent versus exploratory browsing.

    The current turn and accumulated state are both considered. Explicit browsing
    language wins over a generic phrase such as "looking for" until the customer
    supplies a hard constraint. This prevents the evaluator's standard "still
    exploring" message from being routed as buying merely because it names a category.
    """

    if not math.isfinite(buying_threshold) or buying_threshold < 0:
        raise ValueError("buying_threshold must be finite and non-negative")

    text = user_message.casefold()
    phrases = _message_phrases(text)
    constraints = active_constraints or {}
    active_keys = {key.casefold() for key, value in constraints.items() if value}
    signals: list[str] = []
    buying_score = 0.0
    browsing_score = 0.0
    message_has_hard_constraint = False

    def add_buying(name: str, weight: float, *, hard: bool = False) -> None:
        nonlocal buying_score, message_has_hard_constraint
        buying_score += weight
        message_has_hard_constraint = message_has_hard_constraint or hard
        signals.append(name)

    def add_browsing(name: str, weight: float) -> None:
        nonlocal browsing_score
        browsing_score += weight
        signals.append(name)

    if BUDGET_RE.search(text):
        add_buying("message:budget", 2.5, hard=True)
    if SIZE_RE.search(text):
        add_buying("message:size", 2.5, hard=True)
    if _matches_vocabulary(phrases, known_brands):
        add_buying("message:known_brand", 1.5, hard=True)
    if _matches_vocabulary(phrases, known_categories):
        add_buying("message:known_category", 0.5)
    if phrases & COLOR_WORDS:
        add_buying("message:color", 0.5)
    if phrases & MATERIAL_WORDS:
        add_buying("message:material", 0.5)
    if COMMITMENT_RE.search(text):
        add_buying("message:commitment_phrase", 2.0, hard=True)
    if KEY_REQUIREMENT_RE.search(text):
        add_buying("message:key_requirement", 1.5)
    if SHOPPING_REQUEST_RE.search(text):
        add_buying("message:shopping_request", 0.75)

    hard_count = len(active_keys & HARD_CONSTRAINTS)
    soft_count = len(active_keys & SOFT_CONSTRAINTS)
    if hard_count:
        add_buying("state:hard_constraints", min(3.0, hard_count * 1.5))
    if soft_count:
        add_buying("state:soft_constraints", min(1.4, soft_count * 0.35))

    explicit_browsing = bool(EXPLICIT_BROWSING_RE.search(text))
    if explicit_browsing:
        add_browsing("message:explicit_browsing", 3.0)
    elif BROWSING_RE.search(text):
        add_browsing("message:exploration_phrase", 1.5)
    if USE_CASE_RE.search(text) and not message_has_hard_constraint:
        add_browsing("message:generic_use_case", 0.75)
    if not text.strip():
        add_browsing("message:empty", 1.0)

    if explicit_browsing and not message_has_hard_constraint and hard_count == 0:
        intent = "browsing"
    else:
        intent = (
            "buying"
            if buying_score >= buying_threshold and buying_score > browsing_score
            else "browsing"
        )

    total = buying_score + browsing_score
    confidence = (
        0.5
        if total <= 0
        else 0.5 + 0.5 * abs(buying_score - browsing_score) / total
    )
    return IntentDecision(
        intent=intent,
        confidence=confidence,
        buying_score=buying_score,
        browsing_score=browsing_score,
        signals=tuple(signals),
    )


def _best_by_asin(results: Sequence[RankedResult]) -> dict[str, RankedResult]:
    best: dict[str, RankedResult] = {}
    for result in results:
        previous = best.get(result.parent_asin)
        if previous is None or result.rank < previous.rank:
            best[result.parent_asin] = result
    return best


def _sort_fused(results: list[FusedResult]) -> list[FusedResult]:
    results.sort(key=lambda item: (-item.fused_score, item.first_rank, item.parent_asin))
    return results


def reciprocal_rank_fusion(
    sparse_results: Sequence[RankedResult],
    dense_results: Sequence[RankedResult],
    *,
    k: float = 60.0,
    sparse_weight: float = 0.70,
    dense_weight: float = 0.30,
    limit: int | None = None,
) -> list[FusedResult]:
    """Merge source ranks while preserving the established fixed-RRF policy."""

    if not math.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive")
    if sparse_weight < 0 or dense_weight < 0:
        raise ValueError("fusion weights must be non-negative")
    if sparse_weight + dense_weight <= 0:
        raise ValueError("at least one fusion weight must be positive")

    sparse = _best_by_asin(sparse_results)
    dense = _best_by_asin(dense_results)
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
        for asin in set(sparse) | set(dense)
    ]
    ranked = _sort_fused(fused)
    return ranked[:limit] if limit is not None else ranked


def _distribution_scores(
    results: Mapping[str, RankedResult],
    *,
    higher_is_better: bool,
) -> dict[str, float]:
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
    """Normalize incompatible score distributions before weighted fusion."""

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
        for asin in set(sparse) | set(dense)
    ]
    ranked = _sort_fused(fused)
    return ranked[:limit] if limit is not None else ranked


def _single_track_fallback(
    results: Sequence[RankedResult],
    *,
    sparse: bool,
    limit: int,
) -> list[FusedResult]:
    ordered = sorted(
        _best_by_asin(results).values(),
        key=lambda item: (item.rank, item.parent_asin),
    )
    return [
        FusedResult(
            parent_asin=item.parent_asin,
            fused_score=item.score,
            sparse_rank=item.rank if sparse else None,
            dense_rank=None if sparse else item.rank,
        )
        for item in ordered[:limit]
    ]


def route_and_fuse(
    user_message: str,
    active_constraints: Mapping[str, object] | None,
    sparse_results: Sequence[RankedResult],
    dense_results: Sequence[RankedResult],
    *,
    known_brands: Iterable[str] = (),
    known_categories: Iterable[str] = (),
    limit: int = 20,
    config: RoutingConfig = DEFAULT_ROUTING_CONFIG,
    sparse_higher_is_better: bool = False,
    dense_higher_is_better: bool = True,
) -> tuple[list[FusedResult], IntentDecision]:
    """Route buying to RRF and browsing to DBSF, with safe track fallbacks."""

    if limit < 0:
        raise ValueError("limit must not be negative")
    decision = classify_intent(
        user_message,
        active_constraints,
        known_brands=known_brands,
        known_categories=known_categories,
        buying_threshold=config.buying_threshold,
    )
    if limit == 0 or (not sparse_results and not dense_results):
        return [], decision
    if not dense_results:
        return _single_track_fallback(sparse_results, sparse=True, limit=limit), decision
    if not sparse_results:
        return _single_track_fallback(dense_results, sparse=False, limit=limit), decision

    if decision.intent == "buying":
        results = reciprocal_rank_fusion(
            sparse_results,
            dense_results,
            k=config.rrf_k,
            sparse_weight=config.buying_sparse_weight,
            dense_weight=config.buying_dense_weight,
            limit=limit,
        )
    else:
        results = distribution_based_score_fusion(
            sparse_results,
            dense_results,
            sparse_weight=config.browsing_sparse_weight,
            dense_weight=config.browsing_dense_weight,
            sparse_higher_is_better=sparse_higher_is_better,
            dense_higher_is_better=dense_higher_is_better,
            limit=limit,
        )
    return results, decision
