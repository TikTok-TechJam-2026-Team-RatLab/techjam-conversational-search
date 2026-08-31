from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache

from src.data_parser import CatalogItem


TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
AROUND_BUDGET_RE = re.compile(r"\b(?:around|about|approximately|roughly|near)\b", re.I)
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "i'm", "im", "in", "is", "it", "key", "looking", "matters", "me", "my",
    "need", "of", "on", "or", "please", "requirement", "some", "that", "the", "this",
    "to", "want", "what", "with", "would", "you",
})
ATTRIBUTE_WEIGHTS = {
    "category": 1.40,
    "brand": 1.30,
    "material": 1.25,
    "color": 1.25,
    "size": 1.25,
    "budget": 1.15,
    "style": 1.10,
    "use_case": 1.10,
    "feature": 1.00,
    "other": 1.00,
}
NEGATIVE_PENALTY = 2.5
EXACT_PHRASE_BONUS = 0.75
ATTRIBUTE_FIELD_BONUS = 0.20
MINIMUM_PARTIAL_COVERAGE = 0.25
RELIABLE_NEGATIVE_ATTRIBUTES = frozenset({
    "category", "brand", "color", "material", "size", "budget",
})


@lru_cache(maxsize=16_384)
def _normalize(value: str) -> str:
    return " ".join(TOKEN_RE.findall(value.lower()))


def _canonical_token(token: str) -> str:
    """Apply a deliberately small plural normalization for catalog matching."""

    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


@lru_cache(maxsize=16_384)
def _tokens(value: str) -> frozenset[str]:
    return frozenset(
        _canonical_token(token.lower())
        for token in TOKEN_RE.findall(value)
        if token.lower() not in STOPWORDS
    )


def _flatten(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_flatten(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(item) for item in value)
    return str(value)


def _specific_attribute_text(item: CatalogItem, attribute: str) -> str:
    if attribute == "category":
        return " ".join((item.title, _flatten(item.categories)))
    if attribute == "brand":
        brand_details = " ".join(
            f"{key} {_flatten(value)}"
            for key, value in item.details.items()
            if "brand" in str(key).lower() or "manufacturer" in str(key).lower()
        )
        return " ".join((item.store, brand_details))
    return ""


def _document_frequency(items: Sequence[CatalogItem]) -> Counter[str]:
    frequency: Counter[str] = Counter()
    for item in items:
        frequency.update(_tokens(item.dense_text))
    return frequency


def _weighted_coverage(
    constraint_tokens: frozenset[str],
    candidate_tokens: frozenset[str],
    document_frequency: Mapping[str, int],
    candidate_count: int,
) -> float:
    if not constraint_tokens:
        return 0.0
    weights = {
        token: 1.0 + math.log((candidate_count + 1.0) / (document_frequency.get(token, 0) + 1.0))
        for token in constraint_tokens
    }
    total = sum(weights.values())
    return sum(weight for token, weight in weights.items() if token in candidate_tokens) / total


def _text_match_quality(
    constraint: str,
    candidate_text: str,
    document_frequency: Mapping[str, int],
    candidate_count: int,
) -> float:
    constraint_tokens = _tokens(constraint)
    if not constraint_tokens:
        return 0.0
    coverage = _weighted_coverage(
        constraint_tokens,
        _tokens(candidate_text),
        document_frequency,
        candidate_count,
    )
    normalized_constraint = _normalize(constraint)
    normalized_candidate = _normalize(candidate_text)
    exact_phrase = bool(
        normalized_constraint
        and re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_constraint)}(?![a-z0-9])",
            normalized_candidate,
        )
    )
    if coverage < MINIMUM_PARTIAL_COVERAGE and not exact_phrase:
        return 0.0
    return coverage + (EXACT_PHRASE_BONUS if exact_phrase else 0.0)


def _budget_match_quality(constraint: str, item: CatalogItem) -> float:
    if item.price is None:
        return 0.0
    amounts = [float(value) for value in NUMBER_RE.findall(constraint)]
    if not amounts:
        return 0.0
    amount = amounts[-1]
    if amount <= 0:
        return 0.0
    if AROUND_BUDGET_RE.search(constraint):
        return max(0.0, 1.0 - abs(item.price - amount) / amount)
    return 0.25 if item.price <= amount else 0.0


def _recency_weights(constraint_updated_at: Mapping[str, int]) -> dict[str, float]:
    if not constraint_updated_at:
        return {}
    oldest = min(constraint_updated_at.values())
    newest = max(constraint_updated_at.values())
    if oldest == newest:
        return {attribute: 1.0 for attribute in constraint_updated_at}
    return {
        attribute: 1.0 + 0.10 * ((turn - oldest) / (newest - oldest))
        for attribute, turn in constraint_updated_at.items()
    }


def _constraint_score(
    item: CatalogItem,
    constraints: Mapping[str, Sequence[str]],
    *,
    document_frequency: Mapping[str, int],
    candidate_count: int,
    recency_weights: Mapping[str, float],
) -> float:
    score = 0.0
    for attribute, values in constraints.items():
        attribute_weight = ATTRIBUTE_WEIGHTS.get(attribute, 1.0)
        recency_weight = recency_weights.get(attribute, 1.0)
        candidate_text = item.dense_text
        specific_text = _specific_attribute_text(item, attribute)
        for value in values:
            if attribute == "budget":
                quality = _budget_match_quality(str(value), item)
            else:
                quality = _text_match_quality(
                    str(value),
                    candidate_text,
                    document_frequency,
                    candidate_count,
                )
                if specific_text:
                    quality += ATTRIBUTE_FIELD_BONUS * _text_match_quality(
                        str(value),
                        specific_text,
                        document_frequency,
                        candidate_count,
                    )
            score += attribute_weight * recency_weight * quality
    return score


def rerank_candidates(
    candidates: Sequence[tuple[str, float]],
    items_by_asin: Mapping[str, CatalogItem],
    *,
    active_constraints: Mapping[str, Sequence[str]],
    negative_constraints: Mapping[str, Sequence[str]],
    constraint_updated_at: Mapping[str, int] | None = None,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Rerank a bounded retrieval pool using only live state and catalog evidence.

    Original retrieval scores are returned unchanged because sparse and fused scores use
    different scales. The original rank is the stable tie-breaker, so a session with no
    usable constraints retains the retrieval order exactly.
    """

    if top_k <= 0 or not candidates:
        return []

    available = [
        items_by_asin[parent_asin]
        for parent_asin, _ in candidates
        if parent_asin in items_by_asin
    ]
    if not available or (not active_constraints and not negative_constraints):
        return list(candidates[:top_k])

    reliable_negative_constraints = {
        attribute: values
        for attribute, values in negative_constraints.items()
        if attribute in RELIABLE_NEGATIVE_ATTRIBUTES
    }
    document_frequency = _document_frequency(available)
    recency_weights = _recency_weights(constraint_updated_at or {})
    scored: list[tuple[float, int, tuple[str, float]]] = []
    for original_rank, candidate in enumerate(candidates):
        parent_asin, _ = candidate
        item = items_by_asin.get(parent_asin)
        if item is None:
            net_score = 0.0
        else:
            positive_score = _constraint_score(
                item,
                active_constraints,
                document_frequency=document_frequency,
                candidate_count=len(available),
                recency_weights=recency_weights,
            )
            negative_score = _constraint_score(
                item,
                reliable_negative_constraints,
                document_frequency=document_frequency,
                candidate_count=len(available),
                recency_weights=recency_weights,
            )
            net_score = positive_score - NEGATIVE_PENALTY * negative_score
        scored.append((net_score, original_rank, candidate))

    scored.sort(key=lambda result: (-round(result[0], 12), result[1]))
    return [candidate for _, _, candidate in scored[:top_k]]
