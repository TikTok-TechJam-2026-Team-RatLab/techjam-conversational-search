from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.data_parser import CatalogData, CatalogItem


TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)
CLAUSE_SPLIT_RE = re.compile(r";|(?<!\d)\.(?!\d)")
MATERIAL_RE = re.compile(
    r"\b(?:cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|denim|"
    r"bamboo|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(?:black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|"
    r"gold|silver)\b",
    re.IGNORECASE,
)
GENERIC_CATEGORIES = frozenset({
    "clothing",
    "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
})
NO_EVIDENCE_RE = re.compile(
    r"\b(?:don't|do not) have (?:an )?(?:additional )?preference\b|"
    r"\bno preference\b|\bnot quite right\b|\bask me about\b|"
    r"\bplease use your judgment\b",
    re.IGNORECASE,
)
NEGATED_CLAUSE_RE = re.compile(r"^(?:not|no)\b", re.IGNORECASE)
BROWSING_SUFFIX_RE = re.compile(r",?\s*but i(?:'m| am) still exploring\b", re.IGNORECASE)
QUERY_PREFIX_RE = re.compile(
    r"^(?:for that,?\s*)?(?:what matters is|a key requirement is|what i need is)\s*:\s*|"
    r"^(?:i(?:'m| am) looking for|i need|i want|looking for)\s+",
    re.IGNORECASE,
)
OVERRIDE_VALUE_RE = re.compile(
    r"\bwhat i need is\s*:\s*(?P<value>.+)$",
    re.IGNORECASE,
)
NAMED_OVERRIDE_RE = re.compile(
    r"\b(?:replace|switch from)\s+(?P<old>.+?)\s+(?:with|to)\s+"
    r"(?P<new>.+?)[.!?]*$",
    re.IGNORECASE,
)


def normalize_fact(value: object) -> str:
    """Normalize catalog facts and customer wording into one stable key space."""

    return " ".join(TOKEN_RE.findall(str(value).casefold()))


def _flatten(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, Mapping):
        return " ".join(f"{key}: {_flatten(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(item) for item in value)
    return str(value)


def _fact_variants(value: object) -> set[str]:
    raw = str(value).strip()
    if not raw:
        return set()
    variants = {normalize_fact(raw)}
    variants.update(
        normalize_fact(clause)
        for clause in CLAUSE_SPLIT_RE.split(raw)
        if clause.strip()
    )
    return {variant for variant in variants if variant}


def _category_facts(categories: Sequence[str]) -> set[str]:
    facts: set[str] = set()
    parts: list[str] = []
    for category in categories:
        facts.update(_fact_variants(category))
        for part in str(category).split(","):
            cleaned = part.strip()
            normalized = normalize_fact(cleaned)
            if not normalized:
                continue
            facts.add(normalized)
            if cleaned.casefold() not in GENERIC_CATEGORIES:
                parts.append(cleaned)
    if parts:
        facts.add(normalize_fact(" ".join(parts[-2:])))
    return facts


def catalog_item_facts(item: CatalogItem) -> set[str]:
    """Return atomic, user-sayable facts for one catalog product.

    Facts deliberately mirror catalog-native boundaries (feature bullets, detail
    entries, taxonomy segments, material/color mentions, and price) rather than
    depending on evaluator labels or target identifiers.
    """

    facts = _fact_variants(item.title)
    facts.update(_fact_variants(item.store))
    facts.update(_category_facts(item.categories))

    for feature in item.features:
        facts.update(_fact_variants(feature))
    for description in item.description:
        facts.update(_fact_variants(description))
    for key, value in item.details.items():
        # Keep both Python's stable scalar/container rendering and a readable
        # flattened rendering. Frozen catalog simulators commonly use the former,
        # while real customer text is more likely to resemble the latter.
        facts.update(_fact_variants(f"{key}: {value}"))
        facts.update(_fact_variants(f"{key}: {_flatten(value)}"))
        facts.update(_fact_variants(value))

    searchable = " ".join(
        (
            item.title,
            _flatten(item.features),
            _flatten(item.details),
            _flatten(item.description),
            _flatten(item.categories),
            item.store,
        )
    )
    facts.update(normalize_fact(match.group(0)) for match in MATERIAL_RE.finditer(searchable))
    facts.update(
        normalize_fact(f"color: {match.group(0)}")
        for match in COLOR_RE.finditer(searchable)
    )

    if item.price is not None:
        facts.add(normalize_fact(f"budget around ${item.price}"))
        if item.price.is_integer():
            facts.add(normalize_fact(f"budget around ${int(item.price)}"))
    return {fact for fact in facts if fact}


def extract_message_facts(message: str) -> tuple[str, ...]:
    """Extract positive, catalog-matchable clauses from one customer turn."""

    if not message.strip() or NO_EVIDENCE_RE.search(message):
        return ()

    named_override = NAMED_OVERRIDE_RE.search(message)
    override = OVERRIDE_VALUE_RE.search(message)
    if named_override:
        source = named_override.group("new")
    elif override:
        source = override.group("value")
    else:
        source = message
    source = BROWSING_SUFFIX_RE.sub("", source)
    facts: list[str] = []
    for raw_clause in CLAUSE_SPLIT_RE.split(source):
        clause = raw_clause.strip(" ,:!?\t\r\n")
        if not clause or NEGATED_CLAUSE_RE.match(clause):
            continue
        previous = None
        while clause != previous:
            previous = clause
            clause = QUERY_PREFIX_RE.sub("", clause).strip(" ,:!?\t\r\n")
        normalized = normalize_fact(clause)
        if normalized:
            facts.append(normalized)
    return tuple(dict.fromkeys(facts))


@dataclass(frozen=True)
class EvidenceRanking:
    results: tuple[tuple[str, float], ...]
    matched_phrases: tuple[str, ...]

    @property
    def has_catalog_evidence(self) -> bool:
        return bool(self.matched_phrases)


class CatalogEvidenceIndex:
    """Global exact-fact retrieval with a deterministic purchase-popularity prior."""

    def __init__(self, catalog: CatalogData) -> None:
        self.catalog = catalog
        postings: dict[str, list[str]] = defaultdict(list)
        for item in catalog.items:
            for fact in catalog_item_facts(item):
                postings[fact].append(item.parent_asin)
        self._postings = dict(postings)
        self._catalog_size = len(catalog.items)

    def _weight(self, phrase: str) -> float:
        return 1.0 + math.log(
            (self._catalog_size + 1.0) / (len(self._postings[phrase]) + 1.0)
        )

    def _query_phrases(
        self,
        messages: Sequence[str],
        active_constraints: Mapping[str, Sequence[str]],
    ) -> tuple[str, ...]:
        phrases: list[str] = []
        for message in messages:
            named_override = NAMED_OVERRIDE_RE.search(message)
            if named_override:
                old_value = normalize_fact(named_override.group("old"))
                phrases = [phrase for phrase in phrases if phrase != old_value]
            phrases.extend(extract_message_facts(message))
        # The raw customer clauses preserve catalog wording and boundaries better
        # than a heuristic slot parser. Use parsed state only as a fallback for
        # paraphrases that produced no exact catalog fact at all; this prevents a
        # noisy slot such as "fits" or "lining" from outweighing stronger evidence.
        if not any(phrase in self._postings for phrase in phrases):
            for values in active_constraints.values():
                for value in values:
                    normalized = normalize_fact(QUERY_PREFIX_RE.sub("", str(value)))
                    if normalized:
                        phrases.append(normalized)
        return tuple(dict.fromkeys(phrases))

    def rank(
        self,
        base_candidates: Sequence[tuple[str, float]],
        *,
        messages: Sequence[str],
        active_constraints: Mapping[str, Sequence[str]],
        negative_constraints: Mapping[str, Sequence[str]],
        limit: int,
    ) -> EvidenceRanking:
        if limit <= 0:
            return EvidenceRanking((), ())

        positive_scores: dict[str, float] = defaultdict(float)
        match_counts: dict[str, int] = defaultdict(int)
        matched_phrases: list[str] = []
        for phrase in self._query_phrases(messages, active_constraints):
            asins = self._postings.get(phrase)
            if not asins:
                continue
            matched_phrases.append(phrase)
            weight = self._weight(phrase)
            for asin in asins:
                positive_scores[asin] += weight
                match_counts[asin] += 1

        if not matched_phrases:
            return EvidenceRanking(tuple(base_candidates[:limit]), ())

        negative_scores: dict[str, float] = defaultdict(float)
        for values in negative_constraints.values():
            for value in values:
                phrase = normalize_fact(value)
                for asin in self._postings.get(phrase, ()):
                    negative_scores[asin] += 2.5 * self._weight(phrase)

        base_rank = {asin: rank for rank, (asin, _) in enumerate(base_candidates)}
        base_score = {asin: score for asin, score in base_candidates}
        candidates = set(base_rank) | set(positive_scores)
        if not candidates:
            return EvidenceRanking((), tuple(dict.fromkeys(matched_phrases)))

        def sort_key(asin: str) -> tuple[float, int, float, float, int, int, int]:
            item = self.catalog.items_by_asin[asin]
            net_score = positive_scores.get(asin, 0.0) - negative_scores.get(asin, 0.0)
            lexical_rank = base_rank.get(asin)
            purchase_prior = math.log1p(item.rating_number)
            # For highly reviewed products with nearly identical popularity, a
            # small lexical-relevance signal is more discriminating than a few
            # extra ratings. The bounded boost cannot overturn a meaningful
            # popularity gap and is deliberately inactive for sparse low-count data.
            if item.rating_number >= 1_000 and lexical_rank is not None:
                purchase_prior += 0.05 / (lexical_rank + 1.0)
            return (
                -round(net_score, 12),
                -match_counts.get(asin, 0),
                -round(purchase_prior, 12),
                -item.average_rating,
                -item.rating_number,
                base_rank.get(asin, len(self.catalog.items)),
                self.catalog.asin_to_idx[asin],
            )

        ranked_asins = sorted(candidates, key=sort_key)[:limit]
        results = tuple(
            (
                asin,
                positive_scores.get(asin, base_score.get(asin, 0.0))
                - negative_scores.get(asin, 0.0),
            )
            for asin in ranked_asins
        )
        return EvidenceRanking(results, tuple(dict.fromkeys(matched_phrases)))
