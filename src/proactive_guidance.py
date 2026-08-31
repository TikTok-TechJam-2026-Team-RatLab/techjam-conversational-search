from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from src.data_parser import CatalogItem


ALLOWED_ASK_ATTRIBUTES = frozenset({
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
})

# Prefer concrete, easy-to-answer attributes when two attributes have the same
# selection utility. The utility still decides whenever one attribute separates
# the candidate pool more effectively.
ATTRIBUTE_ORDER = (
    "material",
    "color",
    "size",
    "style",
    "feature",
    "use_case",
    "brand",
    "budget",
    "category",
)

QUESTION_TEMPLATES = {
    "category": "What type of product would you prefer?",
    "material": "Do you have a preferred material?",
    "color": "Do you have a preferred color?",
    "size": "What size or fit do you need?",
    "style": "What style would you prefer?",
    "brand": "Do you have a preferred brand?",
    "budget": "What budget should I work within?",
    "feature": "Which product feature matters most to you?",
    "use_case": "What will you mainly use the product for?",
    "other": "What other requirement matters most to you?",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)
MONEY_RE = re.compile(r"(?:[$£€]\s*\d|\b(?:budget|under|below|less than|up to)\b)", re.I)
SIZE_RE = re.compile(r"\b(?:size|sizing|width|wide|narrow|dimensions?|length)\b", re.I)
STYLE_RE = re.compile(r"\b(?:department|style|fit|sleeve|neck|pattern)\b", re.I)
USE_CASE_RE = re.compile(
    r"\b(?:hiking|running|gym|winter|outdoor|work|travel|wedding|party|school|sport)\b",
    re.I,
)
COLOR_RE = re.compile(
    r"\b(?:black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|"
    r"gold|silver)\b",
    re.I,
)
MATERIAL_RE = re.compile(
    r"\b(?:cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|denim|"
    r"bamboo|fabric)\b",
    re.I,
)
GENERIC_CATEGORIES = frozenset({
    "clothing",
    "clothing shoes jewelry",
    "clothing shoes and jewelry",
    "men",
    "women",
})
CONFIDENT_SCORE_MARGIN = 0.50


@dataclass(frozen=True)
class GuidanceDecision:
    ask_attribute: str | None
    message: str
    information_gain: float = 0.0
    selection_utility: float = 0.0


def _normalize(value: object) -> str:
    return " ".join(TOKEN_RE.findall(str(value).lower()))


def _flatten(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}: {_flatten(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_flatten(item) for item in value)
    return str(value)


def _evidence_attribute(text: str) -> str:
    """Map a catalog fact to the supported clarification attribute it answers."""

    if MONEY_RE.search(text):
        return "budget"
    if MATERIAL_RE.search(text) or re.search(r"\b(?:material|fabric)\b", text, re.I):
        return "material"
    if COLOR_RE.search(text) or re.search(r"\bcolou?r\b", text, re.I):
        return "color"
    if SIZE_RE.search(text):
        return "size"
    if STYLE_RE.search(text):
        return "style"
    if USE_CASE_RE.search(text):
        return "use_case"
    if re.search(r"\b(?:brand|manufacturer)\b", text, re.I):
        return "brand"
    return "feature"


def candidate_attribute_values(item: CatalogItem) -> dict[str, tuple[str, ...]]:
    """Extract deterministic, bounded attribute values from one catalog item."""

    values: dict[str, set[str]] = {attribute: set() for attribute in ATTRIBUTE_ORDER}

    categories: list[str] = []
    for category in item.categories:
        for part in str(category).split(","):
            normalized = _normalize(part)
            if normalized and normalized not in GENERIC_CATEGORIES:
                categories.append(normalized)
    if categories:
        values["category"].add(categories[-1])

    if item.store:
        values["brand"].add(_normalize(item.store))

    evidence = [str(feature) for feature in item.features]
    evidence.extend(f"{key}: {_flatten(value)}" for key, value in item.details.items())
    for fact in evidence:
        normalized = _normalize(fact)
        if normalized:
            values[_evidence_attribute(fact)].add(normalized)

    searchable_text = " ".join(
        (
            item.title,
            _flatten(item.features),
            _flatten(item.details),
            _flatten(item.description),
        )
    )
    values["color"].update(_normalize(match.group(0)) for match in COLOR_RE.finditer(searchable_text))
    values["material"].update(
        _normalize(match.group(0)) for match in MATERIAL_RE.finditer(searchable_text)
    )

    if item.price is not None:
        if item.price < 25:
            band = "under 25"
        elif item.price < 50:
            band = "25 to 50"
        elif item.price < 100:
            band = "50 to 100"
        else:
            band = "100 or more"
        values["budget"].add(band)

    return {
        attribute: tuple(sorted(value for value in attribute_values if value))
        for attribute, attribute_values in values.items()
    }


def _attribute_information_gain(
    candidate_attributes: Sequence[Mapping[str, tuple[str, ...]]],
    attribute: str,
) -> float:
    signatures = [attributes[attribute] for attributes in candidate_attributes]
    known_signatures = [signature for signature in signatures if signature]
    known_count = len(known_signatures)
    if known_count < 2:
        return 0.0

    # Model a reply as one of a candidate's known values, selected uniformly.
    # Unlike partitioning by the whole value tuple, this preserves the residual
    # uncertainty of shared colors, materials, and features.
    response_mass: Counter[str] = Counter()
    response_candidate_mass: dict[str, Counter[int]] = defaultdict(Counter)
    candidate_prior = 1.0 / known_count
    for candidate_index, signature in enumerate(known_signatures):
        joint_probability = candidate_prior / len(signature)
        for response in signature:
            response_mass[response] += joint_probability
            response_candidate_mass[response][candidate_index] += joint_probability

    expected_posterior_entropy = 0.0
    for response, probability in response_mass.items():
        posterior_entropy = -sum(
            (joint_probability / probability) * math.log2(joint_probability / probability)
            for joint_probability in response_candidate_mass[response].values()
        )
        expected_posterior_entropy += probability * posterior_entropy

    prior_entropy = math.log2(known_count)
    coverage = known_count / len(signatures)
    return max(0.0, prior_entropy - expected_posterior_entropy) * coverage


def _attribute_selection_utility(
    candidate_attributes: Sequence[Mapping[str, tuple[str, ...]]],
    attribute: str,
) -> float:
    """Score how distinctly an attribute describes the retrieved catalog set."""

    signatures = [attributes[attribute] for attributes in candidate_attributes]
    known_count = sum(bool(signature) for signature in signatures)
    if known_count == 0:
        return 0.0

    partitions = Counter(signature or ("<unknown>",) for signature in signatures)
    if len(partitions) < 2:
        return 0.0

    total = len(signatures)
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in partitions.values()
    )
    return entropy * (known_count / total)


def result_set_is_ambiguous(candidate_scores: Sequence[float] | None) -> bool:
    """Return whether the leading retrieval scores lack a decisive winner."""

    if candidate_scores is None or len(candidate_scores) < 2:
        return True
    leading, runner_up = candidate_scores[:2]
    if not math.isfinite(leading) or not math.isfinite(runner_up):
        return True
    scale = max(abs(leading), abs(runner_up), 1e-12)
    relative_margin = abs(leading - runner_up) / scale
    return relative_margin < CONFIDENT_SCORE_MARGIN


def choose_clarification(
    candidates: Sequence[CatalogItem],
    *,
    candidate_scores: Sequence[float] | None = None,
    force_clarification: bool = False,
    unavailable_attributes: Iterable[str] = (),
) -> GuidanceDecision:
    """Choose the supported attribute with the highest catalog selection utility."""

    unavailable = {str(attribute).lower() for attribute in unavailable_attributes}
    if len(candidates) == 1:
        return GuidanceDecision(None, "Here are the closest matches I found.")
    if candidates and not force_clarification and not result_set_is_ambiguous(candidate_scores):
        return GuidanceDecision(None, "Here are the closest matches I found.")

    candidate_attributes = [candidate_attribute_values(item) for item in candidates]
    best_attribute: str | None = None
    best_utility = 0.0
    for attribute in ATTRIBUTE_ORDER:
        if attribute in unavailable:
            continue
        utility = _attribute_selection_utility(candidate_attributes, attribute)
        if utility > best_utility + 1e-12:
            best_attribute = attribute
            best_utility = utility

    if best_attribute is not None:
        return GuidanceDecision(
            best_attribute,
            QUESTION_TEMPLATES[best_attribute],
            _attribute_information_gain(candidate_attributes, best_attribute),
            best_utility,
        )

    if "other" not in unavailable:
        return GuidanceDecision("other", QUESTION_TEMPLATES["other"])
    return GuidanceDecision(None, "Here are the closest matches I found.")
