from __future__ import annotations

import math
import re
from collections import Counter
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

# Prefer concrete, easy-to-answer attributes when two attributes provide the
# same amount of information. The information-gain calculation still decides
# whenever one attribute separates the candidate pool more effectively.
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


@dataclass(frozen=True)
class GuidanceDecision:
    ask_attribute: str | None
    message: str
    information_gain: float = 0.0


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
    known_count = sum(bool(signature) for signature in signatures)
    if known_count == 0:
        return 0.0

    # An empty signature is a real partition, but coverage discounts questions
    # that many candidate products cannot answer.
    partitions = Counter(signature or ("<unknown>",) for signature in signatures)
    if len(partitions) < 2:
        return 0.0

    total = len(signatures)
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in partitions.values()
    )
    coverage = known_count / total
    return entropy * coverage


def choose_clarification(
    candidates: Sequence[CatalogItem],
    *,
    unavailable_attributes: Iterable[str] = (),
) -> GuidanceDecision:
    """Choose the supported attribute with the highest expected information gain."""

    unavailable = {str(attribute).lower() for attribute in unavailable_attributes}
    if len(candidates) == 1:
        return GuidanceDecision(None, "Here are the closest matches I found.")

    candidate_attributes = [candidate_attribute_values(item) for item in candidates]
    best_attribute: str | None = None
    best_gain = 0.0
    for attribute in ATTRIBUTE_ORDER:
        if attribute in unavailable:
            continue
        gain = _attribute_information_gain(candidate_attributes, attribute)
        if gain > best_gain + 1e-12:
            best_attribute = attribute
            best_gain = gain

    if best_attribute is not None:
        return GuidanceDecision(
            best_attribute,
            QUESTION_TEMPLATES[best_attribute],
            best_gain,
        )

    if "other" not in unavailable:
        return GuidanceDecision("other", QUESTION_TEMPLATES["other"])
    return GuidanceDecision(None, "Here are the closest matches I found.")
