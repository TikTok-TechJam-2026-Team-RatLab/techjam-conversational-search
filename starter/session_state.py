from __future__ import annotations

import re
from dataclasses import dataclass, field


COMMON_COLORS = frozenset({
    "black", "blue", "brown", "gold", "gray", "green", "grey", "orange",
    "pink", "purple", "red", "silver", "white", "yellow",
})
COMMON_MATERIALS = frozenset({"cotton", "denim", "leather", "linen", "polyester", "silk", "wool"})
GENERIC_CATEGORIES = frozenset({"clothing", "clothing shoes jewelry", "women", "men"})
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.IGNORECASE)
SIZE_RE = re.compile(r"\bsize\s*[:#-]?\s*(?P<value>[a-z0-9][a-z0-9./-]*)\b", re.IGNORECASE)
BUDGET_RE = re.compile(
    r"(?:\b(?:budget(?:\s+(?:is|of))?|under|below|less than|up to|max(?:imum)?)\s*:?[ ]*"
    r"(?P<amount>\$?\d+(?:\.\d{1,2})?)\b)|(?P<currency>\$\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
NAMED_OVERRIDE_RE = re.compile(
    r"\b(?:replace|switch from)\s+(?P<old>.+?)\s+(?:with|to)\s+(?P<new>.+?)[.!?]*$",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead)\b.*?\b(?:ignore|replace|forget)\b.*?"
    r"(?:what i need is|(?:i (?:need|want|prefer))|instead)\s*:\s*(?P<new>.+)",
    re.IGNORECASE,
)
REMOVE_ATTRIBUTE_RE = re.compile(
    r"\b(?:i\s+)?(?:no longer|don't|do not)\s+(?:care|have a preference)\s+"
    r"(?:about\s+|for\s+)?(?:the\s+)?(?P<attribute>category|brand|color|colour|material|size|budget)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(r"(?:^|[.;,]\s*|\b(?:but|and)\s+)(?:not|no)\s+(?P<value>[^.;,]+)", re.IGNORECASE)
CONSTRAINT_PREFIX_RE = re.compile(
    r"^(?:for that,?\s*)?(?:what matters is|a key requirement is)\s*:\s*",
    re.IGNORECASE,
)
BROWSING_FILLER_RE = re.compile(r",?\s*but i(?:'m| am) still exploring\b", re.IGNORECASE)
NON_CONSTRAINT_PHRASES = ("ask me about", "not quite right", "please use your judgment")


def _normalize(value: object) -> str:
    return " ".join(TOKEN_RE.findall(str(value).lower()))


def _structured_terms(value: object, include_words: bool = True) -> set[str]:
    values = value if isinstance(value, list) else [value]
    result: set[str] = set()
    for item in values:
        normalized = _normalize(item)
        if normalized and len(normalized) <= 40 and len(normalized.split()) <= 4:
            result.add(normalized)
            if include_words:
                result.update(word for word in normalized.split() if len(word) >= 3)
    return result


@dataclass
class CatalogVocabulary:
    categories: set[str] = field(default_factory=set)
    brands: set[str] = field(default_factory=set)
    colors: set[str] = field(default_factory=lambda: set(COMMON_COLORS))
    materials: set[str] = field(default_factory=lambda: set(COMMON_MATERIALS))

    def add_product(self, product: dict) -> None:
        for category in product.get("categories") or []:
            for part in str(category).split(","):
                normalized = _normalize(part)
                if normalized and normalized not in GENERIC_CATEGORIES:
                    self.categories.add(normalized)
        store = _normalize(product.get("store") or "")
        if store and len(store) <= 60 and len(store.split()) <= 6:
            self.brands.add(store)
        details = product.get("details") or {}
        if not isinstance(details, dict):
            return
        for key, value in details.items():
            normalized_key = _normalize(key)
            if "manufacturer" in normalized_key or "brand" in normalized_key:
                self.brands.update(_structured_terms(value, include_words=False))
            elif "color" in normalized_key or "colour" in normalized_key:
                self.colors.update(_structured_terms(value))
            elif "material" in normalized_key or "fabric" in normalized_key:
                self.materials.update(_structured_terms(value))

    def values_for(self, attribute: str) -> set[str]:
        return {
            "category": self.categories,
            "brand": self.brands,
            "color": self.colors,
            "material": self.materials,
        }.get(attribute, set())


def _find_vocabulary_terms(text: str, vocabulary: set[str]) -> list[str]:
    normalized = _normalize(text)
    words = normalized.split()
    matches: list[str] = []
    occupied: set[int] = set()
    for width in range(min(6, len(words)), 0, -1):
        for start in range(len(words) - width + 1):
            positions = set(range(start, start + width))
            if positions & occupied:
                continue
            candidate = " ".join(words[start:start + width])
            if candidate in vocabulary:
                matches.append(candidate)
                occupied.update(positions)
    return matches


def _remove_terms(text: str, values: list[str]) -> str:
    result = text
    for value in values:
        result = re.sub(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", " ", result, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", result).strip(" ,:!?")


@dataclass
class SessionState:
    user_profile: dict
    vocabulary: CatalogVocabulary = field(default_factory=CatalogVocabulary)
    messages: list[str] = field(default_factory=list)
    active_constraints: dict[str, list[str]] = field(default_factory=dict)
    negative_constraints: dict[str, list[str]] = field(default_factory=dict)
    constraint_updated_at: dict[str, int] = field(default_factory=dict)
    _message_number: int = 0

    def add_message(self, message: str, turn: int | None = None) -> None:
        self.messages.append(message)
        self._message_number += 1
        update_turn = turn if turn is not None else self._message_number

        removal = REMOVE_ATTRIBUTE_RE.search(message)
        if removal:
            self._remove_attribute(removal.group("attribute"), update_turn)
            return

        named_override = NAMED_OVERRIDE_RE.search(message)
        override = named_override or OVERRIDE_RE.search(message)
        if override:
            old = named_override.group("old") if named_override else ""
            self._replace_constraints(override.group("new"), old, update_turn)
            return

        message = BROWSING_FILLER_RE.sub("", message)
        negative_spans: list[str] = []
        for match in NEGATION_RE.finditer(message):
            value = match.group("value").strip(" ,:!?\t\r\n")
            if value:
                self._negate_constraints(value, update_turn)
                negative_spans.append(match.group(0))
        for span in negative_spans:
            message = message.replace(span, " ")
        self._add_constraints(message, update_turn)

    def _parts(self, constraint: str) -> list[tuple[str, str]]:
        has_category = bool(re.search(r"\b(?:looking for|need|want)\b", constraint, re.IGNORECASE))
        remainder = constraint
        recognized: list[tuple[str, str]] = []
        for attribute in ("color", "material", "brand"):
            matches = _find_vocabulary_terms(remainder, self.vocabulary.values_for(attribute))
            if matches:
                recognized.extend((attribute, value) for value in matches)
                remainder = _remove_terms(remainder, matches)
        size_matches = [match.group(0) for match in SIZE_RE.finditer(remainder)]
        if size_matches:
            recognized.extend(("size", _normalize(value)) for value in size_matches)
            remainder = SIZE_RE.sub(" ", remainder)
        budget_matches = [match.group(0) for match in BUDGET_RE.finditer(remainder)]
        if budget_matches:
            recognized.extend(("budget", _normalize(value)) for value in budget_matches)
            remainder = BUDGET_RE.sub(" ", remainder)

        remainder = re.sub(r"\s+", " ", remainder).strip(" ,:!?")
        parts: list[tuple[str, str]] = []
        if has_category and remainder:
            parts.append(("category", remainder))
        elif remainder:
            cleaned = re.sub(
                r"^(?:preferably|prefer|brand|color|colour|material)\s*:?[ ]*", "", remainder,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned:
                category_matches = _find_vocabulary_terms(cleaned, self.vocabulary.categories)
                attribute = "category" if category_matches else "other"
                parts.append((attribute, cleaned))
        parts.extend(recognized)
        return parts

    def _add_constraints(self, message: str, turn: int) -> None:
        for clause in re.split(r"[.;]", message):
            constraint = CONSTRAINT_PREFIX_RE.sub("", clause).strip(" ,:!?\t\r\n")
            lowered = constraint.lower()
            if not constraint or "don't have" in lowered or any(p in lowered for p in NON_CONSTRAINT_PHRASES):
                continue
            for attribute, value in self._parts(constraint):
                self._set_constraint(attribute, value, turn)

    def _set_constraint(self, attribute: str, value: str, turn: int) -> None:
        normalized = _normalize(value)
        if not normalized:
            return
        if attribute != "other":
            self.active_constraints[attribute] = [normalized]
        else:
            values = self.active_constraints.setdefault(attribute, [])
            if normalized not in values:
                values.append(normalized)
        negatives = self.negative_constraints.get(attribute, [])
        self.negative_constraints[attribute] = [item for item in negatives if item != normalized]
        if not self.negative_constraints[attribute]:
            self.negative_constraints.pop(attribute, None)
        self.constraint_updated_at[attribute] = turn

    def _replace_constraints(self, new_value: str, old_value: str, turn: int) -> None:
        parts = self._parts(new_value.strip(" ,:!?\t\r\n"))
        if len(parts) == 1 and parts[0][0] == "other" and old_value:
            old_parts = self._parts(old_value)
            if old_parts:
                parts[0] = (old_parts[-1][0], parts[0][1])
        for attribute, value in parts:
            self.active_constraints.pop(attribute, None)
            self._set_constraint(attribute, value, turn)

    def _negate_constraints(self, value: str, turn: int) -> None:
        parts = self._parts(value)
        for attribute, normalized_value in parts:
            normalized = _normalize(normalized_value)
            current = self.active_constraints.get(attribute, [])
            remaining = [item for item in current if item != normalized]
            if remaining:
                self.active_constraints[attribute] = remaining
            else:
                self.active_constraints.pop(attribute, None)
            negatives = self.negative_constraints.setdefault(attribute, [])
            if normalized not in negatives:
                negatives.append(normalized)
            self.constraint_updated_at[attribute] = turn

    def _remove_attribute(self, attribute: str, turn: int) -> None:
        attribute = "color" if attribute.lower() == "colour" else attribute.lower()
        self.active_constraints.pop(attribute, None)
        self.negative_constraints.pop(attribute, None)
        self.constraint_updated_at[attribute] = turn

    def retrieval_context(self) -> str:
        attributes = sorted(
            self.active_constraints,
            key=lambda attribute: self.constraint_updated_at.get(attribute, 0),
            reverse=True,
        )
        return " ".join(value for attribute in attributes for value in self.active_constraints[attribute])
