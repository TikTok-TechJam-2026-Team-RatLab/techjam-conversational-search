import re
from dataclasses import dataclass, field


NAMED_OVERRIDE_RE = re.compile(
    r"\b(?:replace|switch from)\s+(?P<old>.+?)\s+(?:with|to)\s+(?P<new>.+?)[.!?]*$",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead)\b.*?\b(?:ignore|replace|forget)\b.*?"
    r"(?:what i need is|(?:i (?:need|want|prefer))|instead)\s*:\s*(?P<new>.+)",
    re.IGNORECASE,
)
CONSTRAINT_PREFIX_RE = re.compile(
    r"^(?:for that,?\s*)?(?:what matters is|a key requirement is)\s*:\s*",
    re.IGNORECASE,
)
BROWSING_FILLER_RE = re.compile(r",?\s*but i(?:'m| am) still exploring\b", re.IGNORECASE)
COLORS = {
    "black", "blue", "brown", "gold", "gray", "green", "grey", "orange",
    "pink", "purple", "red", "silver", "white", "yellow",
}
MATERIALS = {"cotton", "denim", "leather", "linen", "polyester", "silk", "wool"}
COLOR_RE = re.compile(r"\b(?:" + "|".join(sorted(COLORS)) + r")\b", re.IGNORECASE)
MATERIAL_RE = re.compile(r"\b(?:" + "|".join(sorted(MATERIALS)) + r")\b", re.IGNORECASE)
SIZE_RE = re.compile(r"\bsize\s*[:#-]?\s*[a-z0-9]+\b", re.IGNORECASE)
BUDGET_RE = re.compile(
    r"\b(?:budget\s*(?:is|of|:)?\s*\$?\d+(?:\.\d+)?|under\s+\$?\d+(?:\.\d+)?)\b"
    r"|\$\d+(?:\.\d+)?",
    re.IGNORECASE,
)
NON_CONSTRAINT_PHRASES = (
    "ask me about",
    "not quite right",
    "please use your judgment",
)


def _constraint_attribute(constraint: str) -> str:
    lowered = constraint.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    if re.search(r"\b(?:looking for|need|want)\b", lowered):
        return "category"
    if words & COLORS or "color" in words or "colour" in words:
        return "color"
    if re.search(r"\bsize\s*[:#-]?\s*[a-z0-9]+\b", lowered):
        return "size"
    if words & MATERIALS or "material" in words:
        return "material"
    if "budget" in words or re.search(r"(?:\$|under\s+)\d", lowered):
        return "budget"
    if "brand" in words:
        return "brand"
    if words & {"men", "mens", "women", "womens", "style", "fit"}:
        return "style"
    return "other"


def _constraint_parts(constraint: str) -> list[tuple[str, str]]:
    """Split recognized attributes out of a potentially mixed clause."""
    remainder = constraint
    recognized: list[tuple[str, str]] = []
    for attribute, pattern in (
        ("color", COLOR_RE),
        ("material", MATERIAL_RE),
        ("size", SIZE_RE),
        ("budget", BUDGET_RE),
    ):
        matches = [match.group(0).strip() for match in pattern.finditer(remainder)]
        if matches:
            recognized.extend((attribute, value) for value in matches)
            remainder = pattern.sub(" ", remainder)

    remainder = re.sub(r"\s+", " ", remainder).strip(" ,:!?")
    has_category = bool(re.search(r"\b(?:looking for|need|want)\b", constraint, re.IGNORECASE))
    parts: list[tuple[str, str]] = []
    if has_category and remainder:
        parts.append(("category", remainder))
    elif remainder:
        cleaned = re.sub(
            r"^(?:preferably|prefer|color|colour|material)\s*:?[ ]*",
            "",
            remainder,
            flags=re.IGNORECASE,
        ).strip()
        if cleaned:
            parts.append((_constraint_attribute(cleaned), cleaned))
    parts.extend(recognized)
    return parts


@dataclass
class SessionState:
    user_profile: dict
    messages: list[str] = field(default_factory=list)
    active_constraints: dict[str, list[str]] = field(default_factory=dict)

    def add_message(self, message: str) -> None:
        self.messages.append(message)
        named_override = NAMED_OVERRIDE_RE.search(message)
        override = named_override or OVERRIDE_RE.search(message)
        if override:
            new_constraint = override.group("new").strip(" ,:!?\t\r\n")
            parts = _constraint_parts(new_constraint)
            if len(parts) == 1 and parts[0][0] == "other" and named_override:
                parts[0] = (_constraint_attribute(named_override.group("old")), parts[0][1])
            for attribute, constraint in parts:
                self.active_constraints.pop(attribute, None)
                self._store_constraint(constraint, attribute)
            return
        self._add_constraints(message)

    def _add_constraints(self, message: str) -> None:
        """Add independently replaceable clauses from a customer message."""
        message = BROWSING_FILLER_RE.sub("", message)
        for clause in re.split(r"[.;]", message):
            constraint = CONSTRAINT_PREFIX_RE.sub("", clause).strip(" ,:!?\t\r\n")
            lowered = constraint.lower()
            if (
                not constraint
                or "don't have" in lowered
                or any(phrase in lowered for phrase in NON_CONSTRAINT_PHRASES)
            ):
                continue
            for attribute, value in _constraint_parts(constraint):
                self._store_constraint(value, attribute)

    def _store_constraint(self, constraint: str, attribute: str) -> None:
        values = self.active_constraints.setdefault(attribute, [])
        if constraint not in values:
            values.append(constraint)

    def retrieval_context(self) -> str:
        return " ".join(
            constraint
            for constraints in self.active_constraints.values()
            for constraint in constraints
        )
