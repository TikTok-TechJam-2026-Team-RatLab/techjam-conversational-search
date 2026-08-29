import re
from dataclasses import dataclass, field


OVERRIDE_RE = re.compile(
    r"\b(?:actually|instead)\b.*?\b(?:ignore|replace|forget)\b.*?"
    r"(?:what i need is|(?:i (?:need|want|prefer))|instead)\s*:\s*(.+)",
    re.IGNORECASE,
)
CONSTRAINT_PREFIX_RE = re.compile(
    r"^(?:for that,?\s*)?(?:what matters is|a key requirement is)\s*:\s*",
    re.IGNORECASE,
)
NON_CONSTRAINT_PHRASES = (
    "ask me about",
    "not quite right",
    "please use your judgment",
    "still exploring",
)


@dataclass
class SessionState:
    user_profile: dict
    messages: list[str] = field(default_factory=list)
    active_constraints: list[str] = field(default_factory=list)

    def add_message(self, message: str) -> None:
        self.messages.append(message)
        override = OVERRIDE_RE.search(message)
        if override:
            if self.active_constraints:
                self.active_constraints.pop()
            self._add_constraints(override.group(1))
            return
        self._add_constraints(message)

    def _add_constraints(self, message: str) -> None:
        """Add independently replaceable clauses from a customer message."""
        for clause in re.split(r"[.;]", message):
            constraint = CONSTRAINT_PREFIX_RE.sub("", clause).strip(" ,:!?\t\r\n")
            lowered = constraint.lower()
            if (
                not constraint
                or "don't have" in lowered
                or any(phrase in lowered for phrase in NON_CONSTRAINT_PHRASES)
            ):
                continue
            if constraint not in self.active_constraints:
                self.active_constraints.append(constraint)

    def retrieval_context(self) -> str:
        return " ".join(self.active_constraints)
