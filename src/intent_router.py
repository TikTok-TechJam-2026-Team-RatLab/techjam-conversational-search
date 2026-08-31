from __future__ import annotations

from enum import Enum
import re


class IntentType(str, Enum):
    BUYING = "buying"
    BROWSING = "browsing"
    INTENT_OVERRIDE = "intent_override"
    BOUNDARY = "boundary"
    CONSTRAINT_UPDATE = "constraint_update"
    GENERAL = "general"


# Regex patterns for detecting intent types
OVERRIDE_PATTERNS = [
    re.compile(r"\b(?:actually|never\s*mind|instead|forget\s+(?:about\s+)?that|ignore\s+(?:my\s+)?earlier|change\s+of\s+mind|changed\s+my\s+mind)\b", re.I),
    re.compile(r"\b(?:what\s+i\s+need\s+is|what\s+i\s+really\s+want\s+is|let['']?s\s+switch\s+to|switch\s+to|rather\s+have)\b", re.I),
    re.compile(r"\b(?:ignore\s+my\s+earlier\s+preference|prioritize\s+the\s+target\s+requirements)\b", re.I),
]

BOUNDARY_PATTERNS = [
    re.compile(r"\b(?:don['']?t\s+have\s+(?:a|any|an\s+additional)\s+preference|no\s+preference|doesn['']?t\s+matter|either\s+is\s+fine|any\s+(?:is|will\s+do))\b", re.I),
    re.compile(r"\b(?:use\s+your\s+judgment|up\s+to\s+you|whatever\s+you\s+think)\b", re.I),
]

BROWSING_PATTERNS = [
    re.compile(r"\b(?:still\s+exploring|just\s+browsing|looking\s+around|exploring\s+options|open\s+to\s+ideas|not\s+sure\s+yet|any\s+recommendations)\b", re.I),
]

BUYING_PATTERNS = [
    re.compile(r"\b(?:key\s+requirement\s+is|must\s+have|specifically\s+looking\s+for|i\s+need|need\s+a|ready\s+to\s+buy|looking\s+to\s+buy)\b", re.I),
    re.compile(r"\b(?:what\s+matters\s+is|for\s+that,\s+what\s+matters\s+is)\b", re.I),
]

CONSTRAINT_REVEAL_PATTERNS = [
    re.compile(r"\b(?:for\s+that,\s+what\s+matters\s+is|what\s+matters\s+is|specifically|preference\s+is)\s*:\s*(.+)", re.I),
]


class IntentRouter:
    """Classifies user turn intent into Buying, Browsing, Intent Override, Boundary, or Constraint Reveal."""

    @staticmethod
    def classify(message: str, turn: int = 1) -> tuple[IntentType, float]:
        """Classifies the given user message and returns (IntentType, confidence)."""
        if not message or not message.strip():
            return IntentType.GENERAL, 0.5

        text = message.strip()

        # 1. Check for explicit intent override first (highest priority)
        for pat in OVERRIDE_PATTERNS:
            if pat.search(text):
                return IntentType.INTENT_OVERRIDE, 0.95

        # 2. Check for boundary / indifference response
        for pat in BOUNDARY_PATTERNS:
            if pat.search(text):
                return IntentType.BOUNDARY, 0.95

        # 3. Check for constraint disclosure (reply to ask_attribute)
        for pat in CONSTRAINT_REVEAL_PATTERNS:
            if pat.search(text):
                return IntentType.CONSTRAINT_UPDATE, 0.90

        # 4. Check for browsing indicators
        for pat in BROWSING_PATTERNS:
            if pat.search(text):
                return IntentType.BROWSING, 0.90

        # 5. Check for buying indicators
        for pat in BUYING_PATTERNS:
            if pat.search(text):
                return IntentType.BUYING, 0.90

        # Fallback heuristics
        if turn == 1:
            # If initial message mentions constraints or specific attributes, lean BUYING; otherwise BROWSING
            if any(k in text.lower() for k in ["requirement", "$", "budget", "size", "color:", "material:"]):
                return IntentType.BUYING, 0.75
            return IntentType.BROWSING, 0.60
        else:
            return IntentType.GENERAL, 0.70

    @staticmethod
    def extract_override_payload(message: str) -> str | None:
        """Extracts the new requirement substring from an override message."""
        # E.g. "Actually, ignore my earlier preference. What I need is: 100% cotton."
        m = re.search(r"(?:what\s+i\s+need\s+is|what\s+i\s+want\s+is|switch\s+to|prioritize)\s*:\s*(.+)", message, re.I)
        if m:
            return m.group(1).strip(" .;")
        
        # E.g. "Actually, ignore my earlier preference. Please find cotton t-shirts."
        sentences = [s.strip() for s in re.split(r"[.!?]", message) if s.strip()]
        for s in reversed(sentences):
            if not any(p.search(s) for p in OVERRIDE_PATTERNS):
                return s
        return None

