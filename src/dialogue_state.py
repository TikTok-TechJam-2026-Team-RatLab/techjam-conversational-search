from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from src.intent_router import IntentRouter, IntentType


MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "fabric", "fleece", "canvas", "denim", "suede", "linen", "cashmere", "mesh",
    "rubber", "synthetic", "down", "gore-tex", "modal", "bamboo", "velvet", "satin",
)

COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "navy", "beige", "khaki", "gold", "silver",
    "tan", "burgundy", "maroon", "teal", "olive", "coral", "turquoise", "charcoal",
)

STYLES = (
    "mens", "womens", "men", "women", "boys", "girls", "unisex", "fit", "slim",
    "loose", "relaxed", "sleeve", "long sleeve", "short sleeve", "sleeveless",
    "neck", "v-neck", "crew neck", "collar", "hooded", "zipper", "pullover",
    "casual", "formal", "vintage", "retro", "classic",
)

USE_CASES = (
    "running", "hiking", "gym", "workout", "winter", "outdoor", "athletic",
    "training", "walking", "work", "travel", "yoga", "swimming", "basketball",
    "soccer", "tennis", "camping", "skiing",
)

MATERIAL_RE = re.compile(rf"\b({'|'.join(MATERIALS)})\b", re.I)
COLOR_RE = re.compile(rf"\b({'|'.join(COLORS)})\b", re.I)
PRICE_RE = re.compile(r"(?:\$|under|around|<=|<|below|budget)\s*(\d+(?:\.\d+)?)", re.I)
SIZE_RE = re.compile(r"\b(?:size\s*(\d+(?:\.\d+)?|[xXsSmMlL]+)|wide|narrow|slim\s*fit)\b", re.I)
CATEGORY_RE = re.compile(r"(?:looking\s+for|search\s+for|need\s+(?:a|an)?|find\s+(?:me)?|exploring\s+(?:options\s+for)?|interested\s+in|browsing\s+for)\s+([^,.;]+)", re.I)


def classify_slot_type(text: str) -> str:
    """Classifies a constraint or term into one of the allowed attribute slots."""
    lowered = text.lower()
    if "budget" in lowered or PRICE_RE.search(lowered) or "$" in lowered:
        return "budget"
    if MATERIAL_RE.search(lowered):
        return "material"
    if COLOR_RE.search(lowered) or "color" in lowered:
        return "color"
    if SIZE_RE.search(lowered) or any(w in lowered for w in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(w in lowered for w in STYLES) or any(w in lowered for w in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(w in lowered for w in USE_CASES):
        return "use_case"
    if any(w in lowered for w in ("brand", "store", "manufacturer")):
        return "brand"
    return "feature"


@dataclass
class TurnRecord:
    turn_idx: int
    user_message: str
    intent: IntentType
    confidence: float
    asked_attribute: str | None = None
    extracted_slots: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class DialogueState:
    session_id: str
    user_profile: dict = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)
    
    # Tracked slots
    categories: list[str] = field(default_factory=list)
    active_slots: dict[str, list[str]] = field(default_factory=lambda: {
        "material": [],
        "color": [],
        "budget": [],
        "size": [],
        "style": [],
        "use_case": [],
        "brand": [],
        "feature": [],
    })
    
    # State tracking collections
    disclosed_constraints: set[str] = field(default_factory=set)
    asked_attributes: list[str] = field(default_factory=list)
    rejected_attributes: set[str] = field(default_factory=set)
    last_asked_attribute: str | None = None
    current_intent: IntentType = IntentType.GENERAL
    override_occurred: bool = False
    
    # Purged slots from previous turns (to prevent query contamination)
    purged_terms: set[str] = field(default_factory=set)

    @property
    def messages(self) -> list[str]:
        return [t.user_message for t in self.turns]

    def add_turn(self, turn_idx: int, user_message: str) -> IntentType:
        """Processes a new user turn, updates intent, tracks slots, and handles overrides."""
        intent, conf = IntentRouter.classify(user_message, turn=turn_idx)
        self.current_intent = intent
        
        extracted: dict[str, list[str]] = {}

        # 1. Handle Intent Override
        if intent == IntentType.INTENT_OVERRIDE:
            self.override_occurred = True
            new_payload = IntentRouter.extract_override_payload(user_message)
            
            # Record superseded active constraints and disclosed constraints to purge list
            for slot_name, values in self.active_slots.items():
                for v in values:
                    self.purged_terms.add(v.lower())
                self.active_slots[slot_name] = []

            for c in self.disclosed_constraints:
                for w in c.split():
                    w_clean = re.sub(r"[^a-zA-Z0-9]", "", w.lower())
                    if len(w_clean) > 2:
                        self.purged_terms.add(w_clean)
            
            self.disclosed_constraints.clear()
            self.asked_attributes.clear()
            self.rejected_attributes.clear()
            self.last_asked_attribute = None
            
            if new_payload:
                self.disclosed_constraints.add(new_payload)
                slot_type = classify_slot_type(new_payload)
                clean_payload = re.sub(r"^(?:color|budget|material|style|use_case)\s*:\s*", "", new_payload, flags=re.I).strip()
                self._add_slot_value(slot_type, clean_payload)
                extracted[slot_type] = [clean_payload]
                self._extract_slots_from_text(new_payload, extracted)

        # 2. Handle Boundary / Rejection of last asked attribute
        elif intent == IntentType.BOUNDARY or "don't have a preference" in user_message.lower() or "don't have an additional preference" in user_message.lower():
            if self.last_asked_attribute:
                self.rejected_attributes.add(self.last_asked_attribute)
            # Check if a specific attribute name is mentioned in the message
            for attr in ("material", "color", "size", "style", "brand", "budget", "feature", "use_case", "category"):
                if attr in user_message.lower():
                    self.rejected_attributes.add(attr)

        # 3. Handle Constraint Reveal ("For that, what matters is: ...")
        elif intent == IntentType.CONSTRAINT_UPDATE or "what matters is:" in user_message.lower():
            m = re.search(r"what\s+matters\s+is\s*:\s*(.+)", user_message, re.I)
            if m:
                payload = m.group(1).strip(" .")
                parts = [p.strip() for p in payload.split(";") if p.strip()]
                for part in parts:
                    self.disclosed_constraints.add(part)
                    st = classify_slot_type(part)
                    clean_val = re.sub(r"^(?:color|budget|material|style|use_case)\s*:\s*", "", part, flags=re.I).strip()
                    self._add_slot_value(st, clean_val)
                    extracted.setdefault(st, []).append(clean_val)
            self._extract_slots_from_text(user_message, extracted)

        # 4. Standard Turn / Turn 1 parsing
        else:
            # Extract category if present
            if turn_idx == 1 or not self.categories:
                cat_match = CATEGORY_RE.search(user_message)
                if cat_match:
                    cat_candidate = cat_match.group(1).strip(" .")
                    # Clean out trailing clauses
                    cat_candidate = re.split(r"\b(?:but|and|with|a key|which)\b", cat_candidate, flags=re.I)[0].strip()
                    if cat_candidate:
                        self.categories.append(cat_candidate)
                        extracted["category"] = [cat_candidate]

            # Extract any explicit slots
            self._extract_slots_from_text(user_message, extracted)

        turn_rec = TurnRecord(
            turn_idx=turn_idx,
            user_message=user_message,
            intent=intent,
            confidence=conf,
            asked_attribute=self.last_asked_attribute,
            extracted_slots=extracted,
        )
        self.turns.append(turn_rec)
        return intent

    def _add_slot_value(self, slot_type: str, value: str) -> None:
        clean_v = re.sub(r"^(?:color|budget|material|style|use_case)\s*:\s*", "", value, flags=re.I).strip()
        if not clean_v:
            return
        if slot_type in self.active_slots:
            if clean_v not in self.active_slots[slot_type]:
                self.active_slots[slot_type].append(clean_v)
        else:
            self.active_slots.setdefault("feature", []).append(clean_v)

    def _extract_slots_from_text(self, text: str, extracted: dict[str, list[str]]) -> None:
        """Extracts recognizable slot keywords from arbitrary text."""
        # Material
        for m in MATERIAL_RE.finditer(text):
            val = m.group(1).lower()
            if val not in self.purged_terms:
                self._add_slot_value("material", val)
                extracted.setdefault("material", []).append(val)

        # Color
        for c in COLOR_RE.finditer(text):
            val = c.group(1).lower()
            if val not in self.purged_terms:
                self._add_slot_value("color", val)
                extracted.setdefault("color", []).append(val)

        # Budget
        p_match = PRICE_RE.search(text)
        if p_match:
            val = f"budget: ${p_match.group(1)}"
            self._add_slot_value("budget", val)
            extracted.setdefault("budget", []).append(val)

        # Use case
        lowered = text.lower()
        for uc in USE_CASES:
            if re.search(rf"\b{uc}\b", lowered):
                self._add_slot_value("use_case", uc)
                extracted.setdefault("use_case", []).append(uc)

    def record_agent_action(self, asked_attribute: str | None) -> None:
        """Records what attribute the agent asked in this turn."""
        self.last_asked_attribute = asked_attribute
        if asked_attribute:
            if asked_attribute not in self.asked_attributes:
                self.asked_attributes.append(asked_attribute)
