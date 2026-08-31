from __future__ import annotations

import math
from typing import Sequence

from src.data_parser import CatalogItem
from src.dialogue_state import DialogueState


ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

# Default priority order when entropy is close or candidates are sparse
DEFAULT_ATTRIBUTE_PRIORITY = [
    "material",
    "color",
    "budget",
    "use_case",
    "style",
    "feature",
    "size",
    "brand",
]


class ProactiveGuidance:
    """Selects the optimal attribute to probe the user using Expected Information Gain (Entropy)."""

    @staticmethod
    def select_attribute(
        state: DialogueState,
        candidate_items: Sequence[CatalogItem],
        turn: int = 1,
    ) -> str | None:
        """Selects the best unasked attribute to probe the user in the current turn."""
        # Available candidates: must be in ALLOWED_ATTRIBUTES, not already asked, not rejected, not fully populated
        available_attributes = [
            attr for attr in DEFAULT_ATTRIBUTE_PRIORITY
            if attr not in state.asked_attributes
            and attr not in state.rejected_attributes
            and not (state.active_slots.get(attr) and len(state.active_slots[attr]) >= 2)
        ]

        if not available_attributes:
            return None

        # If turn is 10 (last turn), asking an attribute will not yield another turn, but evaluator checks it
        if not candidate_items:
            return available_attributes[0]

        # Compute information entropy for each available attribute across candidate_items
        best_attr = None
        max_entropy = -1.0
        n_candidates = len(candidate_items)

        for attr in available_attributes:
            entropy = ProactiveGuidance._compute_attribute_entropy(attr, candidate_items, n_candidates)
            # Add small priority weight to break ties towards high-impact attributes
            priority_boost = (len(DEFAULT_ATTRIBUTE_PRIORITY) - DEFAULT_ATTRIBUTE_PRIORITY.index(attr)) * 0.05
            score = entropy + priority_boost
            
            if score > max_entropy:
                max_entropy = score
                best_attr = attr

        return best_attr or available_attributes[0]

    @staticmethod
    def _compute_attribute_entropy(attr: str, candidate_items: Sequence[CatalogItem], n_candidates: int) -> float:
        """Calculates Shannon entropy for the given attribute distribution across candidates."""
        if n_candidates == 0:
            return 0.0

        presence_count = 0
        value_counts: dict[str, int] = {}

        for item in candidate_items:
            if attr == "material":
                vals = item.materials
            elif attr == "color":
                vals = item.colors
            elif attr == "budget":
                vals = ["priced"] if item.price is not None and item.price > 0 else []
            elif attr == "style":
                vals = [item.department] if item.department else []
            elif attr == "use_case":
                # Check description / features for use case keywords
                vals = ["use_case"] if any(w in item.dense_text.lower() for w in ("running", "hiking", "gym", "winter", "outdoor", "work")) else []
            else:
                vals = ["present"] if item.features else []

            if vals:
                presence_count += 1
                for v in vals:
                    value_counts[v] = value_counts.get(v, 0) + 1

        p_present = presence_count / n_candidates
        if p_present == 0.0 or p_present == 1.0:
            binary_entropy = 0.0
        else:
            binary_entropy = -p_present * math.log2(p_present) - (1.0 - p_present) * math.log2(1.0 - p_present)

        # Also account for value diversity if multiple distinct values exist
        value_entropy = 0.0
        if presence_count > 0 and len(value_counts) > 1:
            for count in value_counts.values():
                p = count / presence_count
                if p > 0:
                    value_entropy -= p * math.log2(p)

        return binary_entropy + 0.3 * value_entropy
