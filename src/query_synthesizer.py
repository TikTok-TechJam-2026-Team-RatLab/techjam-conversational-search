from __future__ import annotations

import re
from typing import Sequence

from src.data_parser import CatalogItem
from src.dialogue_state import DialogueState
from src.intent_router import IntentType


class QuerySynthesizer:
    """Synthesizes high-precision weighted search queries and applies constraint re-ranking."""

    @staticmethod
    def synthesize_query(state: DialogueState) -> str:
        """Constructs an optimized search query string from current dialogue state."""
        query_parts: list[str] = []

        # 1. Primary Category (highest anchor weight)
        if state.categories:
            cat = state.categories[0]
            query_parts.append(f"{cat} {cat}")

        # 2. Active Disclosed Constraints
        for constraint in state.disclosed_constraints:
            clean_c = re.sub(r"^(?:color|budget|material|style|use_case)\s*:\s*", "", constraint, flags=re.I).strip()
            # If constraint contains purged terms, ignore it
            if not any(pt in clean_c.lower() for pt in state.purged_terms):
                query_parts.append(clean_c)

        # 3. Active Slot Keywords
        for slot_type, values in state.active_slots.items():
            for v in values:
                clean_v = re.sub(r"^(?:color|budget|material|style|use_case)\s*:\s*", "", v, flags=re.I).strip()
                if clean_v.lower() not in state.purged_terms:
                    # Boost materials and colors
                    if slot_type in ("material", "color"):
                        query_parts.append(f"{clean_v} {clean_v}")
                    else:
                        query_parts.append(clean_v)

        # 4. If no slots or categories extracted, fallback to recent valid turn message
        if not query_parts:
            for turn in reversed(state.turns):
                if turn.user_message and not any(p in turn.user_message.lower() for p in ("actually", "ignore", "don't have")):
                    query_parts.append(turn.user_message)
                    break

        full_query = " ".join(query_parts).strip()
        
        # Clean multiple spaces
        full_query = re.sub(r"\s+", " ", full_query)
        return full_query

    @staticmethod
    def rerank_candidates(
        candidates: list[tuple[str, float]],
        state: DialogueState,
        items_by_asin: dict[str, CatalogItem],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Reranks retrieved candidate items based on hard constraint matching and slot alignment."""
        if not candidates:
            return []

        # Gather target attributes from state
        target_materials = {m.lower() for m in state.active_slots.get("material", []) if m.lower() not in state.purged_terms}
        target_colors = {c.lower() for c in state.active_slots.get("color", []) if c.lower() not in state.purged_terms}
        target_use_cases = {u.lower() for u in state.active_slots.get("use_case", []) if u.lower() not in state.purged_terms}

        scored_items: list[tuple[str, float]] = []

        for rank, (asin, base_score) in enumerate(candidates):
            item = items_by_asin.get(asin)
            if item is None:
                scored_items.append((asin, base_score))
                continue

            multiplier = 1.0
            bonus = 0.0

            # Match materials
            if target_materials:
                item_materials_lower = {m.lower() for m in item.materials}
                item_text_lower = item.dense_text.lower()
                if any(tm in item_materials_lower or tm in item_text_lower for tm in target_materials):
                    bonus += 0.40
                elif any(pt in item_materials_lower for pt in state.purged_terms):
                    multiplier *= 0.2

            # Match colors
            if target_colors:
                item_colors_lower = {c.lower() for c in item.colors}
                item_text_lower = item.dense_text.lower()
                if any(tc in item_colors_lower or tc in item_text_lower for tc in target_colors):
                    bonus += 0.35
                elif any(pt in item_colors_lower for pt in state.purged_terms):
                    multiplier *= 0.2

            # Match use cases
            if target_use_cases:
                item_text_lower = item.dense_text.lower()
                if any(uc in item_text_lower for uc in target_use_cases):
                    bonus += 0.25

            # Match exact disclosed constraints phrases in dense text
            for dc in state.disclosed_constraints:
                clean_dc = re.sub(r"^(?:color|budget|material|style|use_case)\s*:\s*", "", dc, flags=re.I).strip().lower()
                if len(clean_dc) >= 3:
                    if clean_dc in item.dense_text.lower():
                        bonus += 0.60
                    elif any(w in item.dense_text.lower() for w in clean_dc.split() if len(w) > 3):
                        bonus += 0.30

            # Match primary category words
            if state.categories:
                cat_words = [w.lower() for w in state.categories[0].split() if len(w) > 2]
                if cat_words:
                    matched = sum(1 for w in cat_words if w in item.dense_text.lower())
                    bonus += 0.35 * (matched / len(cat_words))

            final_score = (base_score + bonus) * multiplier
            scored_items.append((asin, final_score))

        # Re-sort candidates by final boosted score
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:top_k]

