from __future__ import annotations
from starter.session_state import CatalogVocabulary, SessionState

import json
import math
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# Category and transactional numeric constraints are treated as hard constraints and
# remain active until the user changes or removes them. Descriptive preferences are
# allowed to fade so stale tastes do not dominate later turns.
HARD_CONSTRAINT_ATTRIBUTES = frozenset({"category", "size", "budget"})
SLOT_DECAY_LAMBDA = 0.35
SOFT_SLOT_MIN_WEIGHT = 0.50


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _slot_weight(attribute: str, updated_at: int, current_turn: int) -> float:
    """Return the retrieval weight for a slot at the current turn.

    Hard constraints do not decay. Soft preferences use exponential decay based on
    the number of turns since they were last added or updated.
    """
    if attribute in HARD_CONSTRAINT_ATTRIBUTES:
        return 1.0
    age = max(0, current_turn - updated_at)
    return math.exp(-SLOT_DECAY_LAMBDA * age)


def _active_retrieval_terms(state: SessionState, current_turn: int) -> list[str]:
    """Build retrieval terms while selectively decaying stale soft preferences."""
    terms: list[str] = []
    for attribute, values in state.active_constraints.items():
        updated_at = state.constraint_updated_at.get(attribute, current_turn)
        weight = _slot_weight(attribute, updated_at, current_turn)
        if attribute not in HARD_CONSTRAINT_ATTRIBUTES and weight < SOFT_SLOT_MIN_WEIGHT:
            continue
        for value in values:
            terms.extend(_terms(value))
    return list(dict.fromkeys(terms))[:40]


def _negative_retrieval_terms(state: SessionState) -> list[str]:
    """Return normalized terms that must be excluded from lexical retrieval."""
    terms: list[str] = []
    for values in state.negative_constraints.values():
        for value in values:
            terms.extend(_terms(value))
    return list(dict.fromkeys(terms))


def _fts_expression(positive_terms: list[str], negative_terms: list[str]) -> str:
    if not positive_terms:
        return ""
    expression = " OR ".join(f'"{term}"' for term in positive_terms)
    for term in negative_terms:
        expression = f'({expression}) NOT "{term}"'
    return expression


class Agent:
    """BM25 retrieval baseline with per-session dialogue state and no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self.vocabulary = CatalogVocabulary()
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                self.vocabulary.add_product(product)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = SessionState(
            user_profile=dict(user_profile),
            vocabulary=self.vocabulary,
        )

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        state = self._sessions[session_id]
        state.add_message(user_message, turn)

        positive_terms = _active_retrieval_terms(state, turn)
        negative_terms = _negative_retrieval_terms(state)
        expression = _fts_expression(positive_terms, negative_terms)

        if not expression:
            recommendations: list[dict] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
