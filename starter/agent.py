from __future__ import annotations
from pathlib import Path

from src.data_parser import load_catalog
from src.dual_index import DualIndex, QueryEmbedder
from src.proactive_guidance import choose_clarification
from starter.session_state import CatalogVocabulary, SessionState


class Agent:
    """State-aware sparse retrieval with validated optional dense-vector fusion."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        embeddings_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        query_embedder: QueryEmbedder | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        artifact_directory = self.catalog_path.parent
        if embeddings_path is None:
            embeddings_path = artifact_directory / "catalog_embeddings.npy"
        if manifest_path is None:
            manifest_path = artifact_directory / "catalog_embeddings.json"
        self.catalog = load_catalog(self.catalog_path)
        self._sessions: dict[str, SessionState] = {}
        self.vocabulary = CatalogVocabulary()
        for item in self.catalog.items:
            self.vocabulary.add_product(item.as_product())
        self.retriever = DualIndex(
            self.catalog,
            embeddings_path=embeddings_path,
            manifest_path=manifest_path,
            query_embedder=query_embedder,
        )

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
        results = self.retriever.search(state.retrieval_context(), top_k=top_k)
        recommendations = [{"parent_asin": parent_asin} for parent_asin, _ in results]
        candidates = [
            self.catalog.items_by_asin[parent_asin]
            for parent_asin, _ in results
            if parent_asin in self.catalog.items_by_asin
        ]
        guidance = choose_clarification(
            candidates,
            unavailable_attributes=(
                set(state.active_constraints)
                | state.asked_attributes
                | state.declined_attributes
            ),
        )
        if guidance.ask_attribute is not None:
            state.record_question(guidance.ask_attribute)
        return {
            "message": guidance.message,
            "ask_attribute": guidance.ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
