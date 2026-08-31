from __future__ import annotations
from pathlib import Path

from src.candidate_reranker import rerank_candidates
from src.data_parser import load_catalog
from src.dual_index import DualIndex, QueryEmbedder
from src.intent_routing import DEFAULT_ROUTING_CONFIG, IntentDecision, RoutingConfig
from src.proactive_guidance import choose_clarification
from starter.session_state import CatalogVocabulary, SessionState


class Agent:
    """State-aware routed retrieval, deterministic reranking, and guidance."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        embeddings_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        query_embedder: QueryEmbedder | None = None,
        enable_intent_routing: bool = True,
        routing_config: RoutingConfig = DEFAULT_ROUTING_CONFIG,
        enable_reranking: bool = True,
        rerank_candidate_pool_size: int = 100,
    ) -> None:
        if rerank_candidate_pool_size <= 0:
            raise ValueError("rerank_candidate_pool_size must be positive")
        self.catalog_path = Path(catalog_path)
        self.enable_intent_routing = enable_intent_routing
        self.routing_config = routing_config
        self.enable_reranking = enable_reranking
        self.rerank_candidate_pool_size = rerank_candidate_pool_size
        artifact_directory = self.catalog_path.parent
        if embeddings_path is None:
            embeddings_path = artifact_directory / "catalog_embeddings.npy"
        if manifest_path is None:
            manifest_path = artifact_directory / "catalog_embeddings.json"
        self.catalog = load_catalog(self.catalog_path)
        self._sessions: dict[str, SessionState] = {}
        self._intent_decisions: dict[str, IntentDecision] = {}
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
        self._intent_decisions.pop(session_id, None)
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
        candidate_limit = top_k
        if self.enable_reranking:
            candidate_limit = min(
                max(top_k, self.rerank_candidate_pool_size),
                len(self.catalog.items),
            )
        if self.enable_intent_routing:
            retrieval_results, decision = self.retriever.search_intent_aware(
                state.retrieval_context(),
                user_message=user_message,
                active_constraints=state.active_constraints,
                known_brands=self.vocabulary.values_for("brand"),
                known_categories=self.vocabulary.values_for("category"),
                top_k=candidate_limit,
                max_price=state.budget_ceiling(),
                routing_config=self.routing_config,
            )
            self._intent_decisions[session_id] = decision
        else:
            retrieval_results = self.retriever.search(
                state.retrieval_context(),
                top_k=candidate_limit,
                max_price=state.budget_ceiling(),
            )
        if self.enable_reranking:
            results = rerank_candidates(
                retrieval_results,
                self.catalog.items_by_asin,
                active_constraints=state.active_constraints,
                negative_constraints=state.negative_constraints,
                constraint_updated_at=state.constraint_updated_at,
                top_k=top_k,
            )
        else:
            results = retrieval_results[:top_k]
        recommendations = [{"parent_asin": parent_asin} for parent_asin, _ in results]
        candidates = [
            self.catalog.items_by_asin[parent_asin]
            for parent_asin, _ in results
            if parent_asin in self.catalog.items_by_asin
        ]
        guidance = choose_clarification(
            candidates,
            # Preserve the established retrieval-confidence policy while using the
            # reranked products to choose the most useful question.
            candidate_scores=[score for _, score in retrieval_results[:top_k]],
            force_clarification=state.guidance_requested,
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
