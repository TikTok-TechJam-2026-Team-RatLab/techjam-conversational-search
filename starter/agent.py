from __future__ import annotations

import json
from pathlib import Path
import sys

_root = str(Path(__file__).resolve().parent.parent)
_libs = str(Path(__file__).resolve().parent.parent / 'libs')

if sys.version_info[:2] == (3, 12):
    sys.path = [_libs, _root] + [p for p in sys.path if 'Python313' not in p and 'Roaming' not in p and p not in (_libs, _root)]
else:
    if _root not in sys.path:
        sys.path.insert(0, _root)

from src.dual_index import DualIndex
from src.dialogue_state import DialogueState
from src.query_synthesizer import QuerySynthesizer
from src.proactive_guidance import ProactiveGuidance


class Agent:
    def __init__(
        self,
        catalog_path: str | Path = 'data/catalog.jsonl',
        embeddings_path: str | Path = 'data/catalog_embeddings.npy',
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.embeddings_path = Path(embeddings_path)
        self.dual_index = DualIndex(
            catalog_path=self.catalog_path,
            embeddings_path=self.embeddings_path,
            load_dense=True,
        )
        self._states: dict[str, DialogueState] = {}

    @property
    def _sessions(self) -> dict[str, DialogueState]:
        return self._states

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._states[session_id] = DialogueState(
            session_id=session_id,
            user_profile=dict(user_profile),
        )

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int = 10,
    ) -> dict:
        if session_id not in self._states:
            raise RuntimeError('reset must be called before respond')

        state = self._states[session_id]
        
        # 1. Update Dialogue State & Extract Slots / Intent
        state.add_turn(turn, user_message)

        # 2. Synthesize High-Precision Search Query
        search_query = QuerySynthesizer.synthesize_query(state)

        # 3. Retrieve Candidate Pool (top-80 for high recall and reranking)
        fetch_k = max(80, top_k * 8)
        if self.dual_index.embeddings is not None:
            raw_results = self.dual_index.search_hybrid_adaptive(
                search_query,
                intent_type=state.current_intent,
                top_k=fetch_k,
            )
        else:
            raw_results = self.dual_index.search_sparse(search_query, top_k=fetch_k)

        # 4. Rerank Candidates with Slot Matching & Purged Term Penalties
        reranked = QuerySynthesizer.rerank_candidates(
            raw_results,
            state=state,
            items_by_asin=self.dual_index.items_by_asin,
            top_k=top_k,
        )

        # 5. Gather candidate items for Proactive Attribute Probing
        candidate_items = [
            self.dual_index.items_by_asin[asin]
            for asin, _ in raw_results[:30]
            if asin in self.dual_index.items_by_asin
        ]

        # 6. Select Optimal ask_attribute via Expected Information Gain
        ask_attr = ProactiveGuidance.select_attribute(state, candidate_items, turn=turn)
        state.record_agent_action(ask_attr)

        recommendations = [{'parent_asin': asin} for asin, _ in reranked[:top_k]]

        return {
            'message': 'Here are the best matching items based on your preferences.',
            'ask_attribute': ask_attr,
            'recommendations': recommendations,
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0},
        }
