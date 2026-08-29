from __future__ import annotations

import json
from pathlib import Path
import sys

_root = str(Path(__file__).resolve().parent.parent)
_libs = str(Path(__file__).resolve().parent.parent / 'libs')
sys.path = [_libs, _root] + [p for p in sys.path if 'Python313' not in p and p not in (_libs, _root)]

from src.dual_index import DualIndex
from starter.session_state import SessionState


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
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(user_profile=dict(user_profile))

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError('reset must be called before respond')

        session = self._sessions[session_id]
        session.add_message(user_message)

        full_context = ' '.join(session.messages)

        if self.dual_index.embeddings is not None:
            results = self.dual_index.search_hybrid(full_context, top_k=top_k)
        else:
            results = self.dual_index.search_sparse(full_context, top_k=top_k)

        recommendations = [{'parent_asin': asin} for asin, _ in results]

        return {
            'message': 'Here are the closest matches I found.',
            'ask_attribute': None,
            'recommendations': recommendations,
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0},
        }
