from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import numpy as np
import bm25s

from src.data_parser import CatalogData, CatalogItem, load_catalog
from src.embedder import Embedder, DEFAULT_MODEL_NAME


def _build_sparse_corpus(items: list[CatalogItem]) -> list[str]:
    corpus = []
    for item in items:
        t = item.title
        c = ' '.join(item.categories)
        m = ' '.join(item.materials)
        col = ' '.join(item.colors)
        dt = item.dense_text
        text = f"{t} {t} {c} {m} {col} {dt}"
        corpus.append(text)

    return corpus



class DualIndex:
    """In-memory dual-track index combining sparse BM25 and dense vector retrieval."""

    def __init__(
        self,
        catalog_path: str | Path = 'data/catalog.jsonl',
        embeddings_path: str | Path = 'data/catalog_embeddings.npy',
        load_dense: bool = True,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.embeddings_path = Path(embeddings_path)
        self.model_name = model_name
        self._embedder: Embedder | None = None

        # 1. Ingest catalog
        self.catalog_data: CatalogData = load_catalog(self.catalog_path)
        self.asin_list = self.catalog_data.asin_list
        self.items_by_asin = self.catalog_data.items_by_asin
        self.asin_to_idx = self.catalog_data.asin_to_idx
        self.num_items = len(self.asin_list)

        # 2. Build sparse BM25 index
        items_ordered = [self.items_by_asin[asin] for asin in self.asin_list]
        sparse_corpus = _build_sparse_corpus(items_ordered)
        self.corpus_tokens = bm25s.tokenize(sparse_corpus, stopwords='en')
        self.bm25 = bm25s.BM25(corpus=self.asin_list)
        self.bm25.index(self.corpus_tokens)

        # 3. Dense Index (In-memory normalized matrix)
        self.embeddings: np.ndarray | None = None
        if load_dense and self.embeddings_path.exists():
            loaded_mat = np.load(str(self.embeddings_path))
            if loaded_mat.shape[0] == self.num_items:
                self.embeddings = loaded_mat
            elif self.num_items <= 100:
                self.embeddings = None
            else:
                raise ValueError(
                    f"Embeddings rows ({loaded_mat.shape[0]}) != catalog size ({self.num_items})"
                )

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(model_name=self.model_name)
        return self._embedder

    def get_item(self, parent_asin: str) -> CatalogItem | None:
        return self.items_by_asin.get(parent_asin)

    def search_sparse(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Executes fast sparse BM25 retrieval."""
        if not query or not query.strip():
            return []
        query_tokens = bm25s.tokenize(query.strip(), stopwords='en', show_progress=False)
        k = min(top_k, self.num_items)
        results, scores = self.bm25.retrieve(query_tokens, k=k, show_progress=False)
        
        top_asins = results[0]
        top_scores = scores[0]
        return [(str(asin), float(score)) for asin, score in zip(top_asins, top_scores) if score > 0]

    def search_dense(self, query_vec: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
        """Executes dense vector dot-product search against normalized embeddings."""
        if self.embeddings is None:
            return []
        if query_vec is None or len(query_vec) == 0:
            return []

        k = min(top_k, self.num_items)
        scores = self.embeddings @ query_vec

        partition_idx = np.argpartition(-scores, k)[:k]
        sorted_idx = partition_idx[np.argsort(-scores[partition_idx])]

        return [(self.asin_list[i], float(scores[i])) for i in sorted_idx]


    def search_hybrid(
        self,
        query: str,
        query_vec: np.ndarray | None = None,
        top_k: int = 50,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
        rrf_k: float = 60.0,
    ) -> list[tuple[str, float]]:
        """Executes hybrid search combining sparse BM25 and dense vector similarity."""
        sparse_res = self.search_sparse(query, top_k=top_k * 2)
        
        if query_vec is None and self.embeddings is not None:
            query_vec = self.embedder.embed_query(query)
            
        dense_res = self.search_dense(query_vec, top_k=top_k * 2) if query_vec is not None else []
        
        combined_scores: dict[str, float] = {}
        
        for rank, (asin, _) in enumerate(sparse_res):
            combined_scores[asin] = combined_scores.get(asin, 0.0) + sparse_weight * (1.0 / (rrf_k + rank + 1))
            
        for rank, (asin, _) in enumerate(dense_res):
            combined_scores[asin] = combined_scores.get(asin, 0.0) + dense_weight * (1.0 / (rrf_k + rank + 1))
            
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    def search_hybrid_adaptive(
        self,
        query: str,
        intent_type: str = 'buying',
        query_vec: np.ndarray | None = None,
        top_k: int = 80,
    ) -> list[tuple[str, float]]:
        """Executes intent-adaptive hybrid retrieval with optimal weights per scenario."""
        intent_lower = intent_type.lower() if intent_type else 'buying'
        
        if 'override' in intent_lower:
            sparse_w, dense_w, rrf_k = 0.75, 0.25, 20.0
        elif 'browsing' in intent_lower:
            sparse_w, dense_w, rrf_k = 0.40, 0.60, 60.0
        elif 'boundary' in intent_lower:
            sparse_w, dense_w, rrf_k = 0.60, 0.40, 40.0
        else: # buying / constraint update
            sparse_w, dense_w, rrf_k = 0.70, 0.30, 30.0
            
        return self.search_hybrid(
            query=query,
            query_vec=query_vec,
            top_k=top_k,
            sparse_weight=sparse_w,
            dense_weight=dense_w,
            rrf_k=rrf_k,
        )
