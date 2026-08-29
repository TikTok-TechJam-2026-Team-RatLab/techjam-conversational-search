from __future__ import annotations

import json
import re
import sqlite3
import warnings
from pathlib import Path
from typing import Protocol

from src.data_parser import CatalogData


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
RRF_CONSTANT = 60.0
EMBEDDING_SCHEMA_VERSION = 1


class EmbeddingArtifactError(ValueError):
    """Raised when saved vectors cannot be proven to match the active catalog."""


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> object:
        ...


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class DualIndex:
    """Exact sparse retrieval with an optional validated dense-vector track."""

    def __init__(
        self,
        catalog: CatalogData,
        *,
        embeddings_path: str | Path = "data/catalog_embeddings.npy",
        manifest_path: str | Path = "data/catalog_embeddings.json",
        query_embedder: QueryEmbedder | None = None,
        sparse_weight: float = 0.7,
        dense_weight: float = 0.3,
    ) -> None:
        if sparse_weight < 0 or dense_weight < 0 or sparse_weight + dense_weight <= 0:
            raise ValueError("Retrieval weights must be non-negative and not both zero")

        self.catalog = catalog
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.connection = sqlite3.connect(":memory:")
        self._build_sparse_index()

        self.embeddings_path = Path(embeddings_path)
        self.manifest_path = Path(manifest_path)
        self._np: object | None = None
        self._dense_matrix: object | None = None
        self._dense_model_name: str | None = None
        self._query_embedder = query_embedder
        self._dense_disabled_reason: str | None = None
        self._load_dense_artifacts_if_present()

    @property
    def dense_available(self) -> bool:
        return self._dense_matrix is not None and self._dense_disabled_reason is None

    def close(self) -> None:
        self.connection.close()

    def _build_sparse_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        for item in self.catalog.items:
            batch.append(item.sparse_fields())
            if len(batch) >= 1000:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def _load_dense_artifacts_if_present(self) -> None:
        vectors_exist = self.embeddings_path.is_file()
        manifest_exists = self.manifest_path.is_file()
        if not vectors_exist and not manifest_exists:
            return
        if vectors_exist != manifest_exists:
            missing = self.manifest_path if vectors_exist else self.embeddings_path
            raise EmbeddingArtifactError(f"Dense retrieval artifact is incomplete; missing {missing}")

        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EmbeddingArtifactError(
                f"Could not read embedding manifest {self.manifest_path}: {error}"
            ) from error
        if not isinstance(manifest, dict):
            raise EmbeddingArtifactError("Embedding manifest must be a JSON object")
        if manifest.get("schema_version") != EMBEDDING_SCHEMA_VERSION:
            raise EmbeddingArtifactError("Unsupported embedding manifest schema version")
        if manifest.get("catalog_sha256") != self.catalog.catalog_sha256:
            raise EmbeddingArtifactError("Embeddings were built from a different catalog file")
        if manifest.get("parent_asins") != self.catalog.asin_list:
            raise EmbeddingArtifactError("Embedding row order does not match the active catalog")
        if not manifest.get("normalized"):
            raise EmbeddingArtifactError("Embedding manifest does not guarantee normalized vectors")

        try:
            import numpy as np
        except ImportError as error:
            raise EmbeddingArtifactError(
                "Loading dense artifacts requires the dependencies in requirements.txt"
            ) from error

        try:
            matrix = np.load(self.embeddings_path, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise EmbeddingArtifactError(
                f"Could not load embedding matrix {self.embeddings_path}: {error}"
            ) from error

        expected_shape = (manifest.get("rows"), manifest.get("dimension"))
        if matrix.ndim != 2 or matrix.shape != expected_shape:
            raise EmbeddingArtifactError(
                f"Embedding matrix shape {matrix.shape} does not match manifest {expected_shape}"
            )
        if matrix.shape[0] != len(self.catalog.asin_list) or matrix.shape[1] == 0:
            raise EmbeddingArtifactError("Embedding matrix dimensions do not match the catalog")
        if matrix.dtype != np.float32:
            raise EmbeddingArtifactError(
                f"Embedding matrix must use float32, found {matrix.dtype}"
            )

        model_name = manifest.get("model_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise EmbeddingArtifactError("Embedding manifest has no model name")

        self._np = np
        self._dense_matrix = matrix
        self._dense_model_name = model_name

    def search_sparse(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        unique_terms = list(dict.fromkeys(_terms(query)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) "
            "FROM products WHERE products MATCH ? ORDER BY 2 LIMIT ?",
            (expression, min(top_k, len(self.catalog.asin_list))),
        ).fetchall()
        return [(str(parent_asin), float(score)) for parent_asin, score in rows]

    def _get_query_embedder(self) -> QueryEmbedder | None:
        if self._query_embedder is not None:
            return self._query_embedder
        if self._dense_model_name is None or self._dense_disabled_reason is not None:
            return None
        try:
            from src.embedder import Embedder

            self._query_embedder = Embedder(
                model_name=self._dense_model_name,
                local_files_only=True,
            )
        except Exception as error:
            self._dense_disabled_reason = str(error)
            warnings.warn(
                "Dense artifacts are present but the query model is unavailable; "
                f"using sparse retrieval only. Details: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        return self._query_embedder

    def search_dense_vector(
        self,
        query_vector: object,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        if self._dense_matrix is None or self._np is None or top_k <= 0:
            return []

        np = self._np
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vector.size != self._dense_matrix.shape[1]:
            raise ValueError(
                f"Query vector dimension {vector.size} does not match catalog dimension "
                f"{self._dense_matrix.shape[1]}"
            )
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return []
        vector = vector / norm
        scores = np.asarray(self._dense_matrix @ vector)

        k = min(top_k, scores.size)
        if k == scores.size:
            ranked_indices = np.argsort(-scores)
        else:
            candidates = np.argpartition(-scores, k - 1)[:k]
            ranked_indices = candidates[np.argsort(-scores[candidates])]
        return [
            (self.catalog.asin_list[int(index)], float(scores[int(index)]))
            for index in ranked_indices[:k]
        ]

    def search_dense(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not query.strip() or self._dense_matrix is None:
            return []
        embedder = self._get_query_embedder()
        if embedder is None:
            return []
        return self.search_dense_vector(embedder.embed_query(query), top_k=top_k)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        candidate_count = min(max(top_k * 5, top_k), len(self.catalog.asin_list))
        sparse = self.search_sparse(query, top_k=candidate_count)
        if self._dense_matrix is None:
            return sparse[:top_k]

        dense = self.search_dense(query, top_k=candidate_count)
        if not dense:
            return sparse[:top_k]

        combined: dict[str, float] = {}
        first_rank: dict[str, int] = {}
        for rank, (parent_asin, _) in enumerate(sparse, start=1):
            combined[parent_asin] = self.sparse_weight / (RRF_CONSTANT + rank)
            first_rank[parent_asin] = rank
        for rank, (parent_asin, _) in enumerate(dense, start=1):
            combined[parent_asin] = combined.get(parent_asin, 0.0) + (
                self.dense_weight / (RRF_CONSTANT + rank)
            )
            first_rank[parent_asin] = min(first_rank.get(parent_asin, rank), rank)

        ranked = sorted(
            combined.items(),
            key=lambda item: (-item[1], first_rank[item[0]], item[0]),
        )[:top_k]
        return ranked
