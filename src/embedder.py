from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.data_parser import CatalogData


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_SCHEMA_VERSION = 1


class EmbeddingDependencyError(RuntimeError):
    """Raised when the optional local embedding dependency is unavailable."""


class Embedder:
    """Generate normalized local embeddings through FastEmbed's ONNX runtime."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        local_files_only: bool = True,
        cache_dir: str | Path | None = None,
        threads: int | None = None,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise EmbeddingDependencyError(
                "Dense retrieval requires the dependencies in requirements.txt"
            ) from error

        options: dict[str, object] = {
            "model_name": model_name,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            options["cache_dir"] = str(cache_dir)
        if threads is not None:
            options["threads"] = threads

        self.model_name = model_name
        try:
            self._model = TextEmbedding(**options)
        except Exception as error:
            mode = "the local model cache" if local_files_only else "the model download"
            raise EmbeddingDependencyError(
                f"Could not initialize {model_name!r} from {mode}: {error}"
            ) from error

    @staticmethod
    def _normalize(vector: object) -> np.ndarray:
        result = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(result))
        if norm > 1e-12:
            result = result / norm
        return result.astype(np.float32, copy=False)

    def embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            return np.empty(0, dtype=np.float32)
        vector = next(iter(self._model.embed([query.strip()], batch_size=1)))
        return self._normalize(vector)

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 256,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        iterator = iter(self._model.embed(texts, batch_size=batch_size))
        try:
            first = self._normalize(next(iterator))
        except StopIteration as error:
            raise RuntimeError("The embedding model returned no catalog vectors") from error

        matrix = np.empty((len(texts), first.size), dtype=np.float32)
        matrix[0] = first
        if progress is not None:
            progress(1, len(texts))

        completed = 1
        for completed, vector in enumerate(iterator, start=2):
            normalized = self._normalize(vector)
            if normalized.size != first.size:
                raise RuntimeError(
                    f"Embedding dimension changed from {first.size} to {normalized.size}"
                )
            matrix[completed - 1] = normalized
            if progress is not None:
                progress(completed, len(texts))

        if completed != len(texts):
            raise RuntimeError(
                f"Embedding model returned {completed} vectors for {len(texts)} texts"
            )
        return matrix


def write_embedding_artifacts(
    *,
    catalog: CatalogData,
    matrix: np.ndarray,
    model_name: str,
    embeddings_path: str | Path,
    manifest_path: str | Path,
) -> None:
    """Atomically save vectors plus enough metadata to prevent row misalignment."""
    embeddings = np.asarray(matrix, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional matrix")
    if embeddings.shape[0] != len(catalog.asin_list):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) do not match catalog rows "
            f"({len(catalog.asin_list)})"
        )
    if embeddings.shape[1] == 0:
        raise ValueError("Embedding vectors cannot have zero dimensions")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains non-finite values")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Embedding matrix contains a zero vector")
    embeddings = embeddings / norms

    vector_path = Path(embeddings_path)
    metadata_path = Path(manifest_path)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    vector_tmp = vector_path.with_suffix(vector_path.suffix + ".tmp")
    with vector_tmp.open("wb") as handle:
        np.save(handle, embeddings.astype(np.float32, copy=False))
    vector_tmp.replace(vector_path)

    manifest = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "model_name": model_name,
        "catalog_sha256": catalog.catalog_sha256,
        "parent_asins": catalog.asin_list,
        "rows": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "dtype": "float32",
        "normalized": True,
    }
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_tmp.write_text(
        json.dumps(manifest, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    metadata_tmp.replace(metadata_path)
