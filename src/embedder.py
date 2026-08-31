from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence
import numpy as np
from fastembed import TextEmbedding
from tqdm import tqdm

DEFAULT_MODEL_NAME = 'BAAI/bge-small-en-v1.5'


class Embedder:
    """Local dense vector embedder using ONNX-quantized FastEmbed models."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, threads: int | None = None) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, threads=threads, local_files_only=True)
        self.embedding_dim = 384 if ('small' in model_name or 'MiniLM' in model_name or 'xs' in model_name) else 768
        self._query_cache: dict[str, np.ndarray] = {}

    def embed_query(self, query: str) -> np.ndarray:
        """Encodes a single search query into an L2-normalized 1D float32 numpy vector with in-memory caching."""
        if not query or not query.strip():
            return np.zeros(self.embedding_dim, dtype=np.float32)
        q_clean = query.strip()
        if q_clean in self._query_cache:
            return self._query_cache[q_clean]
        
        vec = next(self._model.embed([q_clean]))
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        if len(self._query_cache) < 10000:
            self._query_cache[q_clean] = vec
        return vec

    def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 256,
        show_progress: bool = True,
        num_workers: int = 6,
    ) -> np.ndarray:
        """Encodes a sequence of texts into an (N, D) L2-normalized float32 numpy matrix in parallel."""
        n = len(texts)
        if n == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # For small batches, run on single model
        if n <= 500 or num_workers <= 1:
            embeddings_list: list[np.ndarray] = []
            embed_gen = self._model.embed(texts, batch_size=batch_size)
            iterator = tqdm(embed_gen, total=n, desc='Embedding', disable=not show_progress)
            for vec in iterator:
                embeddings_list.append(np.asarray(vec, dtype=np.float32))
            matrix = np.vstack(embeddings_list)
        else:
            # Parallel chunk embedding across multiple ONNX sessions
            models = [TextEmbedding(model_name=self.model_name, local_files_only=True) for _ in range(num_workers)]
            chunk_size = (n + num_workers - 1) // num_workers
            chunks = []
            for i in range(num_workers):
                start = i * chunk_size
                end = min(start + chunk_size, n)
                if start < end:
                    chunks.append((i, texts[start:end]))

            def _worker_task(worker_id: int, sub_texts: Sequence[str]) -> list[np.ndarray]:
                worker_model = models[worker_id]
                res = []
                for v in worker_model.embed(sub_texts, batch_size=batch_size):
                    res.append(np.asarray(v, dtype=np.float32))
                return res

            with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                futures = [executor.submit(_worker_task, cid, ctexts) for cid, ctexts in chunks]
                results = []
                for f in tqdm(futures, desc='Parallel Embedding Workers', disable=not show_progress):
                    results.extend(f.result())
            matrix = np.vstack(results)

        # L2-normalize all vectors
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        return matrix.astype(np.float32)

    @classmethod
    def generate_and_save_catalog_embeddings(
        cls,
        dense_texts: list[str],
        asin_list: list[str],
        output_npy_path: str | Path = 'data/catalog_embeddings.npy',
        output_idx_path: str | Path = 'data/asin_to_idx.json',
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 256,
        num_workers: int = 6,
    ) -> np.ndarray:
        """Generates embeddings for the full catalog and saves to disk."""
        embedder = cls(model_name=model_name)
        matrix = embedder.embed_batch(dense_texts, batch_size=batch_size, show_progress=True, num_workers=num_workers)
        npy_path = Path(output_npy_path)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), matrix)
        idx_path = Path(output_idx_path)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        asin_to_idx = {asin: i for i, asin in enumerate(asin_list)}
        idx_path.write_text(json.dumps(asin_to_idx), encoding='utf-8')
        return matrix

