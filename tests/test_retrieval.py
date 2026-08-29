from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data_parser import CatalogFormatError, load_catalog
from src.dual_index import DualIndex, EmbeddingArtifactError
from src.embedder import Embedder, write_embedding_artifacts


PRODUCTS = [
    {
        "parent_asin": "RED_SHIRT",
        "title": "Red cotton shirt",
        "categories": ["Clothing", "Shirts"],
        "features": ["soft fabric"],
        "details": {"Color": "Red", "Material": "Cotton"},
        "description": ["casual top"],
        "store": "Northwind",
    },
    {
        "parent_asin": "HIKING_BOOT",
        "title": "Waterproof hiking boot",
        "categories": ["Clothing", "Shoes"],
        "features": ["strong arch support"],
        "details": {"Color": "Brown", "Material": "Leather"},
        "description": ["outdoor footwear"],
        "store": "Trailworks",
    },
    {
        "parent_asin": "BLUE_HAT",
        "title": "Blue sun hat",
        "categories": ["Accessories", "Hats"],
        "features": ["wide brim"],
        "details": {"Color": "Blue"},
        "description": ["summer headwear"],
        "store": "Northwind",
    },
]


def write_catalog(path: Path, products: list[dict] = PRODUCTS) -> None:
    path.write_text(
        "".join(json.dumps(product) + "\n" for product in products),
        encoding="utf-8",
    )


class FakeEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.vector


class FakeTextEmbedding:
    last_options: dict[str, object] = {}

    def __init__(self, **options: object) -> None:
        type(self).last_options = options

    def embed(self, texts: list[str], batch_size: int) -> object:
        del batch_size
        for text in texts:
            yield np.asarray([len(text), 1.0], dtype=np.float32)


class DataParserTest(unittest.TestCase):
    def test_catalog_preserves_one_validated_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            write_catalog(catalog_path)

            catalog = load_catalog(catalog_path)

            self.assertEqual(catalog.asin_list, ["RED_SHIRT", "HIKING_BOOT", "BLUE_HAT"])
            self.assertEqual(catalog.asin_to_idx["HIKING_BOOT"], 1)
            self.assertEqual(catalog.items_by_asin["RED_SHIRT"].price, None)
            self.assertIn("Details: Color: Red Material: Cotton", catalog.dense_texts[0])
            self.assertEqual(len(catalog.catalog_sha256), 64)

    def test_invalid_json_is_reported_instead_of_silently_changing_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            catalog_path.write_text('{"parent_asin":"A"}\nnot-json\n', encoding="utf-8")

            with self.assertRaisesRegex(CatalogFormatError, "line 2"):
                load_catalog(catalog_path)

    def test_duplicate_asin_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            write_catalog(catalog_path, [PRODUCTS[0], PRODUCTS[0]])

            with self.assertRaisesRegex(CatalogFormatError, "Duplicate parent_asin"):
                load_catalog(catalog_path)


class EmbedderTest(unittest.TestCase):
    def test_fastembed_adapter_normalizes_batches_and_queries(self) -> None:
        fake_module = types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        with patch.dict(sys.modules, {"fastembed": fake_module}):
            embedder = Embedder(model_name="test-model", local_files_only=True)
            matrix = embedder.embed_batch(["a", "abcd"], batch_size=2)
            query = embedder.embed_query("query")

        self.assertEqual(matrix.shape, (2, 2))
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0])
        self.assertAlmostEqual(float(np.linalg.norm(query)), 1.0)
        self.assertEqual(FakeTextEmbedding.last_options["model_name"], "test-model")
        self.assertEqual(FakeTextEmbedding.last_options["local_files_only"], True)


class DualIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.catalog_path = self.root / "catalog.jsonl"
        self.embeddings_path = self.root / "catalog_embeddings.npy"
        self.manifest_path = self.root / "catalog_embeddings.json"
        write_catalog(self.catalog_path)
        self.catalog = load_catalog(self.catalog_path)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_sparse_index_uses_weighted_catalog_fields(self) -> None:
        index = DualIndex(
            self.catalog,
            embeddings_path=self.embeddings_path,
            manifest_path=self.manifest_path,
        )

        results = index.search("waterproof hiking shoes", top_k=2)

        self.assertEqual(results[0][0], "HIKING_BOOT")

    def test_dense_search_handles_top_k_equal_to_catalog_size(self) -> None:
        matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        write_embedding_artifacts(
            catalog=self.catalog,
            matrix=matrix,
            model_name="test-model",
            embeddings_path=self.embeddings_path,
            manifest_path=self.manifest_path,
        )
        index = DualIndex(
            self.catalog,
            embeddings_path=self.embeddings_path,
            manifest_path=self.manifest_path,
            query_embedder=FakeEmbedder([0.0, 1.0]),
        )

        results = index.search_dense_vector([0.0, 1.0], top_k=10)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], "HIKING_BOOT")

    def test_hybrid_search_can_recover_a_semantic_only_candidate(self) -> None:
        matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        write_embedding_artifacts(
            catalog=self.catalog,
            matrix=matrix,
            model_name="test-model",
            embeddings_path=self.embeddings_path,
            manifest_path=self.manifest_path,
        )
        index = DualIndex(
            self.catalog,
            embeddings_path=self.embeddings_path,
            manifest_path=self.manifest_path,
            query_embedder=FakeEmbedder([0.0, 1.0]),
        )

        results = index.search("completely unseen vocabulary", top_k=1)

        self.assertEqual(results[0][0], "HIKING_BOOT")

    def test_catalog_change_invalidates_saved_embeddings(self) -> None:
        matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        write_embedding_artifacts(
            catalog=self.catalog,
            matrix=matrix,
            model_name="test-model",
            embeddings_path=self.embeddings_path,
            manifest_path=self.manifest_path,
        )
        changed_products = [dict(product) for product in PRODUCTS]
        changed_products[0]["title"] = "Changed product text"
        changed_path = self.root / "changed_catalog.jsonl"
        write_catalog(changed_path, changed_products)

        with self.assertRaisesRegex(EmbeddingArtifactError, "different catalog"):
            DualIndex(
                load_catalog(changed_path),
                embeddings_path=self.embeddings_path,
                manifest_path=self.manifest_path,
            )


if __name__ == "__main__":
    unittest.main()
