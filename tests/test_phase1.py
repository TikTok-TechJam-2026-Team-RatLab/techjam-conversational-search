from __future__ import annotations

import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
_libs = str(Path(__file__).resolve().parent.parent / "libs")
sys.path = [_libs, _root] + [p for p in sys.path if "Python313" not in p and p not in (_libs, _root)]

import numpy as np




from src.data_parser import load_catalog, CatalogData
from src.embedder import Embedder
from src.dual_index import DualIndex


def test_data_parser():
    print('Testing data parser...')
    t0 = time.time()
    data = load_catalog('data/catalog.jsonl')
    t1 = time.time()
    assert len(data.asin_list) == 50000, f'Expected 50000 items, got {len(data.asin_list)}'
    assert len(data.dense_texts) == 50000
    assert len(data.items_by_asin) == 50000
    assert len(data.asin_to_idx) == 50000

    sample_asin = data.asin_list[0]
    sample_item = data.items_by_asin[sample_asin]
    assert sample_item.parent_asin == sample_asin
    assert len(sample_item.dense_text) > 0
    print(f'Data parser passed ({t1 - t0:.2f}s)')


def test_embedder_query():
    print('Testing query embedder...')
    t0 = time.time()
    embedder = Embedder()
    vec = embedder.embed_query('mens running shoes')
    t1 = time.time()
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-4, f'Expected unit norm, got {np.linalg.norm(vec)}'

    zero_vec = embedder.embed_query('')
    assert zero_vec.shape == (384,)
    assert np.linalg.norm(zero_vec) == 0.0
    print(f'Query embedder passed (latency: {(t1 - t0)*1000:.2f}ms)')


def test_dual_index_sparse():
    print('Testing DualIndex sparse search...')
    index = DualIndex('data/catalog.jsonl', load_dense=False)
    assert index.num_items == 50000

    t0 = time.time()
    results = index.search_sparse('cotton graphic t-shirt', top_k=10)
    latency_ms = (time.time() - t0) * 1000
    assert len(results) == 10
    assert latency_ms < 50.0, f'Sparse search latency too high: {latency_ms:.2f}ms'
    print(f'DualIndex sparse search passed (latency: {latency_ms:.2f}ms)')


if __name__ == '__main__':
    test_data_parser()
    test_embedder_query()
    test_dual_index_sparse()
    print('All Phase 1 unit tests passed!')
