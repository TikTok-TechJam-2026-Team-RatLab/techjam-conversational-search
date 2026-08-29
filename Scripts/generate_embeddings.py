from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root and libs to sys.path
_root = str(Path(__file__).resolve().parent.parent)
_libs = str(Path(__file__).resolve().parent.parent / "libs")
sys.path = [_libs, _root] + [p for p in sys.path if "Python313" not in p and p not in (_libs, _root)]


from src.data_parser import load_catalog
from src.embedder import Embedder, DEFAULT_MODEL_NAME


def main():
    catalog_path = Path('data/catalog.jsonl')
    npy_path = Path('data/catalog_embeddings.npy')
    idx_path = Path('data/asin_to_idx.json')

    print(f'Loading catalog from {catalog_path}...')
    t0 = time.time()
    catalog_data = load_catalog(catalog_path)
    total_items = len(catalog_data.asin_list)
    print(f'Loaded {total_items} items in {time.time() - t0:.2f}s')

    print(f'Generating dense embeddings using {DEFAULT_MODEL_NAME}...')
    t1 = time.time()
    embedder = Embedder(model_name=DEFAULT_MODEL_NAME)
    dim = embedder.embedding_dim

    matrix = np.zeros((total_items, dim), dtype=np.float32)
    chunk_size = 5000

    for i in range(0, total_items, chunk_size):
        end = min(i + chunk_size, total_items)
        sub_texts = catalog_data.dense_texts[i:end]
        ct0 = time.time()
        sub_vecs = embedder.embed_batch(sub_texts, batch_size=256, show_progress=False, num_workers=4)
        matrix[i:end] = sub_vecs
        ct1 = time.time()
        rate = (end - i) / max(0.001, ct1 - ct0)
        print(f'Embedded chunk [{i}:{end}] ({end}/{total_items}) in {ct1 - ct0:.1f}s ({rate:.1f} items/s)')

    t2 = time.time()
    print(f'Successfully generated {matrix.shape} embeddings in {t2 - t1:.2f}s ({(t2 - t1)/60:.2f} mins)')
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), matrix)
    print(f'Saved matrix to {npy_path} ({npy_path.stat().st_size / (1024*1024):.2f} MB)')

    asin_to_idx = {asin: idx for idx, asin in enumerate(catalog_data.asin_list)}
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(asin_to_idx), encoding='utf-8')
    print(f'Saved index mapping to {idx_path}')


if __name__ == '__main__':
    main()
