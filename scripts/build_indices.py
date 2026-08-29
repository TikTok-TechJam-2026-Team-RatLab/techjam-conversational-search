import json
import os
import bm25s
import faiss
import numpy as np

from parse_catalog import parse_catalog_to_dense_format

def build_search_indices(raw_catalog_path: str = "data/catalog.jsonl",
                         parsed_catalog_path: str = "data/parsed_catalog.jsonl",
                         embeddings_path: str = "data/embeddings.npy"):
    """
    Builds the Sparse (BM25S) and Dense (FAISS HNSW) indexes in memory.
    """
    # 1. Run the catalog parser again if the parsed file doesn't exist yet
    if not os.path.exists(parsed_catalog_path):
        print(f"Parsed catalog not found. Running parser on {raw_catalog_path}...")
        parse_catalog_to_dense_format(raw_catalog_path, parsed_catalog_path)

    # 2. Read the parsed data
    print(f"Reading parsed catalog from {parsed_catalog_path}...")
    parent_asins = []
    product_texts = []

    with open(parsed_catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            parent_asins.append(item["parent_asin"])
            product_texts.append(item["dense_text"])

    print(f"Loaded {len(parent_asins)} items for indexing to start...")

    # 3. Create Sparse Index (BM25S)
    # Tokenize the product texts (removes stopwords, standardizes text)
    corpus_tokens = bm25s.tokenize(product_texts, stopwords="en")
    
    # Initialize the BM25 model and index the texts
    sparse_retriever = bm25s.BM25()
    sparse_retriever.index(corpus_tokens)
    print("Sparse index (BM25S) built successfully.")

    # 4. Create Dense Index (FAISS HNSW)
    # Load the 768-dimensional BLaIR embeddings (ensure they are float32 for FAISS)
    embeddings = np.load(embeddings_path).astype(np.float32)
    dim = embeddings.shape[1] 
    
    # Initialize the HNSW graph using Inner Product
    dense_index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT) 
    dense_index.hnsw.efConstruction = 200 
    
    # Normalize embeddings so Inner Product behaves like Cosine Similarity
    faiss.normalize_L2(embeddings)
    dense_index.add(embeddings)
    print("Dense index (FAISS HNSW) built successfully.")

    return sparse_retriever, dense_index

def save_indices(sparse_retriever, dense_index, output_dir: str = "data") -> None:
    """
    Serializes the Sparse and Dense Index into disk
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save the Sparse Index
    bm25_save_path = os.path.join(output_dir, "bm25s_index")
    sparse_retriever.save(bm25_save_path)
    print(f"BM25S index saved to {bm25_save_path}/")

    # 2. Save the Dense Index
    faiss_save_path = os.path.join(output_dir, "faiss_hnsw.index")
    faiss.write_index(dense_index, faiss_save_path)
    print(f"FAISS index saved to {faiss_save_path}")

if __name__ == "main":
    sparse_retriever, dense_index = build_search_indices()
    save_indices(sparse_retriever, dense_index)
