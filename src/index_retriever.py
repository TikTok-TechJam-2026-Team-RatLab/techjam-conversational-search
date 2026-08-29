import json
import os
import bm25s
import faiss
import numpy as np

class IndexRetriever:
    def __init__(self, data_dir: str = "data"):
        """
        Loads pre-built indices directly from disk into memory.
        """
        print("Loading pre-built search indices...")

		# 1. Load ASIN mapping
        asins_path = os.path.join(data_dir, "parent_asins.json")
        with open(asins_path, "r", encoding="utf-8") as f:
            self.parent_asins = json.load(f)

        # 2. Load BM25S sparse index
        bm25_path = os.path.join(data_dir, "bm25s_index")
        self.sparse_retriever = bm25s.BM25.load(bm25_path, mmap=True)

        # 3. Load FAISS HNSW dense index
        faiss_path = os.path.join(data_dir, "faiss_hnsw.index")
        self.dense_index = faiss.read_index(faiss_path)
        
    def search_candidates(self, query_text: str, query_vector: np.ndarray, top_k: int = 10):
        # 1. Query the Sparse Index
        # BM25S returns the indices of the matching documents and their scores
        query_tokens = bm25s.tokenize([query_text], stopwords="en")
        sparse_docs, sparse_scores = self.sparse_retriever.retrieve(query_tokens, k=top_k)
        sparse_asins = [self.parent_asins[idx] for idx in sparse_docs[0]]

		# 2. Query the Dense Index 
		# The query vector from the embedding model must also be float32 and normalized
        q_vec = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q_vec)

		# FAISS returns the distances (scores) and the indices of the nearest neighbors
        dense_scores, dense_docs = self.dense_index.search(q_vec, k=top_k)
        dense_asins = [self.parent_asins[idx] for idx in dense_docs[0]]
        
        return sparse_asins, dense_asins
