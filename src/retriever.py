import bm25s
import faiss
import numpy as np

def retrieve_top_candidates(query_text: str, query_vector: np.ndarray, sparse_retriever, dense_index, parent_asins, top_k=10):
    # 1. Query the Sparse Index
    query_tokens = bm25s.tokenize([query_text], stopwords="en")

    # BM25S returns the indices of the matching documents and their scores
    sparse_results, sparse_scores = sparse_retriever.retrieve(query_tokens, k=top_k)
    sparse_indices = sparse_results[0] # The row IDs of the top matches

    # 2. Query the Dense Index 
    # The query vector from the embedding model must also be float32 and normalized
    q_vec = query_vector.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q_vec)
    
    # FAISS returns the distances (scores) and the indices of the nearest neighbors
    dense_scores, dense_results = dense_index.search(q_vec, k=top_k)
    dense_indices = dense_results[0] # The row IDs of the top matches
    
    # 3. Map back to parent_asins
    sparse_asins = [parent_asins[idx] for idx in sparse_indices]
    dense_asins = [parent_asins[idx] for idx in dense_indices]
    
    return sparse_asins, dense_asins