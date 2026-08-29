import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_and_search_index(input_filepath="parsed_catalog.jsonl"):
    asins = []
    texts = []
    
    # 1. Load the parsed catalog into memory
    print("Loading catalog...")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            asins.append(item['parent_asin'])
            texts.append(item['dense_text'])

    # 2. Load the embedding model 
    # This downloads an ~80MB model on the first run, then caches it locally.
    print("Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Generate embeddings
    # normalize_embeddings=True is critical. It scales every vector to unit length, 
    # meaning the dot product naturally calculates Cosine Similarity.
    print(f"Encoding {len(texts)} products (this takes a few minutes on CPU)...")
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    # 4. Build the in-memory FAISS Index
    dimension = embeddings.shape[1]  # all-MiniLM-L6-v2 uses 384 dimensions
    index = faiss.IndexFlatIP(dimension) # Inner Product (IP) index
    index.add(embeddings)
    print(f"Successfully indexed {index.ntotal} items in memory.")

    # 5. Define the Browsing Track search function
    def semantic_search(query_text, top_k=5):
        # The user's query must be encoded the exact same way as the catalog
        query_vector = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        
        # FAISS returns the distance scores and the array indices of the closest matches
        distances, indices = index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "parent_asin": asins[idx],
                "score": float(dist),
                "preview": texts[idx][:100] + "..."
            })
        return results

    # Example Search Simulation
    sample_query = "durable outfit for hiking in the rain"
    print(f"\nBrowsing Track Query: '{sample_query}'")
    hits = semantic_search(sample_query)
    
    for rank, hit in enumerate(hits, 1):
        print(f"{rank}. ASIN: {hit['parent_asin']} | Match Score: {hit['score']:.4f}")
        print(f"   {hit['preview']}")

if __name__ == "__main__":
    build_and_search_index()