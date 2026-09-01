# Competition Data

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

Download `catalog.jsonl.gz` from the official
[Participant Kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit)
and decompress it as `catalog.jsonl` in this directory. Expected row count: 50,000.

# Team Data

## `catalog_embeddings.npy`
This is the compiled dense vector matrix. It is a binary file containing the pre-computed mathematical representations (embeddings) for all 50,000 products in your catalog.jsonl file. By saving these heavy computations offline, your system can load this matrix directly into memory at runtime to perform blazing-fast semantic similarity searches (calculating dot products) without having to run text through the machine learning model on the fly.

## `catalog_embeddings.json`
This is the metadata and mapping registry that pairs with the .npy matrix. Because a NumPy array only holds raw numbers without any labels, this file acts as the vital translation layer. It contains:

- `parent_asins`: A strictly ordered list of product IDs. Row i in the .npy matrix corresponds exactly to the ASIN at index i in this list.

- `model_name`: Tracks the specific model used (BAAI/bge-small-en-v1.5) so the search engine knows the vector dimensions.

- `catalog_sha256`: A security hash that guarantees these embeddings were generated from the exact, current version of the catalog.jsonl file, preventing mismatched data errors if the catalog is updated.
