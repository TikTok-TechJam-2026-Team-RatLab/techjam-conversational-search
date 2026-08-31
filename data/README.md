# Competition Data

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

The complete official frozen catalog is already committed in this directory, so no download, decompression, or file move is required after cloning the repository. It is the organizer-provided `catalog.jsonl.gz` decompressed without modification.

- Expected row count: **50,000**
- SHA-256: `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`
- Official source: [TechJam Participant Kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit)

## Dense retrieval artifacts

The validated dense-retrieval artifacts used by the submitted agent are already committed here:

```text
catalog_embeddings.npy
catalog_embeddings.json
```

They contain normalized `BAAI/bge-small-en-v1.5` embeddings for the official 50,000-product catalog plus a validation manifest recording the model, dimensions, catalog digest, and exact ASIN ordering. The agent validates the artifacts against the local `catalog.jsonl` before using them.

Published SHA-256 digests:

| Artifact | SHA-256 |
| --- | --- |
| `catalog_embeddings.npy` | `56737d96a7e81f4523f8f5122d399180d272420adec8d624e9ac541e4b20eaa6` |
| `catalog_embeddings.json` | `c86c96e733a757b7804bdf4e12c5325b34809787f365fb21ca498177ff5a3bb0` |

Judges do **not** need to download or move the catalog or embedding artifacts manually after cloning the repository. All competition data required by the evaluator is already under `data/`; only the one-time local FastEmbed query-model cache described in the root README is needed for the submitted dense configuration.

Never place API keys, private evaluation data, or participant outputs in this directory.
