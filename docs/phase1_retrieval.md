# Phase 1 Retrieval Integration

## Decision

This integration was rebuilt from current `main` instead of merging any stale Phase 1 branch.
It preserves the existing dialogue-state tracker and selects the strongest reusable ideas from
the parallel prototypes.

| Prototype | Reused | Not selected |
| --- | --- | --- |
| `feature/add-data-parser` | The compact dense-text representation | Its standalone output duplicated catalog state and silently skipped malformed rows |
| `feature/add-local-vector-embedder` | Normalized 384-dimensional local embeddings | Sentence Transformers plus FAISS added heavier dependencies and did not persist validated artifacts |
| `feature/dual-index-construction` | The idea of prebuilding retrieval artifacts | The builder did not generate embeddings or save its ASIN mapping, its entry point was unreachable, and it was not integrated with `Agent` |
| `feature/phase1agy` | Catalog dataclasses, FastEmbed, exact NumPy search, and reciprocal-rank fusion | The committed Windows dependency tree, global `sys.path` mutation, stale Agent implementation, and unvalidated row mapping |

## Resulting Architecture

1. `src.data_parser` validates every catalog row, rejects duplicate identifiers, and records both
   the exact row order and SHA-256 digest.
2. `src.dual_index` preserves the weighted SQLite FTS ranking already evaluated on the public set.
3. `src.embedder` optionally generates normalized local vectors with
   `BAAI/bge-small-en-v1.5` through FastEmbed.
4. The vector matrix is accompanied by a manifest containing the model, dimensions, catalog
   digest, and complete ordered ASIN list. Dense retrieval refuses mismatched artifacts.
5. `starter.Agent` continues to query `SessionState.retrieval_context()`, so intent overrides and
   removed preferences do not reappear in retrieval.
6. Sparse and dense candidates are fused with weighted reciprocal-rank fusion. With no valid
   dense artifacts or no local query model, the evaluated sparse path remains available.

For 50,000 products, an exact NumPy dot product is intentionally preferred over FAISS. It keeps
the implementation portable and deterministic while evaluating only about 19.2 million scalar
products per query for the default 384-dimensional model. FAISS can be reconsidered only after
measured latency shows that exact search is a real bottleneck.

## Reproducibility and Limitations

- Runtime model/API cost: zero; all inference is local.
- Network use: the first embedding-generation run may download the model. Evaluation never
  initiates a download.
- Dense evaluation requires both generated artifacts and access to the same cached query model.
- Generated catalog vectors are roughly 73 MiB and are deliberately excluded from Git.
- The current clarification policy still returns no `ask_attribute`; improving browsing and
  boundary sessions remains later-phase work.

## Validation

The clean branch was compared against the `main` commit from which it was created using the
official 50,000-product catalog and all 200 public sessions.

| Check | Result |
| --- | --- |
| Isolated test suite | 28 tests passed |
| Hit Rate@10 | `0.195` |
| MRR | `0.128956` |
| MTTC | `9.285` |
| Recommended technical score | `0.170487` |

The sparse-fallback metrics are exactly equal to the source `main` baseline. This confirms that
the refactor and state-aware integration introduce no retrieval regression before optional dense
artifacts are enabled. Dense metrics must be reported separately after generating the official
catalog vectors and ensuring the query model is available in the evaluation environment.

The two BM25S corpus layouts from the overlapping prototypes were also run through the same 200
public sessions before selecting the sparse implementation:

| Sparse implementation | Hit Rate@10 | MRR | Technical score |
| --- | ---: | ---: | ---: |
| Existing weighted SQLite FTS | `0.195` | `0.128956` | `0.170487` |
| BM25S with repeated title/category fields | `0.150` | `0.092490` | `0.128647` |
| BM25S over the complete dense text | `0.180` | `0.109835` | `0.154750` |

SQLite FTS therefore remains the sparse track; the BM25S dependency and its generated index are
not carried into the clean branch.
