# Constraint-Aware Candidate Reranking

## Decision

The Agent now retrieves a bounded pool of 100 candidates and deterministically reranks that pool
before returning the requested Top 10. This preserves the recall of sparse or hybrid candidate
generation while using the active dialogue state to improve final ordering.

The reranker uses only information available to the Agent at runtime. It never reads evaluator
labels, target identifiers, intent cards, or per-sample rules.

## Ranking Policy

For each retrieved candidate, the reranker:

1. measures token coverage for every active positive constraint against the catalog text;
2. gives rare, discriminating terms more weight using document frequency within the current
   candidate pool;
3. rewards exact phrase matches and adds a small field-specific bonus for category and brand;
4. gives slightly more weight to recently updated constraint attributes;
5. applies strong penalties only for reliable structured negative attributes such as color,
   material, size, brand, category, and budget;
6. uses proximity for an explicit "around" budget and retains the existing hard price ceiling;
7. preserves the original retrieval rank as the deterministic tie-breaker.

Unstructured negative text is intentionally not used as an exclusion. Catalog facts such as
"no ironing" or "no closure" can appear inside a customer-provided product requirement and are
not reliable evidence that the customer rejects that product.

The original retrieval scores continue to drive the established clarification-confidence test.
Question selection sees the reranked products, but candidate reranking does not manufacture a new
score scale or silently change the ambiguity threshold.

## Public Evaluator Results

All runs use the official 50,000-product catalog and all 200 public sessions.

| Configuration | Hit Rate@10 | MRR | MTTC | Efficiency | Technical score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse + guidance, retrieval ordering | 0.675 | 0.427109 | 5.595 | 0.5405 | 0.573733 |
| **Sparse + guidance + reranking** | **0.810** | **0.556538** | **4.530** | **0.6470** | **0.701361** |
| Hybrid + guidance, retrieval ordering | 0.660 | 0.355950 | 5.660 | 0.5340 | 0.543585 |
| **Hybrid + guidance + reranking** | **0.830** | **0.542571** | **4.220** | **0.6780** | **0.713371** |

The selected hybrid configuration improves TechnicalScore by `0.169786` over the previous hybrid
configuration and by `0.139638` over the strongest previous sparse configuration. The sparse
fallback also improves by `0.127628`, so missing optional dense resources no longer removes the
benefit of this milestone.

### Final scenario metrics

| Scenario | Sparse Hit@10 | Sparse MRR | Sparse MTTC | Hybrid Hit@10 | Hybrid MRR | Hybrid MTTC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Boundary | 0.700000 | 0.566667 | 6.500000 | 0.800000 | 0.533333 | 5.100000 |
| Browsing | 0.925000 | 0.612113 | 3.987500 | 0.887500 | 0.598522 | 4.037500 |
| Buying | 0.725000 | 0.459216 | 4.537500 | 0.762500 | 0.420580 | 4.150000 |
| Intent override | 0.766667 | 0.664484 | 5.300000 | 0.866667 | 0.721759 | 4.600000 |

### Candidate-pool ablation

With the final sparse reranking policy held constant, a pool of 100 scored `0.701361`; increasing
the pool to 200 scored `0.700986`. The larger pool added work and slightly reduced MRR, so 100 is
the selected default. The constructor retains `enable_reranking=False` for retrieval-order
ablation and safe rollback.

## Validation and Limitations

- The implementation is deterministic and reports zero prompt and completion tokens.
- Candidate text normalization is cached within a bounded in-memory LRU cache.
- Sparse fallback and validated optional dense fusion use the same reranking policy.
- Public-set improvements do not guarantee the same magnitude on the 800 private sessions; the
  policy is therefore catalog- and state-driven rather than sample-specific.
- Hybrid retrieval remains slower and requires its published embedding artifacts plus a local
  query-model cache. Sparse reranking remains the portable zero-artifact fallback.

Run the isolated tests and evaluator from the repository root:

```bash
python3 -m unittest discover -v
python3 -m evaluator.local_evaluator
```
