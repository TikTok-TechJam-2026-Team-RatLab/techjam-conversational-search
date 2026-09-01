# Intent-Aware Hybrid Routing

## Decision

Intent-aware fusion is enabled by default. The Agent classifies each turn as
`buying` or `browsing`, fuses independent sparse and dense candidate pools, then
passes the fused pool through the existing constraint reranker and proactive
guidance stages.

The previous fixed weighted-RRF path remains available with
`Agent(..., enable_intent_routing=False)` as a reproducible ablation and rollback.
When dense artifacts or the local query model are unavailable, routed retrieval
returns the sparse ranking and scores unchanged.

## Integration

The implementation preserves the useful parts of the original
`feature/intent-routing` prototype while adapting them to the current pipeline:

1. `SessionState` updates the active multi-turn constraints.
2. `DualIndex.search_tracks` retrieves equally deep sparse and dense pools.
3. The deterministic classifier uses the current message and accumulated state.
4. Buying turns use weighted RRF with the validated `0.70` sparse / `0.30` dense
   weights and smoothing constant `60`.
5. Browsing turns use distribution-based score fusion (DBSF), correctly treating
   SQLite BM25 as lower-is-better and cosine similarity as higher-is-better.
6. The selected browsing weights are `0.30` sparse / `0.70` dense.
7. Constraint-aware reranking reduces the fused pool to the requested Top K.
8. Proactive guidance selects a non-repeating clarification from the reranked set.

Explicit phrases such as "still exploring" override generic shopping language
such as "looking for" until the customer supplies a hard constraint. Active hard
constraints keep short replies such as "blue" on the buying route. Exact intent
overrides are processed by `SessionState` before classification.

Fusion ties use the best source rank before `parent_asin`, avoiding the prototype's
lexicographic reordering of equally scored products. Configuration is centralized
in the immutable `RoutingConfig` dataclass for later metric tuning.

## Public Evaluator Results

All runs use the official 50,000-product catalog and all 200 public sessions. No
evaluator labels, target identifiers, or per-session rules are available to the
Agent. Dense runs use the validated Phase 1 artifacts and locally cached
`BAAI/bge-small-en-v1.5` query model.

| Configuration | Hit@10 | MRR | MTTC | Efficiency | Technical score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed RRF, hybrid (`main` baseline) | 0.830 | 0.542571 | 4.220 | 0.6780 | 0.713371 |
| Routed DBSF, browsing 50/50 | 0.860 | 0.530647 | 3.975 | 0.7025 | 0.729694 |
| Routed DBSF, browsing 35/65 | 0.860 | 0.542575 | 4.005 | 0.6995 | 0.732673 |
| Routed DBSF, browsing 25/75 | 0.865 | 0.538383 | 3.985 | 0.7015 | 0.734315 |
| **Routed DBSF, browsing 30/70** | **0.865** | **0.538770** | **3.970** | **0.7030** | **0.734731** |
| Routed sparse fallback | 0.810 | 0.556538 | 4.530 | 0.6470 | 0.701361 |

The selected hybrid policy improves TechnicalScore by `0.021360` (`3.0%`
relative) over fixed RRF. Session comparison found seven newly successful sessions
and no lost hits.

### Scenario comparison

| Scenario | Fixed Hit@10 | Routed Hit@10 | Fixed MRR | Routed MRR | Fixed MTTC | Routed MTTC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Boundary | 0.800000 | 0.800000 | 0.533333 | 0.480000 | 5.100000 | 5.700000 |
| Browsing | 0.887500 | 0.900000 | 0.598522 | 0.582183 | 4.037500 | 3.787500 |
| Buying | 0.762500 | 0.812500 | 0.420580 | 0.420193 | 4.150000 | 3.837500 |
| Intent override | 0.866667 | 0.933333 | 0.721759 | 0.758796 | 4.600000 | 4.233333 |

Boundary MRR and MTTC regress on the ten public boundary sessions, although Hit@10
is unchanged. The aggregate gain is driven by higher buying and intent-override
recall plus faster conversion. This trade-off should be revisited during broader
validation rather than tuned against ten boundary examples.

## Reproduction

Run the selected routed configuration:

```bash
python3 -m Scripts.evaluate_intent_routing \
  --mode intent \
  --output results.json
```

Run the fixed-RRF rollback:

```bash
python3 -m Scripts.evaluate_intent_routing \
  --mode fixed \
  --output results-fixed.json
```

Verify the sparse fallback without moving local artifacts:

```bash
python3 -m Scripts.evaluate_intent_routing \
  --mode intent \
  --sparse-only \
  --output results-sparse.json
```

The script exposes the routing threshold, RRF constant, and buying/browsing fusion
weights as command-line options. It writes full per-session diagnostics while
printing aggregate and scenario metrics.
