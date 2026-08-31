# Final Catalog-Evidence Agent

## Decision

The submitted default is a deterministic, offline catalog-evidence agent layered over the
existing intent-routed sparse/dense retrieval, constraint reranking, dialogue state, and EIG
fallback. It uses only the participant-visible catalog and live conversation. It does not load
`data/public_set.jsonl`, ground truth, sample IDs, or evaluator internals.

## Architecture

1. **Intent-aware base retrieval** obtains a bounded pool from SQLite FTS5 and, when the validated
   artifacts and cached model are present, the local dense track.
2. **Constraint reranking** applies active positives, structured negatives, budget logic, and
   recency to the bounded pool.
3. **Global catalog-evidence retrieval** indexes normalized taxonomy segments, titles, feature
   bullets, structured details, descriptions, material/color mentions, stores, and price wording
   for all 50,000 products. IDF-like weights reward facts that identify fewer products.
4. **Raw-first state fusion** matches customer clauses directly against catalog facts. Parsed slots
   are used only when the raw wording has no exact catalog match, preventing false slot fragments
   from outweighing stronger evidence. Negatives and named overrides are applied explicitly.
5. **Stable tie resolution** prefers products with stronger exact evidence, more matched facts, and
   then a log-scaled rating-count prior. A small bounded lexical boost resolves near-equal,
   high-review-count ties without overturning a meaningful popularity gap.
6. **Progressive recommendation breadth** emits the best one candidate on turns 1–3, while asking
   broad structured questions, and expands to Top 10 on turn 4. Later turns retain the established
   EIG guidance fallback.

The pipeline is local, reports zero model tokens, does not mutate the catalog, and preserves the
published Agent API.

## Public-set result

Command:

```bash
python3 -m evaluator.local_evaluator
```

Validated with the official 50,000-product catalog, the released 200 public sessions, the existing
`BAAI/bge-small-en-v1.5` catalog vectors, and a locally cached FastEmbed query model:

| Metric | Final |
| --- | ---: |
| Hit Rate@10 | 1.000000 |
| MRR | 0.974048 |
| MTTC | 2.070000 |
| Efficiency | 0.893000 |
| TechnicalScore | **0.970814** |
| Prompt/completion tokens | 0 / 0 |

| Scenario | Samples | Hit@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Boundary | 10 | 1.000000 | 1.000000 | 2.500000 |
| Browsing | 80 | 1.000000 | 0.966667 | 1.900000 |
| Buying | 80 | 1.000000 | 0.976786 | 1.587500 |
| Intent override | 30 | 1.000000 | 0.977778 | 3.666667 |

## Controlled progression

| Configuration | Hit@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| `main`, sparse fallback | 0.810 | 0.556538 | 4.530 | 0.701361 |
| `main`, routed hybrid | 0.865 | 0.538770 | 3.970 | 0.734731 |
| Broad structured clarification only, sparse | 0.955 | 0.652508 | 2.715 | 0.838952 |
| Evidence index + progressive breadth, first pass | 1.000 | 0.954839 | 2.160 | 0.963252 |
| Raw-first evidence with parser-noise isolation | 1.000 | 0.970298 | 2.075 | 0.969589 |
| Final near-tie policy, routed hybrid | **1.000** | **0.974048** | **2.070** | **0.970814** |

Candidate-pool expansion from 100 to 250 and 500 did not improve the broad-clarification hit rate,
confirming that the remaining failures were evidence-boundary and ranking problems rather than
bounded-pool depth alone.

## Robustness changes

- A measurement such as `fits up to 8-inch wrist circumference` is no longer interpreted as an
  eight-dollar budget ceiling.
- Structural words such as `fits`, `other`, and `stainless` are excluded from noisy brand/material
  vocabulary matches.
- A named override removes its stale exact fact before adding the replacement.
- The global evidence track can recover an exact catalog fact outside the base retrieval pool.
- The entire original test suite remains available as feature-level ablations; final-policy tests
  cover evidence recovery, raw-first matching, progressive breadth, overrides, and tie resolution.

## Limitations

- The public simulator reveals catalog-derived constraints, which particularly benefits exact-fact
  matching. Private paraphrasing or freer natural language may shift more work to FTS, dense
  retrieval, and slot fallbacks, so the public result must not be treated as a private guarantee.
- Catalog-fact indexing increases one-time startup work and memory. Per-turn ranking remains bounded
  to postings for matched facts and uses no external service.
- Some products share the same category and disclosed metadata. Rating count is a practical prior,
  not proof of user preference, so a few exact-fingerprint ties can remain below rank 1.
- Broad `other` questions are effective for the published structured simulator; a production UI
  should A/B test them against more natural attribute-specific prompts.
