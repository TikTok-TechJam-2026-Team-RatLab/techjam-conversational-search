# Proactive Guidance and Expected Information Gain

## Goal

The agent now asks a deterministic follow-up question when its current result set is ambiguous.
This implements Phase 3's Proactive Guidance / EIG Engine without adding an LLM, external API,
or token cost.

The official evaluator only reveals another customer constraint after the agent supplies a valid
`ask_attribute`. Previously the agent always returned `null`, so most non-matching sessions repeated
the same generic response until the ten-turn limit.

## Design

1. Retrieval produces the recommendations returned to the caller. When the customer supplies a
   budget, a structured maximum-price filter is applied to both sparse and dense candidates; products
   with unknown prices remain eligible.
2. A top result is treated as decisive only when its score is separated from the runner-up by at least
   50% of their score scale. Otherwise the result set is ambiguous. An explicit customer rejection
   also requests guidance even after a decisive result.
3. The guidance module extracts supported attributes from those candidate products using structured
   categories, features, details, price, store, title, and description fields.
4. For each available attribute, response-level expected information gain uses a uniform candidate
   prior and models the reply as one of that candidate's known values. Shared values therefore retain
   the correct posterior uncertainty. This diagnostic is kept separate from the question-selection
   utility.
5. Question selection uses coverage-discounted entropy over complete catalog value signatures. This
   deterministic diversity proxy favored questions that converged earlier in a controlled policy
   ablation, while avoiding any evaluator labels or target data. The selected attribute is mapped to
   the exact Agent API vocabulary and a deterministic question; concrete attributes win ties.
6. Each session records asked and explicitly declined attributes so an unhelpful question is not
   repeated. Existing constraints are also excluded from future questions.
7. When no supported candidate attribute separates the results, the engine asks `other` once and
   then safely falls back to `null`.

The recommendation order, session isolation, reset behavior, API response keys, and zero-token usage
remain unchanged.

## Public Evaluator Results

The implementation was evaluated on all 200 public sessions with the official 50,000-product catalog.

| Configuration | Hit Rate@10 | MRR | MTTC | Efficiency | Technical score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous sparse fallback | 0.195 | 0.128956 | 9.285 | 0.1715 | 0.170487 |
| Sparse + selected guidance policy | **0.675** | **0.427109** | **5.595** | **0.5405** | **0.573733** |
| Previous hybrid retrieval | 0.255 | 0.125421 | 8.695 | 0.2305 | 0.211226 |
| Hybrid + selected guidance policy | **0.660** | **0.355950** | **5.660** | **0.5340** | **0.543585** |

Policy-only ablation with all retrieval, budget, parsing, and dialogue behavior held constant:

| Selection policy | Sparse score | Hybrid score |
| --- | ---: | ---: |
| Response-level EIG | 0.562922 | 0.536985 |
| Catalog-signature diversity | **0.573733** | **0.543585** |

Hybrid scenario results after proactive guidance:

| Scenario | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: |
| Boundary | 0.700000 | 0.317222 | 5.800000 |
| Browsing | 0.712500 | 0.388358 | 5.537500 |
| Buying | 0.600000 | 0.277163 | 5.587500 |
| Intent override | 0.666667 | 0.492540 | 6.133333 |

All runs report zero prompt and completion tokens. The hybrid result remains stronger than its prior
baseline, although the guided sparse configuration currently ranks known constraints more precisely.
Intent-aware sparse/dense weighting is deliberately left to the separate intent-routing work.
