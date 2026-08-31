# Proactive Guidance and Expected Information Gain

## Goal

The agent now asks a deterministic follow-up question when its current result set is ambiguous.
This implements Phase 3's Proactive Guidance / EIG Engine without adding an LLM, external API,
or token cost.

The official evaluator only reveals another customer constraint after the agent supplies a valid
`ask_attribute`. Previously the agent always returned `null`, so most non-matching sessions repeated
the same generic response until the ten-turn limit.

## Design

1. Retrieval runs exactly as before and produces the recommendations returned to the caller.
2. The guidance module extracts supported attributes from those candidate products using structured
   categories, features, details, price, store, title, and description fields.
3. For each available attribute, the candidate set is partitioned by its normalized values. Shannon
   entropy measures the expected reduction in candidate uncertainty, discounted by catalog coverage.
4. The highest-information attribute is mapped to the exact Agent API vocabulary and a deterministic
   question. Concrete attributes win ties.
5. Each session records asked and explicitly declined attributes so an unhelpful question is not
   repeated. Existing constraints are also excluded from future questions.
6. When no supported candidate attribute separates the results, the engine asks `other` once and
   then safely falls back to `null`.

The recommendation order, session isolation, reset behavior, API response keys, and zero-token usage
remain unchanged.

## Public Evaluator Results

The implementation was evaluated on all 200 public sessions with the official 50,000-product catalog.

| Configuration | Hit Rate@10 | MRR | MTTC | Efficiency | Technical score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous sparse fallback | 0.195 | 0.128956 | 9.285 | 0.1715 | 0.170487 |
| Sparse + proactive guidance | **0.675** | **0.427109** | **5.595** | **0.5405** | **0.573733** |
| Previous hybrid retrieval | 0.255 | 0.125421 | 8.695 | 0.2305 | 0.211226 |
| Hybrid + proactive guidance | **0.660** | **0.356950** | **5.660** | **0.5340** | **0.543885** |

Hybrid scenario results after proactive guidance:

| Scenario | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: |
| Boundary | 0.700000 | 0.317222 | 5.800000 |
| Browsing | 0.712500 | 0.390858 | 5.537500 |
| Buying | 0.600000 | 0.277163 | 5.587500 |
| Intent override | 0.666667 | 0.492540 | 6.133333 |

All runs report zero prompt and completion tokens. The hybrid result remains stronger than its prior
baseline, although the guided sparse configuration currently ranks known constraints more precisely.
Intent-aware sparse/dense weighting is deliberately left to the separate intent-routing work.
