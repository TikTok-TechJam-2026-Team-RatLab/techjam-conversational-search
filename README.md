# RatLab Shopping Copilot

TikTok TechJam 2026 — **Statement 4: Shopping Copilot: AI Conversational Search and Recommendations**

## Project Overview

RatLab Shopping Copilot is a fully local conversational product-search agent built for the frozen 50,000-product Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` catalog supplied in the TechJam participant kit. The goal is to identify the customer's hidden target product in as few turns as possible while handling high-intent buying, open-ended browsing, changing preferences, and vague/boundary conversations.

The submitted agent is deterministic and does not call an external LLM API. It reports **zero model tokens** and keeps evaluation-time retrieval entirely local.

### Final architecture

The pipeline combines several complementary signals rather than relying on one retrieval method:

1. **Multi-turn dialogue state tracking** — accumulates category, brand, color, material, size, budget, and other constraints across turns; supports negation, removal, and explicit intent overrides while preserving unrelated constraints.
2. **Intent-aware routing** — classifies the current search as buying-oriented or browsing-oriented and changes the sparse/dense fusion strategy accordingly.
3. **Hybrid retrieval** — uses weighted SQLite FTS5 as the sparse lexical track and optional local FastEmbed vectors (`BAAI/bge-small-en-v1.5`) as the dense semantic track.
4. **Constraint-aware reranking** — reranks a bounded candidate pool using active positives, reliable negatives, budget logic, constraint recency, catalog evidence, and deterministic tie-breaking.
5. **Global catalog-evidence retrieval** — indexes user-sayable catalog facts across all 50,000 products so an exact catalog clue can recover a product even when it is outside the initial sparse/dense candidate pool.
6. **Raw-first evidence fusion** — gives direct customer wording priority over noisier heuristic slots, while retaining slot state as a fallback for paraphrased or accumulated preferences.
7. **Proactive clarification and progressive recommendation breadth** — asks broad structured questions early, returns only the strongest candidate while evidence is still accumulating, and expands to the full Top 10 from turn 4 onward.

The final public-set configuration is documented in [`docs/final_agent.md`](docs/final_agent.md), with machine-readable metrics in [`docs/final_agent_results.json`](docs/final_agent_results.json).

## Final Public Results

Using the official 50,000-product catalog, all 200 released development sessions, the validated dense artifacts, and a locally cached FastEmbed query model:

| Metric | Result |
| --- | ---: |
| Hit Rate@10 | **1.000000** |
| MRR | **0.974048** |
| MTTC | **2.070000** |
| Efficiency | **0.893000** |
| TechnicalScore | **0.970814** |
| Reported tokens | **0** |

Scenario breakdown:

| Scenario | Sessions | Hit@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Boundary | 10 | 1.000000 | 1.000000 | 2.500000 |
| Browsing | 80 | 1.000000 | 0.966667 | 1.900000 |
| Buying | 80 | 1.000000 | 0.976786 | 1.587500 |
| Intent override | 30 | 1.000000 | 0.977778 | 3.666667 |

These are **public development-set results, not a guarantee of private-set performance**.

## Setup and Installation

### Prerequisites

- Python **3.10+**; development and final validation were performed with Python 3.12.x.
- Git.
- Internet access for the initial dependency installation, catalog/artifact download, and one-time FastEmbed model cache. After those resources are present, evaluator runtime is local and does not initiate downloads.

### 1. Clone the repository

```bash
git clone https://github.com/TikTok-TechJam-2026-Team-RatLab/techjam-conversational-search.git
cd techjam-conversational-search
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` pins the optional dense-retrieval dependencies (`fastembed` and `numpy`). The agent retains a standard-library SQLite FTS fallback if dense resources are unavailable.

### 3. Download the official catalog

Download `catalog.jsonl.gz` from the official [TechJam participant-kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit), verify it against the organizer-provided `SHA256SUMS`, decompress it, and place the result at:

```text
data/catalog.jsonl
```

Expected catalog size: **50,000 products**.

For example on Linux/macOS:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

On Windows, decompress the `.gz` archive with your preferred archive tool and move `catalog.jsonl` into `data/`.

### 4. Install the validated dense artifacts used for the submitted score

Download both files from [Phase 1 Clean Retrieval Artifacts v1](https://github.com/TikTok-TechJam-2026-Team-RatLab/techjam-conversational-search/releases/tag/phase1-clean-artifacts-v1):

```text
data/catalog_embeddings.npy
data/catalog_embeddings.json
```

Published SHA-256 digests:

| Artifact | SHA-256 |
| --- | --- |
| `catalog_embeddings.npy` | `56737d96a7e81f4523f8f5122d399180d272420adec8d624e9ac541e4b20eaa6` |
| `catalog_embeddings.json` | `c86c96e733a757b7804bdf4e12c5325b34809787f365fb21ca498177ff5a3bb0` |

The manifest records the model, dimensions, catalog digest, and exact ASIN ordering. The agent rejects incompatible artifacts instead of silently using a mismatched vector matrix.

Alternatively, regenerate the vectors locally from the official catalog:

```bash
python -m Scripts.generate_embeddings
```

### 5. Cache the query embedding model once

While online, run:

```bash
python -c "from src.embedder import Embedder; print(Embedder(local_files_only=False).embed_query('setup check').shape)"
```

The expected shape is `(384,)`. Once cached, normal evaluator runs do not need network access.

## Steps to Reproduce the Results

The commands below assume the full setup above, including the official catalog, both validated dense artifacts, and the cached query model.

### 1. Run the test suite

```bash
python -m unittest discover -v
```

The final submission branch contains **87 unit/integration tests** covering dialogue state, retrieval, routing/fusion, clarification, reranking, catalog evidence, progressive recommendation breadth, overrides, and tie resolution.

### 2. Run the official local evaluator

```bash
python -m evaluator.local_evaluator
```

The evaluator runs all 200 released development sessions and writes the per-session output plus aggregate metrics to `results.json`.

### 3. Verify the expected aggregate metrics

The submitted configuration should reproduce approximately:

```text
Hit Rate@10:     1.000000
MRR:             0.974048
MTTC:            2.070000
Efficiency:      0.893000
Technical score: 0.970814
Reported tokens: 0
```

The checked-in reference metrics are in [`docs/final_agent_results.json`](docs/final_agent_results.json).

If the dense artifacts or cached query model are absent or invalid, the agent deliberately falls back to sparse retrieval. That fallback is functional, but it is **not** the configuration used for the `0.970814` submitted public score.

### Optional ablation/research commands

To reproduce the earlier intent-routing comparison:

```bash
python -m Scripts.evaluate_intent_routing
```

Additional design decisions and controlled ablations are documented in:

- [`docs/phase1_retrieval.md`](docs/phase1_retrieval.md)
- [`docs/proactive_guidance.md`](docs/proactive_guidance.md)
- [`docs/candidate_reranking.md`](docs/candidate_reranking.md)
- [`docs/intent_routing.md`](docs/intent_routing.md)
- [`docs/final_agent.md`](docs/final_agent.md)

## Limitations and Future Improvements

The final system performs very strongly on the released 200-session simulator, but several limitations matter beyond that development set:

- **Exact catalog evidence benefits from the public simulator's structured clues.** Freer paraphrases or substantially different private wording may rely more heavily on sparse/dense retrieval and dialogue-state fallbacks. With more time, we would evaluate on a larger held-out paraphrase set and add a stronger local semantic reranker or domain-adapted e-commerce encoder.
- **The global evidence index adds startup work and memory use.** Per-turn matching is bounded by postings for matched facts, but a production version should profile startup latency and memory more aggressively, then evaluate compact postings, quantization, or ANN indexing where it provides a measured benefit.
- **Popularity is only a tie-breaking prior.** Rating count helps resolve otherwise indistinguishable catalog fingerprints but is not proof of an individual customer's preference. A learned relevance model or cross-encoder trained/evaluated on held-out conversations would be a stronger ranking signal.
- **Broad early `other` questions are optimized for the competition simulator rather than natural conversation.** In a real shopping assistant, we would A/B test more natural attribute-specific clarification, learned next-best-question selection, and better use of the anonymized user profile.
- **Rule-based slot extraction still has edge cases.** We hardened measurement-vs-budget parsing and noisy brand/material vocabulary, but a more general entity normalizer or lightweight local classifier could reduce brittle hand-written parsing rules.
- **Public-set optimization can overfit.** We used controlled ablations and kept the agent independent of target labels at runtime, but repeated iteration on the same 200 public sessions can still favor their distribution. The most important next step would be broader blind validation before further tuning.

## Team Member Contributions

The repository was developed by a five-person Team RatLab. The contribution summary below reflects the code/prototype work and integration history in this repository:

- **[`gooZXSean`](https://github.com/gooZXSean)** — baseline verification; multi-turn constraint tracking and dialogue-state hardening; clean Phase 1 retrieval integration; proactive guidance; candidate reranking; integration and tuning of intent-aware routing; final global catalog-evidence agent; public evaluation, regression testing, reproducibility documentation, and dense-artifact release.
- **[`Glacialthorn`](https://github.com/Glacialthorn)** — initial robust catalog-to-dense-text parser and local vector-retrieval prototype; contributed the compact dense-text representation and normalized semantic-embedding direction later incorporated into the clean retrieval pipeline.
- **[`bearkerb`](https://github.com/bearkerb)** — developed the `feature/phase1agy` retrieval prototype; contributed catalog dataclass ideas, FastEmbed-based local embeddings, exact NumPy dense search, and reciprocal-rank-fusion concepts reused in the integrated Phase 1 architecture.
- **[`duyLeu`](https://github.com/duyLeu)** — developed the dual-index construction prototype and its setup documentation; contributed the prebuilt retrieval-artifact/index-construction direction evaluated during Phase 1 integration.
- **[`fengjiaqi04`](https://github.com/fengjiaqi04)** — developed the intent-routing and score-fusion prototype; contributed the buying-vs-browsing routing logic and RRF/DBSF fusion work that was subsequently integrated, tested, and tuned in the final pipeline.

Parallel prototypes were deliberately reviewed rather than merged wholesale: useful ideas were retained, while stale, heavier, or lower-scoring implementations were replaced during integration. [`docs/phase1_retrieval.md`](docs/phase1_retrieval.md) records the Phase 1 prototype-selection decisions, and [`docs/intent_routing.md`](docs/intent_routing.md) records the routing integration.

## Repository Structure

```text
starter/agent.py                  final Agent implementation and orchestration
starter/session_state.py          multi-turn dialogue state and constraint tracking
src/catalog_evidence.py           global catalog-fact retrieval and deterministic tie-breaking
src/candidate_reranker.py         constraint-aware candidate reranking
src/data_parser.py                strict, order-preserving catalog parser
src/dual_index.py                 SQLite FTS retrieval and optional dense retrieval
src/embedder.py                   local FastEmbed embedding generation/artifact validation
src/intent_routing.py             deterministic intent classification, RRF, and DBSF
src/proactive_guidance.py         deterministic clarification-question selection
Scripts/generate_embeddings.py    reproducible catalog embedding generation
Scripts/evaluate_intent_routing.py routing/fusion ablation runner
evaluator/local_evaluator.py      official public-set simulator and scorer
data/public_set.jsonl             200 released development sessions
docs/final_agent.md               final architecture, progression, results, and limitations
docs/final_agent_results.json     reference final public metrics
```

## Evaluation Metrics

The local evaluator reports:

- **Hit Rate@10:** fraction of sessions where the target appears in the Top 10 within 10 turns.
- **MRR:** mean reciprocal rank of the target product.
- **MTTC:** mean turn of first successful target retrieval; a miss is assigned turn 11.
- **Efficiency:** `clip((11 - MTTC) / 10, 0, 1)`.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
```

Only exact `parent_asin` equality counts as a hit.

## Agent Interface

The implementation preserves the published participant API:

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "...",
            "ask_attribute": "other",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
```

See [`docs/agent_api_contract.json`](docs/agent_api_contract.json) for the machine-readable contract.

## Data Source and Competition Scope

The organizer-provided catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) before using or redistributing the data.

The competition dataset is read-only. The agent does not inject mock ASINs, read evaluator labels/target IDs at runtime, use private evaluation data, or depend on an external vector database or paid API.
