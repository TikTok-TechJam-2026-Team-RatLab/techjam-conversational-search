# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the official
[Participant Kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit),
then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Agent

Python 3.10 or later is recommended. Without generated embedding artifacts, the agent uses
its standard-library SQLite FTS fallback.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Optional Dense Retrieval

The retrieval layer can fuse the evaluated sparse ranking with local semantic embeddings.
FastEmbed uses the CPU-friendly `BAAI/bge-small-en-v1.5` ONNX model; no paid API is used.

Create an isolated environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Dependency installation requires internet access unless the packages are already cached. On
Windows, use `python` in place of `python3` in the commands below if necessary.

Generate the vectors and their catalog-alignment manifest from the repository root:

```bash
python3 -m Scripts.generate_embeddings
```

The model is downloaded on the first generation run and cached by FastEmbed. At evaluation
time, the agent refuses network downloads: if the query model is unavailable, it safely falls
back to sparse retrieval. Generated vectors, manifests, model caches, and local dependencies
must not be committed.

To avoid regenerating the 50,000 catalog vectors, download both
`catalog_embeddings.npy` and `catalog_embeddings.json` from the
[Phase 1 Clean Retrieval Artifacts v1 release](https://github.com/TikTok-TechJam-2026-Team-RatLab/techjam-conversational-search/releases/tag/phase1-clean-artifacts-v1)
and place them in `data/`. Warm the query-model cache once while online:

```bash
python3 -c "from src.embedder import Embedder; print(Embedder(local_files_only=False).embed_query('setup check').shape)"
```

With vectors generated from the official 50,000-product catalog, the hybrid configuration scored
Hit Rate@10 `0.255`, MRR `0.125421`, MTTC `8.695`, and technical score `0.211226` across all 200
public sessions. The sparse fallback scored `0.170487`, so hybrid retrieval improved the technical
score by 23.9%. See `docs/phase1_retrieval.md` for the full comparison.

After the declared dependencies and query model were downloaded once, the full evaluator was
successfully rerun with network access disabled. The normal setup flow therefore uses internet
access once for dependencies and the model cache, while evaluator runtime remains local. If either
release artifact or the cached query model is missing, the agent deliberately falls back to sparse
retrieval instead of attempting a runtime download.

Run all isolated tests with:

```bash
python3 -m unittest discover -v
```

See `docs/phase1_retrieval.md` for the branch comparison and design decision.

## Constraint-Aware Candidate Reranking

The Agent retrieves a bounded pool of 100 candidates, then deterministically reranks it using the
active positive constraints, reliable structured negatives, budget proximity, catalog evidence,
constraint recency, and the original retrieval order as a stable tie-breaker. This stage uses no
external API and reports zero tokens.

On the official 200 public sessions, sparse retrieval plus guidance and reranking scores Hit
Rate@10 `0.810`, MRR `0.556538`, MTTC `4.530`, and TechnicalScore `0.701361`. With the validated
dense artifacts, hybrid retrieval plus guidance and reranking scores Hit Rate@10 `0.830`, MRR
`0.542571`, MTTC `4.220`, and TechnicalScore `0.713371`.

See `docs/candidate_reranking.md` for the retrieval-order ablation, pool-depth decision, scenario
metrics, and limitations.

## Intent-Aware Hybrid Routing

The Agent retrieves independent sparse and dense candidate pools before fusion. Concrete
buying turns use the validated weighted RRF policy, while exploratory browsing turns use
distribution-based score fusion with a selected `0.30` sparse / `0.70` dense blend. The fused
pool then flows through constraint-aware reranking and proactive guidance.

On all 200 public sessions, intent-aware hybrid routing scores Hit Rate@10 `0.865`, MRR
`0.538770`, MTTC `3.970`, and TechnicalScore `0.734731`, compared with `0.713371` for the
previous fixed-fusion hybrid pipeline. The sparse-only routed fallback reproduces the validated
`0.701361` sparse score exactly. Set `enable_intent_routing=False` on `Agent` to retain the fixed
RRF ablation.

Reproduce routed, fixed, and sparse-only comparisons with
`python3 -m Scripts.evaluate_intent_routing`. See `docs/intent_routing.md` for the controlled
weight ablation, scenario metrics, integration design, and known boundary trade-off.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
docs/phase1_retrieval.md          Phase 1 implementation decision and limitations
src/data_parser.py                validated, order-preserving catalog parser
src/dual_index.py                 sparse index and optional dense fusion
src/intent_routing.py             deterministic intent classification, RRF, and DBSF
src/embedder.py                   local embedding generation and artifact manifest
src/proactive_guidance.py         deterministic clarification-question selection
src/candidate_reranker.py         deterministic constraint-aware candidate reranking
Scripts/generate_embeddings.py    reproducible embedding build command
Scripts/evaluate_intent_routing.py reproducible routing and fusion ablations
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
docs/proactive_guidance.md         guidance design, ablation, and evaluator results
docs/candidate_reranking.md        reranking design, ablations, and evaluator results
docs/intent_routing.md             intent routing integration and evaluator results
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
