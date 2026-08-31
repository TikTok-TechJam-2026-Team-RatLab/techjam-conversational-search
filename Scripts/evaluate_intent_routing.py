from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from src.intent_routing import RoutingConfig
from starter.agent import Agent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce fixed-fusion and intent-aware public evaluator ablations."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--mode", choices=("fixed", "intent"), default="intent")
    parser.add_argument("--sparse-only", action="store_true")
    parser.add_argument("--buying-threshold", type=float, default=2.0)
    parser.add_argument("--rrf-k", type=float, default=60.0)
    parser.add_argument("--buying-sparse-weight", type=float, default=0.70)
    parser.add_argument("--buying-dense-weight", type=float, default=0.30)
    parser.add_argument("--browsing-sparse-weight", type=float, default=0.30)
    parser.add_argument("--browsing-dense-weight", type=float, default=0.70)
    return parser


def main() -> None:
    args = _parser().parse_args()
    catalog_path = Path(args.catalog)
    agent_options: dict[str, object] = {
        "enable_intent_routing": args.mode == "intent",
        "routing_config": RoutingConfig(
            buying_threshold=args.buying_threshold,
            rrf_k=args.rrf_k,
            buying_sparse_weight=args.buying_sparse_weight,
            buying_dense_weight=args.buying_dense_weight,
            browsing_sparse_weight=args.browsing_sparse_weight,
            browsing_dense_weight=args.browsing_dense_weight,
        ),
    }
    if args.sparse_only:
        agent_options.update({
            "embeddings_path": catalog_path.with_name("__disabled_embeddings.npy"),
            "manifest_path": catalog_path.with_name("__disabled_embeddings.json"),
        })

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(catalog_path)
    result = evaluate(
        Agent(catalog_path, **agent_options),
        samples,
        catalog_ids,
        categories,
        products,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
