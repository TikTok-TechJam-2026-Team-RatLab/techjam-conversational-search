from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

_root = str(Path(__file__).resolve().parent.parent)
_libs = str(Path(__file__).resolve().parent.parent / 'libs')

if sys.version_info[:2] == (3, 12):
    sys.path = [_libs, _root] + [p for p in sys.path if 'Python313' not in p and 'Roaming' not in p and p not in (_libs, _root)]
else:
    if _root not in sys.path:
        sys.path.insert(0, _root)

from starter.agent import Agent
from src.query_synthesizer import QuerySynthesizer
from evaluator.local_evaluator import (
    evaluate,
    catalog_index,
    load_jsonl,
)


def run_grid_search(samples_limit: int | None = None) -> None:
    print('Loading catalog and evaluator dataset...', flush=True)
    catalog_ids, categories, products = catalog_index(Path('data/catalog.jsonl'))
    samples = load_jsonl(Path('data/public_set.jsonl'))
    
    if samples_limit:
        samples = samples[:samples_limit]
        print(f'Subsampled to {len(samples)} evaluation sessions for fast grid search.', flush=True)

    print('Initializing base agent...', flush=True)
    agent = Agent(
        catalog_path='data/catalog.jsonl',
        embeddings_path='data/catalog_embeddings.npy',
    )

    param_grid = {
        'phrase_bonus': [0.45, 0.55],
        'material_bonus': [0.40, 0.50],
        'color_bonus': [0.35, 0.45],
        'purged_penalty': [0.10, 0.15],
    }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))
    print(f'Testing {len(combinations)} hyperparameter combinations...', flush=True)

    best_score = -1.0
    best_config = None
    best_metrics = None

    original_rerank = QuerySynthesizer.rerank_candidates

    for idx, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        
        def make_custom_rerank(cfg):
            def custom_rerank(candidates, state, items_by_asin, top_k=10, **kwargs):
                merged_kwargs = {**cfg, **kwargs}
                return original_rerank(
                    candidates=candidates,
                    state=state,
                    items_by_asin=items_by_asin,
                    top_k=top_k,
                    **merged_kwargs,
                )
            return custom_rerank

        QuerySynthesizer.rerank_candidates = staticmethod(make_custom_rerank(config))

        results = evaluate(
            agent=agent,
            samples=samples,
            catalog_ids=catalog_ids,
            categories=categories,
            products=products,
        )

        score = results['recommended_technical_score']
        hit_rate = results['hit_rate_at_10']
        mrr = results['mrr']
        mttc = results['mttc']

        print(f'[{idx+1}/{len(combinations)}] Config: {config} -> Score: {score:.4f} | HitRate: {hit_rate:.3f} | MRR: {mrr:.3f} | MTTC: {mttc:.2f}', flush=True)

        if score > best_score:
            best_score = score
            best_config = config
            best_metrics = {k: v for k, v in results.items() if k != 'sessions'}

    QuerySynthesizer.rerank_candidates = staticmethod(original_rerank)

    print('\n' + '=' * 60, flush=True)
    print('GRID SEARCH COMPLETED', flush=True)
    print(f'Best Score: {best_score:.6f}', flush=True)
    print(f'Best Configuration: {best_config}', flush=True)
    print(f'Best Metrics: {json.dumps(best_metrics, indent=2)}', flush=True)
    print('=' * 60, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Hyperparameter Grid Search')
    parser.add_argument('--samples', type=int, default=40, help='Number of sessions to evaluate per trial')
    args = parser.parse_args()
    run_grid_search(samples_limit=args.samples)
