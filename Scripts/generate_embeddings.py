from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.data_parser import load_catalog
from src.embedder import DEFAULT_MODEL_NAME, Embedder, write_embedding_artifacts


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate validated catalog embeddings for optional dense retrieval."
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default="data/catalog_embeddings.npy")
    parser.add_argument("--manifest", default="data/catalog_embeddings.json")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse model downloads and use an existing FastEmbed cache only.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional shared FastEmbed model-cache directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    catalog_path = _path(args.catalog)
    output_path = _path(args.output)
    manifest_path = _path(args.manifest)
    cache_dir = _path(args.cache_dir) if args.cache_dir else None

    started = time.perf_counter()
    print(f"Loading catalog from {catalog_path}...")
    catalog = load_catalog(catalog_path)
    print(f"Loaded {len(catalog.items)} products")

    print(f"Embedding products with {args.model}...")
    embedder = Embedder(
        model_name=args.model,
        local_files_only=args.local_files_only,
        cache_dir=cache_dir,
    )
    last_reported = 0

    def report(completed: int, total: int) -> None:
        nonlocal last_reported
        if completed == total or completed - last_reported >= 5000:
            print(f"Embedded {completed}/{total} products")
            last_reported = completed

    matrix = embedder.embed_batch(
        catalog.dense_texts,
        batch_size=args.batch_size,
        progress=report,
    )
    write_embedding_artifacts(
        catalog=catalog,
        matrix=matrix,
        model_name=args.model,
        embeddings_path=output_path,
        manifest_path=manifest_path,
    )

    elapsed = time.perf_counter() - started
    size_mib = output_path.stat().st_size / (1024 * 1024)
    print(
        f"Saved {matrix.shape[0]}x{matrix.shape[1]} vectors to {output_path} "
        f"({size_mib:.1f} MiB) in {elapsed / 60:.1f} minutes"
    )
    print(f"Saved alignment manifest to {manifest_path}")


if __name__ == "__main__":
    main()
