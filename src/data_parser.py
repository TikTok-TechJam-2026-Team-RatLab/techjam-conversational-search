from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CatalogFormatError(ValueError):
    """Raised when a catalog row cannot be parsed without losing index alignment."""


def _string_list(value: object) -> list[str]:
    if value in (None, ""):
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _flatten_text(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            flattened = _flatten_text(item)
            if flattened:
                parts.append(f"{key}: {flattened}")
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(part for item in value if (part := _flatten_text(item)))
    return str(value).strip()


def _sparse_text(value: object) -> str:
    """Match the original FTS field serialization so rankings stay reproducible."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _float_or_zero(value: object) -> float:
    parsed = _optional_float(value)
    return 0.0 if parsed is None else parsed


def _int_or_zero(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class CatalogItem:
    parent_asin: str
    title: str
    categories: list[str]
    features: list[str]
    details: dict[str, Any]
    description: list[str]
    price: float | None
    average_rating: float
    rating_number: int
    store: str
    dense_text: str

    def sparse_fields(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.parent_asin,
            _sparse_text(self.title),
            _sparse_text(self.categories),
            _sparse_text(self.features),
            _sparse_text(self.details),
            _sparse_text(self.store),
            _sparse_text(self.description),
        )

    def as_product(self) -> dict[str, object]:
        return {
            "parent_asin": self.parent_asin,
            "title": self.title,
            "categories": self.categories,
            "features": self.features,
            "details": self.details,
            "description": self.description,
            "price": self.price,
            "average_rating": self.average_rating,
            "rating_number": self.rating_number,
            "store": self.store,
        }


@dataclass(frozen=True)
class CatalogData:
    items: list[CatalogItem]
    asin_list: list[str]
    dense_texts: list[str]
    items_by_asin: dict[str, CatalogItem]
    asin_to_idx: dict[str, int]
    catalog_sha256: str


def clean_dense_text(
    *,
    title: str,
    categories: list[str],
    features: list[str],
    details: dict[str, Any],
    description: list[str],
    store: str,
) -> str:
    labelled_values = (
        ("Title", title),
        ("Categories", categories),
        ("Features", features),
        ("Details", details),
        ("Store", store),
        ("Description", description),
    )
    parts = []
    for label, value in labelled_values:
        flattened = _flatten_text(value)
        if flattened:
            parts.append(f"{label}: {flattened}")
    return re.sub(r"\s+", " ", " | ".join(parts)).strip()


def _parse_record(record: object, *, line_number: int) -> CatalogItem:
    if not isinstance(record, dict):
        raise CatalogFormatError(f"Catalog line {line_number} is not a JSON object")

    parent_asin = str(record.get("parent_asin") or "").strip()
    if not parent_asin:
        raise CatalogFormatError(f"Catalog line {line_number} has no parent_asin")

    title = str(record.get("title") or "").strip()
    categories = _string_list(record.get("categories"))
    features = _string_list(record.get("features") or record.get("bullet_point"))
    details_value = record.get("details")
    details = dict(details_value) if isinstance(details_value, dict) else {}
    description = _string_list(record.get("description"))
    store = str(record.get("store") or "").strip()

    return CatalogItem(
        parent_asin=parent_asin,
        title=title,
        categories=categories,
        features=features,
        details=details,
        description=description,
        price=_optional_float(record.get("price")),
        average_rating=_float_or_zero(record.get("average_rating")),
        rating_number=_int_or_zero(record.get("rating_number")),
        store=store,
        dense_text=clean_dense_text(
            title=title,
            categories=categories,
            features=features,
            details=details,
            description=description,
            store=store,
        ),
    )


def load_catalog(catalog_path: str | Path = "data/catalog.jsonl") -> CatalogData:
    """Parse the catalog once while preserving a validated, deterministic row order."""
    path = Path(catalog_path)
    if not path.is_file():
        raise FileNotFoundError(f"Catalog not found at {path}")

    items: list[CatalogItem] = []
    asin_list: list[str] = []
    dense_texts: list[str] = []
    items_by_asin: dict[str, CatalogItem] = {}
    asin_to_idx: dict[str, int] = {}
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError as error:
                raise CatalogFormatError(
                    f"Catalog line {line_number} is not valid UTF-8"
                ) from error
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CatalogFormatError(
                    f"Catalog line {line_number} is invalid JSON: {error.msg}"
                ) from error

            item = _parse_record(record, line_number=line_number)
            if item.parent_asin in items_by_asin:
                raise CatalogFormatError(
                    f"Duplicate parent_asin {item.parent_asin!r} at catalog line {line_number}"
                )

            asin_to_idx[item.parent_asin] = len(items)
            items_by_asin[item.parent_asin] = item
            items.append(item)
            asin_list.append(item.parent_asin)
            dense_texts.append(item.dense_text)

    if not items:
        raise CatalogFormatError(f"Catalog at {path} contains no products")

    return CatalogData(
        items=items,
        asin_list=asin_list,
        dense_texts=dense_texts,
        items_by_asin=items_by_asin,
        asin_to_idx=asin_to_idx,
        catalog_sha256=digest.hexdigest(),
    )
