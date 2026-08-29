from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MATERIAL_WORDS = (
    'cotton', 'polyester', 'nylon', 'leather', 'wool', 'spandex',
    'silk', 'rayon', 'fabric', 'canvas', 'linen', 'denim', 'suede',
    'fleece', 'velvet', 'cashmere', 'rubber', 'mesh', 'satin'
)
MATERIAL_RE = re.compile(r'\b(' + '|'.join(MATERIAL_WORDS) + r')\b', re.I)

COLOR_WORDS = (
    'black', 'white', 'blue', 'red', 'pink', 'green', 'brown',
    'gray', 'grey', 'purple', 'yellow', 'orange', 'gold', 'silver',
    'navy', 'beige', 'khaki', 'tan', 'burgundy', 'maroon', 'teal', 'olive'
)
COLOR_RE = re.compile(r'\b(' + '|'.join(COLOR_WORDS) + r')\b', re.I)

DEPARTMENT_WORDS = ('womens', 'mens', 'women', 'men', 'unisex', 'girls', 'boys', 'kids', 'baby')
DEPARTMENT_RE = re.compile(r'\b(' + '|'.join(DEPARTMENT_WORDS) + r')\b', re.I)


def _flatten_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, dict):
        return ' '.join(f'{k}: {v}' for k, v in value.items() if v not in (None, '', []))
    if isinstance(value, list):
        return ' '.join(str(item) for item in value if item not in (None, ''))
    return str(value).strip()


def _extract_materials(text: str) -> list[str]:
    return list(dict.fromkeys(m.lower() for m in MATERIAL_RE.findall(text)))


def _extract_colors(text: str) -> list[str]:
    return list(dict.fromkeys(c.lower() for c in COLOR_RE.findall(text)))


def _extract_department(text: str) -> str | None:
    match = DEPARTMENT_RE.search(text)
    return match.group(1).lower() if match else None


def clean_dense_text(
    title: str,
    categories: list[str],
    features: list[str] | str,
    details: dict[str, Any] | str,
    description: list[str] | str,
    store: str = '',
) -> str:
    parts: list[str] = []
    if title:
        parts.append(f'Title: {title.strip()}')
    
    cat_str = _flatten_text(categories)
    if cat_str:
        parts.append(f'Categories: {cat_str}')
        
    feat_str = _flatten_text(features)
    if feat_str:
        parts.append(f'Features: {feat_str}')
        
    det_str = _flatten_text(details)
    if det_str:
        parts.append(f'Details: {det_str}')
        
    if store:
        parts.append(f'Store: {store.strip()}')
        
    desc_str = _flatten_text(description)
    if desc_str:
        parts.append(f'Description: {desc_str}')
        
    dense = ' | '.join(parts)
    return re.sub(r'\s+', ' ', dense).strip()


@dataclass
class CatalogItem:
    parent_asin: str
    title: str
    categories: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    description: list[str] = field(default_factory=list)
    price: float | None = None
    average_rating: float = 0.0
    rating_number: int = 0
    store: str = ''
    dense_text: str = ''
    materials: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    department: str | None = None


@dataclass
class CatalogData:
    asin_list: list[str]
    dense_texts: list[str]
    items_by_asin: dict[str, CatalogItem]
    asin_to_idx: dict[str, int]


def load_catalog(catalog_path: str | Path = 'data/catalog.jsonl') -> CatalogData:
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f'Catalog not found at {path}')
        
    asin_list: list[str] = []
    dense_texts: list[str] = []
    items_by_asin: dict[str, CatalogItem] = {}
    asin_to_idx: dict[str, int] = {}
    
    with path.open(encoding='utf-8') as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
                
            parent_asin = str(record.get('parent_asin', '')).strip()
            if not parent_asin:
                continue
                
            title = str(record.get('title') or '').strip()
            categories = [str(c).strip() for c in (record.get('categories') or []) if str(c).strip()]
            
            raw_features = record.get('features') or record.get('bullet_point') or []
            if isinstance(raw_features, str):
                features = [raw_features]
            elif isinstance(raw_features, list):
                features = [str(f).strip() for f in raw_features if str(f).strip()]
            else:
                features = []
                
            details = record.get('details') if isinstance(record.get('details'), dict) else {}
            
            raw_desc = record.get('description') or []
            if isinstance(raw_desc, str):
                description = [raw_desc]
            elif isinstance(raw_desc, list):
                description = [str(d).strip() for d in raw_desc if str(d).strip()]
            else:
                description = []
                
            raw_price = record.get('price')
            try:
                price = float(raw_price) if raw_price is not None and str(raw_price).strip() != '' else None
            except (ValueError, TypeError):
                price = None
                
            try:
                avg_rating = float(record.get('average_rating') or 0.0)
            except (ValueError, TypeError):
                avg_rating = 0.0
                
            try:
                rating_num = int(record.get('rating_number') or 0)
            except (ValueError, TypeError):
                rating_num = 0
                
            store = str(record.get('store') or '').strip()
            
            dense = clean_dense_text(
                title=title,
                categories=categories,
                features=features,
                details=details,
                description=description,
                store=store,
            )
            
            corpus_for_extract = f'{dense} {_flatten_text(details)}'
            materials = _extract_materials(corpus_for_extract)
            colors = _extract_colors(corpus_for_extract)
            dept = _extract_department(f'{_flatten_text(categories)} {_flatten_text(details)}')
            
            item = CatalogItem(
                parent_asin=parent_asin,
                title=title,
                categories=categories,
                features=features,
                details=details,
                description=description,
                price=price,
                average_rating=avg_rating,
                rating_number=rating_num,
                store=store,
                dense_text=dense,
                materials=materials,
                colors=colors,
                department=dept,
            )
            
            items_by_asin[parent_asin] = item
            asin_list.append(parent_asin)
            dense_texts.append(dense)
            asin_to_idx[parent_asin] = len(asin_list) - 1
            
    return CatalogData(
        asin_list=asin_list,
        dense_texts=dense_texts,
        items_by_asin=items_by_asin,
        asin_to_idx=asin_to_idx,
    )
