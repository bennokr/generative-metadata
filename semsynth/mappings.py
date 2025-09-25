from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import DatasetSpec

JSONLD_CONTEXT_URL = "https://w3id.org/semmap/context/v1"


def _mappings_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "mappings"


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def resolve_mapping_json(dataset_spec: DatasetSpec) -> Optional[Path]:
    """Return the curated JSON-LD mapping path for the dataset if it exists."""
    mappings_dir = _mappings_dir()
    candidates = []

    provider = dataset_spec.provider
    provider_id = dataset_spec.id
    dataset_name = dataset_spec.name

    provider_norm = provider.lower().strip() if isinstance(provider, str) and provider.strip() else None
    if provider_norm and provider_norm not in {"openml", "uciml"}:
        provider_norm = _slugify(provider_norm)
    elif provider_norm:
        provider_norm = provider_norm

    if provider_norm and provider_id is not None:
        candidates.append(mappings_dir / f"{provider_norm}-{provider_id}.metadata.json")

    if provider_norm and isinstance(dataset_name, str) and dataset_name.strip():
        candidates.append(mappings_dir / f"{provider_norm}-{_slugify(dataset_name)}.metadata.json")

    if isinstance(dataset_name, str) and dataset_name.strip():
        candidates.append(mappings_dir / f"{_slugify(dataset_name)}.metadata.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_mapping_json(path: Path) -> Dict[str, Any]:
    """Load and minimally validate a curated SemMap JSON-LD mapping."""
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Mapping JSON must be an object")

    return data
