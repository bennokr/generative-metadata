from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

JSONLD_CONTEXT_URL = "https://w3id.org/semmap/context/v1"


def _mappings_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "mappings"


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def resolve_mapping_json(
    provider: Optional[str],
    provider_id: Optional[int],
    dataset_name: Optional[str],
) -> Optional[Path]:
    """Return the curated JSON-LD mapping path for the dataset if it exists."""
    mappings_dir = _mappings_dir()
    candidates = []

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


def canonical_generator_name(name: str) -> str:
    """Return the canonical synthcity plugin name for a user alias."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Generator name must be non-empty")
    aliases = {
        "ctgan": "ctgan",
        "ads-gan": "adsgan",
        "adsgan": "adsgan",
        "pategan": "pategan",
        "dp-gan": "dpgan",
        "dpgan": "dpgan",
        "tvae": "tvae",
        "rtvae": "rtvae",
        "nflow": "nflow",
        "tabularflow": "tabularflow",
        "bn": "bayesiannetwork",
        "bayesiannetwork": "bayesiannetwork",
        "privbayes": "privbayes",
        "arf": "arf",
        "arfpy": "arf",
        "great": "great",
    }
    if key not in aliases:
        raise ValueError(f"Unknown generator alias: {name}")
    return aliases[key]
