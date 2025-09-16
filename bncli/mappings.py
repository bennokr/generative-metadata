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

    dataset = data.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Mapping JSON missing 'dataset' object")
    dataset = dict(dataset)
    dataset.setdefault("@context", JSONLD_CONTEXT_URL)

    var_list = None
    for key in ("disco:variable", "variable", "variables", "columns"):
        val = data.get(key)
        if isinstance(val, list):
            var_list = val
            break
    if var_list is None:
        for key in ("disco:variable", "variable", "variables", "columns"):
            val = dataset.get(key)
            if isinstance(val, list):
                var_list = val
                break
    if var_list is None:
        raise ValueError("Mapping JSON missing disco:variable list")

    normalized_vars = []
    for idx, entry in enumerate(var_list):
        if not isinstance(entry, dict):
            raise ValueError(f"Variable entry at index {idx} must be an object")
        var = dict(entry)
        var.setdefault("@context", dataset["@context"])
        notation = None
        for key in ("skos:notation", "notation", "name"):
            val = var.get(key)
            if isinstance(val, str) and val.strip():
                notation = val.strip()
                break
        if not notation:
            raise ValueError(f"Variable entry at index {idx} missing notation/name")
        normalized_vars.append(var)

    return {"dataset": dataset, "disco:variable": normalized_vars}
