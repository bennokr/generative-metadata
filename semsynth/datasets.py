from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Any

import pandas as pd
import pathlib
import logging

from metadata import DCATSchema
from dataproviders.openml import get_default_openml
from semsynth.dataproviders.openml import load_openml_by_name
from semsynth.dataproviders.uciml import get_default_uciml, load_uciml_by_id

# Local cache directories for dataset payloads (separate from uciml metadata cache)
_DATA_CACHE_ROOT = pathlib.Path('.') / 'downloads-cache'
_OPENML_CACHE_DIR = _DATA_CACHE_ROOT / 'openml'
_UCIML_CACHE_DIR = _DATA_CACHE_ROOT / 'uciml'

for _d in (_OPENML_CACHE_DIR, _UCIML_CACHE_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

@dataclass
class DatasetSpec:
    provider: str  # 'openml' or 'uciml'
    name: Optional[str] = None
    id: Optional[int] = None
    target: Optional[str] = None
    meta: Optional[Any] = None



def specs_from_input(
    provider: str,
    datasets: Optional[Iterable[str]] = None,
    area: str = "Health and Medicine",
) -> List[DatasetSpec]:
    provider = provider.lower()
    if provider not in {"openml", "uciml"}:
        raise ValueError("provider must be 'openml' or 'uciml'")
    if datasets:
        if provider == "openml":
            return [DatasetSpec("openml", name=d) for d in datasets]
        else:
            ids: List[DatasetSpec] = []
            for d in datasets:
                try:
                    ids.append(DatasetSpec("uciml", name=None, id=int(d)))
                except ValueError:
                    raise ValueError(
                        "For uciml provider, datasets must be numeric IDs (as strings)."
                    )
            return ids
    else:
        if provider == "openml":
            return get_default_openml()
        else:
            return get_default_uciml(area=area)


def load_dataset(spec: DatasetSpec) -> Tuple[Any, pd.DataFrame, Optional[pd.Series]]:
    if spec.provider == "openml":
        return load_openml_by_name(spec.name, _OPENML_CACHE_DIR)
    elif spec.provider == "uciml":
        if spec.id is None:
            raise ValueError("uciml dataset requires an 'id'")
        return load_uciml_by_id(spec.id, _UCIML_CACHE_DIR)
    else:
        raise ValueError(f"Unknown provider: {spec.provider}")
