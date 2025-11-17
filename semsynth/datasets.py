from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

import pathlib

import pandas as pd
from makeprov import InPath, OutPath, rule

from .dataproviders.openml import get_default_openml, list_openml, load_openml_by_name
from .dataproviders.uciml import get_default_uciml, list_uciml, load_uciml_by_id
from .specs import DatasetSpec

__all__ = [
    "DatasetSpec",
    "specs_from_input",
    "load_dataset",
    "list_openml",
    "list_uciml",
]

# Local cache directories for dataset payloads (separate from uciml metadata cache)
_DATA_CACHE_ROOT = pathlib.Path(".") / "downloads-cache"
_OPENML_CACHE_DIR = _DATA_CACHE_ROOT / "openml"
_UCIML_CACHE_DIR = _DATA_CACHE_ROOT / "uciml"

for _d in (_OPENML_CACHE_DIR, _UCIML_CACHE_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _as_path(path_like: pathlib.Path | InPath | OutPath | str) -> pathlib.Path:
    """Coerce ``makeprov`` markers or strings into a :class:`~pathlib.Path`."""

    if isinstance(path_like, (InPath, OutPath)):
        return path_like.path or pathlib.Path(".")
    return pathlib.Path(path_like)


@rule()
def specs_from_input(
    provider: str,
    datasets: Optional[Iterable[str]] = None,
    area: str = "Health and Medicine",
    *,
    openml_cache_dir: OutPath = OutPath(str(_OPENML_CACHE_DIR)),
    uciml_cache_dir: OutPath = OutPath(str(_UCIML_CACHE_DIR)),
) -> List[DatasetSpec]:
    _ = (_as_path(openml_cache_dir), _as_path(uciml_cache_dir))
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


@rule()
def load_dataset(
    spec: DatasetSpec,
    *,
    openml_cache_dir: OutPath = OutPath(str(_OPENML_CACHE_DIR)),
    uciml_cache_dir: OutPath = OutPath(str(_UCIML_CACHE_DIR)),
) -> Tuple[Any, pd.DataFrame, Optional[pd.Series]]:
    if spec.provider == "openml":
        return load_openml_by_name(spec.name, _as_path(openml_cache_dir))
    elif spec.provider == "uciml":
        if spec.id is None:
            raise ValueError("uciml dataset requires an 'id'")
        return load_uciml_by_id(spec.id, _as_path(uciml_cache_dir))
    else:
        raise ValueError(f"Unknown provider: {spec.provider}")
