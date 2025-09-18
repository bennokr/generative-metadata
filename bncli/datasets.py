from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Any

import pandas as pd
import requests
import pathlib
import json
import logging

# Local cache directories for dataset payloads (separate from uciml metadata cache)
_DATA_CACHE_ROOT = pathlib.Path('.') / 'downloads-cache'
_OPENML_CACHE_DIR = _DATA_CACHE_ROOT / 'openml'
_OPENML_BY_NAME_DIR = _OPENML_CACHE_DIR / 'by_name'
_UCIML_CACHE_DIR = _DATA_CACHE_ROOT / 'uciml'

for _d in (_OPENML_CACHE_DIR, _OPENML_BY_NAME_DIR, _UCIML_CACHE_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

@dataclass
class DatasetSpec:
    provider: str  # 'openml' or 'uciml'
    name: str
    target: Optional[str] = None
    id: Optional[int] = None  # used by uciml


# ---------------------------
# OpenML
# ---------------------------


def get_default_openml() -> List[DatasetSpec]:
    return [
        DatasetSpec("openml", "adult", target="class"),
        DatasetSpec("openml", "credit-g", target="class"),
        DatasetSpec("openml", "titanic", target="survived"),
        DatasetSpec("openml", "bank-marketing", target="y"),
    ]


def list_openml(name_substr: Optional[str] = None, cat_min: int = 1, num_min: int = 1) -> pd.DataFrame:
    import openml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        logging.info(f'Requesting OpenML datasets')
        df = openml.datasets.list_datasets(output_format="dataframe", status="active")
        numcols = df.select_dtypes(include="float").columns
        df = df.astype({col: "Int64" for col in numcols})
    need = (df["NumberOfSymbolicFeatures"].fillna(0) >= cat_min) & (
        df["NumberOfNumericFeatures"].fillna(0) >= num_min
    )
    sets = df.loc[
        need,
        [
            "did",
            "name",
            "version",
            "NumberOfInstances",
            "NumberOfSymbolicFeatures",
            "NumberOfNumericFeatures",
        ],
    ]
    if name_substr:
        logging.info(f'Filtering OpenML datasets ({name_substr=})')
        mask = sets["name"].str.contains(name_substr, case=False, na=False)
        sets = sets.loc[mask]
    sets = sets.sort_values(["name", "version"]).drop_duplicates("name", keep="last")
    rename = {
        'did': 'id',
        'NumberOfInstances':'n_instances', 
        'NumberOfSymbolicFeatures':'n_categorical', 
        'NumberOfNumericFeatures':'n_numeric'}
    return sets.rename(columns=rename)


def load_openml_by_name(name: str) -> Tuple[Any, pd.DataFrame, Optional[pd.Series]]:
    """Load an OpenML dataset by name, with local caching of the data payload.

    Caching layout:
      - downloads-cache/openml/by_name/{name}.json: minimal metadata with DID and color column
      - downloads-cache/openml/{did}.csv.gz: cached tabular data

    On cache hit, returns a minimal metadata dict instead of the OpenML object.
    """
    # 1) Try cache-by-name first (offline-friendly)
    by_name_meta = _OPENML_BY_NAME_DIR / f"{name}.json"
    if by_name_meta.exists():
        try:
            info = json.loads(by_name_meta.read_text())
            did = int(info.get('did') or info.get('dataset_id') or info.get('id'))
            data_path = _OPENML_CACHE_DIR / f"{did}.csv.gz"
            if data_path.exists():
                df_all = pd.read_csv(data_path)
                for col in list(df_all.columns):
                    if str(col).lower() in {"id", "index"}:
                        df_all = df_all.drop(columns=[col])
                color_series = None
                color_col = info.get('color_column')
                if isinstance(color_col, str) and color_col in df_all.columns:
                    color_series = df_all[color_col]
                else:
                    for c in ["class", "target"]:
                        if c in df_all.columns:
                            color_series = df_all[c]
                            break
                meta = {
                    'name': str(info.get('name') or name),
                    'dataset_id': did,
                    'did': did,
                    'url': f'https://www.openml.org/d/{did}',
                }
                return meta, df_all, color_series
        except Exception:
            pass

    # 2) Fallback to OpenML API and cache results
    import openml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df = openml.datasets.list_datasets(
            output_format="dataframe", status="active", data_name=name
        )
    df = df[df["name"] == name]
    if df.empty:
        raise ValueError(f"No active OpenML dataset named {name!r}.")
    did = int(df.sort_values("version", ascending=False).iloc[0]["did"])
    ds = openml.datasets.get_dataset(did)
    Xy, _, _, _ = ds.get_data(dataset_format="dataframe")
    df_all = Xy.copy()
    for col in list(df_all.columns):
        if str(col).lower() in {"id", "index"}:
            df_all = df_all.drop(columns=[col])
    color_series = None
    color_col: Optional[str] = None
    for c in ["class", "target"]:
        if c in df_all.columns:
            color_series = df_all[c]
            color_col = c
            break

    # Persist cache
    try:
        data_path = _OPENML_CACHE_DIR / f"{did}.csv.gz"
        df_all.to_csv(data_path, index=False, compression='infer')
        by_name_meta.write_text(json.dumps({
            'name': name,
            'did': did,
            'dataset_id': did,
            'color_column': color_col,
        }))
    except Exception:
        pass

    return ds, df_all, color_series


# ---------------------------
# UCI Machine Learning Repository via ucimlrepo
# ---------------------------


def list_uciml(
    area: str = "Health and Medicine", name_substr: Optional[str] = None, 
    cat_min: int = 1, num_min: int = 1
) -> pd.DataFrame:
    """Return (id, name, n_instances, n_categorical, n_numeric) for mixed datasets in area.

    It pulls the dataset list for the given area from the UCI API, then, for each
    dataset, fetches data via ucimlrepo and infers variable types to decide whether
    it is mixed (has at least one categorical and one numeric). Only mixed datasets
    are returned.
    """
    list_url = "https://archive.ics.uci.edu/api/datasets/list"
    logging.info(f'Requesting UCI ML datasets ({area=})')
    resp = requests.get(list_url, params={"area": area}, timeout=30)
    items = resp.json().get("data", []) if resp.ok else []
    pairs = [(int(d["id"]), d["name"]) for d in items]
    if name_substr:
        logging.info(f'Filtering UCI ML datasets ({name_substr=})')
        pairs = [(i, n) for i, n in pairs if name_substr.lower() in n.lower()]

    data_url = "https://archive.ics.uci.edu/api/dataset"
    cachedir = pathlib.Path('.') / 'uciml-cache'
    cachedir.mkdir(exist_ok=True)
    rows = []
    for i, name in pairs:
        cache = (cachedir / f'{i}.json')
        if not cache.exists():
            with cache.open('w') as fw:
                logging.info(f'Requesting UCI ML metadata {i} ({name})')
                r = requests.get(data_url, params={'id':i})
                if r.ok:
                    json.dump(r.json().get('data'), fw)
                else:
                    raise Exception(f'No content at {r}')
        metadata = json.load(cache.open())
        if any('type' in v for v in metadata['variables']):
            vars = pd.DataFrame(metadata['variables'])
            row = {
                'id': i,
                'name': name,
                'n_instances': metadata['num_instances'],
                'n_categorical': vars['type'].isin(['Binary', 'Categorical']).sum(),
                'n_numeric': vars['type'].isin(['Integer', 'Continuous']).sum(),
            }
            if (row['n_categorical'] >= cat_min) and (row['n_numeric'] >= num_min):
                rows.append(row)

    return pd.DataFrame(rows)


def get_default_uciml(area: str = "Health and Medicine") -> List[DatasetSpec]:
    df = list_uciml(area=area)
    return [DatasetSpec("uciml", name=r.name, id=r.id) for r in df.itertuples()]


def load_uciml_by_id(dataset_id: int) -> Tuple[Any, pd.DataFrame, Optional[pd.Series]]:
    """Load a UCI ML dataset by ID, with local caching of the data payload.

    Caching layout:
      - downloads-cache/uciml/{id}.csv.gz: cached tabular data
      - downloads-cache/uciml/{id}.meta.json: minimal metadata (name, color column)
    """
    data_path = _UCIML_CACHE_DIR / f"{dataset_id}.csv.gz"
    meta_path = _UCIML_CACHE_DIR / f"{dataset_id}.meta.json"

    if data_path.exists():
        try:
            df_all = pd.read_csv(data_path)
            # Drop trivial id/index columns if present
            for col in list(df_all.columns):
                if str(col).lower() in {"id", "index"}:
                    df_all = df_all.drop(columns=[col])
            meta_dict = {}
            try:
                meta_dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            except Exception:
                meta_dict = {}
            # Color series from cached metadata or heuristic
            color_series: Optional[pd.Series] = None
            color_col = meta_dict.get('color_column') if isinstance(meta_dict, dict) else None
            if isinstance(color_col, str) and color_col in df_all.columns:
                color_series = df_all[color_col]
            else:
                # Try common fallback names
                for c in ["target", "class"]:
                    if c in df_all.columns:
                        color_series = df_all[c]
                        break
            # Metadata: prefer uciml cached API if available for name
            meta_name = None
            try:
                cache_api = pathlib.Path('.') / 'uciml-cache' / f'{dataset_id}.json'
                if cache_api.exists():
                    api_data = json.loads(cache_api.read_text())
                    meta_name = api_data.get('name')
            except Exception:
                pass
            meta_obj = {
                'id': int(dataset_id),
                'uci_id': int(dataset_id),
                'name': meta_name or meta_dict.get('name') or f'UCI_{dataset_id}',
                'url': f'https://archive.ics.uci.edu/dataset/{int(dataset_id)}',
            }
            return meta_obj, df_all, color_series
        except Exception:
            # Fall back to online path
            pass

    # Online fetch via ucimlrepo and then cache
    from ucimlrepo import fetch_ucirepo

    d = fetch_ucirepo(id=dataset_id)
    X = d.data.features
    y = d.data.targets
    if y is None:
        df_all = X.copy()
        color_series = None
        first_target: Optional[str] = None
    else:
        df_all = pd.concat([X, y], axis=1)
        first_target = y.columns[0] if hasattr(y, "columns") and len(y.columns) else y.name
        color_series = df_all[first_target] if first_target in df_all.columns else None
    for col in list(df_all.columns):
        if str(col).lower() in {"id", "index"}:
            df_all = df_all.drop(columns=[col])

    # Persist cache
    try:
        df_all.to_csv(data_path, index=False, compression='infer')
        name_val = None
        try:
            name_val = getattr(d.metadata, 'name', None)
        except Exception:
            name_val = None
        meta_info = {
            'id': int(dataset_id),
            'uci_id': int(dataset_id),
            'name': name_val,
            'color_column': first_target,
        }
        meta_path.write_text(json.dumps(meta_info))
    except Exception:
        pass

    return d.metadata, df_all, color_series


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
        return load_openml_by_name(spec.name)
    elif spec.provider == "uciml":
        if spec.id is None:
            raise ValueError("uciml dataset requires an 'id'")
        return load_uciml_by_id(spec.id)
    else:
        raise ValueError(f"Unknown provider: {spec.provider}")
