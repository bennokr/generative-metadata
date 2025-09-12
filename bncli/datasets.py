from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests
import pathlib
import json
import logging

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


def load_openml_by_name(name: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
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
    for c in ["class", "target"]:
        if c in df_all.columns:
            color_series = df_all[c]
            break
    return ds,  df_all, color_series


# ---------------------------
# UCI Machine Learning Repository via ucimlrepo
# ---------------------------


def list_uciml(
    area: str = "Health and Medicine", name_substr: Optional[str] = None, 
    cat_min: int = 1, num_min: int = 1
) -> List[Tuple[int, str, int, int, int]]:
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


def load_uciml_by_id(dataset_id: int) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    from ucimlrepo import fetch_ucirepo

    d = fetch_ucirepo(id=dataset_id)
    X = d.data.features
    y = d.data.targets
    if y is None:
        df_all = X.copy()
        color_series = None
    else:
        df_all = pd.concat([X, y], axis=1)
        first_target = y.columns[0] if hasattr(y, "columns") and len(y.columns) else y.name
        color_series = df_all[first_target] if first_target in df_all.columns else None
    for col in list(df_all.columns):
        if str(col).lower() in {"id", "index"}:
            df_all = df_all.drop(columns=[col])
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


def load_dataset(spec: DatasetSpec) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    if spec.provider == "openml":
        return load_openml_by_name(spec.name)
    elif spec.provider == "uciml":
        if spec.id is None:
            raise ValueError("uciml dataset requires an 'id'")
        return load_uciml_by_id(spec.id)
    else:
        raise ValueError(f"Unknown provider: {spec.provider}")
