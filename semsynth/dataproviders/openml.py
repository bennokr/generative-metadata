# ---------------------------
# OpenML
# ---------------------------

import json
import logging
import pathlib
import warnings
from typing import List, Optional, Tuple

import pandas as pd

from ..specs import DatasetSpec


def get_default_openml() -> List[DatasetSpec]:
    return [
        DatasetSpec("openml", "adult", target="class"),
        DatasetSpec("openml", "credit-g", target="class"),
        DatasetSpec("openml", "titanic", target="survived"),
        DatasetSpec("openml", "bank-marketing", target="y"),
    ]


def list_openml(
    name_substr: Optional[str] = None, cat_min: int = 1, num_min: int = 1
) -> pd.DataFrame:
    import openml

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        logging.info("Requesting OpenML datasets")
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
        logging.info(f"Filtering OpenML datasets ({name_substr=})")
        mask = sets["name"].str.contains(name_substr, case=False, na=False)
        sets = sets.loc[mask]
    sets = sets.sort_values(["name", "version"]).drop_duplicates("name", keep="last")
    rename = {
        "did": "id",
        "NumberOfInstances": "n_instances",
        "NumberOfSymbolicFeatures": "n_categorical",
        "NumberOfNumericFeatures": "n_numeric",
    }
    return sets.rename(columns=rename)


def load_openml_by_name(
    name: str, cache_dir: pathlib.Path
) -> Tuple[DatasetSpec, pd.DataFrame, Optional[pd.Series]]:
    """Load an OpenML dataset by name, with local caching of the data payload.

    Caching layout:
      - downloads-cache/openml/by_name/{name}.json: minimal metadata with DID and color column
      - downloads-cache/openml/{did}.csv.gz: cached tabular data

    On cache hit, returns a minimal metadata dict instead of the OpenML object.
    """

    by_name_dir = cache_dir / "by_name"
    by_name_dir.mkdir(parents=True, exist_ok=True)

    spec = DatasetSpec(provider="openml", name=name)

    # 1) Try cache-by-name first (offline-friendly)
    by_name_meta = by_name_dir / f"{name}.json"
    if by_name_meta.exists():
        try:
            info = json.loads(by_name_meta.read_text())
            spec.id = int(info.get("did") or info.get("dataset_id") or info.get("id"))
            data_path = by_name_dir / f"{spec.id}.csv.gz"
            if data_path.exists():
                df_all = pd.read_csv(data_path).convert_dtypes()
                for col in list(df_all.columns):
                    if str(col).lower() in {"id", "index"}:
                        df_all = df_all.drop(columns=[col])
                color_series = None
                spec.target = info.get("target")
                if isinstance(spec.target, str) and spec.target in df_all.columns:
                    color_series = df_all[spec.target]
                else:
                    for c in ["class", "target"]:
                        if c in df_all.columns:
                            color_series = df_all[c]
                            break
                spec.name = str(info.get("name") or name)
                spec.meta = info
                return spec, df_all, color_series
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
    spec.id = int(df.sort_values("version", ascending=False).iloc[0]["did"])
    spec.meta = openml.datasets.get_dataset(spec.id)
    Xy, _, _, _ = spec.meta.get_data(dataset_format="dataframe")
    df_all = Xy.copy()
    for col in list(df_all.columns):
        if str(col).lower() in {"id", "index"}:
            df_all = df_all.drop(columns=[col])
    color_series = None

    for c in ["class", "target"]:
        if c in df_all.columns:
            color_series = df_all[c]
            spec.target = c
            break

    # Persist cache
    try:
        data_path = cache_dir / f"{spec.id}.csv.gz"
        df_all.to_csv(data_path, index=False, compression="infer")
        by_name_meta.write_text(
            json.dumps(
                {
                    "name": name,
                    "id": spec.id,
                    "target": spec.target,
                }
            )
        )
    except Exception:
        pass

    return spec, df_all, color_series
