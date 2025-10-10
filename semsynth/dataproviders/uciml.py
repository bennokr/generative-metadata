# ---------------------------
# UCI Machine Learning Repository via ucimlrepo
# ---------------------------


import json
import logging
import pathlib
from typing import List, Optional, Tuple

import pandas as pd
import requests

from ..specs import DatasetSpec


def list_uciml(
    area: str = "Health and Medicine",
    name_substr: Optional[str] = None,
    cat_min: int = 1,
    num_min: int = 1,
) -> pd.DataFrame:
    """Return (id, name, n_instances, n_categorical, n_numeric) for mixed datasets in area.

    It pulls the dataset list for the given area from the UCI API, then, for each
    dataset, fetches data via ucimlrepo and infers variable types to decide whether
    it is mixed (has at least one categorical and one numeric). Only mixed datasets
    are returned.
    """
    list_url = "https://archive.ics.uci.edu/api/datasets/list"
    logging.info(f"Requesting UCI ML datasets ({area=})")
    resp = requests.get(list_url, params={"area": area}, timeout=30)
    items = resp.json().get("data", []) if resp.ok else []
    pairs = [(int(d["id"]), d["name"]) for d in items]
    if name_substr:
        logging.info(f"Filtering UCI ML datasets ({name_substr=})")
        pairs = [(i, n) for i, n in pairs if name_substr.lower() in n.lower()]

    data_url = "https://archive.ics.uci.edu/api/dataset"
    cachedir = pathlib.Path(".") / "uciml-cache"
    cachedir.mkdir(exist_ok=True)
    rows = []
    for i, name in pairs:
        cache = cachedir / f"{i}.json"
        if not cache.exists():
            with cache.open("w") as fw:
                logging.info(f"Requesting UCI ML metadata {i} ({name})")
                r = requests.get(data_url, params={"id": i})
                if r.ok:
                    json.dump(r.json().get("data"), fw)
                else:
                    raise Exception(f"No content at {r}")
        metadata = json.load(cache.open())
        if any("type" in v for v in metadata["variables"]):
            vars = pd.DataFrame(metadata["variables"])
            row = {
                "id": i,
                "name": name,
                "n_instances": metadata["num_instances"],
                "n_categorical": vars["type"].isin(["Binary", "Categorical"]).sum(),
                "n_numeric": vars["type"].isin(["Integer", "Continuous"]).sum(),
            }
            if (row["n_categorical"] >= cat_min) and (row["n_numeric"] >= num_min):
                rows.append(row)

    return pd.DataFrame(rows)


def get_default_uciml(area: str = "Health and Medicine") -> List[DatasetSpec]:
    df = list_uciml(area=area)
    return [DatasetSpec("uciml", name=r.name, id=r.id) for r in df.itertuples()]


def load_uciml_by_id(
    dataset_id: int, cache_dir: pathlib.Path
) -> Tuple[DatasetSpec, pd.DataFrame, Optional[pd.Series]]:
    """Load a UCI ML dataset by ID, with local caching of the data payload.

    Caching layout:
      - {cache_dir}/{id}.csv.gz: cached tabular data
      - {cache_dir}/{id}.meta.json: minimal metadata (name, color column)
    """
    data_path = cache_dir / f"{dataset_id}.csv.gz"
    meta_path = cache_dir / f"{dataset_id}.meta.json"

    spec = DatasetSpec(provider="uciml", id=dataset_id)

    if data_path.exists():
        try:
            df_all = pd.read_csv(data_path)
            # Drop trivial id/index columns if present
            for col in list(df_all.columns):
                if str(col).lower() in {"id", "index"}:
                    df_all = df_all.drop(columns=[col])
            meta_dict = {}
            try:
                meta_dict = (
                    json.loads(meta_path.read_text()) if meta_path.exists() else {}
                )
            except Exception:
                meta_dict = {}
            # Color series from cached metadata or heuristic
            color_series: Optional[pd.Series] = None
            spec.target = meta_dict.get("target")
            if isinstance(spec.target, str) and spec.target in df_all.columns:
                color_series = df_all[spec.target]
            else:
                # Try common fallback names
                for c in ["target", "class"]:
                    if c in df_all.columns:
                        color_series = df_all[c]
                        break

            spec.name = meta_dict.get("name") or f"UCI_{dataset_id}"
            spec.meta = {
                "url": f"https://archive.ics.uci.edu/dataset/{int(dataset_id)}"
            }
            return spec, df_all, color_series
        except Exception:
            # Fall back to online path
            pass

    # Online fetch via ucimlrepo and then cache
    from ucimlrepo import fetch_ucirepo

    d = fetch_ucirepo(id=dataset_id)
    spec.meta = d.metadata
    X = d.data.features
    y = d.data.targets
    if y is None:
        df_all = X.copy()
        color_series = None
    else:
        df_all = pd.concat([X, y], axis=1)
        spec.target = (
            y.columns[0] if hasattr(y, "columns") and len(y.columns) else y.name
        )
        color_series = df_all[spec.target] if spec.target in df_all.columns else None
    for col in list(df_all.columns):
        if str(col).lower() in {"id", "index"}:
            df_all = df_all.drop(columns=[col])

    # Persist cache
    try:
        df_all.to_csv(data_path, index=False, compression="infer")
        spec.name = getattr(d.metadata, "name", f"UCI_{dataset_id}")
        meta_info = {
            "id": spec.id,
            "name": spec.name,
            "target": spec.target,
        }
        meta_path.write_text(json.dumps(meta_info))
    except Exception:
        pass

    return spec, df_all, color_series
