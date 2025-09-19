from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Union

import numpy as np
import pandas as pd
from zipfile import ZIP_DEFLATED, ZipFile

try:  # Optional dependency for unit-aware dtypes
    from pint_pandas import PintType  # type: ignore
except Exception:  # pragma: no cover - pint is optional
    PintType = None  # type: ignore[assignment]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_all(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _preserve_series_attrs(source: pd.Series, target: pd.Series) -> pd.Series:
    attrs = getattr(source, "attrs", None)
    if isinstance(attrs, dict) and attrs:
        target.attrs.update({k: v for k, v in attrs.items()})
    return target


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s)


def is_discrete_series(s: pd.Series, cardinality_threshold: int = 20) -> bool:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        try:
            n_uniq = s.nunique(dropna=True)
            return n_uniq <= cardinality_threshold
        except Exception:
            return False
    return False


def infer_types(df: pd.DataFrame, cardinality_threshold: int = 20) -> Tuple[List[str], List[str]]:
    disc, cont = [], []
    for c in df.columns:
        s = df[c]
        if is_discrete_series(s, cardinality_threshold):
            disc.append(c)
        elif is_numeric_series(s):
            cont.append(c)
        else:
            disc.append(c)
    return disc, cont


def coerce_discrete_to_category(df: pd.DataFrame, discrete_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in discrete_cols:
        s = df[c]
        if pd.api.types.is_categorical_dtype(s):
            continue
        converted = s.astype("category")
        df[c] = _preserve_series_attrs(s, converted)
    return df


def coerce_continuous_to_float(df: pd.DataFrame, continuous_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in continuous_cols:
        s = df[c]
        converted: Optional[pd.Series] = None
        if pd.api.types.is_integer_dtype(s):
            converted = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            if PintType is not None and isinstance(getattr(s, "dtype", None), PintType):
                try:
                    converted = pd.Series(s.astype("float64"), index=s.index, name=s.name)
                except Exception:
                    try:
                        converted = pd.Series(np.asarray(s), index=s.index, name=s.name).astype(float)
                    except Exception:
                        converted = None
        if converted is not None:
            df[c] = _preserve_series_attrs(s, converted)
    return df


def rename_categorical_categories_to_str(df: pd.DataFrame, discrete_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in discrete_cols:
        s = df[c]
        if pd.api.types.is_categorical_dtype(s):
            try:
                new_cats = [str(cat) for cat in s.cat.categories]
                converted = s.cat.rename_categories(new_cats)
            except Exception:
                mask = s.isna()
                tmp = s.astype(str)
                tmp[mask] = np.nan
                converted = tmp.astype("category")
            df[c] = _preserve_series_attrs(s, converted)
    return df


def summarize_dataframe(df: pd.DataFrame, discrete_cols: List[str], continuous_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        col = df[c]
        na_frac = float(col.isna().mean())
        uniq = int(col.nunique(dropna=True))
        if c in continuous_cols:
            desc = col.describe(percentiles=[0.25, 0.5, 0.75])
            mean = float(desc.get("mean", np.nan)) if not isinstance(desc, float) else float("nan")
            std = float(desc.get("std", np.nan)) if not isinstance(desc, float) else float("nan")
            minv = float(desc.get("min", np.nan)) if not isinstance(desc, float) else float("nan")
            q25 = float(desc.get("25%", np.nan)) if "25%" in desc else float("nan")
            q50 = float(desc.get("50%", np.nan)) if "50%" in desc else float("nan")
            q75 = float(desc.get("75%", np.nan)) if "75%" in desc else float("nan")
            maxv = float(desc.get("max", np.nan)) if not isinstance(desc, float) else float("nan")
            rows.append(
                dict(
                    variable=c,
                    type="continuous",
                    na_frac=na_frac,
                    unique=uniq,
                    mean=mean,
                    std=std,
                    min=minv,
                    q25=q25,
                    median=q50,
                    q75=q75,
                    max=maxv,
                )
            )
        else:
            top = col.value_counts(dropna=True).head(3)
            top_items = "; ".join([f"{k}:{int(v)}" for k, v in top.items()])
            rows.append(
                dict(variable=c, type="discrete", na_frac=na_frac, unique=uniq, top3=top_items)
            )
    return pd.DataFrame(rows)


def dataframe_to_markdown_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    def fmt(x):
        if isinstance(x, float):
            if math.isnan(x):
                return ""
            return float_fmt.format(x)
        return str(x)

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def pick_color_labels(series: Optional[pd.Series]) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    if series is None:
        return None, None
    values = series.astype("category")
    cats = list(values.cat.categories)
    mapping = {cat: i for i, cat in enumerate(cats)}
    labels = values.map(mapping).to_numpy()
    return labels, mapping


@dataclass
class UMAPArtifacts:
    preproc: object
    umap_model: object
    sample_idx: np.ndarray
    embedding: np.ndarray
    label_mapping: Optional[Dict]
    color_labels: Optional[np.ndarray]
