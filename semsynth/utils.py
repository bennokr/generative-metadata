from __future__ import annotations

import os
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # Optional dependency for unit-aware dtypes
    from pint_pandas import PintType  # type: ignore
except Exception:  # pragma: no cover - pint is optional
    PintType = None  # type: ignore[assignment]


def ensure_dir(path: str) -> None:
    """Create the directory at ``path`` if it is missing.

    Args:
        path: Filesystem location to create.

    Examples:
        >>> import os, tempfile
        >>> base = tempfile.mkdtemp()
        >>> target = os.path.join(base, "nested")
        >>> ensure_dir(target)
        >>> os.path.isdir(target)
        True
    """

    os.makedirs(path, exist_ok=True)


def seed_all(seed: int) -> np.random.Generator:
    """Create a deterministic NumPy random generator.

    Args:
        seed: Random seed used for the generator.

    Returns:
        A ``numpy.random.Generator`` seeded with ``seed``.

    Examples:
        >>> rng = seed_all(0)
        >>> rng.integers(0, 10, size=3).tolist()
        [8, 6, 5]
    """

    return np.random.default_rng(seed)


def _preserve_series_attrs(source: pd.Series, target: pd.Series) -> pd.Series:
    attrs = getattr(source, "attrs", None)
    if isinstance(attrs, dict) and attrs:
        target.attrs.update({k: v for k, v in attrs.items()})
    return target


def is_numeric_series(s: pd.Series) -> bool:
    """Return ``True`` when ``s`` represents a numeric series.

    Examples:
        >>> import pandas as pd
        >>> is_numeric_series(pd.Series([1.0, 2.5]))
        True
    """

    return pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s)


def is_discrete_series(s: pd.Series, cardinality_threshold: int = 20) -> bool:
    """Return ``True`` if ``s`` should be treated as discrete.

    Args:
        s: Series to inspect.
        cardinality_threshold: Maximum unique values before treating as continuous.

    Returns:
        Whether the series is discrete.

    Examples:
        >>> import pandas as pd
        >>> is_discrete_series(pd.Series(["a", "b", "a"]))
        True
    """

    if (
        pd.api.types.is_bool_dtype(s)
        or pd.api.types.is_categorical_dtype(s)
        or pd.api.types.is_object_dtype(s)
    ):
        return True
    if pd.api.types.is_integer_dtype(s):
        try:
            n_uniq = s.nunique(dropna=True)
            return n_uniq <= cardinality_threshold
        except Exception:
            return False
    return False


def infer_types(
    df: pd.DataFrame, cardinality_threshold: int = 20
) -> Tuple[List[str], List[str]]:
    """Split dataframe columns into discrete and continuous lists.

    Args:
        df: DataFrame to analyse.
        cardinality_threshold: Threshold used by :func:`is_discrete_series`.

    Returns:
        Two lists containing discrete and continuous column names.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": ["x", "y", "z"]})
        >>> infer_types(df)
        (['a', 'c'], ['b'])
    """

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


def coerce_discrete_to_category(
    df: pd.DataFrame, discrete_cols: List[str]
) -> pd.DataFrame:
    """Convert selected columns to the categorical dtype.

    Args:
        df: Source dataframe.
        discrete_cols: Columns expected to be discrete.

    Returns:
        A copy of ``df`` with categorical conversions applied.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": [1, 1, 2], "b": [0.1, 0.2, 0.3]})
        >>> converted = coerce_discrete_to_category(df, ["a"])
        >>> str(converted.dtypes['a'])
        'category'
    """

    df = df.copy()
    for c in discrete_cols:
        s = df[c]
        if pd.api.types.is_categorical_dtype(s):
            continue
        converted = s.astype("category")
        df[c] = _preserve_series_attrs(s, converted)
    return df


def coerce_continuous_to_float(
    df: pd.DataFrame, continuous_cols: List[str]
) -> pd.DataFrame:
    """Convert continuous columns to floating point when possible.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"value": [1, 2, 3]})
        >>> converted = coerce_continuous_to_float(df, ["value"])
        >>> str(converted.dtypes['value'])
        'float64'
    """

    df = df.copy()
    for c in continuous_cols:
        s = df[c]
        converted: Optional[pd.Series] = None
        if pd.api.types.is_integer_dtype(s):
            numeric = pd.to_numeric(s, errors="coerce")
            converted = pd.Series(
                np.asarray(numeric, dtype="float64"), index=s.index, name=s.name
            )
        else:
            if PintType is not None and isinstance(getattr(s, "dtype", None), PintType):
                try:
                    converted = pd.Series(
                        s.astype("float64"), index=s.index, name=s.name
                    )
                except Exception:
                    try:
                        converted = pd.Series(
                            np.asarray(s), index=s.index, name=s.name
                        ).astype(float)
                    except Exception:
                        converted = None
        if converted is not None:
            df[c] = _preserve_series_attrs(s, converted)
    return df


def rename_categorical_categories_to_str(
    df: pd.DataFrame, discrete_cols: List[str]
) -> pd.DataFrame:
    """Ensure categorical levels are strings to keep outputs JSON-friendly.

    Examples:
        >>> import pandas as pd
        >>> cat = pd.Series(pd.Categorical([1, 2, 1], categories=[1, 2]))
        >>> renamed = rename_categorical_categories_to_str(pd.DataFrame({"value": cat}), ["value"])
        >>> list(renamed['value'].cat.categories)
        ['1', '2']
    """

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


def summarize_dataframe(
    df: pd.DataFrame, discrete_cols: List[str], continuous_cols: List[str]
) -> pd.DataFrame:
    """Summarise dataframe columns with statistics tailored by type.

    Args:
        df: Dataframe to summarise.
        discrete_cols: Columns considered discrete.
        continuous_cols: Columns considered continuous.

    Returns:
        A dataframe with summary statistics per column.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"grade": ["A", "B", "A"], "score": [0.1, 0.2, 0.3]})
        >>> summary = summarize_dataframe(df, ["grade"], ["score"])
        >>> summary.loc[summary['variable'] == 'grade', 'top3'].iloc[0]
        'A:2; B:1'
    """

    rows = []
    for c in df.columns:
        col = df[c]
        na_frac = float(col.isna().mean())
        uniq = int(col.nunique(dropna=True))
        if c in continuous_cols:
            desc = col.describe(percentiles=[0.25, 0.5, 0.75])
            mean = (
                float(desc.get("mean", np.nan))
                if not isinstance(desc, float)
                else float("nan")
            )
            std = (
                float(desc.get("std", np.nan))
                if not isinstance(desc, float)
                else float("nan")
            )
            minv = (
                float(desc.get("min", np.nan))
                if not isinstance(desc, float)
                else float("nan")
            )
            q25 = float(desc.get("25%", np.nan)) if "25%" in desc else float("nan")
            q50 = float(desc.get("50%", np.nan)) if "50%" in desc else float("nan")
            q75 = float(desc.get("75%", np.nan)) if "75%" in desc else float("nan")
            maxv = (
                float(desc.get("max", np.nan))
                if not isinstance(desc, float)
                else float("nan")
            )
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
                dict(
                    variable=c,
                    type="discrete",
                    na_frac=na_frac,
                    unique=uniq,
                    top3=top_items,
                )
            )
    return pd.DataFrame(rows)


def dataframe_to_markdown_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Render a dataframe as a GitHub-flavoured markdown table.

    Args:
        df: Table to render.
        float_fmt: Format string applied to floating values.

    Returns:
        Markdown string representing the table.

    Examples:
        >>> import pandas as pd
        >>> table = pd.DataFrame({"variable": ["grade"], "type": ["discrete"]})
        >>> print(dataframe_to_markdown_table(table))
        | variable | type |
        | --- | --- |
        | grade | discrete |
    """

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
