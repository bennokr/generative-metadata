"""Missingness modeling utilities for backend generators."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from .metrics import per_variable_distances, summarize_distance_metrics


def _make_one_hot_encoder() -> OneHotEncoder:
    """Instantiate a dense ``OneHotEncoder`` compatible with sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - fallback for older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass
class ColumnMissingnessModel:
    """Estimate conditional missingness for a single column."""

    col: str
    p_missing_: float = 0.0
    pipeline_: Optional[Pipeline] = None

    def fit(self, df: pd.DataFrame) -> "ColumnMissingnessModel":
        """Fit the column-level missingness model.

        Args:
            df: Real dataframe that may contain missing values.

        Returns:
            Self after fitting conditional probability estimators.
        """

        y = df[self.col].isna().astype(int)
        self.p_missing_ = float(y.mean())
        if self.p_missing_ == 0.0 or self.p_missing_ == 1.0:
            self.pipeline_ = None
            return self

        X = df.drop(columns=[self.col])
        num_selector = make_column_selector(dtype_exclude=["object", "category", "bool"])
        cat_selector = make_column_selector(dtype_include=["object", "category", "bool"])

        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_selector),
                ("cat", categorical_pipe, cat_selector),
            ],
            remainder="drop",
        )

        clf = LogisticRegression(max_iter=200, solver="lbfgs")

        self.pipeline_ = Pipeline([("pre", pre), ("clf", clf)])
        self.pipeline_.fit(X, y)
        return self

    def sample_mask(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
        """Sample a boolean mask indicating where the column should be missing.

        Args:
            df: Synthetic dataframe prior to applying missingness.
            rng: Random number generator used for reproducibility.

        Returns:
            Boolean series indexed like ``df`` with ``True`` for missing values.
        """

        if self.pipeline_ is None or self.p_missing_ == 0.0:
            if self.p_missing_ == 0.0:
                return pd.Series(False, index=df.index)
            probs = np.full(len(df), self.p_missing_, dtype=float)
        else:
            X = df.drop(columns=[self.col], errors="ignore")
            probs = self.pipeline_.predict_proba(X)[:, 1]
            mean_pred = probs.mean()
            if mean_pred > 0:
                scale = self.p_missing_ / mean_pred
                probs = np.clip(probs * scale, 0.0, 1.0)
            else:
                probs[:] = self.p_missing_

        u = rng.random(len(df))
        mask = u < probs
        return pd.Series(mask, index=df.index)


@dataclass
class DataFrameMissingnessModel:
    """Learn and apply missingness patterns across dataframe columns."""

    random_state: Optional[int] = None
    models_: Dict[str, ColumnMissingnessModel] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "DataFrameMissingnessModel":
        """Fit per-column missingness models on the provided dataframe.

        Args:
            df: Real dataframe used to learn missingness structure.

        Returns:
            Self with fitted column models.
        """

        self.models_ = {}
        for col in df.columns:
            model = ColumnMissingnessModel(col=col)
            model.fit(df)
            self.models_[col] = model
        return self

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned missingness patterns to a synthetic dataframe.

        Args:
            df: Synthetic dataframe before introducing missing values.

        Returns:
            Copy of ``df`` with missingness injected per fitted distributions.
        """

        rng = np.random.default_rng(self.random_state)
        out = df.copy()
        for col, model in self.models_.items():
            if col not in out.columns:
                continue
            mask = model.sample_mask(out, rng)
            out.loc[mask, col] = np.nan
        return out


class MissingnessWrappedGenerator:
    """Wrap a base generator to inject realistic missing values."""

    def __init__(
        self,
        base_generator: Callable[..., pd.DataFrame],
        missingness_model: DataFrameMissingnessModel,
    ) -> None:
        """Initialize the wrapper.

        Args:
            base_generator: Callable that returns a dataframe when invoked with ``n``.
            missingness_model: Learned missingness model to apply to generated data.
        """

        self.base_generator = base_generator
        self.missingness_model = missingness_model

    @classmethod
    def from_real_data(
        cls,
        base_generator: Callable[..., pd.DataFrame],
        real_df: pd.DataFrame,
        random_state: Optional[int] = None,
    ) -> "MissingnessWrappedGenerator":
        """Create a wrapper by fitting missingness to real data.

        Args:
            base_generator: Callable producing synthetic samples.
            real_df: Real dataframe used to estimate missingness.
            random_state: Optional RNG seed for reproducibility.

        Returns:
            Configured ``MissingnessWrappedGenerator`` instance.
        """

        miss_model = DataFrameMissingnessModel(random_state=random_state).fit(real_df)
        return cls(base_generator=base_generator, missingness_model=miss_model)

    def sample(self, n: int, **kwargs) -> pd.DataFrame:
        """Generate ``n`` samples with realistic missing values applied."""

        df_syn = self.base_generator(n, **kwargs)
        return self.missingness_model.apply(df_syn)


def fit_missingness_model(
    df: pd.DataFrame, *, random_state: Optional[int] = None
) -> Optional[DataFrameMissingnessModel]:
    """Fit a dataframe-level missingness model with logging safeguards.

    Args:
        df: Dataframe used to estimate missingness behaviour.
        random_state: Optional seed for reproducibility.

    Returns:
        Fitted :class:`DataFrameMissingnessModel` or ``None`` if fitting failed.
    """

    try:
        model = DataFrameMissingnessModel(random_state=random_state)
        return model.fit(df)
    except Exception:  # pragma: no cover - surfaced via pipeline logging
        logging.exception("Failed to fit missingness model", exc_info=True)
        return None


def apply_missingness_to_outputs(
    *,
    run_dir: Path,
    synth_df: pd.DataFrame,
    missingness_model: DataFrameMissingnessModel,
    real_df: pd.DataFrame,
    disc_cols: Iterable[str],
    cont_cols: Iterable[str],
    backend_name: str,
) -> Tuple[pd.DataFrame, bool]:
    """Apply missingness to backend artefacts and refresh derived metrics.

    Args:
        run_dir: Directory containing backend outputs.
        synth_df: Synthetic dataframe prior to missingness injection.
        missingness_model: Learned missingness model to apply.
        real_df: Real dataframe without missing values for metric refresh.
        disc_cols: Iterable of discrete column names.
        cont_cols: Iterable of continuous column names.
        backend_name: Name of the backend used for logging context.

    Returns:
        Tuple of the updated synthetic dataframe and a boolean indicating
        whether missingness was successfully applied.
    """

    pristine_path = run_dir / "synthetic.nomissing.csv"
    synth_path = run_dir / "synthetic.csv"

    try:
        if not pristine_path.exists():
            synth_df.to_csv(pristine_path, index=False)
        updated_df = missingness_model.apply(synth_df).convert_dtypes()
        updated_df.to_csv(synth_path, index=False)
    except Exception:
        logging.exception(
            "Failed to apply missingness model for backend %s", backend_name,
            exc_info=True,
        )
        return synth_df, False

    _refresh_metrics_after_missingness(
        run_dir=run_dir,
        backend_name=backend_name,
        real_df=real_df,
        synth_df=updated_df,
        disc_cols=disc_cols,
        cont_cols=cont_cols,
    )
    _update_missingness_manifest(
        run_dir=run_dir, missingness_model=missingness_model
    )
    return updated_df, True


def summarize_missingness_model(
    missingness_model: Optional[DataFrameMissingnessModel],
) -> Optional[Dict[str, Any]]:
    """Build a reporting-friendly summary of the missingness model.

    Args:
        missingness_model: Optional fitted missingness model from preprocessing.

    Returns:
        Mapping describing fitted column-level missingness rates or ``None`` if
        no model was provided.
    """

    if missingness_model is None:
        return None

    models = getattr(missingness_model, "models_", {}) or {}
    rows: List[Dict[str, float]] = []
    for column, column_model in sorted(models.items()):
        rate = float(getattr(column_model, "p_missing_", 0.0) or 0.0)
        if rate > 0.0:
            rows.append({"column": column, "missing_rate": rate})

    total_columns = len(models)
    nonzero_count = len(rows)
    zero_count = max(total_columns - nonzero_count, 0)

    return {
        "random_state": getattr(missingness_model, "random_state", None),
        "total_columns": total_columns,
        "nonzero_count": nonzero_count,
        "zero_count": zero_count,
        "rows": rows,
    }


def _refresh_metrics_after_missingness(
    *,
    run_dir: Path,
    backend_name: str,
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    disc_cols: Iterable[str],
    cont_cols: Iterable[str],
) -> None:
    """Recompute per-variable metrics after missingness injection."""

    try:
        dist_df = per_variable_distances(real_df, synth_df, disc_cols, cont_cols)
        dist_df.to_csv(run_dir / "per_variable_metrics.csv", index=False)
        metrics_path = run_dir / "metrics.json"
        metrics_payload: Dict[str, Any] = {}
        if metrics_path.exists():
            try:
                metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                metrics_payload = {}
        metrics_payload.setdefault("backend", backend_name)
        metrics_payload["summary"] = summarize_distance_metrics(dist_df)
        metrics_payload["missingness_wrapped"] = True
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover - surfaced via pipeline logging
        logging.exception(
            "Failed to refresh metrics after missingness injection for backend %s",
            backend_name,
            exc_info=True,
        )


def _update_missingness_manifest(
    *, run_dir: Path, missingness_model: DataFrameMissingnessModel
) -> None:
    """Record missingness configuration in the backend manifest."""

    manifest_path = run_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest.setdefault("missingness", {})
    manifest["missingness"].update(
        {
            "wrapped": True,
            "random_state": missingness_model.random_state,
            "source": "pipeline",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
