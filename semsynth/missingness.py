"""Missingness modeling utilities for backend generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


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
