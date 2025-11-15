"""Unit tests for the missingness modeling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from semsynth.backends.missingness import DataFrameMissingnessModel


def test_missingness_model_matches_marginal_rate() -> None:
    """Calibrated probabilities should reproduce the marginal missing rate."""

    rng = np.random.default_rng(0)
    n = 500
    real = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    mask = rng.random(n) < 0.35
    real.loc[mask, "y"] = np.nan

    model = DataFrameMissingnessModel(random_state=123).fit(real)

    synth = pd.DataFrame({"x": rng.normal(size=2000), "y": rng.normal(size=2000)})
    applied = model.apply(synth)

    expected = real["y"].isna().mean()
    observed = applied["y"].isna().mean()
    assert abs(observed - expected) < 0.05


def test_missingness_model_respects_conditional_structure() -> None:
    """Conditional missingness should follow learned group structure."""

    rng = np.random.default_rng(42)
    n = 800
    groups = np.where(rng.random(n) < 0.5, "high", "low")
    real = pd.DataFrame({"group": groups, "value": rng.normal(size=n)})
    mask = (real["group"] == "high") & (rng.random(n) < 0.75)
    mask |= (real["group"] == "low") & (rng.random(n) < 0.1)
    real.loc[mask, "value"] = np.nan

    model = DataFrameMissingnessModel(random_state=7).fit(real)

    synth_groups = np.where(rng.random(n) < 0.5, "high", "low")
    synth = pd.DataFrame({"group": synth_groups, "value": rng.normal(size=n)})
    applied = model.apply(synth)

    high_rate = applied.loc[applied["group"] == "high", "value"].isna().mean()
    low_rate = applied.loc[applied["group"] == "low", "value"].isna().mean()
    assert high_rate - low_rate > 0.4


def test_missingness_model_ignores_unknown_columns() -> None:
    """Columns missing from the synthetic frame should be ignored gracefully."""

    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "z"]})
    model = DataFrameMissingnessModel(random_state=1).fit(df)
    applied = model.apply(pd.DataFrame({"b": ["x", "y", "z"]}))
    assert applied.shape[1] == 1
    assert applied.columns.tolist() == ["b"]
