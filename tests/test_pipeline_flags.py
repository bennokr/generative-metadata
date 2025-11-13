"""Tests covering pipeline flag behaviors and lazy imports."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pytest

from semsynth.models import ModelConfigBundle, ModelRun
from semsynth.pipeline import PipelineConfig, process_dataset
from semsynth.specs import DatasetSpec


class _DummyUtils:
    """Lightweight stand-in for :mod:`semsynth.utils`."""

    @staticmethod
    def ensure_dir(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def infer_types(df: pd.DataFrame) -> tuple[List[str], List[str]]:
        cont = [c for c in df.select_dtypes(include=["number"]).columns]
        disc = [c for c in df.columns if c not in cont]
        return disc, cont

    @staticmethod
    def coerce_discrete_to_category(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return df

    @staticmethod
    def rename_categorical_categories_to_str(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return df

    @staticmethod
    def coerce_continuous_to_float(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return df

    @staticmethod
    def seed_all(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)


class _DummyReporting:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def write_report_md(self, **kwargs) -> None:  # type: ignore[override]
        self.calls.append(kwargs)


@pytest.fixture()
def dummy_utils() -> _DummyUtils:
    return _DummyUtils()


@pytest.fixture()
def dummy_reporting() -> _DummyReporting:
    return _DummyReporting()


def test_process_dataset_skips_umap_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dummy_utils: _DummyUtils,
    dummy_reporting: _DummyReporting,
) -> None:
    """Disable UMAP and ensure the heavy module is never imported."""

    def fake_import_umap_utils() -> SimpleNamespace:
        raise AssertionError("UMAP utilities should not load when disabled")

    import importlib
    import semsynth.pipeline as pipeline_module

    pipeline_module = importlib.reload(pipeline_module)

    monkeypatch.setattr(pipeline_module, "_import_umap_utils", fake_import_umap_utils)
    monkeypatch.setattr(pipeline_module, "_import_utils", lambda: dummy_utils)
    monkeypatch.setattr(pipeline_module, "_import_reporting", lambda: dummy_reporting)
    monkeypatch.setattr(pipeline_module, "resolve_mapping_json", lambda *_: None)
    monkeypatch.setattr(pipeline_module, "get_uciml_variable_descriptions", lambda *_: {})
    monkeypatch.setattr(pipeline_module, "discover_model_runs", lambda *_: [])
    monkeypatch.setattr("semsynth.utils.coerce_discrete_to_category", lambda df, cols: df)
    monkeypatch.setattr("semsynth.utils.rename_categorical_categories_to_str", lambda df, cols: df)
    monkeypatch.setattr("semsynth.utils.coerce_continuous_to_float", lambda df, cols: df)

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    spec = DatasetSpec(provider="openml", name="demo", id=1, target=None)
    cfg = PipelineConfig(generate_umap=False)
    bundle = ModelConfigBundle(specs=[])

    pipeline_module.process_dataset(
        spec, df, None, str(tmp_path), model_bundle=bundle, pipeline_config=cfg
    )

    assert dummy_reporting.calls, "reporting should still run"


def test_existing_umap_is_respected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dummy_utils: _DummyUtils,
    dummy_reporting: _DummyReporting,
) -> None:
    """Ensure synthetic UMAPs are not regenerated when files already exist."""

    calls: list[tuple[str, str]] = []

    class _Artifacts:
        def __init__(self) -> None:
            self.embedding = np.zeros((2, 2))
            self.color_labels = None

    def fake_build_umap(*_args, **_kwargs) -> _Artifacts:
        return _Artifacts()

    def fake_plot_umap(_embedding, outfile, title, **_kwargs):
        calls.append((outfile, title))
        return ((0.0, 1.0), (0.0, 1.0))

    def fake_transform_with_umap(_art, df: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(df), 2))

    import importlib
    import semsynth.pipeline as pipeline_module

    pipeline_module = importlib.reload(pipeline_module)

    monkeypatch.setattr(
        pipeline_module,
        "_import_umap_utils",
        lambda: SimpleNamespace(
            build_umap=fake_build_umap,
            plot_umap=fake_plot_umap,
            transform_with_umap=fake_transform_with_umap,
        ),
    )
    monkeypatch.setattr(pipeline_module, "_import_utils", lambda: dummy_utils)
    monkeypatch.setattr(pipeline_module, "_import_reporting", lambda: dummy_reporting)
    monkeypatch.setattr(pipeline_module, "resolve_mapping_json", lambda *_: None)
    monkeypatch.setattr(pipeline_module, "get_uciml_variable_descriptions", lambda *_: {})
    monkeypatch.setattr("semsynth.utils.coerce_discrete_to_category", lambda df, cols: df)
    monkeypatch.setattr("semsynth.utils.rename_categorical_categories_to_str", lambda df, cols: df)
    monkeypatch.setattr("semsynth.utils.coerce_continuous_to_float", lambda df, cols: df)

    run_dir = tmp_path / "demo" / "models" / "model_a"
    run_dir.mkdir(parents=True, exist_ok=True)
    synth_csv = run_dir / "synthetic.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(synth_csv, index=False)
    existing_png = run_dir / "umap.png"
    existing_png.write_bytes(b"")

    run = ModelRun(
        name="model_a",
        backend="dummy",
        run_dir=run_dir,
        synthetic_csv=synth_csv,
        per_variable_csv=None,
        metrics_json=None,
        metrics={},
        umap_png=existing_png,
        manifest={},
        privacy_json=None,
        privacy_metrics={},
        downstream_json=None,
        downstream_metrics={},
    )

    monkeypatch.setattr("semsynth.pipeline.discover_model_runs", lambda *_: [run])

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "target": [0, 1, 0]})
    spec = DatasetSpec(provider="openml", name="demo", id=1, target="target")
    bundle = ModelConfigBundle(specs=[])

    pipeline_module.process_dataset(spec, df, None, str(tmp_path), model_bundle=bundle)

    assert len(calls) == 1, "only the real-data UMAP should be plotted"
