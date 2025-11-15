"""Tests covering optional missingness wrapping in the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from semsynth.models import ModelConfigBundle, ModelSpec
from semsynth.pipeline import (
    BackendExecutor,
    DatasetPreprocessor,
    MetricWriter,
    PipelineConfig,
    ReportWriter,
)
from semsynth.specs import DatasetSpec


class _DummyUtils:
    """Minimal stand-in for :mod:`semsynth.utils` utilities."""

    @staticmethod
    def ensure_dir(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def infer_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
        cont = [c for c in df.select_dtypes(include=["number"]).columns]
        disc = [c for c in df.columns if c not in cont]
        return disc, cont

    @staticmethod
    def coerce_discrete_to_category(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df

    @staticmethod
    def rename_categorical_categories_to_str(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df

    @staticmethod
    def coerce_continuous_to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df

    @staticmethod
    def seed_all(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)


class _NoopMetricWriter(MetricWriter):
    """Metric writer that records calls without touching disk."""

    def __init__(self) -> None:
        super().__init__(privacy_summarizer=None, downstream_compare=None)
        self.privacy_calls: list[dict] = []
        self.downstream_calls: list[dict] = []

    def write_privacy(self, *args, **kwargs):  # type: ignore[override]
        self.privacy_calls.append({"args": args, "kwargs": kwargs})
        return {}

    def write_downstream(self, *args, **kwargs):  # type: ignore[override]
        self.downstream_calls.append({"args": args, "kwargs": kwargs})
        return {}


@pytest.fixture()
def dummy_utils() -> _DummyUtils:
    """Provide dummy utility helpers for preprocessing."""

    return _DummyUtils()


def _make_preprocessor(dummy_utils: _DummyUtils) -> DatasetPreprocessor:
    """Build a :class:`DatasetPreprocessor` bound to dummy helpers."""

    return DatasetPreprocessor(
        utils_module=dummy_utils,
        load_mapping=lambda *_: None,
        resolve_mapping=lambda *_: None,
    )


def test_preprocess_builds_missingness_model(
    tmp_path: Path, dummy_utils: _DummyUtils
) -> None:
    """Verify preprocessing optionally fits the missingness model."""

    df = pd.DataFrame(
        {
            "num": [0.0, 1.0, np.nan, 2.0, np.nan, 1.5],
            "cat": ["a", "b", "a", "b", "a", "b"],
        }
    )
    spec = DatasetSpec(provider="demo", name="demo")
    cfg = PipelineConfig(
        enable_missingness_wrapping=True,
        missingness_random_state=123,
        generate_umap=False,
    )
    rng = dummy_utils.seed_all(42)

    preprocessor = _make_preprocessor(dummy_utils)
    result = preprocessor.preprocess(
        spec,
        df,
        color_series=None,
        outdir=tmp_path,
        cfg=cfg,
        rng=rng,
        generate_umap=False,
        umap_utils=None,
    )

    assert result.missingness_model is not None
    assert result.missingness_model.random_state == 123
    assert set(result.missingness_model.models_) == {"num", "cat"}


def test_backend_executor_applies_missingness(
    tmp_path: Path, dummy_utils: _DummyUtils
) -> None:
    """Ensure backend outputs are wrapped with learned missingness."""

    df = pd.DataFrame(
        {
            "num": [0.0, np.nan, 1.0, np.nan, 0.5, 1.5, np.nan, 2.5],
            "cat": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )
    spec = DatasetSpec(provider="demo", name="demo", target=None)
    cfg = PipelineConfig(
        enable_missingness_wrapping=True,
        missingness_random_state=5,
        compute_privacy=False,
        compute_downstream=False,
        generate_umap=False,
    )
    rng = dummy_utils.seed_all(7)

    preprocessor = _make_preprocessor(dummy_utils)
    preprocessed = preprocessor.preprocess(
        spec,
        df,
        color_series=None,
        outdir=tmp_path,
        cfg=cfg,
        rng=rng,
        generate_umap=False,
        umap_utils=None,
    )

    class _StubBackend:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def run_experiment(
            self,
            *,
            df: pd.DataFrame,
            provider,
            dataset_name,
            provider_id,
            outdir: str,
            label: str,
            model_info,
            rows: int,
            seed: int,
            test_size: float,
            semmap_export,
        ) -> Path:
            run_dir = Path(outdir) / "models" / label
            run_dir.mkdir(parents=True, exist_ok=True)
            synth = pd.DataFrame(
                {
                    "num": np.linspace(0.0, 3.0, num=20, endpoint=False),
                    "cat": ["a", "b"] * 10,
                }
            )
            synth.to_csv(run_dir / "synthetic.csv", index=False)
            (run_dir / "manifest.json").write_text(
                json.dumps({"backend": "stub", "name": label}),
                encoding="utf-8",
            )
            (run_dir / "metrics.json").write_text(
                json.dumps({"backend": "stub"}),
                encoding="utf-8",
            )
            self.calls.append({"label": label, "rows": rows, "seed": seed})
            return run_dir

    backend = _StubBackend()
    bundle = ModelConfigBundle(
        specs=[ModelSpec(name="stub", backend="stub", model={}, rows=None, seed=None)]
    )
    metric_writer = _NoopMetricWriter()
    executor = BackendExecutor(
        cfg,
        load_backend=lambda name: backend,
        metric_writer=metric_writer,
    )

    outdir = tmp_path / "dataset"
    outdir.mkdir(parents=True, exist_ok=True)
    executor.run_models(spec, bundle, preprocessed, outdir)

    run_dir = outdir / "models" / "stub"
    wrapped_df = pd.read_csv(run_dir / "synthetic.csv")
    pristine_df = pd.read_csv(run_dir / "synthetic.nomissing.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert wrapped_df.isna().any().any(), "wrapped data should contain missing values"
    assert not pristine_df.isna().any().any(), "original synthetic data should be preserved"
    assert manifest.get("missingness", {}).get("wrapped") is True
    assert manifest.get("missingness", {}).get("source") == "pipeline"
    assert metrics.get("missingness_wrapped") is True
    assert (run_dir / "per_variable_metrics.csv").exists()

    assert not metric_writer.privacy_calls
    assert not metric_writer.downstream_calls


def test_report_writer_includes_missingness_summary(
    tmp_path: Path, dummy_utils: _DummyUtils
) -> None:
    """Report writer should forward missingness metadata to reporting."""

    df = pd.DataFrame(
        {
            "num": [0.0, np.nan, 1.0, 2.0],
            "cat": ["a", "b", "a", "b"],
        }
    )
    spec = DatasetSpec(provider="demo", name="demo")
    cfg = PipelineConfig(
        enable_missingness_wrapping=True,
        missingness_random_state=99,
        generate_umap=False,
    )
    rng = dummy_utils.seed_all(11)

    preprocessor = _make_preprocessor(dummy_utils)
    preprocessed = preprocessor.preprocess(
        spec,
        df,
        color_series=None,
        outdir=tmp_path,
        cfg=cfg,
        rng=rng,
        generate_umap=False,
        umap_utils=None,
    )

    class ReportingStub:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def write_report_md(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

    reporting_stub = ReportingStub()
    reporter = ReportWriter(reporting_stub, umap_utils=None)

    reporter.write_report(
        outdir=tmp_path,
        dataset_spec=spec,
        preprocessed=preprocessed,
        model_runs=[],
        inferred_types=preprocessed.inferred_types,
        variable_descriptions=None,
    )

    assert reporting_stub.calls, "reporting stub should receive a call"
    summary = reporting_stub.calls[0].get("missingness_summary")
    assert summary is not None
    assert summary["random_state"] == 99
    assert summary["total_columns"] == 2
    assert summary["nonzero_count"] >= 1
    assert summary["rows"], "missingness summary should include per-column rates"
