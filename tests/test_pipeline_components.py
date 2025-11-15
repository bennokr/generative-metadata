"""Unit tests for the pipeline helper classes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from semsynth.dataproviders._helpers import clean_dataset_frame
from semsynth.dataproviders.openml import load_openml_by_name
from semsynth.dataproviders.uciml import load_uciml_by_id
from semsynth.models import ModelConfigBundle, ModelSpec
from semsynth.pipeline import (
    BackendExecutor,
    DatasetPreprocessor,
    MetricWriter,
    PipelineConfig,
    PreprocessingResult,
    ReportWriter,
    UmapArtifacts,
    _build_downstream_meta,
)
from semsynth.specs import DatasetSpec


class _StubUtils:
    """Small helper mimicking the subset of utils used by the preprocessor."""

    def __init__(self) -> None:
        self._dirs: List[str] = []

    def ensure_dir(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self._dirs.append(path)

    def infer_types(self, df: pd.DataFrame) -> Any:
        discrete: List[str] = []
        continuous: List[str] = []
        for column in df.columns:
            series = df[column]
            if pd.api.types.is_numeric_dtype(series):
                continuous.append(column)
            else:
                discrete.append(column)
        return discrete, continuous

    def coerce_discrete_to_category(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for column in cols:
            df[column] = pd.Categorical(df[column])
        return df

    def rename_categorical_categories_to_str(
        self, df: pd.DataFrame, cols: List[str]
    ) -> pd.DataFrame:
        return df

    def coerce_continuous_to_float(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for column in cols:
            df[column] = df[column].astype(float)
        return df

    def seed_all(self, seed: int) -> Any:
        import numpy as np

        return np.random.default_rng(seed)


class _StubUmapModule:
    """Simple module stub recording calls to UMAP helpers."""

    def __init__(self) -> None:
        self.build_calls: List[Dict[str, Any]] = []
        self.transform_calls: List[Dict[str, Any]] = []
        self.plot_calls: List[Dict[str, Any]] = []

    def build_umap(self, *args: Any, **kwargs: Any) -> Any:
        self.build_calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(embedding=[[0.0, 0.0]], color_labels=None)

    def plot_umap(self, embedding: Any, path: str, **kwargs: Any) -> Dict[str, float]:
        self.plot_calls.append({"embedding": embedding, "path": path, "kwargs": kwargs})
        Path(path).write_text("png")
        return {"x": 1.0, "y": 2.0}

    def transform_with_umap(self, transformer: Any, df: pd.DataFrame) -> pd.DataFrame:
        self.transform_calls.append({"transformer": transformer, "df": df})
        return pd.DataFrame([[0.0, 0.0]], columns=["x", "y"])


def _create_preprocessing_result(df: pd.DataFrame, tmp_path) -> PreprocessingResult:
    target_series = df["value"].copy()
    target_series.attrs["prov:hadRole"] = "target"
    return PreprocessingResult(
        df_processed=df,
        df_no_na=df,
        df_fit_sample=df,
        disc_cols=["category"],
        cont_cols=["value"],
        inferred_types={"category": "discrete", "value": "continuous"},
        semmap_export={"dummy": True},
        color_series=target_series,
        umap_png_real=None,
        umap_artifacts=UmapArtifacts(transformer=None, real_png=tmp_path / "umap.png", limits=None),
    )


def test_dataset_preprocessor_basic(tmp_path):
    df = pd.DataFrame({"category": ["a", "b"], "value": [1.0, 2.0]})
    spec = DatasetSpec(provider="demo", name="example", target="value")
    utils_stub = _StubUtils()
    preprocessor = DatasetPreprocessor(
        utils_module=utils_stub,
        load_mapping=lambda path: {},
        resolve_mapping=lambda _: None,
    )
    cfg = PipelineConfig(generate_umap=False)
    rng = utils_stub.seed_all(cfg.random_state)

    result = preprocessor.preprocess(
        spec,
        df,
        None,
        tmp_path,
        cfg,
        rng,
        generate_umap=False,
        umap_utils=None,
    )

    assert (tmp_path).exists()
    assert list(result.disc_cols) == ["category"]
    assert list(result.cont_cols) == ["value"]
    assert result.umap_artifacts is None
    assert result.umap_png_real is None
    assert result.df_processed["category"].dtype.name == "category"


def test_dataset_preprocessor_with_umap(tmp_path):
    df = pd.DataFrame({"category": ["a", "b"], "value": [1.0, 2.0]})
    spec = DatasetSpec(provider="demo", name="example", target="value")
    utils_stub = _StubUtils()
    preprocessor = DatasetPreprocessor(
        utils_module=utils_stub,
        load_mapping=lambda path: {},
        resolve_mapping=lambda _: None,
    )
    cfg = PipelineConfig(generate_umap=True)
    rng = utils_stub.seed_all(cfg.random_state)
    umap_stub = _StubUmapModule()

    result = preprocessor.preprocess(
        spec,
        df,
        None,
        tmp_path,
        cfg,
        rng,
        generate_umap=True,
        umap_utils=umap_stub,
    )

    assert result.umap_artifacts is not None
    assert result.umap_png_real == tmp_path / "umap_real.png"
    assert umap_stub.build_calls, "UMAP build should be invoked"
    assert umap_stub.plot_calls, "UMAP plot should be invoked"


def test_metric_writer_writes_metrics(tmp_path):
    real_df = pd.DataFrame({"value": [1.0, 2.0]})
    synth_df = pd.DataFrame({"value": [1.1, 1.9]})

    @dataclass
    class Summary:
        score: float

    def summarizer(_: pd.DataFrame, __: pd.DataFrame, ___: pd.DataFrame) -> Summary:
        return Summary(score=0.5)

    def comparer(_: pd.DataFrame, __: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        schema = metadata.get("dsv:datasetSchema", {})
        columns = schema.get("dsv:column", [])
        assert columns and columns[0]["schema:name"] == "value"
        stats = columns[0].get("dsv:summaryStatistics", {})
        assert stats.get("dsv:statisticalDataType") == "dsv:QuantitativeDataType"
        return {
            "formula": {"metric": 1.0},
            "compare": pd.DataFrame({"sign_match": [1, 1]}),
        }

    run_dir = tmp_path
    synth_df.to_csv(run_dir / "synthetic.csv", index=False)

    writer = MetricWriter(privacy_summarizer=summarizer, downstream_compare=comparer)

    privacy_payload = writer.write_privacy(
        run_dir, real_df, {"value": "continuous"}, synth_df=synth_df
    )
    downstream_payload = writer.write_downstream(
        run_dir, real_df, synth_df, {"value": "continuous"}, target_series=None
    )

    assert privacy_payload == {"score": 0.5}
    assert downstream_payload["sign_match_rate"] == 1.0
    assert (run_dir / "metrics.privacy.json").exists()
    assert (run_dir / "metrics.downstream.json").exists()


def test_build_downstream_meta_matches_dsv_schema():
    df = pd.DataFrame({"feature": ["a", "b", "a"], "target": [0, 1, 1]})
    inferred = {"feature": "discrete", "target": "discrete"}
    target_series = df["target"].copy()
    target_series.attrs["prov:hadRole"] = "target"

    meta = _build_downstream_meta(df, inferred, target_series)

    schema = meta["dsv:datasetSchema"]
    columns = schema["dsv:column"]
    by_name = {col["schema:name"]: col for col in columns}

    assert by_name["target"]["prov:hadRole"] == "target"
    stats = by_name["target"]["dsv:summaryStatistics"]
    assert stats["dsv:statisticalDataType"] == "dsv:NominalDataType"
    assert by_name["target"]["schema:defaultValue"] == "0"

    codebook = (
        by_name["target"]["dsv:columnProperty"]["dsv:hasCodeBook"]["skos:hasTopConcept"]
    )
    assert {concept["skos:notation"] for concept in codebook} == {"0", "1"}
    assert by_name["feature"]["prov:hadRole"] == "predictor"


def test_clean_dataset_frame_detects_target_hint():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "feature": [0.1, 0.2],
            "label": ["a", "b"],
        }
    )
    cleaned, target_name, color_series = clean_dataset_frame(df, target="label")

    assert list(cleaned.columns) == ["feature", "label"]
    assert target_name == "label"
    assert color_series.equals(cleaned["label"])
    assert color_series.attrs.get("prov:hadRole") == "target"


def test_clean_dataset_frame_detects_semmap_role():
    df = pd.DataFrame(
        {
            "feature": [0.1, 0.2],
            "class": ["a", "b"],
        }
    )
    metadata = {
        "dsv:datasetSchema": {
            "dsv:column": [
                {"schema:name": "class", "prov:hadRole": "target"},
                {"schema:name": "feature"},
            ]
        }
    }

    cleaned, target_name, color_series = clean_dataset_frame(
        df, metadata=metadata
    )

    assert target_name == "class"
    assert color_series.attrs.get("prov:hadRole") == "target"


def test_openml_cached_uses_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cache_dir = tmp_path / "openml"
    by_name_dir = cache_dir / "by_name"
    by_name_dir.mkdir(parents=True)
    dataset_name = "demo"
    dataset_id = 123
    meta = {"name": dataset_name, "id": dataset_id, "target": None}
    (by_name_dir / f"{dataset_name}.json").write_text(json.dumps(meta))
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "value": [0.1, 0.2],
            "class": ["a", "b"],
        }
    )
    df.to_csv(by_name_dir / f"{dataset_id}.csv.gz", index=False, compression="infer")

    called: Dict[str, Any] = {}

    def fake_helper(
        data: pd.DataFrame,
        *,
        target: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        cleaned = data.drop(columns=["id"], errors="ignore")
        series = cleaned["class"].copy()
        called.update(
            {
                "called": True,
                "target": target,
                "metadata": metadata,
                "series_id": id(series),
            }
        )
        return cleaned, "class", series

    monkeypatch.setattr(
        "semsynth.dataproviders.openml.clean_dataset_frame", fake_helper
    )

    spec, df_loaded, color_series = load_openml_by_name(dataset_name, cache_dir)

    assert called["called"] is True
    assert called["target"] is None
    assert isinstance(called["metadata"], dict)
    assert "id" not in df_loaded.columns
    assert id(color_series) == called["series_id"]
    assert spec.target == "class"


def test_uciml_cached_uses_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cache_dir = tmp_path
    dataset_id = 456
    meta_path = cache_dir / f"{dataset_id}.meta.json"
    meta_path.write_text(json.dumps({"name": "demo", "target": None}))
    df = pd.DataFrame(
        {
            "index": [0, 1],
            "feature": [1, 2],
            "target": ["x", "y"],
        }
    )
    df.to_csv(cache_dir / f"{dataset_id}.csv.gz", index=False, compression="infer")

    called: Dict[str, Any] = {}

    def fake_helper(
        data: pd.DataFrame,
        *,
        target: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        cleaned = data.drop(columns=["index"], errors="ignore")
        series = cleaned["target"].copy()
        called.update(
            {
                "called": True,
                "target": target,
                "metadata": metadata,
                "series_id": id(series),
            }
        )
        return cleaned, "target", series

    monkeypatch.setattr(
        "semsynth.dataproviders.uciml.clean_dataset_frame", fake_helper
    )

    spec, df_loaded, color_series = load_uciml_by_id(dataset_id, cache_dir)

    assert called["called"] is True
    assert called["target"] is None
    assert isinstance(called["metadata"], dict)
    assert "index" not in df_loaded.columns
    assert id(color_series) == called["series_id"]
    assert spec.target == "target"


def test_backend_executor_runs_models(tmp_path):
    df = pd.DataFrame({"category": pd.Categorical(["a", "b"]), "value": [1.0, 2.0]})
    spec = DatasetSpec(provider="demo", name="example", target="value")
    preprocessed = _create_preprocessing_result(df, tmp_path)

    calls: Dict[str, Any] = {}

    class Backend:
        def run_experiment(self, **kwargs: Any) -> str:
            calls.update(kwargs)
            run_dir = tmp_path / "models" / "test"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "synthetic.csv").write_text("value\n1.0\n")
            return str(run_dir)

    def loader(name: str) -> Backend:
        assert name == "dummy"
        return Backend()

    class DummyMetricWriter:
        def __init__(self) -> None:
            self.privacy_calls: List[Dict[str, Any]] = []
            self.downstream_calls: List[Dict[str, Any]] = []

        def write_privacy(
            self,
            run_dir: Any,
            real_df: pd.DataFrame,
            inferred: Dict[str, str],
            synth_df: pd.DataFrame,
        ) -> Dict[str, Any]:
            self.privacy_calls.append(
                {"run_dir": run_dir, "real_df": real_df, "inferred": inferred, "synth_df": synth_df}
            )
            return {"ok": True}

        def write_downstream(
            self,
            run_dir: Any,
            real_df: pd.DataFrame,
            synth_df: pd.DataFrame,
            inferred: Dict[str, str],
            target_series: Optional[pd.Series],
        ) -> Dict[str, Any]:
            self.downstream_calls.append(
                {
                    "run_dir": run_dir,
                    "real_df": real_df,
                    "synth_df": synth_df,
                    "inferred": inferred,
                    "target": target_series,
                }
            )
            return {"ok": True}

    metric_writer = DummyMetricWriter()

    bundle = ModelConfigBundle(
        specs=[ModelSpec(name="demo", backend="dummy", compute_privacy=True, compute_downstream=True)],
        generate_umap=None,
        compute_privacy=None,
        compute_downstream=None,
    )

    executor = BackendExecutor(
        PipelineConfig(compute_privacy=True, compute_downstream=True),
        load_backend=loader,
        metric_writer=metric_writer,
    )

    executor.run_models(spec, bundle, preprocessed, tmp_path)

    assert calls["semmap_export"] == {"dummy": True}
    assert metric_writer.privacy_calls
    assert metric_writer.downstream_calls


def test_report_writer_generates_outputs(tmp_path):
    df = pd.DataFrame({"category": ["a", "b"], "value": [1.0, 2.0]})
    preprocessed = _create_preprocessing_result(df, tmp_path)
    preprocessed = PreprocessingResult(
        df_processed=preprocessed.df_processed,
        df_no_na=preprocessed.df_no_na,
        df_fit_sample=preprocessed.df_fit_sample,
        disc_cols=preprocessed.disc_cols,
        cont_cols=preprocessed.cont_cols,
        inferred_types=preprocessed.inferred_types,
        semmap_export=preprocessed.semmap_export,
        color_series=None,
        umap_png_real=tmp_path / "umap_real.png",
        umap_artifacts=UmapArtifacts(
            transformer=SimpleNamespace(),
            real_png=tmp_path / "umap_real.png",
            limits={"x": 1.0},
        ),
    )

    umap_stub = _StubUmapModule()
    run_dir = tmp_path / "models" / "demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    synth_path = run_dir / "synthetic.csv"
    synth_path.write_text("category,value\na,1.0\n")

    run = SimpleNamespace(
        name="demo",
        synthetic_csv=synth_path,
        run_dir=run_dir,
        umap_png=None,
    )

    class ReportingModule:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def write_report_md(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

    reporting_stub = ReportingModule()

    reporter = ReportWriter(reporting_stub, umap_stub)

    cfg = PipelineConfig()
    reporter.generate_synthetic_umaps([run], DatasetSpec(provider="demo", name="example"), preprocessed, cfg)
    reporter.write_report(
        outdir=tmp_path,
        dataset_spec=DatasetSpec(provider="demo", name="example"),
        preprocessed=preprocessed,
        model_runs=[run],
        inferred_types=preprocessed.inferred_types,
        variable_descriptions={"category": "demo"},
    )

    assert umap_stub.transform_calls, "UMAP transform should be invoked"
    assert umap_stub.plot_calls, "UMAP plot should be invoked"
    assert run.umap_png is not None and run.umap_png.exists()
    assert reporting_stub.calls, "Report should be written"
