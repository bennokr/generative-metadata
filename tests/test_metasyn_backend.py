"""Unit tests for the MetaSyn backend contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from semsynth.backends import metasyn as metasyn_backend


class _FakeMetaFrame:
    """Minimal stand-in for :class:`metasyn.metaframe.MetaFrame`."""

    def __init__(self, train_df: pd.DataFrame) -> None:
        self._train_df = train_df

    @classmethod
    def fit_dataframe(cls, train_df: pd.DataFrame) -> "_FakeMetaFrame":
        return cls(train_df)

    def save(self, path: str) -> None:
        Path(path).write_text("{}", encoding="utf-8")

    def synthesize(self, n: int) -> pd.DataFrame:
        return self._train_df.head(n).copy()


def _fake_distance(_test: pd.DataFrame, synth: pd.DataFrame, *_args, **_kwargs) -> pd.DataFrame:
    return pd.DataFrame({"variable": synth.columns, "distance": 0.0})


def _fake_summary(_dist: pd.DataFrame) -> Dict[str, float]:
    return {"disc_jsd_mean": 0.0}


def test_run_experiment_writes_expected_artifacts(tmp_path: Path, monkeypatch) -> None:
    """Smoke test ensuring the MetaSyn backend conforms to the protocol."""

    monkeypatch.setattr(metasyn_backend, "_load_metasyn", lambda: _FakeMetaFrame)
    monkeypatch.setattr(metasyn_backend, "per_variable_distances", _fake_distance)
    monkeypatch.setattr(metasyn_backend, "summarize_distance_metrics", _fake_summary)
    monkeypatch.setattr(
        metasyn_backend,
        "model_run_root",
        lambda outdir: Path(outdir) / "models",
    )
    monkeypatch.setattr(metasyn_backend, "ensure_dir", lambda path: Path(path).mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(metasyn_backend, "coerce_continuous_to_float", lambda df, _cols: df)
    monkeypatch.setattr(metasyn_backend, "infer_types", lambda df: ([], list(df.columns)))

    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})

    run_dir = metasyn_backend.run_experiment(
        df=df,
        provider="openml",
        dataset_name="demo",
        provider_id=1,
        outdir=str(tmp_path),
        label="model_a",
        model_info={"type": "fake"},
        rows=2,
        seed=7,
        test_size=0.5,
        semmap_export=None,
    )

    synth_csv = Path(run_dir) / "synthetic.csv"
    metrics_json = Path(run_dir) / "metrics.json"
    manifest_json = Path(run_dir) / "manifest.json"

    assert synth_csv.exists(), "synthetic output should be written"
    assert metrics_json.exists(), "metrics summary should be written"
    assert manifest_json.exists(), "manifest should be written"

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest.get("backend") == "metasyn"
