from __future__ import annotations

import json

import pandas as pd
import pytest

from bncli.mappings import canonical_generator_name
from bncli.synth import discover_synth_runs, run_synth_experiment


@pytest.mark.parametrize(
    "alias, expected",
    [
        ("CTGAN", "ctgan"),
        ("ads-gan", "adsgan"),
        ("arfpy", "arf"),
    ],
)
def test_canonical_generator_name(alias: str, expected: str) -> None:
    assert canonical_generator_name(alias) == expected


def test_run_synth_experiment_creates_artifacts(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "flag": [0, 1, 1, 0] * 8,
            "value": [0.1, 0.5, 1.2, 3.4] * 8,
            "category": ["a", "b", "c", "d"] * 8,
        }
    )
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    try:
        run_dir = run_synth_experiment(
            df=df,
            provider="demo",
            dataset_name="toy",
            provider_id=None,
            outdir=str(out_dir),
            generator="ctgan",
            gen_params_json=json.dumps({"epochs": 1, "batch_size": 64}),
            rows=16,
            seed=42,
            test_size=0.25,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"synthcity dependency missing: {exc}")

    synth_csv = run_dir / "synthetic.csv"
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    assert synth_csv.exists()
    assert manifest_path.exists()
    assert metrics_path.exists()

    synth_df = pd.read_csv(synth_csv)
    assert len(synth_df) == 16

    runs = discover_synth_runs(out_dir, provider="demo", dataset_name="toy")
    assert runs and runs[0].synthetic_csv == synth_csv
