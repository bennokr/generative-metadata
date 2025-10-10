import json
from pathlib import Path
import pandas as pd
import pytest

from semsynth.semmap import SemMapFrameAccessor, _HAVE_PINT


def data_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def test_apply_metadata_and_roundtrip(tmp_path):
    csv_path = data_dir() / "heart.csv"
    meta_path = data_dir() / "heart.metadata.json"

    # Load CSV
    df = pd.read_csv(csv_path)

    # Apply metadata from single JSON file
    df.semmap.apply_json_metadata(str(meta_path), convert_pint=True)

    # Persist with JSON-LD to parquet
    out = tmp_path / "heart.semmap.parquet"
    df.semmap.to_parquet(str(out), index=False)

    # Read back
    df2 = SemMapFrameAccessor.read_parquet(str(out), convert_pint=True)

    # Compare dataset JSON-LD
    assert json.dumps(df.attrs.get("semmap_jsonld"), sort_keys=True) == json.dumps(
        df2.attrs.get("semmap_jsonld"), sort_keys=True
    )

    # Compare per-column JSON-LD
    for col in df.columns:
        j1 = df[col].semmap.jsonld()
        j2 = df2[col].semmap.jsonld()
        assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True)

    # If pint is present, numeric columns should have pint dtype on round-trip
    if _HAVE_PINT:
        from pint_pandas import PintType

        assert isinstance(df2["trestbps"].dtype, PintType)
        assert isinstance(df2["chol"].dtype, PintType)


def test_export_metadata_roundtrip(tmp_path):
    csv_path = data_dir() / "heart.csv"
    meta_path = data_dir() / "heart.metadata.json"

    # Load CSV and apply from fixture
    df = pd.read_csv(csv_path)
    df.semmap.apply_json_metadata(str(meta_path), convert_pint=False)

    # Export
    exported = df.semmap.jsonld()
    with open(meta_path, "r", encoding="utf-8") as f:
        fixture = json.load(f)

    # Compare exported JSON with fixture (order-insensitive for dicts)
    assert json.dumps(exported, sort_keys=True) == json.dumps(fixture, sort_keys=True)

    # Apply exported to a fresh DataFrame and export again (idempotency)
    df2 = pd.read_csv(csv_path)
    df2.semmap.apply_json_metadata(exported, convert_pint=False)
    exported2 = df2.semmap.jsonld()
    assert json.dumps(exported2, sort_keys=True) == json.dumps(exported, sort_keys=True)
