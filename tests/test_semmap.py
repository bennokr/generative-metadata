import json
from pathlib import Path
import pandas as pd
from pint_pandas import PintType

from semsynth.semmap import SemMapFrameAccessor


def data_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def test_apply_metadata_and_roundtrip(tmp_path):
    csv_path = data_dir() / "heart.csv"
    meta_path = data_dir() / "heart.metadata.json"

    # Load CSV
    df = pd.read_csv(csv_path).convert_dtypes()

    # Apply metadata from single JSON file
    df.semmap.from_jsonld(str(meta_path), convert_pint=True)

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
        j1 = df[col].semmap.to_jsonld()
        j2 = df2[col].semmap.to_jsonld()
        assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True)

    #Numeric columns should have pint dtype on round-trip

    assert isinstance(df2["trestbps"].dtype, PintType)
    assert isinstance(df2["chol"].dtype, PintType)


def test_export_metadata_roundtrip(tmp_path):
    csv_path = data_dir() / "heart.csv"
    meta_path = data_dir() / "heart.metadata.json"

    # Load CSV and apply from fixture
    df = pd.read_csv(csv_path).convert_dtypes()
    df.semmap.from_jsonld(str(meta_path), convert_pint=False)

    # Export
    exported = df.semmap.to_jsonld()
    with open(meta_path, "r", encoding="utf-8") as f:
        fixture = json.load(f)

    # Compare exported JSON with fixture (order-insensitive for dicts)
    diff = subset_diff(fixture, exported)
    if diff:
        AssertionError(diff)

    # Apply exported to a fresh DataFrame and export again (idempotency)
    df2 = pd.read_csv(csv_path).convert_dtypes()
    df2.semmap.from_jsonld(exported, convert_pint=False)
    exported2 = df2.semmap.to_jsonld()
    diff = subset_diff(exported, exported2)
    if diff:
        AssertionError(diff)


def subset_diff(expected, actual, path=""):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path}: expected dict, got {type(actual).__name__}"
        for k, v in expected.items():
            if k not in actual:
                return f"{path}.{k}: missing key in actual".lstrip(".")
            diff = subset_diff(v, actual[k], f"{path}.{k}" if path else k)
            if diff:
                return diff
        return None

    elif isinstance(expected, list):
        if not isinstance(actual, list):
            return f"{path}: expected list, got {type(actual).__name__}"
        if len(expected) != len(actual):
            return f"{path}: expected list length {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            diff = subset_diff(e, a, f"{path}[{i}]")
            if diff:
                return diff
        return None

    else:
        if expected != actual:
            return f"{path}: expected {expected!r}, got {actual!r}"
        return None