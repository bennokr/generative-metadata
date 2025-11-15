"""Tests for map_columns.shared helper utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from map_columns.shared import ColumnInfo, DatasetMetadata, load_columns


@pytest.fixture()
def dataset_json(tmp_path: Path) -> Path:
    """Create a dataset JSON file with multiple column shapes."""
    data = {
        "dcterms:title": "Test dataset",
        "dcterms:description": "Example description",
        "dcterms:tableOfContents": "Column A: details",
        "dsv:datasetSchema": {
            "dsv:column": [
                {
                    "schema:name": "col_a",
                    "dcterms:description": "First column",
                    "schema:about": "About A",
                    "schema:unitText": "kg",
                    "prov:hadRole": "measure",
                    "dsv:summaryStatistics": {
                        "dsv:statisticalDataType": "quantitative",
                        "count": 10,
                    },
                },
                {
                    "dcterms:title": "Column B",
                    "schema:identifier": "col_b",
                    "dcterms:description": "Second column",
                },
            ]
        },
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_columns_returns_structured_data(dataset_json: Path) -> None:
    """load_columns should parse dataset and column metadata into dataclasses."""
    columns, meta = load_columns(dataset_json)

    assert isinstance(meta, DatasetMetadata)
    assert meta.as_dict() == {
        "title": "Test dataset",
        "description": "Example description",
        "table_of_contents": "Column A: details",
    }

    assert len(columns) == 2
    first, second = columns
    assert isinstance(first, ColumnInfo)
    assert first.column_id == "col_a"
    assert first.name == "col_a"
    assert first.description == "First column"
    assert first.about == "About A"
    assert first.unit == "kg"
    assert first.role == "measure"
    assert first.statistical_data_type == "quantitative"
    assert first.summary_statistics is not None
    assert first.summary_statistics["count"] == 10

    assert second.column_id == "col_b"
    assert second.name == "Column B"
    assert second.description == "Second column"
    assert second.summary_statistics is None


def test_load_columns_handles_single_column_object(tmp_path: Path) -> None:
    """A solitary dict for dsv:column should be coerced into a list."""
    data = {
        "dsv:datasetSchema": {
            "dsv:column": {
                "schema:identifier": "single",
                "dcterms:description": "Only column",
            }
        }
    }
    path = tmp_path / "single.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    columns, meta = load_columns(path)

    assert meta.title is None
    assert len(columns) == 1
    assert columns[0].name == "single"
    assert columns[0].description == "Only column"


def test_load_columns_ignores_invalid_entries(tmp_path: Path) -> None:
    """Non-dict or unnamed entries should be skipped."""
    data = {
        "dsv:datasetSchema": {
            "dsv:column": [
                "bad",
                {"not_useful": True},
                {"schema:name": "valid", "dcterms:description": "works"},
            ]
        }
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    columns, _ = load_columns(path)
    assert [col.name for col in columns] == ["valid"]
