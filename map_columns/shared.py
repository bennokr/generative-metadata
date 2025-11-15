"""Shared helpers for parsing dataset column metadata."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset-level metadata extracted from DCAT/DSV JSON."""

    title: Optional[str] = None
    description: Optional[str] = None
    table_of_contents: Optional[str] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        """Return the metadata as a dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "table_of_contents": self.table_of_contents,
        }


@dataclass(frozen=True)
class ColumnInfo:
    """Flattened representation of a ``dsv:column`` entry."""

    column_id: Optional[str]
    name: Optional[str]
    description: Optional[str] = None
    about: Optional[str] = None
    unit: Optional[str] = None
    role: Optional[str] = None
    statistical_data_type: Optional[str] = None
    summary_statistics: Optional[Dict[str, Any]] = None


def _ensure_list(value: Any) -> Iterable[Any]:
    """Return ``value`` as an iterable list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_optional_str(value: Any) -> Optional[str]:
    """Convert ``value`` to a trimmed string when possible."""
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return text or None
    return None


def load_columns(path: Path) -> Tuple[List[ColumnInfo], DatasetMetadata]:
    """Load dataset metadata and column definitions from JSON.

    Args:
        path: Location of the DCAT/DSV JSON/JSON-LD file.

    Returns:
        A tuple of ``(columns, dataset_metadata)`` where ``columns`` is a list
        of :class:`ColumnInfo` and ``dataset_metadata`` describes the dataset.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    metadata = DatasetMetadata(
        title=_coerce_optional_str(data.get("dcterms:title")),
        description=_coerce_optional_str(data.get("dcterms:description")),
        table_of_contents=_coerce_optional_str(data.get("dcterms:tableOfContents")),
    )

    schema = data.get("dsv:datasetSchema") or {}
    raw_columns = _ensure_list(schema.get("dsv:column") or [])

    columns: List[ColumnInfo] = []
    for entry in raw_columns:
        if not isinstance(entry, dict):
            continue

        name = (
            _coerce_optional_str(entry.get("schema:name"))
            or _coerce_optional_str(entry.get("dcterms:title"))
            or _coerce_optional_str(entry.get("schema:identifier"))
        )
        if not name:
            continue

        column_id = _coerce_optional_str(entry.get("schema:identifier")) or name

        summary_stats_raw = entry.get("dsv:summaryStatistics")
        summary_stats: Optional[Dict[str, Any]] = None
        statistical_data_type: Optional[str] = None
        if isinstance(summary_stats_raw, dict):
            summary_stats = dict(summary_stats_raw)
            statistical_data_type = _coerce_optional_str(
                summary_stats_raw.get("dsv:statisticalDataType")
            )

        columns.append(
            ColumnInfo(
                column_id=column_id,
                name=name,
                description=_coerce_optional_str(entry.get("dcterms:description")),
                about=_coerce_optional_str(entry.get("schema:about")),
                unit=_coerce_optional_str(entry.get("schema:unitText")),
                role=_coerce_optional_str(entry.get("prov:hadRole")),
                statistical_data_type=statistical_data_type,
                summary_statistics=summary_stats,
            )
        )

    logger.info("Loaded %d columns from %s", len(columns), path)
    return columns, metadata
