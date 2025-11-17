"""Merge SSSOM mappings into SemMap/DSV metadata."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import defopt

from semsynth.semmap import ColumnProperty, Metadata

logger = logging.getLogger(__name__)


PREDICATE_TO_FIELD = {
    "skos:exactMatch": "exactMatch",
    "skos:closeMatch": "closeMatch",
    "skos:relatedMatch": "relatedMatch",
    "skos:broadMatch": "broadMatch",
    "skos:narrowMatch": "narrowMatch",
}


def _normalize_subject_id(subject_id: str) -> str:
    """Extract a column identifier from an SSSOM subject id."""

    token = subject_id.rsplit(":", 1)[-1]
    token = token.rsplit("/", 1)[-1]
    return token


def _load_sssom(tsv_path: Path) -> List[Dict[str, str]]:
    """Load mappings from an SSSOM TSV file."""

    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [row for row in reader]


def _ensure_column_property(prop: Optional[ColumnProperty]) -> ColumnProperty:
    if prop is None:
        return ColumnProperty()
    return prop


def integrate_sssom(
    metadata: Metadata, mappings: List[Dict[str, str]]
) -> Metadata:
    """Attach SSSOM mappings to column properties in the SemMap metadata."""

    column_lookup = {col.name: col for col in metadata.datasetSchema.columns if col.name}
    for row in mappings:
        subj = row.get("subject_id")
        predicate = row.get("predicate_id")
        obj = row.get("object_id")
        if not subj or not predicate or not obj:
            continue
        column_key = _normalize_subject_id(subj)
        column = column_lookup.get(column_key)
        if column is None:
            continue

        field = PREDICATE_TO_FIELD.get(predicate)
        if field is None:
            continue
        column.columnProperty = _ensure_column_property(column.columnProperty)
        existing = getattr(column.columnProperty, field)
        if existing is None:
            setattr(column.columnProperty, field, [obj])
        elif isinstance(existing, list):
            if obj not in existing:
                existing.append(obj)
        else:
            setattr(column.columnProperty, field, [existing, obj])
    return metadata


def main(
    dataset_json: Path,
    sssom_tsv: Path,
    *,
    output_json: Optional[Path] = None,
    indent: int = 2,
) -> None:
    """Merge SSSOM mappings into SemMap metadata and write JSON-LD.

    Args:
        dataset_json: Path to SemMap/DSV JSON-LD metadata file.
        sssom_tsv: Path to an SSSOM TSV file with subject/object pairs.
        output_json: Optional output path for the merged JSON-LD. When omitted,
            the original ``dataset_json`` is overwritten.
        indent: JSON indentation level for the saved metadata.
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    metadata = Metadata.from_dcat_dsv(json.loads(dataset_json.read_text()))
    mappings = _load_sssom(sssom_tsv)
    updated = integrate_sssom(metadata, mappings)

    target = output_json or dataset_json
    target.write_text(
        json.dumps(updated.to_jsonld(), indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote merged SemMap metadata to %s", target)


if __name__ == "__main__":
    defopt.run(main)
