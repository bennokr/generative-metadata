#!/usr/bin/env python3
"""
Use an keyword search index to map dataset columns to codes and emit SSSOM TSV.

It reads DataSet Variable JSON (https://w3id.org/dsv-ontology/) with dsv:datasetSchema.dsv:column[], and for each column query a Datasette `codes` table using dcterms:description as the search term.

Example:

  python kwd_map_columns.py dataset.json \
    --datasette-db-url http://127.0.0.1:8001/terminology \
    --table codes \
    --limit 5 \
    --verbose
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import defopt
import requests

logger = logging.getLogger(__name__)


def load_columns(path: Path) -> List[Dict[str, Any]]:
    """Extract column name + description from dsv:datasetSchema/dsv:column[]."""
    data = json.loads(path.read_text(encoding="utf-8"))

    schema = data.get("dsv:datasetSchema") or {}
    cols_raw = schema.get("dsv:column") or []
    if not isinstance(cols_raw, list):
        cols_raw = [cols_raw]

    columns: List[Dict[str, Any]] = []
    for col in cols_raw:
        if not isinstance(col, dict):
            continue
        name = (
            col.get("schema:name")
            or col.get("dcterms:title")
            or col.get("schema:identifier")
        )
        if not name:
            continue
        desc = col.get("dcterms:description", "") or ""
        columns.append({"name": name, "description": desc})

    logger.info("Loaded %d columns from %s", len(columns), path)
    return columns


def query_codes(
    datasette_db_url: str,
    table: str,
    term: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Query Datasette /<db>/<table>.json using _search on label+synonyms.
    Returns list of row dicts.
    """
    base = datasette_db_url.rstrip("/")
    url = f"{base}/{table}.json"

    # Use _search with both label and synonyms columns (if FTS enabled)
    params = [
        ("_search", term),
        ("_size", str(limit)),
        ("_search_columns", "label"),
        ("_search_columns", "synonyms"),
    ]

    logger.info("GET %s term=%r", url, term)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("rows", [])
    return rows


def main(
    dataset_json: Path,
    *,
    datasette_db_url: str = "http://127.0.0.1:8001/terminology",
    table: str = "codes",
    limit: int = 5,
    verbose: bool = False,
) -> None:
    """
    Suggest candidate terminology rows for each column using Datasette.

    :param dataset_json: Path to JSON/JSON-LD with dsv:datasetSchema/dsv:column[].
    :param datasette_db_url: Base URL of the Datasette database
        (e.g. http://127.0.0.1:8001/terminology).
    :param table: Name of the codes table in that database (default: "codes").
    :param limit: Max number of rows to show per column.
    :param verbose: If True, set log level to INFO.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    columns = load_columns(dataset_json)
    if not columns:
        logger.warning("No columns found, nothing to do")
        return

    for col in columns:
        name = col["name"]
        desc = col["description"].strip()
        if not desc:
            logger.info("Column %s has no dcterms:description; skipping", name)
            continue

        print(f"\n=== Column: {name} ===")
        print(f"Description: {desc}\n")

        try:
            rows = query_codes(datasette_db_url, table, desc, limit)
        except Exception as e:
            logger.warning("Error querying Datasette for %s: %s", name, e)
            continue

        if not rows:
            print("  (no hits)")
            continue

        for r in rows:
            system = r.get("system", "")
            code = r.get("code", "")
            label = r.get("label", "")
            syn = r.get("synonyms", "")
            print(f"- [{system}] {code} :: {label}")
            if syn:
                print(f"    synonyms: {syn}")


if __name__ == "__main__":
    defopt.run(main)
