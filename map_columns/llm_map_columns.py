#!/usr/bin/env python3
"""
Use an LLM to map dataset columns to codes from a Datasette-backed terminology index
(using llm + llm-tools-datasette) and emit SSSOM TSV.

- Input: DCAT style JSON/JSON-LD with dsv:datasetSchema.dsv:column[], and optionally:
    dcterms:title, dcterms:description, dcterms:tableOfContents
- Terminology: any vocabularies exposed via a Datasette `codes` table
  (e.g. SNOMED, LOINC, Wikidata proxy: WD_DISEASE, WD_TEST, ...)

Example:

    python llm_map_columns.py \
    dataset.json \
    --datasette-url http://127.0.0.1:8001/terminology \
    --model gpt-4.1-mini \
    --output mappings.sssom.tsv \
    --extra-prompt "Prefer WD_DISEASE and WD_TEST over SNOMED for this project." \
    --verbose
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import defopt
import llm
from llm_tools_datasette import Datasette

logger = logging.getLogger(__name__)

# Core SSSOM columns + a couple of useful extras
SSSOM_COLUMNS = [
    "subject_id",
    "subject_label",
    "predicate_id",
    "object_id",
    "object_label",
    "mapping_justification",
    "confidence",
    "comment",
]


@dataclass
class ColumnInfo:
    """Flattened view of a dsv:Column entry."""

    column_id: str
    name: str
    description: str
    about: str
    unit: str
    role: str
    statistical_data_type: str


def build_system_prompt(dataset_meta: Dict[str, str], extra_prompt: str = "") -> str:
    """Build the system prompt, including dataset-level metadata and codebook."""
    title = dataset_meta.get("title", "")
    desc = dataset_meta.get("description", "")
    toc = dataset_meta.get("table_of_contents", "")

    base = f"""
You map dataset variables to codes from one or more vocabularies loaded into a Datasette `codes` table.

Dataset metadata:
- title: {title}
- description: {desc}

Additional dataset documentation (may include full codebook and value definitions):
{toc}

The `codes` table has columns:

- system: short identifier for the vocabulary (for example "SNOMED", "LOINC",
          "WD_DISEASE", "WD_TEST", "ICD10", etc.)
- code: the code/id string within that vocabulary
- label: preferred label
- synonyms: optional synonyms / alternate labels

You have a Datasette tool connected to this database. Use it to run SQL queries
against `codes` whenever you need to find candidate codes.

General guidance:

- Use SQL queries that search `label` and `synonyms` and filter by `system` when helpful.
- Inspect the returned rows and pick the best one or two candidates per variable.
- Be conservative: do not map identifiers, free text, dates, or non-semantic flags
  unless there is a clear and widely used code.
- Use skos:exactMatch only for very tight equivalence, skos:closeMatch for strong
  but imperfect matches, and skos:relatedMatch for looser associations.

You will always return SSSOM-style mappings as JSON objects but you must not
invent codes that are not actually present in the `codes` table.
""".strip()

    if extra_prompt:
        base += "\n\nAdditional hints from the user:\n" + extra_prompt.strip()

    return base


def parse_dataset_json(path: Path) -> tuple[List[ColumnInfo], Dict[str, str]]:
    """Parse dsv:datasetSchema/dsv:column[] and dataset-level metadata."""
    data = json.loads(path.read_text(encoding="utf-8"))

    dataset_meta = {
        "title": data.get("dcterms:title", ""),
        "description": data.get("dcterms:description", ""),
        "table_of_contents": data.get("dcterms:tableOfContents", ""),
    }

    schema = data.get("dsv:datasetSchema") or {}
    cols_raw = schema.get("dsv:column") or []
    if not isinstance(cols_raw, list):
        cols_raw = [cols_raw]

    columns: List[ColumnInfo] = []
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

        description = col.get("dcterms:description", "") or ""
        about = col.get("schema:about", "") or ""
        unit = col.get("schema:unitText", "") or ""
        role = col.get("prov:hadRole", "") or ""

        stat_type = ""
        stats = col.get("dsv:summaryStatistics")
        if isinstance(stats, dict):
            stat_type = stats.get("dsv:statisticalDataType", "") or ""

        columns.append(
            ColumnInfo(
                column_id=name,
                name=name,
                description=description,
                about=about,
                unit=unit,
                role=role,
                statistical_data_type=stat_type,
            )
        )

    logger.info("Parsed %d columns from %s", len(columns), path)
    return columns, dataset_meta


def map_column(
    model: llm.Model,
    ds_tool: Datasette,
    column: ColumnInfo,
    system_prompt: str,
    subject_prefix: str = "dataset",
) -> List[Dict[str, Any]]:
    """
    Ask the LLM (with the Datasette tool) to map one column.
    Returns a list of mapping dicts compatible with SSSOM core columns.
    """
    var_id = column.column_id

    user_prompt = f"""
You are mapping ONE dataset variable to one or more codes from the vocabularies
available in the Datasette `codes` table.

Variable:
- internal_id: {var_id}
- column_name: {column.name}
- role: {column.role}
- description: {column.description}
- about: {column.about}
- unit: {column.unit}
- statistical_data_type: {column.statistical_data_type}

Tasks:

1. Decide if this variable should be mapped to codes in any of the vocabularies
   present in the `system` column (for example SNOMED, LOINC, WD_DISEASE,
   WD_TEST, ICD10, or others), or not mapped at all (e.g. identifiers, free
   text, etc.).

2. When mapping, you MUST:
   - Use the Datasette tool to query table `codes`.
   - Use SQL queries that search `label` and `synonyms` and filter by `system`
     where helpful. For example (you do NOT need to quote this in the final answer):

       SELECT system, code, label, synonyms
       FROM codes
       WHERE system IN ('WD_DISEASE', 'SNOMED')
         AND label MATCH 'exercise induced angina'
       LIMIT 10;

   - Inspect the returned rows and pick the best candidate codes.

3. Emit your final result as a JSON array of mapping objects.
   Use this structure exactly:

[
  {{
    "subject_id": "{subject_prefix}:{var_id}",
    "subject_label": "<short human label for the column>",
    "predicate_id": "skos:exactMatch | skos:closeMatch | skos:relatedMatch",
    "object_id": "<SYSTEM>:<code>",
    "object_label": "<preferred term label>",
    "mapping_justification": "semapv:ManualMappingCuration or semapv:LexicalMatching or semapv:LogicalReasoning",
    "confidence": <float between 0 and 1>,
    "comment": "<short free-text justification, may be empty>"
  }}
]

Rules:
- If you find multiple clearly useful mappings (e.g. a measurement that should
  be mapped both to a LOINC lab test and a SNOMED observable entity), include
  multiple objects in the JSON array.
- If you cannot find a suitable concept, return an empty JSON array: [].
- Do NOT include any text before or after the JSON. The entire response must be valid JSON.
"""

    logger.info("Requesting mapping for column %s", column.column_id)

    chain = model.chain(
        user_prompt,
        system=system_prompt,
        tools=[ds_tool],
    )

    text = chain.text().strip()
    if not text:
        logger.warning("Empty response for column %s", column.column_id)
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON for column %s; skipping", column.column_id)
        return []

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        logger.warning("Unexpected JSON structure for column %s; skipping", column.column_id)
        return []

    cleaned: List[Dict[str, Any]] = []
    for m in data:
        if not isinstance(m, dict):
            continue
        row = {
            "subject_id": m.get("subject_id", f"{subject_prefix}:{var_id}"),
            "subject_label": m.get("subject_label", column.name or var_id),
            "predicate_id": m.get("predicate_id", ""),
            "object_id": m.get("object_id", ""),
            "object_label": m.get("object_label", ""),
            "mapping_justification": m.get("mapping_justification", ""),
            "confidence": m.get("confidence", ""),
            "comment": m.get("comment", ""),
        }
        if not row["object_id"]:
            continue
        cleaned.append(row)

    logger.info("Column %s: produced %d mappings", column.column_id, len(cleaned))
    return cleaned


def write_sssom(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Write mappings as SSSOM TSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SSSOM_COLUMNS, delimiter="\t")
        writer.writeheader()
        for m in rows:
            writer.writerow({k: m.get(k, "") for k in SSSOM_COLUMNS})
    logger.info("Wrote %d mappings to %s", len(rows), out_path)


def main(
    dataset_json: Path,
    *,
    datasette_url: str = "http://127.0.0.1:8001/terminology",
    model: str = "gpt-4.1-mini",
    output: Path = Path("mappings.sssom.tsv"),
    extra_prompt: str = "",
    subject_prefix: str = "dataset",
    verbose: bool = False,
) -> None:
    """
    Map dataset columns (from JSON/JSON-LD) to terminology codes via Datasette.

    :param dataset_json: Path to JSON/JSON-LD file with dsv:datasetSchema/dsv:column[].
    :param datasette_url: Base URL of the Datasette database that hosts the `codes` table.
    :param model: llm model ID (as configured in `llm`, e.g. "gpt-4.1-mini").
    :param output: Output SSSOM TSV path.
    :param extra_prompt: Extra hints appended to the system prompt.
    :param subject_prefix: Prefix for subject_id, default "dataset".
    :param verbose: If True, set log level to INFO (else WARNING).
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    columns, dataset_meta = parse_dataset_json(dataset_json)
    if not columns:
        logger.warning("No columns found in %s; nothing to do", dataset_json)
        return

    system_prompt = build_system_prompt(dataset_meta, extra_prompt)

    model_obj = llm.get_model(model)
    ds_tool = Datasette(datasette_url)

    all_mappings: List[Dict[str, Any]] = []
    for col in columns:
        mappings = map_column(
            model=model_obj,
            ds_tool=ds_tool,
            column=col,
            system_prompt=system_prompt,
            subject_prefix=subject_prefix,
        )
        all_mappings.extend(mappings)

    write_sssom(all_mappings, output)


if __name__ == "__main__":
    defopt.run(main)
