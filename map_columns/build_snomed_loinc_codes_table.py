#!/usr/bin/env python3
"""
Build a small terminology index (SNOMED + LOINC) for llm.datasette.io.

Input:
  - SNOMED CT RF2 Description Snapshot (English)
  - LOINC Loinc.csv

Output:
  - codes.tsv with columns: system, code, label, synonyms

Example:

  python build_snomed_loinc_codes_table.py \
      --snomed-description /path/to/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_*.txt \
      --loinc /path/to/Loinc.csv \
      --out codes.tsv \
      --max-snomed 50000

Then:

  sqlite-utils insert terminology.db codes codes.tsv --tsv
  sqlite-utils enable-fts terminology.db codes label synonyms --create-triggers
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set, Optional

# SNOMED description type IDs (FSN + synonym)  
FSN_TYPE_ID = "900000000000003001"
SYN_TYPE_ID = "900000000000013009"

# Rough semantic tags we keep for SNOMED (from the FSN "(tag)" part)
COMMON_SNOMED_TAGS = {
    "disorder",
    "finding",
    "procedure",
    "observable entity",
    "situation",
    "event",
    "body structure",
    "morphologic abnormality",
    "substance",
    "product",
    "qualifier value",
    "specimen",
    "clinical drug",
    "medicinal product",
}

# LOINC column names we will use  
LOINC_CODE_COL = "LOINC_NUM"
LOINC_LONG_NAME_COL = "LONG_COMMON_NAME"
LOINC_SHORT_NAME_COL = "SHORTNAME"
LOINC_COMPONENT_COL = "COMPONENT"
LOINC_COMMON_TEST_RANK_COL = "COMMON_TEST_RANK"
LOINC_COMMON_ORDER_RANK_COL = "COMMON_ORDER_RANK"
LOINC_STATUS_COL = "STATUS"
LOINC_DISPLAY_NAME_COL = "DISPLAY_NAME"
LOINC_CONSUMER_NAME_COL = "CONSUMER_NAME"


# ---------------------------------------------------------------------------
# Utility: parse SNOMED FSN semantic tag
# ---------------------------------------------------------------------------

def extract_semantic_tag(fsn: str) -> str:
    """
    Extract semantic tag from an FSN, e.g.:

        "Angina pectoris (disorder)" -> "disorder"

    Returns empty string if no bracketed tag is found.
    """
    m = re.search(r"\(([^()]+)\)\s*$", fsn)
    if not m:
        return ""
    return m.group(1).strip().lower()


def strip_semantic_tag(fsn: str) -> str:
    """
    Remove semantic tag from FSN to get a nicer label:

        "Angina pectoris (disorder)" -> "Angina pectoris"
    """
    return re.sub(r"\s*\([^()]+\)\s*$", "", fsn).strip()


# ---------------------------------------------------------------------------
# SNOMED: Description RF2 → filtered {system, code, label, synonyms}
# ---------------------------------------------------------------------------

def load_snomed_codes(
    description_path: Path,
    language: str = "en",
    allowed_tags: Optional[Set[str]] = None,
    max_concepts: Optional[int] = None,
) -> List[Tuple[str, str, str, str]]:
    """
    Build a list of (system, code, label, synonyms) for SNOMED.

    - Uses the Description Snapshot file (tab-separated RF2 with header).
    - Keeps only active rows for the given language.
    - Uses FSN rows to select concepts whose semantic tag is in allowed_tags.
    - Collects synonyms for those concepts.
    """
    allowed_tags = allowed_tags or COMMON_SNOMED_TAGS

    fsn_by_concept: Dict[str, str] = {}
    tag_by_concept: Dict[str, str] = {}
    synonyms_by_concept: Dict[str, Set[str]] = defaultdict(set)

    # First pass: capture FSN and synonyms
    with description_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("active") != "1":
                continue
            if row.get("languageCode") != language:
                continue

            concept_id = row.get("conceptId")
            if not concept_id:
                continue

            type_id = row.get("typeId")
            term = (row.get("term") or "").strip()
            if not term:
                continue

            if type_id == FSN_TYPE_ID:
                tag = extract_semantic_tag(term)
                if allowed_tags and tag not in allowed_tags:
                    # Not a concept we care about
                    continue
                # Keep first FSN we see per concept
                if concept_id not in fsn_by_concept:
                    fsn_by_concept[concept_id] = term
                    tag_by_concept[concept_id] = tag

            elif type_id == SYN_TYPE_ID:
                synonyms_by_concept[concept_id].add(term)

    # Build final rows
    codes: List[Tuple[str, str, str, str]] = []
    for concept_id, fsn in fsn_by_concept.items():
        label = strip_semantic_tag(fsn)
        syns = set(synonyms_by_concept.get(concept_id, set()))
        syns.add(label)  # include label itself for search
        synonyms_str = "; ".join(sorted(syns))
        codes.append(("SNOMED", concept_id, label, synonyms_str))

    # Optionally truncate to first N concepts
    if max_concepts is not None and len(codes) > max_concepts:
        codes = codes[:max_concepts]

    return codes


# ---------------------------------------------------------------------------
# LOINC: Loinc.csv → filtered {system, code, label, synonyms}
# ---------------------------------------------------------------------------

def load_loinc_codes(
    loinc_path: Path,
    only_common: bool = True,
) -> List[Tuple[str, str, str, str]]:
    """
    Build a list of (system, code, label, synonyms) for LOINC.

    - Reads Loinc.csv (comma-separated with header).
    - Keeps only ACTIVE records.
    - If only_common=True, keeps only rows where COMMON_TEST_RANK or
      COMMON_ORDER_RANK is populated (Top 20k list).
    """
    codes: List[Tuple[str, str, str, str]] = []

    with loinc_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loinc_num = (row.get(LOINC_CODE_COL) or "").strip()
            if not loinc_num:
                continue

            status = (row.get(LOINC_STATUS_COL) or "").upper()
            if status and status != "ACTIVE":
                # Skip deprecated/suppressed codes
                continue

            if only_common:
                common_test = (row.get(LOINC_COMMON_TEST_RANK_COL) or "").strip()
                common_order = (row.get(LOINC_COMMON_ORDER_RANK_COL) or "").strip()
                if not common_test and not common_order:
                    # Not part of top common orders/results
                    continue

            long_name = (row.get(LOINC_LONG_NAME_COL) or "").strip()
            short_name = (row.get(LOINC_SHORT_NAME_COL) or "").strip()
            component = (row.get(LOINC_COMPONENT_COL) or "").strip()

            label = long_name or short_name or component
            if not label:
                continue

            # Build synonyms from a few fields, deduplicated
            syns: List[str] = []
            for field in [
                LOINC_SHORT_NAME_COL,
                LOINC_COMPONENT_COL,
                LOINC_DISPLAY_NAME_COL,
                LOINC_CONSUMER_NAME_COL,
            ]:
                v = (row.get(field) or "").strip()
                if v:
                    syns.append(v)
            # Remove duplicates preserving order
            seen = set()
            uniq_syns = []
            for s in syns:
                if s not in seen:
                    seen.add(s)
                    uniq_syns.append(s)

            synonyms_str = "; ".join(uniq_syns)
            codes.append(("LOINC", loinc_num, label, synonyms_str))

    return codes


# ---------------------------------------------------------------------------
# Writing TSV
# ---------------------------------------------------------------------------

def write_codes_tsv(
    rows: Iterable[Tuple[str, str, str, str]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["system", "code", "label", "synonyms"])
        for system, code, label, synonyms in rows:
            writer.writerow([system, code, label, synonyms])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build codes.tsv from SNOMED RF2 Description and LOINC Loinc.csv"
    )
    parser.add_argument(
        "--snomed-description",
        type=Path,
        required=True,
        help="Path to SNOMED sct2_Description_Snapshot-en_*.txt",
    )
    parser.add_argument(
        "--loinc",
        type=Path,
        required=True,
        help="Path to LOINC Loinc.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("codes.tsv"),
        help="Output TSV (default: codes.tsv)",
    )
    parser.add_argument(
        "--max-snomed",
        type=int,
        default=None,
        help="Optional max number of SNOMED concepts to keep",
    )
    parser.add_argument(
        "--loinc-all",
        action="store_true",
        help="If set, include all ACTIVE LOINC terms, not just common ones",
    )

    args = parser.parse_args()

    snomed_codes = load_snomed_codes(
        description_path=args.snomed_description,
        language="en",
        allowed_tags=COMMON_SNOMED_TAGS,
        max_concepts=args.max_snomed,
    )

    loinc_codes = load_loinc_codes(
        loinc_path=args.loinc,
        only_common=not args.loinc_all,
    )

    combined = list(snomed_codes) + list(loinc_codes)
    write_codes_tsv(combined, args.out)

    print(
        f"Wrote {len(combined)} rows "
        f"({len(snomed_codes)} SNOMED, {len(loinc_codes)} LOINC) to {args.out}"
    )


if __name__ == "__main__":
    main()
