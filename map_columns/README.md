# Terminology Index and Mapping Scripts

This repository contains several Python scripts designed to build a small medical terminology index and perform mappings between dataset variables and relevant medical codes. You can use both closed (license based) SNOMED CT / LOINC and open data from Wikidata. The mapping can be done with a keyword index or an LLM that wraps this index and re-ranks.

## Table of Contents

- [Scripts Overview](#scripts-overview)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [1. Build SNOMED and LOINC Codes Table](#1-build-snomed-and-loinc-codes-table)
  - [2. Extract Medical Codes from Wikidata](#2-extract-medical-codes-from-wikidata)
  - [3. Map Dataset Columns to Codes using Keyword Search](#3-map-dataset-columns-to-codes-using-keyword-search)
  - [4. Map Dataset Columns to Codes using LLM](#4-map-dataset-columns-to-codes-using-llm)

## Scripts Overview

1. `build_snomed_loinc_codes_table.py` - Builds a TSV containing codes from SNOMED CT and LOINC.
2. `extract_wikidata_medical_codes_table.py` - Extracts medical terminology from Wikidata and outputs a TSV.
3. `kwd_map_columns.py` - Maps dataset columns to codes using a keyword search index over a Datasette-backed terminology index.
4. `llm_map_columns.py` - Utilizes an LLM to map dataset columns to codes from a Datasette-backed terminology index.

## Prerequisites

You can install the necessary libraries using pip:

```bash
pip install requests defopt datasette sqlite-utils sssom
```

To use the LLM script, run:

```bash
pip install llm
llm install llm-tools-datasette
```

## Usage

### 1. Build SNOMED and LOINC Codes Table

This script processes SNOMED CT and LOINC files to create a `codes.tsv` file.

```bash
python build_snomed_loinc_codes_table.py \
    --snomed-description /path/to/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_*.txt \
    --loinc /path/to/Loinc.csv \
    --out codes.tsv \
    --max-snomed 50000
```

After running the script, you can insert the data into a SQLite database:

```bash
sqlite-utils insert terminology.db codes codes.tsv --tsv
sqlite-utils enable-fts terminology.db codes label synonyms --create-triggers
```

### 2. Extract Medical Codes from Wikidata

This script fetches medical codes from Wikidata and outputs them to `codes.tsv`.

```bash
python extract_wikidata_medical_codes_table.py
```

Run the command above to perform the extraction. After execution, the results will be written to `codes.tsv` in the current directory.

### 3. Map Dataset Columns to Codes using Keyword Search

This script maps dataset columns to semantic codes from a Datasette database based on their descriptions.

```bash
python kwd_map_columns.py dataset.json \
    --datasette-db-url http://127.0.0.1:8001/terminology \
    --table codes \
    --limit 5 \
    --verbose
```

Replace `dataset.json` with the path to your JSON file that contains the dataset schema. The results of the mapping will be printed in the terminal.

### 4. Map Dataset Columns to Codes using LLM

This script uses an LLM to intelligently map dataset columns to codes based on their metadata from a Datasette instance.

```bash
python llm_map_columns.py \
    dataset.json \
    --datasette-url http://127.0.0.1:8001/terminology \
    --model gpt-4.1-mini \
    --output mappings.sssom.tsv \
    --extra-prompt "Prefer WD_DISEASE and WD_TEST over SNOMED for this project." \
    --verbose
```

Providing a JSON with dataset metadata as `dataset.json` will produce SSSOM-style mappings and save them to `mappings.sssom.tsv`.
