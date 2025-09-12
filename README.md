# Bayesian Network Reports Pipeline

## Overview
- This project provides a reproducible pipeline to learn and evaluate Bayesian Networks (BNs) on mixed-type tabular datasets and generate human-friendly reports.
- It supports datasets from OpenML and the UCI Machine Learning Repository, learns one or more BN structures using PyBNesian (with configurable search/scoring settings), synthesizes data, and compares fidelity to a MetaSyn baseline with per-variable distance metrics and UMAP visualizations.

## Key features
- Multiple BN configurations in a single run: compare structures, likelihoods, and per-variable distances across configurations.
- BN structure-learning tuning knobs: bn_type, score, operators, max_indegree, seed.
- Arc blacklist: prevent incoming arcs into sensitive variables (sensitive <- non-sensitive), while allowing outgoing arcs from sensitive variables.
- YAML configuration: define multiple BN learning configurations in a file.
- Provider-aware metadata: include links to the dataset’s OpenML/UCI page, JSON-LD metadata, and a merged “Variables and summary” table.

## Installation
- The pipeline depends on packages listed in requirements.txt. The example commands below assume a conda environment named synthdata that already contains the dependencies (including pybnesian) as used in testing.

## CLI usage
Run from the repository root.

- Search for datasets
  - OpenML:
    - `conda run -n synthdata python bn_reports_cli.py search openml --name-substr adult`
  - UCI ML Repository:
    - `conda run -n synthdata python bn_reports_cli.py search uciml --area "Health and Medicine" --name-substr heart`

- Generate reports for specific datasets
  - OpenML (adult):
    - `conda run -n synthdata python bn_reports_cli.py report openml -d adult`
  - UCI ML (dataset id 45: Heart Disease):
    - `conda run -n synthdata python bn_reports_cli.py report uciml -d 45`

### Options (report)
- `--outdir`: Output directory root (default: `docs`). A subdirectory per dataset is created.
- `--configs-yaml`: Path to YAML file defining multiple BN configurations. If omitted, two distinct defaults are used:
  - clg_mi2: CLG BN, score=bic, operators=[arcs], max_indegree=2, seed=42
  - semi_mi5: Semiparametric BN, score=bic, operators=[arcs], max_indegree=5, seed=42
- `--arc-blacklist`: Root variable names (repeatable). If omitted, defaults to:
  - OpenML: [age, sex, race]
  - UCI ML: the metadata “demographics” field, when present (case-insensitive match to columns), otherwise the default above.

## Arc blacklist semantics
- For a set of root variables S and the remaining variables R, the pipeline forbids incoming arcs into root variables from non-root variables.
  - That is, an arc v -> u is disallowed for every v in R and u in S.
  - Outgoing arcs u -> v (from root to non-root) are allowed.

## YAML configuration format
Provide one or more BN structure-learning configurations:

```
configs:
  - name: clg_small
    bn_type: clg
    score: bic
    operators: [arcs]
    max_indegree: 2
    seed: 42
  - name: semi_big
    bn_type: semiparametric
    score: bic
    operators: [arcs]
    max_indegree: 5
    seed: 123
```

## Notes
- Supported bn_type values: clg, semiparametric.
- Supported score values are those accepted by pybnesian.hc (e.g., bic).
- operators is passed directly to hc (e.g., ["arcs"]).
- max_indegree controls parent count constraints and directly affects the learned structure.

## Outputs per dataset
- `report.md` and `index.html`: Full report (HTML rendered from Markdown).
- Dataset metadata: `metadata.json` and `dataset.jsonld`.
- BN structures: `bn_<label>.png` (Graphviz), `structure_<label>.graphml`, `model_<label>.pickle`.
- UMAP images: `umap_real.png`, `umap_metasyn.png`, `umap_bn_<label>.png`.
- MetaSyn model: `metasyn_gmf.json` (when available).

## Troubleshooting
- If log-likelihood columns appear empty in the report, the pipeline automatically ignores non-finite values when calculating the held-out log-likelihood; ensure the training/test splits contain no missing values for the variables used.
- If using `--configs-yaml`, ensure PyYAML is available in your environment.

