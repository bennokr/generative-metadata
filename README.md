# Bayesian Network Reports Pipeline

## Overview
- This project provides a reproducible pipeline to learn and evaluate Bayesian Networks (BNs) on mixed-type tabular datasets and generate human-friendly reports.
- It supports datasets from OpenML and the UCI Machine Learning Repository, learns one or more BN structures using PyBNesian (with configurable search/scoring settings), synthesizes data, and compares fidelity to a MetaSyn baseline with per-variable distance metrics and UMAP visualizations.

## Key features
- Multiple BN configurations in a single run: compare structures, likelihoods, and per-variable distances across configurations.
- BN structure-learning tuning knobs: bn_type, score, operators, max_indegree, seed.
- Arc blacklist: prevent incoming arcs into root variables (root <- others), while allowing outgoing arcs from root variables.
- YAML configuration: define multiple BN learning configurations in a file.
- Provider-aware metadata: include links to the dataset’s OpenML/UCI page, JSON-LD metadata, and a merged “Variables and summary” table.
- Synthcity integration: train tabular generators from the CLI and surface their metrics/plots alongside BN results.

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
- Generate synthetic data with synthcity
  - Example:
    - `conda run -n synthdata python bn_reports_cli.py synth 1590 --provider openml --generator ctgan --gen-params-json '{}' --rows 500 --outdir outputs`

### Options (report)
- `--outdir`: Output directory root (default: `docs`). A subdirectory per dataset is created.
- `--configs-yaml`: Path to YAML file defining multiple BN configurations. If omitted, two distinct defaults are used:
  - clg_mi2: CLG BN, score=bic, operators=[arcs], max_indegree=2, seed=42
  - semi_mi5: Semiparametric BN, score=bic, operators=[arcs], max_indegree=5, seed=42
- `--arc-blacklist`: Root variable names (repeatable). If omitted, defaults to:
  - OpenML: [age, sex, race]
  - UCI ML: the metadata “demographics” field, when present (case-inroot match to columns), otherwise the default above.

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

## Limitation: UCI API metadata vs. actual on-disk datatypes
- For UCI ML Repository datasets, we fetch descriptive metadata from the UCI API (cached under uciml-cache/) but we always infer variable types from the actual data file (via ucimlrepo -> data.csv).
- Some UCI datasets publish pre-discretized/binned numeric attributes as strings (e.g., "112 - 154", "< 3.65", "≥ 1.023"). When loaded, pandas treats these as object dtype. Our type inference therefore marks such columns as discrete, even if the UCI API metadata calls them Integer/Real.
- Integer-coded flags (e.g., 0/1) are also treated as discrete by design (low cardinality integers are considered discrete in bncli/utils.infer_types).
- As a result, the report may show more discrete variables than suggested by the UCI API. This is expected and reflects the data as provided in the file, which is what the models are fit to.

Example: Risk Factor Prediction of Chronic Kidney Disease (UCI id 857)
- UCI metadata lists feature_types = ["Real"] and per-variable types such as Integer/Continuous.
- The actual CSV contains many range-labeled strings and threshold bins for clinical measurements (e.g., bgr, bu, sc, hemo, etc.), so the pipeline infers these as discrete and the report shows mostly discrete variables.

If you need continuous modeling for binned columns
- Preprocess the dataset to convert bin labels to numeric values (e.g., midpoint of range) before running the pipeline, or customize the inference to coerce specific columns to floats.
- Alternatively, the reporting could be extended to display both the UCI-declared type and the inferred type; open an issue if you want this surfaced in the reports.

## Outputs per dataset
- `report.md` and `index.html`: Full report (HTML rendered from Markdown).
- Dataset metadata: `dataset.json`.
- BN structures: `bn_<label>.png` (Graphviz), `structure_<label>.graphml`, `model_<label>.pickle`.
- UMAP images: `umap_real.png`, `umap_metasyn.png`, `umap_bn_<label>.png`.
- MetaSyn model: `metasyn_gmf.json` (when available).

## Troubleshooting
- If log-likelihood columns appear empty in the report, the pipeline automatically ignores non-finite values when calculating the held-out log-likelihood; ensure the training/test splits contain no missing values for the variables used.
- If using `--configs-yaml`, ensure PyYAML is available in your environment.

## Supported synthcity generators
- ctgan
- ads-gan / adsgan
- pategan
- dp-gan / dpgan
- tvae
- rtvae
- nflow / tabularflow
- bn / bayesiannetwork
- privbayes
- arf / arfpy
- great
