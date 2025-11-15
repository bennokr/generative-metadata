# SemSynth 🚀

SemSynth is a compact toolkit to profile tabular datasets, synthesize data with multiple backends, and generate a clean HTML report. It supports datasets from OpenML and the UCI Machine Learning Repository.

## ✨ Features
- Unified model interface: run PyBNesian, SynthCity, and MetaSyn models from a single YAML.
- Uniform outputs: each model writes artifacts under `dataset/models/<model-name>/`.
- Optional MetaSyn baseline: enable or disable per report via the config bundle.
- Provider-aware metadata and UMAP visuals.

## 🔎 Quick start
1. Search datasets
   - OpenML: `python semsynth_reports_cli.py search openml --name-substr adult`
   - UCI ML: `python semsynth_reports_cli.py search uciml --area "Health and Medicine" --name-substr heart`

   The `search` command accepts:
   - `--name-substr` (substring filter applied case-insensitively)
   - `--area` (UCI ML topic area, ignored for OpenML)
   - `--cat-min` / `--num-min` (minimum categorical/numeric columns)
   - `--verbose` (emits info logs while querying providers)

2. Minimal report (metadata only) 🧪
   - Leave `--configs-yaml` empty to skip model execution.
   - Example: `python semsynth_reports_cli.py report uciml --datasets 45 -v`
   - Optional flags for reports:
     - `--datasets` (one or more dataset identifiers)
     - `--outdir` (defaults to `outputs/`)
     - `--configs-yaml` (path to a YAML bundle; omit for metadata-only runs)
     - `--generate-umap` / `--compute-privacy` / `--compute-downstream` (each accepts `auto`, `on`, or `off`)
     - `--overwrite-umap` (regenerate even if projections exist)
     - `--verbose` (turn on info logging)

   With the command above you receive dataset metadata, a real-data UMAP projection, and HTML/Markdown reports under `outputs/<Dataset Name>/`.

3. Full report with synthetic models 🤖
    - Pick a configuration bundle from `configs/`:
      - `configs/simple_config.yaml` (MetaSyn + two PyBNesian models)
      - `configs/advanced_config.yaml` (MetaSyn, PyBNesian, and SynthCity models)
      - `configs/maximal_config.yaml` (enables UMAP, privacy, downstream metrics, and the broadest mix of MetaSyn, PyBNesian, and SynthCity generators)
      - `configs/only_metasyn_config.yaml` (MetaSyn baseline only)
   - Example: `python semsynth_reports_cli.py report openml --datasets adult --configs-yaml configs/advanced_config.yaml --generate-umap on --compute-privacy on --compute-downstream on`

   Report toggles accept `auto` (respect YAML defaults), `on`, or `off`. Use `auto` when your YAML sets global defaults for `generate_umap`, `compute_privacy`, or `compute_downstream`.

## 📄 Unified YAML format
- `configs/simple_config.yaml` mixes MetaSyn with two PyBNesian baselines.
- `configs/advanced_config.yaml` extends the simple bundle with SynthCity generators.
- `configs/maximal_config.yaml` enables every optional report toggle and includes the widest selection of MetaSyn, PyBNesian, and SynthCity generators.
- `configs/only_metasyn_config.yaml` keeps MetaSyn as the single synthetic data baseline.

Example:
```yaml
configs:
  - name: metasyn
    backend: metasyn
  - name: clg_mi2
    backend: pybnesian
    model:
      type: clg
      score: bic
      operators: [arcs]
      max_indegree: 2
      seed: 42
  - name: ctgan_fast
    backend: synthcity
    model:
      type: ctgan
      epochs: 5
      batch_size: 256
    rows: 1000
    seed: 42
```

## 📦 Outputs
- Per dataset (e.g., `outputs/Heart Disease/`):
  - `dataset.json` (schema.org/Dataset JSON-LD)
  - `dataset.semmap.json` (optional, if curated metadata is found)
  - `index.html` and `report.md`
  - `umap_real.png` and optional `umap_metasyn.png`
- Per model (e.g., `outputs/Heart Disease/models/<name>/`):
  - `synthetic.csv`, `per_variable_metrics.csv`, `metrics.json`, `umap.png`
  - PyBNesian-only extras: `bn_<name>.png`, `structure_<name>.graphml`, `model_<name>.pickle`
  - `synthetic.semmap.parquet` (when SemMap metadata is available)

## 🧰 Metadata templates & column mappings
- `uci_template.py` turns the JSON payloads cached under `uciml-cache/` (sometimes referenced as `uci-cache/` in earlier docs) into DCAT + DSV JSON-LD that downstream tools can ingest.
- Scripts under `map_columns/` take that JSON-LD and suggest or write terminology mappings (see `map_columns/README.md`).

### Example: UCI dataset 45 (Heart Disease)
1. Fetch the dataset metadata. Any command that touches the UCI provider will populate `uciml-cache/<id>.json`. For example:
   ```bash
   python semsynth_reports_cli.py report uciml -d 45 --configs-yaml configs/empty.yaml --metasyn false
   ```
   This creates `uciml-cache/45.json` alongside the cached CSV/metadata used by the reporting pipeline.
2. Convert the cached metadata into DCAT + DSV JSON-LD:
   ```bash
   python uci_template.py uciml-cache/45.json heart-dataset.jsonld
   ```
   The resulting `heart-dataset.jsonld` contains dataset-level `dcat:Dataset` fields plus a `dsv:datasetSchema` block with each variable from the Heart Disease dataset.
3. Suggest terminology mappings for every variable description using the keyword search helper:
   ```bash
   python map_columns/kwd_map_columns.py heart-dataset.jsonld \
       --datasette-db-url http://127.0.0.1:8001/terminology \
       --table codes \
       --limit 5 \
       --verbose
   ```
   Swap in `map_columns/llm_map_columns.py` to drive an LLM-backed workflow that writes an `*.sssom.tsv` file (see `map_columns/README.md` for details).

## 📝 Notes
- Metadata-only reports require no YAML file; pass `--configs-yaml` to opt into synthetic runs.
- All models are treated uniformly in the report; UMAPs share the same projection trained on real data.

## 📚 Documentation
- Build the Sphinx site with `sphinx-build -b html sphinx docs`.
