# SemSynth 🚀

SemSynth is a compact toolkit to profile tabular datasets, synthesize data with multiple backends, and generate a clean HTML report. It supports datasets from OpenML and the UCI Machine Learning Repository.

## ✨ Features
- Unified model interface: run both PyBNesian and SynthCity models from a single YAML.
- Uniform outputs: each model writes artifacts under `dataset/models/<model-name>/`.
- Optional MetaSyn baseline: enable or disable per report.
- Provider-aware metadata and UMAP visuals.

## 🔎 Quick start
1. Search datasets
   - OpenML: `python semsynth_reports_cli.py search openml --name-substr adult`
   - UCI ML: `python semsynth_reports_cli.py search uciml --area "Health and Medicine" --name-substr heart`

2. Minimal report (no models, no MetaSyn) 🧪
   - Use the empty config: `configs/empty.yaml`
   - Run: `python semsynth_reports_cli.py report uciml -d 45 --configs-yaml configs/empty.yaml --metasyn false`
   - Outputs go to `docs/<Dataset Name>/` and include metadata, a UMAP of the real data, and a compact HTML report.

3. Report with models 🤖
   - Default models: `configs/default_config.yaml` (two PyBNesian models)
   - Mixed models: `configs/advanced_models.yaml` (PyBNesian + SynthCity)
   - Example: `python semsynth_reports_cli.py report openml -d adult --configs-yaml configs/advanced_models.yaml`

4. Single-dataset synthesis 🧬
   - SynthCity: `python semsynth_reports_cli.py synth 1590 --provider openml --backend synthcity --generator ctgan --gen-params-json '{"epochs": 5}' --rows 1000 --outdir outputs`
   - PyBNesian: `python semsynth_reports_cli.py synth 1590 --provider openml --backend pybnesian --generator clg --gen-params-json '{"score":"bic","operators":["arcs"],"max_indegree":2}' --rows 1000 --outdir outputs`

## 📄 Unified YAML format
- `configs/default_config.yaml` is used when you don’t pass `--configs-yaml`.
- `configs/advanced_models.yaml` mixes multiple backends.

Example:
```yaml
configs:
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
      name: ctgan
      params:
        epochs: 5
        batch_size: 256
    rows: 1000
    seed: 42
```

## 📦 Outputs
- Per dataset (e.g., `docs/Heart Disease/`):
  - `dataset.json` (schema.org/Dataset JSON-LD)
  - `dataset.semmap.json` (optional, if curated metadata is found)
  - `index.html` and `report.md`
  - `umap_real.png` and optional `umap_metasyn.png`
- Per model (e.g., `docs/Heart Disease/models/<name>/`):
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
- Minimal report: use `configs/empty.yaml` and `--metasyn false`.
- All models are treated uniformly in the report; UMAPs share the same projection trained on real data.
