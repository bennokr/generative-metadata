# Provenance workflow guide

This document explains how SemSynth's provenance-aware build graph is assembled with [makeprov](https://github.com/bennokr/makeprov) and how it emits W3C [PROV-O](https://www.w3.org/TR/prov-o/) metadata alongside the reports you generate. It points to the concrete rules, helper functions, and configuration files that control provenance so you can trace every cached download, visualization, and metric back to its source.

## Reference links
- [makeprov README](https://github.com/bennokr/makeprov#readme) and the published [API documentation](https://pypi.org/project/makeprov/) for decorator semantics, CLI commands, and configuration keys.
- [makeprov SHACL shapes](https://raw.githubusercontent.com/bennokr/makeprov/refs/heads/main/tests/prov_shapes.ttl) that validate the generated PROV graphs.
- [PROV-O primer](https://www.w3.org/TR/prov-o/) for the underlying ontology used in emitted `.trig`/`.json` bundles.
- SemSynth CLI module [`semsynth/cli.py`](./cli.py) where makeprov's `COMMANDS` are exposed alongside SemSynth-specific commands.

## Configuration and entry points
- **Default settings** live in [`prov-config.toml`](../prov-config.toml), which sets the base IRI (`https://w3id.org/semsynth/demo#`) and the provenance output directory. You can pass an alternate file to the `prov_conf` argument on any CLI command to override `base_iri`, `prov_dir`, `force`, or `dry_run` through makeprov's `GLOBAL_CONFIG`.
- **CLI integration**: `search` and `report` (both in [`semsynth/cli.py`](./cli.py)) are wrapped with `@rule()` decorators. When you invoke `python semsynth_reports_cli.py search ...` or `python semsynth_reports_cli.py report ...`, defopt exposes every makeprov command (e.g., `build`, `graph`) alongside SemSynth's commands so you can run `build <target>` to materialize missing outputs.
- **File parameters**: Inputs such as `prov_conf` use `InPath`, while outputs like `output` or `outdir` use `OutPath`. makeprov registers these as graph nodes, so running `build outputs/<dataset>/index.html` will trigger the dependent rules that generate that file.

## Dataset discovery and acquisition rules
- `specs_from_input` and `load_dataset` in [`semsynth/datasets.py`](./datasets.py) are annotated with `@rule(out_params=["openml_cache_dir", "uciml_cache_dir"])` so the cache roots under `downloads-cache/` are explicit outputs. Provider-specific helpers (`load_openml_by_name`, `load_uciml_by_id`, and their search counterparts in `semsynth/dataproviders/openml.py` and `semsynth/dataproviders/uciml.py`) are wrapped too, ensuring that API queries and downloaded CSVs populate the graph via `OutPath` markers.
- Because the cache directories are modeled as outputs, makeprov can decide whether to re-run downloads based on whether the files under `downloads-cache/openml/` or `downloads-cache/uciml/` already exist. This is especially useful when combining `build` with the `--force` toggle from the configuration file.

## Pipeline orchestration rules
- The reporting pipeline in [`semsynth/pipeline.py`](./pipeline.py) is rule-aware at every stage:
  - `process_dataset` drives the end-to-end build for a dataset and declares the dataset-level `outdir` as a provenance output.
  - `DatasetPreprocessor.preprocess` marks preprocessing artifacts such as missingness models, semmap exports, and UMAP PNGs as outputs, tying them to upstream cache downloads.
  - `BackendExecutor.run_models` (called per backend) and downstream writers like `MetricWriter.write_privacy` / `MetricWriter.write_downstream` register model run directories, metric CSVs, and JSON summaries as outputs so they participate in dependency resolution.
  - Report assembly (`ReportWriter.write_summary`, `ReportWriter.write_markdown`, `ReportWriter.write_html`) and manifest helpers (`discover_model_runs` in [`semsynth/models.py`](./models.py)) are likewise wrapped, meaning UMAP snapshots, manifest JSON, and HTML/Markdown reports show up as targets in the provenance DAG.
- These rule annotations mirror the tangible file system effects (PNG visualizations, metrics, manifests) so makeprov can construct a consistent dependency graph rather than treating orchestration functions as opaque side effects.

## Provenance outputs and validation
- Running any rule emits provenance metadata into the configured `prov_dir` (defaults in `prov-config.toml`). Each execution records inputs (`InPath`) and outputs (`OutPath`), plus returned Python objects where applicable, serialized into `.trig`/`.json` files aligned with [PROV-O](https://www.w3.org/TR/prov-o/).
- The SHACL shapes shipped with makeprov ([`makeprov/shacl`](https://github.com/bennokr/makeprov/tree/main/makeprov/shacl)) can be used to validate these graphs. Point a SHACL engine at the generated `.trig`/`.json` bundle to confirm class/property constraints before publishing provenance artifacts.
- When debugging, you can render the DAG with `python semsynth_reports_cli.py graph --conf @prov-config.toml` (exposed via makeprov `COMMANDS`) to visualize how cached downloads, preprocessing artifacts, model outputs, and reports depend on one another.

## Practical usage patterns
- **Rebuild a missing artifact**: If a report HTML is missing, run `python semsynth_reports_cli.py build outputs/<dataset>/index.html --conf @prov-config.toml` to regenerate upstream dependencies automatically.
- **Dry runs and forcing recomputation**: Pass `--conf @prov-config.toml --dry-run` to inspect the DAG without executing, or set `force=true` in the TOML to rerun every rule regardless of existing outputs.
- **Extending the workflow**: When adding new provider helpers or report writers, wrap them with `@rule()` and annotate file inputs/outputs with `InPath`/`OutPath` to keep provenance complete. Refer to the decorator examples in the [makeprov README](https://github.com/bennokr/makeprov#readme) and ensure new artifacts respect PROV-O semantics for activities and entities.
