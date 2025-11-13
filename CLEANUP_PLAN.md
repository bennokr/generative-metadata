# Plan to clean up, integrate and simplify the SemSynth codebase

## Summary

- Goal: make the package fast to import (no heavy global imports), clarify and enforce a single backend contract, fix/align the MetaSyn backend, make UMAP image generation optional, and add optional privacy and downstream-fidelity metrics to reports (config-driven).  
- Deliverables: incremental, testable changes grouped into small PRs: lazy imports, well-specified backend interface, MetaSyn backend fix, pipeline/config updates to support optional UMAP/metrics, report integration for privacy & downstream fidelity, tests and docs.  
- Approach: refactor iteratively. Each change is small, keeps backward compatibility where practical, and includes tests and CLI/config options.

## High level design decisions

- Lazy imports: move heavy imports (pandas, umap, synthcity, metasyn, openml, ucimlrepo, pybnesian, sklearn heavy pieces, torch) into the functions or backend modules that actually need them. Export only lightweight public API at import time.
- Single backend interface: define a clear, typed Backend protocol and require every backend to implement it. Use static tests to ensure conformance and runtime checks to fail early with descriptive errors.
- Model-run manifest: unify fields emitted in `manifest.json` and `metrics.json` across backends so reporting code can rely on a stable shape.
- Config-driven optional work: add explicit flags in YAML/CLI (or both) to compute UMAP images, privacy metrics, downstream fidelity, and to control heavy optional dependencies.
- Keep existing CLI compatibility but extend with sensible flags. Document breaking changes if necessary.

## Concrete tasks 
(ordered, with files to change, suggested code/behavior, and tests)

### 1) Make heavy imports lazy across the package

- Files to change: most modules that import heavy libs at top-level, e.g. semsynth/cli.py, semsynth/pipeline.py, semsynth/backends/*, semsynth/dataproviders/*, semsynth/umap_utils.py, semsynth/reporting.py, semsynth/downstream_fidelity.py, semsynth/privacy_metrics.py.
- Change:
  - Move imports like `umap`, `pandas`, `synthcity`, `metasyn.metaframe`, `pybnesian.hc`, `openml`, `ucimlrepo`, `torch`, `matplotlib` into the functions that actually use them.
  - Use importlib.import_module or try/except to raise helpful errors when optional dependency is missing (e.g., "synthcity is required for synthcity backend — install with pip install semsynth[synthcity]").
  - Keep very small, safe imports at module top-level (typing, small stdlib).
- Tests:
  - Unit test that doing `import semsynth` (or from semsynth import cli) doesn't import heavy libs and completes quickly (mock importlib to ensure heavy packages are not loaded).

### 2) Define and enforce a backend interface (Backend Protocol)

- Files to add/change: create `semsynth/backends/__init__.py` and `semsynth/backends/base.py` (new)
- New interface (example):
  - A Backend must implement:
    - `run_experiment(df: pd.DataFrame, *, provider: Optional[str], dataset_name: Optional[str], provider_id: Optional[int], outdir: str, label: str, model_info: Dict[str, Any] | None, rows: Optional[int], seed: int, test_size: float, semmap_export: Optional[Dict[str, Any]] = None) -> Path`
  - Must write:
    - run_dir / "synthetic.csv"
    - run_dir / "metrics.json" (with at least keys: backend, summary)
    - run_dir / "manifest.json" (with unified fields: backend, name, provider, dataset_name, provider_id, generator/type, params, seed, rows, test_size)
- Change pipeline to import the Backend protocol and assert backend modules conform (runtime duck-typing or via typing.Protocol + static tests).
- Files to change: semsynth/pipeline.py for checking backend spec when selecting backend modules.
- Tests:
  - Unit test that `semsynth/backends/pybnesian.py`, `synthcity.py`, and `metasyn.py` satisfy the protocol (simple call signature introspection or a small smoke run with micro-DF).

### 3) Fix and align MetaSyn backend implementation

- File to change: semsynth/backends/metasyn.py
- Issues to address:
  - Currently metasyn backend saves GMF and produces synth_df, but it uses `mf.save` and `mf.synthesize`. Ensure it follows Backend contract exactly (manifest & metrics shape), consistent `model_info` handling, seed usage, and not rely on external non-lazy imports.
  - Accept `model_info` keys similar to other backends (e.g., `type`/`generator`), or ignore if not applicable but preserve structure in manifest `params`.
  - Ensure sampling rows handling uses provided `rows` param correctly and not regenerate UMAP unnecessarily.
  - Ensure error handling: if MetaSyn is missing, raise clear error at runtime or skip if configured that way.
- Change:
  - Move `import metasyn` into function and guard with helpful message.
  - Ensure `manifest` keys match other backends and write manifest via `write_manifest`.
  - Return run_dir Path.
- Tests:
  - Unit test that `run_experiment` runs on a minimal dataframe (mock MetaFrame.fit_dataframe and its synthesize to return pandas DF) and produces synthetic.csv, manifest.json and metrics.json with expected keys.

### 4) Make UMAP generation optional and idempotent

- Files to change: semsynth/pipeline.py, semsynth/umap_utils.py, semsynth/reporting.py
- Changes:
  - Add option in `Config` (pipeline) and in CLI/config YAML to control UMAP generation: e.g., `generate_umap: bool = True`.
  - The pipeline should:
    - Only call build_umap/plot_umap if `cfg.generate_umap` is True.
    - When generating UMAPs for model runs, only do so if run.umap_png does not already exist OR if `overwrite_umap` flag is True.
    - Avoid rebuilding UMAP embedding if not needed. (Currently pipeline builds UMAP once — keep that, but guard plotting.)
  - `umap_utils.plot_umap` should return the limits, but plotting should be conditional.
- CLI: add `--no-umap` or `--generate-umap/--no-generate-umap`.
- Tests:
  - Unit test pipeline with cfg.generate_umap=False ensures no files produced and no heavy imports executed.
  - Test that if UMAP PNG exists and overwrite=False then pipeline does not re-generate.

### 5) Config-driven optional privacy & downstream-fidelity metrics

- Files to change: semsynth/models.py (configs loader), semsynth/pipeline.py, semsynth/reporting.py, semsynth/backends/* (small adaptions)
- Approach:
  - Extend model config YAML (or add a top-level report YAML) to include per-model or global flags:
    - compute_privacy: bool
    - compute_downstream: bool
    - privacy: { eps: 0.1, plugin: 'synthcity' }  (privacy module options)
    - downstream: { cv: 5, m: 20, burnin: 5, max_interactions: 5 }
  - Example YAML snippet to include in docs and schema:
    
    ```yaml
    configs:
      - name: pybn_clg
        backend: pybnesian
        model:
          type: clg
        seed: 42
        rows: 1000
        compute_privacy: true
        compute_downstream: true
    compute_privacy: true          # global fallback
    compute_downstream: false
    ```
  - Extend `load_model_configs` to keep top-level flags and propagate defaults to each ModelSpec (or preserve per-model override).
  - Pipeline changes:
    - After `backend.run_experiment` completes (or after discover_model_runs), for each model run, if `compute_privacy` is set for that run (model-level or global), load synthetic CSV and call `semsynth.privacy_metrics.summarize_privacy_synthcity(df_real, df_synth, meta_df, eps=...)`.
    - For downstream-fidelity, call `semsynth.downstream_fidelity.compare_real_vs_synth` (lazy import inside the code path). Save results:
      - write metrics_privacy.json, metrics_downstream.json or merge into run.metrics and update manifest to point to these files.
  - Guarantee non-blocking if optional dependencies are missing:
    - If synthcity is not installed but compute_privacy requested, log a clear warning or raise according to a `strict` flag.
- Tests:
  - Unit tests that compute_privacy or compute_downstream will be called when flags set. Mock heavy libs (synthcity, MICE, statsmodels) in tests to avoid installing heavy deps.

### 6) Standardize manifest and metrics format across backends

- Files to change: semsynth/models.py (discover_model_runs), all backends (they already mostly write manifest and metrics but with different fields).
- Unified manifest fields recommended:
  - backend (string)
  - name (string)
  - generator (string/None) or type
  - params (object)
  - seed (int)
  - rows (int)
  - provider, dataset_name, provider_id
  - artifacts: { synthetic_csv: 'synthetic.csv', per_variable_metrics: 'per_variable_metrics.csv', metrics_json: 'metrics.json', privacy_json: 'metrics.privacy.json', downstream_json: 'metrics.downstream.json', umap_png: 'umap.png' }
- Ensure discover_model_runs uses these fields to populate ModelRun reliably and does not break on missing fields.
- Tests: unit test discover_model_runs reading manifests written by each backend.

### 7) Centralize orchestration of compute-heavy steps
- Files to change: semsynth/pipeline.py and a new `semsynth/runner.py`
- Tasks:
  - Extract per-model run orchestration into a manager function/class that performs:
    - calling backend.run_experiment
    - optional post-processing steps: compute_privacy, compute_downstream, generate umap for synth
    - writing manifests and metrics
  - Ensure logging and error isolation so one failed model doesn't crash the entire pipeline (fail-fast configurable).
- Tests: integration test running 2 dummy model specs with mocked backends.

### 8) Improve dataset caching and offline modes

- Files to change: semsynth/dataproviders/openml.py, semsynth/dataproviders/uciml.py
- Tasks:
  - Ensure cache usage is robust, use consistent cache layout, and fail gracefully in offline mode. Provide CLI flag `--offline` that forces cache-only behavior.
  - Make cache dir configurable via environment variable or config param.
- Tests: unit tests for load functions with cache hits & misses (mock file system or use tmpdir).

### 9) Tests and CI

- Add test coverage for: lazy import behavior, backend protocol conformance, pipeline (with fake backends), metasyn/pybnesian/synthcity run_experiment smoke tests (mocking external libs), report generation producing expected files, discover_model_runs.
- Use pytest with fixtures and monkeypatch to simulate missing optional dependencies and to mock heavy libs.
- CI:
  - Minimal matrix: python 3.9/3.11, run lint, unit tests; run additional matrix job with extras (pybnesian, synthcity) maybe on-demand.

## 10) Documentation, examples and migration notes

- Files to change/add: README.md, docs/config.md, docs/backends.md, examples/ (example_config.yaml, small demo dataset).
- Document:
  - New config schema, CLI flags (--no-umap, --compute-privacy, --compute-downstream, --offline, --parallel-jobs), and optional extras to install (e.g., `pip install semsynth[pybnesian,synthcity,metasyn]`).
  - How to add a new backend and required contract.
  - Performance tips (caching, do not generate UMAP if not needed).

## 11) Release & migration plan

- Break the work into multiple small releases:
  - 1: Lazy imports + backend protocol + tests
  - 2: Fix metasyn + UMAP optional
  - 3: Privacy & downstream opt-in metrics + reporting changes
  - 4: Orchestration, parallelism, caching improvements
  - Final: Docs, CI, packaging with extras
- For each release, create a CHANGELOG entry documenting any backward-incompatible changes (e.g., manifest keys). Provide a small migration helper if manifest format changes.

## Suggested code / config examples

- Config YAML schema example (global + per-model overrides)

```yaml
# report_config.yaml
compute_privacy: true        # global default for privacy metrics
compute_downstream: false   # default for downstream fidelity
generate_umap: false        # global UMAP generation default
parallel_jobs: 1

configs:
  - name: model_ctgan
    backend: synthcity
    model:
      type: ctgan
      epochs: 10
    rows: 1000
    seed: 42
    compute_privacy: true
    compute_downstream: true

  - name: model_bn
    backend: pybnesian
    model:
      type: clg
      score: bic
    seed: 101
    rows: 500
```

- CLI examples
  - Keep defopt CLI but add flags (or allow top-level config file)
    - `semsynth report openml --outdir docs --configs_yaml report_config.yaml --no-umap`
    - `semsynth report openml --outdir docs --configs_yaml report_config.yaml --compute-privacy --compute-downstream`

## Implementation notes and pitfalls

- Avoid adding heavy dependencies to the package base install. Provide extras: `[pybnesian]`, `[synthcity]`, `[metasyn]`.
- When adding optional privacy/downstream metrics, ensure to catch exceptions from those heavy libs and log helpful messages rather than crash the whole pipeline. Add a `strict` flag to control behavior.
- Unit testing heavy libs: mock their APIs rather than installing them in CI.
- Keep function-level docstrings and typing hints for discoverability and static checking.

## Minimum first PR (small, high impact)

- Implement lazy imports in `semsynth/cli.py` and `semsynth/__init__.py` (expose minimal public API only).
- Add a basic Backend Protocol file (`semsynth/backends/base.py`) and update pipeline selection to import backends only when needed.
- Add tests that importing top-level package doesn't import heavy deps.

## Risk assessment

- Breaking backwards-compatibility of manifests or metrics: mitigated by keeping current keys where possible and adding a mapping layer in discover_model_runs to handle older manifests.
- Tests failing due to mocking complexity: address by keeping mocks simple and covering happy-path and failure path.
- Time estimate depends on familiarity with external libs (pybnesian, metasyn, synthcity) — mocking reduces friction.

## Appendix: concrete small change examples (sketches)


- Lazy import pattern

```python
# bad (current)
import umap
import pandas as pd

# better
def build_umap(...):
    import umap
    import pandas as pd
    ...
```

- Backend protocol (semsynth/backends/base.py)

```python
from typing import Protocol, Optional, Dict, Any
from pathlib import Path
import pandas as pd

class Backend(Protocol):
    def run_experiment(
        self,
        df: pd.DataFrame,
        *,
        provider: Optional[str],
        dataset_name: Optional[str],
        provider_id: Optional[int],
        outdir: str,
        label: str,
        model_info: Dict[str, Any] | None,
        rows: Optional[int],
        seed: int,
        test_size: float,
        semmap_export: Optional[Dict[str, Any]] = None,
    ) -> Path: ...
```

- Pipeline checks before using backend

```python
from .backends.base import Backend
# at runtime when selecting backend module:
if not hasattr(backend_module, "run_experiment"):
    raise RuntimeError(f"Backend {spec.backend} missing run_experiment")
# optionally a runtime signature check (inspect.signature)
```

- Post-run optional privacy/downstream compute in pipeline (sketch)

```python
# after run_dir exists and synthetic CSV written
if spec.compute_privacy or global_compute_privacy:
    try:
        from .privacy_metrics import summarize_privacy_synthcity
        synth_df = pd.read_csv(run_dir / "synthetic.csv")
        privacy_summary = summarize_privacy_synthcity(df_real, synth_df, meta_df, eps=... )
        (run_dir / "metrics.privacy.json").write_text(json.dumps(dataclasses.asdict(privacy_summary)))
    except ImportError:
        logging.warning("synthcity not installed; privacy metrics skipped")
```