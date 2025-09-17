# Synthcity Plan
Below is a concrete plan to add **synthcity** tabular generators to this project. 
The plan adds a CLI surface and a config-driven workflow, with minimal code changes to existing modules.

---

# Goal

* Allow users to choose a **synthcity** generator on the command line **or** in a config file.
* Train the chosen generator on a pandas DataFrame returned by `bncli.datasets.load_dataset`.
* Generate synthetic rows, save artifacts, and run the existing reporting/metrics/UMAP utilities.

---

# Repo recap (from the zip)

* `bncli/cli.py` — defopt-based CLI with commands like `search`, `report` (truncated in zip but shows pattern).
* `bncli/datasets.py` — load datasets from OpenML / UCI and parse user input.
* `bncli/pipeline.py` — central orchestration, coercion helpers, JSON-LD + optional SemMap export, train/test split.
* `bncli/metrics.py`, `bncli/umap_utils.py`, `bncli/reporting.py` — quality metrics and visualization.
* `bncli/mappings.py` — mapping helpers (can host generator aliases).
* `bncli/bn.py` — Bayesian network learning/printing (pybnesian).

We’ll bolt synthcity support onto this structure with a small, isolated adaptor.

---

# New dependencies

Add to `requirements.txt`:

```text
synthcity>=0.2            # or the version you standardize on
torch                     # synthcity runtime; pin as needed for your environment
```

Optional speedups (cpu/gpu) follow your environment; do not force CUDA.

---

# High-level design

1. **Adaptor module**: `bncli/synth.py`

   * Maps human-friendly generator names to **synthcity** plugins.
   * Builds and fits a plugin from name + params.
   * Generates synthetic rows.
   * Saves artifacts (fitted plugin, params, manifest).

2. **CLI**: new `synth` subcommand and extensions to `report`

   * `bncli synth ...` generates data from a single dataset + generator.
   * `bncli report ...` gains optional `--generators` and/or `--configs-yaml` to run multiple generators and produce reports.

3. **Config**: YAML schema that describes generators and their parameters.

   * Supports a single generator or a list.
   * Supports per-generator overrides (rows, seed).

4. **Pipeline integration**

   * Reuse existing coercion (`coerce_*`) so dtypes are stable.
   * Pass the *post*-coercion training frame into synthcity.
   * Reuse `umap_utils` and `metrics` to compare real vs synthetic.
   * Emit JSON/CSV/plots in a structured outdir.

---

# CLI: new subcommand

Add a `synth` function to `bncli/cli.py`. It mirrors the existing style (defopt) and delegates to the adaptor.

```python
# bncli/cli.py (additions)

from .synth import run_synth_experiment
from .datasets import load_dataset, specs_from_input
from .utils import ensure_dir

def synth(
    dataset: str,
    *,
    provider: str = "openml",
    generator: str = "ctgan",
    gen_params_json: str = "",      # '{"lr": 1e-4, "epochs": 300}'
    rows: int | None = None,        # defaults to len(train)
    seed: int = 0,
    outdir: str = "outputs",
    test_size: float = 0.25,
    configs_yaml: str = "",         # optional: run many generators from YAML
) -> None:
    """
    Train a synthcity generator and emit synthetic data + reports.

    dataset: name/id understood by datasets.py; use --provider to pick openml/uciml.
    """
    ensure_dir(outdir)

    # 1) Load dataset
    spec = specs_from_input([dataset], provider=provider)[0]
    df, meta = load_dataset(spec)

    # 2) Single-generator path
    if not configs_yaml:
        run_synth_experiment(
            df=df,
            dataset_name=f"{spec.provider}:{spec.name}",
            outdir=outdir,
            generator=generator,
            gen_params_json=gen_params_json,
            rows=rows,
            seed=seed,
            test_size=test_size,
        )
        return

    # 3) Multi-generator path from YAML
    from .synth import run_from_yaml
    run_from_yaml(
        df=df,
        dataset_name=f"{spec.provider}:{spec.name}",
        outdir=outdir,
        yaml_path=configs_yaml,
        seed=seed,
        test_size=test_size,
    )
```

Register it in `main` (alongside existing `search`, `report`):

```python
def main(argv: Optional[List[str]] = None) -> None:
    defopt.run([search, report, synth], argv=argv)
```

---

# Config file schema (YAML)

Support both single and multiple generators. Example:

```yaml
# configs/synth_generators.yaml

generators:
  - name: ctgan
    params:
      epochs: 300
      batch_size: 512
    rows: 50000
    seed: 42

  - name: tvae
    params:
      epochs: 200
    rows: 50000

  - name: privbayes
    params:
      epsilon: 1.0
    rows: 50000

  - name: great       # LLM-based (GReaT)
    params:
      model: "gpt2"   # example; align with your local policy/models
      max_len: 256
    rows: 20000
```

---

# Name/alias mapping

Add canonical aliases to `bncli/mappings.py`:

```python
# bncli/mappings.py (additions)

SYNTHCITY_ALIASES = {
    # GAN
    "ctgan": "ctgan",
    "ads-gan": "adsgan",
    "adsgan": "adsgan",
    "pategan": "pategan",
    "dp-gan": "dpgan",
    "dpgan": "dpgan",

    # VAE
    "tvae": "tvae",
    "rtvae": "rtvae",

    # Flows
    "nflow": "nflow",           # or "tabularflow" if that is the plugin id you standardize on

    # Bayesian networks
    "bn": "bayesiannetwork",
    "bayesiannetwork": "bayesiannetwork",
    "privbayes": "privbayes",

    # Random forest
    "arf": "arf",
    "arfpy": "arf",

    # LLM-based
    "great": "great",
}

def canonical_generator_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in SYNTHCITY_ALIASES:
        raise ValueError(f"Unknown generator: {name}")
    return SYNTHCITY_ALIASES[key]
```

(Adjust plugin ids if your installed synthcity version uses different strings. Keep this mapping as the single source of truth.)

---

# Adaptor module: `bncli/synth.py`

```python
# bncli/synth.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .mappings import canonical_generator_name
from .utils import ensure_dir, coerce_continuous_to_float, coerce_discrete_to_category
from .umap_utils import build_umap_artifacts, save_umap_plot
from .metrics import (
    # reuse your existing metrics; include any you already expose
    # plug in more as needed (e.g., column-wise PSI/KS, pairwise corr delta, clf-based utility)
)
from .reporting import (
    # reuse whatever reporting helpers exist (tables/figures writing)
)

# synthcity import only where used to avoid hard dependency during non-synth runs
def _get_plugin(name: str, params: Dict[str, Any]):
    from synthcity.plugins import Plugins
    plugin = Plugins().get(name, **(params or {}))
    return plugin

@dataclass
class SynthRunArtifacts:
    plugin_name: str
    plugin_params: Dict[str, Any]
    model_obj: Any
    real_train: pd.DataFrame
    real_test: pd.DataFrame
    synth_df: pd.DataFrame
    seed: int
    rows: int

def fit_and_generate(
    df: pd.DataFrame,
    generator: str,
    gen_params: Dict[str, Any],
    rows: Optional[int],
    seed: int,
    test_size: float,
) -> SynthRunArtifacts:

    # 1) Train/test split and dtype coercion (reuse project utilities)
    df = df.copy()
    df = coerce_continuous_to_float(df)
    df = coerce_discrete_to_category(df)

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)

    # 2) Build and fit
    plugin_name = canonical_generator_name(generator)
    plugin = _get_plugin(plugin_name, gen_params or {})

    # Set seeds if plugin exposes it; also set numpy/torch for reproducibility
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass

    plugin.fit(train_df)

    # 3) Generate
    n = int(rows) if rows else len(train_df)
    synth = plugin.generate(count=n).dataframe() if hasattr(plugin.generate(count=1), "dataframe") \
        else plugin.generate(count=n)  # some versions return DataFrame directly

    return SynthRunArtifacts(
        plugin_name=plugin_name,
        plugin_params=gen_params or {},
        model_obj=plugin,
        real_train=train_df,
        real_test=test_df,
        synth_df=synth,
        seed=seed,
        rows=n,
    )

def _save_artifacts(a: SynthRunArtifacts, dataset_name: str, outdir: str | Path) -> Path:
    out = Path(outdir) / "synth" / dataset_name.replace("/", "_").replace(":", "_") / a.plugin_name
    ensure_dir(out)

    # Synthetic data
    synth_csv = out / "synthetic.csv"
    a.synth_df.to_csv(synth_csv, index=False)

    # Params and manifest
    manifest = {
        "dataset": dataset_name,
        "generator": a.plugin_name,
        "params": a.plugin_params,
        "seed": a.seed,
        "rows": a.rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Save model if supported
    try:
        # synthcity plugins usually offer .save(path) or are picklable
        model_dir = out / "model"
        ensure_dir(model_dir)
        try:
            a.model_obj.save(model_dir)  # type: ignore[attr-defined]
        except Exception:
            import pickle
            with (out / "model.pkl").open("wb") as fw:
                pickle.dump(a.model_obj, fw)
    except Exception as e:
        (out / "save_warning.txt").write_text(str(e), encoding="utf-8")

    return out

def _evaluate_and_visualize(a: SynthRunArtifacts, outdir: Path) -> None:
    # Example: basic UMAP overlay and a few utility metrics.
    # Plug into bncli.metrics/reporting as available.

    # UMAP
    umap_art = build_umap_artifacts(
        real_df=a.real_train,
        synth_df=a.synth_df,
        sample_n=min(10000, len(a.real_train), len(a.synth_df)),
        random_state=a.seed,
    )
    save_umap_plot(umap_art, outdir / "umap_train_vs_synth.png", title="Train vs Synthetic")

    # Placeholder metric calls (replace with your project’s metric functions)
    # metrics = {
    #   "per_column_ks": compute_ks(a.real_test, a.synth_df),
    #   "pairwise_corr_delta": compute_corr_delta(a.real_test, a.synth_df),
    #   "clf_auc_gap": downstream_auc_gap(a.real_train, a.real_test, a.synth_df),
    # }
    # (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

def run_synth_experiment(
    df: pd.DataFrame,
    dataset_name: str,
    outdir: str,
    generator: str,
    gen_params_json: str = "",
    rows: Optional[int] = None,
    seed: int = 0,
    test_size: float = 0.25,
) -> Path:
    params = json.loads(gen_params_json) if gen_params_json else {}
    artifacts = fit_and_generate(
        df=df,
        generator=generator,
        gen_params=params,
        rows=rows,
        seed=seed,
        test_size=test_size,
    )
    run_dir = _save_artifacts(artifacts, dataset_name, outdir)
    _evaluate_and_visualize(artifacts, run_dir)
    return run_dir

def run_from_yaml(
    df: pd.DataFrame,
    dataset_name: str,
    outdir: str,
    yaml_path: str,
    seed: int,
    test_size: float,
) -> None:
    import yaml
    spec = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    gens = spec.get("generators", [])
    for g in gens:
        name = g["name"]
        params = g.get("params", {})
        rows = g.get("rows")
        this_seed = g.get("seed", seed)
        run_synth_experiment(
            df=df,
            dataset_name=dataset_name,
            outdir=outdir,
            generator=name,
            gen_params_json=json.dumps(params),
            rows=rows,
            seed=this_seed,
            test_size=test_size,
        )
```

---

# Reporting

Use the existing reporting module. Minimal additions:

* Write a small reader that scans `outputs/synth/<dataset>/<generator>/` to gather:

  * `manifest.json`
  * `metrics.json` (once you wire metrics)
  * Point to `synthetic.csv` and plot assets
* If `report` already assembles a doc, add a section “Synthetic data (synthcity)” and include:

  * A table of key metrics per generator.
  * Thumbnails of UMAP plots.
  * Links to artifacts.

Example hook:

```python
# bncli/reporting.py (conceptual)
def add_synth_section(doc, synth_root: Path):
    # enumerate runs and append to the report object you already use
    pass
```

---

# Metrics (incremental)

Start with what you already have in `bncli/metrics.py`. If nothing exists yet for synthetic-vs-real, add a few simple, fast checks:

* Numeric columns: KS statistic and PSI.
* Categorical columns: Jensen–Shannon divergence on distribution.
* Pairwise correlations: ΔSpearman on numeric-only subset.
* Simple downstream utility: choose a target column if present; train a baseline model on real/train and synthetic/train and compare AUC/accuracy on real/test.

Keep the column-selection logic robust to missing targets.

```python
# bncli/metrics.py (sketch)
def per_column_ks(real: pd.DataFrame, synth: pd.DataFrame) -> dict: ...
def per_column_jsd(real: pd.DataFrame, synth: pd.DataFrame) -> dict: ...
def corr_delta(real: pd.DataFrame, synth: pd.DataFrame) -> dict: ...
def downstream_auc_gap(real_train, real_test, synth_train) -> float: ...
```

Save into `metrics.json` per run. The adaptor already has a placeholder to write this file.

---

# Config-first workflow

* If the user passes `--configs-yaml`, the CLI iterates `generators[*]`.
* Each generator writes into `outputs/synth/<provider_name>/<generator>/`.
* The `report` command can detect that directory and include a comparison table.

---

# Artifact layout

```text
outputs/
  synth/
    openml_adult/
      ctgan/
        synthetic.csv
        manifest.json
        model/ or model.pkl
        umap_train_vs_synth.png
        metrics.json
      tvae/
        ...
```

---

# Error handling

* Unknown generator → raise `ValueError` from `canonical_generator_name`.
* Bad params → surface synthcity exception with a clear message and write `save_warning.txt` if model saving fails.
* Non-fittable columns → rely on the project’s `coerce_*` helpers; if a column still fails, log and drop with a warning (optional feature flag `--strict` to fail fast).

---

# Tests

Add lightweight tests (use a tiny synthetic dataset):

```python
def test_ctgan_roundtrip(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"x": [0,1,1,0]*50, "y": [1.2, 3.4, 2.2, 0.1]*50})
    out = run_synth_experiment(
        df=df, dataset_name="toy", outdir=tmp_path, generator="ctgan", gen_params_json="{}", rows=50, seed=0
    )
    assert (out / "synthetic.csv").exists()
    assert pd.read_csv(out / "synthetic.csv").shape[0] == 50
```

---

# Minimal changes to existing files

1. **`bncli/cli.py`**: add `synth` function and register it in `main`.
2. **`bncli/mappings.py`**: add alias map + `canonical_generator_name`.
3. **`bncli/synth.py`**: new module (adaptor).
4. **`bncli/reporting.py`**: optional, add a synth section if you generate a consolidated report.
5. **`bncli/metrics.py`**: add real-vs-synth metrics if not present.

---

# Examples

**Single generator from CLI**

```bash
python -m bncli.cli synth 1590 --provider openml --generator ctgan \
  --gen_params_json '{"epochs": 200, "batch_size": 256}' \
  --rows 50000 --seed 42 --outdir outputs
```

**Multiple generators from YAML**

```bash
python -m bncli.cli synth 1590 --provider openml \
  --configs_yaml configs/synth_generators.yaml \
  --outdir outputs --seed 7
```

**Integrate into `report`** (optional extension)

```bash
python -m bncli.cli report --datasets 1590 --provider openml \
  --configs_yaml configs/synth_generators.yaml \
  --outdir docs
```

---

# Notes on datatypes

* Keep your existing dtype coercion (`coerce_continuous_to_float`, `coerce_discrete_to_category`) before calling synthcity. This respects ordinal/categorical intent and avoids bespoke preprocessing at the call site.
* synthcity encodes mixed types internally; pass DataFrame as-is after coercion.

---

# Work plan (sequence)

1. Add `bncli/mappings.py` aliases and create `bncli/synth.py`.
2. Wire `bncli/cli.py` `synth` command with defopt.
3. Implement `_evaluate_and_visualize` with your metrics and UMAP.
4. Add YAML loader + multi-generator loop.
5. Extend reporting (optional).
6. Add tests and sample YAML.
7. Document CLI flags in README and add a “supported generators” section that mirrors your alias map.

This keeps the integration small, testable, and consistent with the current repo shape.
