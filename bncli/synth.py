from __future__ import annotations

import json
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
import logging

from .mappings import canonical_generator_name
from .metrics import per_variable_distances, summarize_distance_metrics
from .umap_utils import build_umap, plot_umap, transform_with_umap
from .utils import (
    coerce_continuous_to_float,
    coerce_discrete_to_category,
    ensure_dir,
    infer_types,
    rename_categorical_categories_to_str,
)
from .models import model_run_root
from .bn import learn_bn


@dataclass
class SynthRunArtifacts:
    plugin_name: str
    plugin_params: Dict[str, Any]
    model_obj: Any
    real_train: pd.DataFrame
    real_test: pd.DataFrame
    synth_df: pd.DataFrame
    discrete_cols: List[str]
    continuous_cols: List[str]
    seed: int
    rows: int
    test_size: float


@dataclass
class SynthReportRun:
    generator: str
    manifest: Dict[str, Any]
    metrics: Dict[str, Any]
    synthetic_csv: Path
    per_variable_csv: Optional[Path]
    umap_png: Optional[Path]
    run_dir: Path


def _ensure_torch_rmsnorm() -> None:
    """Backfill torch.nn.RMSNorm for older torch releases."""

    try:  # pragma: no cover - torch optional at runtime
        import torch
        from torch import nn

        if hasattr(nn, "RMSNorm"):
            return

        class _RMSNorm(nn.Module):
            def __init__(
                self,
                normalized_shape: int | Tuple[int, ...],
                eps: float = 1e-6,
                elementwise_affine: bool = True,
            ) -> None:
                super().__init__()
                if isinstance(normalized_shape, numbers.Integral):
                    normalized_shape = (int(normalized_shape),)
                elif isinstance(normalized_shape, torch.Size):  # pragma: no cover - defensive
                    normalized_shape = tuple(normalized_shape)
                else:
                    normalized_shape = tuple(normalized_shape)

                if not normalized_shape:
                    raise ValueError("normalized_shape must be non-empty")

                self.normalized_shape = normalized_shape
                self.eps = eps
                self.elementwise_affine = elementwise_affine
                if elementwise_affine:
                    self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
                else:
                    self.register_parameter("weight", None)

            def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                dims = tuple(range(-len(self.normalized_shape), 0))
                rms = torch.rsqrt(input.pow(2).mean(dim=dims, keepdim=True) + self.eps)
                output = input * rms
                if self.elementwise_affine:
                    output = output * self.weight
                return output

            def extra_repr(self) -> str:  # pragma: no cover - debug helper
                return (
                    f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
                    f"elementwise_affine={self.elementwise_affine}"
                )

        nn.RMSNorm = _RMSNorm  # type: ignore[attr-defined]
    except Exception:
        pass


def _seed_random_generators(seed: int) -> None:
    np.random.seed(seed)
    try:  # pragma: no cover - torch optional at runtime
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # pragma: no cover - requires cuda
    except Exception:
        pass


def _get_plugin(name: str, params: Dict[str, Any]):
    _ensure_torch_rmsnorm()
    from synthcity.plugins import Plugins
    logging.info("Loading synthcity plugin: %s", name)
    plugin = Plugins().get(name, **(params or {}))
    return plugin


def _normalize_plugin_params(plugin_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {k: v for k, v in (params or {}).items() if v is not None}
    if plugin_name == "ctgan" and "epochs" in normalized and "n_iter" not in normalized:
        normalized["n_iter"] = normalized.pop("epochs")
    return normalized


def _ensure_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "dataframe"):
        df = obj.dataframe()
        if isinstance(df, pd.DataFrame):
            return df
    if hasattr(obj, "to_pandas"):
        df = obj.to_pandas()
        if isinstance(df, pd.DataFrame):
            return df
    if isinstance(obj, np.ndarray):
        return pd.DataFrame(obj)
    return pd.DataFrame(obj)


def _canonical_dataset_dir(outdir: str | Path, provider: str, dataset_name: str, provider_id: Optional[int]) -> Path:
    # Legacy location kept for backward compatibility; new runs use dataset report/models
    root = Path(outdir) / "synth"
    ensure_dir(str(root))
    provider_clean = str(provider or "dataset").strip().lower().replace(" ", "_")
    name_clean = str(dataset_name or "dataset").strip()
    name_clean = name_clean.replace("/", "_").replace(":", "_").replace(" ", "_")
    parts = [provider_clean]
    if provider_id is not None:
        parts.append(str(provider_id))
    if name_clean:
        parts.append(name_clean)
    dataset_dir = root / "_".join(parts)
    ensure_dir(str(dataset_dir))
    return dataset_dir


def fit_and_generate(
    df: pd.DataFrame,
    generator: str,
    gen_params: Dict[str, Any],
    rows: Optional[int],
    seed: int,
    test_size: float,
) -> SynthRunArtifacts:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    logging.info("Starting synthcity fit/generate: generator=%s rows=%s test_size=%s", generator, rows, test_size)
    working = df.copy()
    disc_cols, cont_cols = infer_types(working)
    working = coerce_discrete_to_category(working, disc_cols)
    working = rename_categorical_categories_to_str(working, disc_cols)
    working = coerce_continuous_to_float(working, cont_cols)
    logging.debug("Detected columns: discrete=%d continuous=%d", len(disc_cols), len(cont_cols))

    train_df, test_df = train_test_split(working, test_size=test_size, random_state=seed, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    plugin_name = canonical_generator_name(generator)
    plugin_params = _normalize_plugin_params(plugin_name, gen_params or {})
    plugin = _get_plugin(plugin_name, plugin_params)

    _seed_random_generators(seed)
    logging.info("Fitting plugin %s", plugin_name)
    plugin.fit(train_df)

    n_rows = int(rows) if rows else len(train_df)
    logging.info("Generating synthetic count=%d", n_rows)
    generated = plugin.generate(count=n_rows)
    synth_df = _ensure_dataframe(generated)
    synth_df = synth_df.reindex(columns=train_df.columns)
    for col in disc_cols:
        if col in synth_df.columns:
            synth_df[col] = synth_df[col].astype("category")
    synth_df = rename_categorical_categories_to_str(synth_df, disc_cols)
    synth_df = coerce_continuous_to_float(synth_df, cont_cols)

    return SynthRunArtifacts(
        plugin_name=plugin_name,
        plugin_params=plugin_params,
        model_obj=plugin,
        real_train=train_df,
        real_test=test_df,
        synth_df=synth_df,
        discrete_cols=disc_cols,
        continuous_cols=cont_cols,
        seed=seed,
        rows=n_rows,
        test_size=float(test_size),
    )


def _save_model(artifacts: SynthRunArtifacts, run_dir: Path) -> None:
    try:
        model_dir = run_dir / "model"
        ensure_dir(str(model_dir))
        try:
            artifacts.model_obj.save(model_dir)  # type: ignore[attr-defined]
            logging.debug("Saved synthcity model to %s", model_dir)
            return
        except Exception:
            pass
        import pickle

        with (run_dir / "model.pkl").open("wb") as fw:
            pickle.dump(artifacts.model_obj, fw)
        logging.debug("Saved synthcity model pickle to %s", run_dir / "model.pkl")
    except Exception as exc:  # pragma: no cover - best effort logging
        (run_dir / "save_warning.txt").write_text(str(exc), encoding="utf-8")
        logging.warning("Model save failed: %s", exc)


# Custom UMAP generation removed; the report pipeline will create UMAPs for
# synthcity runs using the same projection as real/BN/MetaSyn.


def _evaluate_and_visualize(artifacts: SynthRunArtifacts, run_dir: Path) -> Dict[str, Any]:
    dist_df = per_variable_distances(
        artifacts.real_test,
        artifacts.synth_df.reindex(columns=artifacts.real_test.columns),
        artifacts.discrete_cols,
        artifacts.continuous_cols,
    )
    dist_path = run_dir / "per_variable_metrics.csv"
    dist_df.to_csv(dist_path, index=False)
    logging.debug("Wrote per-variable metrics to %s", dist_path)

    summary = summarize_distance_metrics(dist_df)

    metrics = {
        "summary": summary,
        "train_rows": len(artifacts.real_train),
        "test_rows": len(artifacts.real_test),
        "synth_rows": len(artifacts.synth_df),
        "discrete_cols": len(artifacts.discrete_cols),
        "continuous_cols": len(artifacts.continuous_cols),
        # UMAP is generated by the report pipeline for consistent projections
        "umap_png": None,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logging.debug("Wrote metrics to %s", metrics_path)
    return metrics


def _write_manifest(
    artifacts: SynthRunArtifacts,
    *,
    run_dir: Path,
    provider: str,
    dataset_name: str,
    provider_id: Optional[int],
    requested_generator: str,
) -> None:
    manifest = {
        "provider": provider,
        "dataset_name": dataset_name,
        "provider_id": provider_id,
        "generator": artifacts.plugin_name,
        "requested_generator": requested_generator,
        "backend": "synthcity",
        "params": artifacts.plugin_params,
        "seed": artifacts.seed,
        "rows": artifacts.rows,
        "test_size": artifacts.test_size,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _save_synthetic_dataframe(artifacts: SynthRunArtifacts, run_dir: Path) -> None:
    synth_path = run_dir / "synthetic.csv"
    artifacts.synth_df.to_csv(synth_path, index=False)


def run_synth_experiment(
    df: pd.DataFrame,
    *,
    provider: str,
    dataset_name: str,
    provider_id: Optional[int],
    outdir: str,
    generator: str,
    gen_params_json: str = "",
    rows: Optional[int] = None,
    seed: int = 0,
    test_size: float = 0.25,
    run_root: str | Path | None = None,
    dir_name: Optional[str] = None,
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

    if run_root is None:
        run_root = model_run_root(Path(outdir))
    run_dir = Path(run_root) / (dir_name or artifacts.plugin_name)
    ensure_dir(str(run_dir))
    logging.info("Writing synthcity artifacts to %s", run_dir)

    _save_synthetic_dataframe(artifacts, run_dir)
    _write_manifest(
        artifacts,
        run_dir=run_dir,
        provider=provider,
        dataset_name=dataset_name,
        provider_id=provider_id,
        requested_generator=generator,
    )
    _save_model(artifacts, run_dir)
    _evaluate_and_visualize(artifacts, run_dir)
    return run_dir


def run_from_yaml(
    df: pd.DataFrame,
    *,
    provider: str,
    dataset_name: str,
    provider_id: Optional[int],
    outdir: str,
    yaml_path: str,
    seed: int,
    test_size: float,
) -> List[Path]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to load generator configs") from exc

    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "generators" in data:
        data = data["generators"]
    if not isinstance(data, list):
        raise ValueError("YAML must define a list of generators under 'generators'")

    outputs: List[Path] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Generator config at index {idx} must be a mapping")
        name = item.get("name")
        if not name:
            raise ValueError(f"Generator config at index {idx} missing 'name'")
        params = item.get("params", {})
        rows = item.get("rows")
        gen_seed = int(item.get("seed", seed))
        outputs.append(
            run_synth_experiment(
                df=df,
                provider=provider,
                dataset_name=dataset_name,
                provider_id=provider_id,
                outdir=outdir,
                generator=str(name),
                gen_params_json=json.dumps(params),
                rows=rows,
                seed=gen_seed,
                test_size=test_size,
                run_root=model_run_root(Path(outdir)),
                dir_name=str(item.get("name") or name),
            )
        )
    return outputs


def discover_synth_runs(
    base_outdir: str | Path,
    *,
    provider: str,
    dataset_name: str,
) -> List[SynthReportRun]:
    # Look under the dataset's models directory only (no legacy fallback)
    base = Path(base_outdir)
    root = base / "models"
    if not root.exists():
        logging.info("No synthcity runs found: %s does not exist", root)
        return []

    provider_norm = str(provider or "").strip().lower()
    dataset_norm = str(dataset_name)
    runs: List[SynthReportRun] = []
    # If scanning legacy tree, there is an extra level of dataset directories
    candidate_dirs: List[Path] = []
    candidate_dirs = [p for p in root.iterdir() if p.is_dir()]

    for run_dir in sorted(candidate_dirs):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(manifest.get("provider", "")).strip().lower() != provider_norm:
            continue
        if manifest.get("dataset_name") != dataset_norm:
            continue
        metrics_path = run_dir / "metrics.json"
        metrics: Dict[str, Any] = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                metrics = {}
        per_var_path = run_dir / "per_variable_metrics.csv"
        umap_png = run_dir / "umap.png"
        runs.append(
            SynthReportRun(
                generator=str(manifest.get("generator", run_dir.name)),
                manifest=manifest,
                metrics=metrics,
                synthetic_csv=run_dir / "synthetic.csv",
                per_variable_csv=per_var_path if per_var_path.exists() else None,
                umap_png=umap_png if umap_png.exists() else None,
                run_dir=run_dir,
            )
        )
    runs.sort(key=lambda r: (r.generator, r.run_dir.name))
    logging.info("Discovered %d synthcity runs under %s", len(runs), root)
    return runs


def run_pybnesian_experiment(
    df: pd.DataFrame,
    *,
    provider: str,
    dataset_name: str,
    provider_id: Optional[int],
    outdir: str,
    bn_type: str,
    bn_params_json: str = "",
    rows: Optional[int] = None,
    seed: int = 0,
    test_size: float = 0.25,
    dir_name: Optional[str] = None,
) -> Path:
    """Fit/generate/evaluate a single PyBNesian model and write artifacts under models/.

    This is a light-weight path used by the synth CLI for backend=pybnesian.
    UMAP images are produced by the report pipeline for consistent projections.
    """
    params = json.loads(bn_params_json) if bn_params_json else {}
    logging.info("Starting PyBNesian synth: type=%s rows=%s test_size=%s", bn_type, rows, test_size)
    working = df.copy()
    disc_cols, cont_cols = infer_types(working)
    working = coerce_discrete_to_category(working, disc_cols)
    working = rename_categorical_categories_to_str(working, disc_cols)
    working = coerce_continuous_to_float(working, cont_cols)

    from sklearn.model_selection import train_test_split as _tts
    train_df, test_df = _tts(working, test_size=test_size, random_state=seed, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Minimal arc blacklist: age, sex, race if present
    default_root = ["age", "sex", "race"]
    cols = list(working.columns)
    col_map = {str(c).lower(): c for c in cols}
    sens_in_cols = [col_map[s] for s in [x.lower() for x in default_root] if s in col_map]
    arc_blacklist_pairs: List[Tuple[str, str]] = []
    for u in sens_in_cols:
        for v in cols:
            arc_blacklist_pairs.append((v, u))

    score = params.get("score")
    operators = params.get("operators")
    max_indegree = params.get("max_indegree")
    logging.info("Learning BN: type=%s score=%s max_indegree=%s seed=%s", bn_type, score, max_indegree, seed)
    bn_art = learn_bn(
        train_df,
        bn_type=str(bn_type),
        random_state=seed,
        arc_blacklist=arc_blacklist_pairs,
        score=score,
        operators=operators,
        max_indegree=max_indegree,
    )
    model = bn_art.model
    n_rows = int(rows) if rows else len(train_df)
    logging.info("Sampling BN synthetic count=%d", n_rows)
    synth = model.sample(n_rows, seed=seed)
    synth_df = synth.to_pandas().reindex(columns=working.columns)
    for c in disc_cols:
        if c in synth_df.columns:
            synth_df[c] = synth_df[c].astype("category")

    # Save under models/<label>
    label = dir_name or f"pybnesian_{bn_type}"
    run_root = model_run_root(Path(outdir))
    run_dir = run_root / label
    ensure_dir(str(run_dir))
    synth_df.to_csv(run_dir / "synthetic.csv", index=False)
    dist_df = per_variable_distances(test_df, synth_df, disc_cols, cont_cols)
    dist_df.to_csv(run_dir / "per_variable_metrics.csv", index=False)
    metrics = {
        "backend": "pybnesian",
        "summary": summarize_distance_metrics(dist_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "synth_rows": len(synth_df),
        "discrete_cols": len(disc_cols),
        "continuous_cols": len(cont_cols),
        "umap_png": None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logging.info("Wrote PyBNesian artifacts to %s", run_dir)
    return run_dir
