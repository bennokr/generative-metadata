from __future__ import annotations

import json
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

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

    return Plugins().get(name, **(params or {}))


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

    working = df.copy()
    disc_cols, cont_cols = infer_types(working)
    working = coerce_discrete_to_category(working, disc_cols)
    working = rename_categorical_categories_to_str(working, disc_cols)
    working = coerce_continuous_to_float(working, cont_cols)

    train_df, test_df = train_test_split(working, test_size=test_size, random_state=seed, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    plugin_name = canonical_generator_name(generator)
    plugin_params = _normalize_plugin_params(plugin_name, gen_params or {})
    plugin = _get_plugin(plugin_name, plugin_params)

    _seed_random_generators(seed)
    plugin.fit(train_df)

    n_rows = int(rows) if rows else len(train_df)
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
            return
        except Exception:
            pass
        import pickle

        with (run_dir / "model.pkl").open("wb") as fw:
            pickle.dump(artifacts.model_obj, fw)
    except Exception as exc:  # pragma: no cover - best effort logging
        (run_dir / "save_warning.txt").write_text(str(exc), encoding="utf-8")


def _plot_train_vs_synth_umap(artifacts: SynthRunArtifacts, outdir: Path) -> Optional[Path]:
    if len(artifacts.real_train) == 0 or len(artifacts.synth_df) == 0:
        return None
    rng = np.random.default_rng(artifacts.seed)
    sample_n = int(min(10000, len(artifacts.real_train), len(artifacts.synth_df)))
    if sample_n <= 1:
        return None
    umap_art = build_umap(
        artifacts.real_train,
        discrete_cols=artifacts.discrete_cols,
        continuous_cols=artifacts.continuous_cols,
        color_series=None,
        rng=rng,
        random_state=artifacts.seed,
        max_sample=sample_n,
    )
    real_emb = umap_art.embedding
    synth_sample = artifacts.synth_df.sample(
        sample_n,
        random_state=artifacts.seed,
        replace=len(artifacts.synth_df) < sample_n,
    )
    synth_sample = synth_sample.reindex(columns=artifacts.real_train.columns).reset_index(drop=True)
    synth_emb = transform_with_umap(umap_art, synth_sample)
    png_path = outdir / "umap_train_vs_synth.png"
    plot_umap(
        np.vstack([real_emb, synth_emb]),
        str(png_path),
        title="Train (blue) vs Synthetic (orange)",
        color_labels=np.concatenate(
            [np.zeros(len(real_emb)), np.ones(len(synth_emb))]
        ),
    )
    return png_path


def _evaluate_and_visualize(artifacts: SynthRunArtifacts, run_dir: Path) -> Dict[str, Any]:
    dist_df = per_variable_distances(
        artifacts.real_test,
        artifacts.synth_df.reindex(columns=artifacts.real_test.columns),
        artifacts.discrete_cols,
        artifacts.continuous_cols,
    )
    dist_path = run_dir / "per_variable_metrics.csv"
    dist_df.to_csv(dist_path, index=False)

    summary = summarize_distance_metrics(dist_df)
    umap_png = _plot_train_vs_synth_umap(artifacts, run_dir)

    metrics = {
        "summary": summary,
        "train_rows": len(artifacts.real_train),
        "test_rows": len(artifacts.real_test),
        "synth_rows": len(artifacts.synth_df),
        "discrete_cols": len(artifacts.discrete_cols),
        "continuous_cols": len(artifacts.continuous_cols),
        "umap_png": umap_png.name if umap_png else None,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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

    dataset_dir = _canonical_dataset_dir(outdir, provider, dataset_name, provider_id)
    run_dir = dataset_dir / artifacts.plugin_name
    ensure_dir(str(run_dir))

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
            )
        )
    return outputs


def discover_synth_runs(
    base_outdir: str | Path,
    *,
    provider: str,
    dataset_name: str,
) -> List[SynthReportRun]:
    root = Path(base_outdir) / "synth"
    if not root.exists():
        return []

    provider_norm = str(provider or "").strip().lower()
    dataset_norm = str(dataset_name)
    runs: List[SynthReportRun] = []
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
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
            umap_png = run_dir / "umap_train_vs_synth.png"
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
    return runs
