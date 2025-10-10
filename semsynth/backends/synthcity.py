from __future__ import annotations

import json
import logging
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..metrics import per_variable_distances, summarize_distance_metrics
from ..utils import (
    coerce_continuous_to_float,
    coerce_discrete_to_category,
    ensure_dir,
    infer_types,
    rename_categorical_categories_to_str,
)
from ..models import model_run_root, write_manifest


def canonical_generator_name(name: str) -> str:
    """Return the canonical synthcity plugin name for a user alias."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Generator name must be non-empty")
    aliases = {
        "ctgan": "ctgan",
        "ads-gan": "adsgan",
        "adsgan": "adsgan",
        "pategan": "pategan",
        "dp-gan": "dpgan",
        "dpgan": "dpgan",
        "tvae": "tvae",
        "rtvae": "rtvae",
        "nflow": "nflow",
        "tabularflow": "tabularflow",
        "bn": "bayesiannetwork",
        "bayesiannetwork": "bayesiannetwork",
        "privbayes": "privbayes",
        "arf": "arf",
        "arfpy": "arf",
        "great": "great",
    }
    if key not in aliases:
        raise ValueError(f"Unknown generator alias: {name}")
    return aliases[key]


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


def _ensure_torch_rmsnorm() -> None:
    try:
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
                else:
                    normalized_shape = tuple(normalized_shape)
                self.normalized_shape = normalized_shape
                self.eps = eps
                self.elementwise_affine = elementwise_affine
                if elementwise_affine:
                    self.weight = nn.Parameter(torch.ones(*self.normalized_shape))
                else:
                    self.register_parameter("weight", None)

            def forward(self, input: "torch.Tensor") -> "torch.Tensor":
                dims = tuple(range(-len(self.normalized_shape), 0))
                rms = torch.rsqrt(input.pow(2).mean(dim=dims, keepdim=True) + self.eps)
                output = input * rms
                if self.elementwise_affine:
                    output = output * self.weight
                return output

        nn.RMSNorm = _RMSNorm  # type: ignore[attr-defined]
    except Exception:
        pass


def _get_plugin(name: str, params: Dict[str, Any]):
    _ensure_torch_rmsnorm()
    from synthcity.plugins import Plugins

    logging.info("Loading synthcity plugin: %s", name)
    return Plugins().get(name, **(params or {}))


def _normalize_plugin_params(
    plugin_name: str, params: Dict[str, Any]
) -> Dict[str, Any]:
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


def run_experiment(
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
) -> Path:
    logging.info("Starting synthcity run: %s", label)
    working = df.copy()
    disc_cols, cont_cols = infer_types(working)
    working = coerce_discrete_to_category(working, disc_cols)
    working = rename_categorical_categories_to_str(working, disc_cols)
    working = coerce_continuous_to_float(working, cont_cols)
    train_df, test_df = train_test_split(
        working, test_size=test_size, random_state=seed, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    model_type = model_info.pop("type")
    plugin_name = canonical_generator_name(model_type)
    plugin_params = _normalize_plugin_params(plugin_name, model_info)
    plugin = _get_plugin(plugin_name, plugin_params)
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

    run_root = model_run_root(Path(outdir))
    run_dir = run_root / label
    ensure_dir(str(run_dir))
    synth_df.to_csv(run_dir / "synthetic.csv", index=False)
    logging.info("Wrote synthetic CSV: %s", run_dir / "synthetic.csv")
    if semmap_export:
        try:
            import copy
            import semsynth.semmap as semmap  # noqa: F401

            sdf = synth_df.copy()
            sdf.semmap.apply_json_metadata(
                copy.deepcopy(semmap_export), convert_pint=False
            )
            sdf.semmap.to_parquet(
                str(run_dir / "synthetic.semmap.parquet"), index=False
            )
        except Exception as e:
            logging.warning("SemMap parquet failed: %s", e)

    # Per-variable distances and summary
    dist_df = per_variable_distances(test_df, synth_df, disc_cols, cont_cols)
    dist_df.to_csv(run_dir / "per_variable_metrics.csv", index=False)
    metrics = {
        "backend": "synthcity",
        "summary": summarize_distance_metrics(dist_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "synth_rows": len(synth_df),
        "discrete_cols": len(disc_cols),
        "continuous_cols": len(cont_cols),
        "umap_png": None,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    manifest = {
        "backend": "synthcity",
        "name": label,
        "provider": provider,
        "dataset_name": dataset_name,
        "provider_id": provider_id,
        "generator": plugin_name,
        "params": plugin_params,
        "seed": seed,
        "rows": n_rows,
        "test_size": test_size,
    }
    write_manifest(run_dir, manifest)
    logging.info("Finished synthcity run: %s", label)
    return run_dir
