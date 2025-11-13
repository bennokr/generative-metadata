"""Pipeline orchestration for dataset processing and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast

from .backends.base import BackendModule, ensure_backend_contract
from .mappings import load_mapping_json, resolve_mapping_json
from .metadata import get_uciml_variable_descriptions
from .models import ModelConfigBundle, discover_model_runs, load_model_configs

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    from .specs import DatasetSpec


_BACKEND_MODULE_PATHS = {
    "pybnesian": "semsynth.backends.pybnesian",
    "synthcity": "semsynth.backends.synthcity",
    "metasyn": "semsynth.backends.metasyn",
}

_BACKEND_CACHE: Dict[str, BackendModule] = {}


@dataclass
class PipelineConfig:
    """Configuration values controlling the reporting pipeline."""

    random_state: int = 42
    max_umap_sample: int = 1000
    fit_on_sample: Optional[int] = 1000
    synthetic_sample: int = 1000
    test_size: float = 0.2
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    generate_umap: bool = False
    compute_privacy: bool = False
    compute_downstream: bool = False
    overwrite_umap: bool = False


def _load_backend_module(name: str) -> BackendModule:
    module_path = _BACKEND_MODULE_PATHS.get(name)
    if not module_path:
        raise ValueError(f"Unknown backend '{name}'")
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - optional deps
        message = f"Failed to import backend '{name}'. "
        message += f"Install with `pip install semsynth[{name}]`."
        raise RuntimeError(message) from exc
    ensure_backend_contract(module)
    typed_module = cast(BackendModule, module)
    _BACKEND_CACHE[name] = typed_module
    return typed_module


def _resolve_flag(
    default_value: bool,
    bundle_value: Optional[bool],
    spec_value: Optional[bool],
) -> bool:
    if spec_value is not None:
        return spec_value
    if bundle_value is not None:
        return bundle_value
    return default_value


def _import_utils():  # pragma: no cover - helper for lazy import
    from . import utils as _utils

    return _utils


def _import_umap_utils():  # pragma: no cover - helper for lazy import
    from . import umap_utils as _umap

    return _umap


def _import_reporting():  # pragma: no cover - helper for lazy import
    from . import reporting as _reporting

    return _reporting


def _build_privacy_metadata(df: "pd.DataFrame", inferred: Dict[str, str]) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for column, kind in inferred.items():
        dtype = "numeric" if kind == "continuous" else "categorical"
        rows.append({"variable": column, "role": "qi", "type": dtype})
    return pd.DataFrame(rows)


def _build_downstream_meta(
    df: "pd.DataFrame",
    inferred: Dict[str, str],
    target: Optional[str],
) -> Dict[str, Any]:
    import pandas as pd

    meta: Dict[str, Any] = {
        "dataset": {"target_name": target, "target_type": None},
        "variables": {},
    }
    for column in df.columns:
        role = "target" if target and column == target else "predictor"
        kind = inferred.get(column, "continuous")
        if kind == "continuous":
            stat_type = "numeric"
            categories: List[str] = []
        else:
            series = df[column].astype("category")
            categories = [str(cat) for cat in series.cat.categories]
            stat_type = "binary" if len(categories) <= 2 else "nominal"
        meta["variables"][column] = {
            "role": role,
            "stat_type": stat_type,
            "categories": categories,
            "reference": categories[0] if categories else None,
            "interaction_ok": True,
            "missing_codes": [],
        }
    if target:
        target_type = meta["variables"][target]["stat_type"]
        if target_type == "nominal" and len(meta["variables"][target]["categories"]) > 2:
            meta["dataset"]["target_type"] = "multiclass"
        else:
            meta["dataset"]["target_type"] = target_type
    else:
        meta["dataset"]["target_type"] = "continuous"
    return meta


def _write_privacy_metrics(
    run_dir: Path,
    real_df: "pd.DataFrame",
    inferred: Dict[str, str],
) -> Dict[str, Any]:
    from .privacy_metrics import summarize_privacy_synthcity

    import pandas as pd

    synth_df = pd.read_csv(run_dir / "synthetic.csv").convert_dtypes()
    metadata = _build_privacy_metadata(real_df, inferred)
    summary = summarize_privacy_synthcity(real_df, synth_df, metadata)
    payload = asdict(summary)
    (run_dir / "metrics.privacy.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _write_downstream_metrics(
    run_dir: Path,
    real_df: "pd.DataFrame",
    synth_df: "pd.DataFrame",
    inferred: Dict[str, str],
    target: Optional[str],
) -> Dict[str, Any]:
    from .downstream_fidelity import compare_real_vs_synth

    meta = _build_downstream_meta(real_df, inferred, target)
    results = compare_real_vs_synth(real_df, synth_df, meta)
    compare = results.get("compare")
    sign_match_rate = float("nan")
    if hasattr(compare, "__getitem__"):
        try:
            series = compare["sign_match"]  # type: ignore[index]
            sign_match_rate = float(series.astype(float).mean())
        except Exception:
            sign_match_rate = float("nan")
    payload = {"formula": results.get("formula"), "sign_match_rate": sign_match_rate}
    (run_dir / "metrics.downstream.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def process_dataset(
    dataset_spec: "DatasetSpec",
    df: "pd.DataFrame",
    color_series: Optional["pd.Series"],
    base_outdir: str,
    *,
    model_bundle: Optional[ModelConfigBundle] = None,
    pipeline_config: Optional[PipelineConfig] = None,
) -> None:
    """Process a dataset end-to-end and generate a report."""

    from . import semmap  # noqa: F401  # register pandas accessor

    utils = _import_utils()
    reporting = _import_reporting()
    import pandas as pd

    bundle = model_bundle or load_model_configs(None)
    cfg = pipeline_config or PipelineConfig()

    outdir = Path(base_outdir) / dataset_spec.name.replace("/", "_")
    utils.ensure_dir(str(outdir))

    semmap_export: Optional[Dict[str, Any]] = None
    mapping_path = resolve_mapping_json(dataset_spec)
    if mapping_path is not None:
        logging.info("Applying curated SemMap metadata from %s", mapping_path)
        try:
            curated = load_mapping_json(mapping_path)
            df.semmap.from_jsonld(curated, convert_pint=True)
            semmap_export = df.semmap.to_jsonld()
        except Exception:
            logging.exception("Failed to apply SemMap metadata", exc_info=True)

    disc_cols, cont_cols = utils.infer_types(df)
    df = utils.coerce_discrete_to_category(df, disc_cols)
    df = utils.rename_categorical_categories_to_str(df, disc_cols)
    df = utils.coerce_continuous_to_float(df, cont_cols)

    inferred_map = {c: ("discrete" if c in disc_cols else "continuous") for c in df.columns}

    df_no_na = df.dropna(axis=0, how="any").reset_index(drop=True)
    if df_no_na.empty:
        df_no_na = df.fillna(method="ffill").fillna(method="bfill").reset_index(drop=True)

    bundle_specs = bundle.specs if bundle.specs else []

    rng = utils.seed_all(cfg.random_state)
    color_series2 = None
    if isinstance(color_series, pd.Series) and color_series.name in df_no_na.columns:
        color_series2 = df_no_na[color_series.name]

    generate_umap_flag = _resolve_flag(
        cfg.generate_umap,
        bundle.generate_umap,
        None,
    )

    umap_utils = _import_umap_utils() if generate_umap_flag else None

    umap_art = None
    umap_png_real = outdir / "umap_real.png"
    umap_lims = None
    if generate_umap_flag and umap_utils is not None:
        logging.info("Fitting UMAP on real data sample")
        umap_art = umap_utils.build_umap(
            df_no_na,
            disc_cols,
            cont_cols,
            color_series=color_series2,
            rng=rng,
            random_state=cfg.random_state,
            max_sample=cfg.max_umap_sample,
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            n_components=cfg.umap_n_components,
        )
        
        umap_lims = umap_utils.plot_umap(
            umap_art.embedding,
            str(umap_png_real),
            title=f"{dataset_spec.name}: real (sample)",
            color_labels=umap_art.color_labels,
        )
    elif not umap_png_real.exists():
        umap_png_real = None

    df_fit_sample = df_no_na
    if cfg.fit_on_sample and cfg.fit_on_sample < len(df_fit_sample):
        df_fit_sample = df_no_na.sample(cfg.fit_on_sample, random_state=cfg.random_state)

    for idx, spec in enumerate(bundle_specs):
        label = spec.name or f"model_{idx + 1}"
        seed = int(spec.seed if spec.seed is not None else cfg.random_state)
        backend_name = spec.backend
        try:
            backend_module = _load_backend_module(backend_name)
        except Exception:
            logging.exception("Failed to load backend %s", backend_name)
            continue

        rows = spec.rows if spec.rows is not None else cfg.synthetic_sample
        try:
            run_dir = backend_module.run_experiment(
                df=df_fit_sample,
                provider=dataset_spec.provider,
                dataset_name=dataset_spec.name,
                provider_id=dataset_spec.id,
                outdir=str(outdir),
                label=label,
                model_info=dict(spec.model or {}),
                rows=min(rows, len(df)),
                seed=seed,
                test_size=cfg.test_size,
                semmap_export=semmap_export,
            )
        except Exception:
            logging.exception("%s run failed for %s", backend_name, label)
            continue

        run_dir_path = Path(run_dir)
        synth_df = pd.read_csv(run_dir_path / "synthetic.csv").convert_dtypes()

        compute_privacy_flag = _resolve_flag(
            cfg.compute_privacy,
            bundle.compute_privacy,
            spec.compute_privacy,
        )
        compute_downstream_flag = _resolve_flag(
            cfg.compute_downstream,
            bundle.compute_downstream,
            spec.compute_downstream,
        )

        if compute_privacy_flag:
            try:
                _write_privacy_metrics(run_dir_path, df_no_na, inferred_map)
                logging.info("Wrote privacy metrics for %s", label)
            except ImportError:
                logging.warning("synthcity not installed; skipping privacy metrics for %s", label)
            except Exception:
                logging.exception("Failed to compute privacy metrics for %s", label)

        if compute_downstream_flag and dataset_spec.target:
            try:
                _write_downstream_metrics(
                    run_dir_path,
                    df_no_na,
                    synth_df,
                    inferred_map,
                    dataset_spec.target,
                )
                logging.info("Wrote downstream metrics for %s", label)
            except ImportError:
                logging.warning(
                    "statsmodels/sklearn not installed; skipping downstream metrics for %s",
                    label,
                )
            except Exception:
                logging.exception("Failed to compute downstream metrics for %s", label)

    var_desc_map: Dict[str, Any] = {}
    if dataset_spec.provider == "uciml" and isinstance(dataset_spec.id, int):
        try:
            var_desc_map = get_uciml_variable_descriptions(dataset_spec.id)
        except Exception:
            var_desc_map = {}

    model_runs = []
    try:
        model_runs = discover_model_runs(outdir)
        if generate_umap_flag and umap_utils is not None and umap_art is not None:
            for run in model_runs:
                try:
                    if run.umap_png and run.umap_png.exists() and not cfg.overwrite_umap:
                        continue
                    s_df = pd.read_csv(run.synthetic_csv).convert_dtypes()
                    if len(s_df) > cfg.max_umap_sample:
                        s_df = s_df.sample(cfg.max_umap_sample)
                    s_df = s_df.reindex(columns=df_no_na.columns)
                    s_emb = umap_utils.transform_with_umap(umap_art, s_df.dropna(axis=0, how="any"))
                    run.umap_png = run.run_dir / "umap.png"
                    umap_utils.plot_umap(
                        s_emb,
                        str(run.umap_png),
                        title=f"{dataset_spec.name}: synthetic ({run.name})",
                        lims=umap_lims,
                    )
                except Exception:
                    logging.exception("Failed to generate UMAP for %s", run.run_dir)
    except Exception:
        logging.exception("Failed to discover model runs for %s", dataset_spec.name)
        model_runs = []

    reporting.write_report_md(
        outdir=str(outdir),
        dataset_name=dataset_spec.name,
        dataset_provider=dataset_spec.provider,
        dataset_provider_id=dataset_spec.id,
        df=df_no_na,
        disc_cols=disc_cols,
        cont_cols=cont_cols,
        umap_png_real=str(umap_png_real) if umap_png_real else None,
        inferred_types=inferred_map or None,
        variable_descriptions=var_desc_map or None,
        semmap_jsonld=semmap_export,
        model_runs=model_runs,
    )

