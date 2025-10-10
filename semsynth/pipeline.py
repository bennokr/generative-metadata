from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, List, Dict
import logging

import pandas as pd
from . import semmap  # noqa: F401  # Register pandas accessors

from .utils import (
    coerce_continuous_to_float,
    coerce_discrete_to_category,
    infer_types,
    rename_categorical_categories_to_str,
    seed_all,
    ensure_dir,
)
from .umap_utils import build_umap, plot_umap, transform_with_umap
from .reporting import write_report_md
from .mappings import resolve_mapping_json, load_mapping_json
from .models import load_model_configs, ModelSpec, discover_model_runs
from .backends import metasyn as backend_meta, pybnesian as backend_pyb, synthcity as backend_syn
from .metadata import (
    get_uciml_variable_descriptions,
)


class Config:
    random_state = 42
    max_umap_sample = 1000
    fit_on_sample = 1000
    synthetic_sample = 1000
    test_size = 0.2
    umap_n_neighbors = 30
    umap_min_dist = 0.1
    umap_n_components = 2


def process_dataset(
    dataset_spec: Any,
    df: pd.DataFrame,
    color_series: Optional[pd.Series],
    base_outdir: str,
    *,
    model_configs: Optional[List[ModelSpec]] = None,
    cfg: Config = Config(),
) -> None:
    logging.info(f"Processing dataset: {dataset_spec.name}")

    outdir = Path(base_outdir) / dataset_spec.name.replace("/", "_")
    ensure_dir(str(outdir))

    # ---- Declared information ---- #
    semmap_export: Optional[Dict[str, Any]] = None
    mapping_path = resolve_mapping_json(dataset_spec)
    if mapping_path is not None:
        logging.info(f"Applying curated SemMap metadata from {mapping_path}")
        try:
            curated_mapping = load_mapping_json(mapping_path)
            df.semmap.apply_json_metadata(curated_mapping, convert_pint=True)
            semmap_export = df.semmap.jsonld()
        except Exception:
            logging.exception("Failed to apply SemMap metadata", exc_info=True)
            semmap_export = None
    else:
        logging.debug(
            "No curated SemMap metadata for provider=%s id=%s dataset=%s",
            dataset_spec.provider,
            dataset_spec.id,
            dataset_spec.name,
        )

    # ---- Inferred information ---- #
    logging.info("Inferring column types (discrete vs continuous)")
    disc_cols, cont_cols = infer_types(df)
    logging.info(
        f"Detected columns — discrete: {len(disc_cols)}, continuous: {len(cont_cols)}"
    )
    df = coerce_discrete_to_category(df, disc_cols)
    df = rename_categorical_categories_to_str(df, disc_cols)
    df = coerce_continuous_to_float(df, cont_cols)

    if len(disc_cols) == 0:
        cand = [
            c
            for c in df.columns
            if c not in disc_cols and pd.api.types.is_integer_dtype(df[c])
        ]
        cand = sorted(cand, key=lambda c: df[c].nunique(dropna=True))[:3]
        if cand:
            df[cand] = df[cand].astype("category")
            disc_cols, cont_cols = infer_types(df)
            df = rename_categorical_categories_to_str(df, disc_cols)
            df = coerce_continuous_to_float(df, cont_cols)

    # TODO Build DCAT JSON-LD, enriched with SemMap & DSV when available

    # Build declared and inferred type maps for reporting
    inferred_map = {
        c: ("discrete" if c in disc_cols else "continuous") for c in df.columns
    }

    logging.info("Dropping rows with any NA for modeling")
    df_no_na = df.dropna(axis=0, how="any").reset_index(drop=True)
    if len(df_no_na) == 0:
        logging.warning(
            "All rows removed by dropna(); imputing missing values as fallback"
        )
        df_imp = df.copy()
        for c in cont_cols:
            if c in df_imp.columns:
                try:
                    med = float(pd.to_numeric(df_imp[c], errors="coerce").median())
                except Exception:
                    med = 0.0
                df_imp[c] = (
                    pd.to_numeric(df_imp[c], errors="coerce").fillna(med).astype(float)
                )
        for c in disc_cols:
            if c in df_imp.columns:
                try:
                    mode = df_imp[c].mode(dropna=True)
                    fill = (
                        mode.iloc[0]
                        if len(mode)
                        else (
                            df_imp[c].cat.categories[0]
                            if pd.api.types.is_categorical_dtype(df_imp[c])
                            and len(df_imp[c].cat.categories)
                            else ""
                        )
                    )
                except Exception:
                    fill = ""
                if pd.api.types.is_categorical_dtype(df_imp[c]):
                    if (
                        isinstance(fill, str) and fill in list(df_imp[c].cat.categories)
                    ) or (
                        not isinstance(fill, str)
                        and fill in list(df_imp[c].cat.categories)
                    ):
                        df_imp[c] = df_imp[c].fillna(fill)
                    else:
                        df_imp[c] = df_imp[c].cat.add_categories([fill]).fillna(fill)
                else:
                    df_imp[c] = df_imp[c].fillna(fill)
        df_no_na = df_imp.reset_index(drop=True)

    # Load model configs. If none provided, use default_config.yaml
    if model_configs is None:
        model_configs = load_model_configs(None)

    rng = seed_all(cfg.random_state)
    color_series2 = None
    if isinstance(color_series, pd.Series) and color_series.name in df_no_na.columns:
        color_series2 = df_no_na[color_series.name]
    logging.info("Fitting UMAP on real data sample")
    umap_art = build_umap(
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
    umap_png_real = outdir / "umap_real.png"
    umap_lims = plot_umap(
        umap_art.embedding,
        str(umap_png_real),
        title=f"{dataset_spec.name}: real (sample)",
        color_labels=umap_art.color_labels,
    )

    # Run models through backends
    df_fit_sample = df_no_na
    if cfg.fit_on_sample and cfg.fit_on_sample < len(df_fit_sample):
        df_fit_sample = df_no_na.sample(
            cfg.fit_on_sample, random_state=cfg.random_state
        )
    
    for idx, spec in enumerate(model_configs):
        label = spec.name or f"model_{idx + 1}"
        seed = int(spec.seed if spec.seed is not None else cfg.random_state)

        if spec.backend == "pybnesian":
            backend_module = backend_pyb
        elif spec.backend == "synthcity":
            backend_module = backend_syn
        elif spec.backend == "metasyn":
            backend_module = backend_meta
        else:
            logging.warning(
                "Unknown backend %s for model %s; skipping", spec.backend, label
            )
            continue

        try:
            backend_module.run_experiment(
                df=df_fit_sample,
                provider=dataset_spec.provider,
                dataset_name=dataset_spec.name,
                provider_id=dataset_spec.id,
                outdir=str(outdir),
                label=label,
                model_info=dict(spec.model or {}),
                rows=min(cfg.synthetic_sample, len(df)),
                seed=seed,
                test_size=cfg.test_size,
                semmap_export=semmap_export,
            )
        except Exception:
            logging.exception("%s run failed for %s", spec.backend, label)

    # If UCI, get variable descriptions from cached metadata for report table
    var_desc_map = {}
    if dataset_spec.provider == "uciml" and isinstance(dataset_spec.id, int):
        try:
            var_desc_map = get_uciml_variable_descriptions(dataset_spec.id)
        except Exception:
            var_desc_map = {}

    # Generate UMAPs for discovered model runs using the same projection
    model_runs = []
    try:
        model_runs = discover_model_runs(outdir)
        for run in model_runs:
            try:
                s_df = pd.read_csv(run.synthetic_csv)
                if len(s_df) > cfg.max_umap_sample:
                    s_df = s_df.sample(cfg.max_umap_sample)
                s_df = s_df.reindex(columns=df_no_na.columns)
                s_no_na = s_df.dropna(axis=0, how="any")
                s_emb = transform_with_umap(umap_art, s_no_na)
                run.umap_png = run.run_dir / "umap.png"
                plot_umap(
                    s_emb,
                    str(run.umap_png),
                    title=f"{dataset_spec.name}: synthetic ({run.name})",
                    lims=umap_lims,
                )
            except Exception:
                logging.exception(
                    "Failed to generate UMAP for model run %s", run.run_dir
                )
    except Exception:
        logging.exception("Failed to discover model runs for %s", dataset_spec.name)
        model_runs = []

    write_report_md(
        outdir=outdir,
        dataset_name=dataset_spec.name,
        # metadata_file=metadata_file,
        # dataset_jsonld=dataset_jsonld,
        dataset_provider=dataset_spec.provider,
        dataset_provider_id=dataset_spec.id,
        df=df_no_na,
        disc_cols=disc_cols,
        cont_cols=cont_cols,
        umap_png_real=str(umap_png_real),
        # declared_types=declared_map or None,
        inferred_types=inferred_map or None,
        variable_descriptions=var_desc_map or None,
        semmap_jsonld=semmap_export,
        model_runs=model_runs,
    )
