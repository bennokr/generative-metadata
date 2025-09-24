from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Any, List, Dict, Sequence, Tuple
import json
import logging

import numpy as np
import pandas as pd
import semmap  # noqa: F401  # Register pandas accessors
from sklearn.model_selection import train_test_split

from .utils import (
    coerce_continuous_to_float,
    coerce_discrete_to_category,
    infer_types,
    rename_categorical_categories_to_str,
    seed_all,
    ensure_dir,
)
from .umap_utils import build_umap, plot_umap, transform_with_umap
from .metrics import per_variable_distances
from .reporting import write_report_md
from .mappings import resolve_mapping_json, load_mapping_json
from .models import load_model_configs, model_run_dir, model_run_root, ModelSpec, discover_model_runs
from .backends import pybnesian as backend_pyb, synthcity as backend_syn
from .metadata import (
    meta_to_dict,
    build_dataset_jsonld,
    select_declared_types,
    resolve_provider_and_id,
    get_uciml_variable_descriptions,
)
from metasyn.metaframe import MetaFrame


class Config:
    random_state = 42
    max_umap_sample = 1000
    synthetic_sample = 1000
    test_size = 0.2
    umap_n_neighbors = 30
    umap_min_dist = 0.1
    umap_n_components = 2



def process_dataset(
    meta: Any,
    df: pd.DataFrame,
    color_series: Optional[pd.Series],
    base_outdir: str,
    *,
    provider: Optional[str] = None,
    provider_id: Optional[int] = None,
    model_configs: Optional[List[ModelSpec]] = None,
    roots: Optional[List[str]] = None,
    run_metasyn: bool = True,
    cfg: Config = Config(),
) -> None:
    name = getattr(meta, 'name', None) or (meta.get('name') if isinstance(meta, dict) else None) or 'dataset'
    logging.info(f"Processing dataset: {name}")

    outdir = Path(base_outdir) / name.replace("/", "_")
    ensure_dir(str(outdir))

    metadata_file = outdir / "metadata.json"
    meta_dict = meta_to_dict(meta)
    with metadata_file.open('w', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2)
    logging.info(f"Wrote raw metadata JSON: {metadata_file}")

    # Attempt to locate curated SemMap metadata before further processing
    mapping_provider, mapping_provider_id = resolve_provider_and_id(
        provider=provider,
        provider_id=provider_id,
        meta_dict=meta_dict,
        dataset_jsonld=None,
    )
    semmap_export: Optional[Dict[str, Any]] = None
    semmap_dataset_meta: Optional[Dict[str, Any]] = None
    mapping_path = resolve_mapping_json(mapping_provider, mapping_provider_id, name)
    if mapping_path is not None:
        logging.info(f"Applying curated SemMap metadata from {mapping_path}")
        try:
            curated_mapping = load_mapping_json(mapping_path)
            df.semmap.apply_json_metadata(curated_mapping, convert_pint=True)
            semmap_export = df.semmap.export_json_metadata()
            if isinstance(semmap_export, dict):
                semmap_dataset_meta = semmap_export.get("dataset")
        except Exception:
            logging.exception("Failed to apply SemMap metadata", exc_info=True)
            semmap_export = None
            semmap_dataset_meta = None
    else:
        logging.debug(
            "No curated SemMap metadata for provider=%s id=%s dataset=%s",
            mapping_provider,
            mapping_provider_id,
            name,
        )

    # Build schema.org/Dataset JSON-LD, enriched with SemMap metadata when available
    dataset_jsonld = build_dataset_jsonld(
        name,
        meta_dict,
        df,
        semmap_dataset=semmap_dataset_meta,
    )
    dataset_jsonld_file = outdir / 'dataset.json'
    with dataset_jsonld_file.open('w', encoding='utf-8') as fw:
        json.dump(dataset_jsonld, fw, indent=2)
    logging.info(f"Wrote JSON-LD metadata: {dataset_jsonld_file}")

    if semmap_export:
        semmap_json_file = outdir / "dataset.semmap.json"
        with semmap_json_file.open('w', encoding='utf-8') as sf:
            json.dump(semmap_export, sf, indent=2)
        logging.info(f"Wrote SemMap metadata JSON: {semmap_json_file}")

    provider_name_final, provider_id_final = resolve_provider_and_id(
        provider=provider,
        provider_id=provider_id,
        meta_dict=meta_dict,
        dataset_jsonld=dataset_jsonld,
    )

    logging.info("Inferring column types (discrete vs continuous)")
    disc_cols, cont_cols = infer_types(df)
    logging.info(f"Detected columns — discrete: {len(disc_cols)}, continuous: {len(cont_cols)}")
    df = coerce_discrete_to_category(df, disc_cols)
    df = rename_categorical_categories_to_str(df, disc_cols)
    df = coerce_continuous_to_float(df, cont_cols)

    if len(disc_cols) == 0:
        cand = [c for c in df.columns if c not in disc_cols and pd.api.types.is_integer_dtype(df[c])]
        cand = sorted(cand, key=lambda c: df[c].nunique(dropna=True))[:3]
        if cand:
            df[cand] = df[cand].astype("category")
            disc_cols, cont_cols = infer_types(df)
            df = rename_categorical_categories_to_str(df, disc_cols)
            df = coerce_continuous_to_float(df, cont_cols)

    # Baseline summary using pandas describe (transposed so variables are rows)
    logging.info("Computing baseline summary via pandas describe")
    baseline_df = df.describe(include='all').transpose()
    baseline_df.index.name = 'variable'

    # Build declared and inferred type maps for reporting
    inferred_map = {c: ("discrete" if c in disc_cols else "continuous") for c in df.columns}

    declared_map = select_declared_types(
        provider=provider_name_final,
        provider_id=provider_id_final,
        meta_obj=meta,
        df_columns=list(df.columns),
        meta_dict=meta_dict,
        dataset_jsonld=dataset_jsonld,
    )

    logging.info("Dropping rows with any NA for modeling")
    df_no_na = df.dropna(axis=0, how="any").reset_index(drop=True)
    if len(df_no_na) == 0:
        logging.info("All rows removed by dropna(); imputing missing values as fallback")
        df_imp = df.copy()
        for c in cont_cols:
            if c in df_imp.columns:
                try:
                    med = float(pd.to_numeric(df_imp[c], errors="coerce").median())
                except Exception:
                    med = 0.0
                df_imp[c] = pd.to_numeric(df_imp[c], errors="coerce").fillna(med).astype(float)
        for c in disc_cols:
            if c in df_imp.columns:
                try:
                    mode = df_imp[c].mode(dropna=True)
                    fill = mode.iloc[0] if len(mode) else (df_imp[c].cat.categories[0] if pd.api.types.is_categorical_dtype(df_imp[c]) and len(df_imp[c].cat.categories) else "")
                except Exception:
                    fill = ""
                if pd.api.types.is_categorical_dtype(df_imp[c]):
                    if (isinstance(fill, str) and fill in list(df_imp[c].cat.categories)) or (
                        not isinstance(fill, str) and fill in list(df_imp[c].cat.categories)
                    ):
                        df_imp[c] = df_imp[c].fillna(fill)
                    else:
                        df_imp[c] = df_imp[c].cat.add_categories([fill]).fillna(fill)
                else:
                    df_imp[c] = df_imp[c].fillna(fill)
        df_no_na = df_imp.reset_index(drop=True)
    train_df, test_df = train_test_split(
        df_no_na, test_size=cfg.test_size, random_state=cfg.random_state, shuffle=True
    )
    logging.info(f"Train/test split sizes: {len(train_df)}/{len(test_df)}")

    # Arc blacklist is backend-specific (handled by pybnesian backend if needed)

    # Load model configs (pybnesian + synthcity). If none provided, use default_config.yaml
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
    plot_umap(umap_art.embedding, str(umap_png_real), title=f"{name}: real (sample)", color_labels=umap_art.color_labels)

    metasyn_semmap_parquet_file: Optional[str] = None
    models_root = model_run_root(outdir)
    # Run models through backends
    for idx, spec in enumerate(model_configs):
        backend = (spec.backend or "pybnesian").lower()
        label = spec.name or f"model_{idx+1}"
        seed = int(spec.seed if spec.seed is not None else cfg.random_state)
        if backend == "pybnesian":
            try:
                backend_pyb.run_experiment(
                    df=df_no_na,
                    provider=provider_name_final,
                    dataset_name=name,
                    provider_id=provider_id_final,
                    outdir=str(outdir),
                    label=label,
                    model_info=spec.model or {},
                    rows=spec.rows,
                    seed=seed,
                    test_size=cfg.test_size,
                    semmap_export=semmap_export,
                )
            except Exception:
                logging.exception("pybnesian run failed for %s", label)
        elif backend == "synthcity":
            try:
                mname = (spec.model or {}).get("name")
                params = (spec.model or {}).get("params", {})
                backend_syn.run_experiment(
                    df=df_no_na,
                    provider=provider_name_final,
                    dataset_name=name,
                    provider_id=provider_id_final,
                    outdir=str(outdir),
                    label=label,
                    model_name=str(mname),
                    params=params,
                    rows=spec.rows,
                    seed=seed,
                    test_size=cfg.test_size,
                    semmap_export=semmap_export,
                )
            except Exception:
                logging.exception("synthcity run failed for %s", label)
        else:
            logging.warning("Unknown backend %s for model %s; skipping", backend, label)

    # ------------- MetaSyn fit and synthesize -------------
    # Fit MetaFrame (GMF) on the same training data
    metasyn_gmf = None
    synth_meta_df = pd.DataFrame(columns=df_no_na.columns)
    if run_metasyn:
        logging.info("Fitting MetaSyn MetaFrame on train")
        mf = MetaFrame.fit_dataframe(train_df)
        metasyn_gmf = outdir / "metasyn_gmf.json"
        try:
            mf.save(str(metasyn_gmf))
        except Exception as e:
            logging.warning(f"Could not save MetaSyn GMF: {e}")
            metasyn_gmf = None

        # Synthesize MetaSyn data
        try:
            synth_meta = mf.synthesize(n=cfg.synthetic_sample)
            import polars as pl
            if isinstance(synth_meta, pl.DataFrame):
                synth_meta_df = synth_meta.to_pandas()
            else:
                synth_meta_df = synth_meta
        except Exception as e:
            logging.warning(f"MetaSyn synthesis failed: {e}")
            synth_meta_df = pd.DataFrame(columns=df_no_na.columns)

        # Align columns and dtypes
        synth_meta_df = synth_meta_df.reindex(columns=df_no_na.columns)
        for c in disc_cols:
            if c in synth_meta_df.columns:
                synth_meta_df[c] = synth_meta_df[c].astype("category")
                if pd.api.types.is_categorical_dtype(df_no_na[c]):
                    synth_meta_df[c] = synth_meta_df[c].cat.set_categories(df_no_na[c].cat.categories)
        synth_meta_df = coerce_continuous_to_float(synth_meta_df, cont_cols)

        if semmap_export:
            try:
                synth_meta_df.semmap.apply_json_metadata(copy.deepcopy(semmap_export), convert_pint=False)
                metasyn_semmap_parquet_path = outdir / "synthetic_metasyn.semmap.parquet"
                synth_meta_df.semmap.to_parquet(str(metasyn_semmap_parquet_path), index=False)
                metasyn_semmap_parquet_file = str(metasyn_semmap_parquet_path)
                logging.info(
                    "Wrote SemMap parquet for MetaSyn synthetic: %s",
                    metasyn_semmap_parquet_path,
                )
            except Exception:
                logging.exception("Failed to serialize SemMap parquet for MetaSyn synthetic")

    # Distances for MetaSyn model (use overall discrete/continuous cols inferred earlier)
    dist_table_meta = per_variable_distances(test_df, synth_meta_df, disc_cols, cont_cols)

    umap_png_meta = None
    if run_metasyn and len(synth_meta_df):
        synth_meta_no_na = synth_meta_df.dropna(axis=0, how="any")
        synth_meta_emb = transform_with_umap(umap_art, synth_meta_no_na)
        umap_png_meta = outdir / "umap_metasyn.png"
        plot_umap(synth_meta_emb, str(umap_png_meta), title=f"{name}: synthetic (MetaSyn sample)")

    # Fidelity summary table
    def summarize_distances(dt: pd.DataFrame) -> dict:
        d_disc = dt[dt['type'] == 'discrete'] if not dt.empty else pd.DataFrame()
        d_cont = dt[dt['type'] == 'continuous'] if not dt.empty else pd.DataFrame()
        res = {
            'disc_jsd_mean': float(d_disc['JSD'].mean()) if ('JSD' in d_disc.columns and len(d_disc)) else float('nan'),
            'disc_jsd_median': float(d_disc['JSD'].median()) if ('JSD' in d_disc.columns and len(d_disc)) else float('nan'),
            'cont_ks_mean': float(d_cont['KS'].mean()) if ('KS' in d_cont.columns and len(d_cont)) else float('nan'),
            'cont_w1_mean': float(d_cont['W1'].mean()) if ('W1' in d_cont.columns and len(d_cont)) else float('nan'),
        }
        return res

    fidelity_rows = []
    if run_metasyn:
        def summarize_distances(dt: pd.DataFrame) -> dict:
            d_disc = dt[dt['type'] == 'discrete'] if not dt.empty else pd.DataFrame()
            d_cont = dt[dt['type'] == 'continuous'] if not dt.empty else pd.DataFrame()
            return {
                'disc_jsd_mean': float(d_disc['JSD'].mean()) if ('JSD' in d_disc.columns and len(d_disc)) else float('nan'),
                'disc_jsd_median': float(d_disc['JSD'].median()) if ('JSD' in d_disc.columns and len(d_disc)) else float('nan'),
                'cont_ks_mean': float(d_cont['KS'].mean()) if ('KS' in d_cont.columns and len(d_cont)) else float('nan'),
                'cont_w1_mean': float(d_cont['W1'].mean()) if ('W1' in d_cont.columns and len(d_cont)) else float('nan'),
            }
        meta_summary = summarize_distances(dist_table_meta)
        fidelity_rows.append(dict(model='MetaSyn', **meta_summary))
    fidelity_table = pd.DataFrame(fidelity_rows)

    # Determine provider and dataset page links
    provider_name, prov_id = provider_name_final, provider_id_final
    # If UCI, get variable descriptions from cached metadata for report table
    var_desc_map = {}
    if provider_name == 'uciml' and isinstance(prov_id, int):
        try:
            var_desc_map = get_uciml_variable_descriptions(prov_id)
        except Exception:
            var_desc_map = {}

    # Generate UMAPs for discovered model runs using the same projection
    model_runs = []
    try:
        model_runs = discover_model_runs(outdir)
        for run in model_runs:
            try:
                s_df = pd.read_csv(run.synthetic_csv)
                s_df = s_df.reindex(columns=df_no_na.columns)
                s_no_na = s_df.dropna(axis=0, how="any")
                s_emb = transform_with_umap(umap_art, s_no_na)
                run.umap_png = run.run_dir / "umap.png"
                plot_umap(s_emb, str(run.umap_png), title=f"{name}: synthetic ({run.name})")
            except Exception:
                logging.exception("Failed to generate UMAP for model run %s", run.run_dir)
    except Exception:
        logging.exception('Failed to discover model runs for %s', name)
        model_runs = []

    write_report_md(
        outdir=outdir,
        dataset_name=name,
        metadata_file=metadata_file,
        dataset_jsonld_file=str(dataset_jsonld_file),
        dataset_jsonld=dataset_jsonld,
        dataset_provider=provider_name,
        dataset_provider_id=prov_id,
        df=df_no_na,
        disc_cols=disc_cols,
        cont_cols=cont_cols,
        baseline_df=baseline_df,
        dist_table_meta=dist_table_meta,
        fidelity_table=fidelity_table,
        umap_png_real=str(umap_png_real),
        umap_png_meta=(str(umap_png_meta) if umap_png_meta else None),
        metasyn_gmf_file=(str(metasyn_gmf) if metasyn_gmf is not None else None),
        metasyn_semmap_parquet=metasyn_semmap_parquet_file,
        declared_types=declared_map or None,
        inferred_types=inferred_map or None,
        variable_descriptions=var_desc_map or None,
        semmap_jsonld=semmap_export,
        model_runs=model_runs,
    )
