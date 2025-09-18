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
from .bn import learn_bn, bn_to_graphviz, save_graphml_structure
from .metrics import heldout_loglik, per_variable_distances
from .reporting import write_report_md
from .mappings import resolve_mapping_json, load_mapping_json
from .synth import discover_synth_runs
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
    bn_configs: Optional[List[Dict[str, Any]]] = None,
    roots: Optional[List[str]] = None,
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

    # Determine arc blacklist variables
    default_root = ["age", "sex", "race"]
    root_vars: List[str]
    if isinstance(roots, (list, tuple)) and len(roots):
        root_vars = [str(x) for x in roots]
    else:
        demo = meta_dict.get('demographics')
        if isinstance(demo, (list, tuple)):
            root_vars = [str(x) for x in demo]
        elif isinstance(demo, dict):
            root_vars = [str(k) for k in demo.keys()]
        else:
            root_vars = default_root
    logging.info(f'Using {root_vars=}')

    # Build arc blacklist pairs: forbid arcs INTO root vars FROM non-root variables only
    cols = list(df.columns)
    col_map = {str(c).lower(): c for c in cols}
    sens_in_cols = []
    for s in root_vars:
        t = str(s).lower()
        if t in col_map:
            sens_in_cols.append(col_map[t])
    logging.info(f'Using {sens_in_cols=}')
    arc_blacklist_pairs: List[Tuple[str, str]] = []
    for u in sens_in_cols:
        for v in cols:
            arc_blacklist_pairs.append((v, u))

    # BN configurations to run
    if isinstance(bn_configs, (list, tuple)) and len(bn_configs):
        configs_to_run = list(bn_configs)
    else:
        # Default: two configs that actually differ
        configs_to_run = [
            dict(name="clg_mi2", bn_type="clg", score="bic", operators=["arcs"], max_indegree=2),
            dict(name="semi_mi5", bn_type="semiparametric", score="bic", operators=["arcs"], max_indegree=5),
        ]

    # Placeholder to collect BN-specific results for reporting
    bn_sections: List[Dict[str, Any]] = []


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

    # Learn and evaluate each BN configuration
    metasyn_semmap_parquet_file: Optional[str] = None

    for idx, cfg_item in enumerate(configs_to_run):
        bn_type = str(cfg_item.get("bn_type", "clg"))
        score = cfg_item.get("score")
        operators = cfg_item.get("operators")
        max_indegree = cfg_item.get("max_indegree")
        seed = int(cfg_item.get("seed", cfg.random_state))
        label = cfg_item.get("name") or f"{bn_type}_{idx+1}"
        logging.info(f"Learning BN model (label={label}, type={bn_type}, score={score}, max_indegree={max_indegree}, seed={seed})")
        bn_art = learn_bn(
            train_df,
            bn_type=bn_type,
            random_state=seed,
            arc_blacklist=arc_blacklist_pairs,
            score=score,
            operators=operators,
            max_indegree=max_indegree,
        )
        model = bn_art.model
        node_types = model.node_types()

        bn_png = outdir / f"bn_{label}.png"
        bn_to_graphviz(model, node_types, str(bn_png), title=f"{name} — {label} BN")

        logging.info(f"Sampling BN synthetic n={cfg.synthetic_sample} for {label}")
        synth = model.sample(cfg.synthetic_sample, seed=seed)
        synth_df = synth.to_pandas()

        logging.info("Computing BN held-out log-likelihood")
        ll = heldout_loglik(model, test_df)
        synth_df = synth_df[df_no_na.columns]
        for c in disc_cols:
            if c in synth_df.columns:
                synth_df[c] = synth_df[c].astype("category")
                if pd.api.types.is_categorical_dtype(df_no_na[c]):
                    synth_df[c] = synth_df[c].cat.set_categories(df_no_na[c].cat.categories)
        bn_semmap_parquet: Optional[str] = None
        if semmap_export:
            try:
                synth_df.semmap.apply_json_metadata(copy.deepcopy(semmap_export), convert_pint=False)
                bn_semmap_parquet_path = outdir / f"synthetic_bn_{label}.semmap.parquet"
                synth_df.semmap.to_parquet(str(bn_semmap_parquet_path), index=False)
                logging.info(
                    "Wrote SemMap parquet for BN synthetic %s: %s",
                    label,
                    bn_semmap_parquet_path,
                )
            except Exception:
                logging.exception("Failed to write SemMap parquet for BN synthetic %s", label)
        logging.info("Per-variable distances for BN computed")
        dist_table_bn = per_variable_distances(test_df, synth_df, bn_art.discrete_cols, bn_art.continuous_cols)

        # UMAP transform for this BN synthetic
        synth_no_na = synth_df.dropna(axis=0, how="any")
        synth_emb = transform_with_umap(umap_art, synth_no_na)
        umap_png_bn = outdir / f"umap_bn_{label}.png"
        plot_umap(synth_emb, str(umap_png_bn), title=f"{name}: synthetic (BN sample: {label})")

        graphml_file = outdir / f"structure_{label}.graphml"
        save_graphml_structure(model, node_types, graphml_file)
        pickle_file = outdir / f"model_{label}.pickle"
        model.save(str(pickle_file))

        bn_sections.append(
            dict(
                label=label,
                bn_type=bn_type,
                bn_png=str(bn_png),
                ll_metrics=ll,
                dist_table=dist_table_bn,
                graphml_file=str(graphml_file),
                pickle_file=str(pickle_file),
                umap_png=str(umap_png_bn),
                params=dict(
                    bn_type=bn_type,
                    score=score or "bic",
                    operators=list(operators) if operators is not None else ["arcs"],
                    max_indegree=int(max_indegree) if max_indegree is not None else 5,
                    seed=seed,
                ),
                semmap_parquet=bn_semmap_parquet,
            )
        )

    # ------------- MetaSyn fit and synthesize -------------
    # Fit MetaFrame (GMF) on the same training data
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

    # UMAP transform for metasyn
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

    meta_summary = summarize_distances(dist_table_meta)
    fidelity_rows = []
    for sect in bn_sections:
        llm = sect['ll_metrics']
        bnsum = summarize_distances(sect['dist_table'])
        fidelity_rows.append(
            dict(model=f"BN:{sect.get('label') or sect['bn_type']}", mean_loglik=llm['mean_loglik'], std_loglik=llm['std_loglik'], sum_loglik=llm['sum_loglik'], **bnsum)
        )
    fidelity_rows.append(
        dict(model='MetaSyn', mean_loglik=float('nan'), std_loglik=float('nan'), sum_loglik=float('nan'), **meta_summary)
    )
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

    synth_runs = []
    try:
        synth_runs = discover_synth_runs(base_outdir, provider=provider_name, dataset_name=name)
    except Exception:
        logging.exception('Failed to load synthcity runs for %s', name)
        synth_runs = []

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
        bn_sections=bn_sections,
        dist_table_meta=dist_table_meta,
        fidelity_table=fidelity_table,
        roots_info=dict(
            root_variables=sens_in_cols,
            n_forbidden_arcs=len(arc_blacklist_pairs),
        ),
        umap_png_real=str(umap_png_real),
        umap_png_bns=[sect['umap_png'] for sect in bn_sections],
        umap_png_meta=str(umap_png_meta),
        metasyn_gmf_file=(str(metasyn_gmf) if metasyn_gmf is not None else None),
        declared_types=declared_map or None,
        inferred_types=inferred_map or None,
        variable_descriptions=var_desc_map or None,
        semmap_jsonld=semmap_export,
        metasyn_semmap_parquet=metasyn_semmap_parquet_file,
        synth_runs=synth_runs,
    )
