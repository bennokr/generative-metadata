from __future__ import annotations

import os
from typing import Optional, Any
import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import (
    coerce_continuous_to_float,
    coerce_discrete_to_category,
    infer_types,
    rename_categorical_categories_to_str,
    seed_all,
    summarize_dataframe,
    dataframe_to_markdown_table,
    ensure_dir,
)
from .umap_utils import build_umap, plot_umap, transform_with_umap
from .bn import learn_bn, bn_to_graphviz, bn_human_readable, save_graphml_structure
from .metrics import heldout_loglik, per_variable_distances
from .reporting import write_report_md


class Config:
    random_state = 42
    max_umap_sample = 1000
    synthetic_sample = 1000
    test_size = 0.2
    umap_n_neighbors = 30
    umap_min_dist = 0.1
    umap_n_components = 2

def string_attributes(obj):
    return {
        name: value
        for name in dir(obj)
        if not name.startswith("_")  # skip dunder attributes
        and isinstance((value := getattr(obj, name, None)), str)
    }

def dict_attributes(obj):
    dict_atts = {
        name: value
        for name in dir(obj)
        if not name.startswith("_")  # skip dunder attributes
        and isinstance((value := getattr(obj, name, None)), dict)
    }
    for key, d in list(dict_atts.items()):
        for k,v in d.items():
            if not isinstance(v, (int, str, float)):
                d[k] = string_attributes(v)
        if all(isinstance(k, int) for k in d.keys()):
            dict_atts[key] = list(d.values())
    return dict_atts



def process_dataset(
    meta: Any,
    df: pd.DataFrame,
    color_series: Optional[pd.Series],
    base_outdir: str,
    bn_type: str = "clg",
    cfg: Config = Config(),
) -> None:
    name = meta.name

    outdir = os.path.join(base_outdir, name.replace("/", "_"))
    ensure_dir(outdir)

    metadata_file = os.path.join(outdir, f"metadata.json")
    try:
        meta_dict = dict(meta)
    except:
        meta_dict = string_attributes(meta)
        meta_dict.update(**dict_attributes(meta))
    json.dump(meta_dict, open(metadata_file, 'w'), indent=2)


    disc_cols, cont_cols = infer_types(df)
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

    baseline = summarize_dataframe(df, disc_cols, cont_cols)
    baseline_md = dataframe_to_markdown_table(baseline)

    df_no_na = df.dropna(axis=0, how="any").reset_index(drop=True)
    train_df, test_df = train_test_split(
        df_no_na, test_size=cfg.test_size, random_state=cfg.random_state, shuffle=True
    )

    bn_art = learn_bn(train_df, bn_type=bn_type, random_state=cfg.random_state)
    model = bn_art.model
    node_types = model.node_types()

    bn_png = os.path.join(outdir, f"bn_{bn_type}.png")
    bn_to_graphviz(model, node_types, bn_png, title=f"{name} — {bn_type.upper()} BN")

    synth = model.sample(cfg.synthetic_sample, seed=cfg.random_state)
    synth_df = synth.to_pandas()

    ll = heldout_loglik(model, test_df)
    synth_df = synth_df[df_no_na.columns]
    for c in disc_cols:
        if c in synth_df.columns:
            synth_df[c] = synth_df[c].astype("category")
            if pd.api.types.is_categorical_dtype(df_no_na[c]):
                synth_df[c] = synth_df[c].cat.set_categories(df_no_na[c].cat.categories)
    dist_table = per_variable_distances(test_df, synth_df, bn_art.discrete_cols, bn_art.continuous_cols)


    rng = seed_all(cfg.random_state)
    color_series2 = None
    if isinstance(color_series, pd.Series) and color_series.name in df_no_na.columns:
        color_series2 = df_no_na[color_series.name]
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
    umap_png_real = os.path.join(outdir, f"umap_real.png")
    plot_umap(umap_art.embedding, umap_png_real, title=f"{name}: real (sample)", color_labels=umap_art.color_labels)
    
    synth_no_na = synth_df.dropna(axis=0, how="any")
    synth_emb = transform_with_umap(umap_art, synth_no_na)
    umap_png_synth = os.path.join(outdir, f"umap_synth.png")
    plot_umap(synth_emb, umap_png_synth, title=f"{name}: synthetic (BN sample)")

    bn_human = bn_human_readable(model)
    graphml_file = os.path.join(outdir, f"structure.graphml")
    save_graphml_structure(model, node_types, graphml_file)
    pickle_file = os.path.join(outdir, f"model.pickle")
    model.save(pickle_file)

    write_report_md(
        outdir=outdir,
        dataset_name=name,
        metadata_file=metadata_file,
        df=df_no_na,
        disc_cols=disc_cols,
        cont_cols=cont_cols,
        baseline_md_table=baseline_md,
        bn_type=bn_type,
        bn_png=bn_png,
        bn_human=bn_human,
        ll_metrics=ll,
        dist_table=dist_table,
        graphml_file=graphml_file,
        pickle_file=pickle_file,
        umap_png_real=umap_png_real,
        umap_png_synth=umap_png_synth,
    )

