from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split

# Integrated BN helper functionality (migrated from bn.py)
from dataclasses import dataclass
from graphviz import Digraph
import networkx as nx
import numpy as np
from pybnesian import hc, CLGNetworkType, SemiparametricBNType


@dataclass
class BNArtifacts:
    model: object
    discrete_cols: List[str]
    continuous_cols: List[str]


def _bn_type_from_str(bn_type: str):
    t = bn_type.lower().strip()
    if t in {"clg", "clgnetwork", "clgnetworktype"}:
        return CLGNetworkType()
    if t in {"semiparametric", "semiparametricbn", "semiparametricbntype", "spbn", "semi"}:
        return SemiparametricBNType()
    raise ValueError(f"Unsupported bn_type: {bn_type!r}. Supported: 'clg', 'semiparametric'")


def learn_bn(
    train_df,
    bn_type: str = "clg",
    random_state: int = 42,
    arc_blacklist: Optional[List[Tuple[str, str]]] = None,
    *,
    score: Optional[str] = None,
    operators: Optional[List[str]] = None,
    max_indegree: Optional[int] = None,
) -> BNArtifacts:
    bn_type_obj = _bn_type_from_str(bn_type)
    score = score or "bic"
    operators = list(operators) if operators is not None else ["arcs"]
    max_indegree = 5 if max_indegree is None else int(max_indegree)
    model = hc(
        train_df,
        bn_type=bn_type_obj,
        score=score,
        operators=operators,
        max_indegree=max_indegree,
        seed=random_state,
        arc_blacklist=arc_blacklist,
    )
    model.fit(train_df)
    node_types = model.node_types()
    discrete_cols, continuous_cols = [], []
    for node, ftype in node_types.items():
        name = type(ftype).__name__.lower()
        if "discrete" in name:
            discrete_cols.append(node)
        else:
            continuous_cols.append(node)
    return BNArtifacts(model=model, discrete_cols=discrete_cols, continuous_cols=continuous_cols)


def bn_to_graphviz(model, node_types: Dict[str, object], out_png: str, title: str = "Learned BN") -> None:
    dot = Digraph(comment=title, format="png")
    dot.attr(rankdir="LR")
    for node, ftype in node_types.items():
        tname = type(ftype).__name__
        if "Discrete" in tname:
            shape = "box"
            fillcolor = "#f3d19c"
        else:
            shape = "ellipse"
            fillcolor = "#9cc9f3"
        dot.node(node, label=node, shape=shape, style="filled", fillcolor=fillcolor)
    for u, v in model.arcs():
        dot.edge(u, v)
    dot.render(filename=out_png.replace('.png', ''), cleanup=True)


def save_graphml_structure(model, node_types: Dict[str, object], out_graphml: str) -> None:
    G = nx.DiGraph()
    for n in model.nodes():
        ftype = node_types[n]
        G.add_node(n, bn_type=type(ftype).__name__)
    for u, v in model.arcs():
        G.add_edge(u, v)
    nx.write_graphml(G, out_graphml)
from ..metrics import per_variable_distances, summarize_distance_metrics, heldout_loglik
from ..utils import (
    coerce_continuous_to_float,
    coerce_discrete_to_category,
    ensure_dir,
    infer_types,
    rename_categorical_categories_to_str,
)
from ..models import model_run_root, write_manifest


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
    """Fit/generate/evaluate a PyBNesian model and write artifacts under models/<label>.

    model_info keys: type ('clg' or 'semiparametric'), score, operators, max_indegree.
    """
    model_info = dict(model_info or {})
    bn_type = str(model_info.get("type", "clg"))
    score = model_info.get("score")
    operators = model_info.get("operators")
    max_indegree = model_info.get("max_indegree")
    logging.info("Starting PyBNesian run: %s (type=%s)", label, bn_type)

    working = df.copy()
    disc_cols, cont_cols = infer_types(working)
    working = coerce_discrete_to_category(working, disc_cols)
    working = rename_categorical_categories_to_str(working, disc_cols)
    working = coerce_continuous_to_float(working, cont_cols)

    train_df, test_df = train_test_split(working, test_size=test_size, random_state=seed, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Simple arc blacklist: common sensitive vars
    default_root = ["age", "sex", "race"]
    cols = list(working.columns)
    col_map = {str(c).lower(): c for c in cols}
    sens_in_cols = [col_map[s] for s in [x.lower() for x in default_root] if s in col_map]
    arc_blacklist_pairs: List[Tuple[str, str]] = []
    for u in sens_in_cols:
        for v in cols:
            arc_blacklist_pairs.append((v, u))

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
    # Sample
    n_rows = int(rows) if rows else len(train_df)
    synth = model.sample(n_rows, seed=seed)
    synth_df = synth.to_pandas().reindex(columns=working.columns)
    for c in disc_cols:
        if c in synth_df.columns:
            synth_df[c] = synth_df[c].astype("category")

    run_root = model_run_root(Path(outdir))
    run_dir = run_root / label
    ensure_dir(str(run_dir))
    synth_df.to_csv(run_dir / "synthetic.csv", index=False)
    logging.info("Wrote synthetic CSV: %s", run_dir / "synthetic.csv")

    # Optional SemMap parquet
    if semmap_export:
        try:
            import copy
            import semmap  # noqa: F401  # register accessor
            sdf = synth_df.copy()
            sdf.semmap.apply_json_metadata(copy.deepcopy(semmap_export), convert_pint=False)
            sdf.semmap.to_parquet(str(run_dir / "synthetic.semmap.parquet"), index=False)
        except Exception as e:
            logging.warning("SemMap parquet failed: %s", e)

    # Distances and summary
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
        "heldout_loglik": heldout_loglik(model, test_df),
        "umap_png": None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # BN visualizations and serialization inside model dir
    try:
        node_types = model.node_types()
        bn_to_graphviz(model, node_types, str(run_dir / f"structure.png"), title=f"{dataset_name or 'dataset'} — {label} BN")
        save_graphml_structure(model, node_types, run_dir / f"structure.graphml")
        model.save(str(run_dir / f"model.pickle"))
    except Exception as e:
        logging.warning("BN serialization failed: %s", e)

    manifest = {
        "backend": "pybnesian",
        "name": label,
        "provider": provider,
        "dataset_name": dataset_name,
        "provider_id": provider_id,
        "type": bn_type,
        "params": dict(type=bn_type, score=score, operators=operators, max_indegree=max_indegree),
        "seed": seed,
        "rows": n_rows,
        "test_size": test_size,
    }
    write_manifest(run_dir, manifest)
    logging.info("Finished PyBNesian run: %s", label)
    return run_dir
