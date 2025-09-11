#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mixed-dtype BN reports:
- Download multiple mixed-dtype datasets from OpenML
- Basic summary tables
- UMAP fit + visualization on a large random sample
- Learn a CLG (hybrid) BN with PyBNesian (structure + parameters)
- Visualize the learned BN (shapes/colors by node type)
- Sample synthetic data from the BN
- Fidelity report: held-out log-likelihood + per-variable distribution distances
- Reuse the same UMAP to visualize synthetic data
- Serialize: human-readable summary in the report + GraphML (interoperable structure) + pickle

Requirements (install first):
  pip install pandas numpy scipy scikit-learn umap-learn matplotlib graphviz networkx pybnesian
  # Plus Graphviz system binaries for layout (e.g., apt-get install graphviz or brew install graphviz)

Notes:
- PyBNesian docs show: hc(), CLGNetworkType, logl(), slogl(), sample(), node_types() etc.
- PyBNesian accepts pandas.DataFrame as input (it wraps to pyarrow internally).
"""

import os
import json
import math
import textwrap
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

import umap
import networkx as nx
from graphviz import Digraph
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

# PyBNesian imports (API: hc(), CLGNetworkType, load/save, BNBase.logl/slogl/sample, node_types(), arcs(), nodes())
from pybnesian import hc, CLGNetworkType, load as pybn_load

# ---------------------------
# Config
# ---------------------------

RANDOM_STATE = 42
MAX_UMAP_SAMPLE = 1000          # "large random sample" upper bound for UMAP fitting/plotting
SYNTHETIC_SAMPLE = 1000         # number of synthetic samples to draw for fidelity + UMAP
TEST_SIZE = 0.2
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2

# Mixed-dtype OpenML datasets (well-known, stable)
# Each item: name or (name, target_column) where target may be used for coloring
DATASETS: List[Tuple[str, Optional[str]]] = [
    ("adult", "class"),           # income <=50K/>50K (categorical target)
    ("credit-g", "class"),        # German credit (good/bad)
    ("titanic", "survived"),      # 0/1
    ("bank-marketing", "y"),      # yes/no
]

# ---------------------------
# Helpers
# ---------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def seed_all(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s)

def is_discrete_series(s: pd.Series, cardinality_threshold: int = 20) -> bool:
    # Treat bool, categorical, object, and low-cardinality integers as discrete
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        # low-card integer -> discrete
        try:
            n_uniq = s.nunique(dropna=True)
            return n_uniq <= cardinality_threshold
        except Exception:
            return False
    return False

def infer_types(df: pd.DataFrame, cardinality_threshold: int = 20) -> Tuple[List[str], List[str]]:
    disc, cont = [], []
    for c in df.columns:
        s = df[c]
        if is_discrete_series(s, cardinality_threshold):
            disc.append(c)
        elif is_numeric_series(s):
            cont.append(c)
        else:
            # default: treat as discrete
            disc.append(c)
    return disc, cont

def coerce_discrete_to_category(df: pd.DataFrame, discrete_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in discrete_cols:
        if not pd.api.types.is_categorical_dtype(df[c]):
            df[c] = df[c].astype("category")
    return df

def coerce_continuous_to_float(df: pd.DataFrame, continuous_cols: List[str]) -> pd.DataFrame:
    """Ensure continuous variables use a floating dtype (required by CLGNetworkType)."""
    df = df.copy()
    for c in continuous_cols:
        s = df[c]
        # pandas boolean should not appear here per our inference, but guard anyway
        if pd.api.types.is_integer_dtype(s):
            df[c] = pd.to_numeric(s, errors="coerce").astype(float)
    return df

def rename_categorical_categories_to_str(df: pd.DataFrame, discrete_cols: List[str]) -> pd.DataFrame:
    """Ensure categorical variables have string-valued categories (required by PyBNesian)."""
    df = df.copy()
    for c in discrete_cols:
        s = df[c]
        if pd.api.types.is_categorical_dtype(s):
            try:
                new_cats = [str(cat) for cat in s.cat.categories]
                df[c] = s.cat.rename_categories(new_cats)
            except Exception:
                # Fallback: cast via string but keep NaN as NaN
                mask = s.isna()
                tmp = s.astype(str)
                tmp[mask] = np.nan
                df[c] = tmp.astype("category")
    return df

def summarize_dataframe(df: pd.DataFrame, discrete_cols: List[str], continuous_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        col = df[c]
        na_frac = float(col.isna().mean())
        uniq = int(col.nunique(dropna=True))
        if c in continuous_cols:
            desc = col.describe(percentiles=[0.25, 0.5, 0.75])
            # handle empty/all-NA
            mean = float(desc.get("mean", np.nan)) if not isinstance(desc, float) else float("nan")
            std = float(desc.get("std", np.nan)) if not isinstance(desc, float) else float("nan")
            minv = float(desc.get("min", np.nan)) if not isinstance(desc, float) else float("nan")
            q25 = float(desc.get("25%", np.nan)) if "25%" in desc else float("nan")
            q50 = float(desc.get("50%", np.nan)) if "50%" in desc else float("nan")
            q75 = float(desc.get("75%", np.nan)) if "75%" in desc else float("nan")
            maxv = float(desc.get("max", np.nan)) if not isinstance(desc, float) else float("nan")
            rows.append(dict(variable=c, type="continuous", na_frac=na_frac, unique=uniq,
                             mean=mean, std=std, min=minv, q25=q25, median=q50, q75=q75, max=maxv))
        else:
            # discrete
            top = col.value_counts(dropna=True).head(3)
            top_items = "; ".join([f"{k}:{int(v)}" for k, v in top.items()])
            rows.append(dict(variable=c, type="discrete", na_frac=na_frac, unique=uniq,
                             top3=top_items))
    return pd.DataFrame(rows)

def dataframe_to_markdown_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    def fmt(x):
        if isinstance(x, float):
            if math.isnan(x):
                return ""
            return float_fmt.format(x)
        return str(x)
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)

def pick_color_labels(series: Optional[pd.Series]) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    if series is None:
        return None, None
    # Convert to categorical labels for coloring
    values = series.astype("category")
    cats = list(values.cat.categories)
    mapping = {cat: i for i, cat in enumerate(cats)}
    labels = values.map(mapping).to_numpy()
    return labels, mapping

@dataclass
class UMAPArtifacts:
    preproc: ColumnTransformer
    umap_model: umap.UMAP
    sample_idx: np.ndarray
    embedding: np.ndarray
    label_mapping: Optional[Dict]
    color_labels: Optional[np.ndarray]

def build_umap(df: pd.DataFrame, discrete_cols: List[str], continuous_cols: List[str],
               color_series: Optional[pd.Series], rng: np.random.Generator) -> UMAPArtifacts:
    # Preprocess: impute + scale for continuous; impute + one-hot for discrete
    cont_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ])
    disc_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("cont", cont_pipe, continuous_cols),
            ("disc", disc_pipe, discrete_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,  # keep sparse if possible
    )

    n = len(df)
    sample_n = min(MAX_UMAP_SAMPLE, n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    df_sample = df.iloc[sample_idx].reset_index(drop=True)

    X = preproc.fit_transform(df_sample)  # sparse or dense
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=UMAP_N_COMPONENTS,
        random_state=RANDOM_STATE,
        metric="euclidean",
        verbose=False,
    )
    embedding = reducer.fit_transform(X)

    labels, mapping = pick_color_labels(color_series.iloc[sample_idx] if color_series is not None else None)
    return UMAPArtifacts(preproc=preproc, umap_model=reducer, sample_idx=sample_idx,
                         embedding=embedding, label_mapping=mapping, color_labels=labels)

def transform_with_umap(art: UMAPArtifacts, df: pd.DataFrame) -> np.ndarray:
    X = art.preproc.transform(df)
    return art.umap_model.transform(X)

def plot_umap(embedding: np.ndarray, outfile: str, title: str,
              color_labels: Optional[np.ndarray] = None) -> None:
    plt.figure(figsize=(7, 6), dpi=140)
    if color_labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=4, alpha=0.6)
    else:
        # simple coloring by integer labels (matplotlib default colormap)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=4, alpha=0.6, c=color_labels)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

# ---------------------------
# Learning & visualization (PyBNesian)
# ---------------------------

@dataclass
class BNArtifacts:
    model: object
    discrete_cols: List[str]
    continuous_cols: List[str]

def learn_clg_bn(train_df: pd.DataFrame) -> BNArtifacts:
    # Let CLGNetworkType infer default FactorTypes from column dtypes.
    # Note: For CLGNetworkType, PyBNesian does not define default operators for the
    # hill-climbing search. Explicitly pass the "arcs" operator set (add/remove/flip arcs).
    bn_type = CLGNetworkType()
    # Use BIC by default (fast, general). Alternatives: "cv-lik", "holdout-lik", "validated-lik", "bge"/"bde" where valid.
    model = hc(
        train_df,
        bn_type=bn_type,
        score="bic",
        operators=["arcs"],
        max_indegree=5,
        seed=RANDOM_STATE,
    )
    # Fit parameters on the training data
    model.fit(train_df)
    # Infer node types
    node_types = model.node_types()  # dict: name -> FactorType
    discrete_cols = []
    continuous_cols = []
    for node, ftype in node_types.items():
        # FactorType class names we expect: DiscreteFactorType vs LinearGaussianCPDType (for CLG)
        name = type(ftype).__name__.lower()
        if "discrete" in name:
            discrete_cols.append(node)
        else:
            continuous_cols.append(node)
    return BNArtifacts(model=model, discrete_cols=discrete_cols, continuous_cols=continuous_cols)

def bn_to_graphviz(model, node_types: Dict[str, object], out_png: str, title: str = "Learned CLG BN") -> None:
    dot = Digraph(comment=title, format="png")
    dot.attr(rankdir="LR")
    # Node appearance by type
    for node, ftype in node_types.items():
        tname = type(ftype).__name__
        if "Discrete" in tname:
            shape = "box"
            fillcolor = "#f3d19c"
        else:
            shape = "ellipse"
            fillcolor = "#9cc9f3"
        dot.node(node, label=node, shape=shape, style="filled", fillcolor=fillcolor)
    # Arcs
    for u, v in model.arcs():
        dot.edge(u, v)
    dot.render(filename=out_png, cleanup=True)

def bn_human_readable(model) -> str:
    # Concise: list nodes with type + arcs + small CPD summary
    node_types = model.node_types()
    lines = []
    lines.append("Nodes and types:")
    for n in model.nodes():
        lines.append(f"  - {n}: {type(node_types[n]).__name__}")
    lines.append("\nArcs:")
    for u, v in model.arcs():
        lines.append(f"  - {u} -> {v}")
    # brief CPD peek: for Discrete show parent list; for LinearGaussian show betas/variance shape
    lines.append("\nParameters (glance):")
    for n in model.nodes():
        cpd = model.cpd(n)
        cname = type(cpd).__name__
        ev = cpd.evidence()
        if "Discrete" in cname:
            lines.append(f"  - {n}: DiscreteFactor | parents={list(ev)}")
        elif "LinearGaussian" in cname:
            try:
                beta = getattr(cpd, "beta", np.array([]))
                variance = getattr(cpd, "variance", None)
                if beta is None: beta = np.array([])
                beta_str = np.array2string(np.asarray(beta), precision=3, floatmode="fixed")
                lines.append(f"  - {n}: LinearGaussian | parents={list(ev)} | beta={beta_str} | var={variance:.4f}" if variance is not None else
                             f"  - {n}: LinearGaussian | parents={list(ev)} | beta={beta_str}")
            except Exception:
                lines.append(f"  - {n}: LinearGaussian | parents={list(ev)}")
        else:
            lines.append(f"  - {n}: {cname} | parents={list(ev)}")
    return "\n".join(lines)

def save_graphml_structure(model, node_types: Dict[str, object], out_graphml: str) -> None:
    """Common interoperable structure format (GraphML). Saves node attrs: 'bn_type'."""
    G = nx.DiGraph()
    for n in model.nodes():
        ftype = node_types[n]
        G.add_node(n, bn_type=type(ftype).__name__)
    for u, v in model.arcs():
        G.add_edge(u, v)
    nx.write_graphml(G, out_graphml)

# ---------------------------
# Fidelity metrics
# ---------------------------

def heldout_loglik(model, df_test: pd.DataFrame) -> Dict[str, float]:
    # Mean and std log-likelihood per instance; total log-likelihood
    arr = model.logl(df_test)  # per-row log-likelihood
    arr = np.asarray(arr).reshape(-1)
    return {
        "mean_loglik": float(np.mean(arr)),
        "std_loglik": float(np.std(arr)),
        "sum_loglik": float(np.sum(arr)),
        "n_rows": int(len(arr)),
    }

def js_divergence_discrete(p: pd.Series, q: pd.Series) -> float:
    # JSD over empirical discrete distributions with small smoothing
    cats = sorted(set(p.dropna().unique()).union(set(q.dropna().unique())))
    if len(cats) == 0:
        return float("nan")
    p_counts = p.value_counts(dropna=True).reindex(cats, fill_value=0).to_numpy(dtype=float)
    q_counts = q.value_counts(dropna=True).reindex(cats, fill_value=0).to_numpy(dtype=float)
    p_probs = (p_counts + 1e-9) / (p_counts.sum() + 1e-9 * len(cats))
    q_probs = (q_counts + 1e-9) / (q_counts.sum() + 1e-9 * len(cats))
    return float(jensenshannon(p_probs, q_probs, base=2.0))

def per_variable_distances(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                           discrete_cols: List[str], continuous_cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in real_df.columns:
        if c in discrete_cols:
            jsd = js_divergence_discrete(real_df[c], synth_df[c])
            rows.append(dict(variable=c, type="discrete", JSD=jsd))
        else:
            a = pd.to_numeric(real_df[c], errors="coerce").dropna()
            b = pd.to_numeric(synth_df[c], errors="coerce").dropna()
            if len(a) == 0 or len(b) == 0:
                rows.append(dict(variable=c, type="continuous", KS=float("nan"), W1=float("nan")))
                continue
            ks = ks_2samp(a, b, method="auto").statistic
            w1 = wasserstein_distance(a, b)
            rows.append(dict(variable=c, type="continuous", KS=ks, W1=w1))
    return pd.DataFrame(rows)

# ---------------------------
# Reporting
# ---------------------------

def write_report_md(
    outdir: str,
    dataset_name: str,
    df: pd.DataFrame,
    disc_cols: List[str],
    cont_cols: List[str],
    baseline_md_table: str,
    umap_png_real: str,
    umap_png_synth: str,
    bn_png: str,
    bn_human: str,
    ll_metrics: Dict[str, float],
    dist_table: pd.DataFrame,
    graphml_file: str,
    pickle_file: str,
) -> None:
    md_path = os.path.join(outdir, f"{dataset_name}_report.md")
    num_rows, num_cols = df.shape
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# BN Report — {dataset_name}\n\n")
        f.write(f"- Rows: {num_rows}\n")
        f.write(f"- Columns: {num_cols}\n")
        f.write(f"- Discrete: {len(disc_cols)}  |  Continuous: {len(cont_cols)}\n\n")
        f.write("## Baseline summary\n\n")
        f.write(baseline_md_table + "\n\n")
        f.write("## UMAP on real data (sample)\n\n")
        f.write(f"![UMAP real]({os.path.basename(umap_png_real)})\n\n")
        f.write("## Learned BN (structure)\n\n")
        f.write(f"![BN graph]({os.path.basename(bn_png)})\n\n")
        f.write("### Human-readable BN summary\n\n")
        f.write("```\n" + bn_human + "\n```\n\n")
        f.write("### Serialization\n\n")
        f.write(f"- Structure (GraphML): `{os.path.basename(graphml_file)}`\n")
        f.write(f"- Full model (pickle): `{os.path.basename(pickle_file)}`\n\n")
        f.write("## Fidelity\n\n")
        f.write(f"- Held-out mean log-likelihood per row: {ll_metrics['mean_loglik']:.4f}\n")
        f.write(f"- Held-out std log-likelihood per row: {ll_metrics['std_loglik']:.4f}\n")
        f.write(f"- Held-out total log-likelihood: {ll_metrics['sum_loglik']:.2f} over n={ll_metrics['n_rows']}\n\n")
        f.write("### Per-variable distances (lower is closer)\n\n")
        if not dist_table.empty:
            f.write(dataframe_to_markdown_table(dist_table) + "\n\n")
        f.write("## UMAP on synthetic data (same projection)\n\n")
        f.write(f"![UMAP synthetic]({os.path.basename(umap_png_synth)})\n\n")
    print(f"[OK] Wrote report: {md_path}")

# ---------------------------
# Main pipeline per dataset
# ---------------------------

def process_dataset(name: str, target: Optional[str], base_outdir: str, rng: np.random.Generator) -> None:
    outdir = os.path.join(base_outdir, name.replace("/", "_"))
    ensure_dir(outdir)
    print(f"\n=== Dataset: {name} ===")

    # 1) Download from OpenML
    print("Downloading from OpenML...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        versions = openml.datasets.list_datasets(
            output_format="dataframe",
            status="active",
            data_name=name,   # server-side filter
        )
        # Exact name match, then pick the highest version
        versions = df[df["name"] == name]
        if versions.empty:
            raise ValueError(f"No active dataset named {name!r}.")
        data_id = int(df.sort_values("version", ascending=False).iloc[0]["did"])
        # Fetch by data_id to avoid version ambiguity
        ds = fetch_openml(data_id=data_id, as_frame=True)
    df: pd.DataFrame = ds.frame.copy()
    # Drop index-like columns commonly found
    for col in df.columns:
        if str(col).lower() in {"id", "index"}:
            df = df.drop(columns=[col])
    # Identify a color/label column if provided and exists
    color_series = None
    if target is not None and target in df.columns:
        color_series = df[target]
    elif "class" in df.columns:
        color_series = df["class"]
    elif "target" in df.columns:
        color_series = df["target"]

    # 2) Infer types and coerce discrete to categorical
    disc_cols, cont_cols = infer_types(df)
    df = coerce_discrete_to_category(df, disc_cols)
    df = rename_categorical_categories_to_str(df, disc_cols)
    df = coerce_continuous_to_float(df, cont_cols)

    # If dataset ends up all one type, try to force a few low-card int columns to discrete
    if len(disc_cols) == 0:
        # Pick up to 3 lowest-card numeric to coerce
        cand = [c for c in df.columns if c not in disc_cols and pd.api.types.is_integer_dtype(df[c])]
        cand = sorted(cand, key=lambda c: df[c].nunique(dropna=True))[:3]
        df[cand] = df[cand].astype("category")
        disc_cols, cont_cols = infer_types(df)
        df = rename_categorical_categories_to_str(df, disc_cols)
        df = coerce_continuous_to_float(df, cont_cols)

    # 3) Baseline summary
    baseline = summarize_dataframe(df, disc_cols, cont_cols)
    baseline_md = dataframe_to_markdown_table(baseline)

    # 4) Train/test split (drop rows with any NA for BN; PyBNesian can filter via HoldOut too, but we keep it simple)
    df_no_na = df.dropna(axis=0, how="any").reset_index(drop=True)
    if len(df_no_na) < 100:
        print("Warning: very few complete rows; proceeding anyway.")
    train_df, test_df = train_test_split(df_no_na, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

    # 5) UMAP on real sample (fit once)
    print("Fitting UMAP on a large random sample...")
    umap_art = build_umap(df_no_na, disc_cols, cont_cols, color_series=df_no_na[color_series.name] if isinstance(color_series, pd.Series) and color_series.name in df_no_na.columns else None, rng=rng)
    umap_png_real = os.path.join(outdir, f"{name}_umap_real.png")
    plot_umap(umap_art.embedding, umap_png_real, title=f"{name}: real (sample)",
              color_labels=umap_art.color_labels)

    # 6) Learn CLG BN with PyBNesian (structure + params)
    print("Learning CLG BN (structure + params)...")
    bn_art = learn_clg_bn(train_df)
    model = bn_art.model
    node_types = model.node_types()

    # 7) Visualize BN
    print("Visualizing BN graph...")
    bn_png = os.path.join(outdir, f"{name}_bn.png")
    bn_to_graphviz(model, node_types, bn_png, title=f"{name} — CLG BN")

    # 8) Sample synthetic data
    print("Sampling synthetic data...")
    synth = model.sample(SYNTHETIC_SAMPLE, seed=RANDOM_STATE)  # returns a pyarrow RecordBatch
    synth_df = synth.to_pandas()  # convert to pandas for metrics/plots

    # 9) Fidelity metrics
    print("Computing fidelity metrics...")
    ll = heldout_loglik(model, test_df)
    # Align columns & dtypes for distance metrics
    synth_df = synth_df[df_no_na.columns]  # same column order
    # For discrete cols, ensure categorical with same categories
    for c in disc_cols:
        if c in synth_df.columns:
            synth_df[c] = synth_df[c].astype("category")
            # Align categories if possible
            if pd.api.types.is_categorical_dtype(df_no_na[c]):
                synth_df[c] = synth_df[c].cat.set_categories(df_no_na[c].cat.categories)
    dist_table = per_variable_distances(test_df, synth_df, bn_art.discrete_cols, bn_art.continuous_cols)

    # 10) UMAP of synthetic data (transform with same preproc + UMAP)
    print("Projecting synthetic data into same UMAP space...")
    # Note: UMAP transform expects same preprocessing. For safety, drop rows with NA produced by sampling (rare).
    synth_no_na = synth_df.dropna(axis=0, how="any")
    synth_emb = transform_with_umap(umap_art, synth_no_na)
    umap_png_synth = os.path.join(outdir, f"{name}_umap_synth.png")
    plot_umap(synth_emb, umap_png_synth, title=f"{name}: synthetic (BN sample)")

    # 11) Serialize
    print("Serializing model...")
    # Human-readable in report (below)
    bn_human = bn_human_readable(model)

    # GraphML (structure only; broad interoperability across graph tools)
    graphml_file = os.path.join(outdir, f"{name}_structure.graphml")
    save_graphml_structure(model, node_types, graphml_file)

    # Pickle full model (PyBNesian-native)
    pickle_file = os.path.join(outdir, f"{name}_model.pickle")
    model.save(pickle_file)  # includes graph; set include_cpd=True to embed CPDs in some versions

    # 12) Report
    write_report_md(
        outdir=outdir,
        dataset_name=name,
        df=df_no_na,
        disc_cols=disc_cols,
        cont_cols=cont_cols,
        baseline_md_table=baseline_md,
        umap_png_real=umap_png_real,
        umap_png_synth=umap_png_synth,
        bn_png=bn_png,
        bn_human=bn_human,
        ll_metrics=ll,
        dist_table=dist_table,
        graphml_file=graphml_file,
        pickle_file=pickle_file,
    )

def main():
    base_outdir = os.path.abspath("bn_reports")
    ensure_dir(base_outdir)
    rng = seed_all(RANDOM_STATE)

    for name, target in DATASETS:
        try:
            process_dataset(name, target, base_outdir, rng)
        except Exception as e:
            print(f"[WARN] Skipped {name} due to error: {e}")
            raise e

    print("\nDone. See the 'bn_reports/' folder for per-dataset subfolders with:")
    print("- Markdown report")
    print("- BN graph PNG")
    print("- UMAP plots (real + synthetic)")
    print("- GraphML structure")
    print("- Pickled BN model")

if __name__ == '__main__':
    main()
