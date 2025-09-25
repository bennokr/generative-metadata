from __future__ import annotations

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import umap

@dataclass
class UMAPArtifacts:
    preproc: object
    umap_model: object
    sample_idx: np.ndarray
    embedding: np.ndarray
    label_mapping: Optional[Dict]
    color_labels: Optional[np.ndarray]


def pick_color_labels(series: Optional[pd.Series]) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    if series is None:
        return None, None
    values = series.astype("category")
    cats = list(values.cat.categories)
    mapping = {cat: i for i, cat in enumerate(cats)}
    labels = values.map(mapping).to_numpy()
    return labels, mapping


def build_umap(
    df: pd.DataFrame,
    discrete_cols: List[str],
    continuous_cols: List[str],
    color_series: Optional[pd.Series],
    rng: np.random.Generator,
    random_state: int = 42,
    max_sample: int = 1000,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    n_components: int = 2,
) -> UMAPArtifacts:
    cont_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=True, with_std=True))]
    )
    disc_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[("cont", cont_pipe, continuous_cols), ("disc", disc_pipe, discrete_cols)],
        remainder="drop",
        sparse_threshold=1.0,
    )

    n = len(df)
    sample_n = min(max_sample, n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    df_sample = df.iloc[sample_idx].reset_index(drop=True)

    X = preproc.fit_transform(df_sample)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric="euclidean",
        verbose=False,
    )
    embedding = reducer.fit_transform(X)

    labels, mapping = pick_color_labels(color_series.iloc[sample_idx] if color_series is not None else None)
    return UMAPArtifacts(
        preproc=preproc,
        umap_model=reducer,
        sample_idx=sample_idx,
        embedding=embedding,
        label_mapping=mapping,
        color_labels=labels,
    )


def transform_with_umap(art: UMAPArtifacts, df: pd.DataFrame) -> np.ndarray:
    X = art.preproc.transform(df)
    return art.umap_model.transform(X)


def plot_umap(embedding: np.ndarray, outfile: str, title: str, color_labels: Optional[np.ndarray] = None, lims: Optional[Tuple[Tuple[float,float], Tuple[float,float]]] = None) -> None:
    fig = plt.figure(figsize=(7, 6), dpi=140)
    if color_labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=4, alpha=0.6)
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=4, alpha=0.6, c=color_labels)
    plt.title(title)
    if lims:
        xlim, ylim = lims
        for ax in fig.axes:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    for ax in fig.axes:
        return ax.get_xlim(), ax.get_ylim()
