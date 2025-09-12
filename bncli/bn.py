from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Iterable

import numpy as np
import networkx as nx
from graphviz import Digraph

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
    arc_blacklist: Optional[Sequence[Tuple[str, str]]] = None,
    *,
    score: Optional[str] = None,
    operators: Optional[Iterable[str]] = None,
    max_indegree: Optional[int] = None,
) -> BNArtifacts:
    bn_type_obj = _bn_type_from_str(bn_type)
    # Defaults if not provided
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


def bn_human_readable(model) -> str:
    node_types = model.node_types()
    lines = []
    lines.append("Nodes and types:")
    for n in model.nodes():
        lines.append(f"  - {n}: {type(node_types[n]).__name__}")
    lines.append("\nArcs:")
    for u, v in model.arcs():
        lines.append(f"  - {u} -> {v}")
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
                if beta is None:
                    beta = np.array([])
                beta_str = np.array2string(np.asarray(beta), precision=3, floatmode="fixed")
                if variance is not None:
                    lines.append(
                        f"  - {n}: LinearGaussian | parents={list(ev)} | beta={beta_str} | var={variance:.4f}"
                    )
                else:
                    lines.append(
                        f"  - {n}: LinearGaussian | parents={list(ev)} | beta={beta_str}"
                    )
            except Exception:
                lines.append(f"  - {n}: LinearGaussian | parents={list(ev)}")
        else:
            lines.append(f"  - {n}: {cname} | parents={list(ev)}")
    return "\n".join(lines)


def save_graphml_structure(model, node_types: Dict[str, object], out_graphml: str) -> None:
    G = nx.DiGraph()
    for n in model.nodes():
        ftype = node_types[n]
        G.add_node(n, bn_type=type(ftype).__name__)
    for u, v in model.arcs():
        G.add_edge(u, v)
    nx.write_graphml(G, out_graphml)
