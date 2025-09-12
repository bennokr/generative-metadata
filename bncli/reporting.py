from __future__ import annotations

import os
from typing import Dict, List
import logging
import markdown

import pandas as pd

from .utils import dataframe_to_markdown_table


def write_report_md(
    outdir: str,
    dataset_name: str,
    metadata_file: str,
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
    md_path = os.path.join(outdir, f"report.md")
    num_rows, num_cols = df.shape
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Data Report — {dataset_name}\n\n")
        f.write(f"Metadata file: {metadata_file}\n")
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
        f.write(
            f"- Held-out total log-likelihood: {ll_metrics['sum_loglik']:.2f} over n={ll_metrics['n_rows']}\n\n"
        )
        f.write("### Per-variable distances (lower is closer)\n\n")
        if not dist_table.empty:
            f.write(dataframe_to_markdown_table(dist_table) + "\n\n")
        f.write("## UMAP on synthetic data (same projection)\n\n")
        f.write(f"![UMAP synthetic]({os.path.basename(umap_png_synth)})\n\n")
    logging.info(f"Wrote report: {md_path}")

    html_path = os.path.join(outdir, f"index.html")
    with open(html_path, 'w') as fw:
        fw.write(markdown.markdown(open(md_path).read(), extensions=['extra']))
    logging.info(f"Converted to HTML: {html_path}")