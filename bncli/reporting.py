from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import logging
import markdown

import pandas as pd


def write_report_md(
    outdir: str,
    dataset_name: str,
    metadata_file: str,
    dataset_jsonld_file: Optional[str],
    dataset_jsonld: Optional[dict],
    df: pd.DataFrame,
    disc_cols: List[str],
    cont_cols: List[str],
    baseline_df: pd.DataFrame,
    bn_type: str,
    bn_png: str,
    bn_human: str,
    ll_metrics: Dict[str, float],
    dist_table_bn: pd.DataFrame,
    dist_table_meta: pd.DataFrame,
    fidelity_table: pd.DataFrame,
    graphml_file: str,
    pickle_file: str,
    umap_png_real: str,
    umap_png_bn: str,
    umap_png_meta: str,
    metasyn_gmf_file: Optional[str] = None,
) -> None:
    md_path = Path(outdir) / "report.md"
    num_rows, num_cols = df.shape
    with md_path.open("w", encoding="utf-8") as f:
        def df_to_markdown(d: pd.DataFrame, index: bool = False) -> str:
            try:
                return d.to_markdown(index=index)
            except Exception:
                return d.to_string(index=index)
        f.write(f"# Data Report — {dataset_name}\n\n")
        mf_name = Path(metadata_file).name
        f.write(f"- Metadata file: [{mf_name}]({mf_name})\n")
        if dataset_jsonld_file:
            jd_name = Path(dataset_jsonld_file).name
            f.write(f"- JSON-LD (schema.org/Dataset): [{jd_name}]({jd_name})\n")
        f.write(f"- Rows: {num_rows}\n")
        f.write(f"- Columns: {num_cols}\n")
        f.write(f"- Discrete: {len(disc_cols)}  |  Continuous: {len(cont_cols)}\n\n")

        if isinstance(dataset_jsonld, dict):
            f.write("## Dataset metadata\n\n")
            name = dataset_jsonld.get("name")
            if name and name != dataset_name:
                f.write(f"- Name: {name}\n")
            desc = dataset_jsonld.get("description")
            if desc:
                f.write("\n### Description\n\n")
                f.write(str(desc).strip() + "\n\n")
            creators = dataset_jsonld.get("creator") or dataset_jsonld.get("author")
            if creators:
                if isinstance(creators, dict):
                    creators = [creators]
                names = []
                for c in creators:
                    if isinstance(c, dict):
                        nm = c.get("name") or " ".join(filter(None, [c.get("givenName"), c.get("familyName")])).strip()
                        if nm:
                            names.append(nm)
                if names:
                    f.write("- Creators: " + ", ".join(names) + "\n")
            date_p = dataset_jsonld.get("datePublished") or dataset_jsonld.get("dateCreated")
            if date_p:
                f.write(f"- Date: {date_p}\n")
            citation = dataset_jsonld.get("citation")
            if citation:
                f.write("- Citation: " + (citation if isinstance(citation, str) else str(citation)) + "\n")
            urls = []
            if dataset_jsonld.get("url"):
                urls.append(("URL", dataset_jsonld.get("url")))
            if dataset_jsonld.get("sameAs"):
                sa = dataset_jsonld.get("sameAs")
                if isinstance(sa, (list, tuple)):
                    for u in sa:
                        urls.append(("sameAs", u))
                else:
                    urls.append(("sameAs", sa))
            if urls:
                f.write("- Links:\n")
                for label, u in urls:
                    f.write(f"  - {label}: {u}\n")
            vars_meas = dataset_jsonld.get("variableMeasured")
            if vars_meas:
                f.write("\n### Variables\n\n")
                rows = []
                if isinstance(vars_meas, dict):
                    vars_meas = [vars_meas]
                for item in vars_meas:
                    if isinstance(item, dict):
                        rows.append({
                            "variable": item.get("name", ""),
                            "measurement": item.get("measurementTechnique", ""),
                            "unit": item.get("unitText", "")
                        })
                if rows:
                    var_df = pd.DataFrame(rows)
                    var_df = var_df.fillna("")
                    f.write(df_to_markdown(var_df, index=False) + "\n\n")
        f.write("## Baseline summary\n\n")
        baseline_out = baseline_df.round(4).reset_index().rename(columns={baseline_df.index.name or 'index': 'variable'})
        baseline_out = baseline_out.fillna("")
        f.write(df_to_markdown(baseline_out, index=False) + "\n\n")
        f.write(f"## Learned {bn_type} BN (structure)\n\n")
        f.write(f"![BN graph]({Path(bn_png).name})\n\n")
        f.write(f"### Human-readable {bn_type} BN summary\n\n")
        f.write("```\n" + bn_human + "\n```\n\n")
        f.write("### Serialization\n\n")
        gname = Path(graphml_file).name
        pname = Path(pickle_file).name
        f.write(f"- Structure (GraphML): [{gname}]({gname})\n")
        f.write(f"- Full model (pickle): [{pname}]({pname})\n")
        if metasyn_gmf_file:
            mname = Path(metasyn_gmf_file).name
            f.write(f"- MetaSyn GMF: [{mname}]({mname})\n\n")
        else:
            f.write("\n")
        f.write("## Fidelity (BN vs MetaSyn)\n\n")
        # Add BN held-out likelihood as text for clarity as well
        f.write(
            f"Held-out BN log-likelihood per row: mean={ll_metrics['mean_loglik']:.4f}, std={ll_metrics['std_loglik']:.4f}; total={ll_metrics['sum_loglik']:.2f} over n={ll_metrics['n_rows']}\n\n"
        )
        if not fidelity_table.empty:
            f.write(df_to_markdown(fidelity_table.round(4).fillna(""), index=False) + "\n\n")

        f.write("### Per-variable distances (lower is closer)\n\n")
        if not dist_table_bn.empty and not dist_table_meta.empty:
            # Merge and build a MultiIndex column table for nested headers
            merged = dist_table_bn.merge(dist_table_meta, on=["variable", "type"], how="outer", suffixes=("_bn", "_meta"))
            base = merged[["variable", "type"]].copy()
            bn_cols = [c for c in ["JSD_bn", "KS_bn", "W1_bn"] if c in merged.columns]
            meta_cols = [c for c in ["JSD_meta", "KS_meta", "W1_meta"] if c in merged.columns]
            bn_part = merged[bn_cols].copy()
            bn_part.columns = pd.MultiIndex.from_tuples([("BN", c.replace("_bn", "")) for c in bn_cols])
            meta_part = merged[meta_cols].copy()
            meta_part.columns = pd.MultiIndex.from_tuples([("MetaSyn", c.replace("_meta", "")) for c in meta_cols])
            comp_df = pd.concat([base, bn_part, meta_part], axis=1)
            f.write(df_to_markdown(comp_df.round(4).fillna(""), index=False) + "\n\n")
        f.write("## UMAP overview (same projection)\n\n")
        tbl = "| Real (sample) | MetaSyn (synthetic) | BN (synthetic) |\n| --- | --- | --- |\n"
        tbl += f"| <img src='{Path(umap_png_real).name}' width='280'/> | <img src='{Path(umap_png_meta).name}' width='280'/> | <img src='{Path(umap_png_bn).name}' width='280'/> |\n\n"
        f.write(tbl)
    logging.info(f"Wrote report: {md_path}")

    html_path = Path(outdir) / "index.html"
    with html_path.open('w', encoding='utf-8') as fw:
        fw.write(markdown.markdown(md_path.read_text(encoding='utf-8'), extensions=['extra']))
    logging.info(f"Converted to HTML: {html_path}")
