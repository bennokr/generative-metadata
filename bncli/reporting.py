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
    dataset_provider: Optional[str],
    dataset_provider_id: Optional[int],
    df: pd.DataFrame,
    disc_cols: List[str],
    cont_cols: List[str],
    baseline_df: pd.DataFrame,
    bn_sections: List[Dict],
    dist_table_meta: pd.DataFrame,
    fidelity_table: pd.DataFrame,
    graphml_files: Optional[List[str]],
    pickle_files: Optional[List[str]],
    roots_info: Optional[Dict],
    umap_png_real: str,
    umap_png_bns: List[str],
    umap_png_meta: str,
    metasyn_gmf_file: Optional[str] = None,
    declared_types: Optional[dict] = None,
    inferred_types: Optional[dict] = None,
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
        f.write(f"- Discrete: {len(disc_cols)}  |  Continuous: {len(cont_cols)}\n")
        # Provider-specific links
        if dataset_provider and dataset_provider_id:
            if dataset_provider == 'openml':
                url = f"https://www.openml.org/search?type=data&id={dataset_provider_id}"
                f.write(f"- OpenML page: {url}\n")
            elif dataset_provider == 'uciml':
                url = f"https://archive.ics.uci.edu/dataset/{dataset_provider_id}"
                f.write(f"- UCI ML page: {url}\n")
        f.write("\n")

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
        # Merge variables and baseline summary into one table
        baseline_out = baseline_df.round(4).reset_index().rename(columns={baseline_df.index.name or 'index': 'variable'})
        baseline_out = baseline_out.fillna("")
        # Build declared and inferred type tables (if provided)
        declared_df = None
        if isinstance(declared_types, dict) and declared_types:
            declared_df = pd.DataFrame(
                [{"variable": k, "declared": v} for k, v in declared_types.items()]
            )
        inferred_df = None
        if isinstance(inferred_types, dict) and inferred_types:
            inferred_df = pd.DataFrame(
                [{"variable": k, "inferred": v} for k, v in inferred_types.items()]
            )
        # Merge into baseline summary
        merged = baseline_out
        if declared_df is not None and not declared_df.empty:
            merged = declared_df.merge(merged, on="variable", how="right")
        if inferred_df is not None and not inferred_df.empty:
            merged = inferred_df.merge(merged, on="variable", how="right")
        f.write("## Variables and summary\n\n")
        f.write(df_to_markdown(merged, index=False) + "\n\n")
        f.write("## Learned BN structures and configurations\n\n")
        if roots_info:
            f.write("### Arc blacklist\n\n")
            sens = roots_info.get('root_variables')
            if sens:
                f.write("- Root variables: " + ", ".join(map(str, sens)) + "\n")
            nf = roots_info.get('n_forbidden_arcs')
            if nf is not None:
                f.write(f"- Forbidden arc count: {nf}\n\n")
        for i, sect in enumerate(bn_sections):
            label = sect.get("label") or sect.get("bn_type", f"BN{i+1}")
            bn_png = sect.get("bn_png")
            params = sect.get("params") or {}
            graphml_file = sect.get("graphml_file")
            pickle_file = sect.get("pickle_file")
            f.write(f"### {label}\n\n")
            # Config table
            if params:
                p_items = list(params.items())
                p_df = pd.DataFrame(p_items, columns=["param", "value"]).astype(str)
                f.write(p_df.to_markdown(index=False) + "\n\n")
            if bn_png:
                f.write(f"![BN graph]({Path(bn_png).name})\n\n")
            f.write("Serialization\n\n")
            if graphml_file:
                gname = Path(graphml_file).name
                f.write(f"- Structure (GraphML): [{gname}]({gname})\n")
            if pickle_file:
                pname = Path(pickle_file).name
                f.write(f"- Full model (pickle): [{pname}]({pname})\n")
            f.write("\n")
        if metasyn_gmf_file:
            mname = Path(metasyn_gmf_file).name
            f.write(f"MetaSyn GMF: [{mname}]({mname})\n\n")
        f.write("## Fidelity (BN vs MetaSyn)\n\n")
        # Add BN held-out likelihood as text for clarity as well
        # If available, include BN likelihoods in the table below; drop separate line
        if not fidelity_table.empty:
            f.write(df_to_markdown(fidelity_table.round(4).fillna(""), index=False) + "\n\n")

        f.write("### Per-variable distances (lower is closer)\n\n")
        # Build merged per-variable distances across BN types and MetaSyn
        if bn_sections:
            base = None
            parts = []
            for sect in bn_sections:
                dt = sect.get("dist_table")
                bnt = sect.get("label") or sect.get("bn_type", "BN")
                if dt is None or dt.empty:
                    continue
                if base is None:
                    base = dt[["variable", "type"]].copy()
                else:
                    base = base.merge(dt[["variable", "type"]], on=["variable", "type"], how="outer")
                cols = [c for c in ["JSD", "KS", "W1"] if c in dt.columns]
                part = dt[cols].copy()
                part.columns = pd.MultiIndex.from_tuples([ (bnt, c) for c in cols ])
                parts.append(part)
            # Add MetaSyn distances
            if isinstance(dist_table_meta, pd.DataFrame) and not dist_table_meta.empty:
                cols = [c for c in ["JSD", "KS", "W1"] if c in dist_table_meta.columns]
                meta_part = dist_table_meta[cols].copy()
                meta_part.columns = pd.MultiIndex.from_tuples([ ("MetaSyn", c) for c in cols])
                parts.append(meta_part)
            if base is not None and parts:
                comp_df = pd.concat([base] + parts, axis=1)
                # Reorder MultiIndex columns as (metric, model)
                mcols = [c for c in comp_df.columns if isinstance(c, tuple)]
                if mcols:
                    sub = comp_df.loc[:, mcols].copy()
                    new_cols = pd.MultiIndex.from_tuples(mcols).swaplevel(0, 1)
                    sub.columns = new_cols
                    # Order metrics as JSD, KS, W1 if present
                    desired = ["JSD", "KS", "W1"]
                    first_levels = [lvl for lvl in desired if lvl in sub.columns.levels[0]]
                    rest = [lvl for lvl in sub.columns.levels[0] if lvl not in first_levels]
                    order = first_levels + rest
                    sub = sub.reindex(columns=order, level=0)
                    comp_df = pd.concat([comp_df[["variable", "type"]], sub], axis=1)
                f.write(df_to_markdown(comp_df.round(4).fillna(""), index=False) + "\n\n")
        f.write("## UMAP overview (same projection)\n\n")
        # Dynamically build columns for UMAP images
        headers = ["Real (sample)", "MetaSyn (synthetic)"] + [f"BN: {sect.get('label') or sect.get('bn_type','')}" for sect in bn_sections]
        tbl = "| " + " | ".join(headers) + " |\n"
        tbl += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        imgs = [
            f"<img src='{Path(umap_png_real).name}' width='280'/>",
            f"<img src='{Path(umap_png_meta).name}' width='280'/>",
        ]
        imgs.extend([f"<img src='{Path(p).name}' width='280'/>" for p in umap_png_bns])
        tbl += "| " + " | ".join(imgs) + " |\n\n"
        f.write(tbl)
    logging.info(f"Wrote report: {md_path}")

    html_path = Path(outdir) / "index.html"
    with html_path.open('w', encoding='utf-8') as fw:
        fw.write(markdown.markdown(md_path.read_text(encoding='utf-8'), extensions=['extra']))
    logging.info(f"Converted to HTML: {html_path}")
