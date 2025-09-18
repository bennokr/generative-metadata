from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import html
import json
import logging
import os
import textwrap

import markdown

from jsonld_to_rdfa import SCHEMA_ORG, render_rdfa

import pandas as pd

from .synth import SynthReportRun


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def _resolve_semmap_context(data: Dict[str, Any]) -> Any:
    if not isinstance(data, dict):
        return SCHEMA_ORG
    if data.get("@context"):
        return data["@context"]
    dataset = data.get("dataset")
    if isinstance(dataset, dict) and dataset.get("@context"):
        return dataset["@context"]
    variables = data.get("disco:variable")
    if isinstance(variables, list):
        for entry in variables:
            if isinstance(entry, dict) and entry.get("@context"):
                return entry["@context"]
    return SCHEMA_ORG


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
    roots_info: Optional[Dict],
    umap_png_real: str,
    umap_png_bns: List[str],
    umap_png_meta: str,
    metasyn_gmf_file: Optional[str] = None,
    declared_types: Optional[dict] = None,
    inferred_types: Optional[dict] = None,
    variable_descriptions: Optional[dict] = None,
    semmap_jsonld: Optional[dict] = None,
    metasyn_semmap_parquet: Optional[str] = None,
    synth_runs: Optional[List[SynthReportRun]] = None,
) -> None:
    md_path = Path(outdir) / "report.md"
    semmap_fragment: Optional[str] = None
    semmap_html_name: Optional[str] = None
    if isinstance(semmap_jsonld, dict) and semmap_jsonld:
        try:
            context = _resolve_semmap_context(semmap_jsonld)
            html_title = f"{dataset_name} — SemMap metadata"
            semmap_fragment = render_rdfa(semmap_jsonld, context, html_title)
            semmap_html_path = md_path.parent / "dataset.semmap.html"
            semmap_html_path.write_text(
                semmap_fragment, encoding="utf-8"
            )
            semmap_html_name = semmap_html_path.name
            logging.info(f"Wrote SemMap metadata HTML: {semmap_html_path}")
        except Exception:
            logging.exception("Failed to render SemMap metadata HTML", exc_info=True)
            semmap_fragment = None
            semmap_html_name = None
    num_rows, num_cols = df.shape
    with md_path.open("w", encoding="utf-8") as f:
        def df_to_markdown(d: pd.DataFrame, index: bool = False) -> str:
            try:
                return d.to_markdown(index=index)
            except Exception:
                return d.to_string(index=index)
        f.write(f"# Data Report — {dataset_name}\n\n")
        
        # Provider-specific links
        if dataset_provider and dataset_provider_id:
            if dataset_provider == 'openml':
                url = f"https://www.openml.org/search?type=data&id={dataset_provider_id}"
                f.write(f"**Source**: [OpenML dataset {dataset_provider_id}]({url})\n")
            elif dataset_provider == 'uciml':
                url = f"https://archive.ics.uci.edu/dataset/{dataset_provider_id}"
                f.write(f"**Source**: [UCI dataset {dataset_provider_id}]({url})\n")
        f.write("\n")
        
        # Overview
        mf_name = Path(metadata_file).name
        f.write(f"- Metadata file: [{mf_name}]({mf_name})\n")
        if dataset_jsonld_file:
            jd_name = Path(dataset_jsonld_file).name
            f.write(f"- JSON-LD (schema.org/Dataset): [{jd_name}]({jd_name})\n")
        if semmap_jsonld:
            semmap_json_name = "dataset.semmap.json"
            if (md_path.parent / semmap_json_name).exists():
                f.write(f"- SemMap JSON-LD: [{semmap_json_name}]({semmap_json_name})\n")
            else:
                f.write(f"- SemMap JSON-LD: {semmap_json_name}\n")
        if semmap_html_name:
            f.write(f"- SemMap HTML: [{semmap_html_name}]({semmap_html_name})\n")
        f.write(f"- Rows: {num_rows}\n")
        f.write(f"- Columns: {num_cols}\n")
        f.write(f"- Discrete: {len(disc_cols)}  |  Continuous: {len(cont_cols)}\n")
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
        if semmap_fragment:
            f.write("\n## Metadata (rich)\n\n")
            if semmap_html_name:
                f.write(f"[Standalone SemMap metadata view]({semmap_html_name})\n\n")
            try:
                semmap_css = (_TEMPLATES_DIR / "semmap_style.css").read_text(encoding="utf-8")
            except Exception:
                semmap_css = ""
            f.write("<style>\n" + semmap_css + "\n</style>\n")
            f.write('<div class="semmap-metadata">\n')
            f.write(semmap_fragment)
            f.write("\n</div>\n\n")
        # Merge variables and baseline summary into one table
        baseline_out = baseline_df.round(4).reset_index().rename(columns={baseline_df.index.name or 'index': 'variable'})
        baseline_out = baseline_out.fillna("")
        # Build declared, inferred, and description tables (if provided)
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
        desc_df = None
        if isinstance(variable_descriptions, dict) and variable_descriptions:
            desc_df = pd.DataFrame(
                [{"variable": k, "description": v} for k, v in variable_descriptions.items()]
            )
        # Merge into baseline summary
        merged = baseline_out
        if declared_df is not None and not declared_df.empty:
            merged = declared_df.merge(merged, on="variable", how="right")
        if inferred_df is not None and not inferred_df.empty:
            merged = inferred_df.merge(merged, on="variable", how="right")
        if desc_df is not None and not desc_df.empty:
            merged = desc_df.merge(merged, on="variable", how="right")
        merged = merged.fillna("")
        
        

        f.write("## Variables and summary\n\n")
        f.write(df_to_markdown(merged, index=False) + "\n\n")

        if synth_runs:
            f.write("## Synthetic data (synthcity)\n\n")
            summary_rows = []
            for run in synth_runs:
                manifest = run.manifest or {}
                summary = run.metrics.get("summary", {}) if isinstance(run.metrics, dict) else {}
                summary_rows.append(
                    {
                        "generator": run.generator,
                        "rows": manifest.get("rows"),
                        "seed": manifest.get("seed"),
                        "disc_jsd_mean": summary.get("disc_jsd_mean"),
                        "disc_jsd_median": summary.get("disc_jsd_median"),
                        "cont_ks_mean": summary.get("cont_ks_mean"),
                        "cont_w1_mean": summary.get("cont_w1_mean"),
                    }
                )
            if summary_rows:
                synth_summary_df = pd.DataFrame(summary_rows)
                f.write(df_to_markdown(synth_summary_df, index=False) + "\n\n")
            for run in synth_runs:
                manifest = run.manifest or {}
                summary = run.metrics.get("summary", {}) if isinstance(run.metrics, dict) else {}
                f.write(f"### Generator: {run.generator}\n\n")
                requested = manifest.get("requested_generator", manifest.get("generator", run.generator))
                if requested and requested != run.generator:
                    f.write(f"- Requested alias: {requested}\n")
                f.write(f"- Seed: {manifest.get('seed')}\n")
                if manifest.get("rows") is not None:
                    f.write(f"- Rows: {manifest.get('rows')}\n")
                params = manifest.get("params") or {}
                if params:
                    f.write("- Params: `" + json.dumps(params, sort_keys=True) + "`\n")
                if summary:
                    stats = []
                    for key in ("disc_jsd_mean", "disc_jsd_median", "cont_ks_mean", "cont_w1_mean"):
                        val = summary.get(key)
                        if val is None:
                            continue
                        try:
                            val_f = float(val)
                        except (TypeError, ValueError):
                            continue
                        if not (val_f != val_f):
                            stats.append(f"{key}={val_f:.4f}")
                    if stats:
                        f.write("- Metrics: " + ", ".join(stats) + "\n")
                def _write_link(label: str, target: Optional[Path]) -> None:
                    if target is None:
                        return
                    if not target.exists():
                        return
                    rel = os.path.relpath(target, start=md_path.parent)
                    f.write(f"- {label}: [{rel}]({rel})\n")
                _write_link("Synthetic CSV", run.synthetic_csv)
                _write_link("Per-variable metrics", run.per_variable_csv)
                _write_link("Metrics JSON", run.run_dir / "metrics.json")
                if run.umap_png and run.umap_png.exists():
                    rel_png = os.path.relpath(run.umap_png, start=md_path.parent)
                    f.write(f"- UMAP: [{Path(rel_png).name}]({rel_png})\n")
                    f.write(f"\n![{run.generator} UMAP]({rel_png})\n\n")
                else:
                    f.write("\n")

        f.write("## Learned structures and configurations\n\n")
        if metasyn_gmf_file:
            mname = Path(metasyn_gmf_file).name
            f.write(f"MetaSyn GMF: [{mname}]({mname})\n\n")
        if metasyn_semmap_parquet:
            f.write("MetaSyn serialization\n\n")
            if metasyn_semmap_parquet:
                pname = Path(metasyn_semmap_parquet).name
                f.write(f"- Synthetic sample (SemMap Parquet): [{pname}]({pname})\n")
            f.write("\n")

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
            semmap_parquet = sect.get("semmap_parquet")
            if semmap_parquet:
                spath = Path(semmap_parquet).name
                f.write(f"- Synthetic sample (SemMap Parquet): [{spath}]({spath})\n")
            f.write("\n")
        
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
                part.columns = pd.MultiIndex.from_tuples([(bnt, c) for c in cols])
                parts.append(part)
            # Add MetaSyn distances
            if isinstance(dist_table_meta, pd.DataFrame) and not dist_table_meta.empty:
                cols = [c for c in ["JSD", "KS", "W1"] if c in dist_table_meta.columns]
                meta_part = dist_table_meta[cols].copy()
                meta_part.columns = pd.MultiIndex.from_tuples([("MetaSyn", c) for c in cols])
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
                    # Promote variable/type to MultiIndex columns for hierarchical headers
                    left = comp_df[["variable", "type"]].copy()
                    left.columns = pd.MultiIndex.from_tuples([(" ", "variable"), (" ", "type")])
                    comp_df = pd.concat([left, sub], axis=1)
                # Write a clean HTML table with hierarchical headers, grouped by metric
                html_table = comp_df.round(4).to_html(index=False, na_rep="", classes=["table", "per-var-dist"], border=0)
                f.write(html_table + "\n\n")
        
        # Dynamically build columns for UMAP images, including synthcity runs
        f.write("## UMAP overview (same projection)\n\n")
        synth_headers: List[str] = []
        synth_imgs: List[str] = []
        if isinstance(synth_runs, list) and synth_runs:
            for run in synth_runs:
                if run.umap_png and run.umap_png.exists():
                    synth_headers.append(f"Synth: {run.generator}")
                    synth_imgs.append(str(run.umap_png))
        headers = ["Real (sample)", "MetaSyn (synthetic)"] + [f"BN: {sect.get('label') or sect.get('bn_type','')}" for sect in bn_sections] + synth_headers
        tbl = "| " + " | ".join(headers) + " |\n"
        tbl += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        imgs = [
            f"<img src='{Path(umap_png_real).name}' width='280'/>",
            f"<img src='{Path(umap_png_meta).name}' width='280'/>",
        ]
        imgs.extend([f"<img src='{Path(p).name}' width='280'/>" for p in umap_png_bns])
        imgs.extend([f"<img src='{Path(p).name}' width='280'/>" for p in synth_imgs])
        tbl += "| " + " | ".join(imgs) + " |\n\n"
        f.write(tbl)
    logging.info(f"Wrote report: {md_path}")

    html_path = Path(outdir) / "index.html"
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["extra"])
    html_body = textwrap.indent(html_body, "    ")
    report_title = f"Data Report — {dataset_name}"
    try:
        css_text = (_TEMPLATES_DIR / "report_style.css").read_text(encoding="utf-8")
    except Exception:
        css_text = ""
    try:
        tpl = (_TEMPLATES_DIR / "report_template.html").read_text(encoding="utf-8")
    except Exception:
        tpl = "<!doctype html><html><head><title>{{TITLE}}</title><style>{{CSS}}</style></head><body><main class=\"report-container\">{{BODY}}</main></body></html>"
    html_out = tpl.replace("{{TITLE}}", html.escape(report_title)).replace("{{CSS}}", css_text).replace("{{BODY}}", html_body)
    html_path.write_text(html_out, encoding="utf-8")
    logging.info(f"Converted to HTML: {html_path}")
