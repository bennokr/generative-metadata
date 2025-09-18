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


_REPORT_STYLE = """<style>
:root { color-scheme: light; }
body { margin: 0; font-family: "Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; background: #f4f7fb; color: #1f2937; line-height: 1.65; }
main.report-container { max-width: 1040px; margin: 0 auto; padding: 3rem 3rem 4rem; background: #ffffff; box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08); border-radius: 18px; }
h1 { font-size: 2.5rem; margin-top: 0; color: #0f172a; letter-spacing: -0.02em; }
h2 { margin-top: 2.75rem; padding-bottom: 0.35rem; border-bottom: 1px solid #e2e8f0; color: #1e293b; font-size: 1.6rem; }
h3 { margin-top: 2.2rem; color: #334155; font-size: 1.25rem; }
p { margin: 0.85rem 0; }
ul, ol { margin: 0.75rem 0 0.75rem 1.5rem; padding: 0; }
li { margin: 0.35rem 0; }
table { border-collapse: collapse; width: 100%; margin: 1.75rem 0; font-size: 0.95rem; }
th, td { border: 1px solid #e2e8f0; padding: 0.6rem 0.75rem; text-align: left; vertical-align: top; }
thead th { background: #f8fafc; text-transform: uppercase; letter-spacing: 0.04em; font-size: 0.75rem; color: #475569; }
tbody tr:nth-child(odd) { background: #fdfefe; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: #f1f5f9; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.85em; }
img { border-radius: 12px; box-shadow: 0 4px 28px rgba(15, 23, 42, 0.12); }
blockquote { border-left: 4px solid #cbd5f5; margin: 1.5rem 0; padding: 0.75rem 1.25rem; background: #f8fafc; color: #1e293b; }
table.per-var-dist { margin-top: 2.25rem; }
table.per-var-dist thead th { text-align: center; }
</style>"""


_SEMMAP_STYLE = """<style>
.semmap-metadata { display: grid; gap: 1.5rem; margin: 1.5rem 0; }
.semmap-metadata .item { border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.25rem 1.5rem; background: linear-gradient(180deg, rgba(248, 250, 252, 0.6), #ffffff); box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05); }
.semmap-metadata .item-title { margin: 0 0 .75rem; font-size: 1.1rem; color: #0f172a; }
.semmap-metadata .prop { margin: .3rem 0; color: #334155; }
.semmap-metadata .name { font-weight: 600; margin-right: .35rem; color: #0f172a; }
.semmap-metadata .prop-table { border-collapse: collapse; width: 100%; margin: .35rem 0 1rem; font-size: .95rem; }
.semmap-metadata .prop-table th, .semmap-metadata .prop-table td { border: 1px solid #e2e8f0; padding: .45rem .6rem; text-align: left; }
.semmap-metadata .prop-table th { background: #f8fafc; color: #475569; text-transform: uppercase; letter-spacing: .04em; font-size: .7rem; }
</style>"""


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
            f.write(_SEMMAP_STYLE + "\n")
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
        
        

        f.write("## Variables and summary

")
        f.write(df_to_markdown(merged, index=False) + "

")

        if synth_runs:
            f.write("## Synthetic data (synthcity)

")
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
                f.write(df_to_markdown(synth_summary_df, index=False) + "

")
            for run in synth_runs:
                manifest = run.manifest or {}
                summary = run.metrics.get("summary", {}) if isinstance(run.metrics, dict) else {}
                f.write(f"### Generator: {run.generator}

")
                requested = manifest.get("requested_generator", manifest.get("generator", run.generator))
                if requested and requested != run.generator:
                    f.write(f"- Requested alias: {requested}
")
                f.write(f"- Seed: {manifest.get('seed')}
")
                if manifest.get("rows") is not None:
                    f.write(f"- Rows: {manifest.get('rows')}
")
                params = manifest.get("params") or {}
                if params:
                    f.write("- Params: `" + json.dumps(params, sort_keys=True) + "`
")
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
                        f.write("- Metrics: " + ", ".join(stats) + "
")
                def _write_link(label: str, target: Optional[Path]) -> None:
                    if target is None:
                        return
                    if not target.exists():
                        return
                    rel = os.path.relpath(target, start=md_path.parent)
                    f.write(f"- {label}: [{rel}]({rel})
")
                _write_link("Synthetic CSV", run.synthetic_csv)
                _write_link("Per-variable metrics", run.per_variable_csv)
                _write_link("Metrics JSON", run.run_dir / "metrics.json")
                if run.umap_png and run.umap_png.exists():
                    rel_png = os.path.relpath(run.umap_png, start=md_path.parent)
                    f.write(f"- UMAP: [{Path(rel_png).name}]({rel_png})
")
                    f.write(f"
![{run.generator} UMAP]({rel_png})

")
                else:
                    f.write("
")

        f.write("## Learned structures and configurations

")
        f.write("## Learned structures and configurations

")
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
        
        # Dynamically build columns for UMAP images
        f.write("## UMAP overview (same projection)\n\n")
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
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["extra"])
    html_body = textwrap.indent(html_body, "    ")
    report_title = f"Data Report — {dataset_name}"
    html_path.write_text(
        (
            "<!doctype html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\">\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"  <title>{html.escape(report_title)}</title>\n"
            f"  {_REPORT_STYLE}\n"
            "</head>\n"
            "<body>\n"
            "  <main class=\"report-container\">\n"
            f"{html_body}\n"
            "  </main>\n"
            "</body>\n"
            "</html>\n"
        ),
        encoding="utf-8",
    )
    logging.info(f"Converted to HTML: {html_path}")
