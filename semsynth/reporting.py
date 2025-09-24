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

from .models import ModelRun


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def _resolve_semmap_context(data: Dict[str, Any]) -> Any:
    if isinstance(data, dict) and data.get("@context"):
        return data["@context"]
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
    dist_table_meta: pd.DataFrame,
    fidelity_table: pd.DataFrame,
    umap_png_real: str,
    umap_png_meta: Optional[str],
    metasyn_gmf_file: Optional[str] = None,
    declared_types: Optional[dict] = None,
    inferred_types: Optional[dict] = None,
    variable_descriptions: Optional[dict] = None,
    semmap_jsonld: Optional[dict] = None,
    metasyn_semmap_parquet: Optional[str] = None,
    model_runs: Optional[List[ModelRun]] = None,
) -> None:
    md_path = Path(outdir) / "report.md"

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

        # Build compact per-variable summary with a single 'dist' column
        def _format_dist(col: pd.Series, *, top_n: int = 10) -> str:
            s = col.dropna()
            if col.name in cont_cols:
                try:
                    x = pd.to_numeric(s, errors="coerce")
                    mean = float(x.mean())
                    std = float(x.std())
                    q25 = float(x.quantile(0.25))
                    q50 = float(x.quantile(0.50))
                    q75 = float(x.quantile(0.75))
                    minv = float(x.min())
                    maxv = float(x.max())
                    def fmt(v: float) -> str:
                        txt = f"{v:.4f}".rstrip('0').rstrip('.')
                        return txt if txt else "0"
                    quantiles = ", ".join([fmt(minv), fmt(q25), fmt(q50), fmt(q75), fmt(maxv)])
                    return f"{mean:.4f} ± {std:.4f} [{quantiles}]"
                except Exception:
                    return ""
            # Discrete
            try:
                vc = s.astype(str).value_counts(dropna=True)
                total = float(vc.sum()) if vc.sum() else 1.0
            except Exception:
                return ""
            if len(vc) == 2:
                labels = list(vc.index)
                def is_true_label(v: str) -> bool:
                    t = str(v).strip().lower()
                    return t in {"true", "1", "yes", "y", "t"}
                pos_label = None
                for lab in labels:
                    if is_true_label(lab):
                        pos_label = lab
                        break
                if pos_label is None:
                    try:
                        nums = [float(x) for x in labels]
                        pos_label = labels[int(nums.index(max(nums)))]
                    except Exception:
                        pos_label = labels[0]
                n_true = int(vc.get(pos_label, 0))
                pct = 100.0 * (n_true / total)
                return f"{n_true} ({pct:.2f}%)"
            parts = []
            shown = 0
            for lab, cnt in vc.items():
                pct = 100.0 * (cnt / total)
                if shown < top_n:
                    parts.append(f"{lab}: {int(cnt)} ({pct:.2f}%)")
                shown += 1
            if shown > top_n:
                parts.append(f"… (+{shown - top_n} more)")
            return "\n".join(parts)

        var_rows = []
        for c in df.columns:
            typ = "continuous" if c in cont_cols else "discrete"
            var_rows.append({"variable": c, "type": typ, "dist": _format_dist(df[c])})
        baseline_out = pd.DataFrame(var_rows)
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

        if model_runs:
            # Unified fidelity summary combining all runs and optional MetaSyn
            f.write("## Fidelity summary\n\n")
            rows = []
            for run in model_runs:
                summary = run.metrics.get("summary", {}) if isinstance(run.metrics, dict) else {}
                rows.append({
                    "model": run.name,
                    "backend": run.backend,
                    "disc_jsd_mean": summary.get("disc_jsd_mean"),
                    "disc_jsd_median": summary.get("disc_jsd_median"),
                    "cont_ks_mean": summary.get("cont_ks_mean"),
                    "cont_w1_mean": summary.get("cont_w1_mean"),
                })
            # Add MetaSyn row if available
            if isinstance(dist_table_meta, pd.DataFrame) and not dist_table_meta.empty:
                d_disc = dist_table_meta[dist_table_meta['type'] == 'discrete'] if 'type' in dist_table_meta.columns else pd.DataFrame()
                d_cont = dist_table_meta[dist_table_meta['type'] == 'continuous'] if 'type' in dist_table_meta.columns else pd.DataFrame()
                rows.append({
                    "model": "MetaSyn",
                    "backend": "metasyn",
                    "disc_jsd_mean": float(d_disc['JSD'].mean()) if ('JSD' in d_disc.columns and len(d_disc)) else None,
                    "disc_jsd_median": float(d_disc['JSD'].median()) if ('JSD' in d_disc.columns and len(d_disc)) else None,
                    "cont_ks_mean": float(d_cont['KS'].mean()) if ('KS' in d_cont.columns and len(d_cont)) else None,
                    "cont_w1_mean": float(d_cont['W1'].mean()) if ('W1' in d_cont.columns and len(d_cont)) else None,
                })
            if rows:
                out = pd.DataFrame(rows)
                f.write(df_to_markdown(out.round(4).fillna(""), index=False) + "\n\n")
            # Model details (links, params)
            f.write("## Models\n\n")
            for run in model_runs:
                manifest = run.manifest or {}
                f.write(f"### Model: {run.name} ({run.backend})\n\n")
                f.write(f"- Seed: {manifest.get('seed')}\n")
                if manifest.get("rows") is not None:
                    f.write(f"- Rows: {manifest.get('rows')}\n")
                params = manifest.get("params") or {}
                if params:
                    f.write("- Params: `" + json.dumps(params, sort_keys=True) + "`\n")
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

                structure_png = run.run_dir / "structure.png"
                if structure_png.exists():
                    rel_png = os.path.relpath(structure_png, start=md_path.parent)
                    f.write(f"- Structure:\n  ![Structure of {run.name}]({rel_png})\n")
                f.write("\n")

        # MetaSyn files
        if metasyn_gmf_file:
            f.write("## MetaSyn\n\n")
            mname = Path(metasyn_gmf_file).name
            f.write(f"- GMF: [{mname}]({mname})\n")
            if metasyn_semmap_parquet:
                if metasyn_semmap_parquet:
                    pname = Path(metasyn_semmap_parquet).name
                    f.write(f"- Synthetic sample (SemMap Parquet): [{pname}]({pname})\n")
            f.write("\n")
        
        # Standalone fidelity table removed (merged into unified summary)

        # Per-variable distances table omitted in unified model view
        
        # Dynamically build columns for UMAP images, including additional model runs
        f.write("## UMAP overview (same projection)\n\n")
        extra_headers: List[str] = []
        extra_imgs: List[str] = []
        if isinstance(model_runs, list) and model_runs:
            for run in model_runs:
                if run.umap_png and run.umap_png.exists():
                    extra_headers.append(f"{run.backend}: {run.name}")
                    rel_png = os.path.relpath(run.umap_png, start=md_path.parent)
                    extra_imgs.append(rel_png)
        headers = ["Real (sample)"] + (["MetaSyn (synthetic)"] if umap_png_meta else []) + extra_headers
        tbl = "| " + " | ".join(headers) + " |\n"
        tbl += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        imgs = [
            f"<img src='{Path(umap_png_real).name}' width='280'/>",
        ]
        if umap_png_meta:
            imgs.append(f"<img src='{Path(umap_png_meta).name}' width='280'/>")
        imgs.extend([f"<img src='{p}' width='280'/>" for p in extra_imgs])
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
