"""Utilities for rendering dataset reports.

The reporting workflow produces a Markdown summary and a corresponding HTML
document for a dataset. The workflow follows three major steps:

1. Assemble an in-memory context describing the dataset, variable summaries and
   model runs.
2. Render Markdown content from a Jinja template using the assembled context.
3. Convert the Markdown output to HTML and embed it inside a static page
   template for distribution.

This module exposes :func:`write_report_md` as the high-level entry point and
contains helpers for computing summary tables, templating and distribution
formatting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import json
import logging
import os
import textwrap

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup
import pandas as pd

from .jsonld_to_rdfa import SCHEMA_ORG, render_rdfa
from .models import ModelRun


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


@dataclass
class _ModelLink:
    """Represents a hyperlink entry in a model's report row."""

    label: str
    href: str


@dataclass
class _ModelRow:
    """Pre-rendered data describing a model run section in the report."""

    name: str
    backend: str
    seed: Any
    rows: Any
    params_json: Optional[str]
    umap_rel: Optional[str]
    structure_rel: Optional[str]
    links: Sequence[_ModelLink]
    missingness: Optional[Dict[str, Any]] = None


def _jinja_environment() -> Environment:
    """Return a configured Jinja environment for report templates."""

    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(("html", "xml"), default=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _resolve_semmap_context(data: Dict[str, Any]) -> Any:
    if isinstance(data, dict) and data.get("@context"):
        return data["@context"]
    return SCHEMA_ORG


def _write_semmap_artifacts(
    output_dir: Path, dataset_name: str, semmap_jsonld: Optional[dict]
) -> tuple[Optional[str], Optional[str]]:
    """Persist SemMap metadata artifacts and return filenames.

    Args:
        output_dir: Target directory for report files.
        dataset_name: Dataset label used for the HTML title.
        semmap_jsonld: JSON-LD payload representing semantic metadata.

    Returns:
        Tuple containing the JSON filename and the rendered HTML filename. Each
        entry can be ``None`` when the corresponding artifact could not be
        generated.
    """

    if not isinstance(semmap_jsonld, dict) or not semmap_jsonld:
        return None, None

    json_name = "dataset.semmap.json"
    json_path = output_dir / json_name
    json_path.write_text(json.dumps(semmap_jsonld, indent=2), encoding="utf-8")

    html_name: Optional[str] = None
    try:
        context = _resolve_semmap_context(semmap_jsonld)
        html_title = f"{dataset_name} — SemMap metadata"
        semmap_fragment = render_rdfa(semmap_jsonld, context, html_title)
        html_path = output_dir / "dataset.semmap.html"
        html_path.write_text(semmap_fragment, encoding="utf-8")
        html_name = html_path.name
        logging.info("Wrote SemMap metadata HTML: %s", html_path)
    except Exception:  # pragma: no cover - log and continue report generation
        logging.exception("Failed to render SemMap metadata HTML", exc_info=True)
    return json_name, html_name


def write_report_md(
    outdir: str,
    dataset_name: str,
    # metadata_file: str,
    # dataset_jsonld_file: Optional[str],
    dataset_provider: Optional[str],
    dataset_provider_id: Optional[int],
    df: "pd.DataFrame",
    disc_cols: List[str],
    cont_cols: List[str],
    umap_png_real: Optional[str] = None,
    declared_types: Optional[dict] = None,
    inferred_types: Optional[dict] = None,
    variable_descriptions: Optional[dict] = None,
    semmap_jsonld: Optional[dict] = None,
    dataset_metadata: Optional[Any] = None,
    model_runs: Optional[List[ModelRun]] = None,
    missingness_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Render Markdown and HTML reports for a dataset.

    Args:
        outdir: Directory where report files should be written.
        dataset_name: Human readable dataset title used in headings.
        dataset_provider: External provider identifier (e.g. ``"openml"``).
        dataset_provider_id: Provider specific identifier for linking back.
        df: Source dataframe containing the analysed dataset.
        disc_cols: Names of discrete variables used for summary statistics.
        cont_cols: Names of continuous variables used for numeric summaries.
        umap_png_real: Optional path to the real-data UMAP image.
        declared_types: Mapping of declared data types per variable.
        inferred_types: Mapping of inferred data types per variable.
        variable_descriptions: Optional free text descriptions for variables.
        semmap_jsonld: Semantic metadata to attach to the report.
        model_runs: Model execution metadata used to build model sections.
        missingness_summary: Optional description of fitted missingness rates used
            for contextual reporting.
    """

    import markdown
    import pandas as pd

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "report.md"

    semmap_json_name, semmap_html_name = _write_semmap_artifacts(
        output_dir, dataset_title, semmap_jsonld
    )

    metadata_dict: Dict[str, Any] = {}
    if dataset_metadata is not None:
        if hasattr(dataset_metadata, "to_jsonld"):
            metadata_dict = dataset_metadata.to_jsonld() or {}
        elif isinstance(dataset_metadata, dict):
            metadata_dict = dataset_metadata

    dataset_description = metadata_dict.get("dcterms:description") or metadata_dict.get("description")
    dataset_purpose = metadata_dict.get("dcterms:purpose") or metadata_dict.get("purpose")
    dataset_table = metadata_dict.get("dcterms:tableOfContents") or metadata_dict.get("tableOfContents")
    dataset_title = metadata_dict.get("dcterms:title") or metadata_dict.get("title") or dataset_name
    dataset_citation = metadata_dict.get("schema:citation") or metadata_dict.get("citation")

    num_rows, num_cols = df.shape
    model_rows = _prepare_model_rows(model_runs or [], output_dir)
    variable_table = _build_variable_summary(
        df=df,
        cont_cols=cont_cols,
        declared_types=declared_types,
        inferred_types=inferred_types,
        variable_descriptions=variable_descriptions,
    )
    fidelity_table = _build_fidelity_table(model_runs=model_runs)
    missingness_table = _build_missingness_table(missingness_summary)

    env = _jinja_environment()
    template = env.get_template("report.md.j2")
    md_text = template.render(
        dataset_name=dataset_title,
        provider_link=_provider_link(dataset_provider, dataset_provider_id),
        semmap_json_name=semmap_json_name,
        semmap_html_name=semmap_html_name,
        num_rows=num_rows,
        num_cols=num_cols,
        num_disc=len(disc_cols),
        num_cont=len(cont_cols),
        variable_table=variable_table,
        fidelity_table=fidelity_table,
        model_rows=model_rows,
        real_umap=os.path.basename(umap_png_real) if umap_png_real else None,
        missingness_summary=missingness_summary,
        missingness_table=missingness_table,
        dataset_description=dataset_description,
        dataset_purpose=dataset_purpose,
        dataset_table=dataset_table,
        dataset_citation=dataset_citation,
    )
    md_path.write_text(md_text, encoding="utf-8")
    logging.info("Wrote report: %s", md_path)

    html_path = output_dir / "index.html"
    html_body = markdown.markdown(md_text, extensions=["extra", "md_in_html"])
    html_body = textwrap.indent(html_body, "    ")
    report_title = f"Data Report — {dataset_title}"
    css_text = _read_template_text("report_style.css")
    html_template = _load_html_template(env)
    html_out = html_template.render(
        TITLE=report_title,
        CSS=Markup(css_text),
        BODY=Markup(html_body),
    )
    html_path.write_text(html_out, encoding="utf-8")
    logging.info("Converted to HTML: %s", html_path)


def _format_dist(col: pd.Series, cont_cols: List[str], *, top_n: int = 10) -> str:
    """Return a concise distribution summary for a column."""

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
                txt = f"{v:.4f}".rstrip("0").rstrip(".")
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


def _build_variable_summary(
    df: pd.DataFrame,
    cont_cols: Iterable[str],
    *,
    declared_types: Optional[dict],
    inferred_types: Optional[dict],
    variable_descriptions: Optional[dict],
) -> str:
    """Create the Markdown table summarising variables."""

    cont_list = list(dict.fromkeys(cont_cols))
    var_rows = [
        {"variable": column, "dist": _format_dist(df[column], cont_list)}
        for column in df.columns
    ]
    baseline = pd.DataFrame(var_rows).fillna("")

    merged = baseline
    for mapping, label in (
        (declared_types, "declared"),
        (inferred_types, "inferred"),
        (variable_descriptions, "description"),
    ):
        if isinstance(mapping, dict) and mapping:
            other = pd.DataFrame(
                [{"variable": key, label: value} for key, value in mapping.items()]
            )
            merged = other.merge(merged, on="variable", how="right").fillna("")

    return _dataframe_to_markdown(merged, index=False)


def _build_fidelity_table(*, model_runs: Optional[Sequence[ModelRun]]) -> Optional[str]:
    """Create the markdown representation of the fidelity table."""

    runs = sorted(model_runs or [], key=lambda run: (run.backend, run.name))
    if not runs:
        return None

    rows: List[Dict[str, Any]] = []
    for run in runs:
        summary = run.metrics.get("summary", {}) if isinstance(run.metrics, dict) else {}
        rows.append(
            {
                "model": run.name,
                "backend": run.backend,
                "disc_jsd_mean": summary.get("disc_jsd_mean"),
                "disc_jsd_median": summary.get("disc_jsd_median"),
                "cont_ks_mean": summary.get("cont_ks_mean"),
                "cont_w1_mean": summary.get("cont_w1_mean"),
                "privacy_overlap": run.privacy_metrics.get("exact_overlap_rate"),
                "downstream_sign_match": run.downstream_metrics.get("sign_match_rate"),
            }
        )

    if not rows:
        return None

    out = pd.DataFrame(rows)
    return _dataframe_to_markdown(out.round(4).fillna(""), index=False)


def _build_missingness_table(summary: Optional[Dict[str, Any]]) -> Optional[str]:
    """Create a markdown table describing modeled missingness rates."""

    if not summary:
        return None

    rows = summary.get("rows") or []
    if not rows:
        return None

    out = pd.DataFrame(rows)
    if "missing_rate" not in out.columns or "column" not in out.columns:
        return None

    out = out.sort_values("missing_rate", ascending=False).reset_index(drop=True)
    out["missing_rate"] = out["missing_rate"].round(4)
    out["missing_pct"] = (out["missing_rate"] * 100).round(2)
    out.rename(
        columns={
            "column": "Column",
            "missing_rate": "Missing rate",
            "missing_pct": "Missing %",
        },
        inplace=True,
    )
    return _dataframe_to_markdown(out[["Column", "Missing rate", "Missing %"]], index=False)


def _prepare_model_rows(model_runs: Iterable[ModelRun], base_dir: Path) -> List[_ModelRow]:
    """Convert model runs into template-friendly rows."""

    prepared: List[_ModelRow] = []
    for run in sorted(model_runs, key=lambda r: (r.backend, r.name)):
        manifest = run.manifest or {}
        params = manifest.get("params") or {}
        params_json = json.dumps(params, sort_keys=True) if params else None

        links = []
        for label, target in (
            ("Synthetic CSV", run.synthetic_csv),
            ("Per-variable metrics", run.per_variable_csv),
            ("Metrics JSON", run.metrics_json or run.run_dir / "metrics.json"),
            ("Privacy metrics", run.privacy_json),
            ("Downstream metrics", run.downstream_json),
        ):
            rel = _relative_path(target, base_dir)
            if rel:
                links.append(_ModelLink(label=label, href=rel))

        prepared.append(
            _ModelRow(
                name=run.name,
                backend=run.backend,
                seed=manifest.get("seed"),
                rows=manifest.get("rows"),
                params_json=params_json,
                umap_rel=_relative_path(run.umap_png, base_dir),
                structure_rel=_relative_path(run.run_dir / "structure.png", base_dir),
                links=tuple(links),
                missingness=manifest.get("missingness"),
            )
        )
    return prepared


def _relative_path(target: Optional[Path], base_dir: Path) -> Optional[str]:
    """Return a relative path when the target exists."""

    if target is None:
        return None
    target_path = Path(target)
    if not target_path.exists():
        return None
    return os.path.relpath(target_path, start=base_dir)


def _provider_link(
    dataset_provider: Optional[str], dataset_provider_id: Optional[int]
) -> Optional[Dict[str, str]]:
    """Build a provider hyperlink descriptor for the template."""

    if not dataset_provider or not dataset_provider_id:
        return None
    provider = dataset_provider.lower()
    if provider == "openml":
        url = f"https://www.openml.org/search?type=data&id={dataset_provider_id}"
        text = f"OpenML dataset {dataset_provider_id}"
    elif provider == "uciml":
        url = f"https://archive.ics.uci.edu/dataset/{dataset_provider_id}"
        text = f"UCI dataset {dataset_provider_id}"
    else:
        return None
    return {"url": url, "text": text}


def _read_template_text(template_name: str) -> str:
    """Load template text from disk, returning an empty string on failure."""

    try:
        return (_TEMPLATES_DIR / template_name).read_text(encoding="utf-8")
    except Exception:  # pragma: no cover - optional styling
        logging.debug("Failed to read template %s", template_name, exc_info=True)
        return ""


def _load_html_template(env: Environment):
    """Load the HTML template with a fallback inline template."""

    try:
        return env.get_template("report_template.html")
    except Exception:  # pragma: no cover - fallback path
        logging.debug("Falling back to inline HTML template", exc_info=True)
        return env.from_string(
            "<!doctype html><html><head><title>{{TITLE}}</title>"
            "<style>{{CSS}}</style></head><body><main class=\"report-container\">"
            "{{BODY}}</main></body></html>"
        )


def _dataframe_to_markdown(df: pd.DataFrame, *, index: bool) -> str:
    """Convert a dataframe to Markdown with a string fallback."""

    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)
