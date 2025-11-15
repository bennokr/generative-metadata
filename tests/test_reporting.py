from __future__ import annotations

from pathlib import Path

import pandas as pd

from semsynth.models import ModelRun
from semsynth.reporting import write_report_md


def _touch(path: Path, content: str = "data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_write_report_md_renders_expected_sections(tmp_path: Path) -> None:
    df = pd.DataFrame({"age": [21, 35, 27], "status": ["yes", "no", "yes"]})
    run_dir = tmp_path / "models" / "demo"

    synthetic_csv = _touch(run_dir / "synthetic.csv")
    per_variable_csv = _touch(run_dir / "per_variable_metrics.csv")
    metrics_json = _touch(run_dir / "metrics.json", "{}")
    privacy_json = _touch(run_dir / "metrics.privacy.json", "{}")
    downstream_json = _touch(run_dir / "metrics.downstream.json", "{}")
    umap_png = _touch(run_dir / "umap.png", "binary")
    structure_png = _touch(run_dir / "structure.png", "binary")

    model_run = ModelRun(
        name="demo_model",
        backend="synth",
        run_dir=run_dir,
        synthetic_csv=synthetic_csv,
        per_variable_csv=per_variable_csv,
        metrics_json=metrics_json,
        metrics={"summary": {"disc_jsd_mean": 0.1, "cont_ks_mean": 0.2}},
        umap_png=umap_png,
        manifest={"seed": 123, "rows": 100, "params": {"alpha": 1}},
        privacy_json=privacy_json,
        privacy_metrics={"exact_overlap_rate": 0.05},
        downstream_json=downstream_json,
        downstream_metrics={"sign_match_rate": 0.9},
    )

    semmap_payload = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": "Demo dataset",
    }

    write_report_md(
        outdir=str(tmp_path),
        dataset_name="Demo dataset",
        dataset_provider="openml",
        dataset_provider_id=42,
        df=df,
        disc_cols=["status"],
        cont_cols=["age"],
        umap_png_real=str(_touch(tmp_path / "umap-real.png", "binary")),
        declared_types={"age": "integer"},
        inferred_types={"status": "categorical"},
        variable_descriptions={"status": "Approval status"},
        semmap_jsonld=semmap_payload,
        model_runs=[model_run],
    )

    markdown_path = tmp_path / "report.md"
    html_path = tmp_path / "index.html"
    semmap_json = tmp_path / "dataset.semmap.json"
    semmap_html = tmp_path / "dataset.semmap.html"

    assert markdown_path.exists()
    assert html_path.exists()
    assert semmap_json.exists()
    assert semmap_html.exists()

    md_text = markdown_path.read_text(encoding="utf-8")
    assert "# Data Report — Demo dataset" in md_text
    assert "**Source**: [OpenML dataset 42]" in md_text
    assert "SemMap JSON-LD" in md_text
    assert "## Variables and summary" in md_text
    assert "## Fidelity summary" in md_text
    assert "## Models" in md_text
    assert "models/demo/umap.png" in md_text
    assert "models/demo/structure.png" in md_text
    assert "Synthetic CSV" in md_text

    html_text = html_path.read_text(encoding="utf-8")
    assert "Data Report — Demo dataset" in html_text
    assert "report-container" in html_text
    assert "<table>" in html_text
