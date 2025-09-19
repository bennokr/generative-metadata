from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .models import model_run_root
from .backends import synthcity as backend_syn


@dataclass
class SynthReportRun:
    generator: str
    manifest: Dict[str, object]
    metrics: Dict[str, object]
    synthetic_csv: Path
    per_variable_csv: Optional[Path]
    umap_png: Optional[Path]
    run_dir: Path


def run_synth_experiment(
    df,
    *,
    provider: str,
    dataset_name: str,
    provider_id: Optional[int],
    outdir: str,
    generator: str,
    gen_params_json: str = "",
    rows: Optional[int] = None,
    seed: int = 0,
    test_size: float = 0.25,
    run_root: str | Path | None = None,
    dir_name: Optional[str] = None,
) -> Path:
    label = dir_name or generator
    params = json.loads(gen_params_json) if gen_params_json else {}
    return backend_syn.run_experiment(
        df=df,
        provider=provider,
        dataset_name=dataset_name,
        provider_id=provider_id,
        outdir=outdir,
        label=label,
        model_name=generator,
        params=params,
        rows=rows,
        seed=seed,
        test_size=test_size,
    )


def run_from_yaml(
    df,
    *,
    provider: str,
    dataset_name: str,
    provider_id: Optional[int],
    outdir: str,
    yaml_path: str,
    seed: int,
    test_size: float,
) -> List[Path]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load generator configs") from exc

    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "generators" in data:
        data = data["generators"]
    if not isinstance(data, list):
        raise ValueError("YAML must define a list of generators under 'generators'")

    outputs: List[Path] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Generator config at index {idx} must be a mapping")
        name = item.get("name")
        if not name:
            raise ValueError(f"Generator config at index {idx} missing 'name'")
        params = item.get("params", {})
        rows = item.get("rows")
        gen_seed = int(item.get("seed", seed))
        outputs.append(
            backend_syn.run_experiment(
                df=df,
                provider=provider,
                dataset_name=dataset_name,
                provider_id=provider_id,
                outdir=outdir,
                label=str(name),
                model_name=str(name),
                params=params,
                rows=rows,
                seed=gen_seed,
                test_size=test_size,
            )
        )
    return outputs


def discover_synth_runs(
    base_outdir: str | Path,
    *,
    provider: str,
    dataset_name: str,
) -> List[SynthReportRun]:
    root = Path(base_outdir) / "models"
    if not root.exists():
        return []
    runs: List[SynthReportRun] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        mpath = run_dir / "manifest.json"
        if not mpath.exists():
            continue
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(manifest.get("backend")).lower() != "synthcity":
            continue
        if str(manifest.get("provider", "")).lower() != str(provider).lower():
            continue
        if manifest.get("dataset_name") != dataset_name:
            continue
        metrics_path = run_dir / "metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        except Exception:
            metrics = {}
        per_var_path = run_dir / "per_variable_metrics.csv"
        umap_png = run_dir / "umap.png"
        runs.append(
            SynthReportRun(
                generator=str(manifest.get("generator", run_dir.name)),
                manifest=manifest,
                metrics=metrics,
                synthetic_csv=run_dir / "synthetic.csv",
                per_variable_csv=per_var_path if per_var_path.exists() else None,
                umap_png=umap_png if umap_png.exists() else None,
                run_dir=run_dir,
            )
        )
    runs.sort(key=lambda r: (r.generator, r.run_dir.name))
    return runs
