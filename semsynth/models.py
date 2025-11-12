from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import logging


@dataclass
class ModelSpec:
    name: str
    backend: str  # 'pybnesian' or 'synthcity'
    model: Dict[str, Any] = field(default_factory=dict)
    rows: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class ModelRun:
    name: str
    backend: str
    run_dir: Path
    synthetic_csv: Path
    per_variable_csv: Optional[Path]
    metrics_json: Optional[Path]
    metrics: Dict[str, Any]
    umap_png: Optional[Path]
    manifest: Dict[str, Any]


def _as_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        if "configs" in data and isinstance(data["configs"], list):
            return data["configs"]  # type: ignore[return-value]
        if "generators" in data and isinstance(data["generators"], list):
            # Backward-compatible alias used by old synth-only YAMLs
            return data["generators"]  # type: ignore[return-value]
        # Single config object given as dict
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("Model config YAML must be a list or an object with 'configs'.")


def load_model_configs(yaml_path: str) -> List[ModelSpec]:
    """Load unified model configs from YAML"""
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("PyYAML is required to load configuration files") from exc

    path = Path(str(yaml_path))
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = _as_list(data)
    logging.info("Loading model configs from %s", path)
    specs: List[ModelSpec] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Config item at index {i} must be a mapping")
        name = str(item.get("name") or f"model_{i + 1}")
        backend = str(item.get("backend") or "pybnesian").strip().lower()
        model = item.get("model") or {}
        rows = item.get("rows")
        seed = item.get("seed")
        specs.append(
            ModelSpec(name=name, backend=backend, model=model, rows=rows, seed=seed)
        )
        logging.debug(
            "Loaded model spec: name=%s backend=%s rows=%s seed=%s",
            name,
            backend,
            rows,
            seed,
        )
    logging.info("Loaded %d model configs", len(specs))
    return specs


def model_run_root(dataset_outdir: Path) -> Path:
    root = dataset_outdir / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_run_dir(dataset_outdir: Path, name: str) -> Path:
    root = model_run_root(dataset_outdir)
    run_dir = root / str(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(run_dir: Path, manifest: Dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logging.debug("Wrote manifest to %s", run_dir / "manifest.json")


def discover_model_runs(dataset_outdir: str | Path) -> List[ModelRun]:
    root = Path(dataset_outdir) / "models"
    if not root.exists():
        logging.info("No model runs found under %s", root)
        return []
    runs: List[ModelRun] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        backend = str(manifest.get("backend") or "").lower()
        name = str(manifest.get("name") or run_dir.name)
        synthetic_csv = run_dir / "synthetic.csv"
        per_var = run_dir / "per_variable_metrics.csv"
        per_var_path = per_var if per_var.exists() else None
        metrics_json = run_dir / "metrics.json"
        metrics: Dict[str, Any] = {}
        if metrics_json.exists():
            try:
                metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
            except Exception:
                metrics = {}
        umap_png = run_dir / "umap.png"
        if not umap_png.exists():
            umap_png = None
        runs.append(
            ModelRun(
                name=name,
                backend=backend,
                run_dir=run_dir,
                synthetic_csv=synthetic_csv,
                per_variable_csv=per_var_path,
                metrics_json=metrics_json if metrics_json.exists() else None,
                metrics=metrics,
                umap_png=umap_png,
                manifest=manifest,
            )
        )
    names = [r.name for r in runs]
    logging.info("Discovered %d model runs under %s %s", len(runs), root, str(names))
    return runs
