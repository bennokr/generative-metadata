from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import logging


DEFAULT_CONFIG_PATH = Path("configs/default_config.yaml")


@dataclass
class ModelSpec:
    name: str
    backend: str  # 'pybnesian' or 'synthcity'
    model: Dict[str, Any]
    rows: Optional[int] = None
    seed: Optional[int] = None


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


def load_model_configs(yaml_path: Optional[str]) -> List[ModelSpec]:
    """Load unified model configs from YAML; fall back to default_config.yaml."""
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("PyYAML is required to load configuration files") from exc

    path = None
    if yaml_path and str(yaml_path).strip():
        path = Path(str(yaml_path))
    else:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = _as_list(data)
    logging.info("Loading model configs from %s", path)
    specs: List[ModelSpec] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Config item at index {i} must be a mapping")
        name = str(item.get("name") or f"model_{i+1}")
        backend = str(item.get("backend") or "pybnesian").strip().lower()
        model = item.get("model") or {}
        rows = item.get("rows")
        seed = item.get("seed")
        specs.append(ModelSpec(name=name, backend=backend, model=model, rows=rows, seed=seed))
        logging.debug("Loaded model spec: name=%s backend=%s rows=%s seed=%s", name, backend, rows, seed)
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
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.debug("Wrote manifest to %s", run_dir / "manifest.json")
