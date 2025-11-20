"""Model configuration loading and run discovery utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from makeprov import OutPath, rule

if False:  # pragma: no cover - imported for type checking only
    from typing import TypedDict  # noqa: F401  # pylint: disable=unused-import


@dataclass
class ModelSpec:
    """Specification for invoking a backend model run."""

    name: str
    backend: str
    model: Dict[str, Any] = field(default_factory=dict)
    rows: Optional[int] = None
    seed: Optional[int] = None
    compute_privacy: Optional[bool] = None
    compute_downstream: Optional[bool] = None


@dataclass
class ModelConfigBundle:
    """Container bundling model specs together with global run flags."""

    specs: List[ModelSpec]
    generate_umap: Optional[bool] = None
    compute_privacy: Optional[bool] = None
    compute_downstream: Optional[bool] = None


@dataclass
class ModelRun:
    """Metadata describing an executed model run on disk."""

    name: str
    backend: str
    run_dir: Path
    synthetic_csv: Path
    per_variable_csv: Optional[Path]
    metrics_json: Optional[Path]
    metrics: Dict[str, Any]
    umap_png: Optional[Path]
    manifest: Dict[str, Any]
    privacy_json: Optional[Path]
    privacy_metrics: Dict[str, Any] = field(default_factory=dict)
    downstream_json: Optional[Path] = None
    downstream_metrics: Dict[str, Any] = field(default_factory=dict)


def _as_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        if isinstance(data.get("configs"), list):
            return list(data["configs"])
        if isinstance(data.get("generators"), list):
            return list(data["generators"])
        return [
            {
                k: v
                for k, v in data.items()
                if k not in {"generate_umap", "compute_privacy", "compute_downstream"}
            }
        ]
    if isinstance(data, list):
        return list(data)
    raise ValueError("Model config YAML must be a list or mapping with 'configs'.")


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "yes", "1", "on"}:
            return True
        if val in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"Cannot coerce value {value!r} to boolean")


def _extract_globals(data: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[bool]]]:
    if isinstance(data, dict):
        items = _as_list(data)
        globals_map = {
            key: _coerce_optional_bool(data.get(key))
            for key in ("generate_umap", "compute_privacy", "compute_downstream")
            if key in data
        }
        return items, globals_map
    items = _as_list(data)
    return items, {}


@rule()
def load_model_configs(
    yaml_path: Optional[str],
) -> ModelConfigBundle:
    """Load model specifications and global flags from YAML.

    If yaml_path is None, return an empty bundle (do not load package default).
    """

    if yaml_path is None:
        return ModelConfigBundle(specs=[], generate_umap=None, compute_privacy=None, compute_downstream=None)

    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to load configuration files") from exc

    resolved_path = Path(yaml_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    text = resolved_path.read_text(encoding="utf-8")
    data: Any = yaml.safe_load(text)
    if data is None:
        data = {}

    items, globals_map = _extract_globals(data)
    logging.info("Loading model configs from %s", resolved_path)
    specs: List[ModelSpec] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Config item at index {idx} must be a mapping")
        name = str(item.get("name") or f"model_{idx + 1}")
        backend = str(item.get("backend") or "pybnesian").strip().lower()
        model = item.get("model") or {}
        rows = item.get("rows")
        seed = item.get("seed")
        compute_privacy = _coerce_optional_bool(item.get("compute_privacy"))
        compute_downstream = _coerce_optional_bool(item.get("compute_downstream"))
        specs.append(
            ModelSpec(
                name=name,
                backend=backend,
                model=dict(model),
                rows=int(rows) if rows is not None else None,
                seed=int(seed) if seed is not None else None,
                compute_privacy=compute_privacy,
                compute_downstream=compute_downstream,
            )
        )
        logging.debug(
            "Loaded model spec: name=%s backend=%s rows=%s seed=%s", name, backend, rows, seed
        )
    logging.info("Loaded %d model configs", len(specs))
    return ModelConfigBundle(
        specs=specs,
        generate_umap=globals_map.get("generate_umap"),
        compute_privacy=globals_map.get("compute_privacy"),
        compute_downstream=globals_map.get("compute_downstream"),
    )


@rule()
def model_run_root(dataset_outdir: Path) -> Path:
    root = dataset_outdir / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


@rule()
def model_run_dir(
    dataset_outdir: Path,
    name: str
) -> Path:
    root = model_run_root(dataset_outdir)
    run_dir = root / str(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@rule()
def write_manifest(
    run_dir: Path,
    manifest: Dict[str, Any],
) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.debug("Wrote manifest to %s", run_dir / "manifest.json")


@rule()
def discover_model_runs(
    dataset_outdir: str | Path
) -> List[ModelRun]:
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
        metrics_json = run_dir / "metrics.json"
        privacy_json = run_dir / "metrics.privacy.json"
        downstream_json = run_dir / "metrics.downstream.json"
        metrics: Dict[str, Any] = {}
        privacy_metrics: Dict[str, Any] = {}
        downstream_metrics: Dict[str, Any] = {}
        for path, target in (
            (metrics_json, metrics),
            (privacy_json, privacy_metrics),
            (downstream_json, downstream_metrics),
        ):
            if path.exists():
                try:
                    target.update(json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    target.clear()

        umap_png = run_dir / "umap.png"
        if not umap_png.exists():
            umap_png = None

        runs.append(
            ModelRun(
                name=name,
                backend=backend,
                run_dir=run_dir,
                synthetic_csv=synthetic_csv,
                per_variable_csv=per_var if per_var.exists() else None,
                metrics_json=metrics_json if metrics_json.exists() else None,
                metrics=metrics,
                umap_png=umap_png,
                manifest=manifest,
                privacy_json=privacy_json if privacy_json.exists() else None,
                privacy_metrics=privacy_metrics,
                downstream_json=downstream_json if downstream_json.exists() else None,
                downstream_metrics=downstream_metrics,
            )
        )

    logging.info("Discovered %d model runs under %s", len(runs), root)
    return runs

