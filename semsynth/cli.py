"""Command-line entry points for SemSynth."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, List, Optional, Literal, Tuple

import defopt

__all__ = ["search", "report", "main"]

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .models import ModelConfigBundle, ModelSpec
    from .specs import DatasetSpec


def search(
    provider: str,
    *,
    name_substr: Optional[str] = None,
    area: str = "Health and Medicine",
    cat_min: int = 1,
    num_min: int = 1,
    verbose: bool = False,
) -> None:
    """Search mixed-type datasets on OpenML or the UCI ML Repository.

    Args:
        provider: Either ``"openml"`` or ``"uciml"``.
        name_substr: Optional substring to filter dataset names (case-insensitive).
        area: Topic area for UCI ML datasets. Ignored for OpenML.
        cat_min: Minimum number of categorical columns.
        num_min: Minimum number of numeric columns.
        verbose: Whether to enable informational logging during execution.
    """
    from .datasets import list_openml, list_uciml  # defer heavy imports

    if verbose:
        logging.root.setLevel(logging.INFO)

    provider_key = provider.lower()
    if provider_key == "openml":
        df = list_openml(name_substr=name_substr, cat_min=cat_min, num_min=num_min)
    elif provider_key == "uciml":
        df = list_uciml(
            area=area,
            name_substr=name_substr,
            cat_min=cat_min,
            num_min=num_min,
        )
    else:
        raise SystemExit("provider must be 'openml' or 'uciml'")
    print(df.to_csv(sep="\t", index=None))


Toggle = Literal["auto", "on", "off"]


def _toggle_value(mode: Toggle) -> Tuple[Optional[bool], bool]:
    value = mode.lower()
    if value == "auto":
        return None, False
    if value == "on":
        return True, True
    if value == "off":
        return False, True
    raise ValueError(f"Unknown toggle mode: {mode}")


def report(
    provider: str = "openml",
    *,
    datasets: List[str] | None = None,
    outdir: str = "outputs",
    configs_yaml: str = "",
    area: str = "Health and Medicine",
    verbose: bool = False,
    generate_umap: Toggle = "auto",
    overwrite_umap: bool = False,
    compute_privacy: Toggle = "auto",
    compute_downstream: Toggle = "auto",
) -> None:
    """Run the report pipeline on a collection of datasets.

    Args:
        provider: Either ``"openml"`` or ``"uciml"``.
        datasets: Dataset identifiers. For OpenML, these are names; for UCI ML, they are
            numeric identifiers as strings. Defaults are used when omitted.
        outdir: Output directory for per-dataset reports.
        configs_yaml: Path to a YAML file defining synthetic data model configurations.
        area: Default topic area for UCI datasets.
        verbose: Whether to enable informational logging during execution.
        generate_umap: Control UMAP generation (``"auto"``, ``"on"``, or ``"off"``).
        overwrite_umap: Whether to regenerate synthetic UMAP plots when files exist.
        compute_privacy: Control privacy metrics computation (``"auto"``, ``"on"``,
            or ``"off"``).
        compute_downstream: Control downstream fidelity metrics computation
            (``"auto"``, ``"on"``, or ``"off"``).
    """
    from .datasets import DatasetSpec, load_dataset, specs_from_input
    from .models import ModelConfigBundle, load_model_configs
    from .pipeline import PipelineConfig, process_dataset
    from .utils import ensure_dir

    if verbose:
        logging.root.setLevel(logging.INFO)

    ensure_dir(outdir)

    dataset_specs: List[DatasetSpec]
    dataset_specs = specs_from_input(provider=provider, datasets=datasets, area=area)

    try:
        bundle: ModelConfigBundle = load_model_configs(
            configs_yaml.strip() or None
        )
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        raise SystemExit(str(exc))

    cfg = PipelineConfig()
    cfg.overwrite_umap = overwrite_umap
    toggles = {
        "generate_umap": generate_umap,
        "compute_privacy": compute_privacy,
        "compute_downstream": compute_downstream,
    }
    toggle_fields = {
        "generate_umap": "override_generate_umap",
        "compute_privacy": "override_compute_privacy",
        "compute_downstream": "override_compute_downstream",
    }
    for key, mode in toggles.items():
        explicit_value, is_explicit = _toggle_value(mode)
        if is_explicit:
            setattr(cfg, toggle_fields[key], explicit_value)

    for dataset_spec in dataset_specs:
        logging.info("Loading dataset %s", dataset_spec)
        try:
            meta, df, color = load_dataset(dataset_spec)
            process_dataset(
                meta,
                df,
                color,
                outdir,
                model_bundle=bundle,
                pipeline_config=cfg,
            )
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            logging.exception(
                "Skipped %s:%s due to error", dataset_spec.provider, dataset_spec.name
            )
            raise SystemExit(str(exc))


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point used by :mod:`defopt`."""

    defopt.run([search, report], argv=argv)


if __name__ == "__main__":
    main(sys.argv[1:])
