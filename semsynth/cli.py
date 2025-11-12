from __future__ import annotations

import sys
from typing import List, Optional
import logging

import defopt

from .datasets import (
    list_openml,
    list_uciml,
    load_dataset,
    specs_from_input,
)
from .models import load_model_configs, ModelSpec
from .utils import ensure_dir
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
    """Search mixed-type datasets on OpenML or UCI ML Repo.

    Parameters:
        provider: 'openml' or 'uciml'.
        name_substr: Optional substring to filter dataset names (case-insensitive).
        area: For 'uciml', the Area to search (default 'Health and Medicine'). Ignored for 'openml'.
        cat_min: Minimum number of categorical columns
        num_min: Minimum number of numeric columns
        verbose: Print logging
    """
    if verbose:
        logging.root.setLevel(logging.INFO)

    p = provider.lower()
    if p == "openml":
        df = list_openml(name_substr=name_substr, cat_min=cat_min, num_min=num_min)
    elif p == "uciml":
        df = list_uciml(
            area=area, name_substr=name_substr, cat_min=cat_min, num_min=num_min
        )
    else:
        raise SystemExit("provider must be 'openml' or 'uciml'")
    print(df.to_csv(sep="\t", index=None))


def report(
    provider: str = "openml",
    *,
    datasets: List[str] = [],
    outdir: str = "docs",
    configs_yaml: str = "",
    area: str = "Health and Medicine",
    verbose: bool = False,
) -> None:
    """Run the report pipeline on a collection of datasets.

    Parameters:
        provider: 'openml' or 'uciml'.
        datasets: For 'openml', dataset names; for 'uciml', dataset IDs as strings. If omitted, defaults are used.
        outdir: Output directory where per-dataset reports are written.
        configs_yaml: YAML file defining a list of synthetic data model configs
        metasyn: Run metasyn inference and synthetic data generation
        area: Default topic area for UCI datasets
    """
    if verbose:
        logging.root.setLevel(logging.INFO)

    ensure_dir(outdir)
    from .pipeline import process_dataset  # defer heavy imports

    ds = datasets if datasets else None
    dataset_specs: List[DatasetSpec] = specs_from_input(
        provider=provider, datasets=ds, area=area
    )
    # Load unified model configs.
    if configs_yaml:
        try:
            model_specs = load_model_configs(configs_yaml.strip())
        except Exception as exc:
            raise SystemExit(str(exc))
    else:
        model_specs = []

    for dataset_spec in dataset_specs:
        logging.info(f"Loading {dataset_spec}")
        try:
            meta, df, color = load_dataset(dataset_spec)
            process_dataset(
                meta,
                df,
                color,
                outdir,
                model_configs=model_specs,
            )
        except Exception as e:
            print(
                f"[WARN] Skipped {dataset_spec.provider}:{dataset_spec.name} due to error: {e}"
            )
            raise


def main(argv: Optional[List[str]] = None) -> None:
    defopt.run([search, report], argv=argv)


if __name__ == "__main__":
    main(sys.argv[1:])
