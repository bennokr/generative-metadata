from __future__ import annotations

import sys
from typing import List, Optional
import logging

import defopt

from .datasets import (
    DatasetSpec,
    list_openml,
    list_uciml,
    load_dataset,
    specs_from_input,
)
from .utils import ensure_dir


def search(provider: str, *, name_substr: Optional[str] = None, area: str = "Health and Medicine", cat_min: int = 1, num_min: int = 1, verbose: bool = False) -> None:
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
        df = list_uciml(area=area, name_substr=name_substr, cat_min=cat_min, num_min=num_min)
    else:
        raise SystemExit("provider must be 'openml' or 'uciml'")
    print(df.to_csv(sep='\t', index=None))


def report(
    provider: str = "openml",
    *,
    datasets: List[str] = [],
    outdir: str = "bn_reports",
    bn_types: List[str] = ("clg", "semiparametric"),
    configs_yaml: str = "",
    arc_blacklist: List[str] = (),
    area: str = "Health and Medicine",
    verbose: bool = False
) -> None:
    """Run the BN report pipeline on a collection of datasets.

    Parameters:
        provider: 'openml' or 'uciml'.
        datasets: For 'openml', dataset names; for 'uciml', dataset IDs as strings. If omitted, defaults are used.
        outdir: Output directory where per-dataset reports are written.
        bn_types: One or more BN types to learn and compare. Supported: 'clg', 'semiparametric'. Ignored if configs_yaml is provided.
        configs_yaml: Optional YAML file defining a list of BN structure-learning configurations. Each item can include: name, bn_type, score, operators, max_indegree, seed. If provided, these override bn_types.
        arc_blacklist: Optional list of variable names treated as sensitive; structure learning forbids arcs from these variables to others. If omitted, defaults to ['age','sex','race'] or for UCI ML, the 'demographics' metadata if present.
        area: For default UCI selection, which Area to pull mixed-dtype datasets from.
    """
    if verbose:
        logging.root.setLevel(logging.INFO)

    ensure_dir(outdir)
    from .pipeline import process_dataset  # defer heavy imports

    ds = datasets if datasets else None
    specs: List[DatasetSpec] = specs_from_input(provider=provider, datasets=ds, area=area)
    # Try to load YAML configurations if provided
    bn_configs = None
    cfg_path = configs_yaml.strip()
    if cfg_path:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise SystemExit("To use configs_yaml you need PyYAML installed. Please install pyyaml.")
        import os
        if not os.path.exists(cfg_path):
            raise SystemExit(f"configs_yaml file not found: {cfg_path}")
        with open(cfg_path, 'r', encoding='utf-8') as fr:
            data = yaml.safe_load(fr)
            if isinstance(data, dict) and 'configs' in data:
                data = data['configs']
            if not isinstance(data, list):
                raise SystemExit("configs_yaml must contain a list of configuration dictionaries")
            # ensure each item is dict
            bn_configs = []
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise SystemExit(f"Config item at index {i} is not a dict")
                bn_configs.append(item)
    for spec in specs:
        logging.info(f'Loading {spec}')
        try:
            meta, df, color = load_dataset(spec)
            # Pass arc_blacklist only if user provided it; otherwise None to use provider-specific default
            abl = list(arc_blacklist) if isinstance(arc_blacklist, (list, tuple)) and len(arc_blacklist) else None
            process_dataset(
                meta,
                df,
                color,
                outdir,
                bn_configs=bn_configs,
                arc_blacklist=abl,
            )
        except Exception as e:
            print(f"[WARN] Skipped {spec.provider}:{spec.name} due to error: {e}")
            raise


def main(argv: Optional[List[str]] = None) -> None:
    defopt.run([search, report], argv=argv)


if __name__ == "__main__":
    main(sys.argv[1:])
