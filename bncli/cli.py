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
    bn_type: str = "clg",
    area: str = "Health and Medicine",
    verbose: bool = False
) -> None:
    """Run the BN report pipeline on a collection of datasets.

    Parameters:
        provider: 'openml' or 'uciml'.
        datasets: For 'openml', dataset names; for 'uciml', dataset IDs as strings. If omitted, defaults are used.
        outdir: Output directory where per-dataset reports are written.
        bn_type: Which BN type to learn. Currently only 'clg' is supported.
        area: For default UCI selection, which Area to pull mixed-dtype datasets from.
    """
    if verbose:
        logging.root.setLevel(logging.INFO)

    ensure_dir(outdir)
    from .pipeline import process_dataset  # defer heavy imports

    ds = datasets if datasets else None
    specs: List[DatasetSpec] = specs_from_input(provider=provider, datasets=ds, area=area)
    for spec in specs:
        logging.info(f'Loading {spec}')
        try:
            meta, df, color = load_dataset(spec)
            process_dataset(meta, df, color, outdir, bn_type=bn_type)
        except Exception as e:
            print(f"[WARN] Skipped {spec.provider}:{spec.name} due to error: {e}")
            raise


def main(argv: Optional[List[str]] = None) -> None:
    defopt.run([search, report], argv=argv)


if __name__ == "__main__":
    main(sys.argv[1:])
