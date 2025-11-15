"""MetaSyn backend implementation for SemSynth."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..metrics import per_variable_distances, summarize_distance_metrics
from ..models import model_run_root, write_manifest
from ..utils import coerce_continuous_to_float, ensure_dir, infer_types

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd


def _load_metasyn() -> Any:
    try:
        from metasyn.metaframe import MetaFrame
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MetaSyn backend requires 'metasyn'; install with pip install semsynth[metasyn]"
        ) from exc
    return MetaFrame


def run_experiment(
    df: "pd.DataFrame",
    *,
    provider: Optional[str],
    dataset_name: Optional[str],
    provider_id: Optional[int],
    outdir: str,
    label: str,
    model_info: Dict[str, Any] | None,
    rows: Optional[int],
    seed: int,
    test_size: float,
    semmap_export: Optional[Dict[str, Any]] = None,
) -> Path:
    """Fit the MetaSyn model and persist artifacts following the backend contract."""

    import pandas as pd
    from sklearn.model_selection import train_test_split

    MetaFrame = _load_metasyn()

    rows = rows or len(df)
    working = df.copy()
    disc_cols, cont_cols = infer_types(working)
    working = coerce_continuous_to_float(working, cont_cols)

    train_df, test_df = train_test_split(
        working, test_size=test_size, random_state=seed, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    run_root = model_run_root(Path(outdir))
    run_dir = run_root / label
    ensure_dir(str(run_dir))

    logging.info("Fitting MetaSyn MetaFrame on train for %s", label)
    mf = MetaFrame.fit_dataframe(train_df)
    gmf_path = run_dir / "metasyn_gmf.json"
    try:
        mf.save(str(gmf_path))
    except Exception:
        logging.warning("Could not serialize MetaSyn GMF", exc_info=True)
        gmf_path = None

    try:
        synth_meta = mf.synthesize(n=rows)
        if hasattr(synth_meta, "to_pandas"):
            synth_df = synth_meta.to_pandas()
        else:
            synth_df = pd.DataFrame(synth_meta)
    except Exception:
        logging.exception("MetaSyn synthesis failed", exc_info=True)
        synth_df = pd.DataFrame(columns=working.columns)

    synth_df = synth_df.reindex(columns=working.columns)
    for column in disc_cols:
        if column in synth_df.columns:
            synth_df[column] = synth_df[column].astype("category")
            if pd.api.types.is_categorical_dtype(df[column]):
                synth_df[column] = synth_df[column].cat.set_categories(
                    df[column].cat.categories
                )
    synth_df = coerce_continuous_to_float(synth_df, cont_cols)

    synth_csv = run_dir / "synthetic.csv"
    synth_df.to_csv(synth_csv, index=False)

    if semmap_export:
        try:
            synth_df.semmap.from_jsonld(
                copy.deepcopy(semmap_export), convert_pint=False
            )
            synth_df.semmap.to_parquet(
                str(run_dir / "synthetic.semmap.parquet"), index=False
            )
        except Exception:
            logging.exception("Failed to serialize SemMap parquet for MetaSyn synthetic")

    dist_df = per_variable_distances(test_df, synth_df, disc_cols, cont_cols)
    dist_df.to_csv(run_dir / "per_variable_metrics.csv", index=False)
    metrics = {
        "backend": "metasyn",
        "summary": summarize_distance_metrics(dist_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "synth_rows": len(synth_df),
        "discrete_cols": len(disc_cols),
        "continuous_cols": len(cont_cols),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = {
        "backend": "metasyn",
        "name": label,
        "provider": provider,
        "dataset_name": dataset_name,
        "provider_id": provider_id,
        "params": dict(model_info or {}),
        "seed": seed,
        "rows": rows,
        "test_size": test_size,
    }
    write_manifest(run_dir, manifest)
    logging.info("Finished MetaSyn run: %s", label)
    return run_dir

