import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from metasyn.metaframe import MetaFrame
from sklearn.model_selection import train_test_split

from ..metrics import per_variable_distances, summarize_distance_metrics
from ..models import model_run_root, write_manifest
from ..utils import coerce_continuous_to_float, ensure_dir, infer_types


def run_experiment(
    df: pd.DataFrame,
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
    rows = rows or len(df)

    train_df, test_df = train_test_split(
        df.copy(), test_size=test_size, random_state=seed, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    run_root = model_run_root(Path(outdir))
    run_dir = run_root / label
    ensure_dir(str(run_dir))

    # Fit MetaFrame (GMF) on the same training data
    logging.info("Fitting MetaSyn MetaFrame on train")
    mf = MetaFrame.fit_dataframe(train_df)
    metasyn_gmf = run_dir / "metasyn_gmf.json"
    try:
        mf.save(str(metasyn_gmf))
    except Exception as e:
        logging.warning(f"Could not save MetaSyn GMF: {e}")
        metasyn_gmf = None

    # Synthesize MetaSyn data
    try:
        synth_meta = mf.synthesize(n=rows)
        import polars as pl

        if isinstance(synth_meta, pl.DataFrame):
            synth_df = synth_meta.to_pandas()
        else:
            synth_df = synth_meta
    except Exception as e:
        logging.warning(f"MetaSyn synthesis failed: {e}")
        synth_df = pd.DataFrame(columns=df.columns)

    # Align columns and dtypes
    disc_cols, cont_cols = infer_types(df)
    synth_df = synth_df.reindex(columns=df.columns)
    for c in disc_cols:
        if c in synth_df.columns:
            synth_df[c] = synth_df[c].astype("category")
            if pd.api.types.is_categorical_dtype(df[c]):
                synth_df[c] = synth_df[c].cat.set_categories(df[c].cat.categories)
    synth_df = coerce_continuous_to_float(synth_df, cont_cols)

    synth_df.to_csv(run_dir / "synthetic.csv", index=False)
    logging.info("Wrote synthetic CSV: %s", run_dir / "synthetic.csv")

    if semmap_export:
        try:
            synth_df.semmap.apply_json_metadata(
                copy.deepcopy(semmap_export), convert_pint=False
            )
            metasyn_semmap_parquet_path = run_dir / "synthetic_metasyn.semmap.parquet"
            synth_df.semmap.to_parquet(str(metasyn_semmap_parquet_path), index=False)
            metasyn_semmap_parquet_file = str(metasyn_semmap_parquet_path)
            logging.info(
                "Wrote SemMap parquet for MetaSyn synthetic: %s",
                metasyn_semmap_parquet_path,
            )
        except Exception:
            logging.exception(
                "Failed to serialize SemMap parquet for MetaSyn synthetic"
            )

    # Distances for MetaSyn model (use overall discrete/continuous cols inferred earlier)
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
        "umap_png": None,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    manifest = {
        "backend": "metasyn",
        "name": label,
        "provider": provider,
        "dataset_name": dataset_name,
        "provider_id": provider_id,
        "seed": seed,
        "rows": rows,
        "test_size": test_size,
    }
    write_manifest(run_dir, manifest)
    logging.info("Finished MetaSyn run: %s", label)
    return run_dir
