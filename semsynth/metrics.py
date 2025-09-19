from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


def heldout_loglik(model, df_test: pd.DataFrame) -> Dict[str, float]:
    try:
        arr = model.logl(df_test)
    except Exception:
        return {
            "mean_loglik": float("nan"),
            "std_loglik": float("nan"),
            "sum_loglik": float("nan"),
            "n_rows": int(0),
        }
    try:
        arr = np.asarray(arr, dtype=float).reshape(-1)
    except Exception:
        arr = np.asarray(arr).reshape(-1)
        try:
            arr = arr.astype(float)
        except Exception:
            arr = np.array([], dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return {
            "mean_loglik": float("nan"),
            "std_loglik": float("nan"),
            "sum_loglik": float("nan"),
            "n_rows": int(0),
        }
    vals = arr[mask]
    return {
        "mean_loglik": float(np.mean(vals)),
        "std_loglik": float(np.std(vals)),
        "sum_loglik": float(np.sum(vals)),
        "n_rows": int(len(vals)),
    }


def js_divergence_discrete(p: pd.Series, q: pd.Series) -> float:
    cats = sorted(set(p.dropna().unique()).union(set(q.dropna().unique())))
    if len(cats) == 0:
        return float("nan")
    p_counts = p.value_counts(dropna=True).reindex(cats, fill_value=0).to_numpy(dtype=float)
    q_counts = q.value_counts(dropna=True).reindex(cats, fill_value=0).to_numpy(dtype=float)
    p_probs = (p_counts + 1e-9) / (p_counts.sum() + 1e-9 * len(cats))
    q_probs = (q_counts + 1e-9) / (q_counts.sum() + 1e-9 * len(cats))
    return float(jensenshannon(p_probs, q_probs, base=2.0))


def per_variable_distances(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    discrete_cols: List[str],
    continuous_cols: List[str],
) -> pd.DataFrame:
    rows = []
    for c in real_df.columns:
        if c in discrete_cols:
            jsd = js_divergence_discrete(real_df[c], synth_df[c])
            rows.append(dict(variable=c, type="discrete", JSD=jsd))
        else:
            a = pd.to_numeric(real_df[c], errors="coerce").dropna()
            b = pd.to_numeric(synth_df[c], errors="coerce").dropna()
            if len(a) == 0 or len(b) == 0:
                rows.append(dict(variable=c, type="continuous", KS=float("nan"), W1=float("nan")))
                continue
            ks = ks_2samp(a, b, method="auto").statistic
            w1 = wasserstein_distance(a, b)
            rows.append(dict(variable=c, type="continuous", KS=ks, W1=w1))
    return pd.DataFrame(rows)

def summarize_distance_metrics(distances: pd.DataFrame) -> Dict[str, float]:
    """Compute aggregate statistics for per-variable distance metrics."""
    if distances.empty:
        return {
            'disc_jsd_mean': float('nan'),
            'disc_jsd_median': float('nan'),
            'cont_ks_mean': float('nan'),
            'cont_w1_mean': float('nan'),
        }
    disc = distances[distances['type'] == 'discrete'] if 'type' in distances.columns else pd.DataFrame()
    cont = distances[distances['type'] == 'continuous'] if 'type' in distances.columns else pd.DataFrame()
    def _agg(frame: pd.DataFrame, column: str, reducer) -> float:
        if frame.empty or column not in frame.columns:
            return float('nan')
        series = pd.to_numeric(frame[column], errors='coerce').dropna()
        if series.empty:
            return float('nan')
        return float(reducer(series))
    return {
        'disc_jsd_mean': _agg(disc, 'JSD', np.nanmean),
        'disc_jsd_median': _agg(disc, 'JSD', np.nanmedian),
        'cont_ks_mean': _agg(cont, 'KS', np.nanmean),
        'cont_w1_mean': _agg(cont, 'W1', np.nanmean),
    }
