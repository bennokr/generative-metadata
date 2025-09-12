from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


def heldout_loglik(model, df_test: pd.DataFrame) -> Dict[str, float]:
    arr = model.logl(df_test)
    arr = np.asarray(arr).reshape(-1)
    return {
        "mean_loglik": float(np.mean(arr)),
        "std_loglik": float(np.std(arr)),
        "sum_loglik": float(np.sum(arr)),
        "n_rows": int(len(arr)),
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

