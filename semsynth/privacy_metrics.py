from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# pip install synthcity
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.metrics.eval_sanity import (
    CommonRowsProportion,
    NearestSyntheticNeighborDistance,
    CloseValuesProbability,
)
from synthcity.metrics.eval_privacy import (
    kAnonymization,
    kMap,
    IdentifiabilityScore,
    DeltaPresence,
)

@dataclass
class DatasetPrivacySummary:
    n_real: int
    n_synth: int
    used_columns: List[str]
    qi_columns: List[str]
    sensitive_columns: List[str]
    exact_overlap_rate: float
    near_duplicate_rate_eps: float          # uses synthcity’s close-values threshold (0.2)
    nn_distance_stats: Dict[str, float]     # {'mean','median','p95','min','max'}
    k_min: Optional[int]
    k_pct_lt5: Optional[float]
    rare_qi_reproduction_rate: Optional[float]
    t_closeness: Dict[str, Dict[str, float]]  # per sensitive var: {'mean','p95','max'}

def _prep(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Small, deterministic typing + NA handling."""
    t = dict(zip(meta.variable, meta.type))
    out = df.copy()
    for c in out.columns:
        kind = t.get(c, "categorical")
        if kind == "numeric":
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].fillna(out[c].median())
        elif kind == "datetime":
            x = pd.to_datetime(out[c], errors="coerce")
            med = x.dropna().astype("int64").median() if x.notna().any() else 0
            out[c] = x.astype("int64").fillna(med)
        else:
            out[c] = out[c].astype("object").where(out[c].notna(), "__MISSING__")
    return out

def _tv(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    return 0.5 * float(np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum())

def _w1(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(x[~np.isnan(x)]); y = np.sort(y[~np.isnan(y)])
    if len(x) == 0 or len(y) == 0: return 0.0
    n = max(len(x), len(y))
    q = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))

def summarize_privacy_synthcity(df_real: pd.DataFrame,
                                df_synth: pd.DataFrame,
                                meta: pd.DataFrame,
                                *,
                                eps: float = 0.1) -> DatasetPrivacySummary:
    # select columns
    assert {'variable','role','type'}.issubset(meta.columns)
    use_meta = meta[~meta.role.isin(['ignore','id'])].copy()
    use_cols = [c for c in use_meta.variable if c in df_real.columns and c in df_synth.columns]
    if not use_cols: raise ValueError("No overlapping usable columns.")
    qi = [c for c in use_meta.loc[use_meta.role=='qi','variable'] if c in use_cols]
    sens = [c for c in use_meta.loc[use_meta.role=='sensitive','variable'] if c in use_cols]

    # preprocess
    df_r = _prep(df_real[use_cols], use_meta.set_index('variable').loc[use_cols].reset_index())
    df_s = _prep(df_synth[use_cols], use_meta.set_index('variable').loc[use_cols].reset_index())
    Xr = GenericDataLoader(df_r, sensitive_features=sens or [])
    Xs = GenericDataLoader(df_s, sensitive_features=sens or [])

    # synthcity metrics (documented interfaces) :contentReference[oaicite:0]{index=0}
    exact_overlap = float(CommonRowsProportion().evaluate_default(Xr, Xs))
    close_prob = float(CloseValuesProbability().evaluate_default(Xr, Xs))  # uses internal 0.2 threshold
    nn_eval = NearestSyntheticNeighborDistance()
    nn_raw = nn_eval.evaluate(Xr, Xs)  # dict with stats
    if isinstance(nn_raw, dict):
        nn_stats = {
            'mean': float(nn_raw.get('mean', np.nan)),
            'median': float(nn_raw.get('median', np.nan)),
            'p95': float(nn_raw.get('p95', np.nan)),
            'min': float(nn_raw.get('min', np.nan)),
            'max': float(nn_raw.get('max', np.nan)),
        }
    else:
        nn_stats = {'mean': float(nn_eval.evaluate_default(Xr, Xs)),
                    'median': np.nan, 'p95': np.nan, 'min': np.nan, 'max': np.nan}

    # k-anon on real QIs and k-map on QIs
    if qi:
        eq_sizes = df_r.groupby(qi, dropna=False).size().to_numpy()
        k_min = int(eq_sizes.min()) if eq_sizes.size else None
        k_pct_lt5 = float((eq_sizes < 5).mean()) if eq_sizes.size else None
        k_map_val = int(kMap().evaluate_default(GenericDataLoader(df_r[qi]), GenericDataLoader(df_s[qi])))
    else:
        k_min = k_pct_lt5 = k_map_val = None

    ident = float(IdentifiabilityScore().evaluate_default(Xr, Xs))
    delta = float(DeltaPresence().evaluate_default(Xr, Xs))

    # rare QI reproduction (real count<=5 or freq<=1%)
    if qi:
        cnt = df_r.groupby(qi, dropna=False).size()
        rare = set(cnt[(cnt <= 5) | (cnt/len(df_r) <= 0.01)].index)
        syn_keys = set(df_s.groupby(qi, dropna=False).size().index)
        rare_rate = len(rare & syn_keys) / len(rare) if rare else 0.0
    else:
        rare_rate = None

    # t-closeness per sensitive var (TV for categoricals; W1 for numerics)
    tmap = dict(zip(use_meta.variable, use_meta.type))
    t_close: Dict[str, Dict[str, float]] = {}
    if qi and sens:
        groups = df_s.groupby(qi, dropna=False)
        for s_col in sens:
            if tmap.get(s_col) == 'numeric':
                gref = df_r[s_col].to_numpy(float)
                vals = [ _w1(g[s_col].to_numpy(float), gref) for _, g in groups ]
            else:
                pref = df_r[s_col].astype('object').value_counts(normalize=True, dropna=False)
                vals = [ _tv(g[s_col].astype('object').value_counts(normalize=True, dropna=False), pref)
                         for _, g in groups ]
            if vals:
                arr = np.array(vals, float)
                t_close[s_col] = {'mean': float(arr.mean()),
                                  'p95': float(np.quantile(arr, 0.95)),
                                  'max': float(arr.max())}
    # package
    return DatasetPrivacySummary(
        n_real=len(df_r),
        n_synth=len(df_s),
        used_columns=use_cols,
        qi_columns=qi,
        sensitive_columns=sens,
        exact_overlap_rate=exact_overlap,
        near_duplicate_rate_eps=close_prob,     # synthcity’s fixed threshold
        nn_distance_stats=nn_stats,
        k_min=k_min,
        k_pct_lt5=k_pct_lt5,
        rare_qi_reproduction_rate=rare_rate,
        t_closeness=t_close,
    )
