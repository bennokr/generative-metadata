"""Measure downstream fidelity of a synthetic dataset by training regression models

# Example usage (sketch)

>>> meta = {
>>>   "dataset": {"target_name": "target", "target_type": "binary"},
>>>   "variables": {
>>>       "target": {"role": "target", "stat_type": "binary"},
>>>       "age":    {"role": "predictor", "stat_type": "numeric", "missing_codes": [], "interaction_ok": True},
>>>       "sex":    {"role": "predictor", "stat_type": "binary", "categories": [0,1], "reference": 0, "interaction_ok": True},
>>>       "cp":     {"role": "predictor", "stat_type": "nominal", "categories": [0,1,2,3], "reference": 0, "interaction_ok": True},
>>>       "thal":   {"role": "predictor", "stat_type": "nominal", "categories": [3,6,7], "reference": 3, "interaction_ok": True},
>>>   }
>>> }

>>> out = compare_real_vs_synth(
>>>   df_real, df_synth, meta,
>>>   m=20, burnin=5, max_interactions=5, cv=5
>>> )
>>> print(out["formula"])
>>> print(out["compare"])
"""

import itertools
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV

import patsy
import statsmodels.api as sm
from statsmodels.imputation import mice

# -----------------------------
# Utilities
# -----------------------------

def _is_cat(vmeta):
    return vmeta["stat_type"] in {"binary", "nominal", "ordinal"}

def _cat_levels(vmeta):
    return list(vmeta.get("categories", []))

def _ref_level(vmeta):
    # choose provided reference, else first category
    return vmeta.get("reference", (vmeta.get("categories") or [None])[0])

def _cat_term(var, vmeta):
    ref = _ref_level(vmeta)
    if vmeta["stat_type"] in {"binary", "nominal"}:
        return f'C({var}, Treatment(reference="{ref}"))'
    if vmeta["stat_type"] == "ordinal":
        # encode as integer if ordered levels provided; else treat as categorical
        if vmeta.get("ordered_levels"):
            return var  # assume df holds ordinal-coded numbers already
        return f'C({var}, Treatment(reference="{ref}"))'
    raise ValueError("not categorical")

def _var_term(var, vmeta):
    return _cat_term(var, vmeta) if _is_cat(vmeta) else var

def _cardinality(df, var, vmeta):
    if _is_cat(vmeta):
        return len(_cat_levels(vmeta)) or df[var].nunique(dropna=True)
    return np.inf

def _replace_missing_codes(df, meta):
    df = df.copy()
    for var, vmeta in meta["variables"].items():
        if "missing_codes" in vmeta and vmeta["missing_codes"]:
            miss = set(vmeta["missing_codes"])
            df[var] = df[var].map(lambda x: np.nan if x in miss else x)
    return df

def _coerce_dtypes_and_levels(df, meta, *, fill_cats_with_missing_token=False):
    """Apply categories and ordinals from metadata. Optionally append '__MISSING__'."""
    df = df.copy()
    for var, vmeta in meta["variables"].items():
        if var not in df.columns:
            continue
        if _is_cat(vmeta):
            levels = list(_cat_levels(vmeta))
            if fill_cats_with_missing_token and "__MISSING__" not in levels:
                levels = levels + ["__MISSING__"]
            if levels:
                df[var] = pd.Categorical(df[var], categories=levels, ordered=vmeta["stat_type"] == "ordinal")
            else:
                # infer then freeze
                cats = list(pd.Series(df[var]).dropna().unique())
                if fill_cats_with_missing_token:
                    cats = cats + ["__MISSING__"]
                df[var] = pd.Categorical(df[var], categories=cats, ordered=vmeta["stat_type"] == "ordinal")
        else:
            df[var] = pd.to_numeric(df[var], errors="coerce")
    return df

def _fill_for_screening(df, meta):
    """Single-imputation for feature screening only."""
    df = df.copy()
    for var, vmeta in meta["variables"].items():
        if var not in df.columns or vmeta["role"] == "target":
            continue
        if _is_cat(vmeta):
            df[var] = df[var].astype("object")
            df[var] = df[var].where(df[var].notna(), "__MISSING__")
        else:
            df[var] = pd.to_numeric(df[var], errors="coerce")
            imp = SimpleImputer(strategy="median")
            df[var] = imp.fit_transform(df[[var]]).ravel()
    # target: drop NA rows for the screening stage
    yname = meta["dataset"]["target_name"]
    df = df[df[yname].notna()]
    return df

# -----------------------------
# Candidate generation
# -----------------------------

def generate_candidates(df, meta, max_interactions=5):
    preds = [v for v, m in meta["variables"].items() if m["role"] == "predictor"]
    # main effects
    main_terms = []
    for v in preds:
        main_terms.append(_var_term(v, meta["variables"][v]))

    # interactions under strong heredity, conservative cardinality rules
    cand_pairs = []
    for a, b in itertools.combinations(preds, 2):
        ma, mb = meta["variables"][a], meta["variables"][b]
        if not ma.get("interaction_ok", True) or not mb.get("interaction_ok", True):
            continue
        ca, cb = _cardinality(df, a, ma), _cardinality(df, b, mb)
        # allow:
        # - num × num
        # - num × cat with cat <= 6
        # - cat × cat if product of levels <= 12
        if not _is_cat(ma) and not _is_cat(mb):
            ok = True
        elif _is_cat(ma) and not _is_cat(mb):
            ok = ca <= 6
        elif not _is_cat(ma) and _is_cat(mb):
            ok = cb <= 6
        else:
            ok = (ca * cb) <= 12
        if ok:
            cand_pairs.append((a, b))
    # cap
    cand_pairs = cand_pairs[:max_interactions]

    inter_terms = []
    for a, b in cand_pairs:
        ta = _var_term(a, meta["variables"][a])
        tb = _var_term(b, meta["variables"][b])
        inter_terms.append(f"{ta}:{tb}")
    return main_terms, inter_terms

# -----------------------------
# Screening model
# -----------------------------

def screen_terms(df, yname, target_type, full_formula, *, cv=5):
    # Build design
    y, X = patsy.dmatrices(f"{yname} ~ {full_formula}", df, return_type="dataframe")
    term_slices = X.design_info.term_slices  # mapping term -> slice of columns
    term_names  = X.design_info.term_names

    # scale numeric columns for stability; keep intercept unscaled
    colnames = X.columns
    intercept_mask = colnames.str.fullmatch("Intercept")
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = X.copy()
    if (~intercept_mask).any():
        X_scaled.loc[:, ~intercept_mask] = scaler.fit_transform(X.loc[:, ~intercept_mask])

    # Pick estimator by target type
    if target_type == "binary":
        # L1 multinomial degenerates to binary for 2 classes
        est = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 10),
            cv=cv,
            penalty="l1",
            solver="saga",
            scoring="neg_log_loss",
            max_iter=1000,
            n_jobs=None,
            fit_intercept=False
        )
        est.fit(X_scaled, y.values.ravel())
        coef = est.coef_.reshape(-1)  # (n_features,)
        nz = np.where(np.abs(coef) > 1e-8)[0]

    elif target_type == "multiclass":
        est = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 10),
            cv=cv,
            penalty="l1",
            solver="saga",
            multi_class="multinomial",
            scoring="neg_log_loss",
            max_iter=2000,
            n_jobs=None,
            fit_intercept=False
        )
        est.fit(X_scaled, y.values.ravel())
        coef = est.coef_  # (K, n_features)
        nz = np.where((np.abs(coef) > 1e-8).any(axis=0))[0]

    elif target_type == "continuous":
        est = LassoCV(alphas=None, cv=cv, max_iter=5000, fit_intercept=False)
        est.fit(X_scaled, y.values.ravel())
        coef = est.coef_
        nz = np.where(np.abs(coef) > 1e-8)[0]

    elif target_type == "count":
        # elastic-net with l1_ratio=1 => lasso; simple CV on alpha
        grid = {"alpha": np.logspace(-4, 1, 12)}
        base = PoissonRegressor(max_iter=2000, fit_intercept=False, alpha=1.0, l1_ratio=1.0, verbose=0)
        est = GridSearchCV(base, grid, cv=cv, scoring="neg_mean_poisson_deviance")
        est.fit(X_scaled, y.values.ravel())
        coef = est.best_estimator_.coef_
        nz = np.where(np.abs(coef) > 1e-8)[0]
    else:
        raise ValueError("target_type must be one of: binary, multiclass, continuous, count")

    # Map selected columns → terms
    selected_terms = set()
    for tname, sl in term_slices.items():
        idxs = range(sl.start, sl.stop)
        if any(i in nz for i in idxs):
            selected_terms.add(tname)

    # Enforce strong heredity: if an interaction is kept, keep both parents
    mains = {t for t in selected_terms if ":" not in t and t != "Intercept"}
    inters = {t for t in selected_terms if ":" in t}
    for t in list(inters):
        a, b = t.split(":", 1)
        mains.add(a)
        mains.add(b)

    # Keep intercept
    if "Intercept" in term_names:
        selected_terms.add("Intercept")

    # Return in stable order
    ordered = [t for t in term_names if (t in mains or t in inters or t == "Intercept")]
    return ordered

def formula_from_selected(yname, selected_terms):
    # Drop "Intercept" from RHS; Patsy includes it by default if not 0+
    rhs_terms = [t for t in selected_terms if t != "Intercept"]
    if not rhs_terms:
        rhs = "1"
    else:
        rhs = " + ".join(rhs_terms)
    return f"{yname} ~ {rhs}"

# -----------------------------
# Public API
# -----------------------------

def auto_formula(df, meta, *, max_interactions=5, cv=5):
    df0 = _replace_missing_codes(df, meta)
    df0 = _coerce_dtypes_and_levels(df0, meta, fill_cats_with_missing_token=True)
    df0 = _fill_for_screening(df0, meta)

    # Build full candidate set
    main_terms, inter_terms = generate_candidates(df0, meta, max_interactions=max_interactions)
    full_formula = " + ".join(main_terms + inter_terms)

    yname = meta["dataset"]["target_name"]
    target_type = meta["dataset"]["target_type"]

    selected_terms = screen_terms(df0, yname, target_type, full_formula, cv=cv)
    return formula_from_selected(yname, selected_terms)

def fit_with_mi(df, formula, meta, *, m=20, burnin=5):
    yname = meta["dataset"]["target_name"]
    target_type = meta["dataset"]["target_type"]

    # Prepare data for MI (keep NaN, apply levels but no single impute)
    df_mi = _replace_missing_codes(df, meta)
    df_mi = _coerce_dtypes_and_levels(df_mi, meta, fill_cats_with_missing_token=False)

    imp = mice.MICEData(df_mi)

    if target_type == "binary":
        model_class = sm.GLM
        init_kwds = {"family": sm.families.Binomial()}
    elif target_type == "continuous":
        model_class = sm.OLS
        init_kwds = {}
    elif target_type == "count":
        model_class = sm.GLM
        init_kwds = {"family": sm.families.Poisson()}
    elif target_type == "multiclass":
        # MNLogit supports formulas; MICE can wrap any model_class with from_formula
        model_class = sm.MNLogit
        init_kwds = {}
    else:
        raise ValueError("unsupported target_type")

    mobj = mice.MICE(formula, model_class, imp, init_kwds=init_kwds)
    res = mobj.fit(n_burnin=burnin, n_imputations=m)  # pooled via Rubin's rules
    return res  # has .params, .bse, etc.

def compare_real_vs_synth(df_real, df_synth, meta, *, m=20, burnin=5, max_interactions=5, cv=5):
    # 1) auto-generate formula on REAL only
    formula = auto_formula(df_real, meta, max_interactions=max_interactions, cv=cv)

    # 2) pooled MI fits on real and synth with identical formula
    res_real  = fit_with_mi(df_real,  formula, meta, m=m, burnin=burnin)
    res_synth = fit_with_mi(df_synth, formula, meta, m=m, burnin=burnin)

    # 3) align coefficients
    out = (pd.DataFrame({
                "beta_real":  res_real.params,
                "se_real":    res_real.bse,
                "beta_synth": res_synth.params,
                "se_synth":   res_synth.bse
           })
           .assign(sign_match=lambda d: np.sign(d.beta_real) == np.sign(d.beta_synth))
          )

    return {"formula": formula, "results_real": res_real, "results_synth": res_synth, "compare": out}


