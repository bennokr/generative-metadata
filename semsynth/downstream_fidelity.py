"""Measure downstream fidelity of a synthetic dataset by training regression models.

# Example usage (sketch)

>>> meta = {
>>>   "dsv:datasetSchema": {
>>>       "dsv:column": [
>>>           {
>>>               "schema:name": "target",
>>>               "prov:hadRole": "target",
>>>               "dsv:summaryStatistics": {
>>>                   "dsv:statisticalDataType": "dsv:NominalDataType"
>>>               },
>>>               "schema:defaultValue": "0",
>>>               "dsv:columnProperty": {
>>>                   "dsv:hasCodeBook": {
>>>                       "skos:hasTopConcept": [
>>>                           {"skos:notation": "0"},
>>>                           {"skos:notation": "1"},
>>>                       ]
>>>                   }
>>>               }
>>>           },
>>>           {
>>>               "schema:name": "age",
>>>               "prov:hadRole": "predictor",
>>>               "dsv:summaryStatistics": {
>>>                   "dsv:statisticalDataType": "dsv:QuantitativeDataType"
>>>               }
>>>           },
>>>       ]
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

def _schema(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    if hasattr(meta, "to_jsonld"):
        meta = meta.to_jsonld() or {}
    schema = meta.get("dsv:datasetSchema") or meta.get("datasetSchema") or {}
    if isinstance(schema, Mapping):
        return schema
    return {}


def _columns(meta: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    schema = _schema(meta)
    columns = schema.get("dsv:column") or schema.get("columns") or []
    return [col for col in columns if isinstance(col, Mapping)]


def _column_name(col_meta: Mapping[str, Any]) -> Optional[str]:
    for key in ("schema:name", "name", "column", "column_name"):
        value = col_meta.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _column_role(col_meta: Mapping[str, Any]) -> str:
    role = col_meta.get("prov:hadRole")
    if isinstance(role, str):
        role_lower = role.strip().lower()
        if role_lower:
            return role_lower
    return "predictor"


def _statistical_type(col_meta: Mapping[str, Any]) -> str:
    nodes: List[Mapping[str, Any]] = []

    maybe_stats = col_meta.get("dsv:summaryStatistics") or col_meta.get("summaryStatistics")
    if isinstance(maybe_stats, Mapping):
        nodes.append(maybe_stats)

    column_prop = col_meta.get("dsv:columnProperty") or col_meta.get("columnProperty")
    if isinstance(column_prop, Mapping):
        prop_stats = column_prop.get("dsv:summaryStatistics") or column_prop.get("summaryStatistics")
        if isinstance(prop_stats, Mapping):
            nodes.append(prop_stats)

    for node in nodes:
        dtype = node.get("dsv:statisticalDataType") or node.get("statisticalDataType")
        if isinstance(dtype, str):
            return dtype
        if isinstance(dtype, Mapping):
            identifier = dtype.get("@id") or dtype.get("id")
            if isinstance(identifier, str):
                return identifier
    return ""


def _is_cat(col_meta: Mapping[str, Any]) -> bool:
    dtype = _statistical_type(col_meta).lower()
    if not dtype:
        return False
    return any(token in dtype for token in ("nominal", "categorical", "ordinal", "binary"))


def _cat_levels(col_meta: Mapping[str, Any]) -> List[str]:
    column_prop = col_meta.get("dsv:columnProperty") or col_meta.get("columnProperty") or {}
    codebook: Mapping[str, Any]
    if isinstance(column_prop, Mapping):
        codebook = column_prop.get("dsv:hasCodeBook") or column_prop.get("hasCodeBook") or {}
    else:
        codebook = {}
    concepts = []
    if isinstance(codebook, Mapping):
        concepts = codebook.get("skos:hasTopConcept") or codebook.get("hasTopConcept") or []
    levels: List[str] = []
    for concept in concepts or []:
        if not isinstance(concept, Mapping):
            continue
        for key in ("skos:notation", "notation", "skos:prefLabel", "prefLabel"):
            value = concept.get(key)
            if isinstance(value, str):
                levels.append(value)
                break
    return levels


def _ref_level(col_meta: Mapping[str, Any]) -> Optional[str]:
    default = col_meta.get("schema:defaultValue") or col_meta.get("defaultValue")
    if isinstance(default, str):
        return default
    if default is not None:
        return str(default)
    levels = _cat_levels(col_meta)
    return levels[0] if levels else None

def _cat_term(var: str, vmeta: Mapping[str, Any]) -> str:
    ref = _ref_level(vmeta)
    dtype = _statistical_type(vmeta).lower()
    if any(token in dtype for token in ("binary", "nominal", "categorical")):
        return f'C({var}, Treatment(reference="{ref}"))'
    if "ordinal" in dtype:
        ordered_levels = vmeta.get("ordered_levels")
        if isinstance(ordered_levels, Sequence) and not isinstance(ordered_levels, (str, bytes)):
            return var
        return f'C({var}, Treatment(reference="{ref}"))'
    raise ValueError("not categorical")


def _var_term(var: str, vmeta: Mapping[str, Any]) -> str:
    return _cat_term(var, vmeta) if _is_cat(vmeta) else var


def _cardinality(df: pd.DataFrame, var: str, vmeta: Mapping[str, Any]) -> float:
    if _is_cat(vmeta):
        return len(_cat_levels(vmeta)) or df[var].nunique(dropna=True)
    return np.inf


def _column_lookup(meta: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        name: col
        for col in _columns(meta)
        if (name := _column_name(col))
    }


def _target_info(
    meta: Mapping[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> Tuple[str, str]:
    lookup = _column_lookup(meta)
    for name, col_meta in lookup.items():
        if _column_role(col_meta) != "target":
            continue
        dtype = _statistical_type(col_meta).lower()
        if "count" in dtype:
            return name, "count"
        if _is_cat(col_meta):
            levels = _cat_levels(col_meta)
            if not levels and df is not None and name in df.columns:
                levels = list(pd.Series(df[name]).dropna().unique())
            if len(levels) <= 2:
                return name, "binary"
            return name, "multiclass"
        return name, "continuous"
    raise ValueError("metadata must define a target column with prov:hadRole = 'target'")


def _missing_codes(col_meta: Mapping[str, Any]) -> List[str]:
    column_prop = col_meta.get("dsv:columnProperty") or col_meta.get("columnProperty") or {}
    codes: List[str] = []
    if isinstance(column_prop, Mapping):
        missing_node = column_prop.get("dsv:missingValueCode") or column_prop.get("missingValueCode")
        if isinstance(missing_node, Sequence) and not isinstance(missing_node, (str, bytes)):
            for item in missing_node:
                if isinstance(item, str):
                    codes.append(item)
        elif isinstance(missing_node, str):
            codes.append(missing_node)
    summary = col_meta.get("dsv:summaryStatistics") or col_meta.get("summaryStatistics") or {}
    if isinstance(summary, Mapping):
        fmt = summary.get("dsv:missingValueFormat") or summary.get("missingValueFormat")
        if isinstance(fmt, str):
            codes.append(fmt)
    return codes


def _replace_missing_codes(df: pd.DataFrame, meta: Mapping[str, Any]) -> pd.DataFrame:
    df = df.copy()
    for name, col_meta in _column_lookup(meta).items():
        if name not in df.columns:
            continue
        codes = _missing_codes(col_meta)
        if codes:
            miss = set(codes)
            df[name] = df[name].map(lambda x: np.nan if x in miss else x)
    return df


def _coerce_dtypes_and_levels(
    df: pd.DataFrame,
    meta: Mapping[str, Any],
    *,
    fill_cats_with_missing_token: bool = False,
) -> pd.DataFrame:
    """Apply categories and ordinals from metadata. Optionally append '__MISSING__'."""

    df = df.copy()
    for name, col_meta in _column_lookup(meta).items():
        if name not in df.columns:
            continue
        if _is_cat(col_meta):
            levels = list(_cat_levels(col_meta))
            if fill_cats_with_missing_token and "__MISSING__" not in levels:
                levels = levels + ["__MISSING__"]
            ordered = "ordinal" in _statistical_type(col_meta).lower()
            if levels:
                df[name] = pd.Categorical(df[name], categories=levels, ordered=ordered)
            else:
                cats = list(pd.Series(df[name]).dropna().unique())
                if fill_cats_with_missing_token:
                    cats = cats + ["__MISSING__"]
                df[name] = pd.Categorical(df[name], categories=cats, ordered=ordered)
        else:
            df[name] = pd.to_numeric(df[name], errors="coerce")
    return df


def _fill_for_screening(df: pd.DataFrame, meta: Mapping[str, Any]) -> pd.DataFrame:
    """Single-imputation for feature screening only."""

    df = df.copy()
    for name, col_meta in _column_lookup(meta).items():
        if name not in df.columns or _column_role(col_meta) == "target":
            continue
        if _is_cat(col_meta):
            df[name] = df[name].astype("object")
            df[name] = df[name].where(df[name].notna(), "__MISSING__")
        else:
            df[name] = pd.to_numeric(df[name], errors="coerce")
            imp = SimpleImputer(strategy="median")
            df[name] = imp.fit_transform(df[[name]]).ravel()
    yname, _ = _target_info(meta, df)
    df = df[df[yname].notna()]
    return df

# -----------------------------
# Candidate generation
# -----------------------------

def generate_candidates(
    df: pd.DataFrame,
    meta: Mapping[str, Any],
    max_interactions: int = 5,
) -> Tuple[List[str], List[str]]:
    lookup = _column_lookup(meta)
    preds = [name for name, col in lookup.items() if _column_role(col) == "predictor"]

    main_terms = [_var_term(name, lookup[name]) for name in preds]

    cand_pairs: List[Sequence[str]] = []
    for a, b in itertools.combinations(preds, 2):
        ma, mb = lookup[a], lookup[b]
        ca, cb = _cardinality(df, a, ma), _cardinality(df, b, mb)
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

    cand_pairs = cand_pairs[:max_interactions]

    inter_terms = []
    for a, b in cand_pairs:
        ta = _var_term(a, lookup[a])
        tb = _var_term(b, lookup[b])
        inter_terms.append(f"{ta}:{tb}")
    return main_terms, inter_terms

# -----------------------------
# Screening model
# -----------------------------

def screen_terms(df, yname, target_type, full_formula, *, cv=5):
    # Build design (RHS only) and use raw target column to avoid patsy expanding it
    X = patsy.dmatrix(full_formula, df, return_type="dataframe")
    term_slices = X.design_info.term_slices  # mapping term -> slice of columns
    term_names = X.design_info.term_names

    y_series = df[yname]
    if target_type in ("binary", "multiclass", "count"):
        if pd.api.types.is_categorical_dtype(y_series):
            y_series = y_series.cat.codes
        y_values = pd.to_numeric(y_series, errors="coerce").fillna(0).astype(int)
    else:
        y_values = pd.to_numeric(y_series, errors="coerce")

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
        est.fit(X_scaled, y_values.values.ravel())
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
        est.fit(X_scaled, y_values.values.ravel())
        coef = est.coef_  # (K, n_features)
        nz = np.where((np.abs(coef) > 1e-8).any(axis=0))[0]

    elif target_type == "continuous":
        est = LassoCV(alphas=None, cv=cv, max_iter=5000, fit_intercept=False)
        est.fit(X_scaled, y_values.values.ravel())
        coef = est.coef_
        nz = np.where(np.abs(coef) > 1e-8)[0]

    elif target_type == "count":
        # elastic-net with l1_ratio=1 => lasso; simple CV on alpha
        grid = {"alpha": np.logspace(-4, 1, 12)}
        base = PoissonRegressor(max_iter=2000, fit_intercept=False, alpha=1.0, l1_ratio=1.0, verbose=0)
        est = GridSearchCV(base, grid, cv=cv, scoring="neg_mean_poisson_deviance")
        est.fit(X_scaled, y_values.values.ravel())
        coef = est.best_estimator_.coef_
        nz = np.where(np.abs(coef) > 1e-8)[0]
    else:
        raise ValueError("target_type must be one of: binary, multiclass, continuous, count")

    # Map selected columns → terms
    selected_terms: set[str] = set()
    for tname, sl in term_slices.items():
        idxs = range(sl.start, sl.stop)
        if any(i in nz for i in idxs):
            selected_terms.add(str(tname))

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

def auto_formula(
    df: pd.DataFrame,
    meta: Mapping[str, Any],
    *,
    max_interactions: int = 5,
    cv: int = 5,
) -> str:
    df0 = _replace_missing_codes(df, meta)
    df0 = _coerce_dtypes_and_levels(df0, meta, fill_cats_with_missing_token=True)
    df0 = _fill_for_screening(df0, meta)

    main_terms, inter_terms = generate_candidates(df0, meta, max_interactions=max_interactions)
    full_formula = " + ".join(main_terms + inter_terms)

    yname, target_type = _target_info(meta, df0)

    selected_terms = screen_terms(df0, yname, target_type, full_formula, cv=cv)
    return formula_from_selected(yname, selected_terms)


def fit_with_mi(
    df: pd.DataFrame,
    formula: str,
    meta: Mapping[str, Any],
    *,
    m: int = 20,
    burnin: int = 5,
):
    yname, target_type = _target_info(meta, df)

    df_mi = _replace_missing_codes(df, meta)
    df_mi = _coerce_dtypes_and_levels(df_mi, meta, fill_cats_with_missing_token=False)
    cat_cols = df_mi.select_dtypes(include="category").columns
    for col in cat_cols:
        codes = df_mi[col].cat.codes.astype("float64")
        df_mi[col] = codes.where(codes >= 0, np.nan)

    if target_type == "multiclass":
        df_single = df_mi.copy()
        for col in df_single.columns:
            if df_single[col].isna().any():
                series = df_single[col]
                if pd.api.types.is_numeric_dtype(series):
                    df_single[col] = series.fillna(series.median())
                else:
                    mode = series.mode(dropna=True)
                    fill_value = mode.iloc[0] if not mode.empty else 0
                    df_single[col] = series.fillna(fill_value)
        model = sm.MNLogit.from_formula(formula, df_single)
        return model.fit(maxiter=200, disp=False)

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


