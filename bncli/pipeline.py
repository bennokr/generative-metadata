from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, List, Dict, Sequence, Tuple
import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import (
    coerce_continuous_to_float,
    coerce_discrete_to_category,
    infer_types,
    rename_categorical_categories_to_str,
    seed_all,
    ensure_dir,
)
from .umap_utils import build_umap, plot_umap, transform_with_umap
from .bn import learn_bn, bn_to_graphviz, save_graphml_structure
from .metrics import heldout_loglik, per_variable_distances
from .reporting import write_report_md
from metasyn.metaframe import MetaFrame


class Config:
    random_state = 42
    max_umap_sample = 1000
    synthetic_sample = 1000
    test_size = 0.2
    umap_n_neighbors = 30
    umap_min_dist = 0.1
    umap_n_components = 2

def string_attributes(obj):
    return {
        name: value
        for name in dir(obj)
        if not name.startswith("_")  # skip dunder attributes
        and isinstance((value := getattr(obj, name, None)), str)
    }

def dict_attributes(obj):
    dict_atts = {
        name: value
        for name in dir(obj)
        if not name.startswith("_")  # skip dunder attributes
        and isinstance((value := getattr(obj, name, None)), dict)
    }
    for key, d in list(dict_atts.items()):
        for k,v in d.items():
            if not isinstance(v, (int, str, float)):
                d[k] = string_attributes(v)
        if all(isinstance(k, int) for k in d.keys()):
            dict_atts[key] = list(d.values())
    return dict_atts



def process_dataset(
    meta: Any,
    df: pd.DataFrame,
    color_series: Optional[pd.Series],
    base_outdir: str,
    *,
    provider: Optional[str] = None,
    provider_id: Optional[int] = None,
    bn_configs: Optional[List[Dict[str, Any]]] = None,
    roots: Optional[List[str]] = None,
    cfg: Config = Config(),
) -> None:
    name = getattr(meta, 'name', None) or (meta.get('name') if isinstance(meta, dict) else None) or 'dataset'
    logging.info(f"Processing dataset: {name}")

    outdir = Path(base_outdir) / name.replace("/", "_")
    ensure_dir(str(outdir))

    metadata_file = outdir / "metadata.json"
    try:
        meta_dict = dict(meta)
    except:
        meta_dict = string_attributes(meta)
        meta_dict.update(**dict_attributes(meta))
    with metadata_file.open('w', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2)
    logging.info(f"Wrote raw metadata JSON: {metadata_file}")

    # Build schema.org/Dataset JSON-LD
    def normalize_creators(val):
        creators = []
        if val is None:
            return creators
        if isinstance(val, str):
            parts = [p.strip() for p in val.replace(';', ',').split(',') if p.strip()]
            for p in parts:
                creators.append({"@type": "Person", "name": p})
        elif isinstance(val, (list, tuple, set)):
            for item in val:
                if isinstance(item, dict):
                    nm = item.get('name') or item.get('fullname') or item.get('full_name')
                    if not nm:
                        nm = " ".join(filter(None, [item.get('givenName'), item.get('familyName')])).strip()
                    if nm:
                        creators.append({"@type": item.get('@type') or "Person", "name": nm})
                else:
                    creators.append({"@type": "Person", "name": str(item)})
        elif isinstance(val, dict):
            nm = val.get('name') or val.get('fullname') or val.get('full_name')
            if nm:
                creators.append({"@type": val.get('@type') or "Person", "name": nm})
        return creators

    def extract_doi(texts):
        import re
        dois = []
        if texts is None:
            return dois
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        pat = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
        for t in texts:
            if not isinstance(t, str):
                continue
            for m in pat.findall(t):
                dois.append("https://doi.org/" + m)
        # dedupe
        return list(dict.fromkeys(dois))

    provider_url = meta_dict.get('url') or meta_dict.get('original_data_url')
    openml_id = meta_dict.get('dataset_id') or meta_dict.get('did')
    if not provider_url and openml_id:
        provider_url = f"https://www.openml.org/d/{openml_id}"
    uci_id = meta_dict.get('id') if isinstance(meta_dict.get('id'), int) else None
    if not provider_url and uci_id:
        provider_url = f"https://archive.ics.uci.edu/dataset/{uci_id}"

    description = meta_dict.get('description') or meta_dict.get('abstract') or meta_dict.get('summary')
    citation = meta_dict.get('citation') or meta_dict.get('bibliography')
    # Build citation from uciml metadata if available and missing
    if not citation and isinstance(meta_dict.get('intro_paper'), dict):
        ip = meta_dict['intro_paper']
        parts = []
        if ip.get('title'):
            parts.append(ip['title'])
        if ip.get('authors'):
            parts.append(ip['authors'])
        if ip.get('venue'):
            parts.append(ip['venue'])
        if ip.get('year'):
            parts.append(str(ip['year']))
        if ip.get('DOI'):
            parts.append('https://doi.org/' + str(ip['DOI']).replace('https://doi.org/',''))
        citation = '. '.join([p for p in parts if p])
    creators_val = meta_dict.get('creators') or meta_dict.get('creator') or meta_dict.get('author') or meta_dict.get('donor')
    creators = normalize_creators(creators_val)
    date_published = meta_dict.get('upload_date') or meta_dict.get('collection_date') or meta_dict.get('year') or meta_dict.get('date')
    same_as = []
    if meta_dict.get('doi'):
        doi_val = meta_dict.get('doi')
        if isinstance(doi_val, str):
            same_as.append('https://doi.org/' + doi_val.replace('https://doi.org/', '').strip())
        elif isinstance(doi_val, (list, tuple)):
            for s in doi_val:
                same_as.append('https://doi.org/' + str(s).replace('https://doi.org/', '').strip())
    same_as.extend(extract_doi([citation, provider_url]))

    variable_measured = []
    try:
        disc_cols_tmp, _cont_cols_tmp = infer_types(df)
        for c in df.columns:
            mt = 'discrete' if c in disc_cols_tmp else 'continuous'
            variable_measured.append({"@type": "PropertyValue", "name": c, "measurementTechnique": mt})
    except Exception:
        pass

    dataset_jsonld = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": name,
        "identifier": openml_id or uci_id,
        "url": provider_url,
        "description": description,
        "creator": creators if creators else None,
        "citation": citation,
        "datePublished": date_published,
        "sameAs": same_as or None,
        "variableMeasured": variable_measured or None,
    }
    dataset_jsonld = {k: v for k, v in dataset_jsonld.items() if v not in (None, [], {})}
    dataset_jsonld_file = outdir / 'dataset.jsonld'
    with dataset_jsonld_file.open('w', encoding='utf-8') as fw:
        json.dump(dataset_jsonld, fw, indent=2)
    logging.info(f"Wrote JSON-LD metadata: {dataset_jsonld_file}")


    logging.info("Inferring column types (discrete vs continuous)")
    disc_cols, cont_cols = infer_types(df)
    logging.info(f"Detected columns — discrete: {len(disc_cols)}, continuous: {len(cont_cols)}")
    df = coerce_discrete_to_category(df, disc_cols)
    df = rename_categorical_categories_to_str(df, disc_cols)
    df = coerce_continuous_to_float(df, cont_cols)

    if len(disc_cols) == 0:
        cand = [c for c in df.columns if c not in disc_cols and pd.api.types.is_integer_dtype(df[c])]
        cand = sorted(cand, key=lambda c: df[c].nunique(dropna=True))[:3]
        if cand:
            df[cand] = df[cand].astype("category")
            disc_cols, cont_cols = infer_types(df)
            df = rename_categorical_categories_to_str(df, disc_cols)
            df = coerce_continuous_to_float(df, cont_cols)

    # Baseline summary using pandas describe (transposed so variables are rows)
    logging.info("Computing baseline summary via pandas describe")
    baseline_df = df.describe(include='all').transpose()
    baseline_df.index.name = 'variable'

    # Build declared and inferred type maps for reporting
    inferred_map = {c: ("discrete" if c in disc_cols else "continuous") for c in df.columns}

    def get_uciml_declared_types(uci_dataset_id: Optional[int]) -> dict:
        declared = {}
        if not uci_dataset_id:
            return declared
        import requests
        import json as _json
        import pathlib as _pl
        cachedir = _pl.Path('.') / 'uciml-cache'
        cachedir.mkdir(exist_ok=True)
        cache = cachedir / f"{uci_dataset_id}.json"
        data_url = "https://archive.ics.uci.edu/api/dataset"
        try:
            if not cache.exists():
                r = requests.get(data_url, params={'id': uci_dataset_id}, timeout=30)
                if r.ok:
                    content = r.json().get('data')
                    cache.write_text(_json.dumps(content))
            if cache.exists():
                data = _json.loads(cache.read_text())
                vars_meta = data.get('variables') or []
                for v in vars_meta:
                    nm = v.get('name')
                    tp = v.get('type')
                    if nm:
                        declared[str(nm)] = str(tp) if tp is not None else ''
        except Exception:
            # Best-effort; swallow errors and return what we have
            pass
        return declared

    def get_openml_declared_types(openml_meta_obj: Any) -> dict:
        declared = {}
        obj = openml_meta_obj
        try:
            # Try OpenML Python dataset object's features list
            feats = getattr(obj, 'features', None)
            if feats is not None:
                for f in feats:
                    # Common attributes on Feature objects: name, data_type
                    nm = getattr(f, 'name', None) or getattr(f, 'index', None)
                    dt = getattr(f, 'data_type', None) or getattr(f, 'dtype', None)
                    if nm is not None:
                        declared[str(nm)] = str(dt) if dt is not None else ''
            # Fallback: data_features dict mapping names to dicts
            if not declared:
                dfeat = getattr(obj, 'data_features', None)
                if isinstance(dfeat, dict):
                    for nm, info in dfeat.items():
                        if isinstance(info, dict):
                            dt = info.get('data_type') or info.get('type')
                        else:
                            dt = None
                        declared[str(nm)] = str(dt) if dt is not None else ''
        except Exception:
            pass
        return declared

    declared_map = {}
    prov_name_lower = (provider or '').lower() if provider else None
    prov_id_value = provider_id
    # Try to resolve provider_name/id similarly to how we prepare links below
    ident = dataset_jsonld.get('identifier') if isinstance(dataset_jsonld, dict) else None
    if isinstance(ident, (int, np.integer)):
        prov_id_value = int(ident)
    if prov_name_lower == 'uciml' or (uci_id and not prov_name_lower):
        declared_map = get_uciml_declared_types(prov_id_value or uci_id)
        prov_name_lower = 'uciml'
    elif prov_name_lower == 'openml' or (openml_id and not prov_name_lower):
        declared_map = get_openml_declared_types(meta)
        prov_name_lower = 'openml'
    # Only keep declared entries for present columns; preserve order of df columns
    if declared_map:
        declared_map = {c: declared_map.get(str(c), '') for c in df.columns}

    logging.info("Dropping rows with any NA for modeling")
    df_no_na = df.dropna(axis=0, how="any").reset_index(drop=True)
    train_df, test_df = train_test_split(
        df_no_na, test_size=cfg.test_size, random_state=cfg.random_state, shuffle=True
    )
    logging.info(f"Train/test split sizes: {len(train_df)}/{len(test_df)}")

    # Determine arc blacklist variables
    default_root = ["age", "sex", "race"]
    root_vars: List[str]
    if isinstance(roots, (list, tuple)) and len(roots):
        root_vars = [str(x) for x in roots]
    else:
        demo = meta_dict.get('demographics')
        if isinstance(demo, (list, tuple)):
            root_vars = [str(x) for x in demo]
        elif isinstance(demo, dict):
            root_vars = [str(k) for k in demo.keys()]
        else:
            root_vars = default_root
    logging.info(f'Using {root_vars=}')

    # Build arc blacklist pairs: forbid arcs INTO root vars FROM non-root variables only
    cols = list(df.columns)
    col_map = {str(c).lower(): c for c in cols}
    sens_in_cols = []
    for s in root_vars:
        t = str(s).lower()
        if t in col_map:
            sens_in_cols.append(col_map[t])
    logging.info(f'Using {sens_in_cols=}')
    arc_blacklist_pairs: List[Tuple[str, str]] = []
    for u in sens_in_cols:
        for v in cols:
            arc_blacklist_pairs.append((v, u))

    # BN configurations to run
    if isinstance(bn_configs, (list, tuple)) and len(bn_configs):
        configs_to_run = list(bn_configs)
    else:
        # Default: two configs that actually differ
        configs_to_run = [
            dict(name="clg_mi2", bn_type="clg", score="bic", operators=["arcs"], max_indegree=2),
            dict(name="semi_mi5", bn_type="semiparametric", score="bic", operators=["arcs"], max_indegree=5),
        ]

    # Placeholder to collect BN-specific results for reporting
    bn_sections: List[Dict[str, Any]] = []


    rng = seed_all(cfg.random_state)
    color_series2 = None
    if isinstance(color_series, pd.Series) and color_series.name in df_no_na.columns:
        color_series2 = df_no_na[color_series.name]
    logging.info("Fitting UMAP on real data sample")
    umap_art = build_umap(
        df_no_na,
        disc_cols,
        cont_cols,
        color_series=color_series2,
        rng=rng,
        random_state=cfg.random_state,
        max_sample=cfg.max_umap_sample,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        n_components=cfg.umap_n_components,
    )
    umap_png_real = outdir / "umap_real.png"
    plot_umap(umap_art.embedding, str(umap_png_real), title=f"{name}: real (sample)", color_labels=umap_art.color_labels)

    # Learn and evaluate each BN configuration
    for idx, cfg_item in enumerate(configs_to_run):
        bn_type = str(cfg_item.get("bn_type", "clg"))
        score = cfg_item.get("score")
        operators = cfg_item.get("operators")
        max_indegree = cfg_item.get("max_indegree")
        seed = int(cfg_item.get("seed", cfg.random_state))
        label = cfg_item.get("name") or f"{bn_type}_{idx+1}"
        logging.info(f"Learning BN model (label={label}, type={bn_type}, score={score}, max_indegree={max_indegree}, seed={seed})")
        bn_art = learn_bn(
            train_df,
            bn_type=bn_type,
            random_state=seed,
            arc_blacklist=arc_blacklist_pairs,
            score=score,
            operators=operators,
            max_indegree=max_indegree,
        )
        model = bn_art.model
        node_types = model.node_types()

        bn_png = outdir / f"bn_{label}.png"
        bn_to_graphviz(model, node_types, str(bn_png), title=f"{name} — {label} BN")

        logging.info(f"Sampling BN synthetic n={cfg.synthetic_sample} for {label}")
        synth = model.sample(cfg.synthetic_sample, seed=seed)
        synth_df = synth.to_pandas()

        logging.info("Computing BN held-out log-likelihood")
        ll = heldout_loglik(model, test_df)
        synth_df = synth_df[df_no_na.columns]
        for c in disc_cols:
            if c in synth_df.columns:
                synth_df[c] = synth_df[c].astype("category")
                if pd.api.types.is_categorical_dtype(df_no_na[c]):
                    synth_df[c] = synth_df[c].cat.set_categories(df_no_na[c].cat.categories)
        logging.info("Per-variable distances for BN computed")
        dist_table_bn = per_variable_distances(test_df, synth_df, bn_art.discrete_cols, bn_art.continuous_cols)

        # UMAP transform for this BN synthetic
        synth_no_na = synth_df.dropna(axis=0, how="any")
        synth_emb = transform_with_umap(umap_art, synth_no_na)
        umap_png_bn = outdir / f"umap_bn_{label}.png"
        plot_umap(synth_emb, str(umap_png_bn), title=f"{name}: synthetic (BN sample: {label})")

        graphml_file = outdir / f"structure_{label}.graphml"
        save_graphml_structure(model, node_types, graphml_file)
        pickle_file = outdir / f"model_{label}.pickle"
        model.save(str(pickle_file))

        bn_sections.append(
            dict(
                label=label,
                bn_type=bn_type,
                bn_png=str(bn_png),
                ll_metrics=ll,
                dist_table=dist_table_bn,
                graphml_file=str(graphml_file),
                pickle_file=str(pickle_file),
                umap_png=str(umap_png_bn),
                params=dict(
                    bn_type=bn_type,
                    score=score or "bic",
                    operators=list(operators) if operators is not None else ["arcs"],
                    max_indegree=int(max_indegree) if max_indegree is not None else 5,
                    seed=seed,
                ),
            )
        )

    # ------------- MetaSyn fit and synthesize -------------
    # Fit MetaFrame (GMF) on the same training data
    logging.info("Fitting MetaSyn MetaFrame on train")
    mf = MetaFrame.fit_dataframe(train_df)
    metasyn_gmf = outdir / "metasyn_gmf.json"
    try:
        mf.save(str(metasyn_gmf))
    except Exception as e:
        logging.warning(f"Could not save MetaSyn GMF: {e}")
        metasyn_gmf = None

    # Synthesize MetaSyn data
    try:
        synth_meta = mf.synthesize(n=cfg.synthetic_sample)
        import polars as pl
        if isinstance(synth_meta, pl.DataFrame):
            synth_meta_df = synth_meta.to_pandas()
        else:
            synth_meta_df = synth_meta
    except Exception as e:
        logging.warning(f"MetaSyn synthesis failed: {e}")
        synth_meta_df = pd.DataFrame(columns=df_no_na.columns)

    # Align columns and dtypes
    synth_meta_df = synth_meta_df.reindex(columns=df_no_na.columns)
    for c in disc_cols:
        if c in synth_meta_df.columns:
            synth_meta_df[c] = synth_meta_df[c].astype("category")
            if pd.api.types.is_categorical_dtype(df_no_na[c]):
                synth_meta_df[c] = synth_meta_df[c].cat.set_categories(df_no_na[c].cat.categories)
    synth_meta_df = coerce_continuous_to_float(synth_meta_df, cont_cols)

    # Distances for MetaSyn model (use overall discrete/continuous cols inferred earlier)
    dist_table_meta = per_variable_distances(test_df, synth_meta_df, disc_cols, cont_cols)

    # UMAP transform for metasyn
    synth_meta_no_na = synth_meta_df.dropna(axis=0, how="any")
    synth_meta_emb = transform_with_umap(umap_art, synth_meta_no_na)
    umap_png_meta = outdir / "umap_metasyn.png"
    plot_umap(synth_meta_emb, str(umap_png_meta), title=f"{name}: synthetic (MetaSyn sample)")

    # Fidelity summary table
    def summarize_distances(dt: pd.DataFrame) -> dict:
        d_disc = dt[dt['type'] == 'discrete'] if not dt.empty else pd.DataFrame()
        d_cont = dt[dt['type'] == 'continuous'] if not dt.empty else pd.DataFrame()
        res = {
            'disc_jsd_mean': float(d_disc['JSD'].mean()) if ('JSD' in d_disc.columns and len(d_disc)) else float('nan'),
            'disc_jsd_median': float(d_disc['JSD'].median()) if ('JSD' in d_disc.columns and len(d_disc)) else float('nan'),
            'cont_ks_mean': float(d_cont['KS'].mean()) if ('KS' in d_cont.columns and len(d_cont)) else float('nan'),
            'cont_w1_mean': float(d_cont['W1'].mean()) if ('W1' in d_cont.columns and len(d_cont)) else float('nan'),
        }
        return res

    meta_summary = summarize_distances(dist_table_meta)
    fidelity_rows = []
    for sect in bn_sections:
        llm = sect['ll_metrics']
        bnsum = summarize_distances(sect['dist_table'])
        fidelity_rows.append(
            dict(model=f"BN:{sect.get('label') or sect['bn_type']}", mean_loglik=llm['mean_loglik'], std_loglik=llm['std_loglik'], sum_loglik=llm['sum_loglik'], **bnsum)
        )
    fidelity_rows.append(
        dict(model='MetaSyn', mean_loglik=float('nan'), std_loglik=float('nan'), sum_loglik=float('nan'), **meta_summary)
    )
    fidelity_table = pd.DataFrame(fidelity_rows)

    # Determine provider and dataset page links
    provider_name = (provider or '').lower() if provider else None
    prov_id = provider_id
    ident = dataset_jsonld.get('identifier') if isinstance(dataset_jsonld, dict) else None
    if isinstance(ident, (int, np.integer)):
        prov_id = int(ident)
    else:
        # Fall back on metadata
        if isinstance(meta_dict.get('dataset_id') or meta_dict.get('did'), (int, np.integer)):
            prov_id = int(meta_dict.get('dataset_id') or meta_dict.get('did'))
            provider_name = provider_name or 'openml'
        elif isinstance(meta_dict.get('uci_id') or meta_dict.get('id'), (int, np.integer)):
            prov_id = int(meta_dict.get('uci_id') or meta_dict.get('id'))
            provider_name = provider_name or 'uciml'

    write_report_md(
        outdir=outdir,
        dataset_name=name,
        metadata_file=metadata_file,
        dataset_jsonld_file=str(dataset_jsonld_file),
        dataset_jsonld=dataset_jsonld,
        dataset_provider=provider_name,
        dataset_provider_id=prov_id,
        df=df_no_na,
        disc_cols=disc_cols,
        cont_cols=cont_cols,
        baseline_df=baseline_df,
        bn_sections=bn_sections,
        dist_table_meta=dist_table_meta,
        fidelity_table=fidelity_table,
        graphml_files=[sect['graphml_file'] for sect in bn_sections],
        pickle_files=[sect['pickle_file'] for sect in bn_sections],
        roots_info=dict(
            root_variables=sens_in_cols,
            n_forbidden_arcs=len(arc_blacklist_pairs),
        ),
        umap_png_real=str(umap_png_real),
        umap_png_bns=[sect['umap_png'] for sect in bn_sections],
        umap_png_meta=str(umap_png_meta),
        metasyn_gmf_file=(str(metasyn_gmf) if metasyn_gmf is not None else None),
        declared_types=declared_map or None,
        inferred_types=inferred_map or None,
    )
