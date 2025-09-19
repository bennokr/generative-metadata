from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import logging

import numpy as np
import pandas as pd

from .utils import infer_types


_SEMMAP_TITLE_KEYS = [
    "dct:title",
    "http://purl.org/dc/terms/title",
    "title",
    "name",
]
_SEMMAP_DESCRIPTION_KEYS = [
    "dct:description",
    "http://purl.org/dc/terms/description",
    "description",
    "summary",
]
_SEMMAP_LICENSE_KEYS = [
    "dct:license",
    "http://purl.org/dc/terms/license",
    "license",
]
_SEMMAP_REP_KEYS = [
    "http://rdf-vocabulary.ddialliance.org/discovery#representation",
    "disco:representation",
    "representation",
]
_SEMMAP_PREFLABEL_KEYS = [
    "skos:prefLabel",
    "prefLabel",
    "label",
]
_SEMMAP_NOTATION_KEYS = [
    "skos:notation",
    "notation",
    "name",
    "termCode",
    "code",
]
_SEMMAP_HAS_TOP_KEYS = [
    "skos:hasTopConcept",
    "hasTopConcept",
]
_SEMMAP_EXACT_MATCH_KEYS = [
    "skos:exactMatch",
    "exactMatch",
]
_SEMMAP_UNIT_TEXT_KEYS = [
    "semmap:pintUnit",
    "pintUnit",
]
_SEMMAP_UNIT_CODE_KEYS = [
    "qudt:hasUnit",
    "hasUnit",
]


def _first_semmap_value(data: Optional[Dict[str, Any]], keys: Sequence[str]) -> Any:
    if not isinstance(data, dict):
        return None
    for key in keys:
        if key in data:
            return data[key]
    return None


def _first_semmap_str(data: Optional[Dict[str, Any]], keys: Sequence[str]) -> Optional[str]:
    val = _first_semmap_value(data, keys)
    if isinstance(val, str) and val.strip():
        return val.strip()
    if isinstance(val, (list, tuple)):
        for entry in val:
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
    return None


def _normalize_semmap_types(rep: Any) -> List[str]:
    if isinstance(rep, list):
        return [str(x) for x in rep]
    if rep is None:
        return []
    return [str(rep)]


def _enrich_variable_with_semmap(item: Dict[str, Any], col_meta: Dict[str, Any]) -> None:
    pref_label = _first_semmap_str(col_meta, _SEMMAP_PREFLABEL_KEYS)
    long_desc = _first_semmap_str(col_meta, _SEMMAP_DESCRIPTION_KEYS)
    if long_desc:
        item["description"] = long_desc
    elif pref_label and pref_label.strip() and pref_label.strip().lower() != str(item.get("name", "")).lower():
        item["description"] = pref_label
    if pref_label:
        item.setdefault("alternateName", pref_label)

    rep = _first_semmap_value(col_meta, _SEMMAP_REP_KEYS)
    if isinstance(rep, dict):
        pint_unit = _first_semmap_str(rep, _SEMMAP_UNIT_TEXT_KEYS)
        qudt_unit = _first_semmap_str(rep, _SEMMAP_UNIT_CODE_KEYS)
        if pint_unit:
            item["unitText"] = pint_unit
        if qudt_unit:
            item["unitCode"] = qudt_unit

        rep_types = _normalize_semmap_types(rep.get("@type"))
        if any("ConceptScheme" in t for t in rep_types):
            item["measurementTechnique"] = "categorical"
            concepts = _first_semmap_value(rep, _SEMMAP_HAS_TOP_KEYS)
            values: List[Dict[str, Any]] = []
            if isinstance(concepts, list):
                for concept in concepts[:5]:
                    if not isinstance(concept, dict):
                        continue
                    entry: Dict[str, Any] = {"@type": "DefinedTerm"}
                    label = _first_semmap_str(concept, _SEMMAP_PREFLABEL_KEYS)
                    notation = _first_semmap_str(concept, _SEMMAP_NOTATION_KEYS)
                    if label:
                        entry["name"] = label
                    if notation:
                        entry["termCode"] = notation
                    exact = _first_semmap_value(concept, _SEMMAP_EXACT_MATCH_KEYS)
                    if isinstance(exact, str) and exact.strip():
                        entry["url"] = exact.strip()
                    elif isinstance(exact, list):
                        for candidate in exact:
                            if isinstance(candidate, str) and candidate.strip():
                                entry["url"] = candidate.strip()
                                break
                    if len(entry) > 1:
                        values.append(entry)
            if values:
                item["valueReference"] = values
        else:
            item["measurementTechnique"] = "continuous"

def _string_attributes(obj: Any) -> Dict[str, str]:
    return {
        name: value
        for name in dir(obj)
        if not name.startswith("_") and isinstance((value := getattr(obj, name, None)), str)
    }


def _dict_attributes(obj: Any) -> Dict[str, Dict[str, Any]]:
    dict_atts: Dict[str, Dict[str, Any]] = {
        name: value
        for name in dir(obj)
        if not name.startswith("_") and isinstance((value := getattr(obj, name, None)), dict)
    }
    # Ensure nested dict values are JSON-serializable
    for key, d in list(dict_atts.items()):
        for k, v in d.items():
            if not isinstance(v, (int, str, float)):
                d[k] = _string_attributes(v)
        # Convert integer-keyed dicts to list for stability
        if all(isinstance(k, int) for k in d.keys()):
            dict_atts[key] = list(d.values())  # type: ignore[assignment]
    return dict_atts


def meta_to_dict(meta: Any) -> Dict[str, Any]:
    """Best-effort conversion of provider metadata object to a plain dict.

    It first tries dict(meta). If that fails, it collects string-valued
    attributes and top-level dict-valued attributes from the object.
    """
    try:
        return dict(meta)  # type: ignore[arg-type]
    except Exception:
        d: Dict[str, Any] = _string_attributes(meta)
        d.update(**_dict_attributes(meta))
        return d


def _normalize_creators(val: Any) -> List[Dict[str, str]]:
    creators: List[Dict[str, str]] = []
    if val is None:
        return creators
    if isinstance(val, str):
        parts = [p.strip() for p in val.replace(";", ",").split(",") if p.strip()]
        for p in parts:
            creators.append({"@type": "Person", "name": p})
    elif isinstance(val, (list, tuple, set)):
        for item in val:
            if isinstance(item, dict):
                nm = item.get("name") or item.get("fullname") or item.get("full_name")
                if not nm:
                    nm = " ".join(filter(None, [item.get("givenName"), item.get("familyName")])).strip()
                if nm:
                    creators.append({"@type": item.get("@type") or "Person", "name": nm})
            else:
                creators.append({"@type": "Person", "name": str(item)})
    elif isinstance(val, dict):
        nm = val.get("name") or val.get("fullname") or val.get("full_name")
        if nm:
            creators.append({"@type": val.get("@type") or "Person", "name": nm})
    return creators


def _extract_doi(texts: Any) -> List[str]:
    import re

    dois: List[str] = []
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


def build_dataset_jsonld(
    name: str,
    meta_dict: Dict[str, Any],
    df: pd.DataFrame,
    *,
    semmap_dataset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a schema.org/Dataset JSON-LD dictionary from metadata and data.

    Attempts to derive provider URL and identifier (OpenML/UCI) and includes a
    per-variable measurementTechnique based on inferred types. When SemMap
    metadata is supplied, dataset- and variable-level entries are enriched with
    its labels, descriptions, units, and value definitions.
    """
    provider_url = meta_dict.get("url") or meta_dict.get("original_data_url")
    openml_id = meta_dict.get("dataset_id") or meta_dict.get("did")
    if not provider_url and openml_id:
        provider_url = f"https://www.openml.org/d/{openml_id}"
    # Resolve UCI dataset id from common keys
    uci_id = None
    _raw_uci = meta_dict.get("id") if meta_dict.get("id") is not None else meta_dict.get("uci_id")
    try:
        if isinstance(_raw_uci, (int, np.integer)):
            uci_id = int(_raw_uci)
        elif isinstance(_raw_uci, str) and _raw_uci.isdigit():
            uci_id = int(_raw_uci)
    except Exception:
        uci_id = None
    if not provider_url and uci_id:
        provider_url = f"https://archive.ics.uci.edu/dataset/{uci_id}"

    description = meta_dict.get("description") or meta_dict.get("abstract") or meta_dict.get("summary")
    semmap_description = _first_semmap_str(semmap_dataset, _SEMMAP_DESCRIPTION_KEYS)
    if semmap_description:
        description = semmap_description
    citation = meta_dict.get("citation") or meta_dict.get("bibliography")
    # Build citation from uciml metadata if available and missing
    if not citation and isinstance(meta_dict.get("intro_paper"), dict):
        ip = meta_dict["intro_paper"]
        parts: List[str] = []
        if ip.get("title"):
            parts.append(ip["title"])
        if ip.get("authors"):
            parts.append(ip["authors"])
        if ip.get("venue"):
            parts.append(ip["venue"])
        if ip.get("year"):
            parts.append(str(ip["year"]))
        if ip.get("DOI"):
            parts.append("https://doi.org/" + str(ip["DOI"]).replace("https://doi.org/", ""))
        citation = ". ".join([p for p in parts if p])

    creators_val = (
        meta_dict.get("creators")
        or meta_dict.get("creator")
        or meta_dict.get("author")
        or meta_dict.get("donor")
    )
    creators = _normalize_creators(creators_val)
    date_published = (
        meta_dict.get("upload_date")
        or meta_dict.get("collection_date")
        or meta_dict.get("year")
        or meta_dict.get("date")
    )
    same_as: List[str] = []
    if meta_dict.get("doi"):
        doi_val = meta_dict.get("doi")
        if isinstance(doi_val, str):
            same_as.append("https://doi.org/" + doi_val.replace("https://doi.org/", "").strip())
        elif isinstance(doi_val, (list, tuple)):
            for s in doi_val:
                same_as.append("https://doi.org/" + str(s).replace("https://doi.org/", "").strip())
    same_as.extend(_extract_doi([citation, provider_url]))

    dataset_title = _first_semmap_str(semmap_dataset, _SEMMAP_TITLE_KEYS) or name
    license_val = _first_semmap_str(semmap_dataset, _SEMMAP_LICENSE_KEYS) or meta_dict.get("license")

    # variableMeasured with inferred categorical/continuous (and uciml descriptions when available)
    variable_measured: List[Dict[str, Any]] = []
    try:
        disc_cols_tmp, _cont_cols_tmp = infer_types(df)
        # If UCI metadata is available, fetch variable descriptions and match to df columns (case-insensitive)
        var_desc_map_lc: Dict[str, str] = {}
        try:
            # Use UCI dataset id if present to fetch cached descriptions
            _raw_uci = meta_dict.get("id") if meta_dict.get("id") is not None else meta_dict.get("uci_id")
            _uci_id = None
            if isinstance(_raw_uci, (int, np.integer)):
                _uci_id = int(_raw_uci)
            elif isinstance(_raw_uci, str) and _raw_uci.isdigit():
                _uci_id = int(_raw_uci)
            if _uci_id:
                for k, v in get_uciml_variable_descriptions(_uci_id).items():  # type: ignore[name-defined]
                    if isinstance(k, str) and isinstance(v, str) and v.strip():
                        var_desc_map_lc[k.lower()] = v.strip()
        except Exception:
            pass
        for c in df.columns:
            mt = "discrete" if c in disc_cols_tmp else "continuous"
            item: Dict[str, Any] = {"@type": "PropertyValue", "name": c, "measurementTechnique": mt}
            desc = var_desc_map_lc.get(str(c).lower())
            if desc:
                item["description"] = desc
            attrs = getattr(df[c], "attrs", None)
            col_meta = attrs.get("semmap_jsonld") if isinstance(attrs, dict) else None
            if isinstance(col_meta, dict):
                _enrich_variable_with_semmap(item, col_meta)
            variable_measured.append(item)
    except Exception:
        pass

    dataset_jsonld: Dict[str, Any] = {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": dataset_title,
        "identifier": openml_id or uci_id,
        "url": provider_url,
        "description": description,
        "creator": creators if creators else None,
        "citation": citation,
        "datePublished": date_published,
        "sameAs": same_as or None,
        "license": license_val,
        "variableMeasured": variable_measured or None,
    }
    # Drop empty/None entries
    dataset_jsonld = {k: v for k, v in dataset_jsonld.items() if v not in (None, [], {})}
    return dataset_jsonld


def load_uciml_cached_metadata(uci_dataset_id: int) -> Optional[Dict[str, Any]]:
    """Load UCI ML dataset metadata from the local cache file if present."""
    try:
        cache = Path(".") / "uciml-cache" / f"{uci_dataset_id}.json"
        if cache.exists():
            return json.loads(cache.read_text())
    except Exception:
        pass
    return None


def get_uciml_variable_descriptions(uci_dataset_id: Optional[int]) -> Dict[str, str]:
    """Return a mapping variable name -> description from UCI cached metadata.

    If no descriptions are available, returns an empty dict.
    """
    res: Dict[str, str] = {}
    if not uci_dataset_id:
        return res
    data = load_uciml_cached_metadata(int(uci_dataset_id))
    try:
        vars_meta = data.get("variables") if isinstance(data, dict) else None
        if isinstance(vars_meta, list):
            for v in vars_meta:
                if not isinstance(v, dict):
                    continue
                nm = v.get("name")
                desc = v.get("description")
                if nm and desc:
                    res[str(nm)] = str(desc)
    except Exception:
        pass
    return res


def get_uciml_declared_types(uci_dataset_id: Optional[int]) -> Dict[str, str]:
    """Fetch declared variable types from the UCI ML API (cached locally)."""
    declared: Dict[str, str] = {}
    if not uci_dataset_id:
        return declared
    import requests
    import json as _json

    cachedir = Path(".") / "uciml-cache"
    cachedir.mkdir(exist_ok=True)
    cache = cachedir / f"{uci_dataset_id}.json"
    data_url = "https://archive.ics.uci.edu/api/dataset"
    try:
        if not cache.exists():
            r = requests.get(data_url, params={"id": uci_dataset_id}, timeout=30)
            if r.ok:
                content = r.json().get("data")
                cache.write_text(_json.dumps(content))
        if cache.exists():
            data = _json.loads(cache.read_text())
            vars_meta = data.get("variables") or []
            for v in vars_meta:
                nm = v.get("name")
                tp = v.get("type")
                if nm:
                    declared[str(nm)] = str(tp) if tp is not None else ""
    except Exception:
        # Best-effort; swallow errors and return what we have
        pass
    return declared


def get_openml_declared_types(openml_meta_obj: Any) -> Dict[str, str]:
    """Extract declared types from an OpenML dataset metadata object (best-effort)."""
    declared: Dict[str, str] = {}
    obj = openml_meta_obj
    try:
        feats = getattr(obj, "features", None)
        if isinstance(feats, (list, tuple)) and len(feats):
            for f in feats:
                nm = getattr(f, "name", None) or getattr(f, "index", None)
                dt = getattr(f, "data_type", None) or getattr(f, "dtype", None)
                if nm is not None:
                    declared[str(nm)] = str(dt) if dt is not None else ""
        # Fallback: data_features dict mapping names to dicts
        if not declared:
            dfeat = getattr(obj, "data_features", None)
            if isinstance(dfeat, dict):
                for nm, info in dfeat.items():
                    if isinstance(info, dict):
                        dt = info.get("data_type") or info.get("type")
                    else:
                        dt = None
                    declared[str(nm)] = str(dt) if dt is not None else ""
    except Exception:
        pass
    return declared


def select_declared_types(
    *,
    provider: Optional[str],
    provider_id: Optional[int],
    meta_obj: Any,
    df_columns: List[str],
    meta_dict: Dict[str, Any],
    dataset_jsonld: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return a mapping variable->declared type, restricted to df_columns.

    Provider resolution:
      - Explicit provider argument wins if provided.
      - Otherwise, use dataset_jsonld.identifier when it's an int.
      - Otherwise, try to infer from meta_dict keys for OpenML/UCI.
    """
    declared_map: Dict[str, str] = {}
    prov_name_lower = (provider or "").lower() if provider else None
    prov_id_value: Optional[int] = provider_id

    # Try to resolve identifier from JSON-LD if available
    ident = None
    if isinstance(dataset_jsonld, dict):
        ident = dataset_jsonld.get("identifier")
    if isinstance(ident, (int, np.integer)):
        prov_id_value = int(ident)

    openml_id = meta_dict.get("dataset_id") or meta_dict.get("did")
    uci_id = meta_dict.get("id") if isinstance(meta_dict.get("id"), int) else None

    if prov_name_lower == "uciml" or (uci_id and not prov_name_lower):
        declared_map = get_uciml_declared_types(prov_id_value or uci_id)  # type: ignore[arg-type]
        prov_name_lower = "uciml"
    elif prov_name_lower == "openml" or (openml_id and not prov_name_lower):
        declared_map = get_openml_declared_types(meta_obj)
        prov_name_lower = "openml"

    if declared_map:
        # Only keep entries for present columns and preserve DataFrame column order
        declared_map = {c: declared_map.get(str(c), "") for c in df_columns}
    return declared_map


def resolve_provider_and_id(
    *,
    provider: Optional[str],
    provider_id: Optional[int],
    meta_dict: Dict[str, Any],
    dataset_jsonld: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """Return a normalized (provider_name, provider_id) for reporting and links."""
    provider_name = (provider or "").lower() if provider else None
    prov_id = provider_id
    ident = dataset_jsonld.get("identifier") if isinstance(dataset_jsonld, dict) else None
    if isinstance(ident, (int, np.integer)):
        prov_id = int(ident)
    else:
        # Fall back on raw metadata
        if isinstance(meta_dict.get("dataset_id") or meta_dict.get("did"), (int, np.integer)):
            prov_id = int(meta_dict.get("dataset_id") or meta_dict.get("did"))
            provider_name = provider_name or "openml"
        elif isinstance(meta_dict.get("uci_id") or meta_dict.get("id"), (int, np.integer)):
            prov_id = int(meta_dict.get("uci_id") or meta_dict.get("id"))
            provider_name = provider_name or "uciml"
    return provider_name, prov_id
