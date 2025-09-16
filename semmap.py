# semmap.py
from __future__ import annotations
import json
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Optional pint support
try:
    import pint
    from pint_pandas import PintType, PintArray
    _HAVE_PINT = True
    _ureg = pint.UnitRegistry()
except Exception:  # pragma: no cover
    _HAVE_PINT = False
    _ureg = None

class Namespace(str):
    def __getattr__(self, key):
        # Avoid hijacking Python special/dunder lookups (e.g., __deepcopy__)
        if key.startswith("__") and key.endswith("__"):
            raise AttributeError(key)
        return Namespace(self + key)

    def __getitem__(self, key):
        return self + key


# ---------- Constants
SEMMAP = Namespace("https://w3id.org/semmap#")
UNIT = Namespace("http://qudt.org/vocab/unit/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
DISCO = Namespace("http://rdf-vocabulary.ddialliance.org/discovery#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DCT = Namespace("http://purl.org/dc/terms/")
DSV = Namespace("https://w3id.org/dsv-ontology#")
DCAT = Namespace("http://www.w3.org/ns/dcat#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")

# Use an external @context. JSON-LD remains valid even if keys are expanded IRIs.
JSONLD_CONTEXT_URL = "https://w3id.org/semmap/context/v1"

# Arrow metadata keys (bytes per Arrow requirements)
_DATASET_JSONLD_KEY = b"jsonld.dataset"
_COLUMN_JSONLD_KEY  = b"jsonld.column"

# Minimal QUDT→pint mapping for the example (extend as needed)

QUDT_TO_PINT = {
    UNIT["MilliM_HG"]: "mmHg",
    UNIT["MilliGM-PER-DeciL"]: "mg/dL",
    UNIT["YR"]: "year",
}
PINT_TO_QUDT = {v: k for k, v in QUDT_TO_PINT.items()}

# Our tiny namespace for implementation hints (kept in JSON-LD but harmless to others)


# ---------- Helpers: JSON-LD builders (blank-node style)

def _ld_get(d: Dict[str, Any], keys: List[str], default=None):
    """Get first present key from a list of alternatives in a JSON-LD-like dict.
    This allows accepting either expanded IRIs, prefixed names (e.g., 'disco:representation'),
    or short names when supplied.
    """
    for k in keys:
        if k in d:
            return d[k]
    return default

def _jsonld_variable_base(
    notation: str,
    pref_label: str,
    source_iri: Optional[str] = None,
) -> Dict[str, Any]:
    # Variable as a blank node (no @id)
    obj = {
        "@context": JSONLD_CONTEXT_URL,
        "@type": [ str(DISCO.Variable), str(DSV.Column)],
        str(SKOS.notation): notation,
        str(SKOS.prefLabel): pref_label,
    }
    if source_iri:
        obj[str(DCT.source)] = source_iri
    return obj


def _jsonld_numeric_representation(
    xsd_datatype_iri: str,
    qudt_unit_iri: Optional[str],
    pint_unit: Optional[str]
) -> Dict[str, Any]:
    rep = {
        "@type": str(DISCO.Representation),
        str(DSV.valueType): xsd_datatype_iri
    }
    if qudt_unit_iri:
        rep[str(QUDT.hasUnit)] = qudt_unit_iri
    if pint_unit:
        rep[str(SEMMAP.pintUnit)] = pint_unit  # implementation hint; optional
    return rep


def _jsonld_codes_concept_scheme(
    code_label_map: Dict[Union[int, str], str],
    exact_match: Optional[Dict[Union[int, str], Union[str, List[str]]]] = None,
    scheme_source_iri: Optional[str] = None
) -> Dict[str, Any]:
    """Build a skos:ConceptScheme with top concepts as blank nodes.
       exact_match: code -> IRI or [IRI, ...]
    """
    scheme = {
        "@type": [
            str(DISCO.Representation),
            str(SKOS.ConceptScheme),
        ],
    }
    if scheme_source_iri:
        scheme[str(DCT.source)] = scheme_source_iri

    tops = []
    for code, label in code_label_map.items():
        cobj = {
            "@type": str(SKOS.Concept),
            str(SKOS.notation): str(code),
            str(SKOS.prefLabel): str(label),
        }
        if exact_match and code in exact_match:
            cobj[str(SKOS.exactMatch)] = exact_match[code]
        tops.append(cobj)
    scheme[str(SKOS.hasTopConcept)] = tops
    return scheme


def _jsonld_dataset(
    title: str,
    description: Optional[str] = None,
    license_iri: Optional[str] = None
) -> Dict[str, Any]:
    obj = {
        "@context": JSONLD_CONTEXT_URL,
        "@type": [ str(DISCO.LogicalDataSet), str(DSV.Dataset), str(DCAT.Dataset) ],
        str(DCT.title): title,
    }
    if description:
        obj[str(DCT.description)] = description
    if license_iri:
        obj[str(DCT.license)] = license_iri
    return obj


# ---------- Accessors

@pd.api.extensions.register_series_accessor("semmap")
class SemMapSeriesAccessor:
    """Series-level accessor to attach JSON-LD (blank-node) metadata."""

    def __init__(self, s: pd.Series) -> None:
        self._s = s
        if "semmap_jsonld" not in self._s.attrs:
            self._s.attrs["semmap_jsonld"] = None

    # ---- Declarative APIs

    def set_numeric(
        self,
        name: str,
        label: str,
        *,
        unit: Optional[str] = None,              # pint string ("mmHg", "mg/dL", "year")
        qudt_unit_iri: Optional[str] = None,     # QUDT IRI; auto-filled from pint if omitted and known
        xsd_datatype_iri: str = XSD.decimal,
        source_iri: Optional[str] = None,
        convert_to_pint: bool = True,
    ) -> "SemMapSeriesAccessor":
        """Attach numeric variable metadata and (optionally) convert dtype to pint."""
        # Prepare pint + QUDT
        pint_unit = unit
        if unit and not qudt_unit_iri:
            qudt_unit_iri = PINT_TO_QUDT.get(unit)

        # Build JSON-LD (blank nodes)
        var = _jsonld_variable_base(notation=name, pref_label=label, source_iri=source_iri)
        var[DISCO.representation] = \
            _jsonld_numeric_representation(xsd_datatype_iri, qudt_unit_iri, pint_unit)

        var["@context"] = JSONLD_CONTEXT_URL  # ensure top-level context
        self._s.attrs["semmap_jsonld"] = var

        # Convert to pint dtype if requested and available
        if convert_to_pint and unit and _HAVE_PINT:
            # Create PintArray if not already
            q = self._s.astype("float64").to_numpy()  # magnitudes
            q = q * _ureg.parse_units(unit)
            self._s = pd.Series(PintArray(q), index=self._s.index, name=self._s.name)
        return self

    def set_categorical(
        self,
        name: str,
        label: str,
        *,
        codes: Dict[Union[int, str], str],
        exact_match: Optional[Dict[Union[int, str], Union[str, List[str]]]] = None,
        scheme_source_iri: Optional[str] = None,
        source_iri: Optional[str] = None,
    ) -> "SemMapSeriesAccessor":
        """Attach categorical variable metadata (integer-coded or strings)."""
        var = _jsonld_variable_base(notation=name, pref_label=label, source_iri=source_iri)
        var[DISCO.representation] = _jsonld_codes_concept_scheme(
            codes, exact_match=exact_match, scheme_source_iri=scheme_source_iri
        )
        var["@context"] = JSONLD_CONTEXT_URL
        self._s.attrs["semmap_jsonld"] = var

        # Keep storage as integer or category if provided
        try:
            # If integer-coded dict, cast to category with ordered codes preserved
            if pd.api.types.is_integer_dtype(self._s.dtype):
                cat = pd.Categorical(self._s.map(lambda v: codes.get(v, v)))
                # store original code as separate attribute for lossless round-trip
                self._s = pd.Series(cat, index=self._s.index, name=self._s.name)
        except Exception:
            pass
        return self

    # ---- Introspection

    def jsonld(self) -> Optional[Dict[str, Any]]:
        return self._s.attrs.get("semmap_jsonld")

    # ---- internal hook (used by DataFrame writer)
    def _ensure_storage_for_parquet(self) -> pd.Series:
        """Ensure the physical storage is parquet-friendly (e.g., strip pint to magnitudes)."""
        s = self._s
        meta = self.jsonld() or {}
        rep = meta.get(str(DISCO.representation), {})
        pint_unit = rep.get(str(SEMMAP.pintUnit))

        if _HAVE_PINT and isinstance(s.dtype, PintType):
            # Store magnitudes; metadata carries units for reconstruction
            s = pd.Series(s.to_numpy().magnitude, index=s.index, name=s.name)
        # For categories, parquet handles pd.Categorical fine.
        return s


@pd.api.extensions.register_dataframe_accessor("semmap")
class SemMapFrameAccessor:
    """DataFrame-level accessor for dataset metadata and Parquet round-trip."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        if "semmap_jsonld" not in self._df.attrs:
            self._df.attrs["semmap_jsonld"] = None

    # ---- Dataset-level metadata

    def set_dataset(
        self,
        title: str,
        description: Optional[str] = None,
        license_iri: Optional[str] = None,
    ) -> "SemMapFrameAccessor":
        self._df.attrs["semmap_jsonld"] = _jsonld_dataset(title, description, license_iri)
        return self

    def jsonld(self) -> Optional[Dict[str, Any]]:
        return self._df.attrs.get("semmap_jsonld")

    # ---- IO: Parquet with Arrow schema/field metadata

    def to_parquet(self, path: str, *, index: bool = False, **pq_kwargs) -> None:
        """Write Parquet with JSON-LD stored in Arrow schema and fields."""
        # 1) normalize columns for parquet storage
        df_store = {}
        for col in self._df.columns:
            s_acc = self._df[col].semmap
            s_norm = s_acc._ensure_storage_for_parquet()
            df_store[col] = s_norm
        pdf = pd.DataFrame(df_store, index=self._df.index if index else None)

        # 2) convert to Arrow table
        table = pa.Table.from_pandas(pdf, preserve_index=index)

        # 3) attach column JSON-LD on each Field
        fields = []
        for field in table.schema:
            s_meta = self._df[field.name].semmap.jsonld()
            fmeta = dict(field.metadata or {})
            if s_meta is not None:
                fmeta[_COLUMN_JSONLD_KEY] = json.dumps(s_meta, ensure_ascii=False).encode("utf-8")
            fields.append(pa.field(field.name, field.type, nullable=field.nullable, metadata=fmeta))
        schema = pa.schema(fields)

        # 4) attach dataset JSON-LD on Schema
        schema_meta = dict(schema.metadata or {})
        d_meta = self.jsonld()
        if d_meta is not None:
            schema_meta[_DATASET_JSONLD_KEY] = json.dumps(d_meta, ensure_ascii=False).encode("utf-8")
        schema = schema.with_metadata(schema_meta)

        # 5) write parquet
        pq.write_table(pa.Table.from_arrays([table.column(i) for i in range(table.num_columns)],
                                            schema=schema), path, **pq_kwargs)

    @staticmethod
    def read_parquet(path: str, *, convert_pint: bool = True, **pq_kwargs) -> pd.DataFrame:
        """Read Parquet and restore JSON-LD + pint units."""
        table = pq.read_table(path, **pq_kwargs)
        schema = table.schema

        # Restore DataFrame
        df = table.to_pandas(types_mapper=None)  # leave as numeric/category

        # Restore dataset JSON-LD
        if schema.metadata and _DATASET_JSONLD_KEY in schema.metadata:
            df.attrs["semmap_jsonld"] = json.loads(schema.metadata[_DATASET_JSONLD_KEY].decode("utf-8"))

        # Restore column JSON-LD, and optionally pint dtypes
        for i, field in enumerate(schema):
            name = field.name
            if field.metadata and _COLUMN_JSONLD_KEY in field.metadata:
                col_jsonld = json.loads(field.metadata[_COLUMN_JSONLD_KEY].decode("utf-8"))
                # attach back to Series
                s = df[name]
                s.attrs["semmap_jsonld"] = col_jsonld

                # reconstruct pint dtype if present and desired
                rep = _ld_get(col_jsonld, [
                    str(DISCO.representation),
                    "disco:representation",
                    "representation",
                ], default={})
                pint_unit = _ld_get(rep, [str(SEMMAP.pintUnit), "semmap:pintUnit", "pintUnit"]) if isinstance(rep, dict) else None
                qudt_unit = _ld_get(rep, [str(QUDT.hasUnit), "qudt:hasUnit", "hasUnit"]) if isinstance(rep, dict) else None
                if convert_pint and _HAVE_PINT and (pint_unit or qudt_unit):
                    unit = pint_unit or QUDT_TO_PINT.get(qudt_unit)
                    if unit:
                        q = df[name].astype("float64").to_numpy() * _ureg.parse_units(unit)
                        df[name] = pd.Series(PintArray(q), index=df.index, name=name)

        return df

    # ---- External metadata loader
    def apply_json_metadata(
        self,
        metadata: Union[str, Dict[str, Any]],
        *,
        convert_pint: bool = True,
    ) -> "SemMapFrameAccessor":
        """Attach dataset and column JSON-LD metadata from a single JSON file/object.

        Expected structure:
          {
            "dataset": { ... JSON-LD for dataset ... },
            "columns": { "colname": { ... JSON-LD for variable/column ... }, ... }
          }
        Unknown columns are ignored. If pint units are present in column metadata
        and pint is available, columns are converted to pint dtype when
        convert_pint=True.
        """
        # Load JSON if path provided
        meta_obj: Dict[str, Any]
        if isinstance(metadata, str):
            with open(metadata, "r", encoding="utf-8") as f:
                meta_obj = json.load(f)
        else:
            meta_obj = metadata

        # Dataset-level metadata
        d_meta = meta_obj.get("dataset")
        if d_meta is not None:
            self._df.attrs["semmap_jsonld"] = d_meta

        # Column-level metadata: expect disco:variable as a list (DDI Discovery),
        # but accept legacy structures for compatibility
        var_list = _ld_get(meta_obj, [str(DISCO.variable), "disco:variable", "variable"])  # top-level
        if var_list is None and isinstance(meta_obj.get("dataset"), dict):
            var_list = _ld_get(meta_obj["dataset"], [str(DISCO.variable), "disco:variable", "variable"])  # nested

        variables: List[tuple[str, Dict[str, Any]]] = []
        if isinstance(var_list, list):
            for v in var_list:
                if not isinstance(v, dict):
                    continue
                name = _ld_get(v, [str(SKOS.notation), "skos:notation", "notation", "name"])  # how to map to column
                if not isinstance(name, str):
                    continue
                variables.append((name, v))
        elif isinstance(var_list, dict):
            for name, v in var_list.items():
                if isinstance(v, dict) and isinstance(name, str):
                    variables.append((name, v))

        for name, cmeta in variables:
            if name not in self._df.columns:
                continue
            s = self._df[name]
            s.attrs["semmap_jsonld"] = cmeta

            # reconstruct pint dtype if desired and available
            rep = _ld_get(cmeta, [
                str(DISCO.representation),
                "disco:representation",
                "representation",
            ], default={})
            pint_unit = _ld_get(rep, [str(SEMMAP.pintUnit), "semmap:pintUnit", "pintUnit"]) if isinstance(rep, dict) else None
            qudt_unit = _ld_get(rep, [str(QUDT.hasUnit), "qudt:hasUnit", "hasUnit"]) if isinstance(rep, dict) else None
            if convert_pint and _HAVE_PINT and (pint_unit or qudt_unit):
                unit = pint_unit or QUDT_TO_PINT.get(qudt_unit)
                if unit:
                    try:
                        q = self._df[name].astype("float64").to_numpy() * _ureg.parse_units(unit)
                        self._df[name] = pd.Series(PintArray(q), index=self._df.index, name=name)
                    except Exception:
                        # leave as-is if conversion fails
                        pass

        return self

    def export_json_metadata(self) -> Dict[str, Any]:
        """Export dataset and per-column JSON-LD metadata into a single JSON object.

        Structure follows DDI Discovery style for variables:
          {
            "dataset": { ... dataset JSON-LD ... },
            "disco:variable": [ { ... column JSON-LD ... }, ... ]
          }
        Only columns that have semmap_jsonld attached are included.
        """
        out: Dict[str, Any] = {}
        d = self.jsonld()
        if d is not None:
            out["dataset"] = d

        vars_list: List[Dict[str, Any]] = []
        for col in self._df.columns:
            meta = self._df[col].semmap.jsonld()
            if not isinstance(meta, dict):
                continue
            notation = _ld_get(meta, [str(SKOS.notation), "skos:notation", "notation", "name"])
            types = meta.get("@type")
            if isinstance(types, str):
                types = [types]
            types = [str(t) for t in (types or [])]
            is_dataset_like = any(
                ("LogicalDataSet" in t) or t.endswith(":Dataset") or t.endswith("/Dataset")
                for t in types
            )
            if is_dataset_like or not isinstance(notation, str):
                continue
            vars_list.append(meta)
        out["disco:variable"] = vars_list
        return out


# ---------- Usage example (run this when the file is executed as a script)
if __name__ == "__main__":
    # Example data
    df = pd.DataFrame({
        "cp": [1, 2, 3, 4, 1],             # chest pain code
        "fbs": [0, 1, 0, 0, 1],            # fasting blood sugar indicator
        "trestbps": [130, 145, 120, 180, 200],  # mmHg
        "chol": [180, 240, 199, 261, 300],      # mg/dL
    })

    # ---- Attach column metadata

    # cp — categorical (SNOMED mappings; DICOM CID 3202 as source)
    cp_codes = {
        1: "Typical angina",
        2: "Atypical angina",
        3: "Non-cardiac chest pain",
        4: "Asymptomatic",
    }
    cp_exact = {
        1: "http://snomed.info/id/429559004",
        2: "http://snomed.info/id/371807002",
        3: "http://snomed.info/id/274668005",
        4: "http://snomed.info/id/161971004",
    }
    df["cp"] = df["cp"].semmap.set_categorical(
        name="cp",
        label="Chest pain type",
        codes=cp_codes,
        exact_match=cp_exact,
        scheme_source_iri="https://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_3202.html",
        source_iri="https://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_3202.html",
    )._s

    # fbs — boolean-like categorical (indicator), source LOINC 2345-7
    df["fbs"] = df["fbs"].semmap.set_categorical(
        name="fbs",
        label="Fasting blood sugar >120 mg/dL (indicator)",
        codes={0: "≤120 mg/dL", 1: ">120 mg/dL"},
        scheme_source_iri="https://loinc.org/2345-7",
        source_iri="https://loinc.org/2345-7",
    )._s

    # trestbps — numeric with unit mmHg; LOINC 8480-6
    df["trestbps"] = df["trestbps"].semmap.set_numeric(
        name="trestbps",
        label="Resting systolic blood pressure",
        unit="mmHg",
        qudt_unit_iri="http://qudt.org/vocab/unit/MilliM_HG",
        xsd_datatype_iri="http://www.w3.org/2001/XMLSchema#integer",
        source_iri="https://loinc.org/8480-6",
        convert_to_pint=True,
    )._s

    # chol — numeric with unit mg/dL; LOINC 2093-3
    df["chol"] = df["chol"].semmap.set_numeric(
        name="chol",
        label="Total cholesterol",
        unit="mg/dL",
        qudt_unit_iri="http://qudt.org/vocab/unit/MilliGM-PER-DeciL",
        xsd_datatype_iri="http://www.w3.org/2001/XMLSchema#integer",
        source_iri="https://loinc.org/2093-3",
        convert_to_pint=True,
    )._s

    # Dataset-level metadata
    df.semmap.set_dataset(
        title="Heart dataset (example)",
        description="Example with variable/column JSON-LD and pint units.",
        license_iri="https://creativecommons.org/publicdomain/zero/1.0/"
    )

    # Write + read back (round-trip)
    out_path = "heart.semmap.parquet"
    df.semmap.to_parquet(out_path, index=False)

    df2 = SemMapFrameAccessor.read_parquet(out_path, convert_pint=True)

    # Quick checks
    assert json.dumps(df.attrs["semmap_jsonld"], sort_keys=True) == json.dumps(df2.attrs["semmap_jsonld"], sort_keys=True)
    for col in df.columns:
        j1 = df[col].semmap.jsonld()
        j2 = df2[col].semmap.jsonld()
        assert json.dumps(j1, sort_keys=True) == json.dumps(j2, sort_keys=True)
    print("Round-trip OK; JSON-LD and pint units restored where available.")
