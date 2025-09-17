# semmap.py
from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from semmap_schema import CodeBook, CodeConcept, Column, ColumnProperty, TableSchema

# Optional pint support
try:  # pragma: no cover
    from pint_pandas import PintArray, PintType
    from ucumvert import PintUcumRegistry

    _ureg = PintUcumRegistry()
    _HAVE_PINT = True
except Exception:  # pragma: no cover
    _HAVE_PINT = False
    PintArray = PintType = None
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

QUDT_TO_UCUM = {
    UNIT["MilliM_HG"]: "mm[Hg]",
    UNIT["MilliGM-PER-DeciL"]: "mg/dL",
    UNIT["YR"]: "a",
}
UCUM_TO_QUDT = {v: k for k, v in QUDT_TO_UCUM.items()}
PINT_TO_UCUM = {
    "mmHg": "mm[Hg]",
    "mg/dL": "mg/dL",
    "year": "a",
}

DSV_NOMINAL = "dsv:NominalDataType"
DSV_QUANTITATIVE = "dsv:QuantitativeDataType"

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

def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _normalize_titles(value: Any) -> Optional[Union[str, List[str]]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return str(value)


def _listify(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _strip_empty(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in obj.items():
            cleaned_value = _strip_empty(value)
            if cleaned_value is None:
                continue
            if isinstance(cleaned_value, dict) and not cleaned_value:
                continue
            if isinstance(cleaned_value, list) and not cleaned_value:
                continue
            cleaned[key] = cleaned_value
        return cleaned
    if isinstance(obj, list):
        cleaned_list = []
        for item in obj:
            cleaned_item = _strip_empty(item)
            if cleaned_item is None:
                continue
            if isinstance(cleaned_item, dict) and not cleaned_item:
                continue
            if isinstance(cleaned_item, list) and not cleaned_item:
                continue
            cleaned_list.append(cleaned_item)
        return cleaned_list
    return obj


def _dataclass_has_data(obj: Any) -> bool:
    return is_dataclass(obj) and bool(_strip_empty(asdict(obj)))


def _dataclass_to_clean_dict(obj) -> Dict[str, Any]:
    return _strip_empty(asdict(obj))


def _parse_code_concept_dict(data: Dict[str, Any]) -> CodeConcept:
    if not isinstance(data, dict):
        raise TypeError("CodeConcept metadata must be a dict")

    kwargs: Dict[str, Any] = {}
    for field in ("notation", "prefLabel"):
        value = data.get(field)
        if value is not None:
            kwargs[field] = str(value)

    for field in ("exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch"):
        value = data.get(field)
        if value is None:
            continue
        kwargs[field] = _listify(value)

    return CodeConcept(**kwargs)


def _parse_codebook_dict(data: Dict[str, Any]) -> Optional[CodeBook]:
    if not isinstance(data, dict):
        raise TypeError("CodeBook metadata must be a dict")

    kwargs: Dict[str, Any] = {}
    top_concepts = []
    for concept in data.get("hasTopConcept", []) or []:
        try:
            parsed = _parse_code_concept_dict(concept)
        except TypeError:
            continue
        if _dataclass_has_data(parsed):
            top_concepts.append(parsed)
    if top_concepts:
        kwargs["hasTopConcept"] = top_concepts

    source = data.get("source")
    if source is not None:
        kwargs["source"] = str(source)

    codebook = CodeBook(**kwargs) if kwargs else CodeBook()
    return codebook if _dataclass_has_data(codebook) else None


def _parse_column_property_dict(data: Dict[str, Any]) -> Optional[ColumnProperty]:
    if not isinstance(data, dict):
        raise TypeError("ColumnProperty metadata must be a dict")

    kwargs: Dict[str, Any] = {}
    for field in (
        "statisticalDataType",
        "valueType",
        "hasQuantityKind",
        "hasUnit",
        "ucumCode",
        "source",
    ):
        value = data.get(field)
        if value is not None:
            kwargs[field] = str(value)

    for field in ("exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch"):
        value = data.get(field)
        if value is not None:
            kwargs[field] = _listify(value)

    if data.get("hasCodeBook") is not None:
        codebook = _parse_codebook_dict(data["hasCodeBook"])
        if codebook is not None:
            kwargs["hasCodeBook"] = codebook

    if not kwargs:
        return None

    column_property = ColumnProperty(**kwargs)
    return column_property if _dataclass_has_data(column_property) else None


def _parse_column_dict(data: Dict[str, Any]) -> Column:
    if not isinstance(data, dict):
        raise TypeError("Column metadata must be a dict")

    name = data.get("name")
    if not isinstance(name, str):
        raise ValueError("Column metadata requires a string name")

    titles = _normalize_titles(data.get("titles"))
    column_property = None
    if data.get("columnProperty") is not None:
        column_property = _parse_column_property_dict(data["columnProperty"])

    kwargs: Dict[str, Any] = {"name": name}
    if titles is not None:
        kwargs["titles"] = titles
    if column_property is not None:
        kwargs["columnProperty"] = column_property

    return Column(**kwargs)


def _parse_table_schema_dict(data: Dict[str, Any]) -> Optional[TableSchema]:
    if not isinstance(data, dict):
        return None

    columns_data = data.get("columns")
    if not isinstance(columns_data, list):
        return None

    columns: List[Column] = []
    for entry in columns_data:
        try:
            column = _parse_column_dict(entry)
        except (TypeError, ValueError):
            continue
        columns.append(column)

    if not columns:
        return None

    return TableSchema(columns=columns)


def _column_to_dict(column: Column) -> Dict[str, Any]:
    return _dataclass_to_clean_dict(column)


def _column_from_metadata_dict(data: Dict[str, Any]) -> Optional[Column]:
    if not isinstance(data, dict):
        return None

    if "name" in data and ("columnProperty" in data or "titles" in data):
        try:
            return _parse_column_dict(data)
        except (TypeError, ValueError):
            return None

    notation = _optional_str(
        _ld_get(data, [str(SKOS.notation), "skos:notation", "notation", "name"])
    )
    if not notation:
        return None

    titles = _normalize_titles(
        _ld_get(data, [str(SKOS.prefLabel), "skos:prefLabel", "prefLabel", "titles"])
    )
    column_source = _optional_str(
        _ld_get(data, [str(DCT.source), "dct:source", "source"])
    )

    rep = _ld_get(
        data,
        [str(DISCO.representation), "disco:representation", "representation"],
        default={},
    )

    column_property = None
    if isinstance(rep, dict):
        cp_kwargs: Dict[str, Any] = {}

        value_type = _optional_str(
            _ld_get(rep, [str(DSV.valueType), "dsv:valueType", "valueType"])
        )
        if value_type:
            cp_kwargs["valueType"] = value_type

        stat_type = _optional_str(
            _ld_get(rep, [str(DSV.statisticalDataType), "dsv:statisticalDataType", "statisticalDataType"])
        )
        if stat_type:
            cp_kwargs["statisticalDataType"] = stat_type

        quantity_kind = _optional_str(
            _ld_get(rep, [str(QUDT.hasQuantityKind), "qudt:hasQuantityKind", "hasQuantityKind"])
        )
        if quantity_kind:
            cp_kwargs["hasQuantityKind"] = quantity_kind

        has_unit = _optional_str(
            _ld_get(rep, [str(QUDT.hasUnit), "qudt:hasUnit", "hasUnit"])
        )
        if has_unit:
            cp_kwargs["hasUnit"] = has_unit

        ucum_code = _optional_str(
            _ld_get(rep, [str(QUDT.ucumCode), "qudt:ucumCode", "ucumCode"])
        )
        if not ucum_code and has_unit:
            ucum_code = QUDT_TO_UCUM.get(has_unit)

        if not ucum_code:
            pint_unit = _optional_str(
                _ld_get(rep, [str(SEMMAP.pintUnit), "semmap:pintUnit", "pintUnit"])
            )
            if pint_unit:
                ucum_code = PINT_TO_UCUM.get(pint_unit)

        if ucum_code:
            cp_kwargs["ucumCode"] = ucum_code

        if column_source:
            cp_kwargs["source"] = column_source

        has_top = _ld_get(rep, [str(SKOS.hasTopConcept), "skos:hasTopConcept", "hasTopConcept"])
        top_concepts = []
        if isinstance(has_top, list):
            for concept in has_top:
                if not isinstance(concept, dict):
                    continue
                concept_kwargs: Dict[str, Any] = {}
                notation_value = _optional_str(
                    _ld_get(concept, [str(SKOS.notation), "skos:notation", "notation"])
                )
                if notation_value:
                    concept_kwargs["notation"] = notation_value

                pref_label_value = _optional_str(
                    _ld_get(concept, [str(SKOS.prefLabel), "skos:prefLabel", "prefLabel", "titles"])
                )
                if pref_label_value:
                    concept_kwargs["prefLabel"] = pref_label_value

                for field in ("exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch"):
                    field_value = _ld_get(
                        concept,
                        [str(getattr(SKOS, field)), f"skos:{field}", field],
                    )
                    if field_value is not None:
                        concept_kwargs[field] = _listify(field_value)

                if concept_kwargs:
                    top_concepts.append(CodeConcept(**concept_kwargs))

        scheme_source = _optional_str(
            _ld_get(rep, [str(DCT.source), "dct:source", "source"])
        )

        if top_concepts or scheme_source:
            codebook_kwargs: Dict[str, Any] = {}
            if top_concepts:
                codebook_kwargs["hasTopConcept"] = top_concepts
            if scheme_source:
                codebook_kwargs["source"] = scheme_source
            cp_kwargs["hasCodeBook"] = CodeBook(**codebook_kwargs)

        if cp_kwargs:
            column_property = ColumnProperty(**cp_kwargs)

    if column_property is None and column_source:
        column_property = ColumnProperty(source=column_source)

    kwargs: Dict[str, Any] = {"name": notation}
    if titles is not None:
        kwargs["titles"] = titles
    if column_property is not None:
        kwargs["columnProperty"] = column_property

    return Column(**kwargs)


def _column_property_to_pint_unit(column_property: Optional[ColumnProperty]):
    if not (_HAVE_PINT and column_property):
        return None

    ucum_code = getattr(column_property, "ucumCode", None)
    if ucum_code:
        try:
            return _ureg.from_ucum(ucum_code).u
        except Exception:  # pragma: no cover
            pass

    has_unit = getattr(column_property, "hasUnit", None)
    if has_unit:
        pint_unit = QUDT_TO_PINT.get(has_unit)
        if pint_unit:
            try:
                return _ureg.parse_units(pint_unit)
            except Exception:  # pragma: no cover
                return None
    return None


def _convert_series_to_pint(
    series: pd.Series, column_property: Optional[ColumnProperty]
) -> Optional[pd.Series]:
    if not (_HAVE_PINT and PintArray is not None):
        return None

    unit = _column_property_to_pint_unit(column_property)
    if unit is None:
        return None

    try:
        magnitudes = series.astype("float64").to_numpy() * unit
    except Exception:  # pragma: no cover
        return None

    return pd.Series(PintArray(magnitudes), index=series.index, name=series.name)


def _jsonld_dataset(
    title: str,
    description: Optional[str] = None,
    license_iri: Optional[str] = None
) -> Dict[str, Any]:
    obj = {
        "@context": JSONLD_CONTEXT_URL,
        "@type": [ str(DISCO.LogicalDataSet), str(DSV.Dataset), str(DCAT.Dataset) ],
        str(DCT["title"]): title,
    }
    if description:
        obj[str(DCT["description"])] = description
    if license_iri:
        obj[str(DCT["license"])] = license_iri
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
        ucum_code: Optional[str] = None,         # UCUM code ("mm[Hg]", "mg/dL", "a")
        qudt_unit_iri: Optional[str] = None,     # QUDT IRI; auto-filled when possible
        value_type_iri: str = str(XSD.decimal),
        statistical_data_type: Optional[str] = DSV_QUANTITATIVE,
        quantity_kind_iri: Optional[str] = None,
        source_iri: Optional[str] = None,
        convert_to_pint: bool = True,
    ) -> "SemMapSeriesAccessor":
        """Attach numeric variable metadata and (optionally) convert dtype to pint."""

        column_name = str(name)
        titles = str(label)

        inferred_ucum = ucum_code or (PINT_TO_UCUM.get(unit) if unit else None)
        inferred_qudt = qudt_unit_iri or (
            UCUM_TO_QUDT.get(inferred_ucum) if inferred_ucum else PINT_TO_QUDT.get(unit)
        )

        cp_kwargs: Dict[str, Any] = {}
        if statistical_data_type:
            cp_kwargs["statisticalDataType"] = str(statistical_data_type)
        if value_type_iri:
            cp_kwargs["valueType"] = str(value_type_iri)
        if quantity_kind_iri:
            cp_kwargs["hasQuantityKind"] = str(quantity_kind_iri)
        if inferred_qudt:
            cp_kwargs["hasUnit"] = str(inferred_qudt)
        if inferred_ucum:
            cp_kwargs["ucumCode"] = str(inferred_ucum)
        if source_iri:
            cp_kwargs["source"] = str(source_iri)

        column_property = ColumnProperty(**cp_kwargs) if cp_kwargs else None
        column = Column(name=column_name, titles=titles, columnProperty=column_property)
        column_metadata = _column_to_dict(column)

        if convert_to_pint:
            converted = _convert_series_to_pint(self._s, column_property)
            if converted is not None:
                self._s = converted
            elif unit and _HAVE_PINT and PintArray is not None:
                try:  # legacy fallback when UCUM mapping is unavailable
                    magnitudes = self._s.astype("float64").to_numpy() * _ureg.parse_units(unit)
                    self._s = pd.Series(PintArray(magnitudes), index=self._s.index, name=self._s.name)
                except Exception:  # pragma: no cover
                    pass
        self._s.attrs["semmap_jsonld"] = column_metadata
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
        statistical_data_type: Optional[str] = DSV_NOMINAL,
    ) -> "SemMapSeriesAccessor":
        """Attach categorical variable metadata (integer-coded or strings)."""

        column_name = str(name)
        titles = str(label)

        top_concepts: List[CodeConcept] = []
        for code, display in codes.items():
            concept_kwargs: Dict[str, Any] = {"notation": str(code)}
            if display is not None:
                concept_kwargs["prefLabel"] = str(display)
            if exact_match and code in exact_match:
                concept_kwargs["exactMatch"] = _listify(exact_match[code])
            clean_kwargs = {k: v for k, v in concept_kwargs.items() if v is not None}
            if clean_kwargs:
                top_concepts.append(CodeConcept(**clean_kwargs))

        codebook = None
        if top_concepts or scheme_source_iri:
            cb_kwargs: Dict[str, Any] = {}
            if top_concepts:
                cb_kwargs["hasTopConcept"] = top_concepts
            if scheme_source_iri:
                cb_kwargs["source"] = str(scheme_source_iri)
            candidate = CodeBook(**cb_kwargs)
            codebook = candidate if _dataclass_has_data(candidate) else None

        cp_kwargs: Dict[str, Any] = {}
        if statistical_data_type:
            cp_kwargs["statisticalDataType"] = str(statistical_data_type)
        if source_iri:
            cp_kwargs["source"] = str(source_iri)
        if codebook is not None:
            cp_kwargs["hasCodeBook"] = codebook

        column_property = ColumnProperty(**cp_kwargs) if cp_kwargs else None
        column = Column(name=column_name, titles=titles, columnProperty=column_property)
        column_metadata = _column_to_dict(column)

        # Keep storage as integer or category if provided
        try:
            # If integer-coded dict, cast to category with ordered codes preserved
            if pd.api.types.is_integer_dtype(self._s.dtype):
                cat = pd.Categorical(self._s.map(lambda v: codes.get(v, v)))
                # store original code as separate attribute for lossless round-trip
                self._s = pd.Series(cat, index=self._s.index, name=self._s.name)
        except Exception:
            pass

        self._s.attrs["semmap_jsonld"] = column_metadata
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

        if _HAVE_PINT and PintType is not None and isinstance(s.dtype, PintType):
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
                column = _column_from_metadata_dict(col_jsonld) if isinstance(col_jsonld, dict) else None
                if column is not None:
                    df[name].attrs["semmap_jsonld"] = _column_to_dict(column)
                else:
                    df[name].attrs["semmap_jsonld"] = col_jsonld

                if convert_pint:
                    column_property = column.columnProperty if column is not None else None
                    converted = _convert_series_to_pint(df[name], column_property)
                    if converted is not None:
                        df[name] = converted
                    elif column_property is None and isinstance(col_jsonld, dict):
                        rep = _ld_get(col_jsonld, [
                            str(DISCO.representation),
                            "disco:representation",
                            "representation",
                        ], default={})
                        pint_unit = _ld_get(rep, [str(SEMMAP.pintUnit), "semmap:pintUnit", "pintUnit"]) if isinstance(rep, dict) else None
                        qudt_unit = _ld_get(rep, [str(QUDT.hasUnit), "qudt:hasUnit", "hasUnit"]) if isinstance(rep, dict) else None
                        unit = pint_unit or QUDT_TO_PINT.get(qudt_unit)
                        if unit and _HAVE_PINT and PintArray is not None:
                            try:
                                magnitudes = df[name].astype("float64").to_numpy() * _ureg.parse_units(unit)
                                df[name] = pd.Series(PintArray(magnitudes), index=df.index, name=name)
                            except Exception:  # pragma: no cover
                                pass

        return df

    # ---- External metadata loader
    def apply_json_metadata(
        self,
        metadata: Union[str, Dict[str, Any]],
        *,
        convert_pint: bool = True,
    ) -> "SemMapFrameAccessor":
        """Attach dataset metadata and column schema from a JSON object."""

        if isinstance(metadata, str):
            with open(metadata, "r", encoding="utf-8") as f:
                meta_obj: Dict[str, Any] = json.load(f)
        else:
            meta_obj = metadata

        dataset_meta = meta_obj.get("dataset")
        if dataset_meta is not None:
            self._df.attrs["semmap_jsonld"] = dataset_meta

        processed: set[str] = set()
        table_schema = _parse_table_schema_dict(meta_obj.get("tableSchema")) if isinstance(meta_obj.get("tableSchema"), dict) else None
        if table_schema is not None:
            for column in table_schema.columns:
                if column.name not in self._df.columns:
                    continue
                self._df[column.name].attrs["semmap_jsonld"] = _column_to_dict(column)
                processed.add(column.name)
                if convert_pint:
                    converted = _convert_series_to_pint(self._df[column.name], column.columnProperty)
                    if converted is not None:
                        self._df[column.name] = converted

        legacy_vars = _ld_get(meta_obj, [str(DISCO.variable), "disco:variable", "variable"])
        if legacy_vars is None and isinstance(meta_obj.get("dataset"), dict):
            legacy_vars = _ld_get(meta_obj["dataset"], [str(DISCO.variable), "disco:variable", "variable"])

        variables: List[tuple[str, Dict[str, Any]]] = []
        if isinstance(legacy_vars, list):
            for entry in legacy_vars:
                if isinstance(entry, dict):
                    name = _ld_get(entry, [str(SKOS.notation), "skos:notation", "notation", "name"])
                    if isinstance(name, str):
                        variables.append((name, entry))
        elif isinstance(legacy_vars, dict):
            for key, value in legacy_vars.items():
                if isinstance(key, str) and isinstance(value, dict):
                    variables.append((key, value))

        for name, raw_meta in variables:
            if name in processed or name not in self._df.columns:
                continue
            column = _column_from_metadata_dict(raw_meta)
            if column is not None:
                self._df[name].attrs["semmap_jsonld"] = _column_to_dict(column)
                processed.add(name)
                if convert_pint:
                    converted = _convert_series_to_pint(self._df[name], column.columnProperty)
                    if converted is not None:
                        self._df[name] = converted
                continue

            # Fallback: store raw metadata and attempt legacy pint conversion
            self._df[name].attrs["semmap_jsonld"] = raw_meta
            if convert_pint and _HAVE_PINT and isinstance(raw_meta, dict):
                rep = _ld_get(raw_meta, [str(DISCO.representation), "disco:representation", "representation"], default={})
                pint_unit = _ld_get(rep, [str(SEMMAP.pintUnit), "semmap:pintUnit", "pintUnit"]) if isinstance(rep, dict) else None
                qudt_unit = _ld_get(rep, [str(QUDT.hasUnit), "qudt:hasUnit", "hasUnit"]) if isinstance(rep, dict) else None
                unit = pint_unit or QUDT_TO_PINT.get(qudt_unit)
                if unit:
                    try:
                        magnitudes = self._df[name].astype("float64").to_numpy() * _ureg.parse_units(unit)
                        self._df[name] = pd.Series(PintArray(magnitudes), index=self._df.index, name=name)
                    except Exception:  # pragma: no cover
                        pass

        return self

    def export_json_metadata(self) -> Dict[str, Any]:
        """Export dataset metadata with a CSVW-style table schema."""

        out: Dict[str, Any] = {}
        dataset_meta = self.jsonld()
        if dataset_meta is not None:
            out["dataset"] = dataset_meta

        columns: List[Column] = []
        for name in self._df.columns:
            meta = self._df[name].semmap.jsonld()
            if not isinstance(meta, dict):
                continue
            column = _column_from_metadata_dict(meta)
            if column is None:
                continue
            columns.append(column)

        out["tableSchema"] = _dataclass_to_clean_dict(TableSchema(columns=columns)) if columns else {"columns": []}
        out.setdefault("@context", JSONLD_CONTEXT_URL)
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
        ucum_code="mm[Hg]",
        qudt_unit_iri="http://qudt.org/vocab/unit/MilliM_HG",
        value_type_iri="xsd:integer",
        source_iri="https://loinc.org/8480-6",
        convert_to_pint=True,
    )._s

    # chol — numeric with unit mg/dL; LOINC 2093-3
    df["chol"] = df["chol"].semmap.set_numeric(
        name="chol",
        label="Total cholesterol",
        unit="mg/dL",
        ucum_code="mg/dL",
        qudt_unit_iri="http://qudt.org/vocab/unit/MilliGM-PER-DeciL",
        value_type_iri="xsd:integer",
        source_iri="https://loinc.org/2093-3",
        convert_to_pint=True,
    )._s

    # Dataset-level metadata
    df.semmap.set_dataset(
        title="Heart dataset (example)",
        description="Example with variable/column JSON-LD and UCUM units.",
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
