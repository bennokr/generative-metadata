from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

from .jsonld import JSONLDMixin

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pint_pandas import PintType

# CONTEXT = json.load(pathlib.Path(__file__).with_name('semmap_context.jsonld'))
CONTEXT = {
    "@context": {
        "csvw": "http://www.w3.org/ns/csvw#",
        "dsv": "https://w3id.org/dsv-ontology#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "qudt": "http://qudt.org/schema/qudt/",
        "unit": "http://qudt.org/vocab/unit/",
        "quantitykind": "http://qudt.org/vocab/quantitykind/",
        "sdmx-dimension": "http://purl.org/linked-data/sdmx/2009/dimension#",
        "schema": "https://schema.org/",
        "wd": "http://www.wikidata.org/entity/",
        "dct": "http://purl.org/dc/terms/",
        "url": "csvw:url",
        "datasetSchema": "dsv:datasetSchema",
        "columns": {"@id": "dsv:column", "@container": "@set"},
        "name": "csvw:name",
        "titles": "csvw:titles",
        "columnProperty": "dsv:columnProperty",
        "columnCompleteness": "dsv:columnCompleteness",
        "SummaryStatistics": "dsv:summaryStatistics",
        "statisticalDataType": {"@id": "dsv:statisticalDataType", "@type": "@id"},
        "hasCodeBook": {"@id": "dsv:hasCodeBook", "@type": "@id"},
        "hasVariable": {"@id": "dsv:hasVariable", "@type": "@id"},
        "datasetCompleteness": "dsv:datasetCompleteness",
        "numberOfRows": "dsv:numberOfRows",
        "numberOfColumns": "dsv:numberOfColumns",
        "missingValueFormat": "dsv:missingValueFormat",
        "notation": "skos:notation",
        "prefLabel": "skos:prefLabel",
        "exactMatch": {"@id": "skos:exactMatch", "@type": "@id", "@container": "@set"},
        "closeMatch": {"@id": "skos:closeMatch", "@type": "@id", "@container": "@set"},
        "broadMatch": {"@id": "skos:broadMatch", "@type": "@id", "@container": "@set"},
        "narrowMatch": {
            "@id": "skos:narrowMatch",
            "@type": "@id",
            "@container": "@set",
        },
        "relatedMatch": {
            "@id": "skos:relatedMatch",
            "@type": "@id",
            "@container": "@set",
        },
        "hasTopConcept": {
            "@id": "skos:hasTopConcept",
            "@type": "@id",
            "@container": "@set",
        },
        "unitText": "schema:unitText",
        "ucumCode": "qudt:ucumCode",
        "hasUnit": "qudt:hasUnit",
        "source": {"@id": "dct:source", "@type": "@id"},
    }
}

# --- SKOS mapping mixin -------------------------------------------------------
@dataclass(kw_only=True)
class SkosMappings(JSONLDMixin):
    exactMatch: Optional[List[str]] = None
    closeMatch: Optional[List[str]] = None
    broadMatch: Optional[List[str]] = None
    narrowMatch: Optional[List[str]] = None
    relatedMatch: Optional[List[str]] = None


# --- Code book (SKOS) ---------------------------------------------------------
@dataclass
class CodeConcept(SkosMappings):
    notation: Optional[str] = None
    prefLabel: Optional[str] = None


@dataclass
class CodeBook(JSONLDMixin):
    hasTopConcept: Optional[List[CodeConcept]] = None
    source: Optional[str] = None  # if different from ColumnProperty source


# --- Column property (DSV + QUDT/UCUM) ---------------------------------------
class StatisticalDataType(Enum):
    Interval = "dsv:IntervalDataType"
    Nominal = "dsv:NominalDataType"
    Numerical = "dsv:NumericalDataType"
    Ordinal = "dsv:OrdinalDataType"
    Ratio = "dsv:RatioDataType"


@dataclass
class SummaryStatistics(JSONLDMixin):
    statisticalDataType: Optional[StatisticalDataType] = None
    columnCompleteness: Optional[float] = None
    datasetCompleteness: Optional[float] = None
    numberOfRows: Optional[int] = None
    numberOfColumns: Optional[int] = None
    missingValueFormat: Optional[str] = None


@dataclass
class Unit(SkosMappings):
    ucumCode: Optional[str] = None  # e.g., "a"


@dataclass
class ColumnProperty(JSONLDMixin):
    summaryStatistics: Optional[SummaryStatistics] = None
    unitText: Optional[str] = None  # e.g., "unit:YR" or "year"
    hasUnit: Optional[Union[str, Unit]] = None  # QUDT IRI or Unit node
    source: Optional[str] = None  # web page with documentation
    hasCodeBook: Optional[CodeBook] = None
    hasVariable: Optional[Union[str, CodeConcept]] = None  # link to variable definition


# --- CSVW/DSV column and schema ----------------------------------------------
@dataclass
class Column(JSONLDMixin):
    name: str  # required
    titles: Optional[Union[str, List[str]]] = None
    columnProperty: Optional[ColumnProperty] = None
    summaryStatistics: Optional[SummaryStatistics] = None


@dataclass
class DatasetSchema(JSONLDMixin):
    __context__ = CONTEXT
    columns: List[Column]  # required


# --- Root document / Dataset --------------------------------------------------
@dataclass
class Metadata(JSONLDMixin):
    __context__ = CONTEXT
    datasetSchema: DatasetSchema  # required
    summaryStatistics: Optional[SummaryStatistics] = None


# Arrow metadata keys (bytes per Arrow requirements)
_DATASET_SEMMAP_KEY = b"semmap.dataset"
_COLUMN_SEMMAP_KEY = b"semmap.column"


@pd.api.extensions.register_series_accessor("semmap")
class SemMapSeriesAccessor:
    """Series-level accessor to attach semantics (blank-node) metadata."""

    def __init__(self, s: pd.Series) -> None:
        self._s = s
        if "semmap" not in self._s.attrs:
            self._s.attrs["semmap"] = None

    # ---- helpers -------------------------------------------------------------

    def _try_convert_to_pint(self, unit_text: Optional[str]) -> None:
        """Best-effort conversion of the Series to a pint dtype using unit_text."""
        if unit_text is None:
            return
        try:
            self._s[:] = self._s.astype(f"pint[{unit_text}]")
        except Exception:
            # Swallow conversion errors—metadata is still attached
            pass

    # ---- Declarative APIs ----------------------------------------------------

    def set_numeric(
        self,
        name: str,
        label: str,
        *,
        unit_text: Optional[str] = None,  # unit string ("mmHg", "mg/dL", "year")
        ucum_code: Optional[str] = None,  # UCUM code ("mm[Hg]", "mg/dL", "a")
        qudt_unit_iri: Optional[str] = None,  # QUDT IRI
        statistical_data_type: Optional[str] = "dsv:QuantitativeDataType",
        source_iri: Optional[str] = None,
        convert_to_pint: bool = True,
    ) -> "SemMapSeriesAccessor":
        """Attach numeric variable metadata and (optionally) convert dtype to pint."""
        # Units node/string based on available inputs
        has_unit: Optional[Union[str, Unit]] = None
        # If both UCUM and QUDT provided, use a Unit node to capture both
        if ucum_code or qudt_unit_iri:
            if ucum_code and qudt_unit_iri:
                has_unit = Unit(ucumCode=ucum_code, exactMatch=[qudt_unit_iri])
            elif ucum_code:
                has_unit = Unit(ucumCode=ucum_code)
            else:
                # Only QUDT IRI -> allow compact representation as a string
                has_unit = qudt_unit_iri

        col_prop = ColumnProperty(
            unitText=unit_text,
            hasUnit=has_unit,
            source=source_iri,
        )

        # Derive unit_text from UCUM if needed
        if col_prop.unitText is None and ucum_code is not None:
            try:
                from ucumvert import PintUcumRegistry

                ureg = PintUcumRegistry()
                col_prop.unitText = str(ureg.from_ucum(ucum_code).units)
            except Exception:
                pass

        column = Column(name=name, titles=label, columnProperty=col_prop)

        # Attach semantics to the Series (serialize dataclasses)
        self._s.attrs["semmap"] = column

        # Optionally convert to pint dtype
        if convert_to_pint:
            self._try_convert_to_pint(col_prop.unitText)

        return self

    def set_categorical(
        self,
        name: str,
        label: str,
        *,
        codes: Dict[Union[int, str], str],
        scheme_source_iri: Optional[str] = None,
        source_iri: Optional[str] = None,
        statistical_data_type: Optional[str] = "dsv:NominalDataType",
        **kwargs,
    ) -> "SemMapSeriesAccessor":
        """Attach categorical variable metadata (integer-coded or strings)."""
        # Build SKOS CodeBook with CodeConcepts
        top_concepts = [
            CodeConcept(notation=str(code), prefLabel=pref) for code, pref in codes.items()
        ]
        code_book = CodeBook(hasTopConcept=top_concepts, source=scheme_source_iri)

        col_prop = ColumnProperty(
            hasCodeBook=code_book,
            source=source_iri,
        )

        column = Column(name=name, titles=label, columnProperty=col_prop)

        # Attach SemMap to the Series
        self._s.attrs["semmap"] = column

        # Ensure pandas categorical dtype if appropriate (best-effort)
        try:
            if not isinstance(self._s.dtype, pd.CategoricalDtype):
                self._s[:] = self._s.astype("category")
        except Exception:
            pass

        return self

    # ---- Introspection -------------------------------------------------------

    def jsonld(self) -> Optional[Dict[str, Any]]:
        return self._s.attrs.get("semmap").to_jsonld()

    # ---- internal hook (used by DataFrame writer) ----------------------------

    def _ensure_storage_for_parquet(self) -> pd.Series:
        """Ensure the physical storage is parquet-friendly (e.g., strip pint to magnitudes)."""
        s = self._s
        if isinstance(s.dtype, PintType):
            # Store magnitudes; metadata carries units for reconstruction
            s = pd.Series(s.to_numpy().magnitude, index=s.index, name=s.name)
        return s


@pd.api.extensions.register_dataframe_accessor("semmap")
class SemMapFrameAccessor:
    """DataFrame-level accessor for dataset metadata and Parquet round-trip."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        if "semmap" not in self._df.attrs:
            self._df.attrs["semmap"] = None

    # ---- Helpers -------------------------------------------------------------

    @staticmethod
    def _maybe_convert_series_to_pint(s: pd.Series, col_semmap: Dict[str, Any]) -> None:
        """Best-effort pint conversion based on column semantics."""
        try:
            col_prop = (col_semmap or {}).get("columnProperty") or {}
            unit_text = col_prop.get("unitText")
            if unit_text:
                s[:] = s.astype(f"pint[{unit_text}]")
        except Exception:
            # Leave as-is if conversion fails
            pass

    def jsonld(self) -> Optional[Dict[str, Any]]:
        meta = self._df.attrs.get("semmap")
        if meta is not None:
            return meta.to_jsonld()

        # Fall back: assemble a Metadata/Schema from per-column semantics if present
        cols: List[Column] = []
        for name in self._df.columns:
            cmeta = self._df[name].semmap
            if cmeta is not None:
                cols.append(cmeta)

        if cols:
            # Use dataclasses to normalize and add context at the dataset level
            md = Metadata(datasetSchema=DatasetSchema(columns=cols))
            return md.to_jsonld()

        return None

    # ---- IO: Parquet with Arrow schema/field metadata ------------------------

    def to_parquet(self, path: str, *, index: bool = False, **pq_kwargs) -> None:
        """Write Parquet with semantics stored in Arrow schema and fields."""
        # 1) normalize columns for parquet storage
        df_store = {}
        for col in self._df.columns:
            s_acc = self._df[col].semmap
            s_norm = s_acc._ensure_storage_for_parquet()
            df_store[col] = s_norm
        pdf = pd.DataFrame(df_store, index=self._df.index if index else None)

        # 2) convert to Arrow table
        table = pa.Table.from_pandas(pdf, preserve_index=index)

        # 3) attach column semantics on each Field
        fields = []
        for field in table.schema:
            s_meta = self._df[field.name].semmap.jsonld()
            fmeta = dict(field.metadata or {})
            if s_meta is not None:
                fmeta[_COLUMN_SEMMAP_KEY] = json.dumps(
                    s_meta, ensure_ascii=False
                ).encode("utf-8")
            fields.append(
                pa.field(
                    field.name, field.type, nullable=field.nullable, metadata=fmeta
                )
            )
        schema = pa.schema(fields)

        # 4) attach dataset semantics on Schema
        schema_meta = dict(schema.metadata or {})
        d_meta = self.jsonld()
        if d_meta is not None:
            schema_meta[_DATASET_SEMMAP_KEY] = json.dumps(
                d_meta, ensure_ascii=False
            ).encode("utf-8")
        schema = schema.with_metadata(schema_meta)

        # 5) write parquet
        pq.write_table(
            pa.Table.from_arrays(
                [table.column(i) for i in range(table.num_columns)], schema=schema
            ),
            path,
            **pq_kwargs,
        )

    @staticmethod
    def read_parquet(
        path: str, *, convert_pint: bool = True, **pq_kwargs
    ) -> pd.DataFrame:
        """Read Parquet and restore semantics + pint units."""
        table = pq.read_table(path, **pq_kwargs)
        schema = table.schema

        # Restore DataFrame
        df = table.to_pandas(types_mapper=None)  # leave as numeric/category

        # Restore dataset semantics
        if schema.metadata and _DATASET_SEMMAP_KEY in schema.metadata:
            df.attrs["semmap"] = Metadata.from_jsonld(json.loads(
                schema.metadata[_DATASET_SEMMAP_KEY].decode("utf-8")
            ))

        # Restore column semantics, and optionally pint dtypes
        for i, field in enumerate(schema):
            name = field.name
            if field.metadata and _COLUMN_SEMMAP_KEY in field.metadata:
                col_semmap = Column.from_jsonld(json.loads(
                    field.metadata[_COLUMN_SEMMAP_KEY].decode("utf-8")
                ))
                # attach back to Series
                df[name].attrs["semmap"] = col_semmap
                if convert_pint:
                    SemMapFrameAccessor._maybe_convert_series_to_pint(
                        df[name], col_semmap
                    )

        return df

    # ---- External metadata loader -------------------------------------------

    def apply_json_metadata(
        self,
        metadata: Union[str, Dict[str, Any]],
        *,
        convert_pint: bool = True,
    ) -> "SemMapFrameAccessor":
        """Attach dataset metadata and column schema from a JSON object."""
        # Load dict if given a path
        if isinstance(metadata, str):
            with open(metadata, "r", encoding="utf-8") as f:
                meta_obj = Metadata.from_jsonld(json.load(f))
        else:
            meta_obj = Metadata.from_jsonld(metadata)

        # Attach dataset semantics verbatim (round-trip equality)
        self._df.attrs["semmap"] = meta_obj

        # Apply per-column metadata if present
        cols = (((meta_obj or {}).get("datasetSchema") or {}).get("columns")) or []
        by_name = {
            c.get("name"): c for c in cols if isinstance(c, dict) and "name" in c
        }

        for name, cmeta in by_name.items():
            if name in self._df.columns:
                self._df[name].attrs["semmap"] = cmeta
                if convert_pint:
                    self._maybe_convert_series_to_pint(self._df[name], cmeta)

        return self
