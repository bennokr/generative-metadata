from __future__ import annotations
import json
from typing import Any, Dict, List, Mapping, Optional, Union
from enum import Enum
from dataclasses import dataclass

from .jsonld import JSONLDMixin

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pint_pandas import PintType

CONTEXT = {
    "@context": {
        "csvw": "http://www.w3.org/ns/csvw#",
        "dsv": "https://w3id.org/dsv-ontology#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "prov": "http://www.w3.org/ns/prov#",
        "qudt": "http://qudt.org/schema/qudt/",
        "unit": "http://qudt.org/vocab/unit/",
        "quantitykind": "http://qudt.org/vocab/quantitykind/",
        "sdmx-dimension": "http://purl.org/linked-data/sdmx/2009/dimension#",
        "schema": "https://schema.org/",
        "wd": "http://www.wikidata.org/entity/",
        "dct": "http://purl.org/dc/terms/",
        "dcat": "http://www.w3.org/ns/dcat#",
        "title": "dct:title",
        "description": "dct:description",
        "abstract": "dct:abstract",
        "purpose": "dct:purpose",
        "tableOfContents": "dct:tableOfContents",
        "landingPage": "dcat:landingPage",
        "accessRights": "dct:accessRights",
        "citation": "schema:citation",
        "identifier": "dct:identifier",
        "funding": "schema:funding",
        "populationType": "schema:populationType",
        "url": "csvw:url",
        "datasetSchema": "dsv:datasetSchema",
        "columns": {"@id": "dsv:column", "@container": "@set"},
        "name": "csvw:name",
        "titles": "csvw:titles",
        "description": "dct:description",
        "identifier": "schema:identifier",
        "about": "schema:about",
        "hadRole": "prov:hadRole",
        "defaultValue": "schema:defaultValue",
        "columnProperty": "dsv:columnProperty",
        "columnCompleteness": "dsv:columnCompleteness",
        "summaryStatistics": "dsv:summaryStatistics",
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
class StatisticalDataType(str, Enum):
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
    meanValue: Optional[float] = None
    medianValue: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class Unit(SkosMappings):
    ucumCode: Optional[str] = None  # e.g., "a"


@dataclass
class ColumnProperty(SkosMappings, JSONLDMixin):
    summaryStatistics: Optional[SummaryStatistics] = None
    unitText: Optional[str] = None  # e.g., "unit:YR" or "year"
    hasUnit: Optional[Unit] = None  # Unit node with possible QUDT IRI skos match
    source: Optional[str] = None  # web page with documentation
    hasCodeBook: Optional[CodeBook] = None
    hasVariable: Optional[Union[str, CodeConcept]] = None  # link to variable definition


# --- CSVW/DSV column and schema ----------------------------------------------
@dataclass
class Column(JSONLDMixin):
    name: str  # required
    titles: Optional[Union[str, List[str]]] = None
    description: Optional[str] = None
    identifier: Optional[str] = None
    about: Optional[str] = None
    hadRole: Optional[str] = None
    defaultValue: Optional[Any] = None
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
    title: Optional[str] = None
    description: Optional[str] = None
    abstract: Optional[str] = None
    purpose: Optional[str] = None
    landingPage: Optional[str] = None
    tableOfContents: Optional[str] = None
    citation: Optional[Any] = None
    provider: Optional[str] = None
    identifier: Optional[Any] = None
    funding: Optional[Any] = None
    populationType: Optional[Any] = None
    accessRights: Optional[Any] = None

    @classmethod
    def from_dcat_dsv(cls, payload: Mapping[str, Any]) -> "Metadata":
        """Build a :class:`Metadata` instance from a DCAT/DSV JSON-LD payload.

        Args:
            payload: Raw JSON-LD mapping created by :mod:`uci_template`.

        Returns:
            Parsed metadata with dataset- and column-level attributes populated.
        """

        def _get(*keys: str) -> Optional[Any]:
            for key in keys:
                if key in payload:
                    return payload.get(key)
            return None

        ds_summary = payload.get("dsv:summaryStatistics") or {}
        ds_stats = SummaryStatistics(
            datasetCompleteness=ds_summary.get("dsv:datasetCompleteness"),
            numberOfRows=ds_summary.get("dsv:numberOfRows"),
            numberOfColumns=ds_summary.get("dsv:numberOfColumns"),
            missingValueFormat=ds_summary.get("dsv:missingValueFormat"),
        ) if isinstance(ds_summary, Mapping) else None

        schema = payload.get("dsv:datasetSchema") or payload.get("datasetSchema") or {}
        columns_json = []
        if isinstance(schema, Mapping):
            columns_json = schema.get("dsv:column") or schema.get("columns") or []

        columns: List[Column] = []
        for col_json in columns_json:
            if not isinstance(col_json, Mapping):
                continue
            summary = col_json.get("dsv:summaryStatistics") or col_json.get("summaryStatistics")
            column_stats = SummaryStatistics.from_jsonld(summary) if isinstance(summary, Mapping) else None
            col_prop_json = col_json.get("dsv:columnProperty") or col_json.get("columnProperty")
            col_prop = ColumnProperty.from_jsonld(col_prop_json) if isinstance(col_prop_json, Mapping) else None
            columns.append(
                Column(
                    name=col_json.get("schema:name") or col_json.get("name"),
                    titles=col_json.get("csvw:titles") or col_json.get("titles") or col_json.get("dcterms:title"),
                    description=col_json.get("dcterms:description"),
                    about=col_json.get("schema:about"),
                    hadRole=col_json.get("prov:hadRole"),
                    identifier=col_json.get("schema:identifier"),
                    defaultValue=col_json.get("schema:defaultValue"),
                    columnProperty=col_prop,
                    summaryStatistics=column_stats,
                )
            )

        dataset_schema = DatasetSchema(columns=columns)

        return cls(
            datasetSchema=dataset_schema,
            summaryStatistics=ds_stats,
            title=_get("dcterms:title"),
            description=_get("dcterms:description"),
            abstract=_get("dcterms:abstract"),
            purpose=_get("dcterms:purpose"),
            landingPage=_get("dcat:landingPage"),
            tableOfContents=_get("dcterms:tableOfContents"),
            citation=_get("schema:citation"),
            provider=_get("dcterms:creator", "prov:wasAttributedTo"),
            identifier=_get("dcterms:identifier"),
            funding=_get("schema:funding"),
            populationType=_get("schema:populationType"),
            accessRights=_get("dcterms:accessRights"),
        )

    def to_privacy_frame(self, inferred: Mapping[str, str]) -> "pd.DataFrame":
        """Build a privacy metadata dataframe from SemMap content.

        The privacy metrics expect roles such as ``qi`` and ``sensitive`` and a
        coarse type mapping (``numeric``/``categorical``/``datetime``). This
        helper normalizes SemMap roles to those expectations and uses
        ``statisticalDataType`` or codebooks to infer the variable types,
        falling back to the provided ``inferred`` mapping when semantics are
        absent.

        Args:
            inferred: Mapping of column names to inferred types (``discrete`` or
                ``continuous``) used as fallback when semantics are missing.

        Returns:
            Dataframe with ``variable``, ``role`` and ``type`` columns.
        """

        import pandas as pd

        def _normalize_role(raw: Optional[str]) -> str:
            if not raw:
                return "qi"
            role = raw.strip().lower()
            if role in {"quasiidentifier", "quasi-identifier", "quasi_identifier"}:
                return "qi"
            if role in {"sensitive", "sensitive_attribute"}:
                return "sensitive"
            if role in {"identifier", "id", "primary_key"}:
                return "id"
            if role in {"ignore", "drop", "exclude"}:
                return "ignore"
            if role in {"target", "label", "outcome"}:
                return "target"
            if role in {"feature", "predictor"}:
                return "qi"
            return role

        rows = []
        for col in self.datasetSchema.columns:
            role = _normalize_role(col.hadRole)
            dtype = None
            stats_nodes = [col.summaryStatistics, getattr(col, "columnProperty", None)]
            for node in stats_nodes:
                stat_node = node.summaryStatistics if getattr(node, "summaryStatistics", None) else node
                if isinstance(stat_node, SummaryStatistics) and stat_node.statisticalDataType:
                    dtype = stat_node.statisticalDataType.value
                    break
                if isinstance(stat_node, ColumnProperty) and stat_node.hasCodeBook:
                    dtype = "dsv:NominalDataType"
                    break
            inferred_kind = inferred.get(col.name, "continuous")
            mapped_dtype = "numeric" if inferred_kind == "continuous" else "categorical"
            if dtype:
                lowered = dtype.lower()
                if "nominal" in lowered or "ordinal" in lowered:
                    mapped_dtype = "categorical"
                elif "interval" in lowered or "ratio" in lowered or "numerical" in lowered:
                    mapped_dtype = "numeric"
            rows.append({"variable": col.name, "role": role, "type": mapped_dtype})
        return pd.DataFrame(rows)

    def update_completeness_from_missingness(
        self,
        df: "pd.DataFrame",
        missingness_model: Optional[Any],
    ) -> None:
        """Refresh completeness and missing-value annotations based on fitted models.

        Args:
            df: Dataframe used to compute dataset completeness.
            missingness_model: Optional :class:`DataFrameMissingnessModel` instance
                containing per-column missingness probabilities.
        """

        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            return

        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        nnz = int(df.notna().to_numpy().sum())
        completeness = nnz / float(total_cells) if total_cells else 1.0

        if not self.summaryStatistics:
            self.summaryStatistics = SummaryStatistics()
        self.summaryStatistics.numberOfRows = n_rows
        self.summaryStatistics.numberOfColumns = n_cols
        self.summaryStatistics.datasetCompleteness = completeness

        model_map = getattr(missingness_model, "models_", {}) or {}
        by_name = {col.name: col for col in self.datasetSchema.columns}
        for col_name, col in by_name.items():
            model = model_map.get(col_name)
            if model is None:
                rate = float(df[col_name].isna().mean()) if col_name in df.columns else 0.0
            else:
                rate = float(getattr(model, "p_missing_", 0.0) or 0.0)
            completeness_val = 1.0 - rate
            if not col.summaryStatistics:
                col.summaryStatistics = SummaryStatistics()
            col.summaryStatistics.columnCompleteness = completeness_val

    def to_jsonld(self) -> Optional[Dict[str, Any]]:  # type: ignore[override]
        return super().to_jsonld()


# Arrow metadata keys (bytes per Arrow requirements)
_DATASET_SEMMAP_KEY = b"semmap.dataset"
_COLUMN_SEMMAP_KEY = b"semmap.column"


@pd.api.extensions.register_series_accessor("semmap")
class SemMapSeriesAccessor:
    """Series-level accessor to attach metadata semantics."""

    def __init__(self, s: pd.Series) -> None:
        self._s = s
        self.col_semmap = None

    # ---- helpers -------------------------------------------------------------

    def _try_convert_to_pint(self) -> None:
        """Best-effort conversion of the Series to a pint dtype using unit_text."""
        # Derive unit_text from UCUM if needed
        if not self.col_semmap: return
        col_prop = self.col_semmap.columnProperty
        ucum_code = col_prop.hasUnit.ucumCode if col_prop.hasUnit else None
        
        if col_prop.unitText is None and ucum_code is not None:
            try:
                from ucumvert import PintUcumRegistry

                ureg = PintUcumRegistry()
                col_prop.unitText = str(ureg.from_ucum(ucum_code).units)
            except Exception:
                pass

        if col_prop.unitText is None:
            return
        try:
            self._s[:] = self._s.astype(f"pint[{col_prop.unitText}]")
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
        source_iri: Optional[str] = None,
        convert_to_pint: bool = True,
    ) -> "SemMapSeriesAccessor":
        """Attach numeric variable metadata and (optionally) convert dtype to pint."""
        # Units node/string based on available inputs
        has_unit: Optional[Union[str, Unit]] = None
        # If both UCUM and QUDT provided, use a Unit node to capture both
        if ucum_code or qudt_unit_iri:
            if qudt_unit_iri:
                has_unit = Unit(ucumCode=ucum_code, exactMatch=[qudt_unit_iri])
            elif ucum_code:
                has_unit = Unit(ucumCode=ucum_code)

        col_prop = ColumnProperty(
            unitText=unit_text,
            hasUnit=has_unit,
            source=source_iri,
        )

        # Attach semantics to the Series (serialize dataclasses)
        self.col_semmap = Column(name=name, titles=label, columnProperty=col_prop)

        # Optionally convert to pint dtype
        if convert_to_pint:
            self._try_convert_to_pint()

        return self

    def set_categorical(
        self,
        name: str,
        label: str,
        *,
        codes: Dict[Union[int, str], str],
        scheme_source_iri: Optional[str] = None,
        source_iri: Optional[str] = None,
    ) -> "SemMapSeriesAccessor":
        """Attach categorical variable metadata (integer-coded or strings)."""
        # Build SKOS CodeBook with CodeConcepts
        top_concepts = [
            CodeConcept(notation=str(code), prefLabel=pref)
            for code, pref in codes.items()
        ]

        col_prop = ColumnProperty(
            hasCodeBook=CodeBook(hasTopConcept=top_concepts, source=scheme_source_iri),
            source=source_iri,
        )

        # Attach SemMap to the Series
        self.col_semmap = Column(name=name, titles=label, columnProperty=col_prop)

        # Ensure pandas categorical dtype if appropriate (best-effort)
        try:
            if not isinstance(self._s.dtype, pd.CategoricalDtype):
                self._s[:] = self._s.astype("category")
        except Exception:
            pass

        return self
    
    def _infer_statistical_data_type(self) -> StatisticalDataType:
        """Heuristic statistical data type inference for the Series."""
        if isinstance(self._s.dtype, pd.CategoricalDtype):
            # pandas categorical supports ordered attribute
            if getattr(getattr(self._s, "cat", None), "ordered", False):
                return StatisticalDataType.Ordinal
            else: 
                return StatisticalDataType.Nominal
        if isinstance(self._s.dtype, pd.BooleanDtype):
            return StatisticalDataType.Nominal
        return StatisticalDataType.Numerical

    # ---- Introspection -------------------------------------------------------
    def __call__(self) -> Column:
        n = len(self._s)
        self.col_semmap.summaryStatistics = SummaryStatistics(
            statisticalDataType=self._infer_statistical_data_type(),
            columnCompleteness=float(self._s.notna().mean()) if n else 1.0,
            numberOfRows=n,
        )
        return self.col_semmap

    def from_jsonld(self, metadata: Dict[str, Any], convert_pint:bool =True) -> "SemMapSeriesAccessor":
        self.col_semmap = Column.from_jsonld(metadata)
        if convert_pint:
            self._try_convert_to_pint()
        return self

    def to_jsonld(self) -> Optional[Dict[str, Any]]:
        return self().to_jsonld()

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
        self.dataset_semmap = None

    # ---- Helpers -------------------------------------------------------------
    
    def __call__(self) -> Metadata:
        # columns
        if not self.dataset_semmap:
            self.dataset_semmap = Metadata(datasetSchema=DatasetSchema(columns=[]))
        cols = [self._df[col].semmap() for col in self._df]
        self.dataset_semmap.datasetSchema.columns = cols

        # stats
        n_rows, n_cols = self._df.shape
        total_cells = n_rows * n_cols
        nnz = int(self._df.notna().to_numpy().sum())
        completeness = nnz / float(total_cells) if total_cells > 0 else 1.0
        self.dataset_semmap.summaryStatistics = SummaryStatistics(
            datasetCompleteness=completeness,
            numberOfRows=n_rows,
            numberOfColumns=n_cols,
        )

        return self.dataset_semmap

    def to_jsonld(self) -> Optional[Dict[str, Any]]:
        return self._df.semmap().to_jsonld()

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
            s_meta = self._df[field.name].semmap.to_jsonld()
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
        d_meta = self.to_jsonld()
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
            df.semmap.from_jsonld(json.loads(
                schema.metadata[_DATASET_SEMMAP_KEY].decode("utf-8")
            ), convert_pint=convert_pint)

        # Restore column semantics, and optionally pint dtypes
        for i, field in enumerate(schema):
            name = field.name
            if field.metadata and _COLUMN_SEMMAP_KEY in field.metadata:
                df[name].semmap.from_jsonld(json.loads(
                    field.metadata[_COLUMN_SEMMAP_KEY].decode("utf-8")
                ), convert_pint=convert_pint)

        return df

    # ---- External metadata loader -------------------------------------------

    def from_jsonld(
        self,
        metadata: Union[str, Dict[str, Any]],
        *,
        convert_pint: bool = True,
    ) -> "SemMapFrameAccessor":
        """Attach dataset metadata and column schema from a JSON object."""
        # Load dict if given a path
        if isinstance(metadata, str):
            with open(metadata, "r", encoding="utf-8") as f:
                meta_jsonld = json.load(f)
        else:
            meta_jsonld = metadata

        # Attach dataset semantics verbatim (round-trip equality)
        self.dataset_semmap = Metadata.from_jsonld(meta_jsonld)

        # Apply per-column metadata if present
        cols = (((meta_jsonld or {}).get("datasetSchema") or {}).get("columns")) or []
        by_name = {
            c.get("name"): c for c in cols if isinstance(c, dict) and "name" in c
        }

        for name, col_jsonld in by_name.items():
            if name in self._df.columns:
                self._df[name].semmap.from_jsonld(col_jsonld, convert_pint=convert_pint)

        return self
