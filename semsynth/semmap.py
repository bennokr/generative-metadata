# semmap.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from pint_pandas import PintType
except ImportError:  # pragma: no cover - optional dependency

    class PintType:  # type: ignore[too-many-ancestors]
        """Placeholder type when pint-pandas is unavailable."""

        pass

    _HAVE_PINT = False
else:
    _HAVE_PINT = True

# Arrow metadata keys (bytes per Arrow requirements)
_DATASET_JSONLD_KEY = b"jsonld.dataset"
_COLUMN_JSONLD_KEY = b"jsonld.column"

DSV_NOMINAL = "dsv:NominalDataType"
DSV_QUANTITATIVE = "dsv:QuantitativeDataType"


@pd.api.extensions.register_series_accessor("semmap")
class SemMapSeriesAccessor:
    """Series-level accessor to attach JSON-LD (blank-node) metadata."""

    def __init__(self, s: pd.Series) -> None:
        self._s = s
        if "semmap_jsonld" not in self._s.attrs:
            self._s.attrs["semmap_jsonld"] = None

    # ---- helpers -------------------------------------------------------------

    def _try_convert_to_pint(self, unit_text: Optional[str]) -> None:
        """Best-effort conversion of the Series to a pint dtype using unit_text."""
        if unit_text is None or not _HAVE_PINT:
            return
        try:
            # pint-pandas supports "pint[<unit>]" dtype strings when pint is available
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
        qudt_unit_iri: Optional[str] = None,  # QUDT IRI; auto-filled when possible
        value_type_iri: str = "http://www.w3.org/2001/XMLSchema#decimal",
        statistical_data_type: Optional[str] = "dsv:QuantitativeDataType",
        quantity_kind_iri: Optional[str] = None,
        source_iri: Optional[str] = None,
        convert_to_pint: bool = True,
    ) -> "SemMapSeriesAccessor":
        """Attach numeric variable metadata and (optionally) convert dtype to pint."""
        col_prop: Dict[str, Any] = {}
        if statistical_data_type is not None:
            col_prop["statisticalDataType"] = statistical_data_type
        if value_type_iri is not None:
            col_prop["valueType"] = value_type_iri
        if quantity_kind_iri is not None:
            col_prop["hasQuantityKind"] = quantity_kind_iri
        if unit_text is not None:
            col_prop["unitText"] = unit_text
        if ucum_code is not None:
            col_prop["ucumCode"] = ucum_code
        if source_iri is not None:
            col_prop["source"] = source_iri
        if qudt_unit_iri is not None:
            # Include as an exactMatch to the QUDT unit IRI if provided
            col_prop.setdefault("exactMatch", [])
            if isinstance(col_prop["exactMatch"], list):
                col_prop["exactMatch"].append(qudt_unit_iri)

        column_jsonld: Dict[str, Any] = {
            "name": name,
            "titles": label,
        }
        if col_prop:
            column_jsonld["columnProperty"] = col_prop

        # Attach JSON-LD to the Series
        self._s.attrs["semmap_jsonld"] = column_jsonld

        # Optionally convert physical storage to pint dtype (kept during runtime;
        # stripped to magnitudes for Parquet in _ensure_storage_for_parquet()).
        if convert_to_pint:
            self._try_convert_to_pint(unit_text)

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
        # Build SKOS CodeBook
        top_concepts = []
        for code, pref in codes.items():
            top_concepts.append({"notation": str(code), "prefLabel": pref})
        code_book: Dict[str, Any] = {"hasTopConcept": top_concepts}
        if scheme_source_iri is not None:
            code_book["source"] = scheme_source_iri

        col_prop: Dict[str, Any] = {
            "hasCodeBook": code_book,
        }
        if statistical_data_type is not None:
            col_prop["statisticalDataType"] = statistical_data_type
        if source_iri is not None:
            col_prop["source"] = source_iri

        column_jsonld: Dict[str, Any] = {
            "name": name,
            "titles": label,
            "columnProperty": col_prop,
        }

        # Attach JSON-LD to the Series
        self._s.attrs["semmap_jsonld"] = column_jsonld

        # Ensure pandas categorical dtype if appropriate (best-effort)
        try:
            if not pd.api.types.is_categorical_dtype(self._s.dtype):
                self._s[:] = self._s.astype("category")
        except Exception:
            pass

        return self

    # ---- Introspection -------------------------------------------------------

    def jsonld(self) -> Optional[Dict[str, Any]]:
        return self._s.attrs.get("semmap_jsonld")

    # ---- internal hook (used by DataFrame writer) ----------------------------

    def _ensure_storage_for_parquet(self) -> pd.Series:
        """Ensure the physical storage is parquet-friendly (e.g., strip pint to magnitudes)."""
        s = self._s

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

    # ---- Helpers -------------------------------------------------------------

    @staticmethod
    def _maybe_convert_series_to_pint(s: pd.Series, col_jsonld: Dict[str, Any]) -> None:
        """Best-effort pint conversion based on column JSON-LD."""
        if not _HAVE_PINT:
            return
        try:
            col_prop = (col_jsonld or {}).get("columnProperty") or {}
            unit_text = col_prop.get("unitText")
            if unit_text:
                s[:] = s.astype(f"pint[{unit_text}]")
        except Exception:
            # Leave as-is if conversion fails
            pass

    def jsonld(self) -> Optional[Dict[str, Any]]:
        meta = self._df.attrs.get("semmap_jsonld")
        if meta is not None:
            return meta

        # Fall back to assembling from per-column JSON-LD if present
        cols: List[Dict[str, Any]] = []
        for name in self._df.columns:
            cmeta = self._df[name].semmap.jsonld()
            if cmeta is not None:
                cols.append(cmeta)

        if cols:
            return {"tableSchema": {"columns": cols}}

        return None

    # ---- IO: Parquet with Arrow schema/field metadata ------------------------

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
                fmeta[_COLUMN_JSONLD_KEY] = json.dumps(
                    s_meta, ensure_ascii=False
                ).encode("utf-8")
            fields.append(
                pa.field(
                    field.name, field.type, nullable=field.nullable, metadata=fmeta
                )
            )
        schema = pa.schema(fields)

        # 4) attach dataset JSON-LD on Schema
        schema_meta = dict(schema.metadata or {})
        d_meta = self.jsonld()
        if d_meta is not None:
            schema_meta[_DATASET_JSONLD_KEY] = json.dumps(
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
        """Read Parquet and restore JSON-LD + pint units."""
        table = pq.read_table(path, **pq_kwargs)
        schema = table.schema

        # Restore DataFrame
        df = table.to_pandas(types_mapper=None)  # leave as numeric/category

        # Restore dataset JSON-LD
        if schema.metadata and _DATASET_JSONLD_KEY in schema.metadata:
            df.attrs["semmap_jsonld"] = json.loads(
                schema.metadata[_DATASET_JSONLD_KEY].decode("utf-8")
            )

        # Restore column JSON-LD, and optionally pint dtypes
        for i, field in enumerate(schema):
            name = field.name
            if field.metadata and _COLUMN_JSONLD_KEY in field.metadata:
                col_jsonld = json.loads(
                    field.metadata[_COLUMN_JSONLD_KEY].decode("utf-8")
                )
                # attach back to Series
                df[name].attrs["semmap_jsonld"] = col_jsonld
                if convert_pint:
                    SemMapFrameAccessor._maybe_convert_series_to_pint(
                        df[name], col_jsonld
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
                meta_obj = json.load(f)
        else:
            meta_obj = metadata

        # Attach dataset JSON-LD verbatim (tests expect round-trip equality)
        self._df.attrs["semmap_jsonld"] = meta_obj

        # Apply per-column metadata if present
        cols = (((meta_obj or {}).get("tableSchema") or {}).get("columns")) or []
        by_name = {
            c.get("name"): c for c in cols if isinstance(c, dict) and "name" in c
        }

        for name, cmeta in by_name.items():
            if name in self._df.columns:
                self._df[name].attrs["semmap_jsonld"] = cmeta
                if convert_pint:
                    self._maybe_convert_series_to_pint(self._df[name], cmeta)

        return self
