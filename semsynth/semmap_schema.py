from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union, Literal
from enum import Enum

from .jsonld import JSONLDMixin

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
        "summaryStatistics": "dsv:summaryStatistics",
        "statisticalDataType": {"@id": "dsv:statisticalDataType", "@type": "@id"},
        "valueType": {"@id": "dsv:valueType", "@type": "@id"},
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
        "hasQuantityKind": {"@id": "qudt:hasQuantityKind", "@type": "@id"},
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
    source: Optional[str] = None # if different from ColumnProperty source


# --- Column property (DSV + QUDT/UCUM) ---------------------------------------
class StatisticalDataType(Enum):
    Interval = "dsv:IntervalDataType"
    Nominal = "dsv:NominalDataType"
    Numerical = "dsv:NumericalDataType"
    Ordinal = "dsv:OrdinalDataType"
    Ratio = "dsv:RatioDataType"


@dataclass
class SummaryStatistics(JSONLDMixin):
    # Can be used for column- and dataset-level stats
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
    valueType: Optional[str] = None  # e.g., "xsd:integer"
    hasQuantityKind: Optional[str] = None  # e.g., "quantitykind:Time"
    unitText: Optional[str] = None  # e.g., "unit:YR"
    hasUnit: Optional[str | Unit] = None
    source: Optional[str] = None  # web page with documentation
    hasCodeBook: Optional[CodeBook] = None
    # Link to a variable definition (SKOS Concept) for the column
    hasVariable: Optional[Union[str, CodeConcept]] = None


# --- CSVW/DSV column and schema ----------------------------------------------
@dataclass
class Column(JSONLDMixin):
    name: str  # required
    titles: Optional[Union[str, List[str]]] = None
    columnProperty: Optional[ColumnProperty] = None
    # DSV allows summary statistics on components (columns)
    summaryStatistics: Optional[SummaryStatistics] = None


@dataclass
class DatasetSchema(JSONLDMixin):
    __context__ = CONTEXT
    columns: List[Column]  # required


# --- Root document / Dataset --------------------------------------------------
@dataclass
class Metadata(JSONLDMixin):
    __context__ = CONTEXT
    # This acts as the dsv:Dataset and links to its dsv:DatasetSchema
    datasetSchema: DatasetSchema  # required
    # Dataset-level summary statistics
    summaryStatistics: Optional[SummaryStatistics] = None