from __future__ import annotations
from dataclasses import dataclass, fields
from typing import List, Optional, Union
import json
import pathlib

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
    "tableSchema": "csvw:tableSchema",
    "columns": { "@id": "csvw:column", "@container": "@set" },
    "name": "csvw:name",
    "titles": "csvw:titles",

    "columnProperty": "dsv:columnProperty",
    "statisticalDataType": { "@id": "dsv:statisticalDataType", "@type": "@id" },
    "valueType": { "@id": "dsv:valueType", "@type": "@id" },
    "hasCodeBook": { "@id": "dsv:hasCodeBook", "@type": "@id" },

    "notation": "skos:notation",
    "prefLabel": "skos:prefLabel",
    "exactMatch": { "@id": "skos:exactMatch", "@type": "@id", "@container": "@set" },
    "closeMatch": { "@id": "skos:closeMatch", "@type": "@id", "@container": "@set" },
    "broadMatch": { "@id": "skos:broadMatch", "@type": "@id", "@container": "@set" },
    "narrowMatch": { "@id": "skos:narrowMatch", "@type": "@id", "@container": "@set" },
    "relatedMatch": { "@id": "skos:relatedMatch", "@type": "@id", "@container": "@set" },
    "hasTopConcept": { "@id": "skos:hasTopConcept", "@type": "@id", "@container": "@set" },

    "hasQuantityKind": { "@id": "qudt:hasQuantityKind", "@type": "@id" },
    "unitText": "schema:unitText",
    "ucumCode": "qudt:ucumCode",
    "source": { "@id": "dct:source", "@type": "@id" }
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
    __context__ = CONTEXT
    hasTopConcept: Optional[List[CodeConcept]] = None
    source: Optional[str] = None


# --- Column property (DSV + QUDT/UCUM) ---------------------------------------
@dataclass
class ColumnProperty(SkosMappings):
    statisticalDataType: Optional[str] = None          # e.g., "dsv:NominalDataType"
    valueType: Optional[str] = None                    # e.g., "xsd:integer"
    hasQuantityKind: Optional[str] = None              # e.g., "quantitykind:Time"
    unitText: Optional[str] = None                      # e.g., "unit:YR"
    ucumCode: Optional[str] = None                     # e.g., "a"
    source: Optional[str] = None                       # e.g., "https://loinc.org/..."
    hasCodeBook: Optional[CodeBook] = None


# --- CSVW/DSV column and schema ----------------------------------------------
@dataclass
class Column(JSONLDMixin):
    __context__ = CONTEXT
    name: str                                          # required
    titles: Optional[Union[str, List[str]]] = None
    columnProperty: Optional[ColumnProperty] = None


@dataclass
class TableSchema(JSONLDMixin):
    __context__ = CONTEXT
    columns: List[Column]                              # required


# --- Root document ------------------------------------------------------------
@dataclass
class Metadata(JSONLDMixin):
    __context__ = CONTEXT
    tableSchema: TableSchema                           # required
