from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union


# --- SKOS mapping mixin -------------------------------------------------------
@dataclass
class SkosMappings:
    exactMatch: Optional[List[str]] = None
    closeMatch: Optional[List[str]] = None
    broadMatch: Optional[List[str]] = None
    narrowMatch: Optional[List[str]] = None
    relatedMatch: Optional[List[str]] = None


# --- Code book (SKOS) ---------------------------------------------------------
@dataclass
class CodeConcept(SkosMappings):
    notation: Optional[str] = None


@dataclass
class CodeBook:
    hasTopConcept: Optional[List[CodeConcept]] = None


# --- Column property (DSV + QUDT/UCUM) ---------------------------------------
@dataclass
class ColumnProperty(SkosMappings):
    statisticalDataType: Optional[str] = None          # e.g., "dsv:NominalDataType"
    hasQuantityKind: Optional[str] = None              # e.g., "quantitykind:Time"
    hasUnit: Optional[str] = None                      # e.g., "unit:YR"
    ucumCode: Optional[str] = None                     # e.g., "a"
    hasCodeBook: Optional[CodeBook] = None


# --- CSVW/DSV column and schema ----------------------------------------------
@dataclass
class Column:
    name: str                                          # required
    titles: Optional[Union[str, List[str]]] = None
    columnProperty: Optional[ColumnProperty] = None


@dataclass
class TableSchema:
    columns: List[Column]                              # required


# --- Root document ------------------------------------------------------------
@dataclass
class Metadata:
    tableSchema: TableSchema                           # required
