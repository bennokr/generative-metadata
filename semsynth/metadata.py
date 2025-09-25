from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from datasets import DatasetSpec
from ..jsonld import JSONLDMixin

JSONLD_CONTEXT = {
  "@context": {
    "dcat": "http://www.w3.org/ns/dcat#",
    "dct": "http://purl.org/dc/terms/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",

    "identifier": "dct:identifier",
    "title": "dct:title",
    "description": "dct:description",
    "publisher": "dct:publisher",
    "keywords": "dcat:keyword",
    "issued": {"@id": "dct:issued", "@type": "xsd:date"},
    "modified": {"@id": "dct:modified", "@type": "xsd:date"},
    "language": "dct:language",
    "distributions": {
      "@id": "dcat:distribution",
      "@type": "@id"
    },

    "Distribution": "dcat:Distribution",
    "access_url": "dcat:accessURL",
    "download_url": "dcat:downloadURL",
    "media_type": "dcat:mediaType"
  }
}
@dataclass
class DCATDistribution(JSONLDMixin):
    __context__ = JSONLD_CONTEXT
    title: str
    access_url: Optional[str] = None
    download_url: Optional[str] = None
    media_type: Optional[str] = None


@dataclass
class DCATDataset(JSONLDMixin):
    __context__ = JSONLD_CONTEXT
    identifier: str
    title: str
    description: str
    publisher: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    issued: Optional[str] = None
    modified: Optional[str] = None
    language: Optional[str] = None
    distributions: List[DCATDistribution] = field(default_factory=list)