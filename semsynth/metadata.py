from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass, field


from makeprov import JSONLDMixin

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
        "distributions": {"@id": "dcat:distribution", "@type": "@id"},
        "Distribution": "dcat:Distribution",
        "access_url": "dcat:accessURL",
        "download_url": "dcat:downloadURL",
        "media_type": "dcat:mediaType",
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


def get_uciml_variable_descriptions(dataset_id: int) -> Dict[str, str]:
    """Best-effort retrieval of UCI ML variable descriptions."""
    return {}
