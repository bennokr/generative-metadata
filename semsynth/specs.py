"""Shared data structures for SemSynth dataset handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DatasetSpec:
    """Container describing how to locate and identify a dataset."""

    provider: str
    name: Optional[str] = None
    id: Optional[int] = None
    target: Optional[str] = None
    meta: Optional[Any] = None
