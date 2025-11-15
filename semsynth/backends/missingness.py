"""Backward-compatible re-export for missingness utilities."""

from __future__ import annotations

from ..missingness import DataFrameMissingnessModel, MissingnessWrappedGenerator

__all__ = ["DataFrameMissingnessModel", "MissingnessWrappedGenerator"]
