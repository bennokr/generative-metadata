"""Backend interface definitions for SemSynth."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
else:  # pragma: no cover - runtime does not require pandas
    pd = Any  # type: ignore


class Backend(Protocol):
    """Protocol describing the callable surface for backend implementations."""

    def run_experiment(
        self,
        df: "pd.DataFrame",
        *,
        provider: Optional[str],
        dataset_name: Optional[str],
        provider_id: Optional[int],
        outdir: str,
        label: str,
        model_info: Optional[Dict[str, Any]],
        rows: Optional[int],
        seed: int,
        test_size: float,
        semmap_export: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Execute the backend for a dataset and return the run directory."""


class BackendModule(Protocol):
    """Protocol applied to backend modules."""

    def run_experiment(
        df: "pd.DataFrame",
        *,
        provider: Optional[str],
        dataset_name: Optional[str],
        provider_id: Optional[int],
        outdir: str,
        label: str,
        model_info: Optional[Dict[str, Any]],
        rows: Optional[int],
        seed: int,
        test_size: float,
        semmap_export: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Execute the backend for a dataset and return the run directory."""


_REQUIRED_KEYWORD_PARAMS = (
    "provider",
    "dataset_name",
    "provider_id",
    "outdir",
    "label",
    "model_info",
    "rows",
    "seed",
    "test_size",
    "semmap_export",
)


def ensure_backend_contract(module: ModuleType) -> None:
    """Validate that a backend module exposes a compliant ``run_experiment``."""

    run_experiment = getattr(module, "run_experiment", None)
    if not callable(run_experiment):
        raise RuntimeError(
            f"Backend module {module.__name__!r} must define a callable 'run_experiment'"
        )
    signature = inspect.signature(run_experiment)
    params = signature.parameters
    if not params:
        raise RuntimeError("run_experiment must accept a pandas DataFrame as the first argument")
    first = next(iter(params.values()))
    if first.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):
        raise RuntimeError("run_experiment must accept the dataset as the first positional argument")
    for name in _REQUIRED_KEYWORD_PARAMS:
        if name not in params:
            raise RuntimeError(f"run_experiment missing required parameter '{name}'")
        if params[name].kind not in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise RuntimeError(f"Parameter '{name}' must be positional-or-keyword or keyword-only")


__all__ = ["Backend", "BackendModule", "ensure_backend_contract"]
