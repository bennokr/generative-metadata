"""Signature compliance tests for backend modules."""

from __future__ import annotations

import importlib
from typing import Dict

import pytest

from semsynth.backends.base import ensure_backend_contract


_BACKEND_DEPENDENCIES: Dict[str, str] = {
    "semsynth.backends.pybnesian": "pybnesian",
    "semsynth.backends.synthcity": "synthcity",
    "semsynth.backends.metasyn": "metasyn",
}


@pytest.mark.parametrize("module_name", sorted(_BACKEND_DEPENDENCIES))
def test_backend_contract(module_name: str) -> None:
    """Ensure each backend exposes a compliant ``run_experiment`` signature."""

    dependency = _BACKEND_DEPENDENCIES[module_name]
    pytest.importorskip(dependency)

    module = importlib.import_module(module_name)
    ensure_backend_contract(module)
