"""Tests ensuring the package keeps imports lightweight."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


_HEAVY_ROOT_MODULES = frozenset({
    "pandas",
    "umap",
    "synthcity",
    "metasyn",
    "pybnesian",
})


@pytest.fixture()
def isolated_semsynth_import():
    """Remove existing ``semsynth`` modules from ``sys.modules`` before a test."""

    existing = [m for m in sys.modules if m == "semsynth" or m.startswith("semsynth.")]
    for module_name in existing:
        sys.modules.pop(module_name)

    heavy_removed = [m for m in sys.modules if m.split(".")[0] in _HEAVY_ROOT_MODULES]
    for module_name in heavy_removed:
        sys.modules.pop(module_name)

    importlib.invalidate_caches()
    yield
    for module_name in existing:
        sys.modules.pop(module_name, None)
    for module_name in heavy_removed:
        sys.modules.pop(module_name, None)



def test_import_is_lightweight(monkeypatch, isolated_semsynth_import):
    """Ensure importing :mod:`semsynth` does not load heavyweight optional deps."""

    original_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root in _HEAVY_ROOT_MODULES:
            raise AssertionError(f"Attempted to import heavy module '{name}' during package import")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)

    import semsynth  # noqa: F401  # pylint: disable=import-outside-toplevel

    for heavy in _HEAVY_ROOT_MODULES:
        assert heavy not in sys.modules
