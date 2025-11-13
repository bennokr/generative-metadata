"""Toolkit to profile, describe and synthesize tabular datasets.

- Unified model interface: run both Metasyn, PyBNesian and SynthCity models from a single config.
- Uniform outputs: each model writes artifacts under dataset/models/<model-name>/.
- Provider-aware metadata and UMAP visuals.
- Generate static HTML reports
"""

from __future__ import annotations

from importlib import metadata

__all__ = ["__version__", "get_version"]


def get_version() -> str:
    """Return the installed SemSynth version string."""

    try:
        return metadata.version("semsynth")
    except metadata.PackageNotFoundError:  # pragma: no cover - fallback for dev installs
        return "0.0.0"


__version__ = get_version()
