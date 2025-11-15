"""Shared helpers for dataset provider modules."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Tuple

import pandas as pd


def clean_dataset_frame(
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Tuple[pd.DataFrame, Optional[str], Optional[pd.Series]]:
    """Return a cleaned dataframe plus detected target column and colour series.

    The helper removes trivial identifier columns, then inspects the provided
    ``target`` hint and metadata for SemMap-style ``prov:hadRole`` annotations to
    identify the target column. When a target column is found, the returned
    series is annotated with ``prov:hadRole = "target"`` for downstream
    consumers.

    Args:
        df: The dataframe to clean.
        target: Optional pre-selected target column name.
        metadata: Optional metadata mapping that may contain SemMap-style
            ``prov:hadRole`` annotations.

    Returns:
        A tuple ``(clean_df, detected_target, color_series)`` where ``clean_df``
        has identifier columns removed, ``detected_target`` is the name of the
        selected target column (if any), and ``color_series`` is the corresponding
        series from ``clean_df`` with ``prov:hadRole`` metadata when available.
    """

    clean_df = df.copy()
    for column in list(clean_df.columns):
        if str(column).lower() in {"id", "index"}:
            clean_df = clean_df.drop(columns=[column])

    candidates = []

    def _append_candidate(name: Optional[str]) -> None:
        if isinstance(name, str) and name and name not in candidates:
            candidates.append(name)

    def _normalise_role(role: object) -> Optional[str]:
        if isinstance(role, str) and role:
            role_lower = role.strip().lower()
            if role_lower == "target":
                return role_lower
        return None

    def _extract_named_role(node: object) -> None:
        if isinstance(node, Mapping):
            role_value = node.get("prov:hadRole")
            role_items: Iterable[str]
            if isinstance(role_value, str):
                role_items = [role_value]
            elif isinstance(role_value, Iterable) and not isinstance(
                role_value, (str, bytes)
            ):
                role_items = [str(item) for item in role_value if isinstance(item, str)]
            else:
                role_items = []

            if any(_normalise_role(r) for r in role_items):
                possible_names = [
                    node.get("schema:name"),
                    node.get("name"),
                    node.get("column"),
                    node.get("column_name"),
                    node.get("field"),
                ]
                for possible in possible_names:
                    if isinstance(possible, str) and possible:
                        _append_candidate(possible)
                        break

            for value in node.values():
                _extract_named_role(value)
        elif isinstance(node, Iterable) and not isinstance(node, (str, bytes)):
            for value in node:
                _extract_named_role(value)

    _append_candidate(target)
    if metadata and isinstance(metadata, Mapping):
        _extract_named_role(metadata)

    detected_target: Optional[str] = None
    color_series: Optional[pd.Series] = None
    for candidate in candidates:
        if candidate in clean_df.columns:
            detected_target = candidate
            color_series = clean_df[candidate].copy()
            color_series.attrs["prov:hadRole"] = "target"
            clean_df[detected_target] = color_series
            break

    return clean_df, detected_target, color_series


__all__ = ["clean_dataset_frame"]

