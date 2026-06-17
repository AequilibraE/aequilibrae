"""Schema-aware routing of free-form IR columns into ``other_attributes`` JSON.

Implements ``_split_attributes`` from plan §4.4. The function decides per
column whether it lands in a same-named existing table column or is JSON-encoded
into ``other_attributes``.

The committer never issues ``ALTER TABLE``: this is the single place that
enforces that property.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Iterable

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


PROTECTED_COLS = {"ogc_fid", "geometry"}
JSON_COL = "other_attributes"


def _is_nan_like(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value == "":
        return False  # empty string is meaningful
    try:
        # pandas NA
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _json_safe(value):
    """Best-effort conversion of a value to a JSON-serialisable form."""
    if _is_nan_like(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    # Fallback: stringify
    try:
        return str(value)
    except Exception:  # pragma: no cover
        return None


def _row_to_json_dropping_nans(row: pd.Series) -> str | None:
    payload = {}
    for key, value in row.items():
        if _is_nan_like(value):
            continue
        payload[str(key)] = _json_safe(value)
    if not payload:
        return None
    return json.dumps(payload, separators=(",", ":"), default=str)


def _merge_json(existing: pd.Series, extras: pd.Series) -> pd.Series:
    """Merge per-row ``other_attributes`` JSON strings.

    ``existing`` is whatever the source already put in the IR column called
    ``other_attributes`` (may be None / NaN); ``extras`` is the JSON computed
    from all the non-DB-mapped IR columns. The extras are merged INTO the
    existing dict (extras win on key collisions, since they represent the more
    recent acquisition).
    """

    def merge_one(existing_value, extras_value):
        base = {}
        if existing_value is not None and not _is_nan_like(existing_value):
            if isinstance(existing_value, str):
                try:
                    parsed = json.loads(existing_value)
                    if isinstance(parsed, dict):
                        base = parsed
                except json.JSONDecodeError:
                    pass
            elif isinstance(existing_value, dict):
                base = dict(existing_value)
        if extras_value is not None and not _is_nan_like(extras_value):
            if isinstance(extras_value, str):
                try:
                    extra = json.loads(extras_value)
                    if isinstance(extra, dict):
                        base.update(extra)
                except json.JSONDecodeError:
                    pass
            elif isinstance(extras_value, dict):
                base.update(extras_value)
        if not base:
            return None
        return json.dumps(base, separators=(",", ":"), default=str)

    return pd.Series(
        [merge_one(e, x) for e, x in zip(existing, extras, strict=False)],
        index=existing.index,
    )


def split_attributes(
    gdf: gpd.GeoDataFrame,
    table_cols: Iterable[str],
) -> tuple[gpd.GeoDataFrame, pd.Series]:
    """Route the columns of ``gdf`` for write into a spatialite table.

    :Returns:
        ``(direct, extra_json)`` where:
          - ``direct`` is a copy of ``gdf`` containing only the columns that
            map to existing real columns of the target table (plus geometry).
          - ``extra_json`` is a per-row JSON string (or ``None``) holding the
            non-mapped columns, merged with any pre-existing
            ``other_attributes`` value the IR supplied.

    See plan §4.4 routing rules.
    """
    table_cols_set = set(table_cols)
    cols = list(gdf.columns)

    known = [
        c
        for c in cols
        if c in table_cols_set
        and c not in PROTECTED_COLS
        and c != JSON_COL
        and not str(c).startswith("_")
    ]
    extras = [
        c
        for c in cols
        if c not in table_cols_set
        and c not in PROTECTED_COLS
        and c != JSON_COL
        and not str(c).startswith("_")
    ]

    if "geometry" in cols:
        direct_cols = known + ["geometry"]
    else:
        direct_cols = known
    direct = gdf[direct_cols].copy()

    if extras:
        extra_json = gdf[extras].apply(_row_to_json_dropping_nans, axis=1)
    else:
        extra_json = pd.Series([None] * len(gdf), index=gdf.index, dtype="object")

    if JSON_COL in cols:
        extra_json = _merge_json(gdf[JSON_COL], extra_json)

    return direct, extra_json
