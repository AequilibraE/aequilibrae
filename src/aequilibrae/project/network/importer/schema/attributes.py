"""Route staged-network attributes into table columns or ``other_attributes``."""

import geopandas as gpd
import json
import math
import pandas as pd
from typing import Iterable

PROT_COLS = {"ogc_fid", "geometry"}
JSON_COL = "other_attributes"


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def to_jsonable(value):
    """Convert a value to a JSON-serialisable form."""
    if is_missing(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return str(value)


def _row_to_json_dropping_nans(row: pd.Series):
    payload = {}
    for key, value in row.items():
        if is_missing(value):
            continue
        payload[str(key)] = to_jsonable(value)
    if not payload:
        return None
    return json.dumps(payload, separators=(",", ":"), default=str)


def _merge_json(existing: pd.Series, extras: pd.Series) -> pd.Series:
    """Merge per-row ``other_attributes`` JSON strings.

    ``existing`` is whatever the source put in the IR ``other_attributes``
    column; ``extras`` is the JSON computed from non-DB-mapped columns.
    The extras are merged INTO the existing dict (extras win on key
    collisions, since they represent the more recent acquisition).
    """

    def merge_one(existing_value, extras_value):
        base = {}
        if not is_missing(existing_value):
            if isinstance(existing_value, str):
                parsed = json.loads(existing_value)
                if isinstance(parsed, dict):
                    base = parsed
            elif isinstance(existing_value, dict):
                base = dict(existing_value)
        if not is_missing(extras_value):
            if isinstance(extras_value, str):
                extra = json.loads(extras_value)
                if isinstance(extra, dict):
                    base.update(extra)
            elif isinstance(extras_value, dict):
                base.update(extras_value)
        if not base:
            return None
        return json.dumps(base, separators=(",", ":"), default=str)

    return pd.Series(
        [merge_one(e, x) for e, x in zip(existing, extras, strict=False)],
        index=existing.index,
    )


def split_attributes(gdf: gpd.GeoDataFrame, table_cols: Iterable[str], ) -> tuple[gpd.GeoDataFrame, pd.Series]:
    """Route the columns of ``gdf`` for write into a spatialite table."""
    col_set = set(table_cols)
    cols = list(gdf.columns)

    known = [c for c in cols if c in col_set and c not in PROT_COLS and c != JSON_COL and not str(c).startswith("_")]
    exts = [c for c in cols if c not in col_set and c not in PROT_COLS and c != JSON_COL and not str(c).startswith("_")]

    direct_cols = known + ["geometry"] if "geometry" in cols else known
    direct = gdf[direct_cols].copy()

    if exts:
        extra_json = gdf[exts].apply(_row_to_json_dropping_nans, axis=1)
    else:
        extra_json = pd.Series([None] * len(gdf), index=gdf.index, dtype="object")

    if JSON_COL in cols:
        extra_json = _merge_json(gdf[JSON_COL], extra_json)

    return direct, extra_json
