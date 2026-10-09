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
    existing = row.get(JSON_COL)
    if isinstance(existing, str):
        existing = json.loads(existing)
    payload = dict(existing) if isinstance(existing, dict) else {}
    for key, value in row.items():
        if key == JSON_COL or is_missing(value):
            continue
        payload[str(key)] = to_jsonable(value)
    if not payload:
        return None
    return json.dumps(payload, separators=(",", ":"), default=str)


def split_attributes(
    gdf: gpd.GeoDataFrame,
    table_cols: Iterable[str],
) -> tuple[gpd.GeoDataFrame, pd.Series]:
    """Route the columns of ``gdf`` for write into a spatialite table."""
    col_set = set(table_cols)
    cols = [c for c in gdf.columns if c not in PROT_COLS and c != JSON_COL and not str(c).startswith("_")]
    direct = gdf[[c for c in cols if c in col_set] + ["geometry"]].copy()
    exts = [c for c in cols if c not in col_set]
    if JSON_COL in gdf.columns:
        exts.append(JSON_COL)

    if exts:
        extra_json = gdf[exts].apply(_row_to_json_dropping_nans, axis=1)
    else:
        extra_json = pd.Series(None, index=gdf.index, dtype="object")

    return direct, extra_json
