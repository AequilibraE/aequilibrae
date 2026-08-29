import sqlite3

import pandas as pd

from aequilibrae.utils.get_table import get_table


def _single_route_value(group: pd.DataFrame, column: str, route_id: int):
    values = group[column].dropna().drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"Cannot export GTFS route_id {route_id}: conflicting values for {column}: {values}")
    return values[0]


def _single_route_text(group: pd.DataFrame, column: str, route_id: int) -> str:
    values = group[column].astype("string").fillna("").str.strip()
    unique = [value for value in pd.unique(values) if value]
    if len(unique) > 1:
        raise ValueError(f"Cannot export GTFS route_id {route_id}: conflicting values for {column}: {unique}")
    return unique[0] if unique else ""


def read_routes(conn: sqlite3.Connection):
    data = get_table("routes", conn).reset_index()
    data.rename(
        columns={
            "description": "route_desc",
            "longname": "route_long_name",
            "shortname": "route_short_name",
        },
        inplace=True,
    )
    headers = ["route_id", "agency_id", "route_short_name", "route_long_name", "route_desc", "route_type"]
    collapsed = []
    for route_id, group in data.sort_values(["route_id", "pattern_id"]).groupby("route_id", sort=True):
        collapsed.append(
            {
                "route_id": route_id,
                "agency_id": _single_route_value(group, "agency_id", route_id),
                "route_short_name": _single_route_text(group, "route_short_name", route_id),
                "route_long_name": _single_route_text(group, "route_long_name", route_id),
                "route_desc": _single_route_text(group, "route_desc", route_id),
                "route_type": _single_route_value(group, "route_type", route_id),
            }
        )

    return pd.DataFrame(collapsed, columns=headers).sort_values("route_id").reset_index(drop=True)
