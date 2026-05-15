import sqlite3

from aequilibrae.utils.get_table import get_table


def read_routes(conn: sqlite3.Connection):
    data = get_table("routes", conn).reset_index()

    data.drop(columns=["pattern_id", "route", "pce", "seated_capacity", "total_capacity", "geometry"], inplace=True)
    data.rename(
        columns={
            "description": "route_desc",
            "longname": "route_long_name",
            "shortname": "route_short_name",
        },
        inplace=True,
    )
    headers = ["route_id", "agency_id", "route_short_name", "route_long_name", "route_desc", "route_type"]
    return data[headers].copy()
