import sqlite3

import pandas as pd
from aequilibrae.utils.get_table import get_table


def read_trips(conn: sqlite3.Connection):
    data = get_table("trips", conn).reset_index()

    pats = pd.read_sql("SELECT pattern_id, route_id FROM routes", conn).drop_duplicates(subset=["pattern_id"])
    data = data.merge(pats, on="pattern_id")
    data.trip = data.trip.astype(str)
    data.rename(
        columns={
            "trip": "trip_headsign",
            "dir": "direction_id",
            "pattern_id": "shape_id",
        },
        inplace=True,
    )
    data = data.assign(service_id=data.shape_id)
    headers = ["route_id", "service_id", "trip_id", "trip_headsign", "direction_id", "shape_id"]
    return data[headers].copy()
