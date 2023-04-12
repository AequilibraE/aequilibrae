import sqlite3

import pandas as pd
from aequilibrae.utils.get_table import get_table

# from polarislib.network.data import DataTableStorage
from aequilibrae.transit.transit_elements import Trip


def read_trips(conn: sqlite3.Connection):
    data = get_table("transit_trips", conn).reset_index()
    data.drop(columns=["seated_capacity", "design_capacity", "total_capacity", "is_artic"], inplace=True)
    data.drop(columns=["number_of_cars"], inplace=True)

    pats = pd.read_sql("Select pattern_id, route_id from Transit_Patterns", conn)
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
    return [Trip().from_row(dt) for _, dt in data.iterrows()]
