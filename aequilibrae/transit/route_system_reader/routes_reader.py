import sqlite3
from aequilibrae.utils.get_table import get_table

# from polarislib.network.data import DataTableStorage
from aequilibrae.transit.transit_elements import Route


def read_routes(conn: sqlite3.Connection):
    data = get_table("transit_routes", conn).reset_index()

    data.drop(columns=["seated_capacity", "design_capacity", "total_capacity", "number_of_cars", "geo"], inplace=True)
    data.rename(
        columns={
            "description": "route_desc",
            "longname": "route_long_name",
            "shortname": "route_short_name",
            "type": "route_type",
        },
        inplace=True,
    )

    return [Route(-1).from_row(dt) for _, dt in data.iterrows()]
