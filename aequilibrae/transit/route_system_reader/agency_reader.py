import sqlite3

from polarislib.network.data import DataTableStorage
from polarislib.network.transit.transit_elements import Agency


def read_agencies(conn: sqlite3.Connection):
    data = DataTableStorage().get_table("transit_agencies", conn).reset_index()
    return [Agency().from_row(dt) for _, dt in data.iterrows() if dt.agency_id > 1]
