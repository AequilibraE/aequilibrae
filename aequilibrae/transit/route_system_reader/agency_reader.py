import sqlite3

# from aequilibrae.data import DataTableStorage
from aequilibrae.utils.get_table import get_table
from aequilibrae.transit.transit_elements import Agency


def read_agencies(conn: sqlite3.Connection):
    data = get_table("transit_agencies", conn).reset_index()
    return [Agency().from_row(dt) for _, dt in data.iterrows() if dt.agency_id > 1]
