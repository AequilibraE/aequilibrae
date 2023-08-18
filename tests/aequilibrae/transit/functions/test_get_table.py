import pandas as pd

from aequilibrae.utils.get_table import get_table
from aequilibrae.utils.list_tables_in_db import list_tables_in_db


def test_get_table(transit_conn):
    tables = get_table("routes", transit_conn)

    assert type(tables) is pd.DataFrame


def test_list_tables_in_db(transit_conn):
    list_table = list_tables_in_db(transit_conn)

    assert len(list_table) >= 5
