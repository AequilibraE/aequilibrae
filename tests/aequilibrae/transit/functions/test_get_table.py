import pandas as pd

from aequilibrae.project import Project
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.functions.data import get_table
from aequilibrae.transit.functions.db_utils import list_tables_in_db


class TestDBUtils:
    def test_get_table(self, project: Project):
        conn = database_connection(table_type="transit")

        tables = get_table("routes", conn)

        assert type(tables) == pd.DataFrame

    def test_list_tables_in_db(self, project: Project):
        conn = database_connection(table_type="transit")

        list_table = list_tables_in_db(conn)

        assert len(list_table) >= 5
