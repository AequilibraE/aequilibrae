import os
from tempfile import gettempdir
import unittest
from uuid import uuid4

import pandas as pd

from aequilibrae.project import Project
from aequilibrae.transit.functions.data import get_table
from aequilibrae.transit.functions.db_utils import list_tables_in_db
from aequilibrae.transit.functions.transit_connection import transit_connection


class TestDBUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.fldr = os.path.join(gettempdir(), uuid4().hex)
        self.prj = Project()
        self.prj.new(self.fldr)

        self.conn = transit_connection(self.fldr)

    def tearDown(self) -> None:
        self.prj.close()

    def test_get_table(self):
        tables = get_table("routes", self.conn)

        self.assertIs(type(tables), pd.DataFrame)

    def test_list_tables_in_db(self):
        list_table = list_tables_in_db(self.conn)

        self.assertGreater(len(list_table), 5)


if __name__ == "__name__":
    unittest.main()
