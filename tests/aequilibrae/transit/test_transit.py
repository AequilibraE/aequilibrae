import os
from tempfile import gettempdir
import unittest
from uuid import uuid4
from aequilibrae.project import Project
from aequilibrae.transit import Transit


class TestTransit(unittest.TestCase):
    def setUp(self) -> None:
        self.fldr = os.path.join(gettempdir(), uuid4().hex)
        self.prj = Project()
        self.prj.new(self.fldr)

    def tearDown(self) -> None:
        self.prj.close()

    def test_new_gtfs(self):

        data = Transit(self.prj)
        transit = data.new_gtfs(agency="", file_path=os.path.join(os.path.abspath(os.path.dirname("tests")), "tests/data/2020-04-01.zip"))

        self.assertEqual(str(type(transit)), "<class 'aequilibrae.transit.lib_gtfs.GTFSRouteSystemBuilder'>")


if __name__ == "__name__":
    unittest.main()
