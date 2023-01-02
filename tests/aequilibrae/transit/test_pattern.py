import os
import shutil
from tempfile import gettempdir
from unittest import TestCase
from uuid import uuid4
from aequilibrae.project import Project
from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit import Transit


class TestTransitPattern(TestCase):
    def setUp(self) -> None:
        self.fldr = os.path.join(gettempdir(), uuid4().hex)
        shutil.copytree(
            src=os.path.join(os.path.abspath(os.path.dirname("tests")), "tests/data/gtfs/AustinProject"),
            dst=self.fldr,
        )
        self.prj = Project()
        self.prj.open(self.fldr)

        self.prj.create_empty_transit()

        self.gtfs_fldr = os.path.join(os.path.abspath(os.path.dirname("tests")), "tests/data/gtfs/2020-04-01.zip")

        agency_id = "Campo"
        description = "CampMetro"
        date = "2020-04-01"

        self.network = database_connection(table_type="transit")

        self.data = Transit(self.prj)
        self.transit = self.data.new_gtfs(agency=agency_id, file_path=self.gtfs_fldr, description=description)
        self.transit.load_date(date)

        self.patterns = self.transit.select_patterns
        self.pat = [x for x in self.patterns.values()][0]

    def test_save_to_database(self):
        self.pat.save_to_database(self.network)

        routes = self.network.execute("SELECT COUNT(*) FROM routes;").fetchone()[0]
        self.assertEqual(routes, 1)

    def test_best_shape(self):
        shp = self.pat.best_shape()
        self.assertEqual(shp, self.pat._stop_based_shape, "Returned the wrong shape")

    def test_get_error(self):
        self.assertEqual(self.pat.get_error(), None, "Resulted a map-matching error when should have returned none")

    def test_map_match(self):
        self.pat.map_match()
        self.pat.save_to_database(self.network)

        pattern_map = self.network.execute("SELECT COUNT(*) FROM pattern_mapping;").fetchone()[0]
        self.assertEqual(pattern_map, 208)
