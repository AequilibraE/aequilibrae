import os
from tempfile import gettempdir
from unittest import TestCase
from uuid import uuid4
from aequilibrae.project import Project
from aequilibrae.transit import Transit
from aequilibrae.transit.functions.transit_connection import transit_connection

from aequilibrae.transit.transit_elements import Pattern


class TestTransitPattern(TestCase):
    def setUp(self) -> None:
        self.fldr = os.path.join(gettempdir(), uuid4().hex)
        self.prj = Project()
        self.prj.new(self.fldr)

        self.gtfs_fldr = os.path.join(os.path.abspath(os.path.dirname("tests")), "tests/data/2020-04-01.zip")

        agency_id = "Campo"
        description = "CampMetro"
        date = "2020-04-01"

        self.network = transit_connection(self.fldr)

        self.data = Transit(self.prj)
        self.transit = self.data.new_gtfs(agency=agency_id, file_path=self.gtfs_fldr, description=description)
        self.transit.load_date(date)

        self.patterns = self.transit.select_patterns  # type: dict
        self.pat = [x for x in self.patterns.values()][0]  # type: Pattern

    def test_save_to_database(self):
        self.pat.save_to_database(self.network)

    def test_best_shape(self):
        shp = self.pat.best_shape()
        self.assertEqual(shp, self.pat._stop_based_shape, "Returned the wrong shape")

    def test_get_error(self):
        self.assertEqual(self.pat.get_error(), None, "Resulted a map-matching error when should have returned none")

    def test_map_match(self):
        self.pat.map_match()
        self.pat.save_to_database(self.network)
