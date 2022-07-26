from unittest import TestCase
import os
from os.path import dirname, join

from aequilibrae.transit.gtfs import GTFS

# Adds the folder with the data to the path and collects the paths to the files
from ...data import gtfs_folder


class TestGTFS(TestCase):
    def setUp(self) -> None:
        spatialite_folder = dirname(dirname(dirname(dirname(os.path.abspath(__file__)))))
        spatialite_folder = join(spatialite_folder, "aequilibrae/project")
        os.environ["PATH"] = f"{spatialite_folder};" + os.environ["PATH"]

        self.gtfs = GTFS()
        self.gtfs.source_folder = gtfs_folder

    def test_load_calendar_dates(self):
        self.gtfs.load_calendar_dates()
        if self.gtfs.schedule_exceptions != set(["FULLW"]):
            self.fail("calendar_dates.txt was read wrong")

    def test_load_agency(self):
        try:
            self.gtfs.load_agency()
        except Exception as err:
            self.fail(f"Agency loader returned an error - {err.__str__()}")
        if self.gtfs.agency.name != "Public Transport":
            self.fail("Agency name was read wrong")

    def test_load_stops(self):
        try:
            self.gtfs.load_stops()
        except Exception as err:
            self.fail(f"stops loader returned an error - {err.__str__()}")
        self.assertEqual(len(self.gtfs.stops), 88, "Not all values read")
        if self.gtfs.stops["88"].name != "Post Office":
            self.fail("GTFS stops not read properly")

    def test_load_routes(self):
        self.gtfs.load_routes()
        if self.gtfs.routes["1415"].long_name != "Braitling and Ciccone":
            self.fail("Route long name not read properly")

        if self.gtfs.routes["1825"].short_name != "500":
            self.fail("Route long name not read properly")

    def test_load_trips(self):
        self.gtfs.load_trips()
        # self.fail()

    def test_load_shapes(self):
        self.gtfs.load_shapes()

    def test_get_routes_shapes(self):
        self.gtfs.load_trips()
        self.gtfs.load_routes()
        self.gtfs.load_shapes()

    def get_data(self):
        self.gtfs = GTFS()
        self.gtfs.source_folder = gtfs_folder
