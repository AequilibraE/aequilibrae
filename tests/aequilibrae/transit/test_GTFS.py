import os, sys
from unittest import TestCase
from aequilibrae.transit.gtfs import GTFS

# Adds the folder with the data to the path and collects the paths to the files
lib_path = os.path.abspath(os.path.join('..', '..'))
sys.path.append(lib_path)
from data import gtfs_folder


class TestGTFS(TestCase):
    def get_data(self):
        self.gtfs = GTFS()
        self.gtfs.source_folder = gtfs_folder

    def test_load_calendar_dates(self):
        self.get_data()
        self.gtfs.load_calendar_dates()
        self.assertEqual(self.gtfs.schedule_exceptions, set(['FULLW']), 'calendar_dates.txt was read wrong')

    def test_load_agency(self):
        try:
            self.get_data()
            self.gtfs.load_agency()
        except:
            self.fail('Agency loader returned an error')
        self.assertEqual(self.gtfs.agency.name, 'Public Transport', 'Agency name was read wrong')

    def test_load_stops(self):
        try:
            self.get_data()
            self.gtfs.load_stops()
        except:
            self.fail('stops loader returned an error')
        self.assertEqual(len(self.gtfs.stops), 88, 'Not all values read')
        self.assertEqual(self.gtfs.stops[88].name, 'Post Office', 'GTFS stops not read properly')

    def test_load_routes(self):
        self.get_data()
        self.gtfs.load_routes()
        self.assertEqual(self.gtfs.routes[1415].long_name, 'Braitling and Ciccone', 'Route long name not read properly')
        self.assertEqual(str(self.gtfs.routes[1825].short_name), '500', 'Route long name not read properly')

    def test_load_trips(self):
        self.get_data()
        self.gtfs.load_trips()
        # self.fail()

    def test_load_shapes(self):
        self.get_data()
        self.gtfs.load_shapes()

    def test_get_routes_shapes(self):
        self.get_data()
        self.gtfs.load_trips()
        self.gtfs.load_routes()
        self.gtfs.load_shapes()

    def get_data(self):
        self.gtfs = GTFS()
        self.gtfs.source_folder = gtfs_folder

