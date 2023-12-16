import os
import zipfile
from os.path import join

import pandas as pd
from pyproj import Transformer

from aequilibrae.project.database_connection import database_connection
from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.transit.gtfs_writer import write_routes, write_agencies, write_fares
from aequilibrae.transit.gtfs_writer import write_stops, write_trips, write_stop_times, write_shapes
from aequilibrae.transit.route_system_reader import read_agencies, read_patterns
from aequilibrae.transit.route_system_reader import read_stop_times, read_stops, read_trips, read_routes
from aequilibrae.utils.db_utils import commit_and_close


class RouteSystem:
    def __init__(self, database_path):
        self.__database_path = database_path

        self.agencies = []
        self.stops = []
        self.routes = []
        self.trips = []
        self.patterns = []
        self.stop_times = pd.DataFrame([])

        self.fare_attributes = []
        self.fare_rules = []
        self.zones = []

        self.transformer = Transformer.from_crs(f"epsg:{get_srid()}", "epsg:4326", always_xy=True)

    def load_route_system(self):
        with commit_and_close(database_connection(join(self.__database_path, "public_transport.sqlite"))) as conn:
            self._read_agencies(conn)
            self._read_stops(conn)
            self._read_routes(conn)
            self._read_patterns(conn)
            self._read_trips(conn)
            self._read_stop_times(conn)

    def _read_agencies(self, conn):
        self.agencies = read_agencies(conn)

    def _read_stops(self, conn):
        self.stops = read_stops(conn, self.transformer)

    def _read_routes(self, conn):
        self.routes = read_routes(conn)

    def _read_patterns(self, conn):
        self.patterns = self.patterns or read_patterns(conn, self.transformer)

    def _read_trips(self, conn):
        self.trips = self.trips or read_trips(conn)

    def _read_stop_times(self, conn):
        self.stop_times = read_stop_times(conn)

    def write_GTFS(self, path_to_folder: str):
        """ """

        with commit_and_close(database_connection(join(self.__database_path, "public_transport.sqlite"))) as conn:
            write_agencies(self.agencies, path_to_folder)
            write_stops(self.stops, path_to_folder)
            write_routes(self.routes, path_to_folder)
            write_shapes(self.patterns, path_to_folder)

            write_trips(self.trips, path_to_folder, conn)
            write_stop_times(self.stop_times, path_to_folder)
            write_fares(path_to_folder, conn)
            self._zip_feed(path_to_folder)

    def _zip_feed(self, path_to_folder: str):
        filename = join(path_to_folder, "polaris_gtfs.zip")
        files = [
            "agency",
            "stops",
            "routes",
            "trips",
            "stop_times",
            "calendar",
            "shapes",
            "fare_attributes",
            "fare_rules",
        ]
        with zipfile.ZipFile(filename, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                zip_file.write(join(path_to_folder, f"{file}.txt"), f"{file}.txt")
                os.unlink(join(path_to_folder, f"{file}.txt"))
