import os
import zipfile
from os.path import join
from typing import Any, cast

import pandas as pd
from pyproj import Transformer

from aequilibrae.transit.functions.get_srid import get_srid
from aequilibrae.transit.gtfs_writer import write_routes, write_agencies, write_fares
from aequilibrae.transit.gtfs_writer import write_stops, write_trips, write_stop_times, write_shapes
from aequilibrae.transit.route_system_reader import read_agencies, read_patterns
from aequilibrae.transit.route_system_reader import read_stop_times, read_stops, read_trips, read_routes


class RouteSystem:
    def __init__(self, project):
        self.project = project

        self.agencies = pd.DataFrame([])
        self.stops = pd.DataFrame([])
        self.routes = pd.DataFrame([])
        self.trips = pd.DataFrame([])
        self.patterns = pd.DataFrame([])
        self.stop_times = pd.DataFrame([])

        self.fare_attributes = pd.DataFrame([])
        self.fare_rules = pd.DataFrame([])
        self.zones = pd.DataFrame([])

        self.transformer = Transformer.from_crs(f"epsg:{get_srid()}", "epsg:4326", always_xy=True)

    def load_route_system(self):
        with self.project.transit_connection as conn:
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
        if self.patterns.empty:
            self.patterns = read_patterns(conn, self.transformer)

    def _read_trips(self, conn):
        if self.trips.empty:
            self.trips = read_trips(conn)

    def _read_stop_times(self, conn):
        self.stop_times = read_stop_times(conn)

    def write_GTFS(self, path_to_folder: str):
        """ """

        with self.project.transit_connection as conn:
            write_agencies(cast(Any, self.agencies), path_to_folder)
            write_stops(cast(Any, self.stops), path_to_folder)
            write_routes(cast(Any, self.routes), path_to_folder)
            write_shapes(cast(Any, self.patterns), path_to_folder)

            write_trips(cast(Any, self.trips), path_to_folder, conn)
            write_stop_times(self.stop_times, path_to_folder)
            write_fares(path_to_folder, conn)
            self._zip_feed(path_to_folder)

    def _zip_feed(self, path_to_folder: str):
        filename = join(path_to_folder, "aequilibrae_gtfs.zip")
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
