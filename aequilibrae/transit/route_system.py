import os
import sqlite3
import zipfile
from os.path import join
from pathlib import Path

import pandas as pd
from pyproj import Transformer

from polarislib.network.tools.geo import Geo
from polarislib.network.data import DataTableStorage
from polarislib.network.database_connection import get_srid
from polarislib.network.starts_logging import logger
from polarislib.network.transit.gtfs_writer import write_routes, write_agencies, write_fares
from polarislib.network.transit.gtfs_writer import write_stops, write_trips, write_stop_times, write_shapes
from polarislib.network.transit.route_system_reader import read_agencies, read_patterns
from polarislib.network.transit.route_system_reader import read_stop_times, read_stops, read_trips, read_routes
from polarislib.utils.database.spatialite_utils import connect_spatialite


class RouteSystem:
    def __init__(self, database_path):
        self.__database_path = database_path
        self.dts = DataTableStorage()
        self.__conn: sqlite3.Connection = None

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
        self._read_agencies()
        self._read_stops()
        self._read_routes()
        self._read_patterns()
        self._read_trips()
        self._read_stop_times()

    def _read_agencies(self):
        self.agencies = read_agencies(self.conn)

    def _read_stops(self):
        self.stops = read_stops(self.conn, self.transformer)

    def _read_routes(self):
        self.routes = read_routes(self.conn)

    def _read_patterns(self):
        self.patterns = self.patterns or read_patterns(self.conn, self.transformer)

    def _read_trips(self):
        self.trips = self.trips or read_trips(self.conn)

    def _read_stop_times(self):
        self.stop_times = read_stop_times(self.conn)

    @property
    def conn(self) -> sqlite3.Connection:
        self.__conn = self.__conn or connect_spatialite(Path(self.__database_path))
        return self.__conn

    def write_GTFS(self, path_to_folder: str):
        """ """
        timezone = self._timezone()

        write_agencies(self.agencies, path_to_folder, timezone)
        write_stops(self.stops, path_to_folder)
        write_routes(self.routes, path_to_folder)
        write_shapes(self.patterns, path_to_folder)

        write_trips(self.trips, path_to_folder, self.conn)
        write_stop_times(self.stop_times, path_to_folder)
        write_fares(path_to_folder, self.conn)
        self._zip_feed(path_to_folder)

    def _timezone(self, allow_error=True):
        geotool = Geo()
        geotool.conn = self.conn
        try:
            return geotool.get_timezone()
        except Exception as e:
            logger.error("Could not retrieve the correct time zone for GTFS exporter. Using Chicago instead")
            if not allow_error:
                raise e
            return "America/Chicago"

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
