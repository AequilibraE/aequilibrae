import sqlite3
from collections import OrderedDict
import os
import shutil
import io
import importlib
import numpy as np
import codecs
import copy
import zipfile
import logging
import csv
from io import BytesIO
from tempfile import gettempdir
from ...reference_files import spatialite_database
from ...utils import WorkerThread

# from ...utils import WorkerThread
from ...parameters import Parameters
from .parse_csv import parse_csv

have_pyqt5 = importlib.util.find_spec("PyQt5")
if have_pyqt5 is None:
    pyqt = False
else:
    from PyQt5.QtCore import pyqtSignal as SIGNAL

    pyqt = True


# TODO : Add control for mandatory and optional files
# TODO: Add constraints for non-negative and limited options (0,1,2, etc) for fields through foreign keys
class create_gtfsdb(WorkerThread):
    if pyqt:
        converting_gtfs = SIGNAL(object)

    def __emit_all(self, *args):
        if pyqt:
            self.converting_gtfs.emit(*args)

    def __init__(self, file_path, save_db, memory_db=False, spatialite_enabled=False, overwrite=False):
        WorkerThread.__init__(self, None)
        self.conn = None
        self.cursor = None
        self.__max_chunk_size = None
        self.spatialite_enabled = spatialite_enabled
        self.available_files = {}
        self.source_path = file_path
        self.save_db = save_db
        self.memory_db = memory_db
        self.overwrite = overwrite
        self.report = []
        self.logger = logging.getLogger("aequilibrae")
        log_level = Parameters().parameters["system"]["logging"]
        if isinstance(log_level, str):
            log_level = log_level.upper()
        self.logger.setLevel(log_level)

        self.logger.info("Starting GTFS import")
        self.logger.info("      " + self.source_path)
        self.logger.info("      " + str(self.save_db))

        if os.path.isfile(self.source_path):
            zip_container = zipfile.ZipFile(self.source_path)
            ret = zip_container.testzip()
            if ret is not None:
                raise ValueError("First bad file in zip: %s" % ret)
            else:
                self.source = "zip"
        elif os.path.isdir(self.source_path):
            self.source = "folder"
        else:
            msg = "Source needs to be zip file or directory/folder"
            self.logger.error(msg)
            raise ValueError(msg)

        # Input consistency checks
        if self.source is None:
            msg = "Input is neither a zip file nor a folder"
            self.logger.error("      " + msg)
            raise ValueError(msg)

        if spatialite_enabled and memory_db:
            msg = "Spatialite is only supported on disk"
            self.logger.error("      " + msg)
            raise ValueError(msg)

        if not self.overwrite:
            if os.path.isfile(os.path.join(save_db)):
                msg = "Output database exists. Please use overwrite=True or choose a different path/name"
                self.logger.error("      " + msg)
                raise ValueError(msg)
        else:
            if os.path.isfile(save_db):
                os.unlink(save_db)

        if self.save_db is None:
            if not self.memory_db:
                save_db = ":memory:"
        else:
            if self.memory_db:
                msg = "You can't have a file name and have the file in memory at the same time"
                self.logger.error("      " + msg)
                raise ValueError(msg)

        OrderedDict([("s", (1, 2)), ("p", (3, 4)), ("a", (5, 6)), ("m", (7, 8))])
        self.column_order = {
            "agency.txt": OrderedDict(
                [
                    ("agency_id", str),
                    ("agency_name", str),
                    ("agency_url", str),
                    ("agency_timezone", str),
                    ("agency_lang", str),
                    ("agency_phone", str),
                    ("agency_fare_url", str),
                    ("agency_email", str),
                ]
            ),
            "routes.txt": OrderedDict(
                [
                    ("route_id", str),
                    ("agency_id", str),
                    ("route_short_name", str),
                    ("route_long_name", str),
                    ("route_desc", str),
                    ("route_type", int),
                    ("route_url", str),
                    ("route_color", str),
                    ("route_text_color", str),
                    ("route_sort_order", int),
                ]
            ),
            "trips.txt": OrderedDict(
                [
                    ("route_id", str),
                    ("service_id", str),
                    ("trip_id", str),
                    ("trip_headsign", str),
                    ("trip_short_name", str),
                    ("direction_id", int),
                    ("block_id", str),
                    ("shape_id", str),
                    ("wheelchair_accessible", int),
                    ("bikes_allowed", int),
                ]
            ),
            "stop_times.txt": OrderedDict(
                [
                    ("trip_id", str),
                    ("arrival_time", str),
                    ("departure_time", str),
                    ("stop_id", str),
                    ("stop_sequence", int),
                    ("stop_headsign", str),
                    ("pickup_type", int),
                    ("shape_dist_traveled", float),
                    ("timepoint", int),
                ]
            ),
            "calendar.txt": OrderedDict(
                [
                    ("service_id", str),
                    ("monday", int),
                    ("tuesday", int),
                    ("wednesday", int),
                    ("thursday", int),
                    ("friday", int),
                    ("saturday", int),
                    ("sunday", int),
                    ("start_date", str),
                    ("end_date", str),
                ]
            ),
            "calendar_dates.txt": OrderedDict([("service_id", str), ("date", str), ("exception_type", int)]),
            "fare_attributes.txt": OrderedDict(
                [
                    ("fare_id", str),
                    ("price", float),
                    ("currency_type", str),
                    ("payment_method", int),
                    ("transfers", int),
                    ("agency_id", str),
                    ("transfer_duration", float),
                ]
            ),
            "fare_rules.txt": OrderedDict(
                [("fare_id", str), ("route_id", str), ("origin_id", str), ("destination_id", str), ("contains_id", str)]
            ),
            "frequencies.txt": OrderedDict(
                [("trip_id", str), ("start_time", str), ("end_time", str), ("headway_secs", str), ("exact_times", int)]
            ),
            "transfers.txt": OrderedDict(
                [("from_stop_id", str), ("to_stop_id", str), ("transfer_type", int), ("min_transfer_time", int)]
            ),
            "feed_info.txt": OrderedDict(
                [
                    ("feed_publisher_name", str),
                    ("feed_publisher_url", str),
                    ("feed_lang", str),
                    ("feed_start_date", str),
                    ("feed_end_date", str),
                    ("feed_version", str),
                ]
            ),
            "stops.txt": OrderedDict(
                [
                    ("stop_id", str),
                    ("stop_code", str),
                    ("stop_name", str),
                    ("stop_desc", str),
                    ("stop_lat", float),
                    ("stop_lon", float),
                    ("zone_id", str),
                    ("stop_url", str),
                    ("location_type", int),
                    ("parent_station", str),
                    ("stop_timezone", str),
                    ("wheelchair_boarding", int),
                ]
            ),
            "shapes.txt": OrderedDict(
                [
                    ("shape_id", str),
                    ("shape_pt_lat", float),
                    ("shape_pt_lon", float),
                    ("shape_pt_sequence", int),
                    ("shape_dist_traveled", float),
                ]
            ),
        }
        self.set_chunk_size(30000)
        self.logger.info("      No errors found in initial checking")
        self.logger.info("      source: " + self.source)

    def set_chunk_size(self, chunk_size):
        if chunk_size is not None:
            if isinstance(chunk_size, int):
                self.__max_chunk_size = chunk_size

    def import_gtfs(self):
        # In case we have not create the database yet
        if self.conn is None:
            self.__emit_all(["text", "Creating container database"])
            self.create_database()

        # Import all tables to SQLITE
        tables = [x.split(".")[0] for x in self.column_order.keys()]
        if self.spatialite_enabled:
            self.__emit_all(["total files", 16])
        else:
            self.__emit_all(["total files", 14])

        for i, tbl in enumerate(tables):
            self.__emit_all(["text", "Loading data from file: " + tbl])
            self.logger.info("      loading table " + tbl)
            self.__load_tables(tbl)
            self.__emit_all(["files counter", i + 1])

        # Creates the geometry
        if self.spatialite_enabled:
            self.__create_geometry()
        self.conn.commit()
        self.conn.close()
        self.__emit_all(["finished_threaded_procedure", 0])

    def create_database(self):
        self.__emit_all(["text", "Creating empty database"])

        if self.spatialite_enabled:
            shutil.copy(spatialite_database, self.save_db)

        self.conn = sqlite3.connect(self.save_db)
        self.cursor = self.conn.cursor()

        self.__create_empty_tables()

    def __create_empty_tables(self):
        self.__create_agency_table()
        self.__create_route_table()
        self.__create_trips_table()
        self.__create_stop_times_table()
        self.__create_calendar_table()
        self.__create_calendar_dates_table()
        self.__create_fare_attributes_table()
        self.__create_fare_rules_table()
        self.__create_frequencies_table()
        self.__create_transfers_table()
        self.__create_feed_info_table()
        self.__create_stops_table()
        self.__create_shapes_table()
        self.conn.commit()

    def __create_agency_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS agency")
        create_query = """CREATE TABLE 'agency' (agency_id VARCHAR PRIMARY KEY UNIQUE,
                                                 agency_name VARCHAR  NOT NULL,
                                                 agency_url VARCHAR  NOT NULL,
                                                 agency_timezone VARCHAR  NOT NULL,
                                                 agency_lang VARCHAR,
                                                 agency_phone VARCHAR,
                                                 agency_fare_url VARCHAR,
                                                 agency_email VARCHAR);"""
        self.cursor.execute(create_query)

    def __create_route_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS routes")
        create_query = """CREATE TABLE 'routes' (route_id VARCHAR PRIMARY KEY UNIQUE NOT NULL,
                                                 agency_id VARCHAR,
                                                 route_short_name VARCHAR,
                                                 route_long_name VARCHAR,
                                                 route_desc VARCHAR,
                                                 route_type NUMERIC NOT NULL,
                                                 route_url VARCHAR,
                                                 route_color VARCHAR,
                                                 route_text_color VARCHAR,
                                                 route_sort_order NUMERIC,
                                                 FOREIGN KEY(agency_id) REFERENCES agency(agency_id));"""

        self.cursor.execute(create_query)

    def __create_trips_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS trips")
        # TODO: Add foreign key to calendar_dates.txt
        create_query = """CREATE TABLE 'trips' (route_id VARCHAR NOT NULL,
                                                service_id VARCHAR NOT NULL,
                                                trip_id VARCHAR PRIMARY KEY UNIQUE NOT NULL,
                                                trip_headsign VARCHAR,
                                                trip_short_name VARCHAR,
                                                direction_id NUMERIC,
                                                block_id VARCHAR,
                                                shape_id VARCHAR,
                                                wheelchair_accessible NUMERIC,
                                                bikes_allowed NUMERIC,
                                                FOREIGN KEY(route_id) REFERENCES routes(route_id)
                                                FOREIGN KEY(service_id) REFERENCES calendar(service_id));"""

        self.cursor.execute(create_query)

    def __create_calendar_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS calendar")
        create_query = """CREATE TABLE 'calendar' (service_id VARCHAR NOT NULL,
                                                   monday NUMERIC NOT NULL DEFAULT 1,
                                                   tuesday NUMERIC NOT NULL DEFAULT 1,
                                                   wednesday NUMERIC NOT NULL DEFAULT 1,
                                                   thursday NUMERIC NOT NULL DEFAULT 1,
                                                   friday NUMERIC NOT NULL DEFAULT 1,
                                                   saturday NUMERIC NOT NULL DEFAULT 1,
                                                   sunday NUMERIC NOT NULL DEFAULT 1,
                                                   start_date VARCHAR,
                                                   end_date VARCHAR,
                                                   FOREIGN KEY(service_id) REFERENCES trips(service_id));"""

        self.cursor.execute(create_query)

    def __create_calendar_dates_table(self):
        # TODO: Add foreign key to calendar.txt
        self.cursor.execute("DROP TABLE IF EXISTS calendar_dates")
        create_query = """CREATE TABLE 'calendar_dates' (service_id VARCHAR NOT NULL,
                                                   'date' VARCHAR NOT NULL,
                                                   exception_type NUMERIC NOT NULL,
                                                   FOREIGN KEY(service_id) REFERENCES trips(service_id));"""
        self.cursor.execute(create_query)

    def __create_stop_times_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS stop_times")
        create_query = """CREATE TABLE 'stop_times' (stop_time_id INTEGER PRIMARY KEY NOT NULL,
                                                     trip_id VARCHAR NOT NULL,
                                                     arrival_time VARCHAR NOT NULL,
                                                     departure_time VARCHAR NOT NULL,
                                                     stop_id VARCHAR NOT NULL,
                                                     stop_sequence NUMERIC NOT NULL,
                                                     stop_headsign VARCHAR,
                                                     pickup_type NUMERIC DEFAULT 0,
                                                     drop_off_type NUMERIC  DEFAULT 0,
                                                     shape_dist_traveled NUMERIC,
                                                     timepoint NUMERIC,
                                                     FOREIGN KEY(trip_id) REFERENCES trips(trip_id)
                                                     FOREIGN KEY(stop_id) REFERENCES stops(stop_id));"""

        self.cursor.execute(create_query)

    def __create_fare_attributes_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS fare_attributes")
        create_query = """CREATE TABLE 'fare_attributes' (fare_id VARCHAR NOT NULL,
                                                          price NUMERIC NOT NULL,
                                                          currency_type VARCHAR NOT NULL,
                                                          payment_method NUMERIC NOT NULL,
                                                          transfers NUMERIC,
                                                          agency_id VARCHAR,
                                                          transfer_duration NUMERIC);"""

        self.cursor.execute(create_query)

    def __create_fare_rules_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS fare_rules")
        create_query = """CREATE TABLE 'fare_rules' (fare_id VARCHAR NOT NULL,
                                                     route_id VARCHAR,
                                                     origin_id VARCHAR,
                                                     destination_id VARCHAR,
                                                     contains_id VARCHAR,
                                                     FOREIGN KEY(fare_id) REFERENCES fare_attributes(fare_id)
                                                     FOREIGN KEY(route_id) REFERENCES routes(route_id)
                                                     FOREIGN KEY(origin_id) REFERENCES stops(stop_id)
                                                     FOREIGN KEY(contains_id) REFERENCES stops(stop_id)
                                                     FOREIGN KEY(destination_id) REFERENCES stops(stop_id));"""
        self.cursor.execute(create_query)

    def __create_frequencies_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS frequencies")
        create_query = """CREATE TABLE 'frequencies' (trip_id VARCHAR NOT NULL,
                                                     start_time VARCHAR NOT NULL,
                                                     end_time VARCHAR NOT NULL,
                                                     headway_secs VARCHAR NOT NULL,
                                                     exact_times NUMERIC,
                                                     FOREIGN KEY(trip_id) REFERENCES trips(trip_id));"""
        self.cursor.execute(create_query)

    def __create_transfers_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS transfers")
        create_query = """CREATE TABLE 'transfers' (from_stop_id VARCHAR NOT NULL,
                                                    to_stop_id VARCHAR NOT NULL,
                                                    transfer_type NUMERIC NOT NULL,
                                                    min_transfer_time NUMERIC NOT NULL,
                                                    FOREIGN KEY(from_stop_id) REFERENCES stops(stop_id)
                                                    FOREIGN KEY(to_stop_id) REFERENCES stops(stop_id));"""
        self.cursor.execute(create_query)

    def __create_feed_info_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS feed_info")
        create_query = """CREATE TABLE 'feed_info' (feed_publisher_name VARCHAR NOT NULL,
                                                    feed_publisher_url VARCHAR NOT NULL,
                                                    feed_lang VARCHAR NOT NULL,
                                                    feed_start_date VARCHAR,
                                                    feed_end_date VARCHAR,
                                                    feed_version VARCHAR);"""
        self.cursor.execute(create_query)

    def __create_stops_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS stops")
        create_query = """CREATE TABLE 'stops' (stop_id VARCHAR PRIMARY KEY UNIQUE NOT NULL,
                                                stop_code VARCHAR,
                                                stop_name VARCHAR NOT NULL,
                                                stop_desc VARCHAR,
                                                stop_lat NUMERIC NOT NULL,
                                                stop_lon NUMERIC NOT NULL,
                                                zone_id VARCHAR,
                                                stop_url VARCHAR,
                                                location_type NUMERIC,
                                                parent_station VARCHAR,
                                                stop_timezone VARCHAR,
                                                wheelchair_boarding NUMERIC);"""
        self.cursor.execute(create_query)

    def __create_shapes_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS shapes")
        create_query = """CREATE TABLE 'shapes' (shape_id VARCHAR NOT NULL,
                                                 shape_pt_lat NUMERIC NOT NULL,
                                                 shape_pt_lon NUMERIC NOT NULL,
                                                 shape_pt_sequence NUMERIC NOT NULL,
                                                 shape_dist_traveled NUMERIC);"""
        self.cursor.execute(create_query)

    def __load_tables(self, table_name):

        # list fields from table
        cursor = self.cursor.execute("select * from " + table_name)
        available_columns = [description[0].lower() for description in cursor.description]

        # create the file name
        file_to_open = table_name + ".txt"

        # Check if exists
        if self.source == "folder":
            data_file = os.path.join(self.source_path, file_to_open)
            self.logger.warning("          " + data_file)
            if not os.path.isfile(data_file):
                self.available_files[file_to_open] = False
                self.logger.warning("          Table " + table_name + " not available")
                return
            else:
                data_file = open(data_file, "r")
        else:
            zip_container = zipfile.ZipFile(self.source_path, "r")
            if file_to_open in zip_container.namelist():
                data_file = zip_container.open(file_to_open, "r")
                data_file = io.TextIOWrapper(data_file, encoding="utf-8-sig")
            else:
                self.available_files[file_to_open] = False
                self.logger.warning("          Table " + table_name + " not available")
                return

        self.available_files[file_to_open] = True
        data = parse_csv(data_file, column_order=self.column_order[file_to_open])

        if not isinstance(data_file, str):
            data_file.close()

        if data is None:
            return
        # we check which columns in the table structure are available in the dataset
        correspondence = []
        for col in data.dtype.names:
            if col in available_columns:
                correspondence.append(col)
        data = data[correspondence]
        cols = data.dtype.names
        dt = tuple(data.tolist())
        fields = ",".join(len(cols) * ["?"])
        try:
            if not isinstance(dt[0], tuple):
                dt = [dt]
            self.cursor.executemany(
                "INSERT into " + table_name + " (" + ",".join(cols) + ") VALUES(" + fields + ")", dt
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error("Could not load data from table " + table_name)
            self.logger.error(str(e))

    def __create_geometry(self):
        # enable extension loading
        self.conn.enable_load_extension(True)
        self.cursor.execute("SELECT load_extension('mod_spatialite')")
        self.conn.commit()
        # We need to create three things here:
        # 1. A geometry column in the stops table
        # 2. A layer of shapes corresponding to each trip
        # 3. A layer of shapes with all the stops for each trip that can be query'd by route_id

        # 1
        self.__emit_all(["text", "Creating stops' geometry"])
        self.__emit_all(["files counter", 14])

        self.cursor.execute("SELECT AddGeometryColumn( 'stops', 'geometry', 4326, 'POINT', 'XY' );")
        self.cursor.execute("update stops set geometry=MakePoint(stop_lon ,stop_lat, 4326);")
        self.cursor.execute("SELECT CreateSpatialIndex( 'stops' , 'geometry' );")

        # 2
        self.__emit_all(["text", "Creating routes' geometry"])
        self.__emit_all(["files counter", 15])
        # We create the table to hold the shapes for each route
        self.cursor.execute("DROP TABLE IF EXISTS shape_routes")
        # TODO: Add foreign key to calendar_dates.txt
        create_query = """CREATE TABLE 'shape_routes' (route_id VARCHAR,
                                                       trip_id VARCHAR,
                                                       shape_id VARCHAR,
                                                       route_text_color VARCHAR,
                                                       FOREIGN KEY(route_id) REFERENCES routes(route_id)
                                                       FOREIGN KEY(trip_id) REFERENCES trips(trip_id));"""
        self.cursor.execute(create_query)
        self.cursor.execute("SELECT AddGeometryColumn( 'shape_routes', 'geometry', 4326, 'LINESTRING', 'XY' );")
        self.cursor.execute("SELECT CreateSpatialIndex( 'shape_routes' , 'geometry' );")

        # We check if we have shapes in the shape layer
        shape_ids = self.cursor.execute("SELECT DISTINCT shape_id from shapes;").fetchall()
        shape_ids = [str(x[0], "utf-8") for x in shape_ids]
        if len(shape_ids) > 0:
            self.__emit_all(["max chunk counter", len(shape_ids)])

            for i, shp in enumerate(shape_ids):
                self.__emit_all(["chunk counter", i])

                qry = self.cursor.execute(
                    "SELECT route_id, trip_id from trips where cast(shape_id as text)='" + shp + "'"
                ).fetchall()
                if len(qry) > 0:
                    route_id, trip_id = qry[0]

                    points = self.cursor.execute(
                        "SELECT shape_pt_lon, shape_pt_lat from shapes where cast(shape_id as text)='"
                        + str(shp)
                        + "' order by shape_pt_sequence"
                    ).fetchall()
                    txt = "LINESTRING (" + ", ".join([str(x[0]) + " " + str(x[1]) for x in points]) + ")"
                    route_text_color = self.cursor.execute(
                        "SELECT route_text_color from routes where cast(route_id as text)='" + str(route_id) + "'"
                    ).fetchall()
                    if len(route_text_color):
                        route_text_color = route_text_color[0]
                        if len(route_text_color) > 0:
                            sql = (
                                "INSERT INTO shape_routes (route_id, trip_id, shape_id, route_text_color, geometry)"
                                + "VALUES (?,?,?,?,"
                                + "LineFromText('"
                                + str(txt)
                                + "', 4326))"
                            )
                            self.cursor.execute(sql, (route_id, trip_id, shp, route_text_color[0]))
        else:
            trip_ids = self.cursor.execute("SELECT DISTINCT trip_id from trips;").fetchall()
            trip_ids = [str(x[0]) for x in trip_ids]
            self.__emit_all(["max chunk counter", len(trip_ids)])

            for i, trip_id in enumerate(trip_ids):
                self.__emit_all(["chunk counter", i])

                route_id = self.cursor.execute(
                    "SELECT route_id from trips where cast(trip_id as text)='" + str(trip_id) + "'"
                ).fetchone()[0]

                sql = (
                    "SELECT stop_lon, stop_lat FROM stop_times INNER JOIN stops"
                    " ON stop_times.stop_id = stops.stop_id WHERE cast(stop_times.trip_id as text)='"
                    + str(trip_id)
                    + "' order by stop_times.stop_sequence"
                )
                qry = self.cursor.execute(sql).fetchall()
                txt = "LINESTRING (" + ", ".join([str(x[0]) + " " + str(x[1]) for x in qry]) + ")"
                route_text_color = self.cursor.execute(
                    "SELECT route_text_color from routes where cast(route_id as text)='" + str(route_id) + "'"
                ).fetchall()[0]
                sql = (
                    "INSERT INTO shape_routes (route_id, trip_id, route_text_color, geometry) "
                    "VALUES (?,?," + "LineFromText('" + txt + "', 4326))"
                )
                self.cursor.execute(sql, (route_id, trip_id, route_text_color))

        # creates the stops table with route ID info
        self.__emit_all(["text", "Associating routes and stops. Sit tight"])

        self.cursor.execute(
            """CREATE TABLE 'shape_stops'
                                AS
                                SELECT
                                a.stop_time_id,
                                a.trip_id,
                                a.stop_id,
                                c.route_id,
                                b.stop_lon,
                                b.stop_lat
                                FROM
                                stop_times a
                                inner join
                                stops  b
                                on a.stop_id  = b.stop_id
                                inner join
                                trips c
                                on a.trip_id = c.trip_id;"""
        )

        self.cursor.execute("SELECT AddGeometryColumn( 'shape_stops', 'geometry', 4326, 'POINT', 'XY' );")
        self.cursor.execute("update shape_stops set geometry=MakePoint(stop_lon ,stop_lat, 4326);")
        self.cursor.execute("SELECT CreateSpatialIndex( 'shape_stops' , 'geometry' );")
        self.conn.commit()
