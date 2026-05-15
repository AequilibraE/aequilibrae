import os
import sqlite3
import zipfile
from os.path import join
from pathlib import Path

import pandas as pd
from aequilibrae.transit.gtfs_writer import write_routes, write_agencies, write_fares
from aequilibrae.transit.gtfs_writer import write_stops, write_trips, write_stop_times, write_shapes
from aequilibrae.transit.route_system_reader import read_agencies, read_patterns
from aequilibrae.transit.route_system_reader import read_stop_times, read_stops, read_trips, read_routes
from aequilibrae.utils.get_table import get_table


def export_gtfs(conn: sqlite3.Connection, path_to_folder: Path) -> None:
    """Exports the current transit database contents to a GTFS ZIP archive.

    :Arguments:
        **path_to_folder** (:obj:`str`): Folder where the GTFS text files and resulting zip archive are written.
    """

    path_to_folder.mkdir(parents=True, exist_ok=True)
    agencies = read_agencies(conn)
    stops = read_stops(conn)
    routes = read_routes(conn)
    patterns = read_patterns(conn)
    trips = read_trips(conn)
    stop_times = read_stop_times(conn)
    service_dates = pd.read_sql("SELECT service_date FROM agencies WHERE agency_id > 1", conn)["service_date"]
    fare_attributes = get_table("fare_attributes", conn)
    fare_rules = get_table("fare_rules", conn)

    write_agencies(agencies, path_to_folder)
    write_stops(stops, path_to_folder)
    write_routes(routes, path_to_folder)
    write_shapes(patterns, path_to_folder)
    write_trips(trips, path_to_folder, service_dates)
    write_stop_times(stop_times, path_to_folder)
    write_fares(fare_attributes, fare_rules, path_to_folder)

    filename = path_to_folder / "aequilibrae_gtfs.zip"
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
